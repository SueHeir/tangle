#!/usr/bin/env python3
"""Train the CT map network on a rented GPU pod (RunPod), driven from this computer.

Run it on the computer that holds the training sets and an SSH key the pod accepts. It needs Python 3.8+ and
the ``ssh`` command (built into Windows 10/11 and macOS), nothing else:

    python pod.py go --ssh "ssh root@IP -p PORT -i ~/.ssh/id_ed25519" --root ROOT --run r16 -- DATA VAL --steps 40000

``--ssh`` is the "SSH over exposed TCP" line from the pod's Connect tab (not the ssh.runpod.io one, which can't
copy files). Everything after ``--`` is train.py's command line without OUT: DATA and VAL are its folder lists
(commas; a folder named twice counts twice), each a folder under ROOT or a full path. Files given to ``--init``
and ``--exclude`` go along.

``go`` runs these steps, each also a command of its own; running ``go`` again picks up where it stopped:

* ``push``: copies the scans the pod doesn't have yet (.npz only; scans listed in ``--exclude`` stay home),
  then checks every scan's zip CRC on the pod and sends any that fail again.
* ``train``: starts train.py on the pod in the background, so it carries on if this computer disconnects. It
  gives every CPU core of the pod to preparing crops, and passes ``--resume``, so a restarted run continues from
  its last checkpoint. The code that trains (train.py, maps.py) is saved with the run.
* ``wait``: prints the progress every few minutes until the run ends.
* ``fetch``: copies the run folder (best.pt, last.pt, config.json, train.log, the code) to ``--dest`` and
  checks the md5s. The pod then stops itself (``--end``), so the GPU stops billing. If nothing fetches the run,
  it stops itself ``--grace`` minutes after the run ends anyway.

``status`` prints where the run is, ``stop`` stops the pod. Nothing goes anywhere but the pod and back.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

WORK = "/workspace/unet"  # on the pod: data/, inputs/, runs/ (the pod's volume disk keeps them while it is stopped)
STAGING = "incoming"  # scans land here and move into data/ once whole
REPO = "https://raw.githubusercontent.com/SueHeir/tangle"
CODE_PATH = "crates/tangle_python/python/examples/ct_unet"
CODE_FILES = ("train.py", "maps.py")
# RunPod keeps the pod's variables (RUNPOD_POD_ID, the pod's API key, PATH) in /etc/rp_environment, which a
# command run over SSH doesn't read by itself.
POD_ENV = "[ -f /etc/rp_environment ] && . /etc/rp_environment; "
POD_SHELL = POD_ENV + f"cd {WORK} && \"$(cat .python)\" pod.py _pod "
FIND_PYTHON = POD_ENV + (
    f"mkdir -p {WORK} && for p in python3 python /usr/bin/python3 /usr/local/bin/python3; do "
    "if command -v $p >/dev/null 2>&1 && $p -c 'import torch' 2>/dev/null; then "
    f"command -v $p > {WORK}/.python; exit 0; fi; done; echo 'no Python with PyTorch on the pod' >&2; exit 1"
)


# ----------------------------------------------------------------------------------------------- this computer


def say(text: str) -> None:
    print(time.strftime("%H:%M:%S ") + text, flush=True)


def hours(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min" if seconds >= 3600 else f"{seconds // 60} min"


class Pod:
    """The pod, reached with the system ``ssh`` from its "SSH over exposed TCP" line."""

    def __init__(self, line: str):
        words = [w.strip("\"'") for w in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', line)]
        if words and Path(words[0]).stem.lower() == "ssh":
            words = words[1:]
        port, key, target, options = "22", None, None, []
        i = 0
        while i < len(words):
            word = words[i]
            if word in ("-p", "-i", "-o") and i + 1 < len(words):
                value = words[i + 1]
                if word == "-p":
                    port = value
                elif word == "-i":
                    key = os.path.expanduser(value)
                else:
                    options += ["-o", value]
                i += 2
                continue
            if not word.startswith("-"):
                target = word
            i += 1
        if not target:
            raise SystemExit(f"--ssh: no user@host in {line!r}")
        if target.endswith("ssh.runpod.io"):
            raise SystemExit("--ssh: that's RunPod's proxy line, which can't copy files. Use the 'SSH over exposed "
                             "TCP' line from the pod's Connect tab (ssh root@IP -p PORT ...).")
        self.key = key
        # New pods reuse addresses with new host keys, so known_hosts is not kept for them.
        self.base = ["ssh", "-p", port, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                     "-o", f"UserKnownHostsFile={os.devnull}", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30",
                     "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
                     *(["-i", key] if key else []), *options, target]

    def run(self, command: str, data: bytes = None, check: bool = True) -> subprocess.CompletedProcess:
        if data is None:
            result = subprocess.run(self.base + [command], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        else:
            result = subprocess.run(self.base + [command], input=data, stdout=subprocess.PIPE)
        if check and result.returncode == 255:
            self.no_access()
        if check and result.returncode != 0:
            raise SystemExit(f"command on the pod failed ({result.returncode}): {command}")
        return result

    def pod(self, *words: str, data: bytes = None, check: bool = True) -> dict:
        """Run ``pod.py _pod WORDS`` on the pod; its last output line is JSON."""
        result = self.run(POD_SHELL + " ".join(shlex.quote(w) for w in words), data=data, check=check)
        lines = result.stdout.decode(errors="replace").strip().splitlines()
        for line in lines[:-1]:
            print(line, flush=True)
        try:
            return json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            print(lines[-1], flush=True)
            return {}

    def install(self) -> None:
        """Put this script on the pod and find the Python that has PyTorch."""
        self.run(f"mkdir -p {WORK} && cat > {WORK}/pod.py", data=Path(__file__).read_bytes())
        self.run(FIND_PYTHON)

    def no_access(self):
        public = Path(self.key + ".pub") if self.key else Path.home() / ".ssh" / "id_ed25519.pub"
        hint = ""
        if public.exists():
            hint = ("\nIf this computer's key isn't the one RunPod knows, add it: in a terminal on the pod, run\n"
                    f"  echo '{public.read_text().strip()}' >> ~/.ssh/authorized_keys\n"
                    "and add the same line under RunPod > Settings > SSH Public Keys, so new pods accept it too.")
        raise SystemExit("Could not reach the pod over SSH (wrong address, the pod is stopped, or it doesn't "
                         "accept this computer's key)." + hint)


def train_inputs(train_args, root: Path):
    """train.py's arguments split into the folders to send, the arguments the pod runs train.py with, the extra
    files to send ({path under WORK: local path}) and the excluded scans."""
    args = []
    for word in train_args:  # --init=X as --init X
        flag, eq, value = word.partition("=")
        args += [flag, value] if eq and flag in ("--init", "--exclude") else [word]
    if len(args) < 2 or args[0].startswith("-") or args[1].startswith("-"):
        raise SystemExit("after --, give train.py's DATA and VAL folder lists first (then its options, no OUT)")
    folders, remote_lists = {}, []
    for listing in args[:2]:
        names = []
        for entry in listing.split(","):
            local = Path(entry) if Path(entry).is_absolute() else root / entry
            if not local.is_dir():  # made on the pod (pod.py make): push checks that it's there
                folders.setdefault(local.name, None)
                names.append(f"{WORK}/data/{local.name}")
                continue
            if folders.setdefault(local.name, local.resolve()) != local.resolve():
                raise SystemExit(f"two different folders are both named {local.name}")
            names.append(f"{WORK}/data/{local.name}")
        remote_lists.append(",".join(names))
    rest, extras, skip = args[2:], {}, set()
    if "--pace" in rest:
        raise SystemExit("--pace waits for scans still being made; on the pod every scan is there from the start")
    for flag in ("--init", "--exclude"):
        if flag in rest:
            k = rest.index(flag)
            local = Path(rest[k + 1])
            local = local if local.is_absolute() or local.exists() else root / local
            if not local.is_file():
                raise SystemExit(f"{flag}: no file {local}")
            name = f"{local.parent.name}_{local.name}" if flag == "--init" else local.name
            extras[f"inputs/{name}"] = local
            rest[k + 1] = f"{WORK}/inputs/{name}"
            if flag == "--exclude":
                skip = set(local.read_text().split())
    return folders, remote_lists + rest, extras, skip


def scan_list(folders: dict, skip: set) -> dict:
    """{path under WORK on the pod: local path} of every scan to train on."""
    out = {}
    for name, folder in folders.items():
        for scan in sorted(folder.glob("*.npz")) if folder else []:
            if f"{name}/{scan.stem}" not in skip:
                out[f"data/{name}/{scan.name}"] = scan
    return out


def send(pod: Pod, files: dict, streams: int = 8) -> None:
    """Send ``files`` ({path under WORK: local path}) to the pod as ``streams`` tars at once, over separate ssh
    connections: one ssh stream tops out well below a fast home connection over a long round trip."""
    import threading

    sizes = {k: v.stat().st_size for k, v in files.items()}
    total = sum(sizes.values())
    lists = [[] for _ in range(max(1, min(streams, len(files))))]
    loads = [0] * len(lists)
    for name in sorted(files, key=sizes.get, reverse=True):  # largest first onto the lightest list
        k = loads.index(min(loads))
        lists[k].append(name)
        loads[k] += sizes[name]
    say(f"sending {len(files)} files, {total / 1e9:.1f} GB, over {len(lists)} connections")
    lock = threading.Lock()
    progress = {"files": 0, "bytes": 0}
    errors = []

    def one(names):
        proc = subprocess.Popen(pod.base + [f"mkdir -p {WORK} && tar -xf - -C {WORK}"], stdin=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|", bufsize=1 << 20) as tar:
                for name in names:
                    if errors:
                        break
                    tar.add(str(files[name]), arcname=name, recursive=False)
                    with lock:
                        progress["files"] += 1
                        progress["bytes"] += sizes[name]
            proc.stdin.close()
        except OSError as error:  # a broken pipe: ssh dropped
            errors.append(f"the copy broke off ({error})")
            proc.kill()
        if proc.wait() != 0 and not errors:
            errors.append("the copy failed on the pod")

    threads = [threading.Thread(target=one, args=(names,), daemon=True) for names in lists]
    for thread in threads:
        thread.start()
    started = time.time()
    while any(t.is_alive() for t in threads):
        for thread in threads:
            thread.join(timeout=60 / len(threads))
        with lock:
            k, done = progress["files"], progress["bytes"]
        rate = done / max(time.time() - started, 1e-6)
        say(f"sent {k}/{len(files)} files, {done / 1e9:.1f} of {total / 1e9:.1f} GB, {rate / 1e6:.1f} MB/s "
            f"({rate * 8 / 1e6:.0f} Mbit/s), about {hours((total - done) / max(rate, 1))} left")
    if errors:
        raise SystemExit(f"{errors[0]}; run the same command again to send the rest")


def deliver(pod: Pod, files: dict, streams: int) -> None:
    """Send scans ({path under WORK: local path}) into a staging folder on the pod, then move each one that passes
    its CRC check into place: a running train.py only ever sees whole scans. Failed ones are sent once more."""
    for attempt in range(2):
        pod.pod("unstage")
        send(pod, {f"{STAGING}/{k}": v for k, v in files.items()}, streams)
        bad = pod.pod("adopt", data=json.dumps(list(files)).encode()).get("bad", [])
        if not bad:
            return
        say(f"{len(bad)} scans failed their CRC check on the pod" + ("; sending them again" if not attempt else ""))
        files = {k: files[k] for k in bad}
    raise SystemExit(f"still failing the CRC check: {sorted(files)}; check these files here")


def add(pod: Pod, args, folders) -> None:
    """Send new scans in ``folders`` (under --root) to a pod, also while it trains, every --every minutes."""
    if not folders:
        raise SystemExit("after --, name the folders (under --root) to send new scans from, e.g. -- data_mixed")
    while True:
        have = pod.pod("list")
        cutoff = time.time() - 120  # leave alone scans still being written here
        todo = {}
        for name in folders:
            local = Path(name) if Path(name).is_absolute() else args.root / name
            if not local.is_dir():
                raise SystemExit(f"no folder {local}")
            todo.update({f"data/{local.name}/{f.name}": f for f in sorted(local.glob("*.npz"))
                         if f.stat().st_mtime < cutoff and have.get(f"data/{local.name}/{f.name}") != f.stat().st_size})
        if todo:
            deliver(pod, todo, args.streams)
            say(f"added {len(todo)} scans; the run picks them up at its next file refresh (--refresh-every)")
        else:
            say("no new scans")
        if not args.every:
            return
        time.sleep(args.every * 60)


def push(pod: Pod, args, train_args) -> list:
    folders, remote_args, extras, skip = train_inputs(train_args, args.root)
    scans = scan_list(folders, skip)
    if args.code_dir:
        for name in CODE_FILES:
            extras[f"inputs/code/{name}"] = args.code_dir / name
    say(f"{len(scans)} scans in {len(folders)} folders ({', '.join(folders)}), {len(skip)} excluded")
    info = pod.pod("setup")
    have = pod.pod("list")
    for name in (n for n, f in folders.items() if f is None):
        count = sum(k.startswith(f"data/{name}/") for k in have)
        if not count:
            raise SystemExit(f"no folder {name} here under {args.root} or on the pod")
        say(f"{name}: {count} scans made on the pod")
    todo = {k: v for k, v in scans.items() if have.get(k) != v.stat().st_size}
    need = sum(v.stat().st_size for v in todo.values())
    if need > info["disk_free"] - 2e9:
        raise SystemExit(f"The pod has {info['disk_free'] / 1e9:.0f} GB free in {WORK} but the scans need "
                         f"{need / 1e9:.0f} GB more. In RunPod, edit the pod and raise its Volume Disk to at least "
                         f"{math.ceil((need + have.get('_bytes', 0)) / 1e9 * 1.2 + 10)} GB (the pod restarts; "
                         f"/workspace is kept), then run this again.")
    say(f"{len(scans) - len(todo)} scans already on the pod, {len(todo)} to send")
    if extras:
        send(pod, extras, args.streams)
    if todo:
        deliver(pod, todo, args.streams)
    names = json.dumps(list(scans)).encode()
    for _ in range(2):
        bad = pod.pod("check", data=names).get("bad", [])
        if not bad:
            say("every scan on the pod passed its CRC check")
            break
        say(f"{len(bad)} scans failed the CRC check on the pod; sending them again")
        deliver(pod, {k: scans[k] for k in bad}, args.streams)
    else:
        raise SystemExit(f"still failing the CRC check: {bad}; check these files here")
    return remote_args


def start(pod: Pod, args, remote_args) -> dict:
    words = ["train", args.run, "--end", args.end, "--grace", str(args.grace), "--ref", args.ref]
    if args.code_dir:
        words.append("--local-code")
    state = pod.pod(*words, "--", *remote_args)
    say(state.get("message", ""))
    return state


def follow(pod: Pod, args) -> dict:
    """Print the progress until the run ends; returns its last status."""
    shown, misses = None, 0
    while True:
        state = pod.pod("status", args.run, check=False)
        if not state:
            misses += 1
            if misses * args.poll > 3600:
                raise SystemExit("lost the pod for an hour; check it in RunPod's console")
            say("the pod isn't answering; trying again")
            time.sleep(args.poll)
            continue
        misses = 0
        if state["state"] in ("done", "failed", "stopped", "none"):
            return state
        line = describe(state, args.price)
        if line != shown:
            say(line)
            shown = line
        time.sleep(args.poll)


def describe(state: dict, price: float) -> str:
    step, steps = state.get("step") or 0, state.get("steps") or 0
    text = f"step {step:,} of {steps:,}"
    if state.get("rate"):
        text += f", {state['rate']:.3f} s/step"
        if state.get("data_wait") is not None:
            text += f" ({state['data_wait']:.3f} of it waiting for crops)"
        left = (steps - step) * state["rate"] * 1.05  # + validation
        text += f", about {hours(left)} left"
        if price:
            text += (f", ${state['elapsed'] / 3600 * price:.2f} so far "
                     f"(about ${(state['elapsed'] + left) / 3600 * price:.2f} for the run)")
    if state.get("gpu_util") is not None:
        text += f", GPU {state['gpu_util']:.0f}% busy"
    if state.get("val"):
        text += f"; {state['val']}"
    return text


def fetch(pod: Pod, args, final: bool) -> Path:
    """Copy runs/RUN here and check the checkpoints' md5s; ``final`` tells the pod it may stop."""
    dest, run = args.dest, args.run
    for _ in range(3):
        sums = pod.pod("md5", run)
        proc = subprocess.Popen(pod.base + [f"tar -cf - --exclude='*.tmp' --exclude='*.pid' --exclude=FETCHED "
                                            f"-C {WORK}/runs {shlex.quote(run)}"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    parts = PurePosixPath(member.name).parts
                    if not member.isfile() or not parts or parts[0] != run or ".." in parts:
                        continue
                    target = dest.joinpath(*parts)
                    make_dir(target.parent)
                    part = target.with_name(target.name + ".part")
                    with tar.extractfile(member) as source, open(part, "wb") as out:
                        shutil.copyfileobj(source, out, 1 << 20)
                    os.replace(part, target)
        except tarfile.ReadError as error:
            raise SystemExit(f"nothing came back for {run} ({error}); is the run name right?")
        if proc.wait() != 0:
            raise SystemExit("copying the run back failed; run fetch again")
        wrong = [name for name, value in sums.items() if md5(dest / run / name) != value]
        if not wrong:
            break
        say(f"md5 mismatch for {wrong} (a checkpoint saved during the copy?); copying again")
    else:
        raise SystemExit("the checkpoints keep arriving different from the pod's; fetch again")
    say(f"run copied to {dest / run} ({', '.join(f'{k} md5 {v}' for k, v in sums.items())})")
    if final:
        pod.run(f"touch {WORK}/runs/{shlex.quote(run)}/FETCHED")
        if args.end != "keep":
            say(f"the pod will {args.end} itself within a minute (training is over and the run is here)")
    return dest / run


def make_dir(path: Path) -> None:
    """mkdir -p that survives Windows network shares, where creating a folder that exists can raise
    FileExistsError (WinError 183) despite exist_ok."""
    for folder in [*reversed(path.parents), path]:
        try:
            folder.mkdir(exist_ok=True)
        except FileExistsError:
            if not folder.is_dir():
                raise
        except PermissionError:  # a network share's root can be entered but not created
            if not folder.exists():
                raise


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make(pod: Pod, args, specs) -> None:
    """Make scans on the pod (make_data.py from --ref, Tangle built there) and follow them until done."""
    if not specs:
        raise SystemExit("after --, give FOLDER:FIRST:COUNT[:make_data flags, comma-separated] for each batch")
    words = ["make", args.run, "--ref", args.ref, "--jobs", str(args.jobs), "--backend", args.backend]
    say(pod.pod(*words, "--", *specs).get("message", ""))
    shown = None
    while True:
        state = pod.pod("make-status", args.run, check=False)
        if not state:
            say("the pod isn't answering; trying again")
        else:
            line = f"{state['phase']}: " + ", ".join(f"{f} {c['made']}/{c['wanted']}" for f, c in state["folders"].items())
            if state.get("per_scan"):
                line += (f"; {state['per_scan']:.0f} s per scan with {state['jobs']} at a time "
                         f"({args.price * state['per_scan'] / 3600:.3f} $/scan), {state['failed']} failed or skipped")
            if line != shown:
                say(line)
                shown = line
            if state["phase"] in ("done", "failed"):
                if state["phase"] == "failed":
                    raise SystemExit(f"making scans failed; see {WORK}/make/{args.run}/supervisor.log on the pod")
                return
        time.sleep(min(args.poll, 120))


def home(args, train_args):
    pod = Pod(args.ssh)
    pod.install()
    if args.command == "make":
        make(pod, args, train_args)
        return
    if args.command == "add":
        add(pod, args, train_args)
        return
    if args.command == "status":
        say(json.dumps(pod.pod("status", args.run), indent=1))
        return
    if args.command == "stop":
        pod.pod("stop", args.end if args.end != "keep" else "stop")
        return
    if args.command == "fetch":
        fetch(pod, args, final=pod.pod("status", args.run).get("state") == "done")
        return
    if args.command in ("go", "push"):
        remote_args = push(pod, args, train_args)
        if args.command == "push":
            return
    if args.command in ("go", "train"):
        if args.command == "train":
            remote_args = train_inputs(train_args, args.root)[1]
        state = start(pod, args, remote_args)
        if args.command == "train":
            return
        if state["state"] == "done":
            fetch(pod, args, final=True)
            return
    state = follow(pod, args)  # go, wait
    if state["state"] not in ("done", "failed"):
        raise SystemExit(f"the run is {state['state']} (the pod restarted, or the run was never started); run go "
                         "again to continue it from its last checkpoint")
    folder = fetch(pod, args, final=state["state"] == "done")
    if state["state"] == "failed":
        for name in ("supervisor.log", "train.log"):
            if (folder / name).exists():
                print(f"--- end of {folder / name}")
                print("\n".join((folder / name).read_text(errors="replace").splitlines()[-25:]))
        raise SystemExit(f"train.py stopped with exit code {state.get('exit')} (logs above). The pod waits "
                         f"{args.grace:g} min for a rerun, then ends itself.")
    say("done")


# ------------------------------------------------------------------------------------------------- on the pod


def pod_env() -> None:
    """Fill in RunPod's variables (pod id, API key) when this shell didn't read them."""
    try:
        for line in Path("/etc/rp_environment").read_text().splitlines():
            m = re.match(r'(?:export\s+)?(\w+)="?(.*?)"?$', line.strip())
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2)
    except OSError:
        pass


def cpus() -> int:
    """The pod's CPU allowance (the container may see all of the host's cores)."""
    if os.environ.get("RUNPOD_CPU_COUNT", "").isdigit():
        return int(os.environ["RUNPOD_CPU_COUNT"])
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            words = Path(quota_file).read_text().split()
            quota, period = words[0], (words[1] if period_file is None else Path(period_file).read_text().strip())
            if quota not in ("max", "-1"):
                return max(1, math.ceil(int(quota) / int(period)))
        except (OSError, ValueError, IndexError):
            pass
    return len(os.sched_getaffinity(0))


def loader_workers() -> int:
    """Crop workers for train.py: one per core, as long as shared memory holds their crops (~150 MB each, two
    queued per worker)."""
    shm = shutil.disk_usage("/dev/shm").free if os.path.isdir("/dev/shm") else 0
    return max(1, min(cpus(), int((shm - 1e9) // 350e6), 32))


def pid_alive(path: Path) -> bool:
    """Whether the pod.py process in ``path`` still runs (a restarted pod reuses process ids)."""
    try:
        return b"pod.py" in Path(f"/proc/{int(path.read_text())}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False


def stop_pod(how: str) -> None:
    pod_env()
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("no RUNPOD_POD_ID: not on a RunPod pod, nothing to stop", flush=True)
        return
    verb = "delete" if how == "terminate" else "stop"
    for command in (["runpodctl", "pod", verb, pod_id],  # current runpodctl, then the older spelling
                    ["runpodctl", "remove" if verb == "delete" else "stop", "pod", pod_id]):
        print("$ " + " ".join(command), flush=True)
        try:
            if subprocess.run(command).returncode == 0:
                return
        except OSError as error:  # no runpodctl
            print(error, flush=True)
            break
    print("could not stop the pod; stop it in RunPod's console", flush=True)


def on_pod(argv):
    pod_env()
    rest = argv[argv.index("--") + 1:] if "--" in argv else []  # train.py's arguments
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(prog="pod.py _pod")
    sub = parser.add_subparsers(dest="what", required=True)
    for name in ("setup", "list", "check", "adopt", "unstage"):
        sub.add_parser(name)
    for name in ("status", "md5"):
        sub.add_parser(name).add_argument("run")
    for name in ("train", "supervise"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--end", default="stop")
        p.add_argument("--grace", type=float, default=60)
        p.add_argument("--ref", default="claude/ct-unet")
        p.add_argument("--local-code", action="store_true")
        p.add_argument("--attempt", default="")
    sub.add_parser("stop").add_argument("how", nargs="?", default="stop")
    for name in ("make", "make-run"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--ref", default="claude/ct-unet")
        p.add_argument("--jobs", type=int, default=0)
        p.add_argument("--backend", default="auto")
        p.add_argument("--attempt", default="")
    sub.add_parser("make-status").add_argument("run")
    args = parser.parse_args(argv)
    args.rest = rest
    work = Path(WORK)
    if args.what == "setup":
        print(json.dumps(pod_setup(work)))
    elif args.what == "list":
        files = {str(p.relative_to(work)): p.stat().st_size for p in (work / "data").glob("*/*.npz")}
        files["_bytes"] = sum(files.values())
        print(json.dumps(files))
    elif args.what == "adopt":
        print(json.dumps({"bad": adopt(work, json.load(sys.stdin))}))
    elif args.what == "unstage":
        shutil.rmtree(work / STAGING, ignore_errors=True)
        print("{}")
    elif args.what == "check":
        print(json.dumps({"bad": crc_check(work, json.load(sys.stdin))}))
    elif args.what == "md5":
        run = work / "runs" / args.run
        print(json.dumps({p.name: md5(p) for p in (run / "best.pt", run / "last.pt") if p.exists()}))
    elif args.what == "status":
        print(json.dumps(run_status(work / "runs" / args.run)))
    elif args.what == "train":
        print(json.dumps(launch(work, args)))
    elif args.what == "supervise":
        supervise(work, args)
    elif args.what == "stop":
        stop_pod(args.how)
        print("{}")
    elif args.what == "make":
        print(json.dumps(make_launch(work, args)))
    elif args.what == "make-run":
        make_run(work, args)
    elif args.what == "make-status":
        print(json.dumps(make_status(work / "make" / args.run)))


def pod_setup(work: Path) -> dict:
    import importlib.util

    for folder in ("data", "inputs", "runs"):
        (work / folder).mkdir(parents=True, exist_ok=True)
    missing = [m for m in ("numpy", "scipy") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"installing {' '.join(missing)}", flush=True)
        env = {**os.environ, "PIP_BREAK_SYSTEM_PACKAGES": "1", "PIP_ROOT_USER_ACTION": "ignore"}
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], env=env, check=True,
                       stdout=sys.stderr)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch on the pod sees no CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    shm = shutil.disk_usage("/dev/shm").total if os.path.isdir("/dev/shm") else 0
    info = {"gpu": props.name, "gpu_gb": round(props.total_memory / 1e9), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cpus": cpus(), "workers": loader_workers(),
            "shm_gb": round(shm / 1e9, 1), "disk_free": shutil.disk_usage(work).free,
            "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9)}
    print(f"pod: {info['gpu']} ({info['gpu_gb']} GB), {info['cpus']} CPUs, {info['ram_gb']} GB RAM, "
          f"{info['shm_gb']} GB shared memory, {info['disk_free'] / 1e9:.0f} GB free in {work}; "
          f"PyTorch {info['torch']}, CUDA {info['cuda']}", flush=True)
    if info["workers"] < info["cpus"]:
        print(f"note: shared memory limits the crop workers to {info['workers']}", flush=True)
    return info


def crc_check(work: Path, names) -> list:
    """Zip CRC test of each scan (npz files are zips), in parallel; ``names`` are paths under ``work``. Scans that
    passed before (same size and time) are not read again."""
    from concurrent.futures import ProcessPoolExecutor

    record = work / "data" / ".crc_ok.json"
    ok = json.loads(record.read_text()) if record.exists() else {}
    paths = [work / n for n in names]
    stamp = {str(p): [p.stat().st_size, p.stat().st_mtime] for p in paths if p.exists()}
    todo = [p for p in paths if str(p) in stamp and ok.get(str(p)) != stamp[str(p)]]
    with ProcessPoolExecutor(max(1, cpus())) as pool:
        results = list(pool.map(_crc_one, map(str, todo), chunksize=8))
    bad = [str(p.relative_to(work)) for p, good in zip(todo, results) if not good]
    bad += [n for n in names if str(work / n) not in stamp]
    ok.update({str(p): stamp[str(p)] for p, good in zip(todo, results) if good})
    record.write_text(json.dumps(ok))
    print(f"CRC-checked {len(todo)} scans ({len(paths) - len(todo)} checked before or missing): {len(bad)} bad",
          flush=True)
    return bad


def adopt(work: Path, names) -> list:
    """Move staged scans that pass their CRC check into place (a rename on the same disk, so train.py never sees
    part of a file); returns the ones that failed, which are deleted."""
    from concurrent.futures import ProcessPoolExecutor

    staged = [work / STAGING / n for n in names]
    with ProcessPoolExecutor(max(1, cpus())) as pool:
        results = list(pool.map(_crc_one, map(str, staged), chunksize=8))
    record = work / "data" / ".crc_ok.json"
    ok = json.loads(record.read_text()) if record.exists() else {}
    bad = []
    for name, source, good in zip(names, staged, results):
        if not good:
            bad.append(name)
            source.unlink(missing_ok=True)
            continue
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        ok[str(target)] = [target.stat().st_size, target.stat().st_mtime]
    record.write_text(json.dumps(ok))
    shutil.rmtree(work / STAGING, ignore_errors=True)
    print(f"moved {len(names) - len(bad)} scans into place, {len(bad)} failed their CRC check", flush=True)
    return bad


def _crc_one(path: str) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def launch(work: Path, args) -> dict:
    run = work / "runs" / args.run
    state = run_status(run)
    if state["state"] == "running":
        return {**state, "message": f"{args.run} is already training on the pod; following it"}
    if state["state"] == "done":
        return {**state, "message": f"{args.run} already finished on the pod"}
    run.mkdir(parents=True, exist_ok=True)
    for marker in ("EXIT", "FETCHED"):
        (run / marker).unlink(missing_ok=True)
    attempt = time.strftime("%Y%m%d-%H%M%S")
    (run / "ATTEMPT").write_text(attempt)
    command = [sys.executable, os.path.abspath(__file__), "_pod", "supervise", args.run, "--end", args.end,
               "--grace", str(args.grace), "--ref", args.ref, "--attempt", attempt,
               *(["--local-code"] if args.local_code else []), "--", *args.rest]
    with open(run / "supervisor.log", "a") as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=work)
    (run / "supervisor.pid").write_text(str(proc.pid))
    time.sleep(10)
    if proc.poll() is not None or (run / "EXIT").exists():
        log_tail = (run / "supervisor.log").read_text().splitlines()[-15:]
        train_tail = (run / "train.log").read_text().splitlines()[-15:] if (run / "train.log").exists() else []
        raise SystemExit("training did not start:\n" + "\n".join(log_tail + train_tail))
    return {"state": "running", "message": f"{args.run} started on the pod (from last.pt if it has one)"}


def place_code(work: Path, run: Path, ref: str, local: bool) -> Path:
    """The run's own copy of train.py and maps.py: kept from an earlier attempt, else this computer's (--code-dir)
    or the repo's at ``ref``."""
    import urllib.request

    code = run / "code"
    if all((code / n).exists() for n in CODE_FILES):
        return code
    code.mkdir(exist_ok=True)
    for name in CODE_FILES:
        if local:
            shutil.copy(work / "inputs" / "code" / name, code / name)
        else:
            with urllib.request.urlopen(f"{REPO}/{ref}/{CODE_PATH}/{name}", timeout=60) as response:
                (code / name).write_bytes(response.read())
    (code / "SOURCE").write_text("--code-dir (the sending computer's copy)\n" if local else f"{REPO}/{ref}\n")
    return code


def supervise(work: Path, args) -> None:
    """Run train.py, then wait for the run to be fetched (or ``--grace`` minutes) and stop the pod."""
    import resource

    run = work / "runs" / args.run
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S ")  # noqa: E731
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else 1 << 20, hard))
    except (ValueError, OSError):
        pass  # crop workers hand tensors over as open files; the default limit is usually enough
    try:
        code = place_code(work, run, args.ref, args.local_code)
        workers = [] if "--workers" in args.rest else ["--workers", str(loader_workers())]
        command = [sys.executable, str(code / "train.py"), *args.rest[:2], str(run), *args.rest[2:], *workers,
                   "--resume"]
        print(stamp() + " ".join(command), flush=True)
        with open(run / "gpu.csv", "a") as gpu_log, open(run / "train.log", "a") as log:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                                        "--format=csv,noheader", "-l", "30"], stdout=gpu_log,
                                       stderr=subprocess.DEVNULL)
            train = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"}, cwd=code)
            (run / "train.pid").write_text(str(train.pid))
            exit_code = train.wait()
            monitor.terminate()
    except Exception as error:  # anything that keeps train.py from running still ends in EXIT and a stop
        print(stamp() + f"could not run train.py: {error!r}", flush=True)
        exit_code = 1
    (run / "EXIT").write_text(f"{exit_code}\n")
    print(stamp() + f"train.py exited with {exit_code}", flush=True)
    deadline = time.time() + args.grace * 60
    while time.time() < deadline and not (run / "FETCHED").exists():
        time.sleep(20)
        if (run / "ATTEMPT").read_text() != args.attempt:
            print(stamp() + "a newer attempt took over; leaving the pod to it", flush=True)
            return
    if args.end == "keep":
        print(stamp() + "leaving the pod running (--end keep)", flush=True)
        return
    fetched = (run / "FETCHED").exists()
    print(stamp() + ("the run was fetched" if fetched else f"nothing fetched the run in {args.grace:g} min")
          + f"; {args.end} the pod", flush=True)
    stop_pod(args.end)


# Scans made on the pod: Tangle built from the repo at --ref, make_data.py run several at a time.
EXAMPLES = "crates/tangle_python/python/examples"
PROBE = """
import sys
import tangle
from tangle.units import um
cell = tangle.Cell([60 * um] * 3, periodic="xyz")
fiber = tangle.Material("fiber", diameter=8 * um, min_bend_radius=40 * um)
recipe = tangle.Recipe(cell)
recipe.insert(tangle.FiberPopulation(material=fiber, count=10, length=(30 * um, 50 * um)))
print(recipe.run(tangle.RelaxationSettings(backend=sys.argv[1], max_iterations=50)))
"""


def make_specs(rest) -> list:
    """FOLDER:FIRST:COUNT[:flags] as (folder, first, count, [flags])."""
    out = []
    for spec in rest:
        parts = spec.split(":", 3)
        if len(parts) < 3:
            raise SystemExit(f"{spec!r}: expected FOLDER:FIRST:COUNT[:flag,flag]")
        out.append((parts[0], int(parts[1]), int(parts[2]), parts[3].split(",") if len(parts) > 3 and parts[3] else []))
    return out


def make_launch(work: Path, args) -> dict:
    job = work / "make" / args.run
    if job.exists() and pid_alive(job / "supervisor.pid"):
        return {"message": f"{args.run} is already making scans on the pod; following it"}
    job.mkdir(parents=True, exist_ok=True)
    make_specs(args.rest)
    (job / "specs.json").write_text(json.dumps(args.rest))
    for marker in ("DONE", "FAILED"):
        (job / marker).unlink(missing_ok=True)
    command = [sys.executable, os.path.abspath(__file__), "_pod", "make-run", args.run, "--ref", args.ref,
               "--jobs", str(args.jobs), "--backend", args.backend, "--", *args.rest]
    with open(job / "supervisor.log", "a") as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=work)
    (job / "supervisor.pid").write_text(str(proc.pid))
    return {"message": f"{args.run}: building Tangle (if needed) and making scans on the pod"}


def build_tangle(work: Path, ref: str, log) -> Path:
    """The repo at ``ref`` in WORK/tangle, with its Python package built and installed (once per commit)."""
    repo = work / "tangle"
    sh = lambda command, **kw: subprocess.run(command, shell=True, check=True, stdout=log,  # noqa: E731
                                              stderr=subprocess.STDOUT, executable="/bin/bash", **kw)
    if not (repo / ".git").exists():
        sh(f"git clone -q https://github.com/SueHeir/tangle.git {repo}")
    sh(f"git -C {repo} fetch -q origin {shlex.quote(ref)} && git -C {repo} checkout -q -f FETCH_HEAD")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    built = repo / ".built"
    if built.exists() and built.read_text().strip() == commit:
        return repo
    env = {**os.environ, "PIP_BREAK_SYSTEM_PACKAGES": "1", "PIP_ROOT_USER_ACTION": "ignore",
           "PATH": f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}"}
    if shutil.which("cc") is None or shutil.which("pkg-config") is None:
        sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential pkg-config "
           "libssl-dev libzstd-dev git curl", env=env)
    if shutil.which("cargo", path=env["PATH"]) is None:
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", env=env)
    sh(f"{sys.executable} -m pip install -q 'maturin>=1.8,<2' tifffile", env=env)
    wheels = work / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    cuda = "--features cuda" if "\ncuda = " in (repo / "crates/tangle_python/Cargo.toml").read_text() else ""
    sh(f"cd {repo} && {sys.executable} -m maturin build -q --release -m crates/tangle_python/Cargo.toml {cuda} "
       f"-i {sys.executable} -o {wheels}", env=env)
    sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps {wheels}/*.whl", env=env)
    built.write_text(commit + "\n")
    return repo


def gpu_env() -> dict:
    """The environment with the NVIDIA libraries pip installed for PyTorch (NVRTC among them) on the library path,
    for Tangle's CUDA backend."""
    import site

    dirs = [str(d) for base in site.getsitepackages() for d in sorted(Path(base).glob("nvidia/*/lib"))]
    path = ":".join(dirs + [p for p in [os.environ.get("LD_LIBRARY_PATH", "")] if p])
    return {**os.environ, "LD_LIBRARY_PATH": path}


def make_run(work: Path, args) -> None:
    """Build Tangle, pick the backend, then run make_data.py on index chunks, ``--jobs`` at a time."""
    job = work / "make" / args.run
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S ")  # noqa: E731
    phase = job / "PHASE"
    try:
        phase.write_text("building Tangle")
        with open(job / "build.log", "a") as log:
            repo = build_tangle(work, args.ref, log)
        env = gpu_env()
        backend = args.backend
        if backend == "auto":  # the first of CUDA, wgpu (Vulkan) that relaxes a small structure, else the CPU
            phase.write_text("testing the GPU relaxation")
            backend = "cpu"
            for trial in ("cuda", "wgpu"):
                try:
                    result = subprocess.run([sys.executable, "-c", PROBE, trial], timeout=600, capture_output=True,
                                            text=True, env=env)
                    ok = result.returncode == 0
                    print(stamp() + f"{trial}: {'works' if ok else result.stderr.strip()[-300:]}", flush=True)
                except subprocess.TimeoutExpired:
                    ok = False
                    print(stamp() + f"{trial}: no answer in 10 minutes", flush=True)
                if ok:
                    backend = trial
                    break
        jobs = args.jobs or max(1, cpus() // 2)
        print(stamp() + f"Tangle at {args.ref} ({(repo / '.built').read_text().strip()[:10]}), backend {backend}, "
              f"{jobs} at a time", flush=True)
        (job / "setup.json").write_text(json.dumps({"backend": backend, "jobs": jobs, "started": time.time()}))
        chunks = []
        for folder, first, count, flags in make_specs(args.rest):
            size = max(1, min(5, count // jobs))
            chunks += [(folder, i, min(size, first + count - i), flags) for i in range(first, first + count, size)]
        phase.write_text("making scans")
        script = repo / EXAMPLES / "ct_unet" / "make_data.py"
        env = {**env, "TANGLE_BACKEND": backend, "PYTHONPATH": str(repo / EXAMPLES),
               "RAYON_NUM_THREADS": str(max(1, cpus() // jobs)), "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        running = []
        while chunks or running:
            running = [r for r in running if r.poll() is None]
            while chunks and len(running) < jobs:
                folder, first, count, flags = chunks.pop(0)
                log = open(job / f"{folder}_{first}.log", "a")
                running.append(subprocess.Popen([sys.executable, str(script), str(work / "data" / folder), str(first),
                                                 str(count), *flags], stdin=subprocess.DEVNULL, stdout=log,
                                                stderr=subprocess.STDOUT, env=env, cwd=script.parent))
            time.sleep(5)
        phase.write_text("done")
        (job / "DONE").write_text(stamp() + "\n")
        print(stamp() + "all scans made", flush=True)
    except Exception as error:
        phase.write_text("failed")
        (job / "FAILED").write_text(f"{error!r}\n")
        print(stamp() + f"failed: {error!r}", flush=True)


def make_status(job: Path) -> dict:
    if not job.exists():
        return {"phase": "none", "folders": {}}
    phase = (job / "PHASE").read_text().strip() if (job / "PHASE").exists() else "starting"
    if phase not in ("done", "failed") and not pid_alive(job / "supervisor.pid"):
        phase = "stopped"
    folders, made = {}, 0
    for folder, first, count, _ in make_specs(json.loads((job / "specs.json").read_text())):
        have = sum((job.parents[1] / "data" / folder / f"varied_{i}.npz").exists() or
                   any((job.parents[1] / "data" / folder).glob(f"*_{i}.npz")) for i in range(first, first + count))
        entry = folders.setdefault(folder, {"made": 0, "wanted": 0})
        entry["made"] += have
        entry["wanted"] += count
        made += have
    logs = [p.read_text(errors="replace") for p in job.glob("*_*.log")]
    state = {"phase": phase, "folders": folders, "made": made,
             "failed": sum(t.count(": failed (") + t.count(": skipped (") for t in logs)}
    try:
        setup = json.loads((job / "setup.json").read_text())
        state["jobs"] = setup["jobs"]
        if made:
            state["per_scan"] = (time.time() - setup["started"]) * setup["jobs"] / made
    except (OSError, ValueError, KeyError):
        pass
    return state


def run_status(run: Path) -> dict:
    if not run.exists():
        return {"state": "none"}
    exit_file = run / "EXIT"
    if exit_file.exists():
        code = int(exit_file.read_text().strip() or 1)
        state = {"state": "done" if code == 0 else "failed", "exit": code}
    else:
        state = {"state": "running" if pid_alive(run / "supervisor.pid") else "stopped"}
    try:
        state["steps"] = json.loads((run / "config.json").read_text())["steps"]
    except (OSError, ValueError, KeyError):
        state["steps"] = None
    log, lines = run / "train.log", []
    if log.exists():
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 200_000))
            lines = f.read().decode(errors="replace").splitlines()
    # train.py's log lines: "step N files F <losses> lr X T s [data W s/step]", T counted from the attempt's start
    steps = [(int(m.group(1)), float(m.group(2)), m.group(3)) for m in
             (re.match(r"step (\d+) .* (\d+) s(?: data ([\d.]+) s/step)?$", line) for line in lines) if m]
    if steps:
        state["step"] = steps[-1][0]
        recent = steps[-1:]
        for s in reversed(steps[:-1]):  # this attempt's latest lines (a resumed run counts its time afresh)
            if len(recent) == 10 or not (s[0] < recent[0][0] and s[1] <= recent[0][1]):
                break
            recent.insert(0, s)
        if len(recent) >= 2:
            state["rate"] = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
        if steps[-1][2] is not None:
            state["data_wait"] = float(steps[-1][2])
    vals = [line.split() for line in lines if line.startswith("VAL step")]
    if vals:
        state["val"] = f"validation loss {vals[-1][4]} at step {int(vals[-1][2]):,}"
    try:
        started = time.mktime(time.strptime((run / "ATTEMPT").read_text(), "%Y%m%d-%H%M%S"))
        state["elapsed"] = time.time() - started
    except (OSError, ValueError):
        state["elapsed"] = 0.0
    try:
        rows = (run / "gpu.csv").read_text().strip().splitlines()[-10:]
        state["gpu_util"] = sum(float(r.split(",")[1].split()[0]) for r in rows) / len(rows)
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    if lines:
        state["last"] = lines[-1][-300:]
    return state


# ------------------------------------------------------------------------------------------------------- main


def main():
    if sys.argv[1:2] == ["_pod"]:
        return on_pod(sys.argv[2:])
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["go", "push", "train", "wait", "status", "fetch", "stop", "make", "add"])
    parser.add_argument("--ssh", required=True, help="the pod's 'SSH over exposed TCP' line, in quotes")
    parser.add_argument("--run", help="run name (its folder under runs/), e.g. r16")
    parser.add_argument("--root", type=Path, default=Path("."), help="the folder holding the training sets")
    parser.add_argument("--dest", type=Path, help="where the run folder comes back to (default ROOT/runs)")
    parser.add_argument("--ref", default="claude/ct-unet",
                        help="branch (or commit) of SueHeir/tangle whose train.py and maps.py the pod runs")
    parser.add_argument("--code-dir", type=Path,
                        help="send this computer's train.py and maps.py from this folder instead of --ref's")
    parser.add_argument("--end", choices=["stop", "terminate", "keep"], default="stop",
                        help="what the pod does once the run is fetched: stop (its disk and the scans stay, at "
                             "RunPod's idle-disk price), terminate (deletes the pod and its disk), keep (stays on)")
    parser.add_argument("--grace", type=float, default=60,
                        help="minutes the pod waits for the run to be fetched after training ends, then ends anyway")
    parser.add_argument("--poll", type=float, default=300, help="seconds between progress checks")
    parser.add_argument("--streams", type=int, default=8, help="push: ssh connections sending scans at once")
    parser.add_argument("--every", type=float, default=0,
                        help="add: look for new scans again every this many minutes, until stopped (Ctrl-C)")
    parser.add_argument("--jobs", type=int, default=0, help="make: scans made at a time (default: one per 2 cores)")
    parser.add_argument("--backend", choices=["auto", "cuda", "wgpu", "cpu"], default="auto",
                        help="make: Tangle's relaxation backend on the pod (auto: CUDA, else wgpu, else the CPU)")
    parser.add_argument("--price", type=float, default=1.59, help="the pod's $ per hour, for the cost estimate")
    args = parser.parse_args(argv)
    ours = {a for action in parser._actions for a in action.option_strings} - {"-h", "--help"}
    misplaced = sorted({w.split("=")[0] for w in train_args} & ours)
    if misplaced:
        parser.error(f"{', '.join(misplaced)} came after --, where train.py's arguments go; put them before --")
    if args.command not in ("stop", "add") and not args.run:
        parser.error("--run is required")
    if args.command in ("go", "push", "train") and not train_args:
        parser.error("give train.py's arguments after --")
    args.dest = args.dest or args.root / "runs"
    home(args, train_args)


if __name__ == "__main__":
    main()
