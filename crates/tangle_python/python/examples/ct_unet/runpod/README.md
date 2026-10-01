# Training on a rented GPU (RunPod)

[`pod.py`](pod.py) trains the CT map network on a rented RunPod pod, driven from the computer that holds the
training sets. One command copies the scans to the pod, starts `train.py` there, prints the progress, copies the
finished run (weights, log, config, the code that trained them) back, and makes the pod stop itself so it stops
billing. Training needs only PyTorch, NumPy and SciPy on the pod; Tangle itself is not built there.

Only the training sets (synthetic scans) and the run go to the pod and back, over SSH. Nothing is uploaded
anywhere else.

## 1. Start a pod

On [runpod.io](https://www.runpod.io), Pods > Deploy:

- **GPU:** see [Which GPU](#which-gpu). An A100 80GB is a good default.
- **Template:** RunPod PyTorch, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8, CUDA 12.8).
- **On-Demand**, not Spot: a spot pod can be taken away mid-run (the run would resume, but only once you start
  it again).
- **Volume Disk:** at least 1.3 times the size of the training sets plus 10 GB (`pod.py` says so if it's too
  small). It is mounted at `/workspace` and kept while the pod is stopped. Container Disk: the default is fine.
- **SSH:** the computer that sends the scans needs its public key in RunPod > Settings > SSH Public Keys before
  the pod starts (or added to a running pod, below).

When it's running, open the pod's **Connect** tab and copy the **SSH over exposed TCP** line, for example
`ssh root@203.0.113.7 -p 40123 -i ~/.ssh/id_ed25519`. The other line (`...@ssh.runpod.io`) goes through
RunPod's proxy, which can't copy files.

If the sending computer's key isn't accepted (`pod.py` prints the exact line to paste), add it from any terminal
that can reach the pod:

```sh
echo 'ssh-ed25519 AAAA... name' >> ~/.ssh/authorized_keys
```

## 2. Train

On the computer with the training sets (Windows or macOS; Python 3.8+ and the built-in `ssh`, nothing to
install), from this folder:

```sh
python pod.py go --ssh "ssh root@IP -p PORT -i ~/.ssh/id_ed25519" --root C:\path\to\unet --run r16 -- data_a,data_web2,data_web2 val_a,val_web2 --steps 40000 --condition --exclude exclude_overlaps.txt
```

Everything after `--` is `train.py`'s usual command line without OUT: DATA and VAL are folders under `--root`
(or full paths), `--init` and `--exclude` files go along, and every other option is passed as it is. The run
folder comes back to `--dest/RUN` (default `ROOT/runs/RUN`).

`go` runs four steps, each also a command of its own (`push`, `train`, `wait`, `fetch`), and picks up where it
stopped when run again:

1. **push:** copies the scans the pod doesn't have yet (`.npz` files only; scans listed in `--exclude` stay
   home) over 8 ssh connections at once (`--streams`; one connection alone reached only ~80 Mbit/s from home),
   printing the speed and the time left, then checks every scan's zip checksum on the pod and sends any
   bad one again. A broken-off copy resumes where it stopped.
2. **train:** starts `train.py` on the pod in the background, so it keeps going if this computer sleeps or the
   connection drops. It always passes `--resume`: a run restarted after a pod restart continues from its last
   checkpoint (saved every 500 steps).
3. **wait:** prints the step, seconds per step, the time and cost left, how busy the GPU is and the last
   validation loss, every 5 minutes (`--poll`).
4. **fetch:** copies `runs/RUN` back and checks the checkpoints' md5s. The pod then stops itself.

Other commands: `status` (where the run is), `stop` (stop the pod now).

**The code that trains** is `train.py` and `maps.py` from the branch `--ref` (default `claude/ct-unet`),
downloaded on the pod. `--code-dir FOLDER` sends this computer's copies instead. Either way the run folder keeps
the copy it trained with, under `code/`.

**When the pod stops:** after the fetch (`--end stop`, the default), or `--grace` minutes (default 60) after
training ends if nothing fetched it, so a forgotten pod costs at most an hour. A stopped pod keeps its disk, and
the scans on it, for the next run; `--end terminate` deletes the pod instead, `--end keep` leaves it running.

## Which GPU

The network is small and trains one 128-voxel crop at a time. Before each step, a CPU core has to cut the crop
from a scan and compute its training targets (axis distances, directions, flips), which can take longer than
the GPU's step itself. `train.py` gets one crop worker per CPU core of the pod. So past a point, a faster GPU
just waits for crops, and **CPU cores matter as much as the GPU**.

The log shows which one is the limit: each progress line says how much of each step went into waiting for
crops (`data ... s/step` in `train.log`). If that is most of the step, the next pod should have more CPU cores,
not a faster GPU; if it is near zero, a faster GPU would help.

RunPod's on-demand prices and CPU cores per GPU, from [runpod.io/pricing](https://www.runpod.io/pricing) on
2026-10-01:

| GPU | $ / hour | CPU cores | RAM |
|---|---|---|---|
| A100 SXM 80GB | 1.59 | 16 | 125 GB |
| A100 PCIe 80GB | 1.59 | 8 | 117 GB |
| H100 PCIe 80GB | 2.89 | 16 | 188 GB |
| H100 SXM 80GB | 3.49 | 20 | 125 GB |
| L40S 48GB | 1.09 | 16 | 94 GB |
| RTX 4090 24GB | 0.74 | 6 | 41 GB |
| RTX 3090 24GB | 0.50 | 16 | 125 GB |

At the same price, the A100 SXM has twice the A100 PCIe's cores. The cheaper 16-core pods (L40S, RTX 3090) are
worth a try once the first run shows how much the crops hold the GPU back.

`--amp bf16` (A100, H100, L40S) runs the network's arithmetic in 16-bit, about twice as fast on the GPU and
with numbers that differ slightly from the 32-bit runs. It only helps when the GPU, not the crops, is the limit.

## Cost

| item | cost |
|---|---|
| the pod while it runs (copying, training, waiting) | its hourly price, e.g. $1.59 for an A100 |
| a stopped pod's disk | $0.20 per GB per month (100 GB: about $20 a month) until the pod is terminated |
| a training run | (steps x seconds per step + about 5% for the validation checks) x the hourly price |

Measured so far: on the Windows PC (RTX 3060 Ti), r13's 40,000 steps took 0.57 s per step, about 7.5 hours with
the validation checks. On an 18-core A100 80GB PCIe pod ($1.59/h), r16's 40,000 steps ran at 0.171 s per step
with `--amp bf16` (GPU about 70% busy, 0.03 s per step waiting for crops): about 2 hours, $3.20.

The first push to a new pod copies every scan over the home internet connection while the pod bills: 47 GB took
about 13 minutes at ~500 Mbit/s over 8 connections (one connection alone managed ~80 Mbit/s). Later runs on the same (stopped and
restarted) pod only send new scans. If training on RunPod becomes routine, a RunPod **network volume** keeps the
scans for $0.07 per GB per month (100 GB: $7) and any new pod in its data center can use them, so they are
copied once; `pod.py` works the same with one (attach it when deploying; it is mounted at `/workspace`).

## If something goes wrong

- **"Could not reach the pod":** wrong address, the pod is stopped, or the key isn't accepted (see step 1).
  An edited or restarted pod can get a new port: copy the line from the Connect tab again.
- **Not enough disk:** edit the pod, raise the Volume Disk, run the same command again.
- **train.py stopped:** `go` prints the end of `train.log`. The pod stays up for `--grace` minutes, so a fixed
  command can resume the run; then it stops itself.
- **"unrecognized arguments" on Windows** when started through PowerShell's `Start-Process`: it splits
  `--ssh "ssh root@... -p ..."` at the spaces. Give that argument its own double quotes inside the argument
  list (`'--ssh', '"ssh root@IP -p PORT -i KEY"'`), or run `python pod.py ...` directly in the terminal.
- **Watching on the pod itself:** `tail -f /workspace/unet/runs/RUN/train.log` in a pod terminal.
