"""Regenerate a training mix from its list of kept scans: every listed scan, and none of the ones cut from it.

    python regenerate_kept.py KEPT.txt SETS.json OUT_ROOT [--only data_mixed,val_mixed] [--jobs 4] [--shard 0/2]
                                                        [--dry-run]

KEPT.txt has one scan per line, as ``FOLDER/NAME`` (``data_mixed/mixed_100123``; ``.npz`` optional; blank lines
and ``#`` comments ignored): the scans a run actually trained and validated on, after its audit cuts and
overlap exclusions. SETS.json says how each folder was made (the family flag, ``--mixed`` etc., comes from each scan's name, so a
folder may hold several families; ``flags`` adds the rest):

    {"commit": "<git commit the scans were made with>",
     "folders": {"data_mixed": {"make_data_flags": ["--mixed", "--fast-relax"], "commit": "<per folder>"},
                 "data": {"make_data_flags": ["--scanned-share", "0.2"], "code": "... commit abc1234 ..."}, ...}}

(``flags`` works as well as ``make_data_flags``; a folder's commit is its ``commit``, else a ``commit <hash>``
named in its ``code`` note, else the top-level ``commit``.)

``make_data.py`` draws everything about a scan from its family and index alone, so a scan comes out the same
when the same code makes it again with the same flags. This script groups each folder's kept indices into runs
of consecutive indices and calls ``make_data.py OUT_ROOT/FOLDER FIRST COUNT FLAGS`` once per run: the cut
scans fall in the gaps between runs and are never made. The family flag comes from the scan name (``mixed_`` ->
``--mixed``; ``dense_hard_`` and ``scanned_`` need none, ``--scanned-share`` decides which an index is).

Reproduction is exact only with the code the scans were made with: check out that commit (SETS.json) first.
The script warns, per folder, when HEAD differs; folders made with different
commits (as on different branches) are best made one commit at a time with ``--only``. Some families live on branches that may not have merged yet (``edges_``:
PR #77, ``networks_``: PR #79). Flags that change the structures (``--fast-relax``, ``--no-overlap-check``)
must match too.

make_data.py skips scans already written, so a stopped run picks up where it stopped. ``--jobs`` runs that many
make_data processes at once (they share the GPU); ``--shard K/N`` takes every Nth run, to split the work over
machines. At the end it lists kept scans that did not appear (a structure that failed to relax or render with
this code, or a code mismatch), and any scan in OUT_ROOT that is not in KEPT.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY_FLAG = {"mixed": "--mixed", "varied": "--varied", "pairs": "--pairs", "hardpairs": "--hard-pairs",
               "bends": "--bends", "empty": "--empty", "edges": "--edges", "networks": "--networks",
               "dense_hard": None, "scanned": None}  # dense_hard / scanned: no flag; --scanned-share picks which


def split_name(name: str) -> tuple[str, int]:
    """``mixed_100123`` -> ("mixed", 100123); ``dense_hard_1001`` -> ("dense_hard", 1001)."""
    family, _, index = name.removesuffix(".npz").rpartition("_")
    if not family or not index.isdigit():
        raise ValueError(f"not a make_data scan name: {name!r}")
    return family, int(index)


def read_kept(path: Path) -> dict[str, list[tuple[str, int]]]:
    """Folder -> [(family, index)], from FOLDER/NAME lines."""
    kept: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        folder, _, name = line.rpartition("/")
        if not folder:
            raise SystemExit(f"{path}:{number}: expected FOLDER/NAME, got {line!r}")
        kept[folder].append(split_name(Path(name).name))
    return kept


def runs(indices: list[int]) -> list[tuple[int, int]]:
    """Sorted indices as (first, count) runs of consecutive indices."""
    out: list[tuple[int, int]] = []
    for i in sorted(set(indices)):
        if out and i == out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((i, 1))
    return out


def family_flag(family: str) -> list[str]:
    """The make_data flag that makes a family's scans (none for dense_hard and scanned)."""
    if family not in FAMILY_FLAG:
        raise SystemExit(f"unknown family {family!r}: add it to FAMILY_FLAG")
    flag = FAMILY_FLAG[family]
    return [flag] if flag else []


def head_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kept", type=Path, help="FOLDER/NAME per line: the scans to make")
    p.add_argument("sets", type=Path, help="per-folder make_data flags and the code commit (JSON)")
    p.add_argument("out_root", type=Path, help="the folders are made under here")
    p.add_argument("--only", help="comma-separated folders to make (default: all in KEPT.txt)")
    p.add_argument("--jobs", type=int, default=1, help="make_data processes at once")
    p.add_argument("--shard", default="0/1", help="K/N: make every Nth run from the Kth")
    p.add_argument("--dry-run", action="store_true", help="print the make_data commands only")
    args = p.parse_args()

    kept = read_kept(args.kept)
    args.out_root = args.out_root.resolve()
    sets = json.loads(args.sets.read_text())
    folders = sets.get("folders", {})
    only = set(args.only.split(",")) if args.only else None
    k, n = (int(v) for v in args.shard.split("/"))

    head = head_commit()

    def recorded(folder: str) -> str | None:
        """The commit a folder was made with: its "commit", or one named in its "code" note."""
        entry = folders.get(folder, {})
        found = re.search(r"\bcommit ([0-9a-f]{7,40})\b", str(entry.get("code", "")))
        return entry.get("commit") or (found.group(1) if found else None) or sets.get("commit")

    for folder in sorted(kept):
        if only and folder not in only:
            continue
        made_with = recorded(folder)
        if made_with is None:
            print(f"note: {folder}: no recorded commit; the scans may not come out the same", file=sys.stderr)
        elif head and not (head.startswith(made_with) or made_with.startswith(head)):
            print(f"warning: {folder} was made with {made_with}, HEAD is {head[:12]}: check that commit out to "
                  "remake it exactly", file=sys.stderr)

    commands: list[list[str]] = []
    for folder in sorted(kept):
        if only and folder not in only:
            continue
        entries = kept[folder]
        # the folder's other flags (--fast-relax, --scanned-share ...); family flags come from the scan names
        entry = folders.get(folder, {})
        given = entry.get("flags", entry.get("make_data_flags", []))
        extra = [f for f in given if f not in FAMILY_FLAG.values()]
        by_flag: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for family, index in entries:
            by_flag[tuple(family_flag(family))].append(index)
        for flag, indices in sorted(by_flag.items()):
            for first, count in runs(indices):
                commands.append([sys.executable, str(HERE / "make_data.py"), str(args.out_root / folder),
                                 str(first), str(count), *flag, *extra])
    commands = commands[k::n]
    print(f"{sum(len(v) for f, v in kept.items() if not only or f in only)} kept scans in "
          f"{len(commands)} runs (shard {k}/{n})", flush=True)

    if args.dry_run:
        for c in commands:
            print(" ".join(c[1:]))
        return

    def run(command: list[str]) -> int:
        return subprocess.run(command, cwd=HERE).returncode

    with ThreadPoolExecutor(max(args.jobs, 1)) as pool:
        failed = [c for c, code in zip(commands, pool.map(run, commands)) if code]
    for c in failed:
        print(f"make_data failed: {' '.join(c[1:])}", file=sys.stderr)

    missing, extra = [], []
    for folder, entries in sorted(kept.items()):
        if only and folder not in only:
            continue
        want = {f"{family}_{index}" for family, index in entries}
        have = {q.stem for q in (args.out_root / folder).glob("*.npz")}
        missing += [f"{folder}/{name}" for name in sorted(want - have)]
        extra += [f"{folder}/{name}" for name in sorted(have - want)]
    if n == 1:  # with shards, the other machines make the rest
        print(f"{len(missing)} kept scans missing" + "".join(f"\n  {m}" for m in missing[:50]), flush=True)
    if extra:
        print(f"{len(extra)} scans in OUT_ROOT not in KEPT.txt" + "".join(f"\n  {e}" for e in extra[:50]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
