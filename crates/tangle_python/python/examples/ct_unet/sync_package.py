"""Copy the CT map network's finding code into the tangle package, where ``tangle.ct.find_fibers`` runs it.

    python sync_package.py            copy (after changing any of the files below)
    python sync_package.py --check    only say whether the package's copy is up to date

The scripts here stay the place to change this code: training and the tutorial's scripts run from this folder,
without Tangle built. ``tangle.ct.find_fibers`` runs the same code from the copy in ``tangle/ct/_network``, made by
this script: each file as it is here, with its imports of the others made package-relative and a first line saying
where it came from. ``tests/test_ct_network.py`` fails while the copy is out of date.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / "tangle" / "ct" / "_network"
MODULES = ("maps", "trace_maps", "fiber_types", "bonds", "diameters", "scan_io", "fiber_outputs", "find_fibers")
HEADER = "# Copied from examples/ct_unet/{name}.py by its sync_package.py: change that file, then run it again.\n"
_NAMES = "|".join(MODULES)
_IMPORT = re.compile(rf"^(\s*)import ({_NAMES})(\s+as\s+\w+)?\s*$")
_FROM = re.compile(rf"^(\s*)from ({_NAMES}) import ")


def package_source(name: str) -> str:
    """The package's copy of ``name``.py: the file here with its imports of the other copied modules relative."""
    lines = []
    for line in (HERE / f"{name}.py").read_text().splitlines():
        line = _IMPORT.sub(lambda m: f"{m[1]}from . import {m[2]}{m[3] or ''}", line)
        line = _FROM.sub(lambda m: f"{m[1]}from .{m[2]} import ", line)
        lines.append(line + "\n")
    return HEADER.format(name=name) + "".join(lines)


def stale() -> list[str]:
    """The modules whose package copy is missing or differs from what this script would write."""
    return [name for name in MODULES
            if not (PACKAGE / f"{name}.py").exists() or (PACKAGE / f"{name}.py").read_text() != package_source(name)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report whether the copy is up to date")
    args = parser.parse_args(argv)
    out_of_date = stale()
    if args.check:
        if out_of_date:
            print("tangle/ct/_network is out of date for: " + ", ".join(out_of_date) + ". Run: python sync_package.py")
            return 1
        print("tangle/ct/_network is up to date")
        return 0
    for name in out_of_date:
        (PACKAGE / f"{name}.py").write_text(package_source(name))
        print(f"copied {name}.py")
    if not out_of_date:
        print("tangle/ct/_network was already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
