"""Fit a large synthetic scan in tiles (``ct.fit_tiled``) and score it.

The scan is ``two_types`` from ``ct_examples.py`` made larger: 7 µm solid
fibers and 19 µm fibers with a bright rim and a dim core, at the same
density, in a cell periodic in the fiber plane, cropped to ``--side`` µm
(480 by default: 384 voxels a side at 1.25 µm). It is fitted from the raw
grey with measured grey profiles, as ``ct_examples.py`` does.

``--mode tiled`` (the default) fits it with ``ct.fit_tiled`` into a
checkpoint under the output folder; running again resumes from the saved
tiles (``--restart`` starts over). ``--mode whole`` fits the same scan with
``ct.fit_fibers`` in one go, for comparison. Each run adds its row
(seconds, peak memory of the process, score) to
``<output>/tiled_scan/summary.md`` and writes ``<mode>.json`` next to it.
Run the two modes as separate commands, so each peak memory is its own.

Usage::

    python ct_tiled_scan.py [--mode tiled|whole] [--side UM] [--tile VOXELS]
                            [--overlap VOXELS] [--redraw-passes N] [--restart]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path


import tangle
import tangle.ct as ct
from tangle.units import um

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct_examples import BACKEND, fit_input, planar_population, relaxed_truth, render_scan  # noqa: E402


def large_two_types(cache: Path, side_um: float) -> tuple[ct.SyntheticScan, list[ct.FiberSpec]]:
    voxel, length = 1.25 * um, (300 * um, 500 * um)
    cell_side = side_um * um + 160 * um
    density = (cell_side / (320 * um)) ** 2  # two_types: 106 fine and 14 coarse in a 320 µm cell
    fine = tangle.Material("fine_7um", diameter=7 * um, min_bend_radius=35 * um)
    coarse = tangle.Material("coarse_19um", diameter=19 * um, min_bend_radius=95 * um)
    cell = tangle.Cell([cell_side] * 3, periodic="xy")
    truth = relaxed_truth(
        cache,
        cell,
        [
            planar_population(fine, int(round(106 * density)), 21, length, 16),
            planar_population(coarse, int(round(14 * density)), 22, length, 16),
        ],
    )
    profiles = [
        (7 * um, ct.CrossSection()),
        (19 * um, ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3)),
    ]
    full = render_scan(truth, voxel, seed=21, profiles=profiles)
    low = int(round((cell_side - side_um * um) / 2 / voxel))
    size = int(round(side_um * um / voxel))
    scan = full.crop((low,) * 3, (low + size,) * 3)
    specs = [
        ct.FiberSpec(diameter=7 * um, min_bend_radius=35 * um, length=400 * um, name="fine_7um"),
        ct.FiberSpec(diameter=19 * um, min_bend_radius=95 * um, length=400 * um, name="coarse_19um"),
    ]
    return scan, specs


def peak_memory_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, KiB on Linux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("tiled", "whole"), default="tiled")
    parser.add_argument("--side", type=float, default=480.0, help="scan side in µm (default 480)")
    parser.add_argument("--tile", type=int, default=192, help="core size in voxels (default 192)")
    parser.add_argument("--overlap", type=int, default=None, help="padding in voxels (default fit_tiled's)")
    parser.add_argument("--redraw-passes", type=int, default=None, help="FitSettings.redraw_passes (default its own)")
    parser.add_argument("--restart", action="store_true", help="delete the saved tiles first")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("TANGLE_CT_OUTPUT", Path(__file__).parent / "output" / "ct")))
    args = parser.parse_args()

    folder = args.output / "tiled_scan"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scan, specs = large_two_types(args.output / ".cache" / f"tiled_scan_{args.side:g}.json", args.side)
    volume, specs, _ = fit_input(scan, specs, "grey")
    prepared = time.perf_counter() - started
    print(f"scan {scan.volume.shape} voxels, {len(scan.centerlines)} true pieces, built in {prepared:.0f} s")

    changes = {"backend": BACKEND}
    if args.redraw_passes is not None:
        changes["redraw_passes"] = args.redraw_passes
    settings = ct.FitSettings(**changes)
    started = time.perf_counter()
    if args.mode == "tiled":
        fit = ct.fit_tiled(
            volume, scan.voxel_size, specs, settings, tile=args.tile, overlap=args.overlap,
            checkpoint=folder / f"checkpoint_{args.side:g}_{args.tile}", restart=args.restart, verbose=True,
        )
    else:
        fit = ct.fit_fibers(volume, scan.voxel_size, specs, settings)
    seconds = time.perf_counter() - started

    report = ct.score(fit, scan)
    summary = {k: v for k, v in report.items() if k not in ("per_true_fiber", "per_type", "per_fitted_fiber")}
    tiles = fit.history[0] if args.mode == "tiled" else None
    row = {
        "mode": args.mode,
        "side (vox)": scan.volume.shape[0],
        "tile / overlap": f"{args.tile} / {tiles['overlap']}" if tiles else "-",
        "tiles": tiles["tiles"] if tiles else "-",
        "joins": tiles["joins"] if tiles else "-",
        "seconds": round(seconds, 1),
        "tile seconds (sum)": tiles["seconds"] if tiles else "-",
        "peak memory (GB)": round(peak_memory_gb(), 2),
        "true": summary["true_fibers_in_volume"],
        "fitted": summary["fitted_fibers"],
        "recovered": summary["recovered"],
        "split": summary["split"],
        "missed": summary["missed"],
        "false": summary["false_fibers"],
        "merged": summary["merged_fibers"],
        "label accuracy": round(summary["voxel_label_accuracy"], 3),
        "centerline F1": round(summary["centerline_f1"], 3) if summary["centerline_f1"] is not None else None,
    }
    print("  " + ", ".join(f"{k} {v}" for k, v in row.items()))
    (folder / f"{args.mode}.json").write_text(
        json.dumps({"row": row, "score": summary, "per_type": report["per_type"], "history": fit.history}, indent=1, default=str) + "\n"
    )
    table = folder / "summary.md"
    if not table.exists():
        table.write_text("| " + " | ".join(row) + " |\n|" + "---|" * len(row) + "\n")
    with table.open("a") as handle:
        handle.write("| " + " | ".join(str(v) for v in row.values()) + " |\n")


if __name__ == "__main__":
    main()
