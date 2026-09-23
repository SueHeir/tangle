"""Two fiber types in one scan: bright 7 µm fibers and dimmer 19 µm fibers
with a bright rim around a dim core.

Grey levels as in the scans this imitates: on a scale where void is 0 and the
7 µm fibers are 1, the 19 µm fibers' rims are about 0.75 and their cores about
0.25, so a large fiber's core overlaps the bright end of the background and
its rim overlaps the edges of the small fibers. No single threshold separates
them.

1. Generate and relax 7 µm and 19 µm fibers (equal volume of each) in a cell
   periodic in the fiber plane.
2. Render it with each type's brightness profile and crop the center.
3. Fit both types together, score each type against the ground truth, and
   write the fit, labels and overlay.

Needs NumPy and SciPy; tifffile and matplotlib add TIFF and PNG outputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import tangle
import tangle.ct as ct
from tangle.units import um

OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("output") / "ct_fit_two_types")
# WGPU runs on the local GPU (Metal on Apple silicon); set TANGLE_BACKEND=cpu
# on a machine without a usable GPU.
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")

CELL = 320 * um
CROP = 200 * um
VOXEL = 1.25 * um
LENGTH = (300 * um, 500 * um)
SMALL = dict(name="fine_7um", diameter=7 * um, count=106, profile=ct.CrossSection())
LARGE = dict(
    name="coarse_19um", diameter=19 * um, count=14,
    profile=ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3),  # core ≈ 0.25
)
TYPES = [SMALL, LARGE]  # truth types and fit types use this order


def material(kind: dict, loosen: float = 1.0) -> tangle.Material:
    return tangle.Material(kind["name"], diameter=kind["diameter"], min_bend_radius=loosen * 5 * kind["diameter"])


def load_or_make_truth() -> tangle.Assembly:
    cache = OUTPUT / "truth_centerlines.json"
    cell = tangle.Cell([CELL] * 3, periodic="xy")
    if not cache.exists():
        recipe = tangle.Recipe(cell)
        for seed, kind in enumerate(TYPES):
            population = tangle.FiberPopulation(
                material=material(kind),
                count=kind["count"],
                segments_per_fiber=16,  # keeps segments longer than a diameter
                seed=21 + seed,
                length=LENGTH,
                curvature_amplitude=(2 * um, 8 * um),
                orientation=tangle.PlanarOrientation(max_tilt=0.35),
            )
            recipe.insert(tangle.generate_fiber_population(cell, population), name=kind["name"])
        started = time.perf_counter()
        run = recipe.run(
            tangle.RelaxationSettings(
                backend=BACKEND, max_iterations=12_000, max_step=1.0 * um, penetration_tolerance=0.1 * um
            )
        )
        print(f"truth: {run} ({time.perf_counter() - started:.0f} s)")
        counts = [kind["count"] for kind in TYPES]
        cache.write_text(json.dumps({"counts": counts, "centerlines": run.centerlines()}) + "\n")
    data = json.loads(cache.read_text())
    assembly = tangle.Assembly(cell)
    start = 0
    for kind, count in zip(TYPES, data["counts"]):
        lines = data["centerlines"][start : start + count]
        start += count
        # Render with a 1% looser bend limit: float32 GPU relaxation can end a
        # hair past it, which export_puma's validator rejects.
        assembly.insert(tangle.FiberCollection.from_centerlines(lines, material(kind, loosen=0.99)), name=kind["name"])
    return assembly


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = [(kind["diameter"], kind["profile"]) for kind in TYPES]
    full = ct.synthetic_ct(load_or_make_truth(), VOXEL, seed=21, profiles=profiles)
    low = int(round((CELL - CROP) / 2 / VOXEL))
    high = low + int(round(CROP / VOXEL))
    scan = full.crop((low,) * 3, (high,) * 3)
    print(f"scan: {scan.volume.shape} voxels; {len(scan.centerlines)} true fiber pieces")

    specs = [
        ct.FiberSpec(
            diameter=kind["diameter"], min_bend_radius=5 * kind["diameter"], length=sum(LENGTH) / 2,
            profile=kind["profile"], name=kind["name"],
        )
        for kind in TYPES
    ]
    started = time.perf_counter()
    fit = ct.fit_fibers(scan.volume, VOXEL, specs, verbose=True)
    elapsed = time.perf_counter() - started
    report = ct.score(fit, scan)
    report["seconds"] = elapsed
    for key in ("fitted_fibers", "recovered", "split", "missed", "stubs", "false_fibers", "merged_fibers",
                "centerline_error_voxels", "voxel_label_accuracy", "seconds"):
        print(f"  {key}: {report[key]}")
    for kind, row in (report["per_type"] or {}).items():
        print(f"  {TYPES[kind]['name']}: {row}")
    fit.write(OUTPUT, volume=scan.volume)
    (OUTPUT / "score.json").write_text(json.dumps(report, indent=1) + "\n")
    ct.save_overlay_figure(OUTPUT / "truth_overlay.png", scan.volume, scan.labels, title="ground truth")


if __name__ == "__main__":
    main()
