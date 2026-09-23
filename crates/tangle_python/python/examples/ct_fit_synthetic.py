"""Fit fibers to a synthetic CT scan rendered from a Tangle config, and score it.

1. Generate and relax a wavy planar fiber population in a closed cell.
2. Render it as a CT-like scan (partial volume, blur, noise, drift) with
   per-voxel ground-truth fiber ids, using ``tangle.ct.synthetic_ct``.
3. Fit fibers to the scan knowing only the fiber diameter and bend radius.
4. Score the fit against the ground truth and write the Tangle config,
   the label volume and the per-fiber overlay.

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

OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("output") / "ct_fit_synthetic")
# WGPU runs on the local GPU (Metal on Apple silicon); set TANGLE_BACKEND=cpu
# on a machine without a usable GPU.
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")

CELL = 240 * um
DIAMETER = 12 * um
MIN_BEND_RADIUS = 4 * DIAMETER
VOXEL = 1.5 * um  # 8 voxels across a fiber


def make_truth() -> tangle.RunResult:
    cell = tangle.Cell([CELL] * 3)
    material = tangle.Material("synthetic fiber", diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    population = tangle.FiberPopulation(
        material=material,
        count=40,
        segments_per_fiber=12,  # segments shorter than a diameter stall the contact solve
        seed=7,
        length=(150 * um, 210 * um),
        curvature_amplitude=(2 * um, 8 * um),
        orientation=tangle.PlanarOrientation(max_tilt=0.35),
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(tangle.generate_fiber_population(cell, population), name="truth")
    return recipe.run(
        tangle.RelaxationSettings(
            backend=BACKEND,
            max_iterations=12_000,
            max_step=1.0 * um,
            penetration_tolerance=0.1 * um,
        )
    )


def load_or_make_truth() -> tangle.Assembly:
    """Relaxing the truth is the slow step, so its centerlines are cached."""
    cache = OUTPUT / "truth_centerlines.json"
    material = tangle.Material("synthetic fiber", diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    if not cache.exists():
        started = time.perf_counter()
        run = make_truth()
        print(f"truth: {run} ({time.perf_counter() - started:.0f} s)")
        cache.write_text(json.dumps({"diameter": DIAMETER, "centerlines": run.centerlines()}) + "\n")
    centerlines = json.loads(cache.read_text())["centerlines"]
    # float32 GPU relaxation can end a hair past the bend limit (a curvature
    # ratio of 1.00001), which export_puma's validator rejects. The rendering
    # doesn't depend on the limit, so render with a 1% looser one.
    material = tangle.Material("synthetic fiber", diameter=DIAMETER, min_bend_radius=0.99 * MIN_BEND_RADIUS)
    assembly = tangle.Assembly(tangle.Cell([CELL] * 3))
    assembly.insert(tangle.FiberCollection.from_centerlines(centerlines, material), name="truth")
    return assembly


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    truth = load_or_make_truth()

    scan = ct.synthetic_ct(truth, VOXEL, seed=7)
    print(f"scan: {scan.volume.shape} voxels, {DIAMETER / VOXEL:.0f} voxels per diameter")

    started = time.perf_counter()
    spec = ct.FiberSpec(diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    fit = ct.fit_fibers(scan.volume, VOXEL, spec, verbose=True)
    print(f"fit: {fit.fiber_count} fibers ({time.perf_counter() - started:.0f} s)")

    report = ct.score(fit, scan)
    for key, value in report.items():
        if key != "per_true_fiber":
            print(f"  {key}: {value}")

    paths = fit.write(OUTPUT, volume=scan.volume)
    ct.save_overlay_figure(OUTPUT / "truth_overlay.png", scan.volume, scan.labels, title="ground truth")
    (OUTPUT / "score.json").write_text(json.dumps(report, indent=1) + "\n")
    print("population for regenerating:", fit.suggested_population())
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
