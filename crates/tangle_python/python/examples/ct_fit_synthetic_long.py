"""Long fibers through a small scan: does the fiber-length prior stop over-splitting?

A real scan is a small window on fibers much longer than it, so most fibers
cross the scan boundary and fiber ends inside the scan are rare (PuMA's
FiberForm crop is like this). This example imitates that:

1. Generate and relax 154 wavy planar fibers of 12 µm diameter and 300-450 µm
   length in a 480 µm cell, periodic in the fiber plane. Periodicity keeps
   fiber ends spread uniformly; in a closed cell relaxation packs them
   against the walls and the central crop would see almost none.
2. Render the whole cell as a CT-like scan and crop the central 240 µm, so
   fibers run through the crop boundary.
3. Fit it three times: without and with ``FiberSpec(length=...)`` on the
   NumPy loop, and with the length prior on Tangle's own solver
   (``FitSettings(engine="tangle")``, the GPU by default). Score each against
   the ground truth, measure its geometry (bend limit, overlaps) and write
   the fits.

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

OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("output") / "ct_fit_synthetic_long")
# WGPU runs on the local GPU (Metal on Apple silicon); set TANGLE_BACKEND=cpu
# on a machine without a usable GPU.
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")

CELL = 480 * um
CROP = 240 * um
DIAMETER = 12 * um
MIN_BEND_RADIUS = 4 * DIAMETER
LENGTH = (300 * um, 450 * um)
VOXEL = 1.5 * um


def load_or_make_truth() -> tangle.Assembly:
    """Relaxing the truth is the slow step, so its centerlines are cached."""
    cache = OUTPUT / "truth_centerlines_periodic.json"
    material = tangle.Material("synthetic fiber", diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    cell = tangle.Cell([CELL] * 3, periodic="xy")
    if not cache.exists():
        population = tangle.FiberPopulation(
            material=material,
            count=154,  # about the fiber volume fraction of ct_fit_synthetic.py
            segments_per_fiber=24,  # keeps segments longer than a diameter
            seed=11,
            length=LENGTH,
            curvature_amplitude=(2 * um, 8 * um),
            orientation=tangle.PlanarOrientation(max_tilt=0.35),
        )
        recipe = tangle.Recipe(cell)
        recipe.insert(tangle.generate_fiber_population(cell, population), name="truth")
        started = time.perf_counter()
        run = recipe.run(
            tangle.RelaxationSettings(
                backend=BACKEND, max_iterations=12_000, max_step=1.0 * um, penetration_tolerance=0.1 * um
            )
        )
        print(f"truth: {run} ({time.perf_counter() - started:.0f} s)")
        cache.write_text(json.dumps({"diameter": DIAMETER, "centerlines": run.centerlines()}) + "\n")
    centerlines = json.loads(cache.read_text())["centerlines"]
    # float32 GPU relaxation can end a hair past the bend limit (a curvature
    # ratio of 1.00001), which export_puma's validator rejects. The rendering
    # doesn't depend on the limit, so render with a 1% looser one.
    material = tangle.Material("synthetic fiber", diameter=DIAMETER, min_bend_radius=0.99 * MIN_BEND_RADIUS)
    assembly = tangle.Assembly(cell)
    assembly.insert(tangle.FiberCollection.from_centerlines(centerlines, material), name="truth")
    return assembly


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full = ct.synthetic_ct(load_or_make_truth(), VOXEL, seed=11)
    low = int(round((CELL - CROP) / 2 / VOXEL))
    high = low + int(round(CROP / VOXEL))
    scan = full.crop((low,) * 3, (high,) * 3)
    print(f"scan: {scan.volume.shape} voxels cropped from {full.volume.shape}; {len(scan.centerlines)} true fiber pieces")

    base = ct.FiberSpec(diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    prior = base.replace(length=sum(LENGTH) / 2)
    numpy_loop = ct.FitSettings()
    runs = {
        "no_length_prior": (base, numpy_loop),
        "length_prior": (prior, numpy_loop),
        "length_prior_solver": (prior, ct.FitSettings(engine="tangle", backend=BACKEND)),
    }
    geometry_keys = ("curvature_ratio_max", "fibers_over_bend_limit", "max_penetration_radii", "overlapping_pairs")
    keys = ("fitted_fibers", "recovered", "split", "missed", "stubs", "false_fibers", "merged_fibers",
            "interior_ends_fit", "interior_ends_truth", "implied_mean_length_fit_m",
            "centerline_error_voxels", "voxel_label_accuracy")
    reports = {}
    for name, (spec, settings) in runs.items():
        started = time.perf_counter()
        fit = ct.fit_fibers(scan.volume, VOXEL, spec, settings, verbose=True)
        elapsed = time.perf_counter() - started
        report = ct.score(fit, scan)
        report["seconds"] = elapsed
        # Measured at the solver's segment length (1.25 diameters) for every fit.
        geometry = ct.geometry_report(
            fit.centerlines, fit.radii, MIN_BEND_RADIUS / VOXEL, spacing=1.25 * DIAMETER / VOXEL
        )
        report.update({key: geometry[key] for key in geometry_keys})
        reports[name] = report
        fit.write(OUTPUT / name, volume=scan.volume)
        (OUTPUT / name / "score.json").write_text(json.dumps(report, indent=1) + "\n")
    ct.save_overlay_figure(OUTPUT / "truth_overlay.png", scan.volume, scan.labels, title="ground truth")

    print(f"{'':32s}" + "".join(f"{name:>18s}" for name in reports))
    for key in keys + geometry_keys + ("seconds",):
        values = [reports[name][key] for name in reports]
        print(f"{key:32s}" + "".join(f"{v:>18.4g}" if isinstance(v, float) else f"{v!s:>18s}" for v in values))


if __name__ == "__main__":
    main()
