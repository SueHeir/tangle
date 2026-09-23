"""Fit Tangle fibers to PuMA's FiberForm micro-CT scan.

PuMA ships a real 200³ micro-CT crop of FiberForm, a rayon-derived carbon
fiber felt (``200_fiberform.tif``, 1.3 µm voxels, NASA Open Source Agreement).
This script downloads it from PuMA's GitHub repository, fits fibers of about
10 µm diameter to it, and writes the Tangle configuration, the label volume
and the per-fiber overlay.

FiberForm fibers have lobed, non-circular cross-sections and carbonized
binder at many crossings, so the circular fibers fitted here are an
equal-area approximation; there is no fiber-level ground truth for this scan.

Needs NumPy, SciPy and tifffile; matplotlib adds the PNG overlay.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import tangle.ct as ct
from tangle.units import mm, um

OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("output") / "ct_fit_fiberform")
SOURCE = "https://raw.githubusercontent.com/nasa/puma/main/python/pumapy/data/200_fiberform.tif"
VOXEL = 1.3 * um

SPEC = ct.FiberSpec(
    diameter=10 * um,
    diameter_tolerance=0.35,  # lobed cross-sections vary in equal-area diameter
    min_bend_radius=40 * um,
    # Fiber-length prior. FiberForm fibers are much longer than this 260 µm
    # crop; 1 mm is an assumed round figure, not a measured one. Set it to
    # None to fit without the prior.
    length=1 * mm,
    name="FiberForm fiber",
)


def main() -> None:
    import tifffile

    OUTPUT.mkdir(parents=True, exist_ok=True)
    scan_path = OUTPUT / "200_fiberform.tif"
    if not scan_path.exists():
        print(f"downloading {SOURCE}")
        urllib.request.urlretrieve(SOURCE, scan_path)
    volume = tifffile.imread(scan_path)
    print(f"scan: {volume.shape} voxels at {VOXEL / um:.1f} µm")

    started = time.perf_counter()
    fit = ct.fit_fibers(volume, VOXEL, SPEC, verbose=True)
    print(f"fit: {fit.fiber_count} fibers ({time.perf_counter() - started:.0f} s)")

    summary = fit.population_summary()
    for key in (
        "diameter_mean", "length_mean", "fibers_touching_boundary", "volume_fraction", "orientation_eigenvalues",
        "interior_ends", "expected_interior_ends", "implied_mean_length",
    ):
        print(f"  {key}: {summary[key]}")
    for name, path in fit.write(OUTPUT, volume=volume).items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
