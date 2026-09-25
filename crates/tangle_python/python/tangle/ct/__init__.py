"""Fit Tangle fibers to CT scans using what the fibers are known to look like.

Give the fiber diameter (and optionally bend radius and minimum length); the
fitter traces initial centerlines, then moves every fiber toward the scan
while keeping its diameter, bending and non-overlap priors, adding, removing
and joining fibers where the image calls for it. The result exports as a Tangle
assembly, a population with the fitted statistics, a per-voxel fiber label
volume, and an overlay of the fibers on the scan with one color per fiber.

    >>> import tangle.ct as ct
    >>> fit = ct.fit_fibers(volume, voxel_size=1.3e-6, spec=ct.FiberSpec(diameter=10e-6))
    >>> fit.write("fit_output", volume=volume)   # fit.json, labels.tif, overlay.*
    >>> assembly = fit.to_assembly()             # continue in Tangle

A whole scan is fitted in tiles, each saved as it finishes so a stopped run
resumes where it stopped:

    >>> scan = ct.open_scan("scan.tif")          # read from disk tile by tile
    >>> fit = ct.fit_tiled(scan, 1.3e-6, spec, checkpoint="fit_tiles")

Requires NumPy and SciPy; ``tifffile`` and ``matplotlib`` are optional (TIFF
and PNG outputs). Install them with ``pip install numpy scipy tifffile
matplotlib``.
"""

from ._evaluate import geometry_report, score
from ._fit import FiberSpec, FitResult, FitSettings, fit_fibers, load_fit
from ._image import Levels
from ._profile import CrossSection
from ._overlay import fiber_palette, overlay_slice, overlay_volume, save_overlay_figure
from ._scanner import Scanner
from ._synthetic import SyntheticScan, read_vti, synthetic_ct
from ._tiles import fit_tiled, load_tiles, open_scan

__all__ = [
    "CrossSection",
    "FiberSpec",
    "FitResult",
    "FitSettings",
    "Levels",
    "Scanner",
    "SyntheticScan",
    "fiber_palette",
    "fit_fibers",
    "fit_tiled",
    "geometry_report",
    "load_fit",
    "load_tiles",
    "open_scan",
    "overlay_slice",
    "overlay_volume",
    "read_vti",
    "save_overlay_figure",
    "score",
    "synthetic_ct",
]
