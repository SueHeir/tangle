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

Requires NumPy and SciPy; ``tifffile`` and ``matplotlib`` are optional (TIFF
and PNG outputs). Install them with ``pip install numpy scipy tifffile
matplotlib``.
"""

from ._evaluate import score
from ._fit import FiberSpec, FitResult, FitSettings, fit_fibers, load_fit
from ._image import Levels
from ._overlay import fiber_palette, overlay_slice, overlay_volume, save_overlay_figure
from ._synthetic import SyntheticScan, read_vti, synthetic_ct

__all__ = [
    "FiberSpec",
    "FitResult",
    "FitSettings",
    "Levels",
    "SyntheticScan",
    "fiber_palette",
    "fit_fibers",
    "load_fit",
    "overlay_slice",
    "overlay_volume",
    "read_vti",
    "save_overlay_figure",
    "score",
    "synthetic_ct",
]
