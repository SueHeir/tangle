"""NumPy-facing wrappers of the Rust volume operations (``tangle_ct`` crate).

Each takes and returns NumPy arrays; the work happens in Rust on the arrays'
own memory. Arrays are made C-contiguous and of the element type the Rust
side reads (a copy only when they are not already).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tangle import _tangle


def _f32(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def _points(points) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def gaussian(image: np.ndarray, sigma: float, order: Sequence[int] = (0, 0, 0)) -> np.ndarray:
    """``scipy.ndimage.gaussian_filter(image, sigma, order=order)`` of a volume, as float32."""
    image = _f32(image)
    out = np.empty_like(image)
    _tangle.ct_gaussian_filter(image, out, float(sigma), tuple(int(o) for o in order))
    return out


def sample(image: np.ndarray, points, fill: float = 0.0) -> np.ndarray:
    """Trilinear samples of a ``(z, y, x)`` volume at ``(x, y, z)`` points (float64)."""
    points = _points(points)
    out = np.empty(len(points))
    _tangle.ct_sample(_f32(image), points, out, float(fill))
    return out


def foreground_depth(foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The foreground's distance transform (float32) and its 3×3×3 maximum."""
    mask = np.ascontiguousarray(foreground, dtype=bool).view(np.uint8)
    edt = np.empty(mask.shape, dtype=np.float32)
    peak = np.empty(mask.shape, dtype=np.float32)
    _tangle.ct_foreground_depth(mask, edt, peak)
    return edt, peak


def core_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Enclosed holes up to ``max_area`` voxels, slice by slice along each axis."""
    values = np.ascontiguousarray(mask, dtype=bool).view(np.uint8)
    out = np.empty(values.shape, dtype=np.uint8)
    _tangle.ct_core_holes(values, out, float(max_area))
    return out.view(bool)


def rasterize(shape, centerlines, radii, reach, signed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-fiber ownership: ``(labels, distance, segment)`` (see ``_geometry.rasterize``)."""
    lines = [np.asarray(line, dtype=np.float64).reshape(-1, 3) for line in centerlines]
    nodes = np.ascontiguousarray(np.concatenate(lines)) if lines else np.zeros((0, 3))
    labels = np.empty(shape, dtype=np.int32)
    distance = np.empty(shape, dtype=np.float32)
    segment = np.empty(shape, dtype=np.int32)
    _tangle.ct_rasterize(
        nodes, [len(line) for line in lines], [float(r) for r in radii], [float(r) for r in reach],
        bool(signed), labels, distance, segment,
    )
    return labels, distance, segment


def paint(target: np.ndarray, line, reach: float, value: int, *, only_empty: bool = False) -> None:
    """Set ``target``'s voxels within ``reach`` of polyline ``line`` to ``value``, in place (int32 target)."""
    if target.dtype != np.int32 or not target.flags.c_contiguous:
        raise ValueError("paint needs a C-contiguous int32 target")
    _tangle.ct_paint(target, _points(line), float(reach), int(value), bool(only_empty))


class Hessian:
    """The Gaussian-scale Hessian of a volume, read at points (``_image.HessianField``)."""

    def __init__(self, image: np.ndarray, sigma: float) -> None:
        self._field = _tangle.CtHessian(_f32(image), float(sigma))
        self.sigma = float(sigma)

    def at(self, points) -> np.ndarray:
        points = _points(points)
        out = np.empty((len(points), 3, 3))
        self._field.at(points, out)
        return out

    def directions(self, points) -> tuple[np.ndarray, np.ndarray]:
        points = _points(points)
        axis = np.empty((len(points), 3))
        tubularity = np.empty(len(points))
        self._field.directions(points, axis, tubularity)
        return axis, tubularity
