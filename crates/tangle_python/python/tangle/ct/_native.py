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

    @property
    def native(self):
        """The ``tangle._tangle.CtHessian`` itself."""
        return self._field

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


def _claimed(claimed: np.ndarray) -> np.ndarray:
    if claimed.dtype != np.int32 or not claimed.flags.c_contiguous:
        raise ValueError("claimed must be a C-contiguous int32 array")
    return claimed


def trace_one_way(
    image: np.ndarray, hessian: Hessian, claimed: np.ndarray, *, radius: float, min_bend_radius: float,
    step: float, start, direction, max_steps: int, own_label: int = 0,
) -> np.ndarray:
    """The points of a one-way trace (see ``_trace.Tracer.trace_one_way``), ``(n, 3)``."""
    points = _tangle.ct_trace_one_way(
        _f32(image), hessian.native, _claimed(claimed), float(radius), float(min_bend_radius), float(step),
        [float(v) for v in start], [float(v) for v in direction], int(max_steps), int(own_label),
    )
    return np.array(points, dtype=np.float64).reshape(-1, 3)


def trace_fibers(
    image: np.ndarray, hessian: Hessian, claimed: np.ndarray, edt: np.ndarray, peak: np.ndarray, *,
    radius: float, min_bend_radius: float, step: float, min_length: float, node_spacing: float,
    label_offset: int, max_fibers: int | None, seed_depth_radii: float,
) -> list[np.ndarray]:
    """Fibers traced from ridge seeds, painting ``claimed`` in place (see ``_trace.trace_fibers``)."""
    lines = _tangle.ct_trace_fibers(
        _f32(image), hessian.native, _claimed(claimed), _f32(edt), _f32(peak), float(radius),
        float(min_bend_radius), float(step), float(min_length), float(node_spacing), int(label_offset),
        None if max_fibers is None else int(max_fibers), float(seed_depth_radii),
    )
    return [np.array(line, dtype=np.float64).reshape(-1, 3) for line in lines]


def _pack(lines) -> tuple[np.ndarray, list[int]]:
    lines = [np.asarray(line, dtype=np.float64).reshape(-1, 3) for line in lines]
    nodes = np.ascontiguousarray(np.concatenate(lines)) if lines else np.zeros((0, 3))
    return nodes, [len(line) for line in lines]


def _unpack(packed) -> list[np.ndarray]:
    flat, counts = packed
    nodes = np.array(flat, dtype=np.float64).reshape(-1, 3)
    return np.split(nodes, np.cumsum(counts)[:-1]) if counts else []


def owners(centerlines, radii, voxels) -> np.ndarray:
    """Owning fiber (one-based, 0 for none) of each ``(k, j, i)`` voxel (see ``_geometry.OwnerLookup``)."""
    nodes, counts = _pack(centerlines)
    voxels = np.ascontiguousarray(np.asarray(voxels, dtype=np.int64).reshape(-1, 3))
    out = np.empty(len(voxels), dtype=np.int32)
    _tangle.ct_owners(nodes, counts, [float(r) for r in radii], voxels, out)
    return out


def end_step(image, centerlines, radii, reach, *, step: float, max_moves: int) -> list[np.ndarray]:
    """Fiber ends grown or trimmed (see ``_refine.end_step``)."""
    nodes, counts = _pack(centerlines)
    return _unpack(_tangle.ct_end_step(
        _f32(image), nodes, counts, [float(r) for r in radii], [float(r) for r in reach], float(step), int(max_moves)
    ))


def cut_void(image, centerlines, radii, *, level, min_gap_radii, bridge_level, bridge_offset_radii,
             aligned_level, aligned_angle_degrees, directions=None):
    """``(pieces, source, (trimmed, splits, bridged))`` (see ``_refine.cut_void``)."""
    nodes, counts = _pack(centerlines)
    packed, source, cut = _tangle.ct_cut_void(
        _f32(image), nodes, counts, [float(r) for r in radii], float(level), float(min_gap_radii),
        float(bridge_level), float(bridge_offset_radii), float(aligned_level), float(aligned_angle_degrees), directions,
    )
    return _unpack(packed), np.array(source, dtype=int), cut


def resample(centerlines, spacing: float) -> list[np.ndarray]:
    """Each polyline resampled to ``spacing`` (see ``_geometry.resample``)."""
    nodes, counts = _pack(centerlines)
    return _unpack(_tangle.ct_resample(nodes, counts, float(spacing)))


def support(image, centerlines) -> np.ndarray:
    nodes, counts = _pack(centerlines)
    return np.array(_tangle.ct_support(_f32(image), nodes, counts))


def curvature_ratio(centerlines, min_bend_radius: float) -> np.ndarray:
    nodes, counts = _pack(centerlines)
    return np.array(_tangle.ct_curvature_ratio(nodes, counts, float(min_bend_radius)))


def _box(corner) -> list[int]:
    return [int(v) for v in corner]


def render_occupancy(low, high, centerlines, radii, edge: float) -> np.ndarray:
    """Soft union occupancy of capsules over box ``[low, high)`` (see ``_moves.render_occupancy``)."""
    low, high = _box(low), _box(high)
    out = np.zeros(tuple(max(high[a] - low[a], 0) for a in (2, 1, 0)))
    if len(centerlines):
        nodes, counts = _pack(centerlines)
        _tangle.ct_render_occupancy(low, high, nodes, counts, [float(r) for r in radii], float(edge), out)
    return out


def local_residual(image, low, high, centerlines, radii, base, edge: float) -> float:
    nodes, counts = _pack(centerlines)
    if base is not None:
        base = np.ascontiguousarray(base, dtype=np.float64)
    return _tangle.ct_local_residual(
        _f32(image), _box(low), _box(high), nodes, counts, [float(r) for r in radii], float(edge), base
    )


def _lines_and_radii(result) -> tuple:
    packed, radii, *count = result
    return (_unpack(packed), np.array(radii, dtype=np.float64), *count)


def trim_duplicates(centerlines, radii, *, min_length: float, closeness: float):
    nodes, counts = _pack(centerlines)
    return _lines_and_radii(
        _tangle.ct_trim_duplicates(nodes, counts, [float(r) for r in radii], float(min_length), float(closeness))
    )


def remove_unsupported(image, centerlines, radii, *, min_length: float, min_support: float):
    nodes, counts = _pack(centerlines)
    return _lines_and_radii(_tangle.ct_remove_unsupported(
        _f32(image), nodes, counts, [float(r) for r in radii], float(min_length), float(min_support)
    ))


def _optional(value) -> float | None:
    return None if value is None else float(value)


def merge_fragments(image, centerlines, radii, *, max_gap, max_angle_degrees, min_bridge_support, min_bend_radius,
                    kink_threshold, end_cost, scale, max_prior_gap, max_prior_angle_degrees):
    nodes, counts = _pack(centerlines)
    return _lines_and_radii(_tangle.ct_merge_fragments(
        _f32(image), nodes, counts, [float(r) for r in radii], float(max_gap), float(max_angle_degrees),
        float(min_bridge_support), _optional(min_bend_radius), float(kink_threshold), float(end_cost), float(scale),
        _optional(max_prior_gap or None), float(max_prior_angle_degrees),
    ))


def split_kinks(centerlines, radii, *, min_bend_radius, min_length, max_length, threshold, min_angle_degrees,
                image, end_cost, scale):
    nodes, counts = _pack(centerlines)
    return _lines_and_radii(_tangle.ct_split_kinks(
        nodes, counts, [float(r) for r in radii], float(min_bend_radius), float(min_length), _optional(max_length),
        float(threshold), float(min_angle_degrees), float(end_cost), float(scale),
        None if image is None else _f32(image),
    ))


def resolve_side_by_side(image, centerlines, radii, *, min_length, reach, end_cost, scale, max_angle_degrees):
    nodes, counts = _pack(centerlines)
    return _lines_and_radii(_tangle.ct_resolve_side_by_side(
        _f32(image), nodes, counts, [float(r) for r in radii], float(min_length), float(reach), float(end_cost),
        float(scale), float(max_angle_degrees),
    ))
