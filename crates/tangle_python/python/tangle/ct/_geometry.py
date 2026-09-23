"""Polyline helpers and capsule rasterization in voxel coordinates.

Internal coordinates are voxel units ordered ``(x, y, z)`` with the center of
voxel ``[k, j, i]`` (array order ``z, y, x``) at ``(i + 0.5, j + 0.5, k + 0.5)``.
That matches the cell-centered grid ``Assembly.export_puma`` writes when the
cell origin is at zero, so ``physical = voxel_coordinate * voxel_size``.
"""

from __future__ import annotations

import numpy as np


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline to (nearly) uniform arc-length spacing."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], steps > 1e-9])
    points = points[keep]
    if len(points) < 2:
        return points.copy()
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    count = max(int(round(arc[-1] / spacing)), 1) + 1
    targets = np.linspace(0.0, arc[-1], count)
    return np.stack([np.interp(targets, arc, points[:, axis]) for axis in range(3)], axis=1)


def tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents at the nodes (central differences, one-sided at ends)."""
    t = np.gradient(points, axis=0) if len(points) > 1 else np.zeros_like(points)
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(norm, 1e-12)


def sample_image(image: np.ndarray, points: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    """Trilinear samples of a ``(z, y, x)`` array at voxel coordinates ``(x, y, z)``."""
    from scipy.ndimage import map_coordinates

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    coordinates = (points[:, ::-1] - 0.5).T
    return map_coordinates(image, coordinates, order=1, mode="constant", cval=fill)


def sample_labels(labels: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-voxel lookup of an integer ``(z, y, x)`` array; outside reads 0."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    index = np.floor(points[:, ::-1]).astype(int)
    inside = np.all((index >= 0) & (index < np.array(labels.shape)), axis=1)
    values = np.zeros(len(points), dtype=labels.dtype)
    values[inside] = labels[index[inside, 0], index[inside, 1], index[inside, 2]]
    return values


def _segment_distances(voxels: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denominator = float(ab @ ab)
    if denominator <= 1e-12:
        return np.linalg.norm(voxels - a, axis=1)
    t = np.clip((voxels - a) @ ab / denominator, 0.0, 1.0)
    return np.linalg.norm(voxels - (a + t[:, None] * ab), axis=1)


def _box_segment_distances(low: np.ndarray, high: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances from the voxel centers of box ``[low, high)`` to segment ``ab``, as ``(z, y, x)``."""
    x = (np.arange(low[0], high[0]) + 0.5 - a[0])[None, None, :]
    y = (np.arange(low[1], high[1]) + 0.5 - a[1])[None, :, None]
    z = (np.arange(low[2], high[2]) + 0.5 - a[2])[:, None, None]
    ab = b - a
    denominator = float(ab @ ab)
    if denominator <= 1e-12:
        return np.sqrt(x * x + y * y + z * z)
    t = np.clip((x * ab[0] + y * ab[1] + z * ab[2]) / denominator, 0.0, 1.0)
    dx, dy, dz = x - t * ab[0], y - t * ab[1], z - t * ab[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def paint(target: np.ndarray, line: np.ndarray, reach: float, value: int, *, only_empty: bool = False) -> None:
    """Set voxels of ``target`` within ``reach`` of polyline ``line`` to ``value`` in place.

    Works on per-segment windows, so it costs nothing proportional to the
    whole volume (unlike :func:`rasterize`).
    """
    line = np.asarray(line, dtype=np.float64).reshape(-1, 3)
    if len(line) == 1:
        line = np.vstack([line, line])
    upper = np.array(target.shape[::-1])
    for a, b in zip(line[:-1], line[1:]):
        low = np.maximum(np.floor(np.minimum(a, b) - reach - 0.5).astype(int), 0)
        high = np.minimum(np.ceil(np.maximum(a, b) + reach + 0.5).astype(int), upper)
        if np.any(high <= low):
            continue
        inside = _box_segment_distances(low, high, a, b) <= reach
        window = target[low[2] : high[2], low[1] : high[1], low[0] : high[0]]
        if only_empty:
            inside &= window == 0
        window[inside] = value


def rasterize(
    shape: tuple[int, int, int],
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    reach: float | np.ndarray | None = None,
    signed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-fiber ownership of every voxel within ``reach`` of a centerline.

    Returns ``(labels, distance, segment)`` arrays in ``(z, y, x)`` order.
    ``labels`` is one-based (0 = unowned). ``distance`` is the distance to the
    owning centerline, minus that fiber's radius when ``signed`` is true, so
    ownership then follows the nearest capsule surface as in Tangle's exporter.
    ``segment`` is a global segment index (``-1`` = unowned); segment ``s`` of
    fiber ``f`` is ``offsets[f] + s`` with offsets from the node counts minus one.
    """
    radii = np.asarray(radii, dtype=np.float64)
    reaches = radii if reach is None else np.broadcast_to(np.asarray(reach, dtype=np.float64), radii.shape)
    labels = np.zeros(shape, dtype=np.int32)
    best = np.full(shape, np.inf, dtype=np.float32)
    segment_ids = np.full(shape, -1, dtype=np.int32)
    offset = 0
    upper = np.array(shape[::-1])
    for index, (line, radius, fiber_reach) in enumerate(zip(centerlines, radii, reaches)):
        line = np.asarray(line, dtype=np.float64)
        for s in range(len(line) - 1):
            a, b = line[s], line[s + 1]
            low = np.maximum(np.floor(np.minimum(a, b) - fiber_reach - 0.5).astype(int), 0)
            high = np.minimum(np.ceil(np.maximum(a, b) + fiber_reach + 0.5).astype(int), upper)
            if np.any(high <= low):
                continue
            distance = _box_segment_distances(low, high, a, b)
            inside = distance <= fiber_reach
            if signed:
                distance = distance - radius
            window = (slice(low[2], high[2]), slice(low[1], high[1]), slice(low[0], high[0]))
            improve = inside & (distance < best[window])
            best[window][improve] = distance[improve]
            labels[window][improve] = index + 1
            segment_ids[window][improve] = offset + s
        offset += max(len(line) - 1, 0)
    return labels, best, segment_ids
