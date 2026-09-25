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


def cross_frames(line: np.ndarray, nodes: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Up to ``nodes`` interior nodes of a centerline, and four unit directions across it at each.

    Returns the nodes ``(n, 3)`` and the directions ``(n, 4, 3)``: ``u``,
    ``-u``, ``w``, ``-w`` for two perpendicular unit normals. None for a line
    with fewer than 3 nodes.
    """
    line = np.asarray(line, dtype=np.float64)
    if len(line) < 3:
        return None
    pick = np.unique(np.linspace(1, len(line) - 2, min(nodes, len(line) - 2)).astype(int))
    axis = tangents(line)[pick]
    helper = np.where(np.abs(axis[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    u = np.cross(axis, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    w = np.cross(axis, u)
    return line[pick], np.stack([u, -u, w, -w], axis=1)


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


def segment_voxels(
    lines: list[np.ndarray],
    pads: np.ndarray,
    limits: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    budget: int = 1 << 21,
):
    """Every segment's voxels in box ``[low, high)`` (x, y, z), many segments per numpy call.

    A segment's voxels are those in its bounding box grown by its line's
    ``pads`` value whose center is within its line's ``limits`` value of the
    segment. Yields chunks ``(voxel, segment, distance)``: the flat index
    into the box's ``(z, y, x)`` array, the global segment index (segment
    ``s`` of line ``f`` is ``offsets[f] + s``, offsets from the node counts
    minus one) and the distance. Segments are batched by box size, so the
    chunks are not in segment order. Drawing capsules one segment at a time
    was a third of single_type's fit time, nearly all of it per-call
    overhead on boxes of a few thousand voxels.
    """
    low = np.asarray(low, dtype=int)
    high = np.asarray(high, dtype=int)
    starts, ends, pad, limit = [], [], [], []
    for line, line_pad, line_limit in zip(lines, pads, limits):
        line = np.asarray(line, dtype=np.float64).reshape(-1, 3)
        count = max(len(line) - 1, 0)
        starts.append(line[:count])
        ends.append(line[1 : count + 1])
        pad.append(np.full(count, float(line_pad)))
        limit.append(np.full(count, float(line_limit)))
    if not starts:
        return
    a = np.concatenate(starts)
    b = np.concatenate(ends)
    pad = np.concatenate(pad)
    limit = np.concatenate(limit)
    lo = np.maximum(np.floor(np.minimum(a, b) - pad[:, None]).astype(int), low)
    hi = np.minimum(np.ceil(np.maximum(a, b) + pad[:, None]).astype(int), high)
    ids = np.nonzero(np.all(hi > lo, axis=1))[0]
    if len(ids) == 0:
        return
    size = hi[ids] - lo[ids]
    ids = ids[np.lexsort((size[:, 0], size[:, 1], size[:, 2]))]  # similar boxes batch together
    ab = b - a
    denominator = np.einsum("ij,ij->i", ab, ab)
    degenerate = denominator <= 1e-12
    denominator = np.where(degenerate, 1.0, denominator)
    strides = np.array([1, high[0] - low[0], (high[0] - low[0]) * (high[1] - low[1])])
    sizes = (hi[ids] - lo[ids]).tolist()  # plain ints: this loop runs once per segment
    start = 0
    while start < len(ids):
        # Grow the batch while its padded boxes stay within the budget and
        # the padding stays under a third of the real boxes' voxels.
        stop = start + 1
        bx, by, bz = sizes[start]
        real = bx * by * bz
        while stop < len(ids):
            ex, ey, ez = sizes[stop]
            gx, gy, gz = max(bx, ex), max(by, ey), max(bz, ez)
            padded = (stop + 1 - start) * gx * gy * gz
            if padded > budget or 3 * padded > 4 * (real + ex * ey * ez):
                break
            bx, by, bz = gx, gy, gz
            real += ex * ey * ez
            stop += 1
        dims = (bx, by, bz)
        batch = ids[start:stop]
        start = stop
        axes = []
        for k in range(3):
            coordinate = lo[batch, k, None] + np.arange(dims[k])[None, :]  # (n, dims[k])
            axes.append((coordinate, coordinate < hi[batch, k, None]))
        x = (axes[0][0] + 0.5 - a[batch, 0, None])[:, None, None, :]
        y = (axes[1][0] + 0.5 - a[batch, 1, None])[:, None, :, None]
        z = (axes[2][0] + 0.5 - a[batch, 2, None])[:, :, None, None]
        u = ab[batch]
        t = (x * u[:, 0, None, None, None] + y * u[:, 1, None, None, None] + z * u[:, 2, None, None, None])
        t = np.clip(t / denominator[batch, None, None, None], 0.0, 1.0)
        t = np.where(degenerate[batch, None, None, None], 0.0, t)
        dx = x - t * u[:, 0, None, None, None]
        dy = y - t * u[:, 1, None, None, None]
        dz = z - t * u[:, 2, None, None, None]
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        keep = (
            (distance <= limit[batch, None, None, None])
            & axes[0][1][:, None, None, :]
            & axes[1][1][:, None, :, None]
            & axes[2][1][:, :, None, None]
        )
        n, i, j, k = np.nonzero(keep)
        voxel = (
            (axes[2][0][n, i] - low[2]) * strides[2]
            + (axes[1][0][n, j] - low[1]) * strides[1]
            + (axes[0][0][n, k] - low[0])
        )
        yield voxel, batch[n], distance[n, i, j, k]


def nearest_segments(size: int, chunks, key: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per voxel of a flat box of ``size``, the nearest segment from :func:`segment_voxels` chunks.

    The distance compared is the segment's distance plus ``key[segment]``
    (for example minus its radius, to compare surfaces); ties go to the
    lower segment index, as drawing segments in order with a strict ``<``
    did. Returns ``(best, segment)`` flat arrays: the compared value
    (``inf`` where none) and the segment (``-1``).
    """
    best = np.full(size, np.inf)
    none = np.iinfo(np.int32).max
    owner = np.full(size, none, dtype=np.int32)
    for voxel, segment, distance in chunks:
        # Sort-free: scatter minimums, then the lowest segment among those
        # at the minimum (a per-chunk lexsort was slower than drawing one
        # segment at a time).
        value = distance + key[segment] if key is not None else distance
        before = best[voxel]
        np.minimum.at(best, voxel, value)
        after = best[voxel]
        at_best = value == after
        voxel, segment = voxel[at_best], segment[at_best]
        owner[voxel[after[at_best] < before[at_best]]] = none  # a nearer segment: the old owner is out
        np.minimum.at(owner, voxel, segment.astype(np.int32))
    owner[owner == none] = -1
    return best, owner


def segment_lines(lines: list[np.ndarray]) -> np.ndarray:
    """The line each global segment index (see :func:`segment_voxels`) belongs to."""
    counts = [max(len(line) - 1, 0) for line in lines]
    return np.repeat(np.arange(len(lines)), counts)


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


class OwnerLookup:
    """``rasterize(shape, lines, radii, signed=True)``'s label at single voxels.

    For callers that read the labels at a few voxels only: the owner of a
    voxel is the fiber whose capsule surface is nearest its center among the
    segments within the fiber's radius of it (0 for none), ties going to the
    lower segment, as :func:`rasterize` decides it.
    """

    def __init__(self, shape: tuple[int, int, int], centerlines: list[np.ndarray], radii: np.ndarray) -> None:
        self.shape = np.array(shape)
        radii = np.asarray(radii, dtype=np.float64)
        line_of = segment_lines(centerlines)
        pieces = [np.asarray(line, dtype=np.float64).reshape(-1, 3) for line in centerlines]
        self.a = np.concatenate([p[:-1] for p in pieces]) if len(line_of) else np.zeros((0, 3))
        b = np.concatenate([p[1:] for p in pieces]) if len(line_of) else np.zeros((0, 3))
        self.ab = b - self.a
        denominator = np.einsum("ij,ij->i", self.ab, self.ab)
        self.degenerate = denominator <= 1e-12
        self.denominator = np.where(self.degenerate, 1.0, denominator)
        self.label = line_of + 1
        self.radius = radii[line_of]

    def __call__(self, index_zyx: tuple[int, int, int]) -> int:
        if not len(self.label):
            return 0
        center = np.asarray(index_zyx[::-1], dtype=np.float64) + 0.5
        offset = center - self.a
        t = np.clip(np.einsum("ij,ij->i", offset, self.ab) / self.denominator, 0.0, 1.0)
        t = np.where(self.degenerate, 0.0, t)
        distance = np.linalg.norm(offset - t[:, None] * self.ab, axis=1)
        near = np.flatnonzero(distance <= self.radius)
        if not len(near):
            return 0
        surface = distance[near] - self.radius[near]
        return int(self.label[near[np.argmin(surface)]])  # argmin: first (lowest) segment on ties


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
    if not len(centerlines):
        return labels, best, segment_ids
    line_of = segment_lines(centerlines)
    chunks = segment_voxels(centerlines, reaches + 0.5, reaches, np.zeros(3, dtype=int), np.array(shape[::-1]))
    value, owner = nearest_segments(
        int(np.prod(shape)), chunks, key=-radii[line_of] if signed else None
    )
    owned = owner >= 0
    labels.ravel()[owned] = line_of[owner[owned]] + 1
    best.ravel()[owned] = value[owned]
    segment_ids.ravel()[owned] = owner[owned]
    return labels, best, segment_ids
