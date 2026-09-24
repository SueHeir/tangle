"""How sure the fit is of each fitted node.

A node is trusted when, where it sits:

* **image**: the scan is fiber across the core of its capsule;
* **surround**: just outside its capsule the scan is void, or is inside
  another fit (fiber there that no fit explains means a fiber is missing,
  or this one is off-center or too thin);
* **ownership**: its core is not also inside another fit's capsule;
* **thickness**: the foreground's thickness there, less the margin by
  which the foreground over-reaches, matches its radius (a fit that runs
  along two touching fibers reads the wrong thickness);
* **stability**: it barely moved in the last solve (when the previous
  positions are given).

Each is a score in [0, 1] and the node's confidence is their product. The
scores are read on rings of points around the centerline, every
``spacing`` along it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._geometry import resample, sample_image, tangents

COMPONENTS = ("image", "surround", "ownership", "thickness", "stability")


def node_confidence(
    image: np.ndarray,
    depth: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    *,
    spacing: float,
    margin: float = 0.0,
    previous: list[np.ndarray] | None = None,
    ring: int = 8,
    thickness_tolerance: float = 0.3,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Confidence of every node of ``lines``, and a summary.

    ``image`` is the normalized scan (void ~0, fiber ~1) and ``depth`` the
    local thickness the fitter classifies with, both ``(z, y, x)``. Returns
    one array per fiber with a value in [0, 1] per node, and a summary with
    each fiber's mean and minimum and the mean of every component.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if not lines:
        return [], {"fibers": 0}
    samples = [
        resample(line, spacing) if len(line) > 1 else np.asarray(line, dtype=np.float64)
        for line in lines
    ]
    segments = _Segments(lines, radii)

    angles = 2.0 * np.pi * np.arange(ring) / ring
    parts: dict[str, list[np.ndarray]] = {name: [] for name in COMPONENTS}
    for f, points in enumerate(samples):
        r = float(radii[f])
        n = len(points)
        u, v = _normals(points)
        directions = (
            np.cos(angles)[None, :, None] * u[:, None, :]
            + np.sin(angles)[None, :, None] * v[:, None, :]
        )

        core = np.concatenate(
            [points[:, None, :], points[:, None, :] + 0.5 * r * directions], axis=1
        )
        values = sample_image(image, core.reshape(-1, 3)).reshape(n, ring + 1)
        parts["image"].append(np.clip((values - 0.5) / 0.4, 0.0, 1.0).mean(axis=1))

        shared = segments.covered(core.reshape(-1, 3), f, extra=0.0).reshape(
            n, ring + 1
        )
        parts["ownership"].append(1.0 - shared.mean(axis=1))

        outside = (points[:, None, :] + (r + margin + 1.0) * directions).reshape(-1, 3)
        fiber = sample_image(image, outside) > 0.5
        explained = segments.covered(outside, f, extra=margin + 0.5)
        parts["surround"].append(
            1.0 - (fiber & ~explained).reshape(n, ring).mean(axis=1)
        )

        thickness = sample_image(depth, points) - margin
        ratio = thickness / r - 1.0
        parts["thickness"].append(np.exp(-0.5 * (ratio / thickness_tolerance) ** 2))

        if previous is not None and len(previous[f]) > 1:
            moved = _distance_to_polyline(
                points, np.asarray(previous[f], dtype=np.float64)
            )
            parts["stability"].append(np.exp(-0.5 * (moved / (0.5 * r)) ** 2))
        else:
            parts["stability"].append(np.ones(n))

    per_sample = []
    for f in range(len(samples)):
        value = np.prod([parts[name][f] for name in COMPONENTS], axis=0)
        per_sample.append(_smooth(value))
    per_node = [
        _to_nodes(line, points, value)
        for line, points, value in zip(lines, samples, per_sample)
    ]
    # The same without stability: a stretch that was just redrawn moved in
    # its solve because it was redrawn, not because it is wrong, so keeping
    # or reverting a redraw is judged on this.
    settled = [
        _to_nodes(
            line,
            points,
            _smooth(
                np.prod(
                    [parts[name][f] for name in COMPONENTS if name != "stability"],
                    axis=0,
                )
            ),
        )
        for f, (line, points) in enumerate(zip(lines, samples))
    ]

    fiber_mean = np.array(
        [float(value.mean()) if len(value) else 0.0 for value in per_sample]
    )
    fiber_min = np.array(
        [float(value.min()) if len(value) else 0.0 for value in per_sample]
    )
    summary: dict[str, Any] = {
        "fibers": len(lines),
        "mean": float(np.mean(np.concatenate(per_sample))),
        "fibers_below_half": int((fiber_min < 0.5).sum()),
        "components": {
            name: round(float(np.mean(np.concatenate(parts[name]))), 3)
            for name in COMPONENTS
        },
        "fiber_mean": fiber_mean,
        "fiber_min": fiber_min,
        "without_stability": settled,
    }
    return per_node, summary


class _Segments:
    """Every fiber's segments, to ask whether points lie inside another fit."""

    def __init__(self, lines: list[np.ndarray], radii: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        a, b, fiber = [], [], []
        for f, line in enumerate(lines):
            line = np.asarray(line, dtype=np.float64).reshape(-1, 3)
            if len(line) == 1:
                line = np.vstack([line, line])
            a.append(line[:-1])
            b.append(line[1:])
            fiber.append(np.full(len(line) - 1, f))
        self.a = np.concatenate(a)
        self.b = np.concatenate(b)
        self.fiber = np.concatenate(fiber)
        self.radius = np.asarray(radii, dtype=np.float64)[self.fiber]
        self.half = 0.5 * np.linalg.norm(self.b - self.a, axis=1)
        self.tree = cKDTree(0.5 * (self.a + self.b))

    def covered(self, points: np.ndarray, skip: int, *, extra: float) -> np.ndarray:
        """Whether each point is within radius + ``extra`` of a fiber other than ``skip``."""
        result = np.zeros(len(points), dtype=bool)
        if len(points) == 0:
            return result
        search = float(self.radius.max() + extra + self.half.max())
        hits = self.tree.query_ball_point(points, search)
        sizes = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if sizes.sum() == 0:
            return result
        point = np.repeat(np.arange(len(points)), sizes)
        segment = np.fromiter(
            (s for h in hits for s in h), dtype=np.int64, count=int(sizes.sum())
        )
        other = self.fiber[segment] != skip
        point, segment = point[other], segment[other]
        if len(point) == 0:
            return result
        p = points[point]
        a, b = self.a[segment], self.b[segment]
        ab = b - a
        denominator = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * ab).sum(axis=1) / denominator, 0.0, 1.0)
        distance = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
        inside = distance <= self.radius[segment] + extra
        result[point[inside]] = True
        return result


def _normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = (
        tangents(points)
        if len(points) > 1
        else np.tile([1.0, 0.0, 0.0], (len(points), 1))
    )
    helper = np.where(np.abs(t[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(t, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    return u, np.cross(t, u)


def _distance_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    a, b = line[:-1], line[1:]
    ab = b - a
    denominator = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / denominator[None], 0.0, 1.0)
    distance = np.linalg.norm(ap - t[..., None] * ab[None], axis=2)
    return distance.min(axis=1)


def _smooth(value: np.ndarray) -> np.ndarray:
    """Median over three neighbors, so one noisy ring does not flag a node."""
    if len(value) < 3:
        return value
    padded = np.concatenate([value[:1], value, value[-1:]])
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)


def _to_nodes(line: np.ndarray, points: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Each node's confidence: the lowest sample within half a segment of it."""
    line = np.asarray(line, dtype=np.float64)
    if len(line) < 2 or len(points) < 2:
        return np.full(len(line), float(value.min()) if len(value) else 0.0)
    node_arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    )
    sample_arc = np.linspace(0.0, node_arc[-1], len(points))
    middles = 0.5 * (node_arc[:-1] + node_arc[1:])
    low = np.concatenate([[-np.inf], middles])
    high = np.concatenate([middles, [np.inf]])
    out = np.empty(len(line))
    for i in range(len(line)):
        pick = (sample_arc >= low[i]) & (sample_arc <= high[i])
        out[i] = (
            float(value[pick].min())
            if pick.any()
            else float(np.interp(node_arc[i], sample_arc, value))
        )
    return out


def coverage_map(
    foreground: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    confidence: list[np.ndarray],
) -> np.ndarray:
    """Per voxel: the confidence of the fit owning it, on the foreground (0 elsewhere).

    Ownership is the nearest capsule surface; the confidence is the mean of
    the owning segment's two nodes.
    """
    from ._geometry import rasterize

    values = np.zeros(foreground.shape, dtype=np.float32)
    if not lines:
        return values
    _, _, segments = rasterize(
        foreground.shape, lines, np.asarray(radii, dtype=np.float64), signed=True
    )
    table = np.concatenate([0.5 * (c[:-1] + c[1:]) for c in confidence]).astype(
        np.float32
    )
    owned = foreground & (segments >= 0)
    values[owned] = table[segments[owned]]
    return values


def sure_coverage(
    foreground: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    confidence: list[np.ndarray],
) -> float:
    """The fraction of the foreground explained by the fit, weighted by confidence.

    Each foreground voxel counts the confidence of the fit that owns it (the
    nearest capsule surface; 0 when no fit does). Missing fibers, unsure
    fits and fits over void all lower it, so a redraw is kept only when it
    raises it.
    """
    total = float(foreground.sum())
    if not lines or total == 0.0:
        return 0.0
    return (
        float(coverage_map(foreground, lines, radii, confidence).sum(dtype=np.float64))
        / total
    )


def residual_map(
    foreground: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    """Per voxel: 1 where the fit disagrees with the foreground, else 0.

    Foreground voxels farther than radius + ``margin`` from every fit are
    unexplained; background voxels inside a fit's capsule are extra.
    """
    from ._geometry import rasterize

    if not lines:
        return foreground.astype(np.int32)
    radii = np.asarray(radii, dtype=np.float64)
    labels, distance, _ = rasterize(
        foreground.shape, lines, radii, reach=radii + margin, signed=True
    )
    unexplained = foreground & (labels == 0)
    extra = ~foreground & (labels > 0) & (distance <= 0.0)
    return (unexplained | extra).astype(
        np.int32
    )  # int32 so differences of sums cannot wrap
