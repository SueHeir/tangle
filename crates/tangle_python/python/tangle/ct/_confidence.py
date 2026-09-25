"""How sure the fit is of each fitted node.

A node is trusted when, where it sits:

* **image**: the scan is fiber across the core of its capsule;
* **surround**: just outside its capsule the scan is void, or is inside
  another fit (fiber there that no fit explains means a fiber is missing,
  or this one is off-center or too thin);
* **ownership**: its core is not also inside another fit's capsule;
* **thickness**: the foreground's depth there, or the scan's
  cross-section there (where it falls to half its peak, pooled over a few
  neighbouring rings so noise averages out), less the margin by which each
  over-reaches, matches its radius (a fit that runs along two touching
  fibers reads the wrong thickness both ways);
* **stability**: it barely moved in the last solve (when the previous
  positions are given).

Each is a score in [0, 1] and the node's confidence is their product. The
scores are read on rings of points around the centerline, every
``spacing`` along it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


COMPONENTS = ("image", "surround", "ownership", "thickness", "stability")


def node_confidence(
    image: np.ndarray,
    depth: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    *,
    spacing: float,
    margin: float = 0.0,
    thickness_margin: float | None = None,
    previous: list[np.ndarray] | None = None,
    ring: int = 8,
    thickness_tolerance: float = 0.3,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Confidence of every node of ``lines``, and a summary.

    ``image`` is the normalized scan (void ~0, fiber ~1) and ``depth`` the
    foreground's local thickness (``_trace.foreground_depth``), both
    ``(z, y, x)``; ``margin`` is how far the foreground over-reaches the
    fibers and ``thickness_margin`` (default ``margin``) how far their
    cross-section radius does (``_fit._Fitter.classify``). Returns
    one array per fiber with a value in [0, 1] per node, and a summary with
    each fiber's mean and minimum and the mean of every component.
    """
    from . import _native

    radii = np.asarray(radii, dtype=np.float64)
    if not lines:
        return [], {"fibers": 0}
    per_node, settled, per_sample, component_parts = _native.node_confidence(
        image, depth, lines, radii, spacing=spacing, margin=margin,
        thickness_margin=margin if thickness_margin is None else thickness_margin, ring=ring,
        thickness_tolerance=thickness_tolerance, previous=previous,
    )
    parts = dict(zip(COMPONENTS, component_parts))

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
