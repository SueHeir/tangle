"""Initial fiber centerlines: seed on the distance-transform ridge, then trace.

Each trace steps along the local tube axis (Hessian eigenvector blended with
momentum), re-centers in the cross-sectional plane on the image intensity,
limits the turn per step by the admissible bend radius, and stops when the
core intensity drops below the fiber/void midpoint. Voxels claimed by earlier
traces are down-weighted, so a trace keeps its direction through a crossing
instead of turning onto the other fiber.
"""

from __future__ import annotations

import numpy as np

from ._geometry import paint, polyline_length, resample, sample_image, sample_labels
from ._image import HessianField


def _perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(direction, e1)


def _disk_offsets(radius: float, spacing: float = 0.5) -> np.ndarray:
    ticks = np.arange(-radius, radius + 1e-9, spacing)
    u, v = np.meshgrid(ticks, ticks)
    keep = u**2 + v**2 <= radius**2
    return np.stack([u[keep], v[keep]], axis=1)


class Tracer:
    def __init__(
        self,
        image: np.ndarray,
        hessian: HessianField,
        *,
        radius: float,
        min_bend_radius: float,
        step: float,
        claimed: np.ndarray,
    ) -> None:
        self.image = image
        self.hessian = hessian
        self.radius = radius
        self.step = step
        self.max_turn = min(step / max(min_bend_radius, 1e-9), np.pi / 4)
        self.claimed = claimed
        self.window = _disk_offsets(1.4 * radius)
        self.core = _disk_offsets(max(0.5 * radius, 0.75))
        self.upper = np.array(image.shape[::-1], dtype=np.float64)

    def _inside(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= 0.0) and np.all(point <= self.upper))

    def _plane(self, center: np.ndarray, direction: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        e1, e2 = _perpendicular_basis(direction)
        return center + offsets[:, :1] * e1 + offsets[:, 1:] * e2

    def core_intensity(self, center: np.ndarray, direction: np.ndarray) -> float:
        return float(sample_image(self.image, self._plane(center, direction, self.core)).mean())

    def recenter(self, center: np.ndarray, direction: np.ndarray, own_label: int = 0) -> np.ndarray:
        points = self._plane(center, direction, self.window)
        weights = np.clip(sample_image(self.image, points), 0.0, None)
        owners = sample_labels(self.claimed, points)
        foreign = (owners > 0) & (owners != own_label)
        weights = np.where(foreign, 0.15 * weights, weights)
        distance2 = ((points - center) ** 2).sum(axis=1)
        weights *= np.exp(-distance2 / (2.0 * (0.8 * self.radius) ** 2))
        if weights.sum() <= 1e-9:
            return center
        shift = (weights[:, None] * (points - center)).sum(axis=0) / weights.sum()
        limit = 0.4 * self.radius
        length = np.linalg.norm(shift)
        if length > limit:
            shift *= limit / length
        return center + shift

    def _turn(self, current: np.ndarray, proposed: np.ndarray) -> np.ndarray:
        cosine = float(np.clip(current @ proposed, -1.0, 1.0))
        angle = np.arccos(cosine)
        if angle <= self.max_turn or angle < 1e-9:
            return proposed
        axis = proposed - cosine * current
        axis /= np.linalg.norm(axis)
        return np.cos(self.max_turn) * current + np.sin(self.max_turn) * axis

    def trace_one_way(
        self, start: np.ndarray, direction: np.ndarray, max_steps: int, own_label: int = 0
    ) -> list[np.ndarray]:
        """Step from ``start`` along ``direction`` until the core leaves the fiber.

        Voxels of ``claimed`` labelled ``own_label`` are not down-weighted
        (the fiber being extended).
        """
        points: list[np.ndarray] = []
        position = start.copy()
        misses = 0
        for _ in range(max_steps):
            candidate = position + self.step * direction
            if not self._inside(candidate):
                break
            candidate = self.recenter(candidate, direction, own_label)
            axis, tubularity = self.hessian.directions(candidate)
            axis = axis[0]
            if axis @ direction < 0:
                axis = -axis
            moved = candidate - position
            moved_length = np.linalg.norm(moved)
            moved = moved / moved_length if moved_length > 1e-9 else direction
            # Trust the Hessian only where the image looks like a single tube.
            hessian_weight = 0.35 if tubularity[0] > 0.05 else 0.0
            proposed = 0.5 * direction + 0.15 * moved + hessian_weight * axis
            proposed /= np.linalg.norm(proposed)
            direction = self._turn(direction, proposed)
            if self.core_intensity(candidate, direction) < 0.5:
                misses += 1
                if misses >= 2:
                    break
            else:
                misses = 0
            position = candidate
            points.append(candidate)
        # drop trailing unsupported points
        while points and self.core_intensity(points[-1], direction) < 0.5:
            points.pop()
        return points

    def trace(self, seed: np.ndarray, max_steps: int) -> np.ndarray:
        axis, _ = self.hessian.directions(seed)
        direction = axis[0] / np.linalg.norm(axis[0])
        center = self.recenter(seed, direction)
        forward = self.trace_one_way(center, direction, max_steps)
        backward = self.trace_one_way(center, -direction, max_steps)
        return np.array(backward[::-1] + [center] + forward)


def _drop_claimed(line: np.ndarray, claimed: np.ndarray, max_covered: float = 0.3) -> np.ndarray:
    """Trim end runs that follow an already-fitted fiber; reject mostly-covered traces.

    A new trace can start in a gap between fits and then follow an existing
    fiber, which would create a second fit on the same fiber.
    """
    if len(line) == 0:
        return line
    index = np.clip(np.floor(line[:, ::-1]).astype(int), 0, np.array(claimed.shape) - 1)
    covered = claimed[index[:, 0], index[:, 1], index[:, 2]] > 0
    start, stop = 0, len(line)
    while start < stop and covered[start]:
        start += 1
    while stop > start and covered[stop - 1]:
        stop -= 1
    line, covered = line[start:stop], covered[start:stop]
    if len(line) == 0 or covered.mean() > max_covered:
        return line[:0]
    return line


def foreground_depth(foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The foreground's distance transform (float32) and its 3×3×3 maximum."""
    from scipy.ndimage import distance_transform_edt, maximum_filter

    edt = distance_transform_edt(foreground).astype(np.float32)
    return edt, maximum_filter(edt, size=3)


def ridge_seeds(
    foreground: np.ndarray,
    radius: float,
    *,
    exclude: np.ndarray | None = None,
    min_depth_radii: float = 0.5,
    depth: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Voxel coordinates of distance-transform ridge points at least
    ``min_depth_radii`` radii deep, deepest first.

    ``depth`` is the foreground's distance transform and its 3×3×3 maximum,
    when the caller already has them (see :func:`foreground_depth`)."""
    edt, peak = depth if depth is not None else foreground_depth(foreground)
    ridge = (edt >= max(min_depth_radii * radius, 1.0)) & (edt >= peak)
    if exclude is not None:
        ridge &= exclude == 0
    z, y, x = np.nonzero(ridge)
    order = np.argsort(-edt[z, y, x], kind="stable")
    return np.stack([x[order], y[order], z[order]], axis=1).astype(np.float64) + 0.5


def trace_fibers(
    image: np.ndarray,
    hessian: HessianField,
    *,
    radius: float,
    min_bend_radius: float,
    min_length: float,
    node_spacing: float,
    claimed: np.ndarray | None = None,
    foreground: np.ndarray | None = None,
    label_offset: int = 0,
    max_fibers: int | None = None,
    seed_depth_radii: float = 0.5,
    depth: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Trace fibers from ridge seeds that are not yet explained by ``claimed``.

    Seeds must be at least ``seed_depth_radii`` radii from the void, so a
    large fiber type is only seeded where the foreground is that thick."""
    if claimed is None:
        claimed = np.zeros(image.shape, dtype=np.int32)
    else:
        claimed = claimed.copy()
    if foreground is None:
        foreground = image > 0.5
    tracer = Tracer(
        image,
        hessian,
        radius=radius,
        min_bend_radius=min_bend_radius,
        step=max(0.75, 0.5 * radius),
        claimed=claimed,
    )
    max_steps = int(4 * sum(image.shape) / tracer.step)
    fibers: list[np.ndarray] = []
    for seed in ridge_seeds(
        foreground, radius, exclude=claimed, min_depth_radii=seed_depth_radii, depth=depth
    ):
        index = tuple(np.floor(seed[::-1]).astype(int))
        if claimed[index] != 0:
            continue
        line = _drop_claimed(tracer.trace(seed, max_steps), claimed)
        label = label_offset + len(fibers) + 1
        if len(line) >= 2 and polyline_length(line) >= min_length:
            line = resample(line, node_spacing)
            fibers.append(line)
            paint(claimed, line, 1.1 * radius, label)
            if max_fibers is not None and len(fibers) >= max_fibers:
                break
        else:
            # Mark the rejected trace (or the seed) so nearby seeds on the
            # same blob are not traced again.
            rejected = line if len(line) else seed[None]
            paint(claimed, rejected, 0.75 * radius, -1, only_empty=True)
    return fibers
