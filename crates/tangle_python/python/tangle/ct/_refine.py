"""Host-side steps of the continuous fit: fiber ends, node spacing, support.

The fibers themselves move in Tangle's solver (``_device``).
"""
from __future__ import annotations

import numpy as np

from ._geometry import resample, sample_image, tangents


def end_step(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    step: float,
    occupied: np.ndarray,
    max_moves: int = 3,
    reach: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Grow or trim fiber ends by up to ``max_moves`` steps of length ``step``.

    A fiber's capsule reaches a radius past its last node, so the scan's
    foreground ends about ``reach`` (the radius, plus any margin by which the
    foreground over-reaches) beyond the true end of the centerline. An end
    grows while the scan is still fiber half a step past that, and is trimmed
    while it is void half a step short of it, which leaves the tip within
    half a step of the true end.
    """
    upper = np.array(image.shape[::-1], dtype=np.float64)
    reach = np.asarray(radii if reach is None else reach, dtype=np.float64)

    def inside(point: np.ndarray) -> bool:
        return bool(np.all(point >= 0.5) and np.all(point <= upper - 0.5))

    def value(point: np.ndarray) -> float:
        return float(sample_image(image, point[None])[0])

    adjusted = []
    for index, line in enumerate(centerlines):
        line = line.copy()
        cap = float(reach[index])
        for end in (0, -1):
            for _ in range(max_moves):
                if len(line) < 3:
                    break
                inner = line[1] if end == 0 else line[-2]
                tip = line[end]
                direction = tip - inner
                direction /= max(np.linalg.norm(direction), 1e-12)
                ahead = tip + (cap + 0.5 * step) * direction
                short = tip + max(cap - 0.5 * step, 0.0) * direction
                owner = (
                    int(occupied[tuple(np.clip(np.floor(ahead[::-1]).astype(int), 0, np.array(image.shape) - 1))])
                    if inside(ahead)
                    else 0
                )
                if inside(ahead) and value(ahead) > 0.55 and owner in (0, index + 1):
                    extended = tip + step * direction
                    line = np.vstack([extended[None], line]) if end == 0 else np.vstack([line, extended[None]])
                elif value(tip) < 0.45 or (inside(short) and value(short) < 0.45):
                    # (A fiber that leaves the scan is not trimmed at the boundary.)
                    line = line[1:] if end == 0 else line[:-1]
                else:
                    break
        adjusted.append(line)
    return adjusted


def respace(centerlines: list[np.ndarray], spacing: float) -> list[np.ndarray]:
    return [resample(line, spacing) for line in centerlines]


def support(image: np.ndarray, centerlines: list[np.ndarray]) -> np.ndarray:
    """Mean normalized intensity along each centerline (about 1 on a real fiber)."""
    return np.array([float(sample_image(image, line).mean()) if len(line) else 0.0 for line in centerlines])


def curvature_ratio(centerlines: list[np.ndarray], min_bend_radius: float) -> np.ndarray:
    """Largest discrete curvature times the admissible bend radius, per fiber."""
    ratios = []
    for line in centerlines:
        if len(line) < 3:
            ratios.append(0.0)
            continue
        t = tangents(line)
        turn = np.arccos(np.clip((t[1:] * t[:-1]).sum(axis=1), -1.0, 1.0))
        spacing = np.linalg.norm(np.diff(line, axis=0), axis=1)
        ratios.append(float((turn / np.maximum(spacing, 1e-9)).max() * min_bend_radius))
    return np.array(ratios)
