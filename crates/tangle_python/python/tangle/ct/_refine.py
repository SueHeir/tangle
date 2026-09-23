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
) -> list[np.ndarray]:
    """Grow or trim fiber ends by up to ``max_moves`` steps of length ``step``."""
    upper = np.array(image.shape[::-1], dtype=np.float64)
    adjusted = []
    for index, line in enumerate(centerlines):
        line = line.copy()
        radius = radii[index]
        for end in (0, -1):
            for _ in range(max_moves):
                if len(line) < 3:
                    break
                inner = line[1] if end == 0 else line[-2]
                tip = line[end]
                direction = tip - inner
                direction /= max(np.linalg.norm(direction), 1e-12)
                probe = tip + max(step, 0.5 * radius) * direction
                inside = np.all(probe >= 0.5) and np.all(probe <= upper - 0.5)
                ahead = float(sample_image(image, probe[None])[0]) if inside else 0.0
                owner = int(occupied[tuple(np.clip(np.floor(probe[::-1]).astype(int), 0, np.array(image.shape) - 1))]) if inside else 0
                here = float(sample_image(image, tip[None])[0])
                if inside and ahead > 0.55 and owner in (0, index + 1):
                    extended = tip + step * direction
                    line = np.vstack([extended[None], line]) if end == 0 else np.vstack([line, extended[None]])
                elif here < 0.45:
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
