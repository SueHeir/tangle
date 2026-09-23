"""Continuous fit: move every fiber toward the image, subject to its priors.

One iteration is an EM-like sweep:

1. Each voxel near a fiber is assigned to the nearest capsule surface
   (the same ownership rule Tangle's voxel exporter uses).
2. Each centerline segment moves laterally toward the intensity-weighted
   centroid of the voxels it owns (the data force).
3. A bending step pulls every interior node toward the midpoint of its
   neighbors (the persistence / bend-radius prior).
4. Each radius moves toward the equivalent radius of the intensity it owns,
   shrunk toward the user's diameter and clamped to the tolerance.
5. Ends grow along their tangent while the image is still fiber there, and
   retreat while it is not.
6. Overlapping capsules of different fibers are pushed apart.
"""

from __future__ import annotations

import numpy as np

from ._geometry import rasterize, resample, sample_image, tangents


def data_step(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    reach_factor: float,
    rate: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Apply the lateral data force; also return each fiber's owned intensity."""
    labels, _, segments = rasterize(
        image.shape, centerlines, radii, reach=reach_factor * radii, signed=True
    )
    owned = segments >= 0
    ids = segments[owned]
    weights = np.clip(image[owned], 0.0, 1.5).astype(np.float64)
    z, y, x = np.nonzero(owned)
    coordinates = np.stack([x, y, z], axis=1) + 0.5
    count = sum(max(len(c) - 1, 0) for c in centerlines)
    total = np.bincount(ids, weights=weights, minlength=count)
    centroid = np.stack(
        [np.bincount(ids, weights=weights * coordinates[:, a], minlength=count) for a in range(3)],
        axis=1,
    ) / np.maximum(total, 1e-9)[:, None]

    # Unclipped intensities for the mass, so zero-mean noise in the void cancels.
    signed_weights = np.clip(image[owned], -1.5, 1.5).astype(np.float64)
    fiber_mass = np.bincount(labels[owned] - 1, weights=signed_weights, minlength=len(centerlines))

    moved = []
    offset = 0
    for line in centerlines:
        n = len(line)
        if n < 2:
            moved.append(line)
            continue
        a, b = line[:-1], line[1:]
        middle = 0.5 * (a + b)
        axis = b - a
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
        shift = centroid[offset : offset + n - 1] - middle
        shift -= (shift * axis).sum(axis=1, keepdims=True) * axis
        shift[total[offset : offset + n - 1] < 1e-6] = 0.0
        node_shift = np.zeros_like(line)
        node_weight = np.zeros(n)
        node_shift[:-1] += shift
        node_shift[1:] += shift
        node_weight[:-1] += 1
        node_weight[1:] += 1
        moved.append(line + rate * node_shift / node_weight[:, None])
        offset += n - 1
    return moved, fiber_mass


def bend_step(centerlines: list[np.ndarray], rate: float, passes: int = 1) -> list[np.ndarray]:
    smoothed = []
    for line in centerlines:
        line = line.copy()
        for _ in range(passes):
            if len(line) > 2:
                line[1:-1] += rate * (0.5 * (line[:-2] + line[2:]) - line[1:-1])
        smoothed.append(line)
    return smoothed


def radius_step(
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    fiber_mass: np.ndarray,
    *,
    prior_radius: float,
    tolerance: float,
    prior_weight: float,
) -> np.ndarray:
    lengths = np.array([np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in centerlines])
    # Owned intensity is (partial-volume) area times length; the two
    # hemispherical caps add 4/3 of a radius of length.
    effective = np.maximum(lengths + 4.0 / 3.0 * radii, 1e-9)
    measured = np.sqrt(np.maximum(fiber_mass, 0.0) / (np.pi * effective))
    blended = (measured + prior_weight * prior_radius) / (1.0 + prior_weight)
    return np.clip(blended, prior_radius * (1 - tolerance), prior_radius * (1 + tolerance))


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


def separate_step(centerlines: list[np.ndarray], radii: np.ndarray, *, passes: int = 2, margin: float = 0.0) -> list[np.ndarray]:
    """Push apart nodes of different fibers closer than the sum of radii."""
    from scipy.spatial import cKDTree

    lines = [line.copy() for line in centerlines]
    if len(lines) < 2:
        return lines
    for _ in range(passes):
        points = np.concatenate(lines)
        owner = np.concatenate([np.full(len(line), i) for i, line in enumerate(lines)])
        starts = np.cumsum([0] + [len(line) for line in lines])
        tree = cKDTree(points)
        pairs = tree.query_pairs(2.0 * float(radii.max()) + margin, output_type="ndarray")
        if len(pairs) == 0:
            break
        pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
        if len(pairs) == 0:
            break
        delta = points[pairs[:, 0]] - points[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        limit = radii[owner[pairs[:, 0]]] + radii[owner[pairs[:, 1]]] + margin
        overlap = limit - distance
        push = overlap > 0
        if not push.any():
            break
        pairs, delta, distance, overlap = pairs[push], delta[push], distance[push], overlap[push]
        direction = delta / np.maximum(distance, 1e-9)[:, None]
        correction = np.zeros_like(points)
        counts = np.zeros(len(points))
        np.add.at(correction, pairs[:, 0], 0.5 * overlap[:, None] * direction)
        np.add.at(correction, pairs[:, 1], -0.5 * overlap[:, None] * direction)
        np.add.at(counts, pairs[:, 0], 1)
        np.add.at(counts, pairs[:, 1], 1)
        points = points + correction / np.maximum(counts, 1)[:, None]
        lines = [points[starts[i] : starts[i + 1]] for i in range(len(lines))]
    return lines


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
