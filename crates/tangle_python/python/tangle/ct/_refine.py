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


def cut_void(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    level: float = 0.3,
    min_gap_radii: float = 2.0,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, int]]:
    """Cut every fit where its centerline sits in void (image below ``level``).

    A fit that has drifted off its fiber into empty space (the solver's image
    force only pulls toward fiber, so void never pushes back) keeps a tail
    there, and another fit can take the place it left. Void nodes at an end
    are trimmed back to the first supported node. An interior void stretch
    shorter than ``min_gap_radii`` radii is kept (a dip, not a drift); a
    longer one is bridged when the straight line between its supported
    neighbors is fiber all the way (the fit bowed off its fiber and back),
    and otherwise splits the fit (a real fiber has no gap). Nodes outside the
    scan are left alone. :func:`end_step` regrows an end where the scan does
    continue.

    Returns ``(pieces, source, counts)``: the pieces, the index of the fit
    each came from, and the numbers of nodes dropped (``"trimmed"``), of
    splits and of stretches bridged (``"bridged"``).
    """
    upper = np.array(image.shape[::-1], dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    pieces: list[np.ndarray] = []
    source: list[int] = []
    counts = {"trimmed": 0, "splits": 0, "bridged": 0}
    for index, line in enumerate(centerlines):
        line = np.asarray(line, dtype=np.float64)
        if len(line) < 2:
            pieces.append(line)
            source.append(index)
            continue
        inside = np.all((line >= 0.5) & (line <= upper - 0.5), axis=1)
        keep = ~((sample_image(image, line) < level) & inside)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        spacing = max(arc[-1] / (len(line) - 1), 1e-6)
        bridges: list[tuple[int, int, np.ndarray]] = []  # (first void node, first node after, new nodes)
        start = None
        for k in range(len(line) + 1):
            weak = k < len(line) and not keep[k]
            if weak and start is None:
                start = k
            elif not weak and start is not None:
                if start > 0 and k < len(line):
                    if arc[k] - arc[start - 1] < min_gap_radii * radii[index]:
                        keep[start:k] = True
                    else:
                        a, b = line[start - 1], line[k]
                        count = max(int(np.ceil(np.linalg.norm(b - a) / spacing)), 2)
                        across = a + (b - a) * np.linspace(0.0, 1.0, count + 1)[1:-1, None]
                        if float(sample_image(image, across).min()) >= level:
                            bridges.append((start, k, across))
                            keep[start:k] = True
                start = None
        counts["trimmed"] += int((~keep).sum())
        if bridges:
            parts, flags, previous = [], [], 0
            for first, after, across in bridges:
                parts += [line[previous:first], across]
                flags += [keep[previous:first], np.ones(len(across), dtype=bool)]
                previous = after
            parts.append(line[previous:])
            flags.append(keep[previous:])
            line, keep = np.vstack(parts), np.concatenate(flags)
            counts["bridged"] += len(bridges)
        runs = np.split(np.arange(len(line)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
        kept = [line[run] for run in runs if keep[run[0]] and len(run) >= 2]
        counts["splits"] += max(len(kept) - 1, 0)
        pieces += kept
        source += [index] * len(kept)
    return pieces, np.array(source, dtype=int), counts


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
