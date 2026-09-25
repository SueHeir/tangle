"""Fiber-length prior: fiber ends are rare, and every end must be paid for.

If fibers of mean length ``L`` have their ends spread uniformly through the
material, a scan contains ``2 Λ / L`` fiber ends, where ``Λ`` is the total
fiber length inside the scan, however the scan boundary cuts the fibers.
Ends on the scan boundary are fibers leaving the volume and do not count.

That gives the fitter two things:

* a check: the length implied by a fit, ``L̂ = 2 Λ / (interior ends)``,
  against the expected length (a much shorter ``L̂`` means over-splitting);
* a price: a break somewhere along a fiber has prior probability of roughly
  ``D / L`` per resolvable position (``D`` the diameter), so every interior
  end costs ``ln(L / D)`` nats. A move that adds ends (a split, a new fiber)
  must improve the match to the scan by at least that much, and a move that
  removes ends (a join) gets it as a bonus.

Image evidence is measured in the same units: the change in squared residual
between two renderings, divided by ``2 σ² A``, where ``σ²`` is the fit's
own mean squared residual per voxel near the fibers (image noise plus model
misfit, such as lobed cross-sections) and ``A = π r²`` is one fiber
cross-section, the scale on which neighboring voxels move together.
"""

from __future__ import annotations

import math

import numpy as np

from ._geometry import polyline_length, rasterize


def end_cost(length: float | None, diameter: float) -> float:
    """Prior cost of one interior fiber end in nats (0 without a length)."""
    if not length or length <= diameter:
        return 0.0
    return math.log(length / diameter)


def length_end_cost(so_far: float, length: float | None, diameter: float, shape: float) -> float:
    """Nats for a fiber end after ``so_far`` of fiber, fiber lengths gamma(``shape``) with mean ``length``.

    The price is ``-ln(h(so_far) D)``, ``h`` the chance per unit length that
    a fiber of that length ends there (the hazard). Shape 1 (exponential
    lengths) gives the constant ``ln(L / D)`` of :func:`end_cost`; a larger
    shape makes ending a short fiber dear and ending one near or past ``L``
    cheap. Always at least one nat (0 without a length: then 1).
    """
    if not length or length <= diameter:
        return 1.0
    from scipy.stats import gamma

    x = max(float(so_far), diameter)
    model = gamma(shape, scale=length / shape)
    return max(float(model.logsf(x) - model.logpdf(x)) - math.log(diameter), 1.0)


def length_join_cost(a: float, b: float, bridge: float, length: float | None, shape: float) -> float:
    """Nats for joining fibers of lengths ``a`` and ``b`` through ``bridge``: the joined fiber's
    cumulative hazard less the two parts' (0 for exponential lengths, growing as the joined
    fiber runs past ``length``)."""
    if not length:
        return 0.0
    from scipy.stats import gamma

    model = gamma(shape, scale=length / shape)
    joined = -float(model.logsf(a + b + bridge))
    return max(joined + float(model.logsf(a)) + float(model.logsf(b)), 0.0)


def interior_end_mask(lines: list[np.ndarray], radii: np.ndarray, shape: tuple[int, int, int], margin_radii: float = 1.5) -> np.ndarray:
    """``(n, 2)`` booleans: is each fiber's start / end away from the scan boundary."""
    upper = np.array(shape[::-1], dtype=np.float64)
    mask = np.zeros((len(lines), 2), dtype=bool)
    for i, line in enumerate(lines):
        if len(line) == 0:
            continue
        margin = margin_radii * float(radii[i])
        for e, tip in enumerate((line[0], line[-1])):
            mask[i, e] = bool(np.all(tip >= margin) and np.all(tip <= upper - margin))
    return mask


def end_statistics(
    lines: list[np.ndarray],
    radii: np.ndarray,
    shape: tuple[int, int, int],
    *,
    length: float | None = None,
) -> dict[str, float | int | None]:
    """Interior ends found, the fiber length they imply, and (with ``length``)
    the number expected. Lengths are in the units of ``lines``."""
    total = float(sum(polyline_length(line) for line in lines))
    interior = int(interior_end_mask(lines, radii, shape).sum())
    stats: dict[str, float | int | None] = {
        "interior_ends": interior,
        "total_length": total,
        "implied_length": 2.0 * total / interior if interior else None,
    }
    if length:
        expected = 2.0 * total / length
        stats["expected_interior_ends"] = expected
        stats["expected_interior_ends_sd"] = math.sqrt(expected)
    return stats


def render_soft(best: np.ndarray, edge: float = 1.2) -> np.ndarray:
    """Soft occupancy from ``rasterize(..., signed=True)`` surface distances."""
    occupancy = np.zeros(best.shape, dtype=np.float32)
    finite = np.isfinite(best)
    occupancy[finite] = np.clip(0.5 - best[finite] / (2.0 * edge), 0.0, 1.0)
    return occupancy


def evidence_scale(image: np.ndarray, lines: list[np.ndarray], radii: np.ndarray, radius: float) -> float:
    """Squared-residual change worth one nat: ``2 σ² π r²``."""
    if not lines:
        return 2.0 * np.pi * radius * radius
    _, best, _ = rasterize(image.shape, lines, radii, reach=radii + 2.0, signed=True)
    near = np.isfinite(best)
    residual = image[near] - render_soft(best)[near]
    variance = float(np.mean(residual * residual)) if residual.size else 1.0
    return 2.0 * max(variance, 1e-4) * np.pi * radius * radius


def local_box(points: np.ndarray, margin: float, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    upper = np.array(shape[::-1])
    low = np.maximum(np.floor(points.min(axis=0) - margin).astype(int), 0)
    high = np.minimum(np.ceil(points.max(axis=0) + margin).astype(int), upper)
    return low, high


def local_residual(
    image: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    base: np.ndarray | None = None,
) -> float:
    """Squared residual of rendering ``lines`` against ``image`` over box ``[low, high)``.

    ``base`` is an already-rendered occupancy of the box (for example the
    unchanged neighbors) that ``lines`` are added to.
    """
    from . import _native

    if np.any(np.asarray(high) <= np.asarray(low)):
        return 0.0
    return _native.local_residual(image, low, high, lines, radii, base, 1.2)


def near_box(lines: list[np.ndarray], radii: np.ndarray, low: np.ndarray, high: np.ndarray, skip: tuple[int, ...] = ()) -> list[int]:
    """Indices of fibers whose bounding boxes reach into ``[low, high)``."""
    return [
        k for k, line in enumerate(lines)
        if k not in skip and len(line) >= 2
        and np.all(line.min(axis=0) - 2 * radii[k] < high)
        and np.all(line.max(axis=0) + 2 * radii[k] > low)
    ]
