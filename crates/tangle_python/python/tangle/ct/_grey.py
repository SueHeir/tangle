"""Fitting the raw grey scan with a measured grey profile per fiber type.

A fiber type's **profile** (``FiberSpec.profile``) is its grey value in the
scan, after the fitter's light denoise, at evenly spaced radii from the axis
(first value) to the surface (last value), for example a bright rim around
a dim core. With profiles the fitter:

* derives each type's grey range (what counts as fiber of that type: the
  bright body of the profile) from the profile and the scan's noise
  (:func:`profile_levels`);
* draws a fit as the grey the scan should show (:func:`render_grey`): each
  voxel takes the profile of the fiber whose surface is nearest, at its
  distance from that fiber's axis, and outside the surface fades from the
  profile's last value to the void grey over ``2 × EDGE`` voxels; that is
  compared with the scan (:func:`squared_residual_map`). A profile is
  measured on the scan itself (:func:`measure_profiles`), so it already
  holds the scan's blur inside the fiber;
* picks each fiber's type by which profile, at that type's radius, best
  matches the grey across the fiber (:func:`profile_types`).

Everything here is in voxel units, points as ``(x, y, z)``, volumes as
``(z, y, x)``.
"""

from __future__ import annotations

import numpy as np

EDGE = 1.2  # soft edge half-width, voxels (as ``_moves.render_occupancy``)


def profile_radii(profile: np.ndarray) -> np.ndarray:
    """The radii, as fractions of the fiber radius, a profile's values are given at."""
    return np.linspace(0.0, 1.0, len(profile))


def profile_levels(
    grey: np.ndarray,
    profiles: list[np.ndarray],
    radii: np.ndarray | None = None,
    *,
    spread: float = 2.5,
    core: float = 1.5,
) -> tuple[float, float, list[tuple[float, float]]]:
    """The void grey, the noise, and each type's grey range.

    The void grey is the median of the voxels darker than every profile's
    dimmest value, and the noise the robust spread (1.4826 × MAD) of those
    below halfway to it. A type's range is its profile's bright core, the
    dimmest profile value within ``core`` voxels of the brightest one (given
    the type's radius in voxels, ``radii``; the brightest value alone
    without), up to the brightest value, widened by ``spread`` noise (at
    least 5% of the contrast), the low end kept at least 60% of the way from
    void to the peak.

    So the core counts as fiber at least ``core`` voxels to each side of
    the peak. A thin fiber's blurred profile only peaks on the axis, and a
    range of the peak alone left it about one voxel wide (two_types lost 28
    of 101 fine fibers); for a thicker fiber the core is near the peak
    anyway. Reaching further down the profile, to its whole bright body,
    let the dim contact between touching fibers read as fiber, so they
    merged (scenario_missed_fiber 12 → 10 of 12; with the whole profile,
    single_type 37 → 21 of 40). Edges are partial fiber through
    ``range_image``'s ramp, and a dim core below the range is filled as a
    hole.
    """
    dimmest = min(float(np.min(p)) for p in profiles)
    below = grey[grey < dimmest]
    if below.size == 0:
        raise ValueError("no voxel is darker than every fiber profile, so there is no void to compare with")
    void = float(np.median(below))
    if void >= dimmest:
        raise ValueError("the void grey is not below the fiber profiles")
    quiet = below[below < 0.5 * (void + dimmest)]
    noise = 1.4826 * float(np.median(np.abs(quiet - np.median(quiet)))) if quiet.size else 0.0
    ranges = []
    for k, p in enumerate(profiles):
        p = np.asarray(p, dtype=np.float64)
        bright = float(np.max(p))
        floor = bright
        if radii is not None:
            at = np.linspace(0.0, 1.0, 101) * float(radii[k])  # voxels from the axis
            values = np.interp(at, profile_radii(p) * float(radii[k]), p)
            peak = at[int(np.argmax(values))]
            floor = float(np.min(values[np.abs(at - peak) <= core]))
        half = max(spread * noise, 0.05 * (bright - void))
        ranges.append((max(floor - half, void + 0.6 * (bright - void)), bright + half))
    return void, noise, ranges


def render_grey(
    low: np.ndarray,
    high: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    profiles: list[np.ndarray],
    void: float,
    edge: float = EDGE,
) -> np.ndarray:
    """The grey the fibers should show over the voxel box ``[low, high)`` (x, y, z); in Rust."""
    from . import _native

    return _native.render_grey(low, high, lines, radii, profiles, void, edge)


def squared_residual_map(
    grey: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    profiles: list[np.ndarray],
    void: float,
) -> np.ndarray:
    """Per voxel, (scan grey − the fit's drawn grey)²."""
    drawn = render_grey(np.zeros(3, dtype=int), np.array(grey.shape[::-1]), lines, radii, profiles, void)
    return ((grey - drawn) ** 2).astype(np.float32)


def evidence_scale(
    grey: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    profiles: list[np.ndarray],
    void: float,
    radius: float,
) -> float:
    """Squared grey residual worth one nat, ``2 σ² π r²``, σ² measured near the fibers."""
    area = np.pi * radius * radius
    if not lines:
        return 2.0 * area
    from ._geometry import rasterize

    _, best, _ = rasterize(grey.shape, lines, radii, reach=np.asarray(radii) + 2.0, signed=True)
    near = np.isfinite(best)
    if not near.any():
        return 2.0 * area
    residual = squared_residual_map(grey, lines, radii, profiles, void)[near]
    contrast = max(max(float(np.max(p)) for p in profiles) - void, 1e-6)
    variance = max(float(np.mean(residual)), 1e-4 * contrast * contrast)
    return 2.0 * variance * area


def profile_types(
    grey: np.ndarray,
    lines: list[np.ndarray],
    type_radii: np.ndarray,
    profiles: list[np.ndarray],
    void: float,
    *,
    decisive: float = 0.7,
    nodes: int = 24,
) -> np.ndarray:
    """Each fiber's type by the profile that best matches the grey across it (−1 when unclear).

    At up to ``nodes`` interior nodes, the grey is sampled across the fiber
    in four directions at 0 … 1.4 of each type's radius and compared with
    that type's profile (void beyond the surface). A type is taken when its
    median squared misfit is below ``decisive`` times the next type's.
    """
    from ._geometry import cross_frames, sample_image

    count = len(profiles)
    out = np.full(len(lines), -1, dtype=int)
    if count < 2:
        return out
    fractions = np.linspace(0.0, 1.4, 8)
    expected = []
    for p, r in zip(profiles, type_radii):
        p = np.asarray(p, dtype=np.float64)
        inside = np.interp(np.minimum(fractions, 1.0), profile_radii(p), p)
        weight = np.clip(1.0 - (fractions - 1.0) * float(r) / (2 * EDGE), 0.0, 1.0)  # as render_grey
        expected.append(void + weight * (inside - void))
    frames = [cross_frames(line, nodes) for line in lines]  # per fiber: sample centers and four directions
    used = [i for i, frame in enumerate(frames) if frame is not None]
    if not used:
        return out
    misfit = np.zeros((len(used), count))
    for k in range(count):
        r = float(type_radii[k])
        batches = [
            frames[i][0][:, None, None, :] + frames[i][1][:, :, None, :] * (fractions * r)[None, None, :, None]
            for i in used
        ]
        sizes = [len(b) for b in batches]
        got = sample_image(grey, np.concatenate(batches).reshape(-1, 3), fill=void).reshape(-1, 4, len(fractions))
        per_node = ((got - expected[k][None, None, :]) ** 2).sum(axis=(1, 2))
        for row, part in enumerate(np.split(per_node, np.cumsum(sizes)[:-1])):
            misfit[row, k] = float(np.median(part))
    order = np.argsort(misfit, axis=1)
    rows = np.arange(len(used))
    clear = misfit[rows, order[:, 0]] < decisive * misfit[rows, order[:, 1]]
    out[np.array(used)[clear]] = order[clear, 0]
    return out


def measure_profiles(
    grey: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    types: np.ndarray,
    count: int,
    samples: int = 9,
) -> list[np.ndarray]:
    """Each type's profile, measured from known fibers: the median grey at each radius fraction.

    ``centerlines`` / ``radii`` / ``types`` describe fibers whose positions
    are known (a few traced by hand, or a synthetic scan's truth). Voxels are
    binned by their distance from the nearest fiber's axis over its radius.
    """
    from ._geometry import rasterize

    radii = np.asarray(radii, dtype=np.float64)
    owner, best, _ = rasterize(grey.shape, centerlines, radii, reach=radii, signed=True)
    owned = owner > 0
    fiber = owner[owned] - 1
    fraction = (best[owned] + radii[fiber]) / radii[fiber]
    values = grey[owned]
    kinds = np.asarray(types, dtype=int)[fiber]
    xs = np.linspace(0.0, 1.0, samples)
    half = 0.5 / (samples - 1)
    profiles = []
    for k in range(count):
        mine = kinds == k
        row = []
        for x in xs:
            window = mine & (np.abs(fraction - x) <= half)
            row.append(float(np.median(values[window])) if window.any() else np.nan)
        row = np.array(row)
        known = np.isfinite(row)
        row[~known] = np.interp(xs[~known], xs[known], row[known]) if known.any() else 0.0
        profiles.append(row)
    return profiles
