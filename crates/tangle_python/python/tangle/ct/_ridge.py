"""Ridge finish (``FitSettings.ridge_finish``): a last pass over the smallest type's fits on the grey ridge.

Inside a packed bundle the range image is saturated, so a fit has nothing to keep it on its own fiber: where
a neighbour comes close it slides across and carries on along the neighbour (the fiber it left looks chopped
and taken over), and fits sit beside their axes. The grey itself, smoothed a little (``recenter_sigma_voxels``,
about half a fine radius), still peaks on every axis. A node is *on the ridge* when the brightest point of a
disc one radius wide across the fit lies within half a radius of it. The finish, on the fits sampled every half
voxel:

1. recenter each fit onto the ridge (disc argmax, half a radius reach, smoothed along the fit);
2. cut out every run of samples off the ridge at least ``_CUT_RUN`` voxels long (a hop's crossing, a stretch
   between fibers, an overhanging end): interior runs split the fit, end runs trim it;
3. grow each end along the ridge while the grey stays at least ``ridge_finish_extend`` of the fit's own median
   grey and no other fit is within half a fiber pitch;
4. join ends of pieces that meet: within 6 voxels, and either both end directions within 30 degrees of the
   bridge, or the ends parallel within 30 degrees and at most 0.6 radius apart across the fiber, or within 4
   voxels running on past each other (the second piece's overlapping start is dropped); at least 80% of the
   straight bridge must be on the ridge;
5. drop pieces shorter than the type's minimum length.

A larger type's fiber is off limits throughout: fine samples inside a coarse fit's oval section are cut (its
bright rim is a ridge too, and a fine fit can follow it), and ends are not grown into one.
"""

from __future__ import annotations

import numpy as np

from ._evaluate import _samples_inside
from ._geometry import sample_image

_STEP = 0.5  # voxels between samples
_CUT_RUN = 3.0  # voxels: shortest off-ridge run cut out
_JOIN_GAP = 6.0  # voxels
_JOIN_COS = float(np.cos(np.radians(30.0)))
_JOIN_RIDGE = 0.8
_JOIN_OVERLAP = 4.0  # voxels: ends this close that run on past each other (side by side) also join


def ridge_finish(fitter, lines, radii, types, grey: np.ndarray):
    """``(lines, radii, types, info)`` after the ridge finish (see the module notes)."""
    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    shape = tuple(int(n) for n in grey.shape)
    fine = int(np.argmin(fitter.radius))
    r = float(fitter.radius[fine])
    min_length = float(fitter.min_length[fine])
    alpha = float(fitter.settings.ridge_finish_extend)
    samples = [_samples_inside(np.asarray(line, dtype=np.float64), shape, _STEP) for line in lines]
    is_fine = [int(k) == fine and len(s) >= 5 for s, k in zip(samples, types)]
    inside_coarse = _coarse_sections(fitter, lines, radii, types, fine)
    # 1. recenter
    samples = [_recenter(grey, s, r, shape) if ok else s for s, ok in zip(samples, is_fine)]
    # 2. cut off-ridge runs
    pieces, others = [], []
    cut_samples = 0
    for s, ok, k, radius in zip(samples, is_fine, types, radii):
        if not ok:
            others.append((s, k, radius))
            continue
        _, dist = _argmax_offset(grey, s, _tangents(s), r, smooth=7)
        keep = np.ones(len(s), dtype=bool)
        for a, b in _runs(dist > 0.5 * r):
            if (b - a) * _STEP >= _CUT_RUN:
                keep[a:b] = False
        keep &= ~inside_coarse(s)
        cut_samples += int((~keep).sum())
        pieces += [(p, radius) for p in _pieces(s, keep) if len(p) >= 4]
    # 3. extend along the ridge
    grown = 0
    all_points = [p for p, _ in pieces] + [s for s, _, _ in others if len(s)]
    for i, (p, radius) in enumerate(pieces):
        if len(p) < 12:
            continue
        rest = [q for j, q in enumerate(all_points) if j != i and len(q)]
        p2, added = _extend(grey, p, rest, r, alpha, shape, forbidden=inside_coarse)
        pieces[i] = (p2, radius)
        all_points[i] = p2
        grown += added
    # 4. join
    joined_pieces, joined = _join(grey, [p for p, _ in pieces], r, shape)
    radius_of_piece = float(np.median([rad for _, rad in pieces])) if pieces else r
    # 5. minimum length
    kept = [p for p in joined_pieces if len(p) * _STEP >= min_length]
    out_lines = [_to_nodes(p, fitter.spacing) for p in kept] + [
        _to_nodes(s, fitter.spacing) if len(s) >= 2 else s for s, _, _ in others
    ]
    out_radii = np.concatenate([np.full(len(kept), radius_of_piece), np.array([rad for _, _, rad in others])])
    out_types = np.concatenate([np.full(len(kept), fine), np.array([k for _, k, _ in others], dtype=int)]).astype(int)
    keep = [i for i, line in enumerate(out_lines) if len(line) >= 2]
    info = {
        "cut_voxels": round(cut_samples * _STEP, 1), "grown_voxels": round(grown * _STEP, 1), "joins": joined,
        "short_dropped": len(joined_pieces) - len(kept),
    }
    return [out_lines[i] for i in keep], out_radii[keep], out_types[keep], info


def _to_nodes(samples: np.ndarray, spacing: float) -> np.ndarray:
    from ._refine import respace

    return respace([np.asarray(samples, dtype=np.float64)], spacing)[0]


def _tangents(s: np.ndarray, half: int = 4) -> np.ndarray:
    n = len(s)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    t = s[i1] - s[i0]
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def _frames(t: np.ndarray):
    helper = np.where(np.abs(t[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(t, helper)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    return e1, np.cross(t, e1)


def _disc(reach: float, step: float = 0.5):
    steps = np.arange(-reach, reach + 1e-9, step)
    u, v = np.meshgrid(steps, steps)
    keep = u**2 + v**2 <= reach**2 + 1e-9
    return u[keep], v[keep]


def _argmax_offset(grey, s, t, reach, smooth=1):
    """Per sample, the offset to the brightest point of a disc ``reach`` wide across the fit, and its length."""
    e1, e2 = _frames(t)
    u, v = _disc(reach)
    points = s[:, None, :] + u[None, :, None] * e1[:, None, :] + v[None, :, None] * e2[:, None, :]
    values = sample_image(grey, points.reshape(-1, 3)).reshape(len(s), len(u))
    if smooth > 1 and len(values) >= smooth:
        k = smooth // 2
        padded = np.concatenate([np.repeat(values[:1], k, 0), values, np.repeat(values[-1:], k, 0)])
        values = sum(padded[j : j + len(values)] for j in range(smooth)) / smooth
    best = np.argmax(values, axis=1)
    return u[best][:, None] * e1 + v[best][:, None] * e2, np.hypot(u[best], v[best])


def _recenter(grey, s, r, shape):
    shift, _ = _argmax_offset(grey, s, _tangents(s), 0.5 * r, smooth=7)
    if len(shift) >= 11:
        padded = np.concatenate([np.repeat(shift[:1], 5, 0), shift, np.repeat(shift[-1:], 5, 0)])
        shift = sum(padded[j : j + len(shift)] for j in range(11)) / 11
    return _samples_inside(s + shift, shape, _STEP)


def _runs(mask):
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _pieces(s, keep):
    return [s[a:b] for a, b in _runs(keep)]


def _coarse_sections(fitter, lines, radii, types, fine):
    """A test ``points -> bool per point``: inside the oval section of a larger type's fit."""
    from ._rescue import _coarse_index, _scaled_distance

    larger = [i for i, k in enumerate(types) if int(k) != fine and len(lines[i]) >= 2]
    if not larger:
        return lambda points: np.zeros(len(points), dtype=bool)
    by_kind = {}
    for i in larger:
        by_kind.setdefault(int(types[i]), []).append(i)
    tests = []
    for kind, ids in by_kind.items():
        index = _coarse_index(
            [np.asarray(lines[i], dtype=np.float64) for i in ids], np.asarray(radii, dtype=np.float64)[ids],
            [fitter.long_axes(np.asarray(lines[i], dtype=np.float64), kind) for i in ids],
        )
        tests.append((index, float(fitter.ratio[kind])))
    return lambda points: np.logical_or.reduce(
        [_scaled_distance(index, np.asarray(points, dtype=np.float64).reshape(-1, 3), ratio) <= 1.0
         for index, ratio in tests]
    )


def _extend(grey, s, others, r, alpha, shape, pitch_half=None, max_steps=80, forbidden=None):
    """``s`` grown at both ends along the ridge; returns it and how many samples were added."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.concatenate(others)) if others else None
    near = pitch_half if pitch_half is not None else 1.02 * r  # about half the pitch of touching fibers
    reference = float(np.median(sample_image(grey, s)))
    u, v = _disc(0.5 * r)
    upper = np.array(shape[::-1], dtype=np.float64)
    added = 0
    line = s
    for side in (0, 1):
        current = line if side == 1 else line[::-1]
        p = current[-1].copy()
        t = current[-1] - current[max(len(current) - 7, 0)]
        t /= max(np.linalg.norm(t), 1e-9)
        grown = []
        for _ in range(max_steps):
            q = p + t
            e1, e2 = _frames(t[None])
            points = q[None, :] + u[:, None] * e1 + v[:, None] * e2
            values = sample_image(grey, points)
            j = int(np.argmax(values))
            q = points[j]
            if values[j] < alpha * reference or np.any(q < 0) or np.any(q >= upper):
                break
            if tree is not None and tree.query(q)[0] < near:
                break
            if forbidden is not None and forbidden(q[None])[0]:
                break
            grown.append(q)
            step = q - p
            step /= max(np.linalg.norm(step), 1e-9)
            t = 0.8 * t + 0.2 * step
            t /= np.linalg.norm(t)
            p = q
        if grown:
            added += len(grown)
            current = np.vstack([current, np.array(grown)])
        line = current if side == 1 else current[::-1]
    return _samples_inside(line, shape, _STEP), added


def _end_direction(s, at_end, k=12):
    t = s[-1] - s[max(len(s) - 1 - k, 0)] if at_end else s[0] - s[min(k, len(s) - 1)]
    return t / max(np.linalg.norm(t), 1e-9)


def _join(grey, pieces, r, shape):
    """Greedy, nearest first: join piece ends that continue each other across a short gap on the ridge."""
    from scipy.spatial import cKDTree

    pieces = {i: p for i, p in enumerate(pieces) if len(p) >= 4}
    joined = 0
    while len(pieces) > 1:
        ends = []
        for i, p in pieces.items():
            ends.append((i, 0, p[0], _end_direction(p, False)))
            ends.append((i, 1, p[-1], _end_direction(p, True)))
        tree = cKDTree(np.array([e[2] for e in ends]))
        candidates = []
        for a, b in tree.query_pairs(max(_JOIN_GAP, _JOIN_OVERLAP)):
            ia, _, pa, ta = ends[a]
            ib, _, pb, tb = ends[b]
            if ia == ib:
                continue
            gap = pb - pa
            length = float(np.linalg.norm(gap))
            facing = length <= _JOIN_GAP and (
                length <= 1e-6 and ta @ (-tb) >= _JOIN_COS
                or length > 1e-6 and ta @ (gap / length) >= _JOIN_COS and tb @ (-gap / length) >= _JOIN_COS
            )
            overlapping = length <= _JOIN_OVERLAP and ta @ (-tb) >= _JOIN_COS
            # Ends that continue each other with a small sideways offset (the facing test fails at a
            # short gap even for a voxel or two sideways): parallel, within half a radius or so across.
            sideways = float(np.linalg.norm(gap - (gap @ ta) * ta)) if length > 1e-6 else 0.0
            aligned = length <= _JOIN_GAP and ta @ (-tb) >= _JOIN_COS and sideways <= 0.6 * r
            if facing or overlapping or aligned:
                candidates.append((length, a, b))
        candidates.sort()
        used, merged = set(), False
        for length, a, b in candidates:
            ia, ea, pa, _ = ends[a]
            ib, eb, pb, _ = ends[b]
            if ia in used or ib in used:
                continue
            n = max(int(np.ceil(length / _STEP)), 1)
            bridge = pa[None] + (pb - pa)[None] * (np.arange(1, n) / n)[:, None]
            if len(bridge):
                ta_end = _end_direction(pieces[ia], ea == 1)
                t = np.repeat(ta_end[None], len(bridge), 0)  # across the fiber, not across the bridge
                _, dist = _argmax_offset(grey, bridge, t, r)
                if np.mean(dist <= 0.5 * r) < _JOIN_RIDGE:
                    continue
            first = pieces[ia] if ea == 1 else pieces[ia][::-1]
            second = pieces[ib] if eb == 0 else pieces[ib][::-1]
            # Pieces that run on past each other: drop the second's samples up to the one nearest the first's end.
            head = second[: max(int(2 * _JOIN_OVERLAP / _STEP), 1)]
            nearest = int(np.argmin(np.linalg.norm(head - first[-1], axis=1)))
            if nearest > 0 and np.dot(second[min(nearest + 1, len(second) - 1)] - first[-1], first[-1] - first[-2]) > 0:
                second = second[nearest:]
                pb = second[0]
                length = float(np.linalg.norm(pb - pa))
                n = max(int(np.ceil(length / _STEP)), 1)
                bridge = pa[None] + (pb - pa)[None] * (np.arange(1, n) / n)[:, None]
            pieces[ia] = _samples_inside(np.vstack([first, bridge, second]) if len(bridge) else np.vstack([first, second]),
                                         shape, _STEP)
            del pieces[ib]
            used.update((ia, ib))
            joined += 1
            merged = True
        if not merged:
            break
    return list(pieces.values()), joined
