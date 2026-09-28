"""The passes after the last solve that put fibers back together and tidy them, each off by default:
``coarse_join``, ``straight_join``, ``bend_finish``, ``clean_finish``, ``merge_short`` and ``through_block``
(see ``FitSettings``). ``coarse_join`` acts on the larger types; ``straight_join`` and ``bend_finish`` act on
the smallest type by default (``straight_join_types``, ``bend_finish_types``), the last three on it only. The
void-relative grey floors read ``fitter.peak_void``: the void grey of the intensity ranges or grey profiles
(zero on a plain grey scan).

Coarse join (``FitSettings.coarse_join``): put a larger type's fiber back together at the end of the fit.

A coarse fiber often ends the fit in pieces the cleanup no longer sees: two pieces facing each other across a
short gap the scan fills, or two pieces that run past each other side by side, each on one rim of the same
dim-cored fiber (their axes about two short radii apart, which the duplicate tests read as two touching
fibers). Pairs of ends are joined best first:

- across a gap: the ends antiparallel within ``_ANGLE`` degrees, at most ``_REACH`` radii apart, each end
  pointing along the bridge within ``_ANGLE`` degrees (or the two lines at most ``_OFFSET`` radii apart
  across the fiber), and the fit image along the bridge at least ``_SUPPORT`` on average;
- side by side: the ends antiparallel within ``_ANGLE`` degrees, running past each other by at most
  ``_REACH`` radii (or stopping at most one radius short of each other), at most ``_SIDE`` radii apart across
  the fiber, and no bright band between them: across the overlap, the grey between the two axes stays below
  the brighter axis plus ``_BAND`` of the smallest type's grey range. Two touching fibers show their rims
  between their axes; two fits on one fiber show its dim core. The overlap is replaced by the midline of the
  two fits. The band is read on the ``coarse_axis_grey_max`` grey (``fitter.axis_grey``); without it there
  are no side-by-side joins.
"""

from __future__ import annotations

import numpy as np

from ._geometry import sample_image

_ANGLE = 35.0  # degrees
_REACH = 4.0  # short radii
_OFFSET = 0.6  # short radii
_SIDE = 2.2  # short radii
_SUPPORT = 0.5
_BAND = 0.1
_END_BASE = 8.0  # voxels of a piece's end that set its direction


def coarse_join(fitter, lines, radii, types):
    """``(lines, radii, types, info)`` after the joins (see the module notes)."""
    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in lines]
    info = {"gap_joins": 0, "side_joins": 0}
    smallest = float(fitter.radius.min())
    band = None
    if fitter.axis_grey is not None:
        low, high = fitter.axis_range
        band = _BAND * (high - low)
    for kind in range(len(fitter.radius)):
        if float(fitter.radius[kind]) <= smallest:
            continue
        r = float(fitter.radius[kind])
        while True:
            ids = [i for i in range(len(lines)) if types[i] == kind and len(lines[i]) >= 4]
            best = None
            for a_pos, a in enumerate(ids):
                for b in ids[a_pos + 1:]:
                    for ea in (0, 1):
                        for eb in (0, 1):
                            found = _candidate(fitter, lines[a], lines[b], ea, eb, r, band)
                            if found is not None and (best is None or found[0] < best[0]):
                                best = (*found, a, b)
            if best is None:
                break
            _, how, joined, a, b = best
            info["gap_joins" if how == "gap" else "side_joins"] += 1
            keep = [i for i in range(len(lines)) if i not in (a, b)]
            radius = 0.5 * (radii[a] + radii[b])
            lines = [lines[i] for i in keep] + [joined]
            radii = np.concatenate([radii[keep], [radius]])
            types = np.concatenate([types[keep], [kind]])
    return lines, radii, types, info


def _candidate(fitter, line_a, line_b, ea, eb, r, band):
    """``(cost, "gap" | "side", joined line)`` for joining end ``ea`` of ``line_a`` to end ``eb`` of ``line_b``, or None."""
    A = line_a if ea == 1 else line_a[::-1]  # A ends at the joined end
    B = line_b if eb == 0 else line_b[::-1]  # B starts at it
    pa, pb = A[-1], B[0]
    da = _unit(A[-1] - A[-4])
    db = _unit(B[0] - B[3])
    cos = np.cos(np.radians(_ANGLE))
    if da @ db > -cos:
        return None
    w = pb - pa
    along = float(w @ da)
    lateral = float(np.linalg.norm(w - along * da))
    if along > 0:
        gap = float(np.linalg.norm(w))
        if gap > _REACH * r:
            return None
        bridge = w / max(gap, 1e-9)
        if lateral > _OFFSET * r and (bridge @ da < cos or -(bridge @ db) < cos):
            # Ends side by side just short of each other: the rims of one fiber, joined at their midpoint.
            if along > r or lateral > _SIDE * r or band is None:
                return None
            if not _one_fiber(fitter.axis_grey, np.stack([pa, A[-2]]), np.stack([pb, B[1]]), band):
                return None
            return gap + lateral, "side", np.concatenate([A[:-1], [0.5 * (pa + pb)], B[1:]])
        points = pa + np.linspace(0.0, 1.0, max(int(np.ceil(gap)) + 1, 2))[:, None] * w
        if float(np.mean(sample_image(fitter.evidence, points))) < _SUPPORT:
            return None
        return gap, "gap", np.concatenate([A, B])
    if -along > _REACH * r or lateral > _SIDE * r or band is None:
        return None
    a_proj = (A - pb) @ da
    b_proj = (B - pa) @ da
    i = int(np.flatnonzero(a_proj <= 0)[-1]) if np.any(a_proj <= 0) else -1
    j = int(np.flatnonzero(b_proj >= 0)[0]) if np.any(b_proj >= 0) else len(B)
    a_keep, a_over = A[: i + 1], A[i + 1:]
    b_over, b_keep = B[:j], B[j:]
    if len(a_keep) < 2 or len(b_keep) < 2 or not len(a_over) or not len(b_over):
        return None
    dense_a, dense_b = _dense(A), _dense(B)
    near_b = dense_b[np.argmin(np.linalg.norm(a_over[:, None, :] - dense_b[None, :, :], axis=2), axis=1)]
    near_a = dense_a[np.argmin(np.linalg.norm(b_over[:, None, :] - dense_a[None, :, :], axis=2), axis=1)]
    if np.max(np.linalg.norm(a_over - near_b, axis=1)) > _SIDE * r * 1.1:
        return None
    if not _one_fiber(fitter.axis_grey, np.concatenate([a_over, b_over]), np.concatenate([near_b, near_a]), band):
        return None
    mids = np.concatenate([0.5 * (a_over + near_b), 0.5 * (b_over + near_a)])
    mids = mids[np.argsort((mids - pa) @ da)]
    return float(-along + lateral), "side", np.concatenate([a_keep, mids, b_keep])


def _one_fiber(grey, p, q, band) -> bool:
    """Whether the grey between each pair of points ``p``, ``q`` stays below the brighter end plus ``band`` (median over pairs)."""
    excess = []
    for x, y in zip(p, q):
        d = float(np.linalg.norm(y - x))
        if d < 4.0:
            excess.append(-np.inf)
            continue
        s = np.arange(1.5, d - 1.5 + 1e-9, 0.5)
        between = sample_image(grey, x + s[:, None] * (y - x) / d)
        ends = sample_image(grey, np.stack([x, y]))
        excess.append(float(between.max() - ends.max()))
    return float(np.median(excess)) <= band


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _dense(line, step: float = 0.5):
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return line
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, line[:, j]) for j in range(3)], axis=1)


def straight_join(fitter, lines, radii, types, kinds, max_gap, max_angle, max_offset, max_overlap, support):
    """``(lines, radii, types, joins)``: pieces of the types ``kinds`` that continue each other in a straight line
    joined (``FitSettings.straight_join``).

    A pair of ends qualifies when the two end directions (over the last ``_END_BASE`` voxels) are antiparallel
    within ``max_angle`` degrees and each piece's extended axis passes within ``max_offset`` radii of the other's
    end; then either the ends face each other across a gap of at most ``max_gap`` voxels whose straight bridge
    reads at least ``support`` in the fit image on average, or the pieces run past each other along the same
    axis by at most ``max_overlap`` voxels (the second piece's overlapping start is dropped). Candidates are
    taken shortest first, each end joins at most once, and no join closes a loop.
    """
    from scipy.spatial import cKDTree

    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in lines]
    cos = np.cos(np.radians(max_angle))
    ids = [i for i in range(len(lines)) if int(types[i]) in kinds and len(lines[i]) >= 2]
    ends, owner = [], []
    for i in ids:
        line = lines[i]
        for e in (0, 1):
            p = line[0] if e == 0 else line[-1]
            ends.append((p, _end_direction(line, e)))
            owner.append((i, e))
    if len(ends) < 2:
        return lines, radii, types, 0
    points = np.array([p for p, _ in ends])
    tree = cKDTree(points)
    candidates = []
    for a, b in tree.query_pairs(max(max_gap, max_overlap) + 1e-9):
        (ia, _), (ib, _) = owner[a], owner[b]
        if ia == ib or types[ia] != types[ib]:
            continue
        r = float(fitter.radius[int(types[ia])])
        pa, da = ends[a]
        pb, db = ends[b]
        if float(da @ db) > -cos:
            continue
        w = pb - pa
        along = float(w @ da)
        off_a = float(np.linalg.norm(w - along * da))
        along_b = float(-w @ db)
        off_b = float(np.linalg.norm(-w - along_b * db))
        if max(off_a, off_b) > max_offset * r:
            continue
        if along > 0:
            gap = float(np.linalg.norm(w))
            if gap > max_gap:
                continue
            if support is not None and gap > 1.0:
                bridge = pa + np.linspace(0.0, 1.0, max(int(np.ceil(gap)) + 1, 2))[:, None] * w
                if float(np.mean(sample_image(fitter.evidence, bridge))) < support:
                    continue
            candidates.append((gap + max(off_a, off_b), a, b, False))
        elif -along <= max_overlap:
            candidates.append((max(off_a, off_b), a, b, True))
    candidates.sort(key=lambda c: c[0])
    partner: dict[int, tuple[int, bool]] = {}
    group = {i: i for i in ids}

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    joins = 0
    for _, a, b, overlap in candidates:
        if a in partner or b in partner:
            continue
        ra, rb = root(owner[a][0]), root(owner[b][0])
        if ra == rb:
            continue
        partner[a] = (b, overlap)
        partner[b] = (a, overlap)
        group[ra] = rb
        joins += 1
    if not joins:
        return lines, radii, types, 0
    end_index = {o: k for k, o in enumerate(owner)}
    used = set()
    out_lines, out_radii, out_types = [], [], []
    for i in range(len(lines)):
        if i not in group:
            out_lines.append(lines[i]), out_radii.append(radii[i]), out_types.append(types[i])
            continue
        if i in used:
            continue
        # walk to one end of the chain: the end of piece i with no partner, if any
        start, e = i, 0
        seen = {i}
        k = end_index[(start, e)]
        while k in partner:
            nxt = owner[partner[k][0]]
            if nxt[0] in seen:
                break
            seen.add(nxt[0])
            start, e = nxt[0], 1 - nxt[1]
            k = end_index[(start, e)]
        # now walk forward from (start, e): piece oriented so that end e is its beginning
        chain, members = [], []
        piece, begin = start, e
        while True:
            used.add(piece)
            members.append(piece)
            line = lines[piece] if begin == 0 else lines[piece][::-1]
            prev = next((c for c, _ in reversed(chain) if len(c) >= 2), None)
            if prev is not None and chain[-1][1]:  # overlap: drop this piece's start behind the previous end
                d = _end_direction(prev, 1)
                keep = (line - prev[-1]) @ d > 0
                first = int(np.argmax(keep)) if keep.any() else len(line)
                line = line[first:]
            k_out = end_index[(piece, 1 - begin)]
            nxt = partner.get(k_out)
            chain.append((line, bool(nxt[1]) if nxt else False))
            if nxt is None or owner[nxt[0]][0] in used:
                break
            piece, begin = owner[nxt[0]]
        joined = np.concatenate([c for c, _ in chain if len(c)])
        if len(joined) < 2:
            continue
        out_lines.append(joined)
        out_radii.append(float(np.median([radii[m] for m in members])))
        out_types.append(types[i])
    return out_lines, np.asarray(out_radii, dtype=np.float64), np.asarray(out_types, dtype=int), joins


def _end_direction(line, e, base=_END_BASE):
    """Unit direction pointing out of end ``e`` (0 = first node) of ``line``, over its last ``base`` voxels."""
    line = line if e == 1 else line[::-1]
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    back = np.cumsum(seg[::-1])
    k = int(np.searchsorted(back, base)) + 1
    k = min(max(k, 1), len(line) - 1)
    return _unit(line[-1] - line[-1 - k])


def bend_finish(fitter, lines, radii, types, kinds, grey, grey_min, iterations=400, bends=None):
    """``(lines, radii, types, info)``: the fits of the types ``kinds`` smoothed toward their type's bend limit
    (``FitSettings.bend_finish``; ``bends``: per-type limits in voxels in place of ``fitter.bend``).

    Each fit is resampled at the node spacing; wherever a node sits more than the sagitta of the limit,
    ``s**2 / (2 R)`` with ``s`` the mean segment length before smoothing, off the midpoint of its neighbours, it
    is pulled toward that midpoint (half the excess per sweep, up to ``iterations`` sweeps). The steps after the
    last solve (ridge finish, hole births, joins) do not keep to the limit, so this is where it is restored. Two
    cases stop short of it: smoothing shortens a rough trace's segments, so the sagitta set before it allows
    bends tighter than ``R``, and a long fit under a very large ``R`` needs more than ``iterations`` sweeps.
    With ``grey_min``, runs of at least 3 voxels where the smoothed fit reads below that share of the way from
    void (``fitter.peak_void``) to the median grey along the fits (``grey``) are then cut: a trace that wandered
    along noise has no fiber under its smoothed path. Pieces that cut leaves shorter than the type's minimum
    length are dropped.
    """
    from ._geometry import polyline_length, resample

    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    spacing = float(fitter.spacing)
    shape = np.array(grey.shape[::-1], dtype=np.float64) if grey is not None else None
    reference = None
    void = float(getattr(fitter, "peak_void", 0.0) or 0.0)
    if grey is not None and grey_min is not None:
        own = [resample(np.asarray(l, dtype=np.float64), 1.0) for l, k in zip(lines, types)
               if int(k) in kinds and len(l) >= 2]
        own = [q for q in own if len(q)]
        if own:
            reference = float(np.median(sample_image(grey, np.concatenate(own))))
    out_lines, out_radii, out_types = [], [], []
    moved_total, cut_voxels, dropped = 0.0, 0.0, 0
    for line, radius, kind in zip(lines, radii, types):
        line = np.asarray(line, dtype=np.float64)
        if int(kind) not in kinds or len(line) < 3:
            out_lines.append(line), out_radii.append(radius), out_types.append(kind)
            continue
        bend = float(fitter.bend[int(kind)] if bends is None else bends[int(kind)])
        p = resample(line, spacing)
        if len(p) < 3:
            out_lines.append(line), out_radii.append(radius), out_types.append(kind)
            continue
        start = p.copy()
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = float(np.mean(seg)) if len(seg) else spacing
        sagitta = s * s / (2.0 * bend)
        for _ in range(iterations):
            dev = p[1:-1] - 0.5 * (p[:-2] + p[2:])
            size = np.linalg.norm(dev, axis=1)
            over = size > sagitta * 1.001
            if not over.any():
                break
            shrink = np.where(over, 0.5 * (1.0 - sagitta / np.maximum(size, 1e-12)), 0.0)
            p[1:-1] -= dev * shrink[:, None]
        moved_total += float(np.sum(np.linalg.norm(p - start, axis=1)))
        pieces, was_cut = [p], False
        if reference is not None:
            q = resample(p, 0.5)
            inside = np.all((q >= 0) & (q <= shape - 1), axis=1)
            dim = (sample_image(grey, q) - void < grey_min * (reference - void)) & inside
            keep = np.ones(len(q), dtype=bool)
            edges = np.flatnonzero(np.diff(np.r_[0, dim.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if (b - a) * 0.5 >= 3.0:
                    keep[a:b] = False
                    cut_voxels += (b - a) * 0.5
                    was_cut = True
            if not was_cut:
                keep[:] = True
            pieces = []
            edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                pieces.append(q[a:b])
        min_length = float(fitter.min_length[int(kind)]) if was_cut else 0.0
        for piece in pieces:
            if len(piece) >= 2 and polyline_length(piece) >= min_length:
                out_lines.append(resample(piece, spacing) if len(piece) > 2 else piece)
                out_radii.append(radius)
                out_types.append(kind)
            else:
                dropped += 1
    info = {"moved_voxels_mean": round(moved_total / max(sum(len(l) for l in out_lines), 1), 3),
            "cut_voxels": round(float(cut_voxels), 1), "pieces_dropped": dropped}
    return out_lines, np.asarray(out_radii, dtype=np.float64), np.asarray(out_types, dtype=int), info


def clean_finish(fitter, lines, radii, types, kinds, grey, reach_radii, min_length):
    """``(lines, radii, types, info)``: fits of the types ``kinds`` with doubled stretches and short pieces removed
    (``FitSettings.clean_finish``).

    Shortest fit first, every sample within ``reach_radii`` radii of another live fit of the same kinds is a
    doubled stretch when the grey (``grey``) at the midpoint between the two is at least the dimmer of the two
    axes' grey less one unit: no darker gap between them, so they sit on one fiber (two touching fibers show a
    dip at their contact). Doubled stretches are cut out of the shorter fit; then, with ``min_length``, the
    pieces of the types ``kinds`` shorter than that many voxels are dropped.
    """
    from scipy.spatial import cKDTree

    from ._geometry import polyline_length, resample

    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    ids = [i for i in range(len(lines)) if int(types[i]) in kinds and len(lines[i]) >= 2]
    dense = {i: resample(np.asarray(lines[i], dtype=np.float64), 1.0) for i in ids}
    order = sorted(ids, key=lambda i: polyline_length(np.asarray(lines[i], dtype=np.float64)))
    alive = set(ids)
    pieces_of: dict[int, list[np.ndarray]] = {}
    doubled = 0
    for i in order:
        others = [j for j in alive if j != i and len(dense[j])]
        q = dense[i]
        if not others or not len(q):
            continue
        reach = reach_radii * float(fitter.radius[int(types[i])])
        points = np.concatenate([dense[j] for j in others])
        d, k = cKDTree(points).query(q)
        near = d <= reach
        if not near.any():
            continue
        mid = 0.5 * (q + points[k])
        one = sample_image(grey, mid) >= np.minimum(sample_image(grey, q), sample_image(grey, points[k])) - 1.0
        dup = near & one
        if not dup.any():
            continue
        doubled += int(dup.sum())
        edges = np.flatnonzero(np.diff(np.r_[0, (~dup).astype(int), 0]))
        pieces = [q[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= 2]
        pieces_of[i] = pieces
        dense[i] = np.concatenate(pieces) if pieces else q[:0]
        if not pieces:
            alive.discard(i)
    out_lines, out_radii, out_types, short = [], [], [], 0
    for i, (line, radius, kind) in enumerate(zip(lines, radii, types)):
        pieces = pieces_of.get(i, [np.asarray(line, dtype=np.float64)])
        shortest = 0.0 if min_length is None else float(min_length)
        for piece in pieces:
            if int(kind) in kinds and (len(piece) < 2 or polyline_length(piece) < shortest):
                short += 1
                continue
            out_lines.append(piece if i not in pieces_of else resample(piece, float(fitter.spacing)))
            out_radii.append(radius)
            out_types.append(kind)
    info = {"doubled_voxels": doubled, "short_dropped": short}
    return out_lines, np.asarray(out_radii, dtype=np.float64), np.asarray(out_types, dtype=int), info


def merge_short(fitter, lines, radii, types, kinds, short_length, max_gap, max_offset_radii):
    """``(lines, radii, types, merged)``: short pieces of the types ``kinds`` (under ``short_length`` voxels) added to
    the end of a longer fit they continue (``FitSettings.merge_short``).

    A short piece's own direction says little, so only the long fit's is used: the piece qualifies when its
    nearest point lies ahead of the long fit's end (along the end direction) by at most ``max_gap`` voxels and
    all its samples lie within ``max_offset_radii`` radii of the end's extended axis. Each short piece goes to the
    nearest qualifying end; a long end takes pieces in order of distance, each extending it.
    """
    from ._geometry import polyline_length, resample

    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in lines]
    length = np.array([polyline_length(line) if len(line) >= 2 else 0.0 for line in lines])
    shorts = [i for i in range(len(lines)) if int(types[i]) in kinds and len(lines[i]) >= 2 and length[i] < short_length]
    longs = [i for i in range(len(lines)) if int(types[i]) in kinds and len(lines[i]) >= 2 and length[i] >= short_length]
    if not shorts or not longs:
        return lines, radii, types, 0
    ends = []
    for i in longs:
        for e in (0, 1):
            p = lines[i][0] if e == 0 else lines[i][-1]
            ends.append((i, e, p, _end_direction(lines[i], e)))
    claims = []  # (distance, short, long, end)
    for s in shorts:
        q = resample(lines[s], 0.5)
        r = float(fitter.radius[int(types[s])])
        best = None
        for i, e, p, d in ends:
            if types[i] != types[s]:
                continue
            w = q - p
            along = w @ d
            lateral = np.linalg.norm(w - along[:, None] * d, axis=1)
            if along.min() < -r or along.min() > max_gap or lateral.max() > max_offset_radii * r:
                continue
            dist = float(max(along.min(), 0.0))
            if best is None or dist < best[0]:
                best = (dist, s, i, e)
        if best is not None:
            claims.append(best)
    claims.sort()
    extended = {}
    used = set()
    for dist, s, i, e in claims:
        if s in used:
            continue
        used.add(s)
        base = extended.get(i, lines[i])
        piece = lines[s]
        if e == 1:
            # orient the piece to run away from the end
            if np.linalg.norm(piece[0] - base[-1]) > np.linalg.norm(piece[-1] - base[-1]):
                piece = piece[::-1]
            base = np.concatenate([base, piece])
        else:
            if np.linalg.norm(piece[-1] - base[0]) > np.linalg.norm(piece[0] - base[0]):
                piece = piece[::-1]
            base = np.concatenate([piece, base])
        extended[i] = base
    out_lines, out_radii, out_types = [], [], []
    for i, (line, radius, kind) in enumerate(zip(lines, radii, types)):
        if i in used:
            continue
        out_lines.append(extended.get(i, line)), out_radii.append(radius), out_types.append(kind)
    return out_lines, np.asarray(out_radii, dtype=np.float64), np.asarray(out_types, dtype=int), len(used)


def extend_through_ridge(fitter, lines, radii, types, kinds, grey, void, reference, support, dim_run, bend,
                         margin=4.0, clearance=0.9, cross_angle=30.0, reach_radii=0.3):
    """``(lines, info)``: interior ends of the types ``kinds`` walked along their own grey ridge toward the block's
    faces (for ``through_block``).

    Each step moves one voxel along the current direction; the brightest point of ``grey`` on a disc
    ``reach_radii`` radii wide across the step aims the next direction, which may turn by at most one voxel over
    ``bend`` voxels (the bend limit's curvature) per step. A walk starts only from an end more than ``margin``
    voxels inside the block and stops at that margin, at a fit running within ``cross_angle`` degrees closer
    than ``clearance`` of the two radii' sum (steeper crossings are passed: fibers cross over and under each
    other), or after ``dim_run`` dim voxels (below ``support`` of the way from ``void`` to ``reference``); the
    end moves to the last voxel that read bright enough. What earlier walks grew blocks a walk too, so two ends
    walking toward each other on one fiber meet instead of both filling the gap. A walk still going after
    twice the block's diagonal is circling a closed ridge and leaves its end where it was; an end with no
    direction (coincident end nodes) is not walked.
    """
    from scipy.spatial import cKDTree

    from ._geometry import resample

    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in lines]
    upper = np.array(grey.shape[::-1], dtype=np.float64) - 1.0
    dense = [resample(line, 1.0) if len(line) >= 2 else line for line in lines]
    owner = np.concatenate([np.full(len(q), i) for i, q in enumerate(dense)]) if dense else np.zeros(0, int)
    tree = cKDTree(np.concatenate(dense)) if dense else None
    level = void + support * (reference - void)
    turn = 1.0 / max(float(bend), 1.0)
    cos_cross = np.cos(np.radians(cross_angle))
    grown = reached = 0
    stops = {"face": 0, "blocked": 0, "dim": 0, "loop": 0}
    max_steps = int(2.0 * np.linalg.norm(upper)) + 1  # no walk to a face is longer

    def inside(p):
        return float(min(np.min(p), np.min(upper - p)))

    for i in range(len(lines)):
        if int(types[i]) not in kinds or len(lines[i]) < 2:
            continue
        r = float(fitter.radius[int(types[i])])
        steps = np.arange(-reach_radii * r, reach_radii * r + 1e-9, 0.25)
        du, dv = np.meshgrid(steps, steps)
        keep = du**2 + dv**2 <= (reach_radii * r) ** 2
        du, dv = du[keep], dv[keep]
        for e in (0, 1):
            line = lines[i]
            p = (line[0] if e == 0 else line[-1]).copy()
            if inside(p) <= margin:
                continue
            t = _end_direction(line, e, 20.0)
            if float(np.linalg.norm(t)) < 0.5:  # coincident end nodes: no direction to walk
                continue
            last_good, dim, path = 0, 0, []
            for _ in range(max_steps):
                q0 = p + t
                if inside(q0) <= margin:
                    stops["face"] += 1
                    break
                helper = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                e1 = _unit(np.cross(t, helper))
                e2 = np.cross(t, e1)
                disc = q0[None, :] + du[:, None] * e1 + dv[:, None] * e2
                best = disc[int(np.argmax(sample_image(grey, disc)))]
                want = _unit(best - p)
                angle = float(np.arccos(np.clip(want @ t, -1.0, 1.0)))
                if angle > turn:
                    axis = want - (want @ t) * t
                    if np.linalg.norm(axis) > 1e-9:
                        axis = _unit(axis)
                        t = _unit(np.cos(turn) * t + np.sin(turn) * axis)
                else:
                    t = want
                q = p + t
                blocked = False
                for h in tree.query_ball_point(q, clearance * (r + float(radii.max()))):
                    j = owner[h]
                    if j == i or np.linalg.norm(tree.data[h] - q) >= clearance * (r + float(radii[j])):
                        continue
                    a = h - 1 if h > 0 and owner[h - 1] == j else h
                    b = h + 1 if h + 1 < len(owner) and owner[h + 1] == j else h
                    tj = tree.data[b] - tree.data[a]
                    if np.linalg.norm(tj) > 0 and abs(float(_unit(tj) @ t)) < cos_cross:
                        continue
                    blocked = True
                    break
                if blocked:
                    stops["blocked"] += 1
                    break
                path.append(q.copy())
                p = q
                if float(sample_image(grey, q[None])[0]) >= level:
                    last_good, dim = len(path), 0
                else:
                    dim += 1
                    if dim >= dim_run:
                        stops["dim"] += 1
                        break
            else:  # circling a closed ridge: the end stays where it was
                stops["loop"] += 1
                last_good = 0
            if last_good:
                add = np.array(path[:last_good])
                lines[i] = np.concatenate([add[::-1], line]) if e == 0 else np.concatenate([line, add])
                grown += last_good
                # later walks stop at this growth too
                dense[i] = resample(lines[i], 1.0)
                owner = np.concatenate([np.full(len(d), k) for k, d in enumerate(dense)])
                tree = cKDTree(np.concatenate(dense))
                if inside(add[-1]) <= margin + 1.0:
                    reached += 1
    return lines, {"grown_voxels": grown, "ends_reached_face": reached, **{f"stop_{k}": v for k, v in stops.items()}}


_THROUGH_LEVELS = ((0.3, 10, 120.0), (0.3, 25, 200.0), (0.25, 40, 240.0))  # (grey support, dim run, join gap) per round
_THROUGH_CLEARANCE = 0.45  # of the two radii's sum: a walk stops closer than this to a fit running its way
_THROUGH_CROSS = 10.0  # degrees: a fit running at a steeper angle is a crossing and is passed (at a shallower
# pass angle a walk slides along a nearly parallel neighbour and doubles it)
_THROUGH_JOIN_ANGLE = 12.0  # degrees
_THROUGH_JOIN_OFFSET = 1.5  # radii
_THROUGH_OVERLAP = 20.0  # voxels
_GHOST_LEVEL = 0.3  # of the way from void to the fits' median grey
_GHOST_RUN = 5.0  # voxels
_THROUGH_GIVE = 2.0  # diameters: an end this close to a face counts as at the face (``through_share``); a fiber
# leaving the block still ends a little inside it, where its last stretch blurs into the edge


def through_block(fitter, lines, radii, types, kinds, grey, sharp, void, bend):
    """``(lines, radii, types, info)`` for ``FitSettings.through_block``: fits of the types ``kinds`` pushed to run
    through the block.

    Over the rounds of ``_THROUGH_LEVELS``, loosening the dim stretches a walk may cross and the joins' reach: join
    ends that continue each other (``straight_join``, no bridge check: the walk checks the grey), walk every end
    still inside the block along its ridge on ``sharp`` (``extend_through_ridge``, bend limit ``bend`` voxels),
    and join again. Then cut runs of at least ``_GHOST_RUN`` voxels where ``grey`` reads below ``_GHOST_LEVEL`` of
    the way from ``void`` to the fits' median grey (a walk or join over empty space), and drop pieces shorter than
    the type's minimum length.
    """
    from ._geometry import polyline_length, resample

    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in lines]
    own = [resample(l, 1.0) for l, k in zip(lines, types) if int(k) in kinds and len(l) >= 2]
    if not own:
        return lines, radii, types, {}
    reference = float(np.median(sample_image(grey, np.concatenate(own))))
    spacing = float(fitter.spacing)
    joins = grown = 0
    for support, dim_run, gap in _THROUGH_LEVELS:
        for walk in (True, False):
            lines, radii, types, j = straight_join(
                fitter, lines, radii, types, kinds, gap, _THROUGH_JOIN_ANGLE, _THROUGH_JOIN_OFFSET, _THROUGH_OVERLAP, None
            )
            joins += j
            lines = [resample(l, spacing) if len(l) >= 2 else l for l in lines]
            if walk:
                lines, info = extend_through_ridge(
                    fitter, lines, radii, types, kinds, sharp, void, reference, support, dim_run, bend,
                    clearance=_THROUGH_CLEARANCE, cross_angle=_THROUGH_CROSS,
                )
                grown += info["grown_voxels"]
    out_lines, out_radii, out_types, ghost = [], [], [], 0
    for line, radius, kind in zip(lines, radii, types):
        if int(kind) not in kinds or len(line) < 2:
            out_lines.append(line), out_radii.append(radius), out_types.append(kind)
            continue
        q = resample(line, 1.0)
        low = sample_image(grey, q) - void < _GHOST_LEVEL * (reference - void)
        edges = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
        keep = np.ones(len(q), dtype=bool)
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= _GHOST_RUN:
                keep[a:b] = False
                ghost += int(b - a)
        if keep.all():
            out_lines.append(line), out_radii.append(radius), out_types.append(kind)
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if polyline_length(q[a:b]) >= float(fitter.min_length[int(kind)]):
                out_lines.append(resample(q[a:b], spacing)), out_radii.append(radius), out_types.append(kind)
    upper = np.array(grey.shape[::-1], dtype=np.float64) - 1.0
    through = total = 0
    for line, kind in zip(out_lines, out_types):
        if int(kind) in kinds and len(line) >= 2:
            total += 1
            give = _THROUGH_GIVE * 2.0 * float(fitter.radius[int(kind)])
            through += all(min(np.min(p), np.min(upper - p)) <= give for p in (line[0], line[-1]))
    info = {"joins": joins, "grown_voxels": grown, "ghost_voxels_cut": ghost,
            "through_share": round(through / max(total, 1), 3)}
    return out_lines, np.asarray(out_radii, dtype=np.float64), np.asarray(out_types, dtype=int), info
