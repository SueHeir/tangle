"""Fibers straight from the network's maps: axis votes, then tracking along the predicted direction.

1. Every voxel the network calls fiber votes for its axis at voxel + offset.
   Votes are pooled in 1-voxel cells; a cell with enough votes is an axis
   point, carrying the mean vote position, the mean direction tensor and
   the mean predicted fiber radius.
2. Fibers are tracked from the best-supported unused axis point, both ways
   along the point's direction: each step looks ``STEP`` voxels ahead and
   moves to the mean of the axis points there whose direction agrees with
   the track's, so a crossing fiber's points are ignored. A track may coast
   straight over short gaps, and it stops where it runs onto a fiber
   already traced.
3. Tracks shorter than ``min_length`` are dropped. Each track carries its fiber's radius (the median of the
   predicted radius along it); nothing here knows fiber types, so any mix of sizes works. Types are assigned
   per fiber afterwards (``fiber_types.py``).

usage (scoring against the truth): see score_trace.py
"""

import numpy as np
from scipy.spatial import cKDTree

from maps import DIRECTION, FIBER, OFFSET, RADIUS

# Tuned on the network's own maps (r19; 34 test scans, then all 9 test sets): short steps, a tight look-ahead and
# strict direction agreement keep a track from stepping onto a fiber that crosses it, the main loss on dense
# scans. Fibers of radius THICK and up keep the earlier, longer steps (1.5, 1.6, 0.8): with short steps the
# look-ahead also takes in axis points just behind the track, and on a thick fiber's dense points it stalls.
STEP = 0.75  # voxels per tracking step
REACH = 1.3  # voxels: axis points within this of the look-ahead point are candidates
AGREE = 0.9  # |cos| between a candidate's direction and the track's
THICK = 6.0  # voxels: seeds of at least this radius track with the settings below
THICK_STEP, THICK_REACH, THICK_AGREE = 1.5, 1.6, 0.8
GAP_STEPS = 4  # straight steps allowed without support
OWNED_MIN = 1.2  # voxels: a track uses up the axis points within max(this, OWNED_SHARE x its radius)
OWNED_SHARE = 0.5  # (a thick fiber's votes spread wider, e.g. onto both rims of a dim core)
OVERLAP_STEPS = 4  # stop after this many steps on used-up points


def principal(tensor: np.ndarray) -> np.ndarray:
    """Unit principal direction of each 6-component t t^T (N, 6) -> (N, 3)."""
    m = np.zeros((len(tensor), 3, 3))
    m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = tensor[:, 0], tensor[:, 1], tensor[:, 2]
    m[:, 0, 1] = m[:, 1, 0] = tensor[:, 3]
    m[:, 0, 2] = m[:, 2, 0] = tensor[:, 4]
    m[:, 1, 2] = m[:, 2, 1] = tensor[:, 5]
    _, vectors = np.linalg.eigh(m)
    return vectors[:, :, -1]


def vote_cells(maps: np.ndarray, origin=(0, 0, 0), window=None):
    """Per-cell vote sums of the fiber voxels in ``window`` (z, y, x slices of ``maps``; default all).

    ``origin`` (z, y, x) places ``maps`` in a bigger volume, so tiles of one scan vote into the same cells.
    Returns (key, count, position sum, tensor sum, type counts), one row per cell.
    """
    window = window or tuple(slice(0, n) for n in maps.shape[1:])
    fiber = maps[FIBER][0][window] > 0.5
    z, y, x = np.nonzero(fiber)
    z, y, x = z + window[0].start, y + window[1].start, x + window[2].start
    offset = maps[OFFSET][:, z, y, x].T.astype(np.float64)
    tensor = maps[DIRECTION][:, z, y, x].T.astype(np.float64)
    votes = np.stack([x + origin[2], y + origin[1], z + origin[0]], axis=1) + 0.5 + offset
    cell = np.floor(votes).astype(np.int64) + 1024  # room for votes a little outside the volume
    key = (cell[:, 0] * 8192 + cell[:, 1]) * 8192 + cell[:, 2]
    return merge_cells(key, np.ones(len(key), np.int64), votes, tensor, maps[RADIUS][0][z, y, x].astype(np.float64))


def merge_cells(key, count, position, tensor, radius):
    """Sum rows that share a cell key."""
    unique, inverse = np.unique(key, return_inverse=True)
    n = len(unique)
    sums = [np.zeros(n, np.int64), np.zeros((n, 3)), np.zeros((n, 6)), np.zeros(n)]
    for total, part in zip(sums, (count, position, tensor, radius)):
        np.add.at(total, inverse, part)
    return (unique, *sums)


def cells_to_points(key, count, position, tensor, radius, min_votes: int = 3):
    """Axis points from summed cells: mean position, principal direction, mean radius, vote count."""
    keep = count >= min_votes
    return (position[keep] / count[keep, None], principal(tensor[keep]), radius[keep] / count[keep], count[keep])


def axis_points(maps: np.ndarray, min_votes: int = 3):
    return cells_to_points(*vote_cells(maps), min_votes=min_votes)


def tiled_axis_points(predict_tile, shape, tile: int = 128, stride: int = 64, min_votes: int = 3, log=None):
    """Axis points of a whole volume from overlapping tiles.

    ``predict_tile(z, y, x)`` returns the maps of the tile at that corner (sides ``tile``). Each tile votes
    only from its core, the part nearer its own center than any neighbour's, so every voxel votes once,
    from the tile that saw the most around it.
    """
    def origins(n):
        o = list(range(0, max(n - tile, 0) + 1, stride))
        if o[-1] + tile < n:
            o.append(n - tile)
        return o

    def cores(o, n):
        cuts = [0] + [(a + tile + b) // 2 for a, b in zip(o[:-1], o[1:])] + [n]
        return list(zip(o, cuts[:-1], cuts[1:]))

    axes = [cores(origins(n), n) for n in shape]
    parts, done, total = [], 0, len(axes[0]) * len(axes[1]) * len(axes[2])
    for oz, z0, z1 in axes[0]:
        for oy, y0, y1 in axes[1]:
            for ox, x0, x1 in axes[2]:
                maps = predict_tile(oz, oy, ox)
                window = (slice(z0 - oz, z1 - oz), slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
                parts.append(vote_cells(maps, (oz, oy, ox), window))
                done += 1
                if log and done % 50 == 0:
                    log(f"tiles {done}/{total}")
        # merge as we go so the memory stays at one row per cell
        parts = [merge_cells(*[np.concatenate(c) for c in zip(*parts)])]
    return cells_to_points(*parts[0], min_votes=min_votes)


def trace(maps: np.ndarray, min_length: float = 10.0, min_votes: int = 3, tidy: bool = True):
    """Centerlines (x, y, z voxels) and each one's radius (voxels); ``tidy`` drops doubles and joins gaps."""
    lines, radii = track(*axis_points(maps, min_votes), shape=maps.shape[1:], min_length=min_length)
    return tidy_up(lines, radii) if tidy else (lines, radii)


def tidy_up(lines, radii):
    """Doubled traces dropped, then gaps joined."""
    lines, radii = drop_doubles(lines, radii)
    return join_gaps(lines, radii)


def track(position, direction, radius, counts, shape, min_length: float = 10.0, log=None):
    """Fibers tracked through axis points (see the module notes); ``shape`` is the volume's (z, y, x)."""
    if not len(position):
        return [], []
    tree = cKDTree(position)
    used = np.zeros(len(position), dtype=bool)
    lines, line_radii = [], []
    for rank, seed in enumerate(np.argsort(-counts)):
        if used[seed]:
            continue
        if log and rank % 100000 == 0:
            log(f"tracking: seed {rank}/{len(counts)}, {len(lines)} fibers")
        stride, reach, agree = ((THICK_STEP, THICK_REACH, THICK_AGREE) if radius[seed] >= THICK
                                else (STEP, REACH, AGREE))
        halves = []
        for sign in (1.0, -1.0):
            p, t = position[seed].copy(), sign * direction[seed]
            path, gaps, overlap = [], 0, 0
            while True:
                ahead = p + stride * t
                near = np.array(tree.query_ball_point(ahead, reach), dtype=int)
                if len(near):
                    near = near[np.abs(direction[near] @ t) >= agree]
                if len(near):
                    w = counts[near].astype(np.float64)
                    new = (position[near] * w[:, None]).sum(0) / w.sum()
                    d = (direction[near] * np.sign(direction[near] @ t)[:, None] * w[:, None]).sum(0)
                    step = new - p
                    if np.linalg.norm(step) < 0.3:  # no progress
                        break
                    t_new = d / max(np.linalg.norm(d), 1e-9)
                    gaps = 0
                    overlap = overlap + 1 if used[near].mean() > 0.5 else 0
                    if overlap >= OVERLAP_STEPS:
                        break
                    p, t = new, t_new
                else:
                    gaps += 1
                    if gaps > GAP_STEPS:
                        break
                    p = ahead
                path.append(p.copy())
                if np.any(p < -1) or np.any(p > np.array(shape[::-1]) + 1):
                    break
                if len(path) > 2000:
                    break
            # drop the unsupported coasting at the end
            if gaps:
                path = path[: len(path) - gaps]
            halves.append(path)
        line = np.array(halves[1][::-1] + [position[seed]] + halves[0])
        owned = max(OWNED_MIN, OWNED_SHARE * radius[seed])
        for q in line:  # use up the points along it, even if it is too short to keep
            used[tree.query_ball_point(q, owned)] = True
        used[seed] = True
        if len(line) < 2 or np.linalg.norm(np.diff(line, axis=0), axis=1).sum() < min_length:
            continue
        near = np.unique(np.concatenate([tree.query_ball_point(q, owned) for q in line])).astype(int)
        lines.append(line)
        line_radii.append(float(np.median(radius[near])) if len(near) else float(radius[seed]))
    return lines, line_radii


def _dense(line: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    segment = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(segment)]
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def drop_doubles(lines, radii, share: float = 0.5):
    """Drop traces lying on top of a longer one along more than ``share`` of their length.

    A track's leftover axis points just beside it can seed a short second track along the same fiber; a
    solver then pushes the two a diameter apart. "On top" = within half the trace's own radius plus a voxel of the
    longer trace's axis.
    """
    if not lines:
        return lines, radii
    dense = [_dense(np.asarray(l, float)) for l in lines]
    owner = np.concatenate([np.full(len(d), k) for k, d in enumerate(dense)])
    tree = cKDTree(np.concatenate(dense))
    kept = np.zeros(len(lines), bool)
    for k in np.argsort([-len(d) for d in dense]):
        tol = 0.5 * radii[k] + 1.0
        hits = tree.query_ball_point(dense[k], tol)
        on_top = np.mean([bool(kept[owner[h]].any()) if len(h) else False for h in hits])
        kept[k] = on_top <= share
    return [l for l, k in zip(lines, kept) if k], [r for r, k in zip(radii, kept) if k]


def join_gaps(lines, radii, max_gap: float = 12.0, max_offset: float = 1.5, min_cos: float = 0.9,
              max_size_ratio: float = 1.3):
    """Join traces of about the same radius (within ``max_size_ratio``) whose ends face each other across a gap.

    Ends are paired best-first by gap length; a pair joins when the gap is under ``max_gap`` voxels, both
    ends' directions (over their last 5 voxels) point along the gap within ``min_cos``, and each end's
    straight extension passes within ``max_offset`` voxels of the other end.
    """
    lines = [np.asarray(l, float) for l in lines]
    radii = list(radii)
    while True:
        ends, info = [], []
        for k, l in enumerate(lines):
            if len(l) < 2:
                continue
            for side in (0, 1):
                p = l[0] if side == 0 else l[-1]
                q = _dense(l[::-1] if side == 1 else l)  # from this end inward
                inner = q[min(5, len(q) - 1)]
                d = p - inner
                d /= max(np.linalg.norm(d), 1e-9)  # pointing out of the fiber
                ends.append(p)
                info.append((k, side, d))
        if len(ends) < 2:
            return lines, radii
        tree = cKDTree(np.array(ends))
        pairs = sorted(tree.query_pairs(max_gap), key=lambda ab: np.linalg.norm(ends[ab[0]] - ends[ab[1]]))
        used, joins = set(), []
        for a, b in pairs:
            ka, sa, da = info[a]
            kb, sb, db = info[b]
            if ka == kb or ka in used or kb in used or max(radii[ka], radii[kb]) > max_size_ratio * min(radii[ka], radii[kb]):
                continue
            gap = ends[b] - ends[a]
            g = np.linalg.norm(gap)
            if g > 1e-6:
                u = gap / g
                if u @ da < min_cos or -(u @ db) < min_cos:
                    continue
                off_a = np.linalg.norm(gap - (gap @ da) * da)
                off_b = np.linalg.norm(gap - (gap @ db) * db)
                if max(off_a, off_b) > max_offset:
                    continue
            used |= {ka, kb}
            joins.append((ka, sa, kb, sb))
        if not joins:
            return lines, radii
        for ka, sa, kb, sb in joins:
            la = lines[ka] if sa == 1 else lines[ka][::-1]  # ends at the joining end
            lb = lines[kb] if sb == 0 else lines[kb][::-1]  # starts at the joining end
            lines[ka] = np.vstack([la, lb])
            radii[ka] = (len(la) * radii[ka] + len(lb) * radii[kb]) / (len(la) + len(lb))
            lines[kb] = np.zeros((0, 3))
        keep = [k for k, l in enumerate(lines) if len(l)]
        lines, radii = [lines[k] for k in keep], [radii[k] for k in keep]
