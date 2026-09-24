"""Discrete topology moves: remove duplicates and unsupported fibers, merge
fragments that continue each other, and add fibers where the image is not
yet explained (births are done by re-tracing, see ``_fit``)."""

from __future__ import annotations

import numpy as np

from ._geometry import polyline_length, sample_image


def _others_tree(lines: list[np.ndarray], skip: int):
    from scipy.spatial import cKDTree

    others = [line for j, line in enumerate(lines) if j != skip and len(line)]
    if not others:
        return None
    return cKDTree(np.concatenate(others))


def trim_duplicates(
    centerlines: list[np.ndarray], radii: np.ndarray, *, min_length: float, closeness: float = 0.8
) -> tuple[list[np.ndarray], np.ndarray]:
    """Remove fibers mostly lying inside another fiber, and trim end runs that do.

    Two distinct fibers never have centerlines closer than the sum of their
    radii, so nodes within ``closeness * radius`` of another fiber's nodes are
    re-traces of that fiber.
    """
    lines = [line.copy() for line in centerlines]
    radii = radii.copy()
    order = np.argsort([polyline_length(line) for line in lines])  # shortest first
    removed = np.zeros(len(lines), dtype=bool)
    for index in order:
        live = [line if not removed[j] else np.empty((0, 3)) for j, line in enumerate(lines)]
        tree = _others_tree(live, index)
        if tree is None:
            continue
        line = lines[index]
        distance, _ = tree.query(line)
        covered = distance < closeness * radii[index]
        if covered.mean() > 0.5:
            removed[index] = True
            continue
        start, stop = 0, len(line)
        while start < stop and covered[start]:
            start += 1
        while stop > start and covered[stop - 1]:
            stop -= 1
        trimmed = line[start:stop]
        if len(trimmed) < 2 or polyline_length(trimmed) < min_length:
            removed[index] = True
        else:
            lines[index] = trimmed
    keep = ~removed
    return [line for line, k in zip(lines, keep) if k], radii[keep]


def remove_unsupported(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_length: float,
    min_support: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    keep = []
    for line in centerlines:
        ok = len(line) >= 2 and polyline_length(line) >= min_length
        if ok:
            ok = float(sample_image(image, line).mean()) >= min_support
        keep.append(ok)
    keep = np.array(keep, dtype=bool)
    return [line for line, k in zip(centerlines, keep) if k], radii[keep]


def _end(line: np.ndarray, which: int) -> tuple[np.ndarray, np.ndarray]:
    """End point and outward unit tangent (averaged over a few nodes)."""
    count = min(len(line) - 1, 3)
    if which == 0:
        tip, inner = line[0], line[count]
    else:
        tip, inner = line[-1], line[-1 - count]
    tangent = tip - inner
    return tip, tangent / max(np.linalg.norm(tangent), 1e-12)


def merge_fragments(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    max_gap: float,
    max_angle_degrees: float = 35.0,
    min_bridge_support: float = 0.45,
    min_bend_radius: float | None = None,
    kink_threshold: float = 2.0,
    end_cost: float = 0.0,
    scale: float = 1.0,
    max_prior_gap: float | None = None,
    max_prior_angle_degrees: float = 45.0,
    chains: list[list[tuple[int, int]]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Join pairs of ends that continue each other across a short gap.

    With ``min_bend_radius``, a join that would create a kink the split step
    would cut again is not made.

    With an ``end_cost`` (the fiber-length prior, see ``_ends``), the fixed
    gap, angle and bridge tests are replaced by :func:`_merge_with_prior`,
    which also fills ``chains`` (not supported without the prior).
    """
    if end_cost > 0.0:
        return _merge_with_prior(
            image, centerlines, radii,
            max_gap=max_prior_gap or 4.0 * max_gap, max_angle_degrees=max_prior_angle_degrees,
            min_bend_radius=min_bend_radius, kink_threshold=kink_threshold, end_cost=end_cost, scale=scale,
            chains=chains,
        )
    if chains is not None:
        raise ValueError("chains needs the length prior (end_cost > 0)")
    lines = [line.copy() for line in centerlines]
    radii = radii.copy()
    cos_limit = np.cos(np.radians(max_angle_degrees))
    merges = 0
    rejected: set[tuple[int, int]] = set()
    while True:
        best = None
        ends = [(i, e, *_end(line, e)) for i, line in enumerate(lines) if len(line) >= 2 for e in (0, 1)]
        for a in range(len(ends)):
            i, ei, pi, ti = ends[a]
            for b in range(a + 1, len(ends)):
                j, ej, pj, tj = ends[b]
                if i == j or (id(lines[i]), id(lines[j])) in rejected:
                    continue
                gap = pj - pi
                distance = float(np.linalg.norm(gap))
                if distance > max_gap:
                    continue
                if float(ti @ -tj) < cos_limit:
                    continue
                if distance > 0.5 * min(radii[i], radii[j]):
                    unit = gap / distance
                    if float(unit @ ti) < cos_limit or float(-unit @ tj) < cos_limit:
                        continue
                    bridge = pi[None] + np.linspace(0.0, 1.0, max(int(distance), 2))[:, None] * gap[None]
                    if float(sample_image(image, bridge).mean()) < min_bridge_support:
                        continue
                score = distance * (2.0 - float(ti @ -tj))
                if best is None or score < best[0]:
                    best = (score, i, ei, j, ej)
        if best is None:
            break
        _, i, ei, j, ej = best
        first = lines[i] if ei == 1 else lines[i][::-1]
        second = lines[j] if ej == 0 else lines[j][::-1]
        joined = np.vstack([first, second])
        if min_bend_radius is not None and _kink_index(joined, min_bend_radius, kink_threshold) is not None:
            rejected.add((id(lines[i]), id(lines[j])))
            continue
        length_i, length_j = polyline_length(lines[i]), polyline_length(lines[j])
        radius = (radii[i] * length_i + radii[j] * length_j) / max(length_i + length_j, 1e-9)
        keep = [k for k in range(len(lines)) if k not in (i, j)]
        lines = [lines[k] for k in keep] + [joined]
        radii = np.concatenate([radii[keep], [radius]])
        merges += 1
    return lines, radii, merges


def _join(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Join ``first`` (ending at the junction) to ``second`` (starting there).

    Both are cut back at the plane through the midpoint of their tips, so
    pieces separated by a gap are bridged and pieces that overlap are trimmed.
    """
    _, out = _end(first, 1)
    _, back = _end(second, 0)
    axis = out - back
    axis /= max(np.linalg.norm(axis), 1e-12)
    middle = 0.5 * (first[-1] + second[0])
    stop = len(first)
    while stop > 2 and float((first[stop - 1] - middle) @ axis) > 0.0:
        stop -= 1
    start = 0
    while start < len(second) - 2 and float((second[start] - middle) @ axis) < 0.0:
        start += 1
    return np.vstack([first[:stop], second[start:]])


def _oriented(line: np.ndarray, end: int, at_start: bool) -> np.ndarray:
    """``line`` ordered so that its end ``end`` comes first (``at_start``) or last."""
    if at_start:
        return line if end == 0 else line[::-1]
    return line if end == 1 else line[::-1]


def _merge_with_prior(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    max_gap: float,
    max_angle_degrees: float,
    min_bend_radius: float | None,
    kink_threshold: float,
    end_cost: float,
    scale: float,
    chains: list[list[tuple[int, int]]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Join aligned end pairs whose join the scan and the length prior favor.

    Every pair of ends within ``max_gap`` whose directions (and the gap
    between them) agree within ``max_angle_degrees`` is a candidate; the ends
    may also overlap by up to two radii. A candidate is scored in nats: the
    drop in squared residual when the neighborhood is rendered with the joined
    fiber instead of the two pieces, divided by ``scale``, plus
    ``2 * end_cost`` for the two ends the join removes. Joins that would kink
    beyond the bend limit are never made.

    Pairings are chosen together: candidates are taken best first, each end
    joins at most once, and joins that would close a loop are skipped, so
    pieces meeting at a crossing are paired the way the scan supports best.

    ``chains``, when given, is filled with the input fibers each output fiber
    was joined from, in order along it, as ``(fiber, end it enters by)``: end
    1 means the fiber runs reversed in the output.
    """
    from scipy.spatial import cKDTree

    from ._ends import local_box, local_residual, near_box

    lines = [np.asarray(line, dtype=np.float64) for line in centerlines]
    radii = np.asarray(radii, dtype=np.float64)
    ends = [(i, e, *_end(line, e)) for i, line in enumerate(lines) if len(line) >= 3 for e in (0, 1)]
    if chains is not None:
        chains[:] = [[(i, 0)] for i in range(len(lines))]
    if len(ends) < 2:
        return lines, radii, 0
    cos_limit = np.cos(np.radians(max_angle_degrees))
    tips = np.array([tip for _, _, tip, _ in ends])
    # Geometric candidates first (cheap), keeping each end's nearest few.
    geometric: list[tuple[float, int, int]] = []
    for a, b in cKDTree(tips).query_pairs(max_gap):
        i, ei, pi, ti = ends[a]
        j, ej, pj, tj = ends[b]
        if i == j or float(ti @ -tj) < cos_limit:
            continue
        r = 0.5 * (radii[i] + radii[j])
        gap = pj - pi
        distance = float(np.linalg.norm(gap))
        along = float(gap @ ti)
        if along < -2.0 * r or float(-gap @ tj) < -2.0 * r:
            continue
        if distance > r:
            lateral = float(np.linalg.norm(gap - along * ti))
            if along <= 0.0 and lateral > 1.5 * r:
                continue
            if along > 0.0 and (along / distance < cos_limit or float(-gap @ tj) / distance < cos_limit):
                continue
            # Quick bound from the straight bridge: filling a gap of length g
            # with mean intensity m changes the residual by about
            # π r² g (1 - 2m); skip joins that can't come close to paying.
            bridge = pi[None] + np.linspace(0.0, 1.0, max(int(distance), 2))[:, None] * gap[None]
            mean = float(sample_image(image, bridge).mean())
            if np.pi * r * r * max(along, 0.0) * (2.0 * mean - 1.0) / scale + 4.0 * end_cost < 0.0:
                continue
        geometric.append((distance, a, b))
    per_end: dict[int, int] = {}
    shortlist = []
    for distance, a, b in sorted(geometric):
        if per_end.get(a, 0) >= 4 or per_end.get(b, 0) >= 4:
            continue
        per_end[a] = per_end.get(a, 0) + 1
        per_end[b] = per_end.get(b, 0) + 1
        shortlist.append((a, b))

    candidates = []
    for a, b in shortlist:
        i, ei, _, _ = ends[a]
        j, ej, _, _ = ends[b]
        r = 0.5 * (radii[i] + radii[j])
        joined = _join(_oriented(lines[i], ei, at_start=False), _oriented(lines[j], ej, at_start=True))
        if min_bend_radius is not None:
            junction = len(_oriented(lines[i], ei, at_start=False))
            excess = _kink_excess(joined, min_bend_radius, kink_threshold)
            window = excess[max(junction - 6, 0) : junction + 3]
            if len(window) and window.max() > 1.0:
                continue
        region = np.vstack([lines[i][-2:] if ei == 1 else lines[i][:2], lines[j][-2:] if ej == 1 else lines[j][:2]])
        low, high = local_box(region, 2.0 * r, image.shape)
        others = near_box(lines, radii, low, high, skip=(i, j))
        base = render_occupancy(low, high, [lines[k] for k in others], radii[others]) if others else None
        length_i, length_j = polyline_length(lines[i]), polyline_length(lines[j])
        radius = (radii[i] * length_i + radii[j] * length_j) / max(length_i + length_j, 1e-9)
        apart = local_residual(image, low, high, [lines[i], lines[j]], np.array([radii[i], radii[j]]), base)
        together = local_residual(image, low, high, [joined], np.array([radius]), base)
        odds = (apart - together) / scale + 2.0 * end_cost
        if odds > 0.0:
            candidates.append((odds, i, ei, j, ej))

    # Best first; each end once; no loops (union-find over fibers).
    parent = list(range(len(lines)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    link: dict[tuple[int, int], tuple[int, int]] = {}
    for _, i, ei, j, ej in sorted(candidates, reverse=True):
        if (i, ei) in link or (j, ej) in link or root(i) == root(j):
            continue
        link[(i, ei)] = (j, ej)
        link[(j, ej)] = (i, ei)
        parent[root(i)] = root(j)
    if not link:
        return lines, radii, 0

    # Walk each chain from a fiber with a free end.
    used = np.zeros(len(lines), dtype=bool)
    merged_lines, merged_radii, members = [], [], []
    for start in range(len(lines)):
        if used[start]:
            continue
        free = [e for e in (0, 1) if (start, e) not in link]
        if not free:
            continue  # interior of a chain; reached from its free end
        i, entry = start, free[0]
        path = _oriented(lines[i], entry, at_start=True)
        weight, total = radii[i] * polyline_length(lines[i]), polyline_length(lines[i])
        used[i] = True
        chain = [(i, entry)]
        while (i, 1 - entry) in link:
            j, ej = link[(i, 1 - entry)]
            path = _join(path, _oriented(lines[j], ej, at_start=True))
            weight += radii[j] * polyline_length(lines[j])
            total += polyline_length(lines[j])
            used[j] = True
            chain.append((j, ej))
            i, entry = j, ej
        merged_lines.append(path)
        merged_radii.append(weight / max(total, 1e-9))
        members.append(chain)
    if chains is not None:
        chains[:] = members
    return merged_lines, np.array(merged_radii), len(link) // 2


def split_kinks(
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_bend_radius: float,
    min_length: float,
    max_length: float | None = None,
    threshold: float = 2.0,
    min_angle_degrees: float = 35.0,
    image: np.ndarray | None = None,
    end_cost: float = 0.0,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Split fibers at kinks sharper than the bend limit, and overlong fibers.

    A trace that runs from one fiber onto another develops a kink once the fit
    pulls each part onto its own fiber. The turn is measured over two node
    spacings on each side to ignore node-scale noise; ``threshold`` is the
    allowed multiple of the admissible curvature. Fibers longer than
    ``max_length`` are cut where the image support along them is weakest.

    With an ``end_cost`` (the fiber-length prior, see ``_ends``), a kink is
    first smoothed out locally; the fiber is only cut when the smoothed
    fiber matches ``image`` worse than the kinked one by more than the cost of
    the two new ends. A trace that jumped onto another fiber cannot be smoothed
    without leaving both fibers, so it is still cut.
    """
    from ._ends import local_box, local_residual, near_box

    context_lines = [np.asarray(line) for line in centerlines]
    context_radii = np.asarray(radii, dtype=np.float64)
    # Each queued piece remembers which input fiber it came from, so that
    # fiber is left out of the rendering context around its own kink.
    queue = [(line, radius, source) for source, (line, radius) in enumerate(zip(centerlines, radii))]
    done: list[tuple[np.ndarray, float]] = []
    splits = 0
    kept = 0
    while queue:
        line, radius, source = queue.pop()
        cut = _kink_index(line, min_bend_radius, threshold, min_angle_degrees)
        if cut is not None and end_cost > 0 and image is not None and kept < 10 * len(centerlines):
            smoothed = _smooth_kink(line, cut, min_bend_radius, threshold, min_angle_degrees)
            if smoothed is not None:
                window = np.vstack([line[max(cut - 6, 0) : cut + 7], smoothed[max(cut - 6, 0) : cut + 7]])
                low, high = local_box(window, 2.5 * radius, image.shape)
                others = near_box(context_lines, context_radii, low, high, skip=(source,))
                base = render_occupancy(low, high, [context_lines[k] for k in others], context_radii[others]) if others else None
                kinked = local_residual(image, low, high, [line], np.array([radius]), base)
                smooth = local_residual(image, low, high, [smoothed], np.array([radius]), base)
                if (kinked - smooth) / scale + 2.0 * end_cost > 0.0:
                    kept += 1
                    queue.append((smoothed, radius, source))
                    continue
        if cut is None and max_length is not None and polyline_length(line) > max_length:
            cut = _weakest_index(line, image, min_length)
        if cut is None:
            done.append((line, radius))
            continue
        splits += 1
        for part in (line[: cut + 1], line[cut:]):
            if len(part) >= 2 and polyline_length(part) >= min_length:
                queue.append((part, radius, source))
    return [line for line, _ in done], np.array([radius for _, radius in done]), splits


def _kink_excess(line: np.ndarray, min_bend_radius: float, threshold: float, min_angle_degrees: float = 35.0, k: int = 3) -> np.ndarray:
    """Turn over ``k`` node spacings on each side of node ``k..n-k``, as a multiple of the allowed turn."""
    if len(line) < 2 * k + 1:
        return np.zeros(0)
    before = line[k:-k] - line[: -2 * k]
    after = line[2 * k :] - line[k:-k]
    lengths = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.clip((before * after).sum(axis=1) / np.maximum(lengths, 1e-12), -1.0, 1.0)
    window = 0.5 * (np.linalg.norm(before, axis=1) + np.linalg.norm(after, axis=1))
    angle = np.arccos(cosine)
    limit = np.maximum(threshold * window / min_bend_radius, np.radians(min_angle_degrees))
    return angle / limit


def _kink_index(
    line: np.ndarray, min_bend_radius: float, threshold: float, min_angle_degrees: float = 35.0, k: int = 3
) -> int | None:
    excess = _kink_excess(line, min_bend_radius, threshold, min_angle_degrees, k)
    if len(excess) == 0:
        return None
    worst = int(np.argmax(excess))
    if excess[worst] <= 1.0:
        return None
    return worst + k


def _smooth_kink(
    line: np.ndarray,
    cut: int,
    min_bend_radius: float,
    threshold: float,
    min_angle_degrees: float = 35.0,
    k: int = 3,
    passes: int = 40,
) -> np.ndarray | None:
    """Relax the nodes around ``cut`` until no kink is left there, or give up (``None``)."""
    low, high = max(cut - 2 * k, 1), min(cut + 2 * k, len(line) - 2)
    if high <= low:
        return None
    smoothed = line.copy()
    for _ in range(passes):
        smoothed[low : high + 1] += 0.5 * (0.5 * (smoothed[low - 1 : high] + smoothed[low + 1 : high + 2]) - smoothed[low : high + 1])
        excess = _kink_excess(smoothed, min_bend_radius, threshold, min_angle_degrees, k)
        near = excess[max(low - k, 0) : max(high - k + 1, 0)]
        if len(near) == 0:
            return None
        if near.max() <= 1.0:
            return smoothed
    return None


def _weakest_index(line: np.ndarray, image: np.ndarray | None, min_length: float) -> int | None:
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    allowed = np.flatnonzero((arc >= min_length) & (arc <= arc[-1] - min_length))
    if len(allowed) == 0:
        return None
    if image is None:
        return int(allowed[len(allowed) // 2])
    values = sample_image(image, line)
    smooth = np.convolve(values, np.ones(3) / 3.0, mode="same")
    return int(allowed[np.argmin(smooth[allowed])])


def render_occupancy(
    box_low: np.ndarray, box_high: np.ndarray, lines: list[np.ndarray], radii: np.ndarray, edge: float = 1.2
) -> np.ndarray:
    """Soft union occupancy of capsules over the voxel box ``[low, high)`` (x, y, z)."""
    from ._geometry import segment_lines, segment_voxels

    box_low = np.asarray(box_low, dtype=int)
    box_high = np.asarray(box_high, dtype=int)
    occupancy = np.zeros(tuple(int(n) for n in (box_high - box_low)[::-1]))
    if not len(lines):
        return occupancy
    radii = np.asarray(radii, dtype=np.float64)
    radius_of = radii[segment_lines(lines)]
    flat = occupancy.ravel()
    # Occupancy is zero from a surface distance of ``edge`` on.
    for voxel, segment, distance in segment_voxels(lines, radii + 2 * edge, radii + edge, box_low, box_high):
        np.maximum.at(flat, voxel, np.clip(0.5 - (distance - radius_of[segment]) / (2 * edge), 0.0, 1.0))
    return occupancy


def resolve_side_by_side(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_length: float,
    reach: float = 2.3,
    end_cost: float = 0.0,
    scale: float = 1.0,
    max_angle_degrees: float = 30.0,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Decide, from the image, whether two adjacent parallel fits are one fiber.

    When two fits land on one fiber, contact pushes them apart until each
    sits about a radius off the true axis, where they look like two touching
    fibers. For every pair running side by side (at least 3 nodes of one fit
    within ``reach`` radii of the other, with tangents within
    ``max_angle_degrees``), the image in their neighborhood is compared with
    two renderings: both fibers as they are, and one fiber along their
    midline (the weaker fit is removed there). The rendering with the
    smaller squared residual wins. With an ``end_cost`` (the fiber-length
    prior), each fiber end a rendering adds or removes is charged or
    credited ``end_cost`` nats, the residual being measured in units of
    ``scale``. Fits that only cross are not tested.
    """
    from scipy.spatial import cKDTree

    from ._geometry import tangents

    lines = [line.copy() for line in centerlines]
    radii = np.asarray(radii, dtype=np.float64).copy()
    upper = np.array(image.shape[::-1])
    cos_limit = np.cos(np.radians(max_angle_degrees))
    changed = 0

    def index():
        """One tree over every fit's nodes; rebuilt only after a change."""
        live = [k for k in range(len(lines)) if len(lines[k]) >= 2]
        points = np.concatenate([lines[k] for k in live]) if live else np.zeros((0, 3))
        owner = np.concatenate([np.full(len(lines[k]), k) for k in live]) if live else np.zeros(0, dtype=int)
        starts = dict(zip(live, np.cumsum([0] + [len(lines[k]) for k in live])[:-1]))
        lows = np.array([lines[k].min(axis=0) - 2 * radii[k] if len(lines[k]) >= 2 else np.full(3, np.inf) for k in range(len(lines))])
        highs = np.array([lines[k].max(axis=0) + 2 * radii[k] if len(lines[k]) >= 2 else np.full(3, -np.inf) for k in range(len(lines))])
        return cKDTree(points) if len(points) else None, points, owner, starts, lows, highs

    tree, points, owner, starts, lows, highs = index()
    order = list(np.argsort([float(sample_image(image, line).mean()) for line in lines]))
    done: set[int] = set()
    while order and tree is not None:
        i = order.pop(0)
        if i in done or len(lines[i]) < 2:
            continue
        # Nearest node of any other fit, for each node of fit i: ask for
        # enough neighbors to get past fit i's own nodes within ``reach``.
        segment = float(np.median(np.linalg.norm(np.diff(lines[i], axis=0), axis=1)))
        own = int(2.0 * reach * radii[i] / max(segment, 1e-6)) + 1
        k_near = min(own + 8, len(points))
        if k_near < 2:
            break
        distance, nearest = tree.query(lines[i], k=k_near, distance_upper_bound=reach * radii[i])
        other = (nearest < len(points)) & (owner[np.minimum(nearest, len(points) - 1)] != i)
        has = other.any(axis=1)
        if has.sum() < 3:
            continue
        first = np.argmax(other, axis=1)
        rows = np.arange(len(lines[i]))
        near_point = nearest[rows, first]
        close = has & (distance[rows, first] < reach * radii[i])
        partner = np.where(close, owner[np.minimum(near_point, len(points) - 1)], -1)
        partners, counts = np.unique(partner[close], return_counts=True)
        if counts.size == 0 or counts.max() < 3:
            continue
        j = int(partners[np.argmax(counts)])
        mine = np.flatnonzero(partner == j)
        theirs = near_point[mine] - starts[j]
        # Crossing fits come close at a few nodes but are not parallel.
        cosine = np.abs((tangents(lines[i])[mine] * tangents(lines[j])[theirs]).sum(axis=1))
        if np.median(cosine) < cos_limit:
            continue
        region = np.vstack([lines[i][mine], lines[j][theirs]])
        margin = 2.5 * max(radii[i], radii[j])
        low = np.maximum(np.floor(region.min(axis=0) - margin).astype(int), 0)
        high = np.minimum(np.ceil(region.max(axis=0) + margin).astype(int), upper)
        observed = image[low[2] : high[2], low[1] : high[1], low[0] : high[0]]
        overlap = np.all(lows < high, axis=1) & np.all(highs > low, axis=1)
        overlap[[i, j]] = False
        neighbors = list(np.flatnonzero(overlap))
        context = [lines[k] for k in neighbors]
        context_radii = radii[neighbors]
        both = render_occupancy(low, high, context + [lines[i], lines[j]], np.concatenate([context_radii, [radii[i], radii[j]]]))
        merged_j = lines[j].copy()
        merged_j[theirs] = 0.5 * (lines[j][theirs] + lines[i][mine])
        keep_i = np.ones(len(lines[i]), dtype=bool)
        keep_i[mine] = False
        pieces = _runs(lines[i], keep_i, min_length)
        one = render_occupancy(
            low, high, context + [merged_j] + pieces,
            np.concatenate([context_radii, [radii[j]], np.full(len(pieces), radii[i])]),
        )
        added_ends = 2 * len(pieces) - 2
        gain = (((observed - both) ** 2).sum() - ((observed - one) ** 2).sum()) / scale
        if gain - added_ends * end_cost > 0.0:
            lines[j] = merged_j
            lines[i] = pieces[0] if pieces else np.empty((0, 3))
            for piece in pieces[1:]:
                lines.append(piece)
                radii = np.append(radii, radii[i])
            done.add(j)
            changed += 1
            tree, points, owner, starts, lows, highs = index()
        done.add(i)
    keep = [k for k, line in enumerate(lines) if len(line) >= 2]
    return [lines[k] for k in keep], radii[keep], changed


def _runs(line: np.ndarray, keep: np.ndarray, min_length: float) -> list[np.ndarray]:
    pieces, start = [], None
    for k, flag in enumerate(list(keep) + [False]):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            piece = line[start:k]
            if len(piece) >= 2 and polyline_length(piece) >= min_length:
                pieces.append(piece)
            start = None
    return pieces
