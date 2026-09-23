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
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Join pairs of ends that continue each other across a short gap.

    With ``min_bend_radius``, a join that would create a kink the split step
    would cut again is not made.
    """
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
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Split fibers at kinks sharper than the bend limit, and overlong fibers.

    A trace that runs from one fiber onto another develops a kink once the fit
    pulls each part onto its own fiber. The turn is measured over two node
    spacings on each side to ignore node-scale noise; ``threshold`` is the
    allowed multiple of the admissible curvature. Fibers longer than
    ``max_length`` are cut where the image support along them is weakest.
    """
    queue = [(line, radius) for line, radius in zip(centerlines, radii)]
    done: list[tuple[np.ndarray, float]] = []
    splits = 0
    while queue:
        line, radius = queue.pop()
        cut = _kink_index(line, min_bend_radius, threshold, min_angle_degrees)
        if cut is None and max_length is not None and polyline_length(line) > max_length:
            cut = _weakest_index(line, image, min_length)
        if cut is None:
            done.append((line, radius))
            continue
        splits += 1
        for part in (line[: cut + 1], line[cut:]):
            if len(part) >= 2 and polyline_length(part) >= min_length:
                queue.append((part, radius))
    return [line for line, _ in done], np.array([radius for _, radius in done]), splits


def _kink_index(
    line: np.ndarray, min_bend_radius: float, threshold: float, min_angle_degrees: float = 35.0, k: int = 3
) -> int | None:
    if len(line) < 2 * k + 1:
        return None
    before = line[k:-k] - line[: -2 * k]
    after = line[2 * k :] - line[k:-k]
    lengths = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.clip((before * after).sum(axis=1) / np.maximum(lengths, 1e-12), -1.0, 1.0)
    window = 0.5 * (np.linalg.norm(before, axis=1) + np.linalg.norm(after, axis=1))
    angle = np.arccos(cosine)
    limit = np.maximum(threshold * window / min_bend_radius, np.radians(min_angle_degrees))
    excess = angle / limit
    worst = int(np.argmax(excess))
    if excess[worst] <= 1.0:
        return None
    return worst + k


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
    from ._geometry import _box_segment_distances

    occupancy = np.zeros(tuple(int(n) for n in (box_high - box_low)[::-1]))
    for line, radius in zip(lines, radii):
        for a, b in zip(line[:-1], line[1:]):
            if np.any(np.minimum(a, b) - radius - 2 * edge > box_high) or np.any(np.maximum(a, b) + radius + 2 * edge < box_low):
                continue
            distance = _box_segment_distances(box_low, box_high, a, b)
            np.maximum(occupancy, np.clip(0.5 - (distance - radius) / (2 * edge), 0.0, 1.0), out=occupancy)
    return occupancy


def resolve_side_by_side(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_length: float,
    reach: float = 2.3,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Decide, from the image, whether two adjacent parallel fits are one fiber.

    When two fits land on one fiber, the non-overlap step pushes them apart
    until each sits about a radius off the true axis, where they look like two
    touching fibers. For every pair running side by side, the image in their
    neighborhood is compared with two renderings: both fibers as they are, and
    one fiber along their midline (the weaker fit is removed there). The
    rendering with the smaller squared residual wins.
    """
    from scipy.spatial import cKDTree

    lines = [line.copy() for line in centerlines]
    radii = np.asarray(radii, dtype=np.float64).copy()
    upper = np.array(image.shape[::-1])
    changed = 0
    index = 0
    order = list(np.argsort([float(sample_image(image, line).mean()) for line in lines]))
    done: set[int] = set()
    while order:
        i = order.pop(0)
        if i in done or len(lines[i]) < 2:
            continue
        others = [j for j in range(len(lines)) if j != i and len(lines[j]) >= 2]
        if not others:
            break
        points = np.concatenate([lines[j] for j in others])
        owner = np.concatenate([np.full(len(lines[j]), j) for j in others])
        starts = {j: s for j, s in zip(others, np.cumsum([0] + [len(lines[j]) for j in others])[:-1])}
        distance, nearest = cKDTree(points).query(lines[i])
        close = distance < reach * radii[i]
        if close.sum() < 3:
            continue
        partners, counts = np.unique(owner[nearest[close]], return_counts=True)
        j = int(partners[np.argmax(counts)])
        mine = np.flatnonzero(close & (owner[nearest] == j))
        if len(mine) < 3:
            continue
        theirs = nearest[mine] - starts[j]
        region = np.vstack([lines[i][mine], lines[j][theirs]])
        margin = 2.5 * max(radii[i], radii[j])
        low = np.maximum(np.floor(region.min(axis=0) - margin).astype(int), 0)
        high = np.minimum(np.ceil(region.max(axis=0) + margin).astype(int), upper)
        observed = image[low[2] : high[2], low[1] : high[1], low[0] : high[0]]
        neighbors = [
            k for k in range(len(lines))
            if k not in (i, j) and len(lines[k]) >= 2
            and np.all(lines[k].min(axis=0) - 2 * radii[k] < high)
            and np.all(lines[k].max(axis=0) + 2 * radii[k] > low)
        ]
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
        if ((observed - one) ** 2).sum() < ((observed - both) ** 2).sum():
            lines[j] = merged_j
            lines[i] = pieces[0] if pieces else np.empty((0, 3))
            for piece in pieces[1:]:
                lines.append(piece)
                radii = np.append(radii, radii[i])
            done.add(j)
            changed += 1
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
