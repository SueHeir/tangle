"""Redraw the unsure parts of a fit by growing the sure parts into them.

After a fit, every node has a confidence (``_confidence``). A redraw pass:

1. cuts the unsure stretches (confidence below a threshold) out of every
   fiber and keeps the sure pieces;
2. grows each cut end forward along its own direction into the unsure
   region with the tracer (it follows the scan, keeps within the bend
   limit, and keeps its direction through other fibers), stopping where the
   scan ends or where it would run into another piece;
3. leaves joining the grown ends, and new traces in what is still
   unexplained, to the fitter's usual steps;
4. and gives the solver the sure pieces as anchors: nodes on them are
   pinned, so they stay where they are and the rest fits around them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._confidence import _Segments
from ._geometry import paint, polyline_length, resample, tangents


Box = tuple[np.ndarray, np.ndarray]  # (low, high) corners, voxel coordinates (x, y, z)


@dataclass
class Cut:
    pieces: list[np.ndarray]
    parent: np.ndarray  # each piece's fiber index in the cut fit
    # (piece, 0 = start / -1 = end) where an unsure stretch was removed
    cut_ends: list[tuple[int, int]]
    anchors: list[np.ndarray]  # the sure pieces, less a free stretch at each cut end
    regions: list[Box]  # one box around each cluster of removed stretches
    removed_nodes: int
    removed_fibers: int


def cut_unsure(
    lines: list[np.ndarray],
    confidence: list[np.ndarray],
    radii: np.ndarray,
    *,
    threshold: float,
    spacing: float,
    min_piece_radii: float = 2.0,
    free_radii: float = 2.0,
    widen: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
    skip: list[Box] | None = None,
) -> Cut | None:
    """Split every fiber at its unsure stretches; ``None`` when nothing is unsure.

    ``widen`` lists boxes ``(low, high, extra)``: an unsure node inside one
    also removes the nodes within ``extra`` (arc length) of it, for a region
    whose earlier redraw failed. Nodes inside a ``skip`` box (a region given
    up on) are left as they are. The removed stretches are clustered (those
    within two radii of each other) into ``regions``, each a box padded by
    two radii.
    """
    radii = np.asarray(radii, dtype=np.float64)
    pad = 2.0 * float(radii.max()) if len(radii) else 0.0
    pieces: list[np.ndarray] = []
    parent: list[int] = []
    cut_ends: list[tuple[int, int]] = []
    anchors: list[np.ndarray] = []
    removed: list[np.ndarray] = []
    removed_nodes = 0
    removed_fibers = 0
    for f, (line, value) in enumerate(zip(lines, confidence)):
        r = float(radii[f])
        fine, fine_value = _refine_with_values(
            np.asarray(line, dtype=np.float64), np.asarray(value), spacing
        )
        unsure = fine_value < threshold
        if widen and unsure.any():
            unsure = _widen(fine, unsure, widen)
        if skip and unsure.any():
            unsure &= ~_inside_any(fine, skip)
        kept_mask = np.zeros(len(fine), dtype=bool)
        kept = 0
        for start, stop in _runs(~unsure):
            piece = fine[start:stop]
            if (
                len(fine)
                and unsure.any()
                and polyline_length(piece) < min_piece_radii * r
            ):
                continue
            kept_mask[start:stop] = True
            index = len(pieces)
            pieces.append(piece)
            parent.append(f)
            kept += 1
            cut_start, cut_stop = start > 0, stop < len(fine)
            if cut_start:
                cut_ends.append((index, 0))
            if cut_stop:
                cut_ends.append((index, -1))
            anchor = _trim_arc(
                piece,
                free_radii * r if cut_start else 0.0,
                free_radii * r if cut_stop else 0.0,
            )
            if len(anchor) >= 2:
                anchors.append(anchor)
        removed_nodes += int((~kept_mask).sum())
        for start, stop in _runs(~kept_mask):
            removed.append(fine[start:stop])
        if kept == 0:
            removed_fibers += 1
    if removed_nodes == 0:
        return None
    return Cut(
        pieces,
        np.array(parent, dtype=int),
        cut_ends,
        anchors,
        _cluster_boxes(removed, pad),
        removed_nodes,
        removed_fibers,
    )


def touched_regions(lines: list[np.ndarray], regions: list[Box]) -> list[set[int]]:
    """For every fiber, the regions its nodes enter."""
    touched = []
    for line in lines:
        line = np.asarray(line, dtype=np.float64).reshape(-1, 3)
        touched.append(
            {
                k
                for k, (low, high) in enumerate(regions)
                if np.any(np.all((line >= low) & (line <= high), axis=1))
            }
        )
    return touched


def region_components(region_count: int, *touch_lists: list[set[int]]) -> np.ndarray:
    """Group regions that share a fiber (before or after the redraw); a component id per region.

    A redraw is kept or reverted one component at a time, so a fiber is
    never half kept.
    """
    root = list(range(region_count))

    def find(k: int) -> int:
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for touches in touch_lists:
        for regions in touches:
            regions = sorted(regions)
            for other in regions[1:]:
                root[find(other)] = find(regions[0])
    labels = np.array([find(k) for k in range(region_count)], dtype=int)
    _, component = np.unique(labels, return_inverse=True)
    return component.astype(int)


def choose(
    old_touch: list[set[int]],
    new_touch: list[set[int]],
    component: np.ndarray,
    accepted: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Which old and which new fibers make up the merged fit.

    New fibers are kept unless they are in a reverted component; old fibers
    come back only for a reverted component.
    """

    def verdict(regions: set[int]) -> bool | None:
        if not regions:
            return None
        return bool(accepted[component[next(iter(regions))]])

    keep_new = [
        i for i, regions in enumerate(new_touch) if verdict(regions) is not False
    ]
    keep_old = [i for i, regions in enumerate(old_touch) if verdict(regions) is False]
    return keep_old, keep_new


def lost_fibers(
    lines: list[np.ndarray],
    candidates: list[int],
    kept: list[np.ndarray],
    radii: np.ndarray,
) -> list[int]:
    """Of ``candidates`` (indices into ``lines``), those less than half followed by a ``kept`` fiber."""
    if not candidates:
        return []
    if not kept:
        return list(candidates)
    segments = _Segments(kept, np.zeros(len(kept)))
    lost = []
    for i in candidates:
        line = np.asarray(lines[i], dtype=np.float64).reshape(-1, 3)
        if segments.covered(line, -1, extra=0.5 * float(radii[i])).mean() < 0.5:
            lost.append(i)
    return lost


def _widen(
    line: np.ndarray,
    unsure: np.ndarray,
    widen: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    )
    out = unsure.copy()
    for i in np.flatnonzero(unsure):
        extra = max(
            (
                e
                for low, high, e in widen
                if np.all((line[i] >= low) & (line[i] <= high))
            ),
            default=0.0,
        )
        if extra > 0.0:
            out |= np.abs(arc - arc[i]) <= extra
    return out


def _inside_any(points: np.ndarray, boxes: list[Box]) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for low, high in boxes:
        inside |= np.all((points >= low) & (points <= high), axis=1)
    return inside


def _cluster_boxes(stretches: list[np.ndarray], pad: float) -> list[Box]:
    """Boxes around clusters of stretches that come within ``pad`` of each other."""
    from scipy.spatial import cKDTree

    stretches = [s for s in stretches if len(s)]
    if not stretches:
        return []
    points = np.concatenate(stretches)
    owner = np.concatenate([np.full(len(s), k) for k, s in enumerate(stretches)])
    root = list(range(len(stretches)))

    def find(k: int) -> int:
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for a, b in cKDTree(points).query_pairs(max(pad, 1e-6)):
        ra, rb = find(int(owner[a])), find(int(owner[b]))
        if ra != rb:
            root[rb] = ra
    groups: dict[int, list[int]] = {}
    for k in range(len(stretches)):
        groups.setdefault(find(k), []).append(k)
    boxes = []
    for members in groups.values():
        cluster = np.concatenate([stretches[k] for k in members])
        boxes.append((cluster.min(axis=0) - pad, cluster.max(axis=0) + pad))
    return boxes


def grow_cut_ends(
    pieces: list[np.ndarray],
    cut_ends: list[tuple[int, int]],
    radii: np.ndarray,
    *,
    tracer_for,
    shape: tuple[int, int, int],
    spacing: float,
    max_length: float,
) -> tuple[list[np.ndarray], int]:
    """Extend every cut end along its own direction into the unsure region.

    ``tracer_for(piece_index, claimed)`` returns the ``_trace.Tracer`` for
    that piece's type, reading ``claimed`` (a one-based label volume of
    every piece, updated as they grow). The longest pieces grow first. A
    grown stretch may cross another piece, but one that runs onto another
    piece stops where it entered it, so two ends growing toward each other
    leave a small gap for the join rather than running side by side. Returns the pieces and the
    total length grown.
    """
    pieces = [piece.copy() for piece in pieces]
    claimed = np.zeros(shape, dtype=np.int32)
    for index, piece in enumerate(pieces):
        paint(claimed, piece, 1.1 * float(radii[index]), index + 1)
    order = sorted(cut_ends, key=lambda item: -polyline_length(pieces[item[0]]))
    grown = 0.0
    for index, end in order:
        piece = pieces[index]
        if len(piece) < 2:
            continue
        tracer = tracer_for(index, claimed)
        tip = piece[end]
        inner = (
            piece[min(2, len(piece) - 1)] if end == 0 else piece[max(len(piece) - 3, 0)]
        )
        direction = tip - inner
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction = direction / norm
        steps = max(int(max_length / tracer.step), 1)
        extension = tracer.trace_one_way(tip, direction, steps, own_label=index + 1)
        extension = _stop_before_others(
            np.array(extension).reshape(-1, 3), claimed, index + 1, pieces
        )
        if len(extension) == 0:
            continue
        grown += polyline_length(np.vstack([tip[None], extension]))
        paint(claimed, extension, 1.1 * float(radii[index]), index + 1, only_empty=True)
        joined = (
            np.vstack([extension[::-1], piece])
            if end == 0
            else np.vstack([piece, extension])
        )
        pieces[index] = resample(joined, spacing)
    return pieces, grown


def pinned_flags(
    lines: list[np.ndarray], anchors: list[np.ndarray], tolerance: float
) -> list[np.ndarray]:
    """Per node: whether it lies on an anchor (within ``tolerance``)."""
    if not anchors:
        return [np.zeros(len(line), dtype=bool) for line in lines]
    segments = _Segments(anchors, np.zeros(len(anchors)))
    return [
        segments.covered(
            np.asarray(line, dtype=np.float64).reshape(-1, 3), -1, extra=tolerance
        )
        for line in lines
    ]


def _stop_before_others(
    extension: np.ndarray,
    claimed: np.ndarray,
    own: int,
    pieces: list[np.ndarray],
    max_angle_degrees: float = 30.0,
) -> np.ndarray:
    """Cut a grown stretch where it runs onto another piece; let it cross one.

    Passing through another piece at more than ``max_angle_degrees`` to it
    is a crossing and is kept. Running along one (a smaller angle), or
    ending inside one, means the growth ran onto that piece, so it stops
    where it entered.
    """
    if len(extension) == 0:
        return extension
    index = np.clip(
        np.floor(extension[:, ::-1]).astype(int), 0, np.array(claimed.shape) - 1
    )
    owner = claimed[index[:, 0], index[:, 1], index[:, 2]]
    foreign = (owner > 0) & (owner != own)
    cos_limit = np.cos(np.radians(max_angle_degrees))
    for start, stop in _runs(foreign):
        if stop == len(extension):
            return extension[:start]
        label = int(np.bincount(owner[start:stop]).argmax())
        other = pieces[label - 1]
        if len(other) < 2:
            continue
        before = extension[max(start - 1, 0)]
        heading = extension[min(stop, len(extension) - 1)] - before
        heading /= max(np.linalg.norm(heading), 1e-12)
        middle = extension[(start + stop - 1) // 2]
        k = int(np.argmin(np.linalg.norm(other - middle, axis=1)))
        along = tangents(other)[k]
        if abs(float(heading @ along)) >= cos_limit:
            return extension[:start]
    return extension


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _refine_with_values(
    line: np.ndarray, value: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """``line`` resampled every ``spacing``, with ``value`` interpolated along it."""
    if len(line) < 2:
        return line, value
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    )
    fine = resample(line, spacing)
    fine_arc = np.linspace(0.0, arc[-1], len(fine))
    return fine, np.interp(fine_arc, arc, value)


def _trim_arc(line: np.ndarray, start: float, stop: float) -> np.ndarray:
    if len(line) < 2:
        return line
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    )
    keep = (arc >= start) & (arc <= arc[-1] - stop)
    return line[keep]
