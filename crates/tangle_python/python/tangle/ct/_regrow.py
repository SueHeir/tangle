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


@dataclass
class Cut:
    pieces: list[np.ndarray]
    parent: np.ndarray  # each piece's fiber index in the cut fit
    cut_ends: list[
        tuple[int, int]
    ]  # (piece, 0 = start / -1 = end) where an unsure stretch was removed
    anchors: list[np.ndarray]  # the sure pieces, less a free stretch at each cut end
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
) -> Cut | None:
    """Split every fiber at its unsure stretches; ``None`` when nothing is unsure."""
    if not any(bool((np.asarray(c) < threshold).any()) for c in confidence):
        return None
    pieces: list[np.ndarray] = []
    parent: list[int] = []
    cut_ends: list[tuple[int, int]] = []
    anchors: list[np.ndarray] = []
    removed_nodes = 0
    removed_fibers = 0
    for f, (line, value) in enumerate(zip(lines, confidence)):
        r = float(radii[f])
        fine, fine_value = _refine_with_values(
            np.asarray(line, dtype=np.float64), np.asarray(value), spacing
        )
        sure = fine_value >= threshold
        removed_nodes += int((~sure).sum())
        runs = _runs(sure)
        kept = 0
        for start, stop in runs:
            piece = fine[start:stop]
            if polyline_length(piece) < min_piece_radii * r:
                continue
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
        if kept == 0:
            removed_fibers += 1
    return Cut(
        pieces,
        np.array(parent, dtype=int),
        cut_ends,
        anchors,
        removed_nodes,
        removed_fibers,
    )


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
