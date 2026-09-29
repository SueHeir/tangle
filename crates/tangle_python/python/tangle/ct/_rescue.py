"""Coarse rescue (``FitSettings.coarse_rescue``): put the largest type's fibers back after a fit that traced
the smallest type first.

Traced first, the fine type lays fits along a coarse fiber's bright rim and claims it, so the coarse tracer
rarely gets the fiber. At the end of the fit, the coarse type is traced again on the scan, ignoring the fit's
claims. A candidate is kept when it is dim inside (the 90th percentile of the grey over its cross-section is
below ``coarse_rescue_grey`` of the way from void to the fine fits' axis grey, as a dim-cored coarse fiber is and
a bundle of fine fibers is not) and few fine fits run along its axis (under ``coarse_rescue_shared`` of its
nodes within 3 voxels of one). Kept candidates replace coarse fits that lie mostly inside them, and fine nodes
inside a coarse fit's oval section are cut out (the rim fits), keeping pieces of at least 10 voxels.
"""

from __future__ import annotations

import numpy as np

from ._geometry import polyline_length, sample_image, tangents

_MIN_COARSE = 20.0  # voxels: shortest candidate piece kept
_MIN_FINE = 10.0  # voxels: shortest fine piece kept after the cut


def coarse_rescue(fitter, lines, radii, types, void: float):
    """``(lines, radii, types, info)`` after the rescue (see the module notes)."""
    from scipy.spatial import cKDTree

    from . import _moves
    from ._trace import trace_fibers

    s = fitter.settings
    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in lines]
    fine, coarse = int(np.argmin(fitter.radius)), int(np.argmax(fitter.radius))
    if fine == coarse or not lines:
        return lines, radii, types, {}
    r = float(fitter.radius[coarse])
    ratio = float(fitter.ratio[coarse])
    candidates = trace_fibers(
        fitter.trace_image(coarse), fitter.hessian(coarse), radius=r, min_bend_radius=float(fitter.bend[coarse]),
        min_length=float(fitter.min_length[coarse]), node_spacing=fitter.spacing, claimed=None,
        foreground=fitter.foreground, depth=(fitter.edt, fitter.depth), label_offset=0, seed_depth_radii=0.7,
        bright_seed_strength=s.bright_seed_strength, peak_floor=s.trace_peak_floor, claim_radii=s.trace_claim_radii,
    )
    traced = len(candidates)
    if s.grey_checked_traces and candidates:
        candidates = fitter._grey_checked(candidates, coarse)
    if s.coarse_trace_axis_check and candidates:
        bright = fitter._bright_axis(candidates)
        if bright is not None:
            candidates = [line for line, b in zip(candidates, bright) if not b]
    if s.coarse_disc_max is not None and candidates:
        holds = fitter._holds_fine(candidates, coarse)
        candidates = [line for line, h in zip(candidates, holds) if not h]
    grey = fitter.axis_grey
    fine_lines = [line for line, k in zip(lines, types) if k == fine and len(line) >= 2]
    if grey is None or not fine_lines or not candidates:
        return lines, radii, types, {"candidates": traced, "accepted": 0}
    reference = float(np.median(np.concatenate([sample_image(grey, line) for line in fine_lines])))
    fine_tree = cKDTree(np.concatenate([_dense(line, 1.0) for line in fine_lines]))
    accepted = []
    for line in candidates:
        line = np.asarray(line, dtype=np.float64)
        p90 = _disc_p90(grey, line, r)
        shared = float(np.mean(fine_tree.query(_dense(line, 1.0))[0] <= 3.0))
        if (p90 - void) < s.coarse_rescue_grey * (reference - void) and shared < s.coarse_rescue_shared:
            accepted.append(line)
    if len(accepted) > 1:
        accepted, _ = _moves.trim_duplicates(accepted, np.full(len(accepted), r), min_length=_MIN_COARSE)
        accepted = [np.asarray(a, dtype=np.float64) for a in accepted]
    info = {"candidates": traced, "accepted": len(accepted)}
    if not accepted:
        return lines, radii, types, info
    new_index = _coarse_index(accepted, np.full(len(accepted), r), [fitter.long_axes(a, coarse) for a in accepted])
    replaced = {
        i for i in np.flatnonzero(types == coarse)
        if np.mean(_scaled_distance(new_index, lines[i], ratio) <= 1.0) >= 0.5
    }
    keep = [i for i in range(len(lines)) if i not in replaced]
    lines, radii, types = [lines[i] for i in keep], radii[keep], types[keep]
    old = [i for i in range(len(lines)) if types[i] == coarse]
    old_index = _coarse_index(
        [lines[i] for i in old], radii[old], [fitter.long_axes(lines[i], coarse) for i in old]
    ) if old else None
    added = []
    for line in accepted:
        added += _cut(line, _scaled_distance(old_index, line, ratio) <= 1.0, _MIN_COARSE)
    lines = lines + added
    radii = np.concatenate([radii, np.full(len(added), r)])
    types = np.concatenate([types, np.full(len(added), coarse)])
    coarse_ids = np.flatnonzero(types == coarse)
    index = _coarse_index(
        [lines[i] for i in coarse_ids], radii[coarse_ids], [fitter.long_axes(lines[i], coarse) for i in coarse_ids]
    )
    out_lines, out_radii, out_types, cut_nodes = [], [], [], 0
    for line, radius, kind in zip(lines, radii, types):
        if kind != fine:
            out_lines.append(line)
            out_radii.append(radius)
            out_types.append(kind)
            continue
        inside = _scaled_distance(index, line, ratio) <= 1.0
        cut_nodes += int(inside.sum())
        for piece in (_cut(line, inside, _MIN_FINE) if inside.any() else [line]):
            out_lines.append(piece)
            out_radii.append(radius)
            out_types.append(kind)
    info.update(replaced=len(replaced), added=len(added), fine_nodes_cut=cut_nodes)
    return out_lines, np.asarray(out_radii, dtype=np.float64), np.asarray(out_types, dtype=int), info


def _dense(line: np.ndarray, step: float) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64)
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if len(line) < 2 or s[-1] <= 0:
        return line
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, line[:, j]) for j in range(3)], axis=1)


def _frame(line: np.ndarray):
    t = tangents(line)
    helper = np.where(np.abs(t[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(t, helper)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    return t, e1, np.cross(t, e1)


def _disc_p90(grey: np.ndarray, line: np.ndarray, r: float) -> float:
    """90th percentile of the grey over discs of radius ``r`` across the line's interior nodes."""
    inner = line[1:-1] if len(line) > 2 else line
    _, e1, e2 = _frame(inner)
    steps = np.arange(-r, r + 1e-9, 0.75)
    u, v = np.meshgrid(steps, steps)
    keep = u**2 + v**2 <= r * r
    u, v = u[keep], v[keep]
    points = inner[:, None, :] + u[None, :, None] * e1[:, None, :] + v[None, :, None] * e2[:, None, :]
    return float(np.percentile(sample_image(grey, points.reshape(-1, 3)), 90))


def _coarse_index(lines, radii, long_axes):
    """Dense points of coarse lines with their long axis, tangent, short radius and end flags, and a KD-tree."""
    from scipy.spatial import cKDTree

    P, A, T, R, E = [], [], [], [], []
    for line, radius, axes in zip(lines, radii, long_axes):
        line = np.asarray(line, dtype=np.float64)
        if len(line) < 2:
            continue
        if axes is None:
            axes = _frame(line)[1]
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] <= 0:
            continue
        q = np.arange(0.0, s[-1] + 1e-9, 0.5)
        points = np.stack([np.interp(q, s, line[:, j]) for j in range(3)], axis=1)
        axes = np.asarray(axes, dtype=np.float64).copy()
        for i in range(1, len(axes)):
            if axes[i] @ axes[i - 1] < 0:
                axes[i] = -axes[i]
        a = np.stack([np.interp(q, s, axes[:, j]) for j in range(3)], axis=1)
        t = tangents(points)
        a = a - np.sum(a * t, axis=1, keepdims=True) * t
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        ends = np.zeros(len(points), dtype=bool)
        ends[0] = ends[-1] = True
        P.append(points), A.append(a), T.append(t), R.append(np.full(len(points), radius)), E.append(ends)
    if not P:
        return None
    P = np.concatenate(P)
    return {"P": P, "A": np.concatenate(A), "T": np.concatenate(T), "R": np.concatenate(R),
            "E": np.concatenate(E), "tree": cKDTree(P)}


def _scaled_distance(index, points: np.ndarray, ratio: float, near: int = 8) -> np.ndarray:
    """Smallest oval-scaled distance from each point to a coarse line, in units of its short radius (inf past ends)."""
    points = np.asarray(points, dtype=np.float64)
    if index is None or not len(points):
        return np.full(len(points), np.inf)
    d, j = index["tree"].query(points, k=min(near, len(index["P"])))
    j = np.atleast_2d(j).reshape(len(points), -1)
    best = np.full(len(points), np.inf)
    for column in range(j.shape[1]):
        jj = j[:, column]
        offset = points - index["P"][jj]
        t, a = index["T"][jj], index["A"][jj]
        along = np.sum(offset * t, axis=1)
        across = offset - along[:, None] * t
        u = np.sum(across * a, axis=1)
        w2 = np.maximum(np.sum(across * across, axis=1) - u * u, 0.0)
        scaled = np.sqrt(w2 + (u / ratio) ** 2) / index["R"][jj]
        scaled[index["E"][jj] & (np.abs(along) > 1.0)] = np.inf
        best = np.minimum(best, scaled)
    return best


def _cut(line: np.ndarray, bad: np.ndarray, min_length: float) -> list[np.ndarray]:
    """``line`` less its ``bad`` nodes: the remaining runs at least ``min_length`` voxels long."""
    out, k, good = [], 0, ~np.asarray(bad, dtype=bool)
    while k < len(line):
        if not good[k]:
            k += 1
            continue
        j = k
        while j < len(line) and good[j]:
            j += 1
        piece = line[k:j]
        if len(piece) >= 2 and polyline_length(piece) >= min_length:
            out.append(piece)
        k = j
    return out
