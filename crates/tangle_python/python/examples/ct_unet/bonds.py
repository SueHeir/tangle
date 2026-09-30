"""Bonds from the network's binder map: which traced fibers each blob of binder joins.

A bond is the binder (voxels the network calls binder, probability > 0.5) between one pair of traced fibers:
every binder voxel goes to the two fibers whose surfaces are nearest it, and a pair with enough voxels is a
bond. Its center is their centroid, its size their count. Binder next to one fiber only (a coating) is no bond.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from maps import BINDER

MIN_VOXELS = 4
REACH = 2.5  # voxels from a traced fiber's surface


def _dense(line, spacing=1.0):
    line = np.asarray(line, float)
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def extract_bonds(maps: np.ndarray, lines, radii, threshold: float = 0.5):
    """[{"fibers": (i, j) indices into ``lines``, "center": (x, y, z) voxels, "voxels": n}, ...]

    Each binder voxel is given to the pair of traced fibers whose surfaces are nearest it (both within
    ``REACH``); a pair with at least ``MIN_VOXELS`` voxels is a bond, centered on their centroid. Binder that
    touches one fiber only (a coating) makes no bond. Pairs, not connected blobs, so the neck around one
    crossing counts once even if it reads as two pieces, and neighbouring crossings stay apart."""
    z, y, x = np.nonzero(maps[BINDER][0] > threshold)
    return bonds_from_binder(np.stack([x, y, z], 1) + 0.5, lines, radii)


def bonds_from_binder(voxels: np.ndarray, lines, radii):
    """``extract_bonds`` for binder voxel centers (x, y, z) already picked out (e.g. tile by tile)."""
    if len(voxels) == 0 or not lines:
        return []
    points = [_dense(l) for l in lines]
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(points)])
    radius = np.asarray(radii, float)[owner]
    tree = cKDTree(np.concatenate(points))
    dist, idx = tree.query(voxels, k=24, distance_upper_bound=float(np.max(radii)) + REACH)
    pairs = {}
    for v in range(len(voxels)):
        ok = np.isfinite(dist[v])
        if ok.sum() < 2:
            continue
        gap = dist[v][ok] - radius[idx[v][ok]]
        fib = owner[idx[v][ok]]
        best = {}
        for f, g in zip(fib, gap):
            if g <= REACH and g < best.get(f, np.inf):
                best[f] = g
        if len(best) < 2:
            continue
        a, b = sorted(best, key=best.get)[:2]
        pairs.setdefault((min(a, b), max(a, b)), []).append(v)
    bonds = []
    for (a, b), members in pairs.items():
        if len(members) >= MIN_VOXELS:
            bonds.append({"fibers": (int(a), int(b)), "center": voxels[members].mean(0).tolist(),
                          "voxels": len(members)})
    return bonds


def score_bonds(found, true_centers, tolerance: float = 4.0):
    """Precision, recall and F1 of found bond centers against the true ones (each matched once, within
    ``tolerance`` voxels, nearest first)."""
    true_centers = np.asarray(true_centers, float).reshape(-1, 3)
    found_centers = np.asarray([b["center"] for b in found], float).reshape(-1, 3)
    if len(true_centers) == 0 and len(found_centers) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "found": 0, "true": 0}
    if len(true_centers) == 0 or len(found_centers) == 0:
        return {"precision": 0.0 if len(found_centers) else 1.0, "recall": 0.0 if len(true_centers) else 1.0,
                "f1": 0.0, "found": len(found_centers), "true": len(true_centers)}
    d = np.linalg.norm(found_centers[:, None] - true_centers[None], axis=2)
    matched, used_f, used_t = 0, set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[i, j] > tolerance:
            break
        if i in used_f or j in used_t:
            continue
        used_f.add(i), used_t.add(j)
        matched += 1
    p, r = matched / len(found_centers), matched / len(true_centers)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12), "found": len(found_centers),
            "true": len(true_centers)}


def extract_bond_points(maps: np.ndarray, lines, radii, threshold: float = 0.5):
    """Bonds from the bond-point map: every local peak above ``threshold`` is a bond, joining the two traced
    fibers whose surfaces are nearest it (both within ``REACH``)."""
    from maps import BONDPT

    heat = maps[BONDPT][0]
    peaks = (heat > threshold) & (heat == ndimage.maximum_filter(heat, size=5))
    z, y, x = np.nonzero(peaks)
    return bonds_at(np.stack([x, y, z], 1) + 0.5, heat[z, y, x], lines, radii)


def bond_peaks(maps: np.ndarray, threshold: float = 0.5, window=None, origin=(0, 0, 0)):
    """(centers (x, y, z), strengths) of the bond-point peaks above ``threshold`` inside ``window``."""
    from maps import BONDPT

    heat = maps[BONDPT][0]
    peaks = (heat > threshold) & (heat == ndimage.maximum_filter(heat, size=5))
    if window is not None:
        keep = np.zeros_like(peaks)
        keep[window] = True
        peaks &= keep
    z, y, x = np.nonzero(peaks)
    return np.stack([x + origin[2], y + origin[1], z + origin[0]], 1) + 0.5, heat[z, y, x]


def bonds_at(centers: np.ndarray, strengths, lines, radii):
    """Bonds at given points: each joins the two traced fibers whose surfaces are nearest it (within REACH)."""
    if len(centers) == 0 or not lines:
        return []
    points = [_dense(l) for l in lines]
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(points)])
    radius = np.asarray(radii, float)[owner]
    tree = cKDTree(np.concatenate(points))
    dist, idx = tree.query(centers, k=32, distance_upper_bound=float(np.max(radii)) + REACH)
    bonds = []
    for v in range(len(centers)):
        ok = np.isfinite(dist[v])
        best = {}
        for f, g in zip(owner[idx[v][ok]], dist[v][ok] - radius[idx[v][ok]]):
            if g <= REACH and g < best.get(f, np.inf):
                best[f] = g
        if len(best) >= 2:
            a, b = sorted(best, key=best.get)[:2]
            bonds.append({"fibers": (int(min(a, b)), int(max(a, b))), "center": centers[v].tolist(),
                          "strength": float(strengths[v])})
    return bonds


def merge_bonds(*groups, tolerance: float = 3.0):
    """One list from several bond finders: a bond is dropped if an earlier one joins the same fiber pair or
    lies within ``tolerance`` voxels of it."""
    kept = []
    for group in groups:
        for bond in group:
            c = np.asarray(bond["center"])
            if any(b["fibers"] == bond["fibers"] or np.linalg.norm(np.asarray(b["center"]) - c) < tolerance
                   for b in kept):
                continue
            kept.append(bond)
    return kept
