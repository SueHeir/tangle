# Copied from examples/ct_unet/bonds.py by its sync_package.py: change that file, then run it again.
"""Bonds from the network's binder map: which traced fibers each blob of binder joins.

The network only says, voxel by voxel, background, fiber or binder; bonds are found here. The binder outside the
traced fibers falls into blobs (26-connected voxels). A blob that touches one fiber only is a coating, and one that
touches none is loose binder (dust, a blob in the void): neither is a bond. A blob that touches two or more fibers
bonds them, but only pair by pair where the binder sits in the gap between the two: a voxel counts for a pair when
it lies within ``REACH`` of both fibers' surfaces, and a pair with at least ``MIN_VOXELS`` such voxels is a bond. So
one web spanning several fibers bonds the neighbours it fills the gaps between, not every fiber it touches to every
other. A bond's center is the centroid of its voxels and its size their count.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .maps import BINDER

TOUCH = 1.5  # voxels from a traced fiber's surface: binder this close touches the fiber
REACH = 2.5  # voxels from both fibers' surfaces: binder this close lies in the gap between them
MIN_TOUCH = 2  # voxels of a blob that must touch a fiber for the blob to hold it
MIN_VOXELS = 4  # voxels in the gap between a pair for a bond
NEIGHBOURS = 64  # nearest centerline points looked at per voxel (1 voxel apart, so a thick fiber takes many)


def _dense(line, spacing=1.0):
    line = np.asarray(line, float)
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def extract_bonds(maps: np.ndarray, lines, radii, threshold: float = 0.5, **settings):
    """[{"fibers": (i, j) indices into ``lines``, "center": (x, y, z) voxels, "voxels": n}, ...] from whole-scan
    maps: the voxels the network calls binder (probability > ``threshold``) outside the traced fibers, read by
    ``bonds_from_binder``."""
    z, y, x = np.nonzero(maps[BINDER][0] > threshold)
    voxels = np.stack([x, y, z], 1) + 0.5
    voxels = voxels[outside_fibers(voxels, lines, radii)]
    return bonds_from_binder(voxels, lines, radii, **settings)


def blobs(voxels: np.ndarray) -> np.ndarray:
    """Which 26-connected blob each voxel center (x, y, z) belongs to: (n,) labels from 0."""
    if len(voxels) == 0:
        return np.zeros(0, int)
    pairs = cKDTree(voxels).query_pairs(1.75, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(voxels),) * 2)
    return connected_components(graph, directed=False)[1]


def surface_gaps(voxels: np.ndarray, lines, radii, reach: float = REACH):
    """(voxel, fiber, gap) arrays: each voxel's distance to the surface of every traced fiber within ``reach`` of
    it (the fiber's capsule of its radius around its centerline; the nearest point of each fiber)."""
    points = [_dense(l) for l in lines]
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(points)])
    radius = np.asarray(radii, float)[owner]
    tree = cKDTree(np.concatenate(points))
    k = min(NEIGHBOURS, len(owner))
    dist, idx = tree.query(voxels, k=k, distance_upper_bound=float(np.max(radii)) + reach)
    dist, idx = dist.reshape(len(voxels), k), idx.reshape(len(voxels), k)
    ok = np.isfinite(dist)
    v = np.nonzero(ok)[0]
    fiber = owner[idx[ok]]
    gap = dist[ok] - radius[idx[ok]]
    near = gap <= reach
    v, fiber, gap = v[near], fiber[near], gap[near]
    order = np.lexsort((gap, fiber, v))  # nearest point of each fiber first
    v, fiber, gap = v[order], fiber[order], gap[order]
    first = np.r_[True, (v[1:] != v[:-1]) | (fiber[1:] != fiber[:-1])]
    return v[first], fiber[first], gap[first]


def bonds_from_binder(voxels: np.ndarray, lines, radii, *, touch: float = TOUCH, reach: float = REACH,
                      min_touch: int = MIN_TOUCH, min_voxels: int = MIN_VOXELS):
    """Bonds from binder voxel centers (x, y, z) outside the traced fibers (see the module notes)."""
    if len(voxels) == 0 or not lines:
        return []
    voxels = np.asarray(voxels, float)
    blob = blobs(voxels)
    v, fiber, gap = surface_gaps(voxels, lines, radii, max(reach, touch))
    # the fibers each blob touches
    touching = gap <= touch
    held = {}
    if touching.any():
        contacts, counts = np.unique(np.stack([blob[v[touching]], fiber[touching]], 1), axis=0, return_counts=True)
        for (b, f), n in zip(contacts, counts):
            if n >= min_touch:
                held.setdefault(int(b), set()).add(int(f))
    held = {b: fs for b, fs in held.items() if len(fs) >= 2}
    if not held:
        return []
    # per voxel of those blobs, the held fibers within reach of it
    within = {}
    for i, f, g in zip(v, fiber, gap):
        fs = held.get(int(blob[i]))
        if fs is not None and g <= reach and int(f) in fs:
            within.setdefault(int(i), []).append(int(f))
    pairs = {}
    for i, fs in within.items():
        fs = sorted(fs)
        for a in range(len(fs)):
            for c in range(a + 1, len(fs)):
                pairs.setdefault((int(blob[i]), fs[a], fs[c]), []).append(i)
    found = {}
    for (b, f, g), members in pairs.items():
        if len(members) < min_voxels:
            continue
        found.setdefault((f, g), []).append(members)
    bonds = []
    for (f, g), groups in sorted(found.items()):
        # one bond per pair and blob; the same pair bonded through two blobs close together (one neck the
        # threshold split) is one bond
        groups.sort(key=len, reverse=True)
        kept = []
        for members in groups:
            center = voxels[members].mean(0)
            for k in kept:
                if np.linalg.norm(k["center"] - center) < 2.0 * max(float(radii[f]), float(radii[g])) + reach:
                    k["members"] += members
                    k["center"] = voxels[k["members"]].mean(0)
                    break
            else:
                kept.append({"members": list(members), "center": center})
        for k in kept:
            bonds.append({"fibers": (f, g), "center": k["center"].tolist(), "voxels": len(k["members"])})
    return bonds


def outside_fibers(voxels: np.ndarray, lines, radii) -> np.ndarray:
    """Which of the voxel centers (x, y, z) lie outside every traced fiber (a capsule of its radius around its
    centerline): where binder can be, since the fibers always win."""
    keep = np.ones(len(voxels), bool)
    if len(voxels) == 0 or not lines:
        return keep
    points = [_dense(l) for l in lines]
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(points)])
    radius = np.asarray(radii, float)[owner]
    tree = cKDTree(np.concatenate(points))
    dist, idx = tree.query(voxels, k=8, distance_upper_bound=float(np.max(radii)) + 1.0)
    ok = np.isfinite(dist)
    return ~(ok & (dist <= radius[np.where(ok, idx, 0)])).any(axis=1)


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
