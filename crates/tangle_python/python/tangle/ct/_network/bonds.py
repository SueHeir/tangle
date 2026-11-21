# Copied from examples/ct_unet/bonds.py by its sync_package.py: change that file, then run it again.
"""Bonds from the network's binder map: which traced fibers each blob of binder joins.

The network only says, voxel by voxel, background, fiber or binder; bonds are found here. Two fibers are bonded
when binder, with no fiber material in between, runs from one fiber's surface to the other's without passing a
third fiber on the way: a web bonds each fiber to its direct neighbours, not every fiber it touches to every other.

The binder outside the traced fibers falls into blobs (26-connected voxels). A blob holds the fibers it touches
(binder within ``TOUCH`` of the surface, ``MIN_TOUCH`` voxels or more). Each voxel of a blob goes to the fiber
whose surface is nearest it (a Voronoi split of the binder), or to none when that fiber is not one the blob holds.
Two held fibers are bonded where their regions meet: 26-neighbouring voxels, one in each, a voxel touching both
(a gap one voxel wide), or a path between the two regions through binder that belongs to no fiber. So a coating (a blob holding one fiber) or loose binder
(holding none) bonds nothing, while touching coated fibers whose coatings merge are bonded. One bond per contact:
the voxels where the two regions meet, in 26-connected clusters (and a pair's contacts in two blobs within
2 radii + ``REACH`` of each other, one neck a threshold split, are one); a bond's center is the centroid of its
meeting voxels and its size their count.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .maps import BINDER

TOUCH = 1.5  # voxels from a traced fiber's surface: binder this close touches the fiber
REACH = 2.5  # voxels: two contacts of a pair in different blobs closer than 2 radii + this are one bond
MIN_TOUCH = 1  # voxels of a blob that must touch a fiber for the blob to hold it
NEIGHBOURS = 64  # nearest centerline points looked at per voxel (1 voxel apart, so a thick fiber takes many)
CHUNK = 50_000  # voxels per query in surface_gaps


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


def _neighbour_pairs(voxels: np.ndarray) -> np.ndarray:
    """(m, 2) index pairs of 26-neighbouring voxel centers."""
    return cKDTree(voxels).query_pairs(1.75, output_type="ndarray").reshape(-1, 2)


def _components(n: int, pairs: np.ndarray) -> np.ndarray:
    graph = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def blobs(voxels: np.ndarray) -> np.ndarray:
    """Which 26-connected blob each voxel center (x, y, z) belongs to: (n,) labels from 0."""
    if len(voxels) == 0:
        return np.zeros(0, int)
    return _components(len(voxels), _neighbour_pairs(voxels))


def surface_gaps(voxels: np.ndarray, lines, radii, reach: float = REACH):
    """(voxel, fiber, gap) arrays: each voxel's distance to the surface of every traced fiber within ``reach`` of
    it (the fiber's capsule of its radius around its centerline; the nearest point of each fiber; ``reach`` inf:
    the fibers of the nearest centerline points). The voxels go in chunks, so a scan full of binder takes bounded
    memory."""
    points = [_dense(l) for l in lines]
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(points)])
    radius = np.asarray(radii, float)[owner]
    tree = cKDTree(np.concatenate(points))
    k = min(NEIGHBOURS, len(owner))
    parts = []
    for start in range(0, len(voxels), CHUNK):
        chunk = voxels[start:start + CHUNK]
        dist, idx = tree.query(chunk, k=k, distance_upper_bound=float(np.max(radii)) + reach)
        dist, idx = dist.reshape(len(chunk), k), idx.reshape(len(chunk), k)
        ok = np.isfinite(dist)
        v = np.nonzero(ok)[0]
        fiber = owner[idx[ok]]
        gap = dist[ok] - radius[idx[ok]]
        near = gap <= reach
        v, fiber, gap = v[near] + start, fiber[near], gap[near]
        order = np.lexsort((gap, fiber, v))  # nearest point of each fiber first
        v, fiber, gap = v[order], fiber[order], gap[order]
        first = np.ones(len(v), bool)  # (np.r_[True, ...] would be one long when no voxel is near a fiber)
        first[1:] = (v[1:] != v[:-1]) | (fiber[1:] != fiber[:-1])
        parts.append((v[first], fiber[first], gap[first]))
    if not parts:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return tuple(np.concatenate(x) for x in zip(*parts))


def bonds_from_binder(voxels: np.ndarray, lines, radii, *, touch: float = TOUCH, reach: float = REACH,
                      min_touch: int = MIN_TOUCH):
    """Bonds from binder voxel centers (x, y, z) outside the traced fibers (see the module notes)."""
    if len(voxels) == 0 or not lines:
        return []
    voxels = np.asarray(voxels, float)
    n = len(voxels)
    pairs = _neighbour_pairs(voxels)
    blob = _components(n, pairs)
    v, fiber, gap = surface_gaps(voxels, lines, radii, np.inf)
    # each voxel's nearest fiber surface
    order = np.lexsort((gap, v))
    v, fiber, gap = v[order], fiber[order], gap[order]
    first = np.ones(len(v), bool)
    first[1:] = v[1:] != v[:-1]
    nearest = np.full(n, -1)
    nearest[v[first]] = fiber[first]
    # the fibers each blob holds: touched by min_touch voxels or more
    t = gap <= touch
    if not t.any():
        return []
    contact, counts = np.unique(np.stack([blob[v[t]], fiber[t]], 1), axis=0, return_counts=True)
    contact = contact[counts >= min_touch]
    held = set(map(tuple, contact.tolist()))
    region = np.array([f if (int(b), int(f)) in held else -1 for b, f in zip(blob, nearest)]) if n else nearest
    # where two regions meet: directly (neighbouring voxels of two fibers' regions) ...
    ra, rb = region[pairs[:, 0]], region[pairs[:, 1]]
    meet = {}
    direct = (ra >= 0) & (rb >= 0) & (ra != rb)
    for (i, j), fa, fb in zip(pairs[direct], ra[direct], rb[direct]):
        key = (int(blob[i]),) + ((int(fa), int(fb)) if fa < fb else (int(fb), int(fa)))
        meet.setdefault(key, set()).update((int(i), int(j)))
    # ... or in one voxel touching both (a gap one voxel wide, where neither region gets a voxel of its own)
    tv, tf = v[t], fiber[t]
    keep = np.array([(int(blob[i]), int(f)) in held for i, f in zip(tv, tf)], bool) if len(tv) else np.zeros(0, bool)
    tv, tf = tv[keep], tf[keep]
    for d in range(1, len(tv)):
        same = tv[d:] == tv[:-d]
        if not same.any():
            break
        for i, fa, fb in zip(tv[:-d][same], tf[:-d][same], tf[d:][same]):
            key = (int(blob[i]),) + ((int(fa), int(fb)) if fa < fb else (int(fb), int(fa)))
            meet.setdefault(key, set()).add(int(i))
    # ... or through binder that belongs to no fiber: every two regions bordering one such patch
    free = region < 0
    if free.any():
        both = free[pairs[:, 0]] & free[pairs[:, 1]]
        idx = np.nonzero(free)[0]
        local = np.full(n, -1)
        local[idx] = np.arange(len(idx))
        patch = np.full(n, -1)
        patch[idx] = _components(len(idx), local[pairs[both]])
        border = {}
        for i, j in pairs[free[pairs[:, 0]] != free[pairs[:, 1]]]:
            u, w = (i, j) if free[i] else (j, i)
            border.setdefault(int(patch[u]), {}).setdefault(int(region[w]), set()).add(int(w))
        for p, sides in border.items():
            fs = sorted(sides)
            for a in range(len(fs)):
                for c in range(a + 1, len(fs)):
                    w0 = next(iter(sides[fs[a]]))
                    meet.setdefault((int(blob[w0]), fs[a], fs[c]), set()).update(sides[fs[a]] | sides[fs[c]])
    # one bond per contact: the meeting voxels of a pair in a blob, 26-connected clusters
    found = {}
    for (b, f, g), members in meet.items():
        members = np.array(sorted(members))
        parts = _components(len(members), _neighbour_pairs(voxels[members])) if len(members) > 1 else np.zeros(1, int)
        for c in np.unique(parts):
            m = members[parts == c]
            found.setdefault((f, g), []).append((len(m), voxels[m].mean(0)))
    bonds = []
    for (f, g), groups in sorted(found.items()):
        groups.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for size, center in groups:
            for k in kept:
                if np.linalg.norm(k["center"] - center) < 2.0 * max(float(radii[f]), float(radii[g])) + reach:
                    k["center"] = (k["center"] * k["voxels"] + center * size) / (k["voxels"] + size)
                    k["voxels"] += size
                    break
            else:
                kept.append({"voxels": size, "center": np.asarray(center, float)})
        for k in kept:
            bonds.append({"fibers": (f, g), "center": k["center"].tolist(), "voxels": int(k["voxels"])})
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
    # every found-true pair within tolerance (a sparse list: scans can hold tens of thousands of bonds)
    near = cKDTree(found_centers).sparse_distance_matrix(cKDTree(true_centers), tolerance, output_type="ndarray")
    matched, used_f, used_t = 0, set(), set()
    for i, j in zip(near["i"][np.argsort(near["v"], kind="stable")], near["j"][np.argsort(near["v"], kind="stable")]):
        if i in used_f or j in used_t:
            continue
        used_f.add(i), used_t.add(j)
        matched += 1
    p, r = matched / len(found_centers), matched / len(true_centers)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12), "found": len(found_centers),
            "true": len(true_centers)}
