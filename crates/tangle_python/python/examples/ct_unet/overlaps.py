"""Fibers that pass through each other in a true structure, measured from its geometry.

The measure is the **covered share**: for a pair of fibers, the largest share of one fiber's cross-section that
lies inside the other, at the worst point along it. Each section is sampled at 37 area-weighted points, and each
sample is tested against the other fiber's solid (its offset from the nearest point on that fiber's curved axis,
inside its oval; a rounded cap past its ends). It holds for crossing, side-by-side, curved and ending fibers alike.
For crossing round fibers, a share of 0.15 is about a quarter of the thinner fiber's thickness inside the other,
0.45 about half (its axis reaches the other's surface) and 0.9 or more all of it.

Candidate pairs come from a quick screen: the overlap along the line joining the two axes where they come
closest, h_i(n) + h_j(n) - d with h each section's half-width along n (exact for crossing fibers, an
overstatement for side-by-side ovals, so the screen misses nothing).

``from_structure`` measures a relaxed structure (make_data's reject check); ``from_npz`` measures a finished
make_data volume, whose ovals' long axes it recovers from the label volume (the principal direction of each
fiber's voxels across its axis).

usage: python overlaps.py OUT DATA [DATA ...] [--workers N]
(DATA: make_data folders or .npz files; writes OUT/scans.csv, one row per volume, and OUT/pairs.csv, every pair
with a share above 0.1)
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

QUARTER, HALF, WHOLE = 0.15, 0.45, 0.9  # covered shares (see above)
SPACING = 0.25  # voxels between axis points
STEP = 4  # the screen looks at every 4th axis point
SCREEN = 0.15  # joining-line overlap, over the thinner fiber's thickness, that earns a pair the full measure
KEEP = 0.1  # pairs.csv keeps pairs with a share above this

# Section samples: rings at radius fractions 0, 0.35, 0.65, 0.9 of 1, 6, 12, 18 points, each weighted by its
# ring's share of the area (an affine map keeps area shares, so the same holds for ovals).
_RINGS = [(0.0, 1, 0.0, 0.175), (0.35, 6, 0.175, 0.5), (0.65, 12, 0.5, 0.775), (0.9, 18, 0.775, 1.0)]
SAMPLES = np.array([(f * np.cos(2 * np.pi * (k + 0.5 * (n == 12)) / n), f * np.sin(2 * np.pi * (k + 0.5 * (n == 12)) / n))
                    for f, n, _, _ in _RINGS for k in range(n)])
WEIGHTS = np.array([(hi * hi - lo * lo) / n for _, n, lo, hi in _RINGS for _ in range(n)])


class Fibers:
    """Axis points of every fiber, SPACING apart, stored fiber by fiber: positions and unit tangents (x, y, z
    voxels, tangents toward the fiber's end), the oval's long axis at each point (zeros for round fibers), and
    per fiber its long and short semi-axes ``a >= b`` (voxels)."""

    def __init__(self, points, tangents, fiber, long_axes, a, b):
        self.P, self.T, self.U = points, tangents, long_axes
        self.F = np.asarray(fiber, dtype=np.int64)  # 0-based fiber of each point
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        n = len(self.a)
        self.start = np.searchsorted(self.F, np.arange(n), side="left")
        self.end = np.searchsorted(self.F, np.arange(n), side="right")
        self.oval = self.a > 1.001 * self.b


def from_structure(source, voxel_size: float) -> Fibers:
    """The fibers of an ``Assembly`` or ``RunResult`` (lengths in voxels of ``voxel_size``)."""
    from tangle.ct._geometry import resample

    assembly = getattr(source, "assembly", source)
    assembly = assembly() if callable(assembly) else assembly
    lines = [np.asarray(line, dtype=np.float64) / voxel_size for line in assembly.centerlines()]
    semi = np.asarray(assembly.section_semi_axes(), dtype=np.float64).reshape(-1, 2) / voxel_size
    axes = assembly.long_axes() if np.any(semi[:, 0] > 1.001 * semi[:, 1]) else None
    P, T, U, F = [], [], [], []
    for k, line in enumerate(lines):
        dense = resample(line, SPACING)
        t = np.gradient(dense, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        u = np.zeros_like(dense)
        if axes is not None and semi[k, 0] > 1.001 * semi[k, 1]:
            nodes = np.asarray(axes[k], dtype=np.float64)
            _, near = cKDTree(line).query(dense)
            u = nodes[near] - (nodes[near] * t).sum(1, keepdims=True) * t
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        P.append(dense), T.append(t), U.append(u), F.append(np.full(len(dense), k))
    return Fibers(np.concatenate(P), np.concatenate(T), np.concatenate(F), np.concatenate(U),
                  semi.max(1), semi.min(1))


def from_scan(scan) -> Fibers:
    """The true fibers of a ``SyntheticScan`` (its centerlines, semi-axes and long axes are in voxels)."""
    from tangle.ct._geometry import resample

    n = len(scan.centerlines)
    if scan.semi_axes is not None:
        semi = np.asarray(scan.semi_axes, dtype=np.float64).reshape(-1, 2)
    else:
        semi = np.repeat(np.asarray(scan.radii, dtype=np.float64)[:, None], 2, axis=1)
    P, T, U, F = [], [], [], []
    for k in range(n):
        line = np.asarray(scan.centerlines[k], dtype=np.float64)
        dense = resample(line, SPACING)
        t = np.gradient(dense, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        u = np.zeros_like(dense)
        if scan.long_axes is not None and semi[k].max() > 1.001 * semi[k].min():
            nodes = np.asarray(scan.long_axes[k], dtype=np.float64)
            _, near = cKDTree(line).query(dense)
            u = nodes[near] - (nodes[near] * t).sum(1, keepdims=True) * t
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        P.append(dense), T.append(t), U.append(u), F.append(np.full(len(dense), k))
    return Fibers(np.concatenate(P), np.concatenate(T), np.concatenate(F), np.concatenate(U),
                  semi.max(1), semi.min(1))


def from_npz(data) -> Fibers:
    """The true fibers of a make_data volume (``np.load`` of its .npz)."""
    P = data["p_pos"].astype(np.float64)
    T = data["p_tan"].astype(np.float64)
    F = data["p_fid"].astype(np.int64) - 1
    n = int(F.max()) + 1
    if "semi_axes" in data.files and len(data["semi_axes"]) == n:
        semi = data["semi_axes"].astype(np.float64)
        a, b = semi.max(1), semi.min(1)
    else:
        first = np.searchsorted(F, np.arange(n))
        a = b = data["p_rad"].astype(np.float64)[np.minimum(first, len(F) - 1)]
    fibers = Fibers(P, T, F, np.zeros_like(P), a, b)
    if fibers.oval.any():
        fibers.U = _long_axes_from_labels(data["labels"], fibers)
    return fibers


def _long_axes_from_labels(labels, fibers: Fibers, stretch: int = 16):
    """Each oval's long axis per axis point: the leading direction of its voxels' offsets across the axis, over
    stretches of ``stretch`` points (each borrowing half of its neighbours' voxels)."""
    U = np.zeros_like(fibers.P)
    ids = np.nonzero(fibers.oval)[0] + 1  # labels are one-based
    zz, yy, xx = np.nonzero(np.isin(labels, ids))
    lab = labels[zz, yy, xx]
    order = np.argsort(lab, kind="stable")
    zz, yy, xx, lab = zz[order], yy[order], xx[order], lab[order]
    for f in ids - 1:
        s, e = fibers.start[f], fibers.end[f]
        lo, hi = np.searchsorted(lab, [f + 1, f + 2])
        if e - s < 2 or hi - lo < 3:
            continue
        voxels = np.stack([xx[lo:hi], yy[lo:hi], zz[lo:hi]], axis=1) + 0.5
        _, k = cKDTree(fibers.P[s:e]).query(voxels)
        t = fibers.T[s:e][k]
        o = voxels - fibers.P[s:e][k]
        o -= (o * t).sum(1, keepdims=True) * t
        M = np.zeros(((e - s + stretch - 1) // stretch, 3, 3))
        np.add.at(M, k // stretch, o[:, :, None] * o[:, None, :])
        shared = M.copy()
        shared[1:] += 0.5 * M[:-1]
        shared[:-1] += 0.5 * M[1:]
        U[s:e] = np.linalg.eigh(shared)[1][:, :, 2][np.arange(e - s) // stretch]
    return U


def _half_width(n, a, b, u):
    c = np.einsum("ij,ij->i", n, u)
    return np.sqrt(b * b + (a * a - b * b) * c * c)


def _joining(fib: Fibers, A, B):
    """Overlap along the line joining axis points A and B, and the distance between them."""
    v = fib.P[B] - fib.P[A]
    d = np.linalg.norm(v, axis=1)
    n = v / np.maximum(d, 1e-9)[:, None]
    meet = d < 0.05  # the axes (nearly) meet: their common normal
    if meet.any():
        c = np.cross(fib.T[A[meet]], fib.T[B[meet]])
        n[meet] = c / np.maximum(np.linalg.norm(c, axis=1), 1e-9)[:, None]
    fa, fb = fib.F[A], fib.F[B]
    return (_half_width(n, fib.a[fa], fib.b[fa], fib.U[A]) + _half_width(n, fib.a[fb], fib.b[fb], fib.U[B]) - d,
            d)


def _closest_approaches(fib: Fibers):
    """Per pair of fibers within reach: the axis points where the joining-line overlap is largest."""
    coarse = np.unique(np.concatenate([np.arange(s, e, STEP) for s, e in zip(fib.start, fib.end) if e > s]
                                      + [fib.end[fib.end > fib.start] - 1]))
    Fc = fib.F[coarse]
    reach = fib.a[Fc]
    sizes = np.unique(np.round(reach, 2))
    if len(sizes) > 8:  # many sizes: eight groups, each searched at its largest
        edges = np.quantile(reach, np.linspace(0, 1, 9)[1:])
        group = np.minimum(np.searchsorted(edges, reach), 7)
    else:
        group = np.searchsorted(sizes, np.round(reach, 2))
    members = [np.nonzero(group == g)[0] for g in range(group.max() + 1)]
    tops = [reach[m].max() if len(m) else 0.0 for m in members]
    trees = [cKDTree(fib.P[coarse[m]]) if len(m) else None for m in members]
    As, Bs = [], []
    for gi in range(len(members)):
        for gj in range(gi, len(members)):
            if trees[gi] is None or trees[gj] is None:
                continue
            r = tops[gi] + tops[gj] + 0.75
            if gi == gj:
                pairs = trees[gi].query_pairs(r, output_type="ndarray")
                A, B = members[gi][pairs[:, 0]], members[gi][pairs[:, 1]]
            else:
                near = trees[gi].sparse_distance_matrix(trees[gj], r, output_type="ndarray")
                A, B = members[gi][near["i"]], members[gj][near["j"]]
            keep = Fc[A] != Fc[B]
            As.append(coarse[A[keep]])
            Bs.append(coarse[B[keep]])
    A = np.concatenate(As) if As else np.zeros(0, np.int64)
    B = np.concatenate(Bs) if Bs else np.zeros(0, np.int64)
    if not len(A):
        return A, B, np.zeros(0), np.zeros(0)
    over, _ = _joining(fib, A, B)
    keep = over > -1.0
    A, B, over = A[keep], B[keep], over[keep]
    fa, fb = fib.F[A], fib.F[B]
    A, B = np.where(fa > fb, B, A), np.where(fa > fb, A, B)
    key = np.minimum(fa, fb) * (len(fib.a) + 1) + np.maximum(fa, fb)
    order = np.lexsort((-over, key))
    best = order[np.unique(key[order], return_index=True)[1]]
    A, B = A[best], B[best]
    # refine on every axis point within STEP of each
    offsets = np.arange(-STEP, STEP + 1)
    fa, fb = fib.F[A], fib.F[B]
    AA = np.clip(A[:, None] + offsets, fib.start[fa][:, None], fib.end[fa][:, None] - 1)
    BB = np.clip(B[:, None] + offsets, fib.start[fb][:, None], fib.end[fb][:, None] - 1)
    AA = np.repeat(AA[:, :, None], len(offsets), axis=2).reshape(len(A), -1)
    BB = np.repeat(BB[:, None, :], len(offsets), axis=1).reshape(len(A), -1)
    over, d = _joining(fib, AA.ravel(), BB.ravel())
    over, d = over.reshape(len(A), -1), d.reshape(len(A), -1)
    k = over.argmax(1)
    rows = np.arange(len(A))
    return AA[rows, k], BB[rows, k], over[rows, k], d[rows, k]


def _inside(fib: Fibers, points, f, lo, hi):
    """Which ``points`` lie inside fiber f, from its axis points lo..hi-1."""
    span = np.arange(lo, hi)
    _, k = cKDTree(fib.P[span]).query(points)
    q = span[k]
    w = points - fib.P[q]
    t = fib.T[q]
    along = (w * t).sum(1)
    across = w - along[:, None] * t
    past_end = ((q == fib.start[f]) & (along < 0)) | ((q == fib.end[f] - 1) & (along > 0))
    beyond = ((q == lo) | (q == hi - 1)) & ~past_end & (np.abs(along) > 0.3)  # nearest point is past the span
    a, b = fib.a[f], fib.b[f]
    if fib.oval[f]:
        u = fib.U[q] - (fib.U[q] * t).sum(1, keepdims=True) * t
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        wu = (across * u).sum(1)
        r2 = (wu / a) ** 2 + np.maximum((across * across).sum(1) - wu * wu, 0.0) / b**2
    else:
        r2 = (across * across).sum(1) / b**2
    r2 = np.where(past_end, r2 + (along / b) ** 2, r2)  # a rounded cap
    return (r2 < 1.0) & ~beyond


def covered_share(fib: Fibers, i, j, at_i, at_j):
    """The largest share of fiber i's section inside fiber j near axis points ``at_i`` (on i) and ``at_j`` (on j),
    and the length (voxels) of i's axis inside j there."""
    reach = int((2 * (fib.a[i] + fib.a[j]) + 2) / SPACING)
    along_i = np.arange(max(fib.start[i], at_i - reach), min(fib.end[i], at_i + reach + 1), 2)
    lo, hi = max(fib.start[j], at_j - 2 * reach), min(fib.end[j], at_j + 2 * reach + 1)
    if len(along_i) == 0 or hi - lo < 2:
        return 0.0, 0.0
    t = fib.T[along_i]
    u = fib.U[along_i] - (fib.U[along_i] * t).sum(1, keepdims=True) * t
    nu = np.linalg.norm(u, axis=1, keepdims=True)
    other = np.cross(t, np.where(np.abs(t[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    u = np.where(nu > 0.5, u / np.maximum(nu, 1e-12), other / np.linalg.norm(other, axis=1, keepdims=True))
    v = np.cross(t, u)
    samples = (fib.P[along_i][:, None, :] + fib.a[i] * SAMPLES[None, :, :1] * u[:, None, :]
               + fib.b[i] * SAMPLES[None, :, 1:] * v[:, None, :]).reshape(-1, 3)
    hit = _inside(fib, samples, j, lo, hi).reshape(len(along_i), -1)
    return float((hit @ WEIGHTS).max()), 2 * SPACING * float(hit[:, 0].sum())


def overlapping_pairs(fib: Fibers) -> list[dict]:
    """Every pair of fibers with a covered share above 0, worst first: fibers (0-based), share, the length of
    axis inside the other fiber (voxels), the crossing angle (degrees) and where (x, y, z voxels)."""
    A, B, over, _ = _closest_approaches(fib)
    out = []
    if not len(A):
        return out
    fa, fb = fib.F[A], fib.F[B]
    thin = 2 * np.minimum(fib.b[fa], fib.b[fb])
    for m in np.nonzero(over > SCREEN * thin)[0]:
        i, j = int(fa[m]), int(fb[m])
        s_ij, l_ij = covered_share(fib, i, j, A[m], B[m])
        s_ji, l_ji = covered_share(fib, j, i, B[m], A[m])
        if max(s_ij, s_ji) <= 0:
            continue
        cos = min(abs(float(fib.T[A[m]] @ fib.T[B[m]])), 1.0)
        out.append({"fibers": (i, j), "share": round(max(s_ij, s_ji), 3), "inside": round(max(l_ij, l_ji), 2),
                    "angle": round(float(np.degrees(np.arccos(cos))), 1),
                    "at": [round(float(x), 1) for x in 0.5 * (fib.P[A[m]] + fib.P[B[m]])]})
    return sorted(out, key=lambda p: -p["share"])


def summary(fib: Fibers, pairs: list[dict]) -> dict:
    """Counts for one structure: pairs past a quarter, half and all of the thinner fiber, the worst share."""
    shares = np.array([p["share"] for p in pairs])
    return {"fibers": len(fib.a), "oval_fibers": int(fib.oval.sum()),
            "worst_share": float(shares.max()) if len(shares) else 0.0,
            "quarter": int((shares > QUARTER).sum()), "half": int((shares > HALF).sum()),
            "whole": int((shares > WHOLE).sum())}


def _audit(path):
    started = time.perf_counter()
    try:
        fib = from_npz(np.load(path))
        pairs = overlapping_pairs(fib)
        return str(path), {**summary(fib, pairs), "seconds": round(time.perf_counter() - started, 2)}, pairs, None
    except Exception as error:  # an unreadable volume is reported, not fatal
        return str(path), None, [], repr(error)


def main() -> None:
    args = sys.argv[1:]
    workers = 1
    if "--workers" in args:
        k = args.index("--workers")
        workers = int(args[k + 1])
        del args[k:k + 2]
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for arg in map(Path, args[1:]):
        files += sorted(arg.glob("*.npz")) if arg.is_dir() else [arg]
    files = [f for f in files if not f.name.startswith("._")]
    started = time.perf_counter()
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            results = list(pool.imap_unordered(_audit, files))
    else:
        results = [_audit(f) for f in files]
    results.sort()
    keys = next((list(s) for _, s, _, _ in results if s), [])
    with open(out / "scans.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["set", "scan"] + keys + ["error"])
        for path, s, _, error in results:
            p = Path(path)
            w.writerow([p.parent.name, p.stem] + ([s[k] for k in keys] if s else [""] * len(keys)) + [error or ""])
    with open(out / "pairs.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["set", "scan", "fiber_a", "fiber_b", "share", "inside", "angle", "x", "y", "z"])
        for path, _, pairs, _ in results:
            p = Path(path)
            for q in pairs:
                if q["share"] > KEEP:
                    w.writerow([p.parent.name, p.stem, q["fibers"][0] + 1, q["fibers"][1] + 1, q["share"],
                                q["inside"], q["angle"], *q["at"]])
    bad = sum(1 for _, s, _, _ in results if s and s["quarter"])
    errors = sum(1 for *_, e in results if e)
    print(f"{len(results)} volumes, {bad} with a pair past a quarter, {errors} unreadable "
          f"({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
