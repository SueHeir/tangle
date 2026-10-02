# Copied from examples/ct_unet/fiber_types.py by its sync_package.py: change that file, then run it again.
"""Fiber types from traced fibers: per-fiber features, then clustering into K types.

The network only finds fibers; types are decided per fiber, from features averaged along its whole length, so
noise washes out and any number of types works without retraining:

* ``log_radius``: log of the fiber's median predicted radius (voxels);
* ``grey``: the median scan grey along its axis, scaled so the void is 0 and bright fibers ~1;
* ``bend``: how far it strays from straight: over stretches 10 diameters long, the largest distance from the
  straight line joining a stretch's ends, as a share of the stretch's length (the median over the stretches, as
  a log). A long stretch keeps the tracer's jitter of a few tenths of a voxel far below a real bend, so straight
  and bendable fibers of one material and size still separate (``make_data --bends``);
* ``curvature`` (optional): its median curvature over ~5 voxels, in 1/radius units (noisier than ``bend``).

Types that differ in size or grey separate as before with bend added (typing accuracy unchanged within noise on
the varied and mixed test sets), and types that differ only in bending now separate where the tracer follows
the fibers cleanly.

``assign_types(features, k)`` clusters them with a Gaussian mixture (k-means++ start, a few EM rounds) and numbers
the types by size (0 = thinnest). With ``k=None`` it picks k in 1-4 by BIC: a guess, where giving k is reliable.
"""

import numpy as np
from scipy.ndimage import map_coordinates

FEATURES = ("log_radius", "grey", "bend")


def _dense(line: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    segment = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(segment)]
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def bend(line, radius: float) -> float:
    """How far a fiber strays from straight (see the module notes): 0 for a straight one, ~0.1 for one bending
    round about ten diameters."""
    p = _dense(np.asarray(line, float))  # 1 voxel apart
    w = max(int(round(20.0 * radius)), 8)  # 10 diameters
    stretches = [(0, len(p) - 1)] if len(p) <= w + 1 else [(a, a + w) for a in range(0, len(p) - w, max(w // 2, 1))]
    shares = []
    for a, b in stretches:
        chord = p[b] - p[a]
        length = float(np.linalg.norm(chord))
        if length < 1e-6:
            continue
        u = chord / length
        off = p[a:b + 1] - p[a]
        off -= (off @ u)[:, None] * u
        shares.append(float(np.linalg.norm(off, axis=1).max()) / length)
    return float(np.median(shares)) if shares else 0.0


def features(lines, radii, volume=None, grey=None, curvature: bool = False, bends: bool = True) -> np.ndarray:
    """(fibers, n) features: log radius, grey (with ``volume`` (z, y, x) and ``grey`` = (void, bright) levels),
    curvature (``curvature``), bend (``bends``, on by default)."""
    cols = [np.log(np.maximum(np.asarray(radii, float), 0.3))]
    if volume is not None:
        low, high = grey
        g = []
        for line in lines:
            p = _dense(np.asarray(line, float)) - 0.5
            v = map_coordinates(volume, [p[:, 2], p[:, 1], p[:, 0]], order=1, mode="nearest", output=np.float32)
            g.append((float(np.median(v)) - low) / max(high - low, 1e-6))
        cols.append(np.asarray(g))
    if curvature:
        c = []
        for line, r in zip(lines, radii):
            p = _dense(np.asarray(line, float), 2.5)
            if len(p) < 3:
                c.append(0.0)
                continue
            t = np.diff(p, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            turn = np.arccos(np.clip((t[1:] * t[:-1]).sum(1), -1, 1)) / 2.5
            c.append(float(np.median(turn)) * r)
        cols.append(np.asarray(c))
    if bends:
        cols.append(np.log(np.asarray([bend(line, r) for line, r in zip(lines, radii)]) + 0.005))
    return np.stack(cols, axis=1)


def _gmm(x, k, rng, rounds=60):
    n, d = x.shape
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):  # k-means++
        d2 = np.min([((x - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(x[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else x[rng.integers(n)])
    mu = np.array(centers)
    cov = np.array([np.cov(x.T).reshape(d, d) + 1e-3 * np.eye(d)] * k)
    w = np.full(k, 1.0 / k)
    for _ in range(rounds):
        logp = np.empty((n, k))
        for j in range(k):
            inv = np.linalg.inv(cov[j])
            diff = x - mu[j]
            logp[:, j] = (np.log(w[j]) - 0.5 * np.linalg.slogdet(cov[j])[1]
                          - 0.5 * np.einsum("ni,ij,nj->n", diff, inv, diff) - 0.5 * d * np.log(2 * np.pi))
        top = logp.max(1, keepdims=True)
        like = top[:, 0] + np.log(np.exp(logp - top).sum(1))
        resp = np.exp(logp - like[:, None])
        nk = resp.sum(0) + 1e-9
        w = nk / n
        mu = (resp.T @ x) / nk[:, None]
        for j in range(k):
            diff = x - mu[j]
            cov[j] = (resp[:, j, None] * diff).T @ diff / nk[j] + 1e-3 * np.eye(d)
    params = k * (d + d * (d + 1) / 2) + k - 1
    return resp.argmax(1), float(-2 * like.sum() + params * np.log(n)), mu


def assign_types(feats: np.ndarray, k: int | None = None, seed: int = 0):
    """(type per fiber, k): types numbered by size, 0 = thinnest. ``k=None``: k in 1-4 by BIC."""
    x = np.asarray(feats, float)
    if len(x) == 0:
        return np.zeros(0, int), k or 1
    scale = x.std(0)
    x = (x - x.mean(0)) / np.where(scale > 0, scale, 1.0)
    rng = np.random.default_rng(seed)
    choices = [k] if k else range(1, min(4, len(x)) + 1)
    best = None
    for kk in choices:
        if kk == 1:
            labels, bic, mu = np.zeros(len(x), int), float("inf") if k is None and len(x) < 2 else 0.0, x.mean(0)[None]
            if k is None:  # the one-type BIC, for comparison
                d = x.shape[1]
                cov = np.cov(x.T).reshape(d, d) + 1e-3 * np.eye(d)
                diff = x - mu[0]
                ll = (-0.5 * np.linalg.slogdet(cov)[1] - 0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
                      - 0.5 * d * np.log(2 * np.pi)).sum()
                bic = float(-2 * ll + (d + d * (d + 1) / 2) * np.log(len(x)))
        else:
            runs = [_gmm(x, kk, rng) for _ in range(4)]
            labels, bic, mu = min(runs, key=lambda r: r[1])
        if best is None or bic < best[1]:
            best = (labels, bic, mu, kk)
    labels, _, mu, kk = best
    order = np.argsort(np.argsort(mu[:, 0]))  # number the types by radius
    return order[labels], kk


def assign_by_size(radii, diameters) -> np.ndarray:
    """Types when the fiber sizes are known: each fiber gets the type whose diameter (voxels) is nearest its
    own measured diameter, in log terms. Types are numbered in the order of ``diameters``."""
    d = np.log(np.maximum(2.0 * np.asarray(radii, float), 0.3))
    known = np.log(np.asarray(diameters, float))
    return np.argmin(np.abs(d[:, None] - known[None, :]), axis=1)
