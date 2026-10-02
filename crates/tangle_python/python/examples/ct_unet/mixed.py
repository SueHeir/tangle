"""The mixed family of training scans: every kind of variety the map network should meet, mixed at random inside
each scan (``make_data.py OUT FIRST COUNT --mixed``; scans ``mixed_<index>``).

Each scan draws, independently:

* Fiber types: 1-4, each with its own diameter (3.5-28 voxels; some with a continuous spread of diameters around
  it, some thinning along each fiber), section (round or oval), core (solid, dim or hollow), brightness and
  phase ratio.
* Shape, per type: really straight, gently bent, tightly curled (coils) or wavy (crimped like wool: a smooth wave
  or a zigzag, in a plane or helical); in some types a share of the fibers loops round or doubles back (hairpins).
* Packing, per type: single fibers, parallel bundles, twisted yarn-like bundles (low to high twist) or touching
  side-by-side pairs and triplets; the whole scan from nearly empty to dense.
* Orientation: flat layers, cross-ply layers, fully random, or aligned along one axis.
* Things that aren't fibers: short broken fiber pieces, binder (bonds and webs at junctions, and blobs stuck to
  fibers, some with air bubbles inside), dust particles (a few dense enough to streak), and voids (pockets the
  fibers keep out of).
* Edges: a cut face of the sample with air beyond it, or the edge of the scanner's field of view.
* Scan quality: noise, blur, contrast (some samples embedded in resin) and phase halos varied; ring artifacts,
  streaks (fewer projections, dense particles), beam-hardening cupping and slice-to-slice brightness drift.

The hard spots come with these: fiber ends inside the volume (short fibers, broken pieces, thinning tips),
shallow-angle crossings (aligned and layered scans) and fibers leaving the crop (training crops are smaller than
the volume).

The true structure is relaxed by Tangle, so fibers never pass through each other, nor through themselves at
loops, hairpins and coils (``overlaps.py`` checks pairs, ``self_overlaps`` here each fiber). It is rendered here
(fiber shading, binder, dust, edges) and scanned with the scan simulator's physics (``tangle.ct._scanner``).

Indices: training from FIRST up, validation VAL, test TEST.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SIDE = 160  # voxels (1 um): the volume; training crops are 128
FIRST = 100_001  # training scans from here up
VAL = range(99_001, 99_033)
TEST = range(99_501, 99_549)
BENDS_FIRST = 200_001  # bends_<index> (bends_settings): training from here up
BENDS_VAL = range(199_001, 199_033)
BENDS_TEST = range(199_501, 199_549)
SEAMS_FIRST = 129_001  # mixed scans from here up may also have binder seams (val 129001-129032, test 129501-129548,
# training from 130001); the scans below keep exactly what they were made with
SEED = 1_000_000  # each scan's random stream is default_rng(SEED + index): no other family's seeds are met
MAX_FIBERS = 900  # a cap on fibers per scan (broken pieces included), for the relaxation's sake
RELAX_MORE = (3000, 9000)  # more relaxation steps, in turn, while fibers still pass through each other
PLACEMENTS = 3  # then new placements
BINDER_CAP = 6.0  # voxels: the largest bond or blob radius (thick fibers' bonds would otherwise fill the gaps)


# -- the draw ----------------------------------------------------------------------------------------------------


def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def mixed_settings(index: int) -> dict:
    """Everything random about ``mixed_<index>`` (lengths in voxels), drawn from the index alone."""
    rng = np.random.default_rng(SEED + index)
    k = int(rng.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.22, 0.13]))
    while True:  # neighbouring types at least 30% apart in diameter, so they are different materials
        d = np.sort(np.exp(rng.uniform(np.log(3.5), np.log(28.0), k)))
        if k == 1 or np.all(d[1:] / d[:-1] >= 1.3):
            break
    fill = str(rng.choice(["nearly empty", "sparse", "medium", "dense"], p=[0.08, 0.25, 0.4, 0.27]))
    total = float(rng.uniform(*{"nearly empty": (0.002, 0.012), "sparse": (0.012, 0.06), "medium": (0.06, 0.17),
                                "dense": (0.17, 0.3)}[fill]))
    share = rng.dirichlet(np.ones(k))
    bright = rng.uniform(0.4, 1.0, k)
    bright /= bright.max()
    orientation = str(rng.choice(["planar", "biaxial", "isotropic", "aligned"], p=[0.35, 0.15, 0.25, 0.25]))
    types = [_type_settings(rng, float(d[t]), float(share[t]), float(bright[t])) for t in range(k)]

    pockets = []
    if rng.random() < 0.2:  # voids: pockets the fibers keep out of
        for _ in range(int(rng.integers(1, 4))):
            pockets.append({"center": rng.uniform(0, SIDE, 3).round(1).tolist(),
                            "radii": rng.uniform(10, 35, 3).round(1).tolist()})

    binder = None
    mode = str(rng.choice(["none", "bonds", "blobs", "both"], p=[0.45, 0.3, 0.1, 0.15]))
    if mode != "none":
        binder = {"brightness": round(float(rng.uniform(0.3, 1.2)), 2),
                  "delta_beta": round(float(rng.uniform(4.0, 16.0)), 1)}
        if mode in ("bonds", "both"):
            binder["bonds"] = {"shape": str(rng.choice(["bridge", "meniscus", "meniscus", "blob"])),
                               "probability": round(float(rng.uniform(0.3, 1.0)), 2),
                               "gap": round(float(rng.uniform(0.3, 2.0)), 2),
                               "radius_ratio": round(float(rng.uniform(0.6, 2.0)), 2)}
        if mode in ("blobs", "both"):
            binder["blobs"] = {"count": int(rng.integers(5, 60)), "size": [0.8, round(float(rng.uniform(1.2, 2.5)), 2)],
                               "bubbles": round(float(rng.uniform(0.0, 0.6)), 2)}
    if index >= SEAMS_FIRST:
        # Seams: binder filling the groove along fibers lying side by side, over their whole shared length (glued
        # pairs, rafts, bound bundles). From a stream of its own, so the draws below stay as they were.
        rs = np.random.default_rng([SEED + index, 9])
        if (binder and rs.random() < 0.6) or (not binder and rs.random() < 0.25):
            binder = binder or {"brightness": round(float(rs.uniform(0.3, 1.2)), 2),
                                "delta_beta": round(float(rs.uniform(4.0, 16.0)), 1)}
            binder["seams"] = {"share": round(float(rs.uniform(0.3, 1.0)), 2),  # of the side-by-side stretches
                               "size": round(float(rs.uniform(0.5, 1.5)), 2),  # x the thinner fiber's radius
                               "gap": round(float(rs.uniform(0.3, 1.5)), 2),  # voxels between the two surfaces
                               "angle": round(float(rs.uniform(8.0, 20.0)), 1)}  # degrees between the fibers

    dust = None
    if rng.random() < 0.3:
        dust = {"count": int(rng.integers(3, 50)), "radius": [0.7, round(float(rng.uniform(1.5, 4.0)), 2)],
                "brightness": [0.4, round(float(rng.uniform(1.0, 3.0)), 2)],
                "dense": int(rng.integers(1, 6)) if rng.random() < 0.25 else 0,
                "delta_beta": round(float(rng.uniform(3.0, 12.0)), 1)}

    edge = None
    if rng.random() < 0.12:  # a cut face of the sample, air beyond it
        axis = np.zeros(3)
        axis[int(rng.integers(0, 3))] = rng.choice([-1.0, 1.0])
        normal = axis + rng.normal(0.0, 0.3, 3)
        edge = {"normal": (normal / np.linalg.norm(normal)).round(4).tolist(),
                "outside": round(float(rng.uniform(0.1, 0.45)), 3),
                "wave": round(float(rng.uniform(0.0, 4.0)), 2), "wavelength": round(float(rng.uniform(30, 120)), 1),
                "phase": round(float(rng.uniform(0, 2 * np.pi)), 3)}
    fov = None
    if rng.random() < 0.06:  # the edge of the reconstructed field of view (rotation about z)
        angle = float(rng.uniform(0, 2 * np.pi))
        fov = {"angle": round(angle, 3), "outside": round(float(rng.uniform(0.05, 0.35)), 3),
               "fill": str(rng.choice(["zero", "void"], p=[0.6, 0.4]))}

    blur = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
    thinnest = float(d[0])
    scanner = {  # the noise as in the earlier families (blotchier noise with fewer photons), up to 1.5x cleaner
        "photons": round(1300 * _log_uniform(rng, 0.5, 3.0) / max(1.0, 23 * blur**2)),
        "noise_blur": blur,
        "resolution": round(min(float(rng.uniform(1.0, 4.0)), max(1.0, thinnest)) if rng.random() < 0.7
                            else float(rng.uniform(1.0, 4.0)), 2),
        "propagation": round(0.0 if rng.random() < 0.3 else _log_uniform(rng, 0.5, 12.0), 2),
        "ring_strength": round(0.0 if rng.random() < 0.6 else _log_uniform(rng, 0.002, 0.01), 4),
        "fiber_motion": round(float(rng.uniform(0.0, 1.5)) if rng.random() < 0.5 else 0.0, 2),
        "drift": round(float(rng.uniform(0.0, 1.0)) if rng.random() < 0.2 else 0.0, 2),
        "angles": round(float(rng.uniform(0.3, 0.8)), 2) if rng.random() < 0.3 else None,  # of the slice width
        "beam_hardening": round(_log_uniform(rng, 0.2, 3.0), 2) if rng.random() < 0.4 else 0.0,
        "slice_drift": round(float(rng.uniform(0.01, 0.08)), 3) if rng.random() < 0.25 else 0.0,
        # resin around the fibers (x a brightness-1 fiber's attenuation), at most about half the dimmest fiber's
        "matrix": round(float(rng.uniform(0.15, 0.55) * bright.min()), 3) if rng.random() < 0.12 else 0.0,
        "matrix_delta_beta": round(float(rng.uniform(4.0, 16.0)), 1),
        "shading": round(float(rng.uniform(0.0, 0.15)), 3),  # slow brightness change across each slice
    }
    # Touching fibers stay separable: packed fibers (bundles, yarns, pairs) are never blurred past 0.8 of their
    # diameter. Resin's lower contrast is made up with photons, so the noise against the fibers stays the same.
    packed = [t["diameter"] for t in types if t["packing"] != "single"]
    if packed:
        scanner["resolution"] = round(min(scanner["resolution"], max(1.0, 0.8 * min(packed))), 2)
    if scanner["matrix"]:
        scanner["photons"] = round(scanner["photons"] / (1.0 - scanner["matrix"] / float(bright.min())) ** 2)
    return {"index": index, "fill": fill, "total": round(total, 4), "orientation": orientation,
            "tilt": round(float(rng.uniform(0.05, 0.5)), 2), "aligned_axis": int(rng.integers(0, 3)),
            "types": types, "pockets": pockets, "binder": binder, "dust": dust, "edge": edge, "fov": fov,
            "scanner": scanner, "brightness_spread": round(float(rng.uniform(0.0, 0.4)), 2), "seed": SEED + index}


def bends_settings(index: int) -> dict:
    """``bends_<index>``: two fiber types of one material and one size in equal amounts, one really straight and
    one bendable, so only bending tells them apart. The bendable one bends round a few diameters at its tightest
    (half the time), is crimped (a third) or curled; fibers are 4-12 voxels across, so they run at least about 13
    diameters through the volume and a bend has room to show. Section, core and scanner vary as in
    ``mixed_settings``; no binder, dust, voids or edges."""
    s = mixed_settings(index)
    rng = np.random.default_rng([SEED + index, 5])  # its own stream: the mixed draws above stay as they are
    d = _log_uniform(rng, 4.0, 12.0)
    base = _type_settings(rng, d, 0.5, 1.0)
    base.update(packing="single", loops=0.0, broken=0.0, length=[round(float(rng.uniform(0.45, 0.7)), 2),
                                                                  round(float(rng.uniform(0.8, 1.1)), 2)])
    straight = {**base, "shape": "straight", "bend": round(_log_uniform(rng, 40.0, 400.0), 1),
                "bow": round(float(rng.uniform(0.0, 0.01)), 4)}
    kind = str(rng.choice(["bent", "wavy", "curled"], p=[0.5, 0.35, 0.15]))
    if kind == "bent":
        bendy = {**base, "shape": "bent", "bend": round(float(rng.uniform(2.5, 5.0)), 1),
                 "curvature": round(float(rng.uniform(0.7, 0.95)), 2)}
    elif kind == "wavy":
        bendy = {**base, "shape": "wavy", "wave": str(rng.choice(["sine", "zigzag"])), "helical": bool(rng.random() < 0.3),
                 "amplitude": round(min(float(rng.uniform(0.5, 1.5)), 10.0 / d), 3),
                 "wavelength": round(float(rng.uniform(6.0, 16.0)), 2), "corner": round(float(rng.uniform(1.0, 3.0)), 2)}
    else:
        bendy = {**base, "shape": "curled", "coil": round(max(min(float(rng.uniform(1.0, 3.0)), 12.0 / d), 0.8), 3),
                 "pitch": round(float(rng.uniform(2.5, 8.0)), 2)}
    fill = str(rng.choice(["sparse", "medium", "dense"], p=[0.3, 0.45, 0.25]))
    total = float(rng.uniform(*{"sparse": (0.02, 0.06), "medium": (0.06, 0.15), "dense": (0.15, 0.25)}[fill]))
    return {**s, "fill": fill, "total": round(total, 4), "types": [straight, bendy], "pockets": [], "binder": None,
            "dust": None, "edge": None, "fov": None}


def _type_settings(rng, d: float, share: float, brightness: float) -> dict:
    """One fiber type of diameter ``d`` voxels (lengths below in diameters unless they say voxels)."""
    t = {"diameter": round(d, 2), "share": round(share, 4), "brightness": round(brightness, 3),
         "delta_beta": round(float(rng.uniform(4.0, 20.0)), 1)}
    t["ratio"] = round(float(rng.uniform(0.35, 0.9)), 2) if rng.random() < 0.35 else 1.0
    round_ = t["ratio"] == 1.0
    t["hollow"] = round(float(rng.uniform(0.3, 0.6)), 2) if round_ and d >= 7 and rng.random() < 0.15 else 0.0
    t["lumen"] = round(float(rng.uniform(0.0, 0.12)), 3)  # a hollow core's brightness
    t["rim"] = (round(float(rng.uniform(0.15, 0.3)), 3) if not t["hollow"] and d > 8 and rng.random() < 0.25
                else 0.0)  # a dim core: the rim's thickness (diameters) ...
    t["core"] = round(float(rng.uniform(0.4, 0.8)), 2)  # ... and the core's brightness
    t["taper"] = round(float(rng.uniform(0.55, 0.9)), 2) if round_ and d >= 6 and rng.random() < 0.2 else 1.0
    t["spread"] = round(float(rng.uniform(0.05, 0.2)), 3) if rng.random() < 0.35 else 0.0  # sd of log diameter

    shape = str(rng.choice(["straight", "bent", "curled", "wavy"], p=[0.25, 0.3, 0.15, 0.3]))
    if shape == "curled" and d > 12:  # a coil of a thick fiber would not fit the volume
        shape = "wavy"
    t["shape"] = shape
    if shape == "straight":
        t["bend"] = round(_log_uniform(rng, 40.0, 400.0), 1)  # the bend limit, diameters
        t["bow"] = round(float(rng.uniform(0.0, 0.01)), 4)  # sag over the length
    elif shape == "bent":
        t["bend"] = round(float(rng.uniform(4.0, 30.0)), 1)
        t["curvature"] = round(float(rng.uniform(0.25, 0.85)), 2)  # of the limit, at the tightest
    elif shape == "curled":
        coil = min(float(rng.uniform(1.0, 3.0)), 12.0 / d)
        pitch = float(rng.uniform(max(2.5, 1.4), 8.0))
        t.update(coil=round(max(coil, 0.8), 3), pitch=round(pitch, 2))
    else:
        t["wave"] = str(rng.choice(["sine", "zigzag"], p=[0.6, 0.4]))
        t["helical"] = bool(rng.random() < 0.3)
        t["amplitude"] = round(min(float(rng.uniform(0.3, 2.0)), 10.0 / d), 3)
        t["wavelength"] = round(float(rng.uniform(5.0, 25.0)), 2)
        t["corner"] = round(float(rng.uniform(1.0, 3.0)), 2)  # zigzag corners' bend radius
    t["loops"] = round(float(rng.uniform(0.05, 0.3)), 2) if rng.random() < 0.3 else 0.0

    packing = "single"
    if d < 14 and shape != "curled":
        packing = str(rng.choice(["single", "bundle", "yarn", "pairs"], p=[0.45, 0.2, 0.15, 0.2]))
    t["packing"] = packing
    if packing == "bundle":
        t["members"] = int(rng.choice([3, 5, 7, 7, 10, 19]))
    elif packing == "yarn":
        t["members"] = int(rng.choice([7, 12, 19]))
        t["twist"] = round(float(rng.uniform(5.0, 35.0)), 1)  # degrees, at the yarn's surface
    elif packing == "pairs":
        t["members"] = int(rng.choice([2, 2, 2, 3]))
    short = rng.random() < 0.2  # short fibers: many ends inside the volume
    t["length"] = ([round(float(rng.uniform(0.1, 0.25)), 2), round(float(rng.uniform(0.3, 0.5)), 2)] if short
                   else [round(float(rng.uniform(0.25, 0.6)), 2), round(float(rng.uniform(0.7, 1.1)), 2)])  # x SIDE
    t["broken"] = round(float(rng.uniform(0.1, 0.5)), 2) if rng.random() < 0.35 else 0.0  # pieces per fiber
    return t


# -- shapes ------------------------------------------------------------------------------------------------------


def resample(points, step: float) -> np.ndarray:
    """A polyline resampled at (about) ``step`` along its length, keeping both ends."""
    p = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / step)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


def max_curvature(points) -> float:
    """1 / the tightest bend radius along a polyline (turning angle per length at its nodes)."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return 0.0
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1)
    cos = np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]) / np.maximum(n[:-1] * n[1:], 1e-30), -1.0, 1.0)
    return float(np.max(np.arccos(cos) / np.maximum(0.5 * (n[:-1] + n[1:]), 1e-30)))


def _length(points) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def _frames(path):
    """Unit tangents and a rotation-minimizing normal along a polyline."""
    t = np.gradient(path, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.zeros_like(t)
    a = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n[0] = a - (a @ t[0]) * t[0]
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(t)):
        v = n[i - 1] - (n[i - 1] @ t[i]) * t[i]
        n[i] = v / max(np.linalg.norm(v), 1e-12)
    return t, n, np.cross(t, n)


def _axis(rng, t: dict, length: float, d: float, kink: bool, reach: float = 0.0) -> tuple[np.ndarray, float]:
    """A fiber's path in its own frame (along +x, centered near the origin) and the bend limit it needs
    (voxels): the type's shape, and a loop or hairpin when ``kink``. Drawn finely along x, then resampled evenly
    along its length (coils and waves run much longer than their extent in x). The axis of a group whose members
    sit up to ``reach`` off it bends no tighter than twice that, so the members stay smooth too."""
    x = np.arange(-0.5 * length, 0.5 * length + 1e-9, max(0.03 * d, 0.05))
    shape = t["shape"]
    if shape == "straight":
        bow = t["bow"] * length * rng.uniform(0.0, 1.0)
        y = bow * (1.0 - (2.0 * x / length) ** 2)
        path = np.stack([x, y, np.zeros_like(x)], axis=1)
        bend = t["bend"] * d
    elif shape == "bent":
        bend = t["bend"] * d
        y, z = np.zeros_like(x), np.zeros_like(x)
        for mode in (0.5, 1.0, 1.5, 2.0):
            phase = rng.uniform(0, 2 * np.pi, 2)
            weight = rng.normal(0.0, 1.0, 2) / mode**2
            y += weight[0] * np.sin(2 * np.pi * mode * x / length + phase[0])
            z += weight[1] * np.sin(2 * np.pi * mode * x / length + phase[1])
        path = np.stack([x, y, z], axis=1)
        kappa = max_curvature(resample(path, max(0.15 * d, 0.25))) or 1e-9  # amplitude 1: scale to the target
        scale = t["curvature"] * float(rng.uniform(0.6, 1.0)) / (bend * kappa)
        path[:, 1:] *= scale
    elif shape == "curled":
        radius = t["coil"] * d * float(rng.uniform(0.85, 1.15))
        pitch = t["pitch"] * d * float(rng.uniform(0.85, 1.15))
        theta = 2 * np.pi * x / pitch + rng.uniform(0, 2 * np.pi)
        path = np.stack([x, radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        c = pitch / (2 * np.pi)
        bend = 0.8 * (radius**2 + c**2) / radius
    else:  # wavy
        amplitude = t["amplitude"] * d * float(rng.uniform(0.8, 1.2))
        wavelength = t["wavelength"] * d * float(rng.uniform(0.85, 1.15))
        # the wave must not bend tighter than about a diameter
        wavelength = max(wavelength, 2 * np.pi * np.sqrt(amplitude * 1.2 * d), 4.0 * amplitude)
        phase = 2 * np.pi * x / wavelength + rng.uniform(0, 2 * np.pi)
        jitter = 1.0 + 0.15 * np.sin(2 * np.pi * x / (wavelength * rng.uniform(2.5, 5.0)) + rng.uniform(0, 6))
        if t["wave"] == "zigzag":  # a triangle wave whose corners are rounded to the corner radius
            corner = max(t["corner"] * d, 1.2 * d)
            raw = (2 / np.pi) * np.arcsin(np.sin(phase))
            sigma = max(corner / max(wavelength, 1e-9) * 2.0, 0.05)  # rounding, in wave periods
            from scipy.ndimage import gaussian_filter1d

            step = x[1] - x[0] if len(x) > 1 else 1.0
            wave = gaussian_filter1d(raw, sigma * wavelength / step / (2 * np.pi), mode="nearest")
        else:
            wave = np.sin(phase)
        y = amplitude * jitter * wave
        z = amplitude * jitter * np.cos(phase) if t["helical"] else np.zeros_like(x)
        path = np.stack([x, y, z], axis=1)
    path = resample(path, max(0.15 * d, 0.25))
    for _ in range(3):  # a group's axis: soften its bends (the sideways part of the path) to 2 x reach
        kappa = max_curvature(path)
        if reach <= 0 or kappa * 2.0 * reach <= 1.0:
            break
        path[:, 1:] *= 0.95 / (kappa * 2.0 * reach)
        path = resample(path, max(0.15 * d, 0.25))
    if shape == "wavy":
        bend = 0.8 / max(max_curvature(path), 1e-9)
    elif shape == "straight":  # stiff, and no straighter than its slight sag
        bend = min(bend, 0.9 / max(max_curvature(path), 1e-9))
    elif reach > 0:
        bend = max(min(bend, 0.9 / max(max_curvature(path), 1e-9)), 2.0 * reach)
    if kink:
        path, kink_bend = _kink(rng, path, d)
        bend = min(bend, kink_bend)
    return path, max(bend, 0.75 * d)


def _kink(rng, path, d):
    """``path`` with a loop (a full turn, rising by more than a diameter so it clears itself) or a hairpin (a
    U-turn back alongside itself) at a random point; returns it and the bend radius the turn needs."""
    n = len(path)
    cut = int(n * rng.uniform(0.3, 0.7))
    t, nrm, bi = _frames(path)
    p0, t0, n0, b0 = path[cut], t[cut], nrm[cut], bi[cut]
    step = max(np.linalg.norm(path[1] - path[0]), 1e-6)
    if rng.random() < 0.5:  # loop
        radius = float(rng.uniform(1.5, 4.0)) * d
        radius = min(radius, 15.0)
        rise = float(rng.uniform(1.4, 2.2)) * d
        theta = np.arange(0, 2 * np.pi, step / radius)
        loop = (p0 + np.outer(radius * np.sin(theta), t0) + np.outer(radius * (1 - np.cos(theta)), n0)
                + np.outer(rise * (theta - np.sin(theta)) / (2 * np.pi), b0))  # rises smoothly, level at both ends
        out = np.concatenate([path[:cut], loop, path[cut:] + rise * b0])
        return out, 0.85 * radius
    radius = min(float(rng.uniform(1.2, 3.5)) * d, 15.0)  # hairpin
    theta = np.arange(0, np.pi, step / radius)
    turn = p0 + np.outer(radius * np.sin(theta), t0) + np.outer(radius * (1 - np.cos(theta)), n0)
    # back alongside: the stretch before the turn, mirrored in the plane half way across the turn
    back = path[:cut][::-1]
    leg = back - 2 * np.outer((back - p0) @ n0 - radius, n0)
    keep = max(int(len(leg) * rng.uniform(0.4, 1.0)), 2)
    out = np.concatenate([path[:cut], turn, leg[:keep]])
    return out, 0.85 * radius


def _slots(t: dict) -> list[tuple[float, float]]:
    """Member positions across a group (in member spacings): a row for pairs, hexagonal rings for bundles."""
    count = t.get("members", 1)
    if t["packing"] == "pairs":
        return [(k - 0.5 * (count - 1), 0.0) for k in range(count)]
    slots = [(0.0, 0.0)]
    ring = 1
    while len(slots) < count:
        corners = [ring * np.array([np.cos(a), np.sin(a)]) for a in np.arange(6) * np.pi / 3]
        for k in range(6):
            for step in range(ring):
                slots.append(tuple(corners[k] + (corners[(k + 1) % 6] - corners[k]) * step / ring))
        ring += 1
    return slots[:count]


def _spacing(t: dict, d_mean: float) -> float:
    """Members' center spacing: a hair apart, more in a twisted yarn (its members lean)."""
    return 1.03 * d_mean / (np.cos(np.radians(t["twist"])) if t["packing"] == "yarn" else 1.0)


def _reach(t: dict, d_mean: float) -> float:
    """How far a group's members sit off its axis (voxels; 0 for single fibers)."""
    if t["packing"] == "single":
        return 0.0
    return max(np.hypot(a, b) for a, b in _slots(t)) * _spacing(t, d_mean)


def _members(rng, t: dict, axis: np.ndarray, d_mean: float) -> list[np.ndarray]:
    """The fibers of one group along ``axis``: one (single), a parallel bundle, a twisted yarn, or touching
    side-by-side pairs or triplets."""
    packing = t["packing"]
    if packing == "single":
        return [axis]
    slots = _slots(t)
    pitch = _spacing(t, d_mean)
    reach = _reach(t, d_mean)
    tan, nrm, bi = _frames(axis)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(axis, axis=0), axis=1))])
    twist = np.tan(np.radians(t["twist"])) / reach if packing == "yarn" and reach > 0 else 0.0  # turn per voxel
    roll = rng.uniform(0, 2 * np.pi)
    out = []
    for a, b in slots:
        ang = roll + twist * s
        ca, sa = np.cos(ang), np.sin(ang)
        u, v = a * ca - b * sa, a * sa + b * ca
        member = axis + pitch * (u[:, None] * nrm + v[:, None] * bi)
        cut = rng.integers(0, max(len(member) // 10, 1) + 1, size=2)  # staggered ends
        member = member[cut[0]: len(member) - cut[1]]
        if len(member) >= 4:
            out.append(member)
    return out


# -- placement ---------------------------------------------------------------------------------------------------


def _direction(rng, s: dict, layer: int | None) -> np.ndarray:
    tilt = s["tilt"]
    o = s["orientation"]
    if o == "isotropic":
        z = rng.uniform(-1, 1)
        a = rng.uniform(0, 2 * np.pi)
        return np.array([np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z])
    if o == "aligned":
        c = rng.uniform(np.cos(tilt), 1.0)
        a = rng.uniform(0, 2 * np.pi)
        v = np.array([c, np.sqrt(1 - c * c) * np.cos(a), np.sqrt(1 - c * c) * np.sin(a)]) * rng.choice([-1, 1])
        return np.roll(v, s["aligned_axis"])
    if o == "biaxial":
        base = (s["seed"] % 628) / 100.0
        a = base + (0.5 * np.pi if (layer or 0) % 2 else 0.0) + rng.normal(0.0, 0.08)
    else:
        a = rng.uniform(0, 2 * np.pi)
    e = rng.uniform(-tilt, tilt) * (0.4 if o == "biaxial" else 1.0)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _rotation(rng, direction) -> np.ndarray:
    """A rotation taking +x to ``direction``, rolled at random about it."""
    e1 = direction / np.linalg.norm(direction)
    a = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e2 = a - (a @ e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    r = rng.uniform(0, 2 * np.pi)
    e2, e3 = np.cos(r) * e2 + np.sin(r) * e3, -np.sin(r) * e2 + np.cos(r) * e3
    return np.stack([e1, e2, e3], axis=1)


def _inside_runs(points, low, high, min_points):
    """The longest stretch of ``points`` inside the box [low, high] (None if under ``min_points``)."""
    inside = np.all((points >= low) & (points <= high), axis=1)
    best, start = None, None
    for k, flag in enumerate(list(inside) + [False]):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            if best is None or k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    if best is None or best[1] - best[0] < min_points:
        return None
    return points[best[0]: best[1]]


def _in_pockets(points, pockets, margin) -> bool:
    for p in pockets:
        q = (points - np.asarray(p["center"])) / (np.asarray(p["radii"]) + margin)
        if np.any((q * q).sum(axis=1) < 1.0):
            return True
    return False


# -- the structure -----------------------------------------------------------------------------------------------


@dataclass
class Fiber:
    line: np.ndarray  # voxels, (n, 3)
    diameter: float  # voxels: the long width for ovals
    ratio: float  # thickness / width (1 = round)
    bend: float  # voxels: the bend limit
    type: int
    kind: str = "fiber"  # "fiber", "loop", "hairpin" or "broken"
    taper: float = 1.0  # the radius at the thin end over the full radius
    flip: bool = False  # which end is thin


def build_fibers(s: dict, crowd: float = 1.0, placement: int = 0) -> list[Fiber]:
    """The fibers of ``mixed_<index>`` as placed (before relaxation): ``crowd`` scales how full the box is."""
    rng = np.random.default_rng([s["seed"], 7, placement])
    side = float(SIDE)
    fibers: list[Fiber] = []
    target_total = s["total"] * crowd * side**3
    for ti, t in enumerate(s["types"]):
        d0 = t["diameter"]
        area = np.pi * (0.5 * d0) ** 2 * t["ratio"]
        target = t["share"] * target_total
        volume = 0.0
        layers = max(3, int(side / (4 * d0))) if s["orientation"] == "biaxial" else None
        tries = 0
        while volume < target and len(fibers) < MAX_FIBERS and tries < 4000:
            tries += 1
            d_mean = d0 * (float(np.exp(rng.normal(0.0, t["spread"]))) if t["spread"] else 1.0)
            d_mean = float(np.clip(d_mean, 0.75 * d0, 1.35 * d0))
            length = max(side * float(rng.uniform(*t["length"])), 3.0 * d_mean)  # thick short fibers: stubby rods
            # loops and hairpins in fibers thin enough to turn round within the volume (radius 1.2-4 diameters)
            kink = (t["loops"] > 0 and t["packing"] == "single" and t["shape"] in ("straight", "bent")
                    and d_mean <= 10.0 and rng.random() < t["loops"])
            axis, bend = _axis(rng, t, length, d_mean, kink, _reach(t, d_mean))
            group = _members(rng, t, axis, d_mean)
            if len(group) > 1:  # members off the axis (on helices, in a yarn) bend tighter than it
                bend = min(bend, 0.9 / max(max(max_curvature(m) for m in group), 1e-9))
            layer = int(rng.integers(0, layers)) if layers else None
            rot = _rotation(rng, _direction(rng, s, layer))
            placed = [m @ rot.T for m in group]
            low, high = np.min([p.min(0) for p in placed], axis=0), np.max([p.max(0) for p in placed], axis=0)
            margin = 0.5 * d_mean + 0.6
            for _ in range(20):
                center = rng.uniform(margin, side - margin, 3) - 0.5 * (low + high)
                if layers:
                    center[2] = (layer + 0.5) * side / layers + rng.normal(0.0, 0.1 * side / layers) - 0.5 * (low[2] + high[2])
                members = []
                for p in placed:
                    run = _inside_runs(p + center, margin, side - margin, 4)
                    if run is not None and _length(run) >= 2.0 * d_mean:
                        members.append(run)
                if not members or (s["pockets"] and any(_in_pockets(m, s["pockets"], margin) for m in members)):
                    continue
                for m in members:  # a group's members share its diameter
                    fibers.append(Fiber(m, d_mean, t["ratio"], bend, ti, "loop" if kink else "fiber", t["taper"],
                                        bool(rng.random() < 0.5)))
                    volume += np.pi * (0.5 * d_mean) ** 2 * t["ratio"] * _length(m)
                break
        if t["broken"] > 0:  # short broken pieces of this type, lying among the rest
            count = int(t["broken"] * sum(1 for f in fibers if f.type == ti)) + 2
            for _ in range(count):
                if len(fibers) >= MAX_FIBERS:
                    break
                d_mean = float(np.clip(d0 * (np.exp(rng.normal(0.0, t["spread"])) if t["spread"] else 1.0),
                                       0.75 * d0, 1.35 * d0))
                length = max(float(rng.uniform(1.5, 6.0)) * d_mean, 4.0)
                x = np.arange(-0.5 * length, 0.5 * length + 1e-9, max(0.15 * d_mean, 0.25))
                bow = rng.uniform(0.0, 0.05) * length * (1 - (2 * x / length) ** 2)
                piece = np.stack([x, bow, np.zeros_like(x)], axis=1)
                if s["orientation"] == "isotropic" or rng.random() < 0.5:
                    direction = _direction(rng, {**s, "orientation": "isotropic"}, None)
                else:
                    direction = _direction(rng, s, None)
                piece = piece @ _rotation(rng, direction).T
                margin = 0.5 * d_mean + 0.6
                for _ in range(20):
                    p = piece + rng.uniform(margin, side - margin, 3) - piece.mean(0)
                    if np.all((p >= margin) & (p <= side - margin)) and not (
                            s["pockets"] and _in_pockets(p, s["pockets"], margin)):
                        bend = min(40.0 * d_mean, 0.9 / max(max_curvature(p), 1e-9))  # stiff, as bent as it is
                        fibers.append(Fiber(p, d_mean, t["ratio"], bend, ti, "broken"))
                        break
    return fibers


def segment_points(f: Fiber) -> np.ndarray:
    """The fiber's polyline for Tangle: segments of 0.4-1.25 diameters, no longer than a third of its bend
    limit, at least four of them."""
    step = float(np.clip(min(0.33 * f.bend, 1.25 * f.diameter), 0.4 * f.diameter, 1.25 * f.diameter))
    length = _length(f.line)
    step = min(step, length / 4.0)
    return resample(f.line, step)


def _long_axes(line, ratio, layered: bool):
    """An oval's long axis at each node: across the fiber and, in layered scans, lying flat (in the xy plane)."""
    t = np.gradient(line, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    u = np.cross(t, ref) if layered else np.cross(t, np.array([0.3, -0.5, 0.81]))
    bad = np.linalg.norm(u, axis=1) < 1e-3
    u[bad] = np.cross(t[bad], np.array([1.0, 0.0, 0.0]))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def _collection(fibers: list[Fiber], lines, layered: bool, loosen: bool, axes=None):
    """One FiberCollection of every fiber (its own Material each), from ``lines`` (voxels); ovals' long axes
    ``axes`` (per node, as a relaxation left them) or, by default, lying across each fiber (flat in layers)."""
    import tangle
    from tangle.units import um

    collection = tangle.FiberCollection("mixed")
    for k, (f, line) in enumerate(zip(fibers, lines)):
        line = np.asarray(line, dtype=np.float64)
        # a fiber's polyline is its rest shape, which must keep within its bend limit; a relaxed line (rendering)
        # may have ended a little past the limit, so the limit is loosened to what it reached
        bend = min(0.99 * f.bend if loosen else f.bend, (0.9 if loosen else 0.97) / max(max_curvature(line), 1e-30))
        oval = {"thickness": f.ratio * f.diameter * um} if f.ratio < 1.0 else {}
        material = tangle.Material(f"m{k}", diameter=f.diameter * um, min_bend_radius=bend * um, **oval)
        if f.ratio < 1.0:
            long_axis = axes[k] if axes is not None else _long_axes(line, f.ratio, layered).tolist()
            collection.add_fiber((line * um).tolist(), material, long_axis=long_axis)
        else:
            collection.add_fiber((line * um).tolist(), material)
    return collection


def relax(fibers: list[Fiber], s: dict, cache_file: Path, steps: int | None = None) -> None:
    """Relax the placed fibers (Tangle, ``ct_examples.TRUTH_RELAXATION`` plus ``steps`` more) and cache them."""
    import time

    import tangle
    import ct_examples as ex
    from tangle.units import um

    cell = tangle.Cell([SIDE * um] * 3)
    layered = s["orientation"] in ("planar", "biaxial")
    recipe = tangle.Recipe(cell)
    recipe.insert(_collection(fibers, [segment_points(f) for f in fibers], layered, loosen=False), name="mixed")
    settings = dict(ex.TRUTH_RELAXATION)
    if steps:
        settings["max_iterations"] = settings.get("max_iterations", 12_000) + steps
    started = time.perf_counter()
    run = recipe.run(tangle.RelaxationSettings(backend=ex.BACKEND, max_step=1.0 * um, penetration_tolerance=0.1 * um,
                                               **settings))
    print(f"  truth relaxed: {run} ({time.perf_counter() - started:.0f} s)", flush=True)
    data = {"lines": [(np.asarray(line) / um).round(4).tolist() for line in run.centerlines()],
            "more_steps": steps or 0}
    if any(f.ratio < 1.0 for f in fibers):
        data["long_axes"] = run.assembly.long_axes()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data) + "\n")


def truth_from_cache(fibers: list[Fiber], s: dict, cache_file: Path):
    """The relaxed structure as a Tangle Assembly (each fiber its own material), or bonded (a RunResult)."""
    import tangle
    import ct_examples as ex
    from tangle.units import um

    data = json.loads(cache_file.read_text())
    cell = tangle.Cell([SIDE * um] * 3)
    layered = s["orientation"] in ("planar", "biaxial")
    collection = _collection(fibers, data["lines"], layered, loosen=True, axes=data.get("long_axes"))
    bonds = (s["binder"] or {}).get("bonds")
    if not bonds:
        assembly = tangle.Assembly(cell)
        assembly.insert(collection, name="mixed")
        return assembly
    recipe = tangle.Recipe(cell)
    recipe.insert(collection, name="mixed")
    recipe.capture_junctions(tangle.JunctionPolicy(
        "binder", "bond", max_surface_gap=bonds["gap"] * um, min_crossing_angle=0.0,
        probability=bonds["probability"], seed=s["seed"], max_per_fiber_pair=1,
        candidate_capacity=1 << 21))  # dense scans of finely segmented fibers touch in very many places
    return recipe.run(tangle.RelaxationSettings(backend=ex.BACKEND, max_iterations=300, max_step=0.5 * um,
                                                penetration_tolerance=0.1 * um))


def self_overlaps(fib, quarter: float = 0.5) -> int:
    """How many fibers pass through themselves (``fib``: ``overlaps.Fibers``): two stretches of one fiber,
    more than two diameters apart along it, closer than ``2 - quarter`` short semi-axes (about a quarter of its
    thickness inside itself at 0.5, as overlaps.QUARTER is for two fibers)."""
    count = 0
    for f in range(len(fib.a)):
        s, e = fib.start[f], fib.end[f]
        if e - s < 8:
            continue
        b = fib.b[f]
        pts = fib.P[s:e:2]
        pairs = cKDTree(pts).query_pairs((2.0 - quarter) * b, output_type="ndarray")
        if len(pairs) and np.any(np.abs(pairs[:, 0] - pairs[:, 1]) * 2 * 0.25 > 4.0 * b):
            count += 1
    return count


# -- the scan ----------------------------------------------------------------------------------------------------


@dataclass
class MixedScan:
    """A rendered mixed scan and its truth (voxels, x y z), as make_data stores it."""

    volume: np.ndarray
    labels: np.ndarray
    table: dict  # the point table (pos, tan, fid, rad, typ), per-point radius (thinning fibers)
    centerlines: list
    radii: np.ndarray
    types: np.ndarray
    semi_axes: np.ndarray | None
    bond_labels: np.ndarray | None
    bonds: list | None
    binder_occupancy: np.ndarray | None
    debris: np.ndarray  # dust voxels
    type_info: np.ndarray  # per type present: equal-area diameter, thickness / width, hollow
    flags: np.ndarray  # broken pieces, dust, voids present
    voxel_size: float = 1e-6
    extra: dict = field(default_factory=dict)


def _ellipsoid(rng, shape, center, radius, aspect=(0.5, 1.0)):
    """Soft occupancy (0-1) of a randomly turned ellipsoid, in a box around ``center`` (x, y, z voxels)."""
    axes = radius * np.array([1.0, rng.uniform(*aspect), rng.uniform(*aspect)])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    half = int(np.ceil(axes.max() + 2))
    c = np.floor(center).astype(int)
    lo = np.maximum(c - half, 0)
    hi = np.minimum(c + half + 1, np.array(shape[::-1]))
    if np.any(hi <= lo):
        return None, None
    z, y, x = np.mgrid[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    p = np.stack([x + 0.5, y + 0.5, z + 0.5], axis=-1) - center
    local = p @ q
    rho = np.sqrt(((local / axes) ** 2).sum(-1))
    occ = np.clip(0.5 - (rho - 1.0) * axes.min(), 0.0, 1.0)
    return (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0])), occ.astype(np.float32)


def _seams(lines, radius, labels, seams: dict, rng) -> np.ndarray:
    """Binder seams (one id each, 0 elsewhere): where two fibers lie side by side, surfaces within ``seams["gap"]``
    voxels and axes within ``seams["angle"]`` degrees, for at least two diameters, the groove between them is filled
    (void voxels whose distances to the two surfaces add up to at most ``seams["size"]`` x the thinner radius) all
    along the shared stretch; ``seams["share"]`` of such stretches get one."""
    from scipy.ndimage import distance_transform_edt

    shape = labels.shape
    out = np.zeros(shape, np.int32)
    pts, fib, tan = [], [], []
    for k, line in enumerate(lines):
        dense = resample(line, 1.0)
        if len(dense) < 3:
            continue
        t = np.gradient(dense, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        pts.append(dense), fib.append(np.full(len(dense), k)), tan.append(t)
    if len(pts) < 2:
        return out
    pts, fib, tan = np.concatenate(pts), np.concatenate(fib), np.concatenate(tan)
    reach = 2.0 * float(radius.max()) + seams["gap"]
    pairs = cKDTree(pts).query_pairs(reach, output_type="ndarray")
    if not len(pairs):
        return out
    a, b = pairs[:, 0], pairs[:, 1]
    fa, fb = fib[a], fib[b]
    gap = np.linalg.norm(pts[a] - pts[b], axis=1) - radius[fa] - radius[fb]
    keep = ((fa != fb) & (gap <= seams["gap"])
            & (np.abs((tan[a] * tan[b]).sum(1)) >= np.cos(np.radians(seams["angle"]))))
    a, b, fa, fb = a[keep], b[keep], fa[keep], fb[keep]
    swap = fa > fb
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    fa, fb = np.minimum(fa, fb), np.maximum(fa, fb)
    seam_id = 0
    for i, j in sorted(set(zip(fa.tolist(), fb.tolist()))):
        mine = np.unique(a[(fa == i) & (fb == j)])  # points of fiber i alongside fiber j, in order along i
        runs = np.split(mine, np.nonzero(np.diff(mine) > 2)[0] + 1)
        size = seams["size"] * float(min(radius[i], radius[j]))
        for run in runs:
            if len(run) < 4.0 * min(radius[i], radius[j]) or rng.random() >= seams["share"]:  # two diameters
                continue
            p = pts[run]
            pad = int(np.ceil(size + 2 * max(radius[i], radius[j]) + 3))
            lo = np.maximum(np.floor(p.min(0)).astype(int) - pad, 0)[::-1]  # (z, y, x)
            hi = np.minimum(np.floor(p.max(0)).astype(int) + pad + 1, np.array(shape[::-1]))[::-1]
            box = tuple(slice(l, h) for l, h in zip(lo, hi))
            lab = labels[box]
            if not ((lab == i + 1).any() and (lab == j + 1).any()):
                continue
            d_i = distance_transform_edt(lab != i + 1)
            d_j = distance_transform_edt(lab != j + 1)
            z, y, x = np.indices(lab.shape)
            local = np.stack([x + lo[2] + 0.5, y + lo[1] + 0.5, z + lo[0] + 0.5], axis=-1)
            near, _ = cKDTree(p).query(local.reshape(-1, 3), distance_upper_bound=float(radius[i] + radius[j] + size))
            region = (d_i + d_j <= size + 1.0) & (lab == 0) & np.isfinite(near).reshape(lab.shape)
            seam_id += 1
            out[box][region & (out[box] == 0)] = seam_id
    return out


def render(truth, fibers: list[Fiber], s: dict) -> MixedScan:
    """Scan the relaxed structure: shading (brightness, dim or hollow cores, thinning), binder, dust and edges,
    then the scan simulator (``tangle.ct._scanner``) with the scan's settings."""
    import tempfile
    from dataclasses import replace

    from scipy.ndimage import distance_transform_edt, gaussian_filter

    import tangle.ct as ct
    import ct_examples as ex
    from tangle.ct._scanner import acquire, motion_warp
    from tangle.ct._synthetic import _bond_truth, _shape_binder, read_vti
    from tangle.units import um

    rng = np.random.default_rng([s["seed"], 11])
    voxel = 1 * um
    bonds = (s["binder"] or {}).get("bonds")
    with tempfile.TemporaryDirectory() as tmp:
        report = truth.export_puma(Path(tmp) / "truth.puma", voxel, include_fiber_ids=True, include_interface=True,
                                   **({"bond_radius_ratio": bonds["radius_ratio"]} if bonds else {}))
        labels = read_vti(report.fiber_ids_path).astype(np.int32)
        occ = np.clip(read_vti(report.interface_path).astype(np.float32) / 255.0, 0.0, 1.0)
        bond_labels = binder_occ = None
        if bonds and report.bond_ids_path is not None:
            bond_labels = read_vti(report.bond_ids_path).astype(np.int32)
            binder_occ = np.clip(read_vti(report.binder_interface_path).astype(np.float32) / 255.0, 0.0, 1.0)
    shape = labels.shape
    lines = [np.asarray(line, dtype=np.float64) / voxel for line in truth.centerlines()]
    n = len(lines)
    diam = np.array([f.diameter for f in fibers])
    ratio = np.array([f.ratio for f in fibers])
    radius = 0.5 * diam * np.sqrt(ratio)  # equal-area
    ftype = np.array([f.type for f in fibers])
    tset = s["types"]

    # Shading: each occupied voxel's own fiber and its distance from that fiber's axis.
    pts, pfib, pfrac = [], [], []
    for k, line in enumerate(lines):
        dense = resample(line, 0.5)
        pts.append(dense)
        pfib.append(np.full(len(dense), k))
        pfrac.append(np.linspace(0.0, 1.0, len(dense)))
    pts, pfib, pfrac = np.concatenate(pts), np.concatenate(pfib), np.concatenate(pfrac)
    tree = cKDTree(pts)
    zz, yy, xx = np.nonzero(occ > 0)
    q = np.stack([xx, yy, zz], axis=1) + 0.5
    dist, idx = tree.query(q, k=8)
    idx = np.minimum(idx, len(pts) - 1)
    own = labels[zz, yy, xx] - 1
    nearest_fiber = pfib[idx]
    match = nearest_fiber == np.where(own >= 0, own, nearest_fiber[:, 0])[:, None]
    pick = np.argmax(match, axis=1)
    rows = np.arange(len(q))
    f = pfib[idx[rows, pick]]
    rho = dist[rows, pick]
    frac = pfrac[idx[rows, pick]]
    lost = np.nonzero(~match.any(axis=1))[0]  # labelled one fiber, nearer eight points of others: ask that fiber
    for k in np.unique(own[lost]):
        mine = lost[own[lost] == k]
        span = np.nonzero(pfib == k)[0]
        dk, ik = cKDTree(pts[span]).query(q[mine])
        f[mine], rho[mine], frac[mine] = k, dk, pfrac[span[ik]]
    spread = s["brightness_spread"]
    factor = np.exp(rng.normal(0.0, spread, n)) if spread else np.ones(n)
    bright = np.array([tset[t]["brightness"] for t in ftype]) * factor
    level = bright[f].astype(np.float64)
    hollow = np.array([tset[t]["hollow"] for t in ftype])
    lumen = np.array([tset[t]["lumen"] for t in ftype])
    rim = np.array([tset[t]["rim"] for t in ftype]) * diam
    core = np.array([tset[t]["core"] for t in ftype])
    inner_h = hollow[f] * 0.5 * diam[f]
    level = np.where((hollow[f] > 0) & (rho < inner_h), lumen[f], level)  # air (or nearly) inside
    inner_r = np.maximum(radius[f] - rim[f], 0.0)
    level = np.where((rim[f] > 0) & (rho < inner_r), core[f] * bright[f], level)
    taper = np.array([fb.taper for fb in fibers])
    flip = np.array([fb.flip for fb in fibers])
    local_r = radius[f] * (1.0 - (1.0 - taper[f]) * np.where(flip[f], 1.0 - frac, frac))
    thin = np.clip(local_r - rho + 0.5, 0.0, 1.0)
    shaded = np.zeros(shape, np.float32)
    raw = occ.copy()
    occ_t = occ[zz, yy, xx] * np.where(taper[f] < 1.0, thin, 1.0)
    raw[zz, yy, xx] = occ_t
    shaded[zz, yy, xx] = occ_t * level
    labels[zz, yy, xx] = np.where((taper[f] < 1.0) & (thin < 0.5) & (labels[zz, yy, xx] > 0), 0, labels[zz, yy, xx])
    dbeta = np.zeros(shape, np.float32)
    dbeta[zz, yy, xx] = np.array([tset[t]["delta_beta"] for t in ftype])[f]

    # Binder: bonds at junctions (the exporter's bridges, reshaped), and blobs stuck to fibers.
    b = s["binder"]
    if bonds:
        # bonds no bigger than BINDER_CAP: the exporter's bridges only where they stay that small, otherwise
        # fillets drawn here around the same junctions with the fibers' radii capped for the bond size
        shape_ = bonds["shape"]
        if shape_ == "bridge" and bonds["radius_ratio"] * radius.max() > BINDER_CAP:
            shape_ = "meniscus"
        binder = ct.Binder(radius_ratio=bonds["radius_ratio"], brightness=b["brightness"], delta_beta=b["delta_beta"],
                           shape=shape_)
        if bond_labels is None:
            bond_labels, binder_occ = np.zeros(shape, np.int32), np.zeros(shape, np.float32)
        elif binder.shape != "bridge":
            bond_labels, binder_occ = _shape_binder(bond_labels, binder_occ, labels,
                                                    np.minimum(radius, BINDER_CAP / bonds["radius_ratio"]), binder,
                                                    s["seed"])
        binder_occ = binder_occ * (labels == 0)
    if b and b.get("seams"):
        seam_labels = _seams(lines, radius, labels, b["seams"], rng)
        if seam_labels.any():
            if bond_labels is None:
                bond_labels, binder_occ = np.zeros(shape, np.int32), np.zeros(shape, np.float32)
            free = (bond_labels == 0) & (seam_labels > 0)
            bond_labels[free] = seam_labels[free] + bond_labels.max()
            binder_occ = np.maximum(binder_occ, np.clip(gaussian_filter((seam_labels > 0).astype(np.float32), 0.5),
                                                        0.0, 1.0) * (labels == 0))
    voids = bool(s["pockets"])
    if b and b.get("blobs"):
        blobs = b["blobs"]
        blob = np.zeros(shape, np.float32)
        bubble = np.zeros(shape, np.float32)
        weights = np.array([_length(line) for line in lines])
        for _ in range(blobs["count"]):
            k = int(rng.choice(n, p=weights / weights.sum()))
            line = lines[k]
            i = int(rng.integers(0, len(line)))
            t_ = np.gradient(line, axis=0)[i]
            t_ /= max(np.linalg.norm(t_), 1e-9)
            u = rng.normal(size=3)
            u -= (u @ t_) * t_
            u /= max(np.linalg.norm(u), 1e-9)
            size = float(np.clip(rng.uniform(*blobs["size"]) * radius[k], 1.5, BINDER_CAP))
            center = line[i] + u * (radius[k] + size * rng.uniform(-0.3, 0.4))
            box, e = _ellipsoid(rng, shape, center, size, aspect=(0.6, 1.0))
            if box is None:
                continue
            blob[box] = np.maximum(blob[box], e)
            if size >= 2.5 and rng.random() < blobs["bubbles"]:
                for _ in range(int(rng.integers(1, 3))):
                    bbox, be = _ellipsoid(rng, shape, center + rng.normal(0, 0.25 * size, 3),
                                          size * rng.uniform(0.25, 0.5), aspect=(0.8, 1.0))
                    if bbox is not None:
                        bubble[bbox] = np.maximum(bubble[bbox], be)
                        voids = True
        blob = np.clip(blob - bubble, 0.0, 1.0) * (labels == 0)
        binder_occ = blob if binder_occ is None else np.maximum(binder_occ, blob)
        if bond_labels is None:
            bond_labels = np.zeros(shape, np.int32)

    # Dust: grains in the free space (union of one to three ellipsoids each), a few dense.
    dust = s["dust"]
    dust_att = np.zeros(shape, np.float32)
    debris = np.zeros(shape, bool)
    if dust:
        solid = (labels > 0) | ((binder_occ if binder_occ is not None else 0) > 0.5)
        free = distance_transform_edt(~solid)
        for j in range(dust["count"] + dust["dense"]):
            dense = j >= dust["count"]
            r = (_log_uniform(rng, 0.7, 2.5) if dense else _log_uniform(rng, *dust["radius"]))
            for _ in range(60):
                z, y, x = (int(rng.integers(0, m)) for m in shape)
                if free[z, y, x] > r + 0.7:
                    break
            else:
                continue
            c = np.array([x, y, z]) + 0.5
            grain_b = _log_uniform(rng, 20.0, 150.0) if dense else float(rng.uniform(*dust["brightness"]))
            for _ in range(int(rng.integers(1, 4))):
                box, e = _ellipsoid(rng, shape, c + rng.normal(0, 0.3 * r, 3), r * rng.uniform(0.6, 1.0))
                if box is None:
                    continue
                e = e * ~solid[box]
                dust_att[box] = np.maximum(dust_att[box], e * grain_b)
                debris[box] |= e > 0.5

    # A cut face: the sample ends at a gently waving plane; material beyond it is gone (air).
    inside = np.ones(shape, np.float32)
    edge = s["edge"]
    zc, yc, xc = np.meshgrid(*(np.arange(m, dtype=np.float32) + 0.5 for m in shape), indexing="ij", sparse=True)
    if edge:
        nrm = np.asarray(edge["normal"])
        proj = nrm[0] * xc + nrm[1] * yc + nrm[2] * zc
        level_ = np.quantile(proj, 1.0 - edge["outside"])
        across = np.cross(nrm, [0.3, 0.5, 0.81])
        across /= np.linalg.norm(across)
        wave = edge["wave"] * np.sin(2 * np.pi * (across[0] * xc + across[1] * yc + across[2] * zc)
                                     / edge["wavelength"] + edge["phase"])
        inside = np.clip(level_ + wave - proj + 0.5, 0.0, 1.0).astype(np.float32)
        shaded *= inside
        raw *= inside
        dust_att *= inside
        if binder_occ is not None:
            binder_occ *= inside
        cut = inside < 0.5
        labels[cut] = 0
        debris[cut] = False
        if bond_labels is not None:
            bond_labels[cut] = 0

    # The sample: fibers, binder and dust on air (or resin), and its phase.
    sc = s["scanner"]
    base = ex.SCANNER
    fa = base.fiber_attenuation
    binder_b = b["brightness"] if b else 0.0
    binder_occ_ = binder_occ if binder_occ is not None else np.zeros(shape, np.float32)
    filled = np.clip(raw + binder_occ_ + np.minimum(dust_att, 1.0), 0.0, 1.0)
    void = base.void_attenuation + sc["matrix"] * fa
    sample = (void * (1.0 - filled) * inside + base.void_attenuation * (1.0 - inside)
              + fa * (shaded + binder_b * binder_occ_ + dust_att)).astype(np.float32)
    phase = (fa * (shaded * dbeta + binder_b * (b["delta_beta"] if b else 0.0) * binder_occ_
                   + dust_att * (dust["delta_beta"] if dust else 0.0))
             + sc["matrix"] * fa * sc["matrix_delta_beta"] * (1.0 - filled) * inside).astype(np.float32)
    width = int(np.ceil(np.hypot(shape[1], shape[2]))) + 4
    scanner = replace(base, photons=sc["photons"], noise_blur=sc["noise_blur"], resolution=sc["resolution"] * um,
                      propagation=sc["propagation"], fiber_motion=sc["fiber_motion"] * um, drift=sc["drift"] * um,
                      ring_strength=sc["ring_strength"], delta_beta=0.0,
                      angles=int(sc["angles"] * width) if sc["angles"] else None,
                      beam_hardening=sc["beam_hardening"], slice_drift=sc["slice_drift"])
    srng = np.random.default_rng(s["seed"])
    warp = motion_warp(labels, scanner, voxel, srng) if (scanner.fiber_motion > 0 or scanner.drift > 0) else None
    attenuation = acquire(sample, scanner, srng, voxel, phase, warp) / fa
    if sc["shading"]:
        ny, nx = shape[1:]
        attenuation += sc["shading"] * (np.cos(np.pi * (xc - nx / 2) / nx) * np.cos(np.pi * (yc - ny / 2) / ny))
    low, high = np.percentile(attenuation, [0.5, 99.5])
    volume = np.clip((attenuation - low) / (high - low) * 65535, 0, 65535).astype(np.uint16)

    # The field of view's edge: no data beyond it (zero, or the void's grey).
    fov = s["fov"]
    if fov:  # a circle about the rotation axis (z), its center well outside the volume
        far = 2.5 * SIDE
        cx, cy = 0.5 * SIDE - far * np.cos(fov["angle"]), 0.5 * SIDE - far * np.sin(fov["angle"])
        r2 = (xc[0] - cx) ** 2 + (yc[0] - cy) ** 2  # (y, x)
        out = np.broadcast_to(r2 > np.quantile(r2, 1.0 - fov["outside"]), shape)
        volume[out] = 0 if fov["fill"] == "zero" else np.uint16(np.median(volume[~out]))
        labels[out] = 0
        debris[out] = False
        if bond_labels is not None:
            bond_labels[out] = 0
        if binder_occ is not None:
            binder_occ[out] = 0.0
        inside = inside * ~out

    # The truth: each fiber's pieces still in the sample (and seen), per-point radii, and labels by piece.
    keep_inside = inside >= 0.5
    pieces, piece_of, piece_r, piece_t = [], [], [], []
    for k, line in enumerate(lines):
        dense = resample(line, 0.25)
        idx3 = np.clip(np.floor(dense).astype(int), 0, np.array(shape[::-1]) - 1)
        ok = keep_inside[idx3[:, 2], idx3[:, 1], idx3[:, 0]]
        frac_k = np.linspace(0.0, 1.0, len(dense))
        r_k = radius[k] * (1.0 - (1.0 - taper[k]) * (1.0 - frac_k if flip[k] else frac_k))
        start = None
        for i, flag in enumerate(list(ok) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= 2:
                    pieces.append(dense[start:i])
                    piece_of.append(k)
                    piece_r.append(r_k[start:i])
                    piece_t.append(fibers[k].type)
                start = None
    new_labels = np.zeros_like(labels)
    by_fiber: dict[int, list[int]] = {}
    for p, k in enumerate(piece_of):
        by_fiber.setdefault(k, []).append(p)
    for k, ps in by_fiber.items():
        mask = labels == k + 1
        if len(ps) == 1:
            new_labels[mask] = ps[0] + 1
            continue
        z, y, x = np.nonzero(mask)
        nodes = np.concatenate([pieces[p] for p in ps])
        owner = np.concatenate([np.full(len(pieces[p]), p) for p in ps])
        _, near = cKDTree(nodes).query(np.stack([x, y, z], axis=1) + 0.5)
        new_labels[z, y, x] = owner[near] + 1
    present = sorted(set(piece_t))
    remap = {t: i for i, t in enumerate(present)}
    table = {"pos": [], "tan": [], "fid": [], "rad": [], "typ": []}
    for p, (line, r_p) in enumerate(zip(pieces, piece_r)):
        tg = np.gradient(line, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
        table["pos"].append(line)
        table["tan"].append(tg)
        table["fid"].append(np.full(len(line), p + 1))
        table["rad"].append(r_p)
        table["typ"].append(np.full(len(line), remap[piece_t[p]]))
    if pieces:
        table = {"pos": np.concatenate(table["pos"]).astype(np.float32),
                 "tan": np.concatenate(table["tan"]).astype(np.float32),
                 "fid": np.concatenate(table["fid"]).astype(np.int32),
                 "rad": np.concatenate(table["rad"]).astype(np.float32),
                 "typ": np.concatenate(table["typ"]).astype(np.int8)}
    else:
        table = {"pos": np.zeros((0, 3), np.float32), "tan": np.zeros((0, 3), np.float32),
                 "fid": np.zeros(0, np.int32), "rad": np.zeros(0, np.float32), "typ": np.zeros(0, np.int8)}
    semi = None
    if np.any(ratio < 1.0):
        semi = np.array([[0.5 * diam[k], 0.5 * diam[k] * ratio[k]] for k in piece_of], np.float32).reshape(-1, 2)
    info = []
    for t in present:
        members = [k for k in set(piece_of) if fibers[k].type == t and fibers[k].kind != "broken"] or \
                  [k for k in set(piece_of) if fibers[k].type == t]
        info.append([float(np.median(2 * radius[members])), tset[t]["ratio"], float(tset[t]["hollow"] > 0)])
    broken = any(fibers[k].kind == "broken" for k in piece_of)
    bond_truth = _bond_truth(bond_labels, new_labels) if bond_labels is not None else (None, None)
    return MixedScan(
        volume=volume, labels=new_labels, table=table, centerlines=pieces,
        radii=np.array([radius[k] for k in piece_of]), types=np.array([remap[t] for t in piece_t], int),
        semi_axes=semi, bond_labels=bond_truth[0], bonds=bond_truth[1], binder_occupancy=binder_occ,
        debris=debris, type_info=np.array(info, np.float32).reshape(-1, 3),
        flags=np.array([broken, bool(debris.any()), voids], bool),
        extra={"bond_ratio": bonds["radius_ratio"] if bonds else None})


class FibersOverlap(Exception):
    """The relaxed truth still has fibers passing through each other or themselves."""


def check(truth) -> dict:
    """overlaps.summary of the truth, plus how many fibers pass through themselves."""
    import overlaps
    from tangle.units import um

    fib = overlaps.from_structure(truth, 1 * um)
    report = overlaps.summary(fib, overlaps.overlapping_pairs(fib))
    report["self"] = self_overlaps(fib)
    return report


def mixed_scan(index: int, cache: Path, check_overlaps: bool = True, settings=mixed_settings):
    """Relax (cached), check and scan ``mixed_<index>`` (or, with ``settings=bends_settings``, ``bends_<index>``);
    returns (MixedScan, settings with the check's report)."""
    s = settings(index)
    for placement in range(PLACEMENTS if check_overlaps else 1):
        for crowd in (1.0, 0.6, 0.35):
            fibers = build_fibers(s, crowd, placement)
            if not fibers:
                raise ValueError("no fibers placed")
            # keyed on the placed fibers themselves, so a cache never pairs one placement's relaxed lines with
            # another's fibers (as when the placement code changes between runs)
            placed = hashlib.sha1()
            for f in fibers:
                placed.update(np.round(f.line, 3).tobytes() + np.float64([f.diameter, f.ratio, f.bend]).tobytes())
            key = hashlib.sha1(f"{placed.hexdigest()}{crowd}{placement}".encode()).hexdigest()[:8]
            cache_file = cache / f"mixed_{index}-{key}.json"
            try:
                if not cache_file.exists():
                    relax(fibers, s, cache_file)
                truth = truth_from_cache(fibers, s, cache_file)
            except Exception as error:
                if "could not place" in str(error) or "capacity" in str(error):
                    continue
                raise
            report = None
            if check_overlaps:
                report = check(truth)
                for steps in RELAX_MORE:
                    if not (report["quarter"] or report["self"]):
                        break
                    relax(fibers, s, cache_file, steps)
                    truth = truth_from_cache(fibers, s, cache_file)
                    report = check(truth)
                if report["quarter"] or report["self"]:
                    print(f"mixed_{index}: placement {placement}: {report['quarter']} pairs and {report['self']} "
                          f"fibers through themselves past a quarter", flush=True)
                    break
            scan = render(truth, fibers, s)
            binder = 0.0 if scan.binder_occupancy is None else round(float((scan.binder_occupancy > 0.5).mean()), 4)
            out = {**s, "crowd": crowd, **({"placement": placement} if placement else {}), "binder_share": binder,
                   **({"truth_check": report} if report else {}),
                   "fibers": {kind: sum(1 for f in fibers if f.kind == kind) for kind in ("fiber", "loop", "broken")}}
            return scan, out
    raise FibersOverlap(f"fibers still pass through each other after {PLACEMENTS} placements")
