"""Training scans at the edge of a reconstruction's field of view (``edges_<index>``).

A real scan is reconstructed inside a circle about the rotation axis (the field of view: the detector's width);
outside it there is no data, and the export fills it with one value, most often 0 (black). Tiles of a real scan
that hold part of that circle confused the network: a black region beside the grey, a bright or dark rim at the
circle, rings bent round the axis. ``empty.py`` already taught it that black holds no fibers; these scans give it
the edge itself as a reconstruction makes it, with three kinds:

* ``black``: the whole volume beyond the circle (all fill).
* ``blank``: the circle crossing the volume, empty space (air, or the scan's resin) inside, no fibers.
* ``fibers``: the circle crossing a ``mixed`` scan (``mixed.py``): fibers in the grey part only.

The edge comes from the physics of truncated filtered back-projection, about the true rotation axis (somewhere
outside the volume, the circle 120-4000 voxels in radius: from a tight arc to a nearly straight cut):

* Truncation. The sample is usually wider than the field of view (local tomography): rays through the sample
  outside the circle are cut off at the detector's edge, and the ramp filter turns that into a smooth error
  that grows towards the circle (a bright rim or a dark cupping, depending on how the reconstruction pads the
  projections: zeros, the edge value, or a taper). Sometimes the sample is a little narrower than the circle,
  so a ring of air shows between the sample's edge and the circle. The error is worked out exactly for a
  round sample of the scan's mean attenuation (``truncation_profile``: the sinogram of a disc, cut to the
  detector, filtered as the scan simulator filters, back-projected) and added to the scan simulator's own
  reconstruction of the volume, whose noise and fiber edges stay as they were.
* Rings about the true axis (a detector column's gain error is a ring at its radius), and sometimes a thin
  bright or dark ring right at the circle (the detector's outermost columns).
* The fill: 0 (most often), a dark constant above 0, the empty-space grey, or no mask at all (the
  reconstruction runs on past the circle as a fading halo with thinner noise).

Labels beyond the circle are background whatever the fill. Each scan stores ``grey`` (empty space, bright
fiber) as ``empty.py``'s do: the greys a whole scan of this sample would be scaled by (its empty space taken
from the inside of the circle, away from the rim), so training scales an edge tile as ``find_fibers`` scales
each tile of a real scan by the whole scan's greys, with the fill left out (``scan_io.check_grey``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mixed import SIDE, MixedScan, _log_uniform, mixed_settings

FIRST = 400_001  # training scans from here up
VAL = range(399_001, 399_033)
TEST = range(399_501, 399_549)
KINDS = ("black", "blank", "fibers")


def edges_settings(index: int) -> dict:
    """Everything random about ``edges_<index>``: the mixed family's draw for the index (fibers, binder, dust,
    scanner), with the field of view's edge from a stream of its own. The mixed family's own field-of-view
    edge is off, and so are its rings (they would be about the volume's middle, not the true axis)."""
    s = mixed_settings(index)
    rng = np.random.default_rng([s["seed"], 31])
    kind = str(rng.choice(KINDS, p=[0.15, 0.35, 0.5]))
    radius = _log_uniform(rng, 120.0, 4000.0)
    beyond = kind == "fibers" or rng.random() < 0.75  # the sample wider than the field of view
    field = {
        "kind": kind,
        "radius": round(radius, 1),
        "angle": round(float(rng.uniform(0, 2 * np.pi)), 3),  # where the axis lies, seen from the volume
        "outside": 1.0 if kind == "black" else round(float(rng.uniform(0.15, 0.85)), 3),  # share of a slice
        "sample": round(float(rng.uniform(1.05, 3.0) if beyond else rng.uniform(0.85, 0.98)), 3),  # x radius
        # the mean attenuation of the sample outside the volume (x a fiber's), over the scan's empty space;
        # fibers scans also take their own volume's mean into it
        "density": round(float(rng.uniform(0.03, 0.3)), 3),
        "padding": str(rng.choice(["zero", "edge", "taper"], p=[0.25, 0.5, 0.25])),
        "taper": round(float(rng.uniform(0.1, 0.4)), 3),  # x radius, for the taper
        "fill": str(rng.choice(["zero", "dark", "void", "unmasked"], p=[0.6, 0.15, 0.1, 0.15]) if kind != "black"
                    else rng.choice(["zero", "dark", "void"], p=[0.75, 0.15, 0.1])),
        "dark": round(float(rng.uniform(0.1, 0.8)), 3),  # the dark fill, as a share of the way up to empty space
        "rings": round(_log_uniform(rng, 0.01, 0.12), 4) if rng.random() < 0.5 else 0.0,  # x fiber contrast
        "rim": ({"offset": round(float(rng.uniform(0.0, 4.0)), 2), "width": round(float(rng.uniform(0.7, 2.5)), 2),
                 "height": round(float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.6)), 3)}
                if rng.random() < 0.3 else None),
        # greys as empty.py's: zero this far below empty space (x fiber contrast), headroom above a bright fiber,
        # and for a quarter of the scans empty space a little off the scan's empty-space grey
        "floor": round(_log_uniform(rng, 0.2, 1.5), 3),
        "headroom": round(float(rng.uniform(0.3, 2.0)), 3),
        "offset": round(float(rng.uniform(-0.3, 0.15)), 3) if rng.random() < 0.25 else 0.0,
    }
    sc = {**s["scanner"], "ring_strength": 0.0}
    return {**s, "scanner": sc, "fov": None, "field": field}


def axis_position(field: dict) -> tuple[float, float]:
    """The rotation axis (x, y, voxels) that puts ``field["outside"]`` of each slice beyond the circle."""
    radius = field["radius"]
    c = np.arange(SIDE) + 0.5
    x, y = np.meshgrid(c, c, indexing="xy")
    ux, uy = np.cos(field["angle"]), np.sin(field["angle"])
    half = 0.5 * np.sqrt(2.0) * SIDE

    def axis(d):
        return 0.5 * SIDE - d * ux, 0.5 * SIDE - d * uy

    def share(d):
        cx, cy = axis(d)
        return float(np.mean(np.hypot(x - cx, y - cy) > radius))

    if field["outside"] >= 1.0:
        return axis(radius + half + 2.0)
    lo, hi = radius - half - 2.0, radius + half + 2.0  # share 0 at lo, 1 at hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if share(mid) < field["outside"] else (lo, mid)
    return axis(0.5 * (lo + hi))


def truncation_profile(radius: float, sample: float, density: float, padding: str, taper: float,
                       r: np.ndarray, angles: int = 2048) -> np.ndarray:
    """The reconstruction at distances ``r`` from the axis of a round sample (radius ``sample``, attenuation
    ``density`` per voxel) scanned with a detector reaching ``radius`` from the axis: its sinogram (the same at
    every angle) cut to the detector, padded (``zero``, ``edge`` value or a cosine ``taper`` over ``taper``
    times the radius), filtered by the scan simulator's Shepp-Logan ramp and back-projected. Beyond the radius
    no ray was measured; the back-projection there is the fading halo of an unmasked reconstruction."""
    from tangle.ct._scanner import _filter

    half = int(np.ceil(radius))
    u = np.arange(-half, half + 1, dtype=np.float64)
    p = 2.0 * density * np.sqrt(np.clip(sample**2 - u**2, 0.0, None))
    # Pad well past the detector, so the filter sees the padding and not a wrap-around.
    pad = 3 * half + 16
    line = np.zeros(len(u) + 2 * pad)
    line[pad:pad + len(u)] = p
    if padding != "zero":
        reach = np.arange(1, pad + 1, dtype=np.float64)
        roll = np.ones(pad) if padding == "edge" else np.where(
            reach < taper * radius, 0.5 * (1.0 + np.cos(np.pi * reach / max(taper * radius, 1.0))), 0.0)
        line[pad + len(u):] = p[-1] * roll
        line[:pad] = p[0] * roll[::-1]
    q = _filter(line[None].astype(np.float32))[0].astype(np.float64)[pad:pad + len(u)]  # the measured columns
    theta = (np.arange(angles) + 0.5) * (np.pi / angles)
    out = np.empty(len(r))
    for i in range(0, len(r), 256):
        s_ = np.asarray(r[i:i + 256], np.float64)[:, None] * np.cos(theta)[None]
        seen = np.abs(s_) <= radius
        out[i:i + 256] = (np.interp(s_, u, q) * seen).sum(axis=1) * (np.pi / angles)
    return out


def _radial(field: dict, density: float, r: np.ndarray, rng: np.random.Generator):
    """The reconstruction's error at distances ``r`` (truncation, rings, rim), on a fine radial grid, and the
    error's typical value inside the circle away from the rim (which the scan's empty-space grey includes)."""
    radius, sample = field["radius"], field["radius"] * field["sample"]
    lo, hi = max(float(r.min()) - 2.0, 0.0), float(r.max()) + 2.0
    grid = np.arange(lo, hi + 0.25, 0.25)
    truth = np.where(grid < sample, density, 0.0)
    error = truncation_profile(radius, sample, density, field["padding"], field["taper"], grid) - truth
    # Inside, away from the rim: area-weighted median of the error over 0-0.85 of the radius.
    inner = np.linspace(0.0, 0.85 * radius, 400)
    inner_error = truncation_profile(radius, sample, density, field["padding"], field["taper"], inner) - \
        np.where(inner < sample, density, 0.0)
    order = np.argsort(inner_error)
    weight = np.cumsum((inner + 1.0)[order])
    reference = float(inner_error[order][np.searchsorted(weight, 0.5 * weight[-1])])
    if field["rings"]:  # one gain error per detector column: a ring at each whole-voxel radius
        from scipy.ndimage import gaussian_filter1d

        columns = gaussian_filter1d(rng.standard_normal(int(hi) + 3), 0.6)
        columns *= field["rings"] / max(float(columns.std()), 1e-9)
        error = error + np.interp(grid, np.arange(len(columns)), columns)
    if field["rim"]:
        rim = field["rim"]
        error = error + rim["height"] * np.exp(-0.5 * ((grid - (radius - rim["offset"])) / rim["width"]) ** 2)
    return grid, error, reference


def finish(attenuation: np.ndarray, s: dict, empty_space: np.ndarray, density: float, rng: np.random.Generator):
    """The scan of ``s`` from the scan simulator's reconstruction ``attenuation`` (``(z, y, x)``, in fiber
    attenuations): the field of view's errors added, the greys set, the outside filled. ``empty_space`` marks
    the voxels with nothing in them (for the empty-space grey and the noise). Returns (uint16 volume, the
    outside as a (z, y, x) mask, the greys (empty space, bright fiber) in the volume's units)."""
    from scipy.ndimage import gaussian_filter

    f = s["field"]
    shape = attenuation.shape
    cx, cy = axis_position(f)
    c = np.arange(SIDE) + 0.5
    x, y = np.meshgrid(c, c, indexing="xy")
    r = np.hypot(x - cx, y - cy)  # (y, x)
    outside = r > f["radius"]
    grid, error, reference = _radial(f, density, r, rng)
    att = attenuation + np.interp(r, grid, error)[None].astype(np.float32)

    space = attenuation[empty_space & ~outside[None]] if (empty_space & ~outside[None]).sum() > 1000 \
        else attenuation[empty_space] if empty_space.sum() > 1000 else attenuation.ravel()
    level = float(np.median(space))  # the empty space's grey, without the field of view's errors
    noise = float(np.std(space))
    if f["fill"] == "unmasked":  # past the circle: the halo, with noise thinned by the share of angles seen
        seen = 1.0 - (2.0 / np.pi) * np.arccos(np.clip(f["radius"] / np.maximum(r, 1e-9), 0.0, 1.0))
        speckle = gaussian_filter(rng.standard_normal(shape).astype(np.float32), (0.7, 0.9, 0.9))
        speckle *= noise / max(float(speckle.std()), 1e-9)
        halo = level + np.interp(r, grid, error)
        att = np.where(outside[None], (halo + 0.0)[None] + np.sqrt(seen)[None] * speckle, att).astype(np.float32)

    # Greys as empty.py's: in fiber contrasts above the scan's empty space, zero at -floor.
    empty = level + reference - f["offset"]
    inside_values = att[:, ~outside] if (~outside).any() else att.reshape(len(att), -1)
    low_q, high_q = np.quantile(inside_values, [1e-4, 1.0 - 1e-4]) if inside_values.size else (empty, empty + 1)
    floor = max(f["floor"], 3.5 * noise, empty - float(low_q) + 0.05)
    headroom = max(f["headroom"], float(high_q) - empty - 1.0)
    to_grey = 65535.0 / (floor + 1.0 + headroom)

    def grey(a):
        return (np.asarray(a, np.float64) - empty + floor) * to_grey

    volume = np.clip(grey(att), 1, 65535).astype(np.uint16)  # 0 is the fill's alone
    mask = np.broadcast_to(outside[None], shape)
    if f["fill"] == "zero":
        volume[mask] = 0
    elif f["fill"] == "dark":
        volume[mask] = np.uint16(max(1.0, f["dark"] * floor * to_grey))
    elif f["fill"] == "void":
        volume[mask] = np.uint16(np.clip(grey(empty), 1, 65535))
    return volume, mask, np.array([grey(empty), grey(empty + 1.0)], np.float32)


def blank_scan(index: int, s: dict | None = None) -> tuple[MixedScan, dict]:
    """Scan a ``black`` or ``blank`` ``edges_<index>``: empty space (air, or the scan's resin) through the scan
    simulator, then the field of view's edge (``finish``)."""
    import ct_examples as ex
    from tangle.ct._scanner import acquire
    from tangle.units import um

    s = s or edges_settings(index)
    sc = s["scanner"]
    base = ex.SCANNER
    fa = base.fiber_attenuation
    shape = (SIDE,) * 3
    sample = np.full(shape, base.void_attenuation + sc["matrix"] * fa, np.float32)
    width = int(np.ceil(np.hypot(SIDE, SIDE))) + 4
    scanner = replace(base, photons=sc["photons"], noise_blur=sc["noise_blur"], resolution=sc["resolution"] * um,
                      propagation=sc["propagation"], ring_strength=0.0, delta_beta=0.0,
                      angles=int(sc["angles"] * width) if sc["angles"] else None,
                      beam_hardening=sc["beam_hardening"], slice_drift=sc["slice_drift"])
    rng = np.random.default_rng(s["seed"])
    attenuation = acquire(sample, scanner, rng, 1 * um) / fa
    if sc["shading"]:
        y, x = np.meshgrid(np.arange(SIDE) + 0.5, np.arange(SIDE) + 0.5, indexing="ij")
        attenuation += sc["shading"] * (np.cos(np.pi * (x - SIDE / 2) / SIDE) * np.cos(np.pi * (y - SIDE / 2) / SIDE))
    density = s["field"]["density"] + sc["matrix"]
    volume, outside, grey = finish(attenuation, s, np.ones(shape, bool), density, np.random.default_rng([s["seed"], 32]))
    table = {"pos": np.zeros((0, 3), np.float32), "tan": np.zeros((0, 3), np.float32),
             "fid": np.zeros(0, np.int32), "rad": np.zeros(0, np.float32), "typ": np.zeros(0, np.int8)}
    scan = MixedScan(
        volume=volume, labels=np.zeros(shape, np.int32), table=table, centerlines=[], radii=np.zeros(0),
        types=np.zeros(0, int), semi_axes=None, bond_labels=None, bonds=None, binder_occupancy=None,
        debris=np.zeros(shape, bool),
        # the hints still give fiber types: a user says what the fibers are even where a tile has none
        type_info=np.array([[t["diameter"], t["ratio"], float(t["hollow"] > 0)] for t in s["types"]],
                           np.float32).reshape(-1, 3),
        flags=np.zeros(3, bool), extra={"grey": grey})
    return scan, {**s, "outside_share": round(float(outside.mean()), 4)}


def edges_scan(index: int, cache, check_overlaps: bool = True) -> tuple[MixedScan, dict]:
    """Scan ``edges_<index>``: fiber-free (``blank_scan``) or a mixed scan with the edge (``mixed.render``)."""
    s = edges_settings(index)
    if s["field"]["kind"] != "fibers":
        return blank_scan(index, s)
    import mixed

    return mixed.mixed_scan(index, cache, check_overlaps=check_overlaps, settings=edges_settings)
