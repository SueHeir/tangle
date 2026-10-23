"""Fiber-free training scans: heavy background noise, and pure black regions (``empty_<index>``).

The network should find nothing in either. Real scans have both: stretches of empty space whose noise is
strong against the fibers' contrast, and black (zero) regions where there is no data at all (beyond the
reconstructed field of view, missing slices at the ends of a stack, masked-out parts). Nothing in the earlier
families taught the network that these hold no fibers.

Two kinds, chosen from the index:

* ``noise``: empty space only (air or resin) with every scanner flaw the mixed family draws (blotchy noise,
  rings, slice drift, beam hardening, shading), its noise then scaled to 0.3-1.0 of the bright-fiber contrast
  (standard deviation; the mixed scans' empty space has about 0.1-0.3).
* ``black``: the same empty space with 0.1-0.6 noise, and one or two black regions in it: the whole volume, a
  field-of-view circle, a flat cut, slabs of missing slices at the top or bottom, or boxes.

The noise is the scan simulator's own (filtered back-projection texture), scaled up after reconstruction rather
than made with fewer photons: below a few tens of photons per ray the counts hit zero and the noise stops being
CT-like.

Grey levels: make_data normally scales each scan by its own median and 99.5th percentile (``maps.levels``). A
scan with nothing in it has no fiber grey, so that would blow its noise up to fiber contrast (or, all black,
divide by zero). These scans store ``grey``: the empty-space and bright-fiber greys a scan of this sample with
fibers in it would have, and training scales them by those, as ``find_fibers`` scales each tile of a real scan
by the whole scan's greys. The hints still give fiber types (drawn as the mixed family draws them): a user
says what the fibers are even for a part of the scan that has none, so a hint must never mean "fibers here".
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mixed import SIDE, MixedScan, _log_uniform, mixed_settings

FIRST = 300_001  # training scans from here up
VAL = range(299_001, 299_033)
TEST = range(299_501, 299_549)
KINDS = ("all", "fov", "cut", "slabs", "boxes")  # black region shapes


def empty_settings(index: int) -> dict:
    """Everything random about ``empty_<index>``: the mixed family's draw for the index (types, resin, scanner),
    with the noise level and the black regions from a stream of its own."""
    s = mixed_settings(index)
    rng = np.random.default_rng([s["seed"], 21])
    kind = "noise" if rng.random() < 0.5 else "black"
    # the noise's standard deviation, in bright-fiber contrasts
    noise = _log_uniform(rng, 0.3, 1.0) if kind == "noise" else _log_uniform(rng, 0.1, 0.6)
    sc = dict(s["scanner"])
    if rng.random() < 0.4:  # more blotchy, correlated noise than the mixed family's blur choices give
        sc["noise_blur"] = round(float(rng.uniform(0.5, 1.5)), 2)
    black = []
    if kind == "black":
        for shape in rng.choice(KINDS, size=int(rng.choice([1, 2], p=[0.7, 0.3])), replace=False,
                                p=[0.08, 0.3, 0.27, 0.2, 0.15]):
            region = {"shape": str(shape)}
            if shape == "fov":  # a circle about z: from a gentle arc (far center) to a tight one
                region.update(angle=round(float(rng.uniform(0, 2 * np.pi)), 3),
                              far=round(float(_log_uniform(rng, 0.6, 4.0)), 2),
                              outside=round(float(rng.uniform(0.1, 0.9)), 3))
            elif shape == "cut":  # a flat face at any angle
                normal = rng.normal(size=3)
                region.update(normal=(normal / np.linalg.norm(normal)).round(4).tolist(),
                              outside=round(float(rng.uniform(0.1, 0.9)), 3))
            elif shape == "slabs":  # missing slices at the top, the bottom, or both
                ends = str(rng.choice(["top", "bottom", "both"]))
                region.update(ends=ends, slices=[int(rng.integers(5, 70)) for _ in range(2 if ends == "both" else 1)])
            elif shape == "boxes":
                region["boxes"] = [[[int(rng.integers(0, SIDE - 20)) for _ in range(3)],
                                    [int(rng.integers(20, 100)) for _ in range(3)]]
                                   for _ in range(int(rng.integers(1, 5)))]
            black.append(region)
    # Where the black sits below empty space, in bright-fiber contrasts (exports put zero 0.2-1.5 below it, and
    # never closer than 3.5 noise deviations, so the noise is not cut off into black specks); how
    # far the bright fiber sits from the top of the grey range; and, for a quarter of the scans, empty space a
    # little off the scan's empty-space grey (air beside resin, a lighter patch).
    return {**s, "kind": kind, "scanner": sc, "noise": round(float(noise), 3), "black": black,
            "floor": round(max(float(_log_uniform(rng, 0.2, 1.5)), 3.5 * noise), 3),
            "headroom": round(float(rng.uniform(0.3, 2.0)), 3),
            "offset": round(float(rng.uniform(-0.4, 0.15)), 3) if rng.random() < 0.25 else 0.0}


def black_mask(s: dict) -> np.ndarray:
    """The voxels of ``s``'s black regions, (z, y, x)."""
    shape = (SIDE,) * 3
    z, y, x = np.meshgrid(*(np.arange(SIDE) + 0.5 for _ in range(3)), indexing="ij")
    out = np.zeros(shape, bool)
    for region in s["black"]:
        kind = region["shape"]
        if kind == "all":
            out[:] = True
        elif kind == "fov":
            far = region["far"] * SIDE
            cx = 0.5 * SIDE - far * np.cos(region["angle"])
            cy = 0.5 * SIDE - far * np.sin(region["angle"])
            r2 = (x[0] - cx) ** 2 + (y[0] - cy) ** 2
            out |= (r2 > np.quantile(r2, 1.0 - region["outside"]))[None]
        elif kind == "cut":
            n = np.asarray(region["normal"])
            proj = n[0] * x + n[1] * y + n[2] * z
            out |= proj > np.quantile(proj[::4, ::4, ::4], 1.0 - region["outside"])
        elif kind == "slabs":
            ends = ["top", "bottom"] if region["ends"] == "both" else [region["ends"]]
            for end, count in zip(ends, region["slices"]):
                out[SIDE - count:] = True if end == "top" else out[SIDE - count:]
                out[:count] = True if end == "bottom" else out[:count]
        elif kind == "boxes":
            for low, size in region["boxes"]:  # (z, y, x) corners and sizes
                out[tuple(slice(lo, lo + n) for lo, n in zip(low, size))] = True
    return out


def empty_scan(index: int) -> tuple[MixedScan, dict]:
    """Scan ``empty_<index>``: empty space (air, or the scan's resin) through the scan simulator, black regions
    set to zero; returns (MixedScan, settings) with the scan's reference greys in ``extra["grey"]``."""
    import ct_examples as ex
    from tangle.ct._scanner import acquire
    from tangle.units import um

    s = empty_settings(index)
    sc = s["scanner"]
    base = ex.SCANNER
    fa = base.fiber_attenuation
    shape = (SIDE,) * 3
    void = base.void_attenuation + sc["matrix"] * fa
    sample = np.full(shape, void, np.float32)
    width = int(np.ceil(np.hypot(SIDE, SIDE))) + 4
    scanner = replace(base, photons=sc["photons"], noise_blur=sc["noise_blur"], resolution=sc["resolution"] * um,
                      propagation=sc["propagation"], ring_strength=sc["ring_strength"], delta_beta=0.0,
                      angles=int(sc["angles"] * width) if sc["angles"] else None,
                      beam_hardening=sc["beam_hardening"], slice_drift=sc["slice_drift"])
    attenuation = acquire(sample, scanner, np.random.default_rng(s["seed"]), 1 * um) / fa
    level = float(attenuation.mean())  # the empty space's reconstructed attenuation, in fiber units
    attenuation = level + (attenuation - level) * (s["noise"] / max(float(attenuation.std()), 1e-9))
    if sc["shading"]:
        y, x = np.meshgrid(np.arange(SIDE) + 0.5, np.arange(SIDE) + 0.5, indexing="ij")
        attenuation += sc["shading"] * (np.cos(np.pi * (x - SIDE / 2) / SIDE) * np.cos(np.pi * (y - SIDE / 2) / SIDE))

    # Greys in bright-fiber contrasts above the scan's empty space (which a fiber-filled scan's median would be):
    # zero at -floor, the bright fiber at 1, the top of the range headroom above that.
    empty = level - s["offset"]  # the empty-space grey the scan's greys are taken from
    span = s["floor"] + 1.0 + max(s["headroom"], 3.5 * s["noise"] - 1.0)
    to_grey = 65535.0 / span

    def grey(a):
        return (np.asarray(a, np.float64) - empty + s["floor"]) * to_grey

    volume = np.clip(grey(attenuation), 0, 65535).astype(np.uint16)
    black = black_mask(s)
    volume[black] = 0
    labels = np.zeros(shape, np.int32)
    table = {"pos": np.zeros((0, 3), np.float32), "tan": np.zeros((0, 3), np.float32),
             "fid": np.zeros(0, np.int32), "rad": np.zeros(0, np.float32), "typ": np.zeros(0, np.int8)}
    types = s["types"]
    scan = MixedScan(
        volume=volume, labels=labels, table=table, centerlines=[], radii=np.zeros(0), types=np.zeros(0, int),
        semi_axes=None, bond_labels=None, bonds=None, binder_occupancy=None, debris=np.zeros(shape, bool),
        type_info=np.array([[t["diameter"], t["ratio"], float(t["hollow"] > 0)] for t in types],
                           np.float32).reshape(-1, 3),
        flags=np.zeros(3, bool),
        extra={"grey": np.array([float(grey(empty)), float(grey(empty + 1.0))], np.float32)})
    clipped = float(((volume == 0) & ~black).mean() + (volume == 65535).mean())
    return scan, {**s, "black_share": round(float(black.mean()), 4), "clipped": round(clipped, 5)}
