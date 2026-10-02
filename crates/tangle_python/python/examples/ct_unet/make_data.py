"""Simulated CT scans and their truth, for training the CT map network.

Each volume is one ``ct_examples`` structure (``dense_hard`` or ``scanned``
settings) at an index well past the test sets, so the seeds never meet the
structures the fitter is scored on (``dense_hard_1``-``36``).

Per volume, ``<name>.npz`` holds the scan (uint16, ``(z, y, x)``), the true
labels, and for every voxel within ``REACH`` voxels of an axis the index of
the nearest point on a true centerline (-1 elsewhere). The point table holds
each point's position (x, y, z voxels), unit tangent, fiber id, radius (the
equivalent radius; ovals also their semi-axes) and type, so the training
targets (axis heatmap, offset to the axis, direction, type) are gathers at
load time.

usage: python make_data.py OUT FIRST COUNT [--scanned-share 0.2] [--varied | --pairs | --hard-pairs | --mixed]
    [--blur-cap 0.4]

``--varied`` draws every structure from ``varied_settings``: 1-4 fiber types of any size, shape and brightness,
any orientation, and a wide range of scanner settings (see there).

``--mixed`` makes ``mixed_<index>`` scans (``mixed.py``): every kind of variety at random inside each scan
(fiber shapes, packings, sections, binder, dust, broken pieces, voids, sample edges, scanner artifacts). Their
volumes also hold ``type_info`` (per fiber type: equal-area diameter, thickness / width, hollow), ``hint_flags``
(broken pieces, dust, voids present) and ``debris_mask`` (dust voxels), for the network's hints.
Indices: training from 100001, validation 99001-99032, test 99501-99548.

``--bends`` makes ``bends_<index>`` scans (``mixed.bends_settings``): two fiber types of one material and size,
one straight and one bendable, so only bending tells them apart. Indices: training from 200001, validation
199001-199032, test 199501-199548.

``--blur-cap F`` keeps every scan's blur at most F times its thinnest fiber type's diameter (``cap_blur``), in any
family, to leave out scans blurrier than the scanners the network is meant for. (On the test scans the network
finds fibers much worse past about half the thinnest diameter: F1 about 0.7, against 0.88 or better up to 0.4.)
Without the option every family draws the blur it always has, so earlier sets can be made again exactly.
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REACH = 12.0  # voxels: past every coarse fiber's long semi-axis


def point_table(scan) -> dict:
    """Every true centerline resampled to 0.25 voxels, with tangent, fiber, radius and type."""
    from tangle.ct._geometry import resample

    pos, tan, fid, rad, typ = [], [], [], [], []
    for index, line in enumerate(scan.centerlines):
        dense = resample(np.asarray(line, dtype=np.float64), 0.25)
        if len(dense) < 2:
            continue
        t = np.gradient(dense, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        pos.append(dense)
        tan.append(t)
        fid.append(np.full(len(dense), index + 1))
        rad.append(np.full(len(dense), scan.radii[index]))
        typ.append(np.full(len(dense), 0 if scan.types is None else scan.types[index]))
    return {
        "pos": np.concatenate(pos).astype(np.float32),
        "tan": np.concatenate(tan).astype(np.float32),
        "fid": np.concatenate(fid).astype(np.int32),
        "rad": np.concatenate(rad).astype(np.float32),
        "typ": np.concatenate(typ).astype(np.int8),
    }


def nearest_points(shape, pos: np.ndarray) -> np.ndarray:
    """For each voxel center, the index of the nearest point within REACH (-1 beyond)."""
    tree = cKDTree(pos)
    nz, ny, nx = shape
    out = np.full(shape, -1, dtype=np.int32)
    xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="xy")
    for z in range(nz):
        pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z + 0.5)], axis=1)
        d, i = tree.query(pts, distance_upper_bound=REACH)
        i[~np.isfinite(d)] = -1
        out[z] = i.reshape(ny, nx)
    return out


VARIED_SIDE = 160  # voxels (1 um); everything below is in voxels, the physical scale does not matter to the network


FOCUS_FIRST = 4001  # indices from here on use the focused draw (varied_settings(focus=True))
BOND_FIRST = 5001  # indices from here on may have binder bonds at their fiber junctions
SHAPES_FIRST = 7001  # ... and from here on bonds of several shapes (bridge, meniscus, blob) and coatings
STIFF_FIRST = 9001  # ... and from here on stiff, nearly straight fibers, bonded at their tight crossings
WEB_FIRST = 11001  # ... and from here on big binder webs and fillets (no coatings), some flat fibers, milder noise
EASY_FIRST = 12001  # ... and from here on the same, made clean (an easy start for training on webs)
EASY_MAX_BINDER = 0.04  # easy scans with more binder than this share of the volume are skipped
MIXED_MAX_BINDER = 0.15  # ... and mixed scans with more than this (dense, nearly all crossings bonded with big bonds)
PAIRS_FIRST = 20001  # pairs_<index> (--pairs): dense_hard scans whose fine fibers lie in touching pairs
MOTION_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))  # cap_blur: fiber motion (RMS) as a Gaussian blur's width
HARD_PAIRS_FIRST = 21001  # hardpairs_<index> (--hard-pairs): the same made hard to split (see hard_pairs_settings)
# The true structure must not have fibers passing through each other (overlaps.py): while a pair still shares more
# than a quarter of the thinner fiber's thickness, relax further (steps, in turn), then try a new placement.
RELAX_MORE = (3000, 9000, 15000)
PLACEMENTS = 3


def varied_settings(index: int, focus: bool | None = None) -> dict:
    """A random structure and scanner for ``varied_<index>`` (lengths in voxels = um at 1 um voxels).

    1-4 fiber types. Diameters are log-uniform in 3.5-28 voxels, and neighbouring types differ by at least 30%:
    ``synthetic_ct`` gives each fiber the profile of the type whose diameter is nearest its own, so two types of
    nearly the same size cannot be rendered as different materials. Each type has its own brightness (the
    brightest 1), oval flattening, dim core, bundling, bend limit and phase ratio.

    ``focus`` (default: index >= FOCUS_FIRST) weights the draw toward what the first round found hard: the thinnest
    type 3.5-6 voxels 70% of the time, more bundles, more random 3D orientations, and a blur no wider than the
    thinnest fiber.
    """
    rng = np.random.default_rng(30_000 + index)
    focus = index >= FOCUS_FIRST if focus is None else focus
    k = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]))
    while True:
        d = np.sort(np.exp(rng.uniform(np.log(3.5), np.log(28.0), k)))
        if focus and rng.random() < 0.7:  # the focused round: the thinnest type 3.5-6 voxels most of the time
            d[0] = float(np.exp(rng.uniform(np.log(3.5), np.log(6.0))))
            d = np.sort(d)
        if k == 1 or np.all(d[1:] / d[:-1] >= 1.3):
            break
    total = float(rng.uniform(0.05, 0.25))
    share = rng.dirichlet(np.ones(k))
    bright = rng.uniform(0.4, 1.0, k)
    bright /= bright.max()
    types = []
    for t in range(k):
        types.append({
            "diameter": round(float(d[t]), 2),
            "fraction": round(float(total * share[t]), 4),
            "brightness": round(float(bright[t]), 3),
            "ratio": round(float(rng.uniform(0.55, 0.9)), 2) if rng.random() < 0.35 else 1.0,
            "rim": round(float(rng.uniform(0.15, 0.3) * d[t]), 2) if (d[t] > 8 and rng.random() < 0.35) else None,
            "core": round(float(rng.uniform(0.4, 0.8)), 2),
            "bundle": (int(rng.choice([1, 7, 19], p=[0.4, 0.35, 0.25] if focus else [0.6, 0.25, 0.15]))
                       if d[t] < 12 else 1),
            "bend": round(float(rng.uniform(3.0, 8.0)), 2),
            "delta_beta": round(float(rng.uniform(4.0, 16.0)), 1),
        })
    bonds = None
    if index >= BOND_FIRST and rng.random() < 0.65:  # a third of the bond round stays bond-free
        bonds = {
            "probability": round(float(rng.uniform(0.3, 1.0)), 2),  # of the touching crossings
            "gap": round(float(rng.uniform(0.3, 1.5)), 2),  # voxels: surfaces this close count as touching
            "radius_ratio": round(float(rng.uniform(0.5, 1.5)), 2),
            "brightness": round(float(rng.choice([rng.uniform(0.3, 0.8), rng.uniform(0.8, 1.2)])), 2),
            "delta_beta": round(float(rng.uniform(4.0, 16.0)), 1),
        }
        if index >= SHAPES_FIRST:
            bonds["shape"] = str(rng.choice(["bridge", "meniscus", "meniscus", "blob"]))
            bonds["coating"] = round(float(rng.uniform(0.1, 0.6)), 2) if rng.random() < 0.4 else 0.0
            bonds["coating_thickness"] = round(float(rng.uniform(0.7, 2.0)), 2)
    elif index >= SHAPES_FIRST and rng.random() < 0.3:  # some bond-free scans still carry a coating
        bonds = {"probability": 0.0, "gap": 0.5, "radius_ratio": 1.0,
                 "delta_beta": round(float(rng.uniform(4.0, 16.0)), 1),
                 "brightness": round(float(rng.uniform(0.3, 1.2)), 2), "shape": "bridge",
                 "coating": round(float(rng.uniform(0.1, 0.6)), 2),
                 "coating_thickness": round(float(rng.uniform(0.7, 2.0)), 2)}
    orientation = str(rng.choice(["planar", "planar", "aligned", "biaxial", "isotropic"]
                                 + (["isotropic"] if focus else [])))
    if index >= STIFF_FIRST:
        # stiff round: bend limits of 40-400 diameters and little waviness, most scans bonded at tight crossings
        # (surfaces within 0.2-0.8 voxel), fibers mostly in layers
        rs = np.random.default_rng(90_000 + index)  # a separate stream: the draws above stay as they were
        for t in types:
            t["bend"] = round(float(np.exp(rs.uniform(np.log(40.0), np.log(400.0)))), 1)
        if bonds is not None or rs.random() < 0.85:
            base = bonds or {"delta_beta": round(float(rs.uniform(4.0, 16.0)), 1), "shape": "bridge", "coating": 0.0,
                             "coating_thickness": 1.0, "radius_ratio": 1.0,
                             "brightness": round(float(rs.uniform(0.3, 1.2)), 2)}
            bonds = {**base, "probability": round(float(rs.uniform(0.6, 1.0)), 2),
                     "gap": round(float(rs.uniform(0.2, 0.8)), 2),
                     "radius_ratio": round(float(rs.uniform(0.5, 1.5)), 2),
                     "shape": str(rs.choice(["bridge", "meniscus", "meniscus", "blob"]))}
        orientation = str(rs.choice(["planar", "planar", "biaxial", "biaxial", "aligned", "isotropic"]))
    if index >= WEB_FIRST:
        # web round: binder as fillets and webs (meniscus shape) that also span fibers a few voxels apart (a wide capture
        # gap and a bond size of 1-2.2 fiber radii), never coatings; about half the types flat (width ratio
        # 0.3-0.6); binder about as bright as the fibers
        rw = np.random.default_rng(110_000 + index)
        for t in types:
            if rw.random() < 0.5:
                t["ratio"] = round(float(rw.uniform(0.3, 0.6)), 2)
        if bonds is not None:
            bonds.update(probability=round(float(rw.uniform(0.5, 1.0)), 2),
                         gap=round(float(rw.uniform(0.5, 2.5)), 2),
                         radius_ratio=round(float(rw.uniform(1.0, 2.2)), 2),
                         brightness=round(float(rw.uniform(0.7, 1.1)), 2),
                         shape="meniscus",
                         coating=0.0)
    blur = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
    settings = {
        "types": types,
        "orientation": orientation,
        "tilt": round(float(rng.uniform(0.1, 0.5)), 2),
        "length_fraction": [round(float(rng.uniform(0.35, 0.6)), 2), round(float(rng.uniform(0.7, 0.98)), 2)],
        "curvature": ([round(float(rng.uniform(0.05, 0.2)), 2), round(float(rng.uniform(0.3, 0.8)), 2)]
                      if index < STIFF_FIRST else [0.0, round(float(rng.uniform(0.02, 0.15)), 2)]),
        "brightness_spread": round(float(rng.uniform(0.0, 0.4)), 2),
        "photons": round(1300 * float(rng.uniform(0.5, 2.0)) / max(1.0, 23 * blur**2)
                         * (2.0 if index >= WEB_FIRST else 1.0)),
        "noise_blur": blur,
        # the focused round caps the blur (FWHM, um = voxels) at the thinnest fiber's diameter: blurrier scans
        # merge touching thin fibers beyond what any method can undo
        "resolution": round(float(min(rng.uniform(1.2, 4.0), d[0])) if focus else float(rng.uniform(1.2, 4.0)), 2),
        "propagation": round(float(rng.choice([0.0, rng.uniform(1.0, 8.0)])), 2),
        "fiber_motion": round(float(rng.uniform(0.0, 2.0)), 2),
        "drift": round(float(rng.uniform(0.0, 1.0)), 2) if rng.random() < 0.3 else 0.0,
        "ring_strength": round(float(rng.uniform(0.0, 0.01)), 4) if rng.random() < 0.2 else 0.0,
        "seed": 30_000 + index,
        **({"bonds": bonds} if index >= BOND_FIRST else {}),
    }
    if index >= EASY_FIRST:
        # easy web round: the web round made clean, to start training on: no blur, drift or rings, at least
        # 1000 photons, a resolution no wider than half the thinnest fiber, smaller and fewer webs (binder a few
        # percent of the volume)
        re_ = np.random.default_rng(120_000 + index)
        settings.update(noise_blur=0.0, drift=0.0, ring_strength=0.0, fiber_motion=round(float(re_.uniform(0.0, 0.5)), 2),
                        photons=max(int(settings["photons"]), int(re_.integers(1000, 4000))),
                        resolution=round(float(min(settings["resolution"], max(1.0, 0.5 * float(d[0])))), 2))
        if settings.get("bonds") and settings["bonds"].get("probability", 0) > 0:
            settings["bonds"].update(radius_ratio=round(float(re_.uniform(0.8, 1.5)), 2),
                                     probability=round(float(re_.uniform(0.4, 0.8)), 2),
                                     gap=round(float(re_.uniform(0.3, 1.5)), 2))
    return settings


def pairs_settings(index: int) -> dict:
    """``pairs_<index>``: ``dense_hard_settings`` with the fine fibers in touching side-by-side pairs (bundles of
    two, a hair apart, in the fibers' plane) in place of bundles of 7 or 19."""
    import ct_examples as ex

    return {**ex.dense_hard_settings(index), "per_bundle": 2}


def hard_pairs_settings(index: int) -> dict:
    """``hardpairs_<index>``: ``pairs_settings`` made hard to split: fine fibers 3.5-6 voxels across (so more of
    them), a scanner blur from half to all of their diameter, 40-100% of the photons and more fiber motion."""
    import ct_examples as ex

    rng = np.random.default_rng(210_000 + index)
    fine = round(float(rng.uniform(3.5, 6.0)), 2)
    return {**pairs_settings(index), "fine_um": fine,
            "resolution_um": round(float(rng.uniform(0.5, 1.0) * fine), 2),
            "photons": round(ex.SCANNER.photons * float(rng.uniform(0.4, 1.0))),
            "fiber_motion_um": round(float(rng.uniform(0.5, 1.5)), 2)}


def _max_curvature(line) -> float:
    """The largest turning angle per unit length along a polyline (1 / its tightest bend radius)."""
    p = np.asarray(line, float)
    if len(p) < 3:
        return 0.0
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1)
    cos = np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]) / np.maximum(n[:-1] * n[1:], 1e-30), -1.0, 1.0)
    return float(np.max(np.arccos(cos) / np.maximum(0.5 * (n[:-1] + n[1:]), 1e-30)))


def cached_recipe(cache_file: Path, cell, populations):
    """A recipe holding the cached relaxed structure (and the cache's contents)."""
    import tangle
    import ct_examples as ex

    data = json.loads(cache_file.read_text())
    recipe = tangle.Recipe(cell)
    start = 0
    for population, count in zip(populations, data["counts"]):
        lines = data["centerlines"][start:start + count]
        axes = data["long_axes"][start:start + count] if "long_axes" in data else None
        start += count
        material = ex._material(population)
        bend = 0.99 * material.min_bend_radius
        if bend:  # stiff fibers can end their relaxation a little past their limit: loosen it to what they reached
            bend = min(bend, 0.9 / max(max(_max_curvature(line) for line in lines), 1e-30)) if lines else bend
        looser = tangle.Material(material.name, diameter=material.diameter, min_bend_radius=bend,
                                 thickness=material.thickness)
        collection = tangle.FiberCollection(material.name)
        for k, line in enumerate(lines):
            if material.is_oval and axes is not None:
                collection.add_fiber(line, looser, long_axis=axes[k])
            else:
                collection.add_fiber(line, looser)
        recipe.insert(collection, name=material.name)
    return recipe, data


def with_bonds(cache_file: Path, cell, populations, bonds: dict, seed: int):
    """The cached relaxed structure again, with its touching crossings captured as bonded junctions and a short
    relaxation to settle them (a RunResult that export_puma can draw bonds for)."""
    import tangle
    import ct_examples as ex
    from tangle.units import um

    recipe, _ = cached_recipe(cache_file, cell, populations)
    recipe.capture_junctions(tangle.JunctionPolicy(
        "binder", "bond", max_surface_gap=bonds["gap"] * um, min_crossing_angle=0.0,
        probability=bonds["probability"], seed=seed, max_per_fiber_pair=1))
    return recipe.run(tangle.RelaxationSettings(backend=ex.BACKEND, max_iterations=300, max_step=0.5 * um,
                                                penetration_tolerance=0.1 * um))


class FibersOverlap(Exception):
    """The relaxed truth still has fibers passing through each other."""


def relax_further(cache_file: Path, cell, populations, steps: int) -> None:
    """Relax the structure again from its placement for ``steps`` more than last time, and cache the result.

    Not from the cache: rebuilt from relaxed centerlines, each fiber's bent shape would be its rest shape, so its
    bend limit has to be loosened to what the worst fiber reached (``cached_recipe``), and over thousands of
    steps every fiber then bends that far. Starting over repeats the earlier steps, but every fiber keeps its own
    rest shape and bend limit.
    """
    import ct_examples as ex

    more = json.loads(cache_file.read_text()).get("more_steps", 0) + steps
    saved = dict(ex.TRUTH_RELAXATION)
    ex.TRUTH_RELAXATION["max_iterations"] = saved.get("max_iterations", 12_000) + more
    try:
        cache_file.unlink()
        ex.relaxed_truth(cache_file, cell, populations)
    finally:
        ex.TRUTH_RELAXATION.clear()
        ex.TRUTH_RELAXATION.update(saved)
    data = json.loads(cache_file.read_text())
    data["more_steps"] = more
    cache_file.write_text(json.dumps(data) + "\n")


def overlap_report(truth) -> dict:
    """``overlaps.summary`` of a true structure (lengths in 1 um voxels)."""
    import overlaps
    from tangle.units import um

    fibers = overlaps.from_structure(truth, 1 * um)
    return overlaps.summary(fibers, overlaps.overlapping_pairs(fibers))


def cap_blur(settings: dict, cap: float) -> dict:
    """``settings`` (of any family) with the scan's blur cut back to at most ``cap`` times its thinnest fiber
    type's diameter. The blur is the scanner's (``resolution``, a full width at half maximum) and the fibers' motion
    (``fiber_motion``, an RMS displacement, counted as a Gaussian blur of that sigma: width ``MOTION_FWHM`` times
    it) together, added in quadrature. Where they come to more than the cap, both shrink by the same factor, so the
    scan keeps its mix of the two; the rest of the draw stays as it was. The result records ``blur_cap``."""
    s = copy.deepcopy(settings)
    if "scanner" in s:  # mixed and bends
        where, resolution_key, motion_key = s["scanner"], "resolution", "fiber_motion"
        thinnest = min(float(t["diameter"]) for t in s["types"])
    elif "resolution_um" in s:  # the ct_examples families: dense_hard, scanned, pairs, hard pairs
        where, resolution_key, motion_key = s, "resolution_um", "fiber_motion_um"
        thinnest = min(float(s[k]) for k in ("fine_um", "coarse_um") if s.get(k))
    else:  # varied
        where, resolution_key, motion_key = s, "resolution", "fiber_motion"
        thinnest = min(float(t["diameter"]) for t in s["types"])
    resolution, motion = float(where.get(resolution_key) or 0.0), float(where.get(motion_key) or 0.0)
    blur = math.hypot(resolution, MOTION_FWHM * motion)
    if blur > cap * thinnest:
        shrink = cap * thinnest / blur
        where[resolution_key] = round(resolution * shrink, 3)
        where[motion_key] = round(motion * shrink, 3)
    s["blur_cap"] = cap
    return s


def varied_scan(index: int, cache: Path, check: bool = True, blur_cap: float | None = None):
    """Relax (cached) and scan ``varied_<index>``; returns (SyntheticScan, settings). ``blur_cap``: see
    ``cap_blur``."""
    import hashlib
    from dataclasses import replace

    import tangle
    import tangle.ct as ct
    import ct_examples as ex
    from tangle.units import um

    v = varied_settings(index)
    if blur_cap:
        v = cap_blur(v, blur_cap)
    for placement in range(PLACEMENTS if check else 1):  # fibers still through each other: place them anew
        for crowd in (1.0, 0.6, 0.35):  # too crowded to place every fiber: fewer fibers, same everything else
            try:
                return _varied_scan(v, index, cache, crowd, placement, check)
            except FibersOverlap as error:
                print(f"varied_{index}: placement {placement}: {error}", flush=True)
                break
            except Exception as error:
                if "could not place" not in str(error) or crowd == 0.35:
                    raise
    raise FibersOverlap(f"fibers still pass through each other after {PLACEMENTS} placements")


def _varied_scan(v, index, cache, crowd, placement=0, check=True):
    import hashlib
    from dataclasses import replace

    import tangle
    import tangle.ct as ct
    import ct_examples as ex
    from tangle.units import um

    side = VARIED_SIDE * um
    cell = tangle.Cell([side] * 3)
    length = tuple(f * side for f in v["length_fraction"])
    populations, profiles = [], []
    for t, spec in enumerate(v["types"]):
        diameter = spec["diameter"] * um
        oval = {"thickness": spec["ratio"] * diameter} if spec["ratio"] < 1.0 else {}
        material = tangle.Material(f"type{t}_{spec['diameter']:g}", diameter=diameter,
                                   min_bend_radius=spec["bend"] * diameter, **oval)
        mean = 0.5 * sum(length)
        area = np.pi * (diameter / 2) ** 2 * (spec["ratio"] if oval else 1.0)
        count = int(np.clip(round(spec["fraction"] * side**3 / (area * mean)), 3, 350))
        orientation = {
            "planar": lambda: tangle.PlanarOrientation(max_tilt=v["tilt"]),
            "aligned": lambda: tangle.AlignedOrientation("x", max_angle=v["tilt"]),
            "biaxial": lambda: tangle.LayeredBiaxialOrientation(max_tilt=v["tilt"], seed=v["seed"]),
            "isotropic": lambda: tangle.IsotropicOrientation(),
        }[v["orientation"]]()
        position = (tangle.LayeredPosition(max(3, int(side / (4 * diameter))), jitter_fraction=0.25)
                    if v["orientation"] == "biaxial" else tangle.UniformPosition())
        population = tangle.FiberPopulation(
            material=material, count=max(int(count * crowd) // spec["bundle"], 1),
            segments_per_fiber=max(4, int(length[0] / (1.25 * diameter))), seed=v["seed"] + t + 1000 * placement,
            length=length,
            # thick fibers in a 160-voxel box cannot wave by a large share of their diameter within their bend
            # limit; a retry (crowd < 1) also halves the waviness
            curvature_amplitude=tuple(c * diameter * min(1.0, 8.0 / spec["diameter"]) * (1.0 if crowd == 1.0 else 0.5)
                                      for c in v["curvature"]),
            orientation=orientation, position=position,
        )
        populations.append(ex.bundles(cell, population, spec["bundle"]) if spec["bundle"] > 1 else population)
        shade = {"rim": spec["rim"] * um, "core": spec["core"]} if spec["rim"] else {}
        profiles.append((diameter, ct.CrossSection(brightness=spec["brightness"], **shade)))
    tried = {**v, "crowd": crowd, **({"placement": placement} if placement else {})}
    key = hashlib.sha1(json.dumps(tried, sort_keys=True).encode()).hexdigest()[:8]
    cache_file = cache / f"varied_{index}-{key}.json"

    def final_truth():
        truth = ex.relaxed_truth(cache_file, cell, populations)
        return with_bonds(cache_file, cell, populations, v["bonds"], v["seed"]) if v.get("bonds") else truth

    truth = final_truth()
    report = None
    if check:
        report = overlap_report(truth)
        for steps in RELAX_MORE:
            if not report["quarter"]:
                break
            relax_further(cache_file, cell, populations, steps)
            truth = final_truth()
            report = overlap_report(truth)
        if report["quarter"]:
            raise FibersOverlap(f"{report['quarter']} fiber pairs past a quarter after "
                                f"{json.loads(cache_file.read_text()).get('more_steps', 0)} more steps")
        report["more_steps"] = json.loads(cache_file.read_text()).get("more_steps", 0)
    binder = None
    if v.get("bonds"):
        b = v["bonds"]
        binder = ct.Binder(radius_ratio=b["radius_ratio"], brightness=b["brightness"], delta_beta=b["delta_beta"],
                           shape=b.get("shape", "bridge"), coating=b.get("coating", 0.0),
                           coating_thickness=b.get("coating_thickness", 1.0))
    v = {**tried, **({"truth_check": report} if report else {})}
    scanner = replace(ex.SCANNER, photons=v["photons"], noise_blur=v["noise_blur"], resolution=v["resolution"] * um,
                      propagation=v["propagation"], fiber_motion=v["fiber_motion"] * um, drift=v["drift"] * um,
                      ring_strength=v["ring_strength"],
                      delta_beta=tuple(spec["delta_beta"] for spec in v["types"]))
    scan = ct.synthetic_ct(truth, 1 * um, seed=v["seed"], profiles=profiles, scanner=scanner, binder=binder,
                           **({"brightness_spread": v["brightness_spread"]} if v["brightness_spread"] else {}))
    return scan, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("first", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("--scanned-share", type=float, default=0.2)
    parser.add_argument("--varied", action="store_true")
    parser.add_argument("--fast-relax", action="store_true",
                        help="relax the truth structures for at most 3000 steps with a tighter neighbor skin (about "
                             "8x faster, the same structures to within a few percent); for training sets, not tests")
    parser.add_argument("--pairs", action="store_true",
                        help="dense_hard scans with the fine fibers in touching pairs (pairs_<index>, from 20001)")
    parser.add_argument("--hard-pairs", action="store_true",
                        help="pairs made hard to split: thinner fibers, more blur, fewer photons (hardpairs_<index>)")
    parser.add_argument("--mixed", action="store_true",
                        help="every kind of variety at random inside each scan (mixed_<index>; see mixed.py)")
    parser.add_argument("--bends", action="store_true",
                        help="two types of one material and size, one straight, one bendable (bends_<index>)")
    parser.add_argument("--no-overlap-check", action="store_true",
                        help="keep structures whose fibers pass through each other (as every set made before had them)")
    parser.add_argument("--blur-cap", type=float, default=None,
                        help="keep each scan's blur (scanner and fiber motion together) at most this many times its "
                             "thinnest fiber type's diameter (see cap_blur); off by default")
    args = parser.parse_args()

    def capped(settings):  # a family's settings with the blur cap, when one is asked for
        return (lambda index: cap_blur(settings(index), args.blur_cap)) if args.blur_cap else settings

    if args.fast_relax:
        import ct_examples as ex
        ex.TRUTH_RELAXATION = {"max_iterations": 3000, "neighbor_skin_scale": 0.5, "neighbor_capacity": 192}
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import ct_examples as ex

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / ".cache"
    for index in range(args.first, args.first + args.count):
        # Which settings family is a fixed function of the index, so reruns agree.
        family = ("varied" if args.varied else "pairs" if args.pairs else "hardpairs" if args.hard_pairs
                  else "mixed" if args.mixed else "bends" if args.bends
                  else "scanned" if np.random.default_rng(index).random() < args.scanned_share else "dense_hard")
        name = f"{family}_{index}"
        target = args.out / f"{name}.npz"
        if target.exists():
            continue
        started = time.perf_counter()
        try:
            if family == "varied":
                scan, varied = varied_scan(index, cache, check=not args.no_overlap_check, blur_cap=args.blur_cap)
            elif family in ("mixed", "bends"):
                import mixed

                scan, varied = mixed.mixed_scan(index, cache, check_overlaps=not args.no_overlap_check,
                                                settings=capped(mixed.bends_settings if family == "bends" else
                                                                mixed.mixed_settings))
            else:
                settings = {"dense_hard": ex.dense_hard_settings, "pairs": pairs_settings,
                            "hardpairs": hard_pairs_settings}.get(family, ex.scanned_settings)
                example = ex.scanned(index, capped(settings))(cache / f"{name}.json")
                scan, varied = example.scan, None
                if not args.no_overlap_check:  # no relaxing further here: a structure that fails is left out
                    import overlaps

                    fibers = overlaps.from_scan(scan)
                    report = overlaps.summary(fibers, overlaps.overlapping_pairs(fibers))
                    if report["quarter"]:
                        raise FibersOverlap(f"{report['quarter']} fiber pairs past a quarter")
                    varied = {"family": family, "truth_check": report}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # a structure that fails to relax or render (a Tangle panic too): skip it
            print(f"{name}: failed ({error})", flush=True)
            continue
        most = EASY_MAX_BINDER if family == "varied" and index >= EASY_FIRST else (
            MIXED_MAX_BINDER if family in ("mixed", "bends") else None)
        if most is not None and scan.binder_occupancy is not None and float((scan.binder_occupancy > 0.5).mean()) > most:
            print(f"{name}: skipped (binder {float((scan.binder_occupancy > 0.5).mean()):.1%} > {most:.0%})", flush=True)
            continue
        made_here = family in ("mixed", "bends")  # rendered by mixed.py: its own point table (thinning fibers' radii)
        table = scan.table if made_here else point_table(scan)
        if not len(table["pos"]):  # e.g. a sparse scan whose fibers all lie past a cut face: nothing to learn from
            print(f"{name}: skipped (no fibers in the volume)", flush=True)
            continue
        near = nearest_points(scan.volume.shape, table["pos"])
        extra = {}
        if made_here:
            extra.update(type_info=scan.type_info, hint_flags=scan.flags, debris_mask=scan.debris)
        if scan.bond_labels is not None:
            extra["bond_labels"] = scan.bond_labels.astype(np.uint16)
            ratio = scan.extra.get("bond_ratio") if made_here else varied["bonds"]["radius_ratio"]
            if ratio:
                extra["bond_ratio"] = np.float32(ratio)
            if scan.binder_occupancy is not None:
                extra["binder_mask"] = scan.binder_occupancy > 0.5
            extra["bond_pairs"] = np.array([b["fibers"] + [0] * (2 - len(b["fibers"])) for b in scan.bonds]
                                           or np.zeros((0, 2)), dtype=np.int32).reshape(-1, 2)
            extra["bond_centers"] = np.array([b["center"] for b in scan.bonds] or np.zeros((0, 3)),
                                             dtype=np.float32).reshape(-1, 3)
        if scan.semi_axes is not None:
            extra["semi_axes"] = np.asarray(scan.semi_axes, dtype=np.float32)
        np.savez_compressed(
            target, volume=scan.volume.astype(np.uint16), labels=scan.labels.astype(np.uint16), near=near,
            **{f"p_{k}": v for k, v in table.items()}, **extra,
        )
        (args.out / f"{name}.json").write_text(json.dumps({
            "voxel_um": scan.voxel_size * 1e6, "fibers": len(scan.centerlines),
            "types": None if scan.types is None else np.bincount(scan.types).tolist(),
            "settings": varied,
        }) + "\n")
        print(f"{name}: {len(scan.centerlines)} fibers, {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
