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

usage: python make_data.py OUT FIRST COUNT [--scanned-share 0.2] [--varied]

``--varied`` draws every structure from ``varied_settings``: 1-4 fiber types of any size, shape and brightness,
any orientation, and a wide range of scanner settings (see there).
"""

import argparse
import json
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
    blur = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
    return {
        "types": types,
        "orientation": orientation,
        "tilt": round(float(rng.uniform(0.1, 0.5)), 2),
        "length_fraction": [round(float(rng.uniform(0.35, 0.6)), 2), round(float(rng.uniform(0.7, 0.98)), 2)],
        "curvature": [round(float(rng.uniform(0.05, 0.2)), 2), round(float(rng.uniform(0.3, 0.8)), 2)],
        "brightness_spread": round(float(rng.uniform(0.0, 0.4)), 2),
        "photons": round(1300 * float(rng.uniform(0.5, 2.0)) / max(1.0, 23 * blur**2)),
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


def with_bonds(cache_file: Path, cell, populations, bonds: dict, seed: int):
    """The cached relaxed structure again, with its touching crossings captured as bonded junctions and a short
    relaxation to settle them (a RunResult that export_puma can draw bonds for)."""
    import tangle
    import ct_examples as ex
    from tangle.units import um

    data = json.loads(cache_file.read_text())
    recipe = tangle.Recipe(cell)
    start = 0
    for population, count in zip(populations, data["counts"]):
        lines = data["centerlines"][start:start + count]
        axes = data["long_axes"][start:start + count] if "long_axes" in data else None
        start += count
        material = ex._material(population)
        looser = tangle.Material(material.name, diameter=material.diameter,
                                 min_bend_radius=0.99 * material.min_bend_radius, thickness=material.thickness)
        collection = tangle.FiberCollection(material.name)
        for k, line in enumerate(lines):
            if material.is_oval and axes is not None:
                collection.add_fiber(line, looser, long_axis=axes[k])
            else:
                collection.add_fiber(line, looser)
        recipe.insert(collection, name=material.name)
    recipe.capture_junctions(tangle.JunctionPolicy(
        "binder", "bond", max_surface_gap=bonds["gap"] * um, min_crossing_angle=0.0,
        probability=bonds["probability"], seed=seed, max_per_fiber_pair=1))
    return recipe.run(tangle.RelaxationSettings(backend=ex.BACKEND, max_iterations=300, max_step=0.5 * um,
                                                penetration_tolerance=0.1 * um))


def varied_scan(index: int, cache: Path):
    """Relax (cached) and scan ``varied_<index>``; returns (SyntheticScan, settings)."""
    import hashlib
    from dataclasses import replace

    import tangle
    import tangle.ct as ct
    import ct_examples as ex
    from tangle.units import um

    v = varied_settings(index)
    for crowd in (1.0, 0.6, 0.35):  # too crowded to place every fiber: fewer fibers, same everything else
        try:
            return _varied_scan(v, index, cache, crowd)
        except Exception as error:
            if "could not place" not in str(error) or crowd == 0.35:
                raise


def _varied_scan(v, index, cache, crowd):
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
            segments_per_fiber=max(4, int(length[0] / (1.25 * diameter))), seed=v["seed"] + t, length=length,
            # thick fibers in a 160-voxel box cannot wave by a large share of their diameter within their bend
            # limit; a retry (crowd < 1) also halves the waviness
            curvature_amplitude=tuple(c * diameter * min(1.0, 8.0 / spec["diameter"]) * (1.0 if crowd == 1.0 else 0.5)
                                      for c in v["curvature"]),
            orientation=orientation, position=position,
        )
        populations.append(ex.bundles(cell, population, spec["bundle"]) if spec["bundle"] > 1 else population)
        shade = {"rim": spec["rim"] * um, "core": spec["core"]} if spec["rim"] else {}
        profiles.append((diameter, ct.CrossSection(brightness=spec["brightness"], **shade)))
    key = hashlib.sha1(json.dumps({**v, "crowd": crowd}, sort_keys=True).encode()).hexdigest()[:8]
    truth = ex.relaxed_truth(cache / f"varied_{index}-{key}.json", cell, populations)
    binder = None
    if v.get("bonds"):
        b = v["bonds"]
        truth = with_bonds(cache / f"varied_{index}-{key}.json", cell, populations, b, v["seed"])
        binder = ct.Binder(radius_ratio=b["radius_ratio"], brightness=b["brightness"], delta_beta=b["delta_beta"],
                           shape=b.get("shape", "bridge"), coating=b.get("coating", 0.0),
                           coating_thickness=b.get("coating_thickness", 1.0))
    v = {**v, "crowd": crowd}
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
    args = parser.parse_args()
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import ct_examples as ex

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / ".cache"
    for index in range(args.first, args.first + args.count):
        # Which settings family is a fixed function of the index, so reruns agree.
        family = "varied" if args.varied else (
            "scanned" if np.random.default_rng(index).random() < args.scanned_share else "dense_hard")
        name = f"{family}_{index}"
        target = args.out / f"{name}.npz"
        if target.exists():
            continue
        started = time.perf_counter()
        try:
            if family == "varied":
                scan, varied = varied_scan(index, cache)
            else:
                settings = ex.dense_hard_settings if family == "dense_hard" else ex.scanned_settings
                example = ex.scanned(index, settings)(cache / f"{name}.json")
                scan, varied = example.scan, None
        except Exception as error:  # a structure that fails to relax or render: skip it
            print(f"{name}: failed ({error})", flush=True)
            continue
        table = point_table(scan)
        near = nearest_points(scan.volume.shape, table["pos"])
        extra = {}
        if scan.bond_labels is not None:
            extra["bond_labels"] = scan.bond_labels.astype(np.uint16)
            extra["bond_ratio"] = np.float32(varied["bonds"]["radius_ratio"])
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
