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

usage: python make_data.py OUT FIRST COUNT [--scanned-share 0.2]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("first", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("--scanned-share", type=float, default=0.2)
    args = parser.parse_args()
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import ct_examples as ex

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / ".cache"
    for index in range(args.first, args.first + args.count):
        # Which settings family is a fixed function of the index, so reruns agree.
        family = "scanned" if np.random.default_rng(index).random() < args.scanned_share else "dense_hard"
        name = f"{family}_{index}"
        target = args.out / f"{name}.npz"
        if target.exists():
            continue
        started = time.perf_counter()
        settings = ex.dense_hard_settings if family == "dense_hard" else ex.scanned_settings
        try:
            example = ex.scanned(index, settings)(cache / f"{name}.json")
        except Exception as error:  # a structure that fails to relax or render: skip it
            print(f"{name}: failed ({error})", flush=True)
            continue
        scan = example.scan
        table = point_table(scan)
        near = nearest_points(scan.volume.shape, table["pos"])
        extra = {}
        if scan.semi_axes is not None:
            extra["semi_axes"] = np.asarray(scan.semi_axes, dtype=np.float32)
        np.savez_compressed(
            target, volume=scan.volume.astype(np.uint16), labels=scan.labels.astype(np.uint16), near=near,
            **{f"p_{k}": v for k, v in table.items()}, **extra,
        )
        (args.out / f"{name}.json").write_text(json.dumps({
            "voxel_um": scan.voxel_size * 1e6, "fibers": len(scan.centerlines),
            "types": None if scan.types is None else np.bincount(scan.types).tolist(),
            "specs": [{"name": s.name, "diameter_um": s.diameter * 1e6} for s in
                      (example.spec if isinstance(example.spec, list) else [example.spec])],
        }) + "\n")
        print(f"{name}: {len(scan.centerlines)} fibers, {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
