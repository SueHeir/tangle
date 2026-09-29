"""Centerline F1 (and, with several fiber types, typing accuracy) of traced fibers against the truth.

usage: python score_trace.py MAPS CACHE_DIR EXAMPLE... [--no-tidy] [--min-votes N]
       python score_trace.py MAPS - FILE.npz...                 (make_data volumes, e.g. the varied test set)

MAPS is a checkpoint (.pt), ``truth`` (the true maps: tests the tracer alone), or ``fit:DIR`` (a grey fit's
fit.json files, to check this scorer against ct.score). CACHE_DIR holds ct_examples truth caches, the same ones
the grey-fitter runs used, so the scans are identical.

Typing: each traced fiber is typed by ``fiber_types`` with k = the true number of types (from its radius and axis
grey), matched to the true fiber it lies on, and the share of traced length typed right is reported.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tangle.ct._evaluate import _samples_inside, centerline_agreement  # noqa: E402
from fiber_types import assign_by_size, assign_types, features  # noqa: E402
from maps import BINDER, CHANNELS, DIRECTION, FIBER, HEAT, OFFSET, RADIUS, levels, targets  # noqa: E402
from bonds import extract_bonds, score_bonds  # noqa: E402
from trace_maps import trace  # noqa: E402


def true_maps(data) -> np.ndarray:
    """The maps a perfect network would give, from make_data-style truth (``near``, labels, point table)."""
    t = targets(data)
    shape = t["heat"].shape[1:]
    maps = np.zeros((CHANNELS, *shape), dtype=np.float32)
    maps[HEAT], maps[OFFSET], maps[DIRECTION] = t["heat"], t["offset"], t["direction"]
    maps[FIBER] = t["fiber"]
    maps[RADIUS] = np.exp(t["radius"]) * t["own"] + (1 - t["own"])
    maps[BINDER] = t["binder"]
    return maps


def truth_of(scan):
    """make_data-style truth arrays for a SyntheticScan."""
    from make_data import nearest_points, point_table

    table = point_table(scan)
    return {"near": nearest_points(scan.volume.shape, table["pos"]), "labels": scan.labels,
            **{f"p_{k}": v for k, v in table.items()}}


def rendered(name, cache: Path):
    """A ct_examples scan, rendered once and then read back from CACHE/<name>_scan.npz (rendering takes seconds)."""
    from types import SimpleNamespace

    saved = cache / f"{name}_scan.npz"
    if saved.exists():
        d = np.load(saved, allow_pickle=False)
        ends = d["ends"]
        return SimpleNamespace(volume=d["volume"], labels=d["labels"], radii=d["radii"],
                               types=d["types"] if d["types"].size else None,
                               centerlines=np.split(d["points"], ends[:-1]))
    import ct_examples as ex

    scan = ex.EXAMPLES[name](cache / f"{name}.json").scan
    np.savez(saved, volume=scan.volume, labels=scan.labels, radii=np.asarray(scan.radii),
             types=np.asarray(scan.types if scan.types is not None else []),
             points=np.concatenate(scan.centerlines), ends=np.cumsum([len(l) for l in scan.centerlines]))
    return scan


def from_npz(path):
    """(volume, centerlines, radii, types, truth dict) of a make_data volume."""
    d = dict(np.load(path))
    fid = d["p_fid"]
    lines, radii, types = [], [], []
    for f in np.unique(fid):
        m = fid == f
        lines.append(d["p_pos"][m].astype(np.float64))
        radii.append(float(d["p_rad"][m][0]))
        types.append(int(d["p_typ"][m][0]))
    return d["volume"], lines, np.array(radii), np.array(types), d


def typing_accuracy(lines, predicted, truth_lines, truth_types):
    """Share of traced length (on some true fiber) whose type matches, under the best numbering of the types."""
    truth = np.concatenate(truth_lines)
    tid = np.concatenate([np.full(len(l), t) for l, t in zip(truth_lines, truth_types)])
    tree = cKDTree(truth)
    true_of, weight = [], []
    for line in lines:
        d, i = tree.query(np.asarray(line))
        true_of.append(int(np.bincount(tid[i]).argmax()))
        weight.append(len(line))
    true_of, weight, predicted = np.array(true_of), np.array(weight), np.asarray(predicted)
    kinds = sorted(set(true_of) | set(predicted))
    best = 0.0
    for perm in itertools.permutations(kinds):
        mapping = dict(zip(kinds, perm))
        best = max(best, float((weight * (np.array([mapping[p] for p in predicted]) == true_of)).sum() / weight.sum()))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps")
    parser.add_argument("cache")
    parser.add_argument("examples", nargs="+")
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--no-tidy", action="store_true")
    parser.add_argument("--bond-hint", action="store_true", help="tell the network whether the scan is bonded")
    parser.add_argument("--bond-threshold", type=float, default=0.5, help="binder probability that counts as binder")
    parser.add_argument("--known-sizes", action="store_true",
                        help="give the network (if conditioned) and the typing each type's true diameter")
    parser.add_argument("--min-length", type=float, help="voxels (default: 3 x the smallest true diameter)")
    args = parser.parse_args()
    model = None
    if args.maps.endswith(".pt"):
        import torch
        from maps import load, predict

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = load(args.maps, device)
    scores, typing = [], []
    for name in args.examples:
        if name.endswith(".npz"):
            volume, truth_lines, truth_radii, truth_types, data = from_npz(name)
            label = Path(name).stem
        else:
            scan = rendered(name, Path(args.cache))
            volume, truth_lines, truth_radii = scan.volume, scan.centerlines, np.asarray(scan.radii)
            truth_types = np.zeros(len(truth_lines), int) if scan.types is None else np.asarray(scan.types)
            data, label = None, name
        shortest = args.min_length or 6.0 * float(truth_radii.min())
        started = time.perf_counter()
        if args.maps.startswith("fit:"):
            saved = json.loads((Path(args.maps[4:]) / name / "fit.json").read_text())
            lines = [np.asarray(f["centerline"]) / saved["voxel_size"] for f in saved["fibers"]]
            radii = None
        else:
            sizes = [2.0 * float(np.median(truth_radii[truth_types == t])) for t in np.unique(truth_types)]
            if model is not None:
                bonded = data is not None and "bond_labels" in data and bool(np.any(data["bond_labels"]))
                maps = predict(model, volume, device, diameters=sizes if args.known_sizes else None,
                               bonds=bonded if args.bond_hint else None)
            else:  # truth
                maps = true_maps(data if data is not None else truth_of(scan))
            lines, radii = trace(maps, min_length=shortest, min_votes=args.min_votes, tidy=not args.no_tidy)
        seconds = time.perf_counter() - started
        shape = volume.shape[::-1]
        # True pieces shorter in the volume than the minimum length are stubs, left out as ct.score does.
        stubs = {g for g, line in enumerate(truth_lines) if 0.5 * len(_samples_inside(line, shape)) < shortest}
        agree = centerline_agreement(lines, truth_lines, truth_radii, shape, skip=stubs)
        scores.append(agree["f1"])
        k = len(set(truth_types.tolist()))
        extra = ""
        if radii is not None and k > 1 and len(lines) >= k:
            grey = levels(volume[::2, ::2, ::2])
            if args.known_sizes:
                kinds = assign_by_size(radii, sizes)
            else:
                kinds, _ = assign_types(features(lines, radii, volume, grey), k)
            acc = typing_accuracy(lines, kinds, truth_lines, truth_types)
            typing.append(acc)
            extra = f", {k} types: typed right {acc:.3f}"
        bond_scores = getattr(main, "bond_scores", [])
        main.bond_scores = bond_scores
        if data is not None and "p_pos" in data and (model is None or getattr(model, "layout", "") == "bonds") \
                and radii is not None and ("bond_centers" in data or "bond_labels" not in data):
            found = extract_bonds(maps, lines, radii, threshold=args.bond_threshold)
            bs = score_bonds(found, data["bond_centers"] if "bond_centers" in data else np.zeros((0, 3)))
            bond_scores.append(bs)
            extra += f", bonds {bs['found']}/{bs['true']} P {bs['precision']:.2f} R {bs['recall']:.2f}"
        print(f"{label}: true {len(truth_lines) - len(stubs)} traced {len(lines)} recall {agree['recall']:.3f} "
              f"precision {agree['precision']:.3f} F1 {agree['f1']:.3f}{extra} ({seconds:.1f} s)", flush=True)
    bond_scores = getattr(main, "bond_scores", [])
    bonded = [b for b in bond_scores if b["true"] > 0]
    free = [b for b in bond_scores if b["true"] == 0]
    print(f"mean F1 {np.mean(scores):.3f} over {len(scores)}"
          + (f"; typing {np.mean(typing):.3f} over {len(typing)} multi-type scans" if typing else "")
          + (f"; bonds F1 {np.mean([b['f1'] for b in bonded]):.3f} over {len(bonded)} bonded scans" if bonded else "")
          + (f", {np.mean([b['found'] for b in free]):.1f} false bonds per bond-free scan" if free else ""))


if __name__ == "__main__":
    main()
