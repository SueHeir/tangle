"""How closely two sets of fibers agree: found fibers against true ones (a demo scan's truth, a hand tracing,
or another run of the same region).

    python compare_fibers.py FOUND TRUTH [--tolerance 0.5]

FOUND and TRUTH are find_fibers.py result folders (or any folder holding fibers.csv and centerlines.csv in the
same columns), or Tangle fit.json files. Both must describe the same region, in micrometers from its corner.

Scores, as in docs/ct_unet.md: recall is the share of the true fibers' length with a found centerline within
``--tolerance`` true radii; precision is the share of the found length lying that close to a true centerline;
F1 combines them. Then, per true fiber type, the found diameters against the true ones, and the share of the
found length typed like the true fiber it lies on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fiber_outputs import resample


def read_fibers(path: str | Path):
    """(centerlines in um, diameters in um, types from 1) of a result folder or a fit.json."""
    path = Path(path).expanduser()
    if path.is_dir():
        if (path / "centerlines.csv").exists() and (path / "fibers.csv").exists():
            return _read_csv(path)
        if (path / "fit.json").exists():
            path = path / "fit.json"
        else:
            raise SystemExit(f"compare_fibers: {path} holds neither fibers.csv + centerlines.csv nor fit.json")
    if not path.exists():
        raise SystemExit(f"compare_fibers: can't find {path}")
    data = json.loads(path.read_text())
    lines = [np.asarray(f["centerline"], float) * 1e6 for f in data["fibers"]]
    diameters = np.array([f["diameter"] * 1e6 for f in data["fibers"]])
    types = np.array([int(f.get("type", 0)) + 1 for f in data["fibers"]], int)
    return lines, diameters, types


def _read_csv(folder: Path):
    with open(folder / "fibers.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    number = {int(row["fiber"]): k for k, row in enumerate(rows)}
    diameters = np.array([float(row["diameter_um"]) for row in rows])
    types = np.array([int(float(row.get("type") or 1)) for row in rows], int)
    table = np.loadtxt(folder / "centerlines.csv", delimiter=",", skiprows=1, ndmin=2)
    lines = [np.zeros((0, 3))] * len(rows)
    if len(table):
        order = np.lexsort((table[:, 1], table[:, 0]))
        table = table[order]
        ids, starts = np.unique(table[:, 0].astype(int), return_index=True)
        for fid, part in zip(ids, np.split(table[:, 2:5], starts[1:])):
            if fid in number:
                lines[number[fid]] = part
    return lines, diameters, types


def samples(lines, spacing: float):
    """Points every ``spacing`` along every line, and the number of the line each belongs to."""
    parts, owners = [], []
    for k, line in enumerate(lines):
        if len(line) >= 2:
            points = resample(line, spacing)
            parts.append(points)
            owners.append(np.full(len(points), k))
    if not parts:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.concatenate(parts), np.concatenate(owners)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="compare_fibers.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("found", help="find_fibers.py result folder or fit.json")
    parser.add_argument("truth", help="the fibers to compare with: folder or fit.json")
    parser.add_argument("--tolerance", type=float, default=0.5, help="in true radii (default 0.5)")
    args = parser.parse_args(argv)
    found, found_d, found_t = read_fibers(args.found)
    truth, true_d, true_t = read_fibers(args.truth)
    spacing = max(0.25 * float(np.min(true_d)), 0.1) if len(true_d) else 1.0
    t_pts, t_owner = samples(truth, spacing)
    f_pts, f_owner = samples(found, spacing)
    if not len(t_pts) or not len(f_pts):
        print(f"found {len(found)} fibers, true {len(truth)}: nothing to compare")
        return 0
    t_radius = 0.5 * true_d[t_owner]
    distance, _ = cKDTree(f_pts).query(t_pts)
    recall = float(np.mean(distance <= args.tolerance * t_radius))
    distance, nearest = cKDTree(t_pts).query(f_pts)
    on_truth = distance <= args.tolerance * t_radius[nearest]
    precision = float(np.mean(on_truth))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    def length(lines):
        return sum(float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in lines if len(l) > 1)

    print(f"found {len(found)} fibers, {length(found) / 1000:.3g} mm; true {len(truth)} fibers, "
          f"{length(truth) / 1000:.3g} mm")
    print(f"centerlines within {args.tolerance:g} true radii: recall {recall:.3f}, precision {precision:.3f}, "
          f"F1 {f1:.3f}")
    # each found fiber lies on the true fiber most of its matched samples are nearest
    match = np.full(len(found), -1)
    weight = np.zeros(len(found))
    for k in range(len(found)):
        mine = (f_owner == k) & on_truth
        weight[k] = float(np.sum(f_owner == k))
        if mine.any():
            match[k] = int(np.bincount(t_owner[nearest[mine]]).argmax())
    for t in np.unique(true_t):
        on = [k for k in range(len(found)) if match[k] >= 0 and true_t[match[k]] == t]
        text = f"true type {t}: {int(np.sum(true_t == t))} fibers, median diameter {np.median(true_d[true_t == t]):.3g} um"
        if on:
            text += f"; {len(on)} found fibers lie on them, median diameter {np.median(found_d[on]):.3g} um"
        print(text)
    matched = match >= 0
    if matched.any() and len(np.unique(true_t)) > 1:  # both number their types by size, 1 = the thinnest
        kinds, how = found_t, ""
        if len(np.unique(found_t)) != len(np.unique(true_t)):  # numbered differently: pair them by diameter
            sizes = {t: np.log(np.median(true_d[true_t == t])) for t in np.unique(true_t)}
            pair = {f: min(sizes, key=lambda t: abs(sizes[t] - np.log(np.median(found_d[found_t == f]))))
                    for f in np.unique(found_t)}
            kinds = np.array([pair[f] for f in found_t], int)
            how = (f" ({len(np.unique(found_t))} found types against {len(np.unique(true_t))} true ones, each found "
                   "type taken as the true type nearest in diameter)")
        right = kinds[matched] == true_t[match[matched]]
        share = float((weight[matched] * right).sum() / weight[matched].sum())
        print(f"typed like the true fiber they lie on: {100 * share:.1f}% of the matched found length{how}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
