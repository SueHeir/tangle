"""How well the per-fiber measures in fibers.csv (``fiber_types.measures``: bend, curvature, axis grey) come out
of traced fibers, on make_data volumes with known truth.

usage: python check_measures.py CHECKPOINT FILE.npz... [--out TABLE.csv]

Each traced fiber is matched to the true fiber most of its points lie on (within a radius). Bend and curvature
are measured the same way on the traced line and on the true one, so the difference is the tracer's alone. Grey
is the median scan grey along the axis, traced and true; where the scan's settings (its .json) give each fiber
type's brightness, it is compared with that too, by true diameter, to show how much blur dims thin fibers.
A per-fiber brightness spread in the settings (``brightness_spread``) is not stored, so it shows as scatter.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from fiber_types import measures
from maps import levels, load, predict
from trace_maps import trace


def true_lines(data):
    fid = data["p_fid"]
    ids = np.unique(fid)
    lines = [data["p_pos"][fid == f].astype(np.float64) for f in ids]
    radii = np.array([float(np.median(data["p_rad"][fid == f])) for f in ids])
    types = np.array([int(data["p_typ"][fid == f][0]) for f in ids])
    return lines, radii, types


def type_brightness(path: Path, types: np.ndarray):
    """Each true fiber's type brightness from the scan's settings, or None where they don't say it plainly
    (no settings, or not every type of the settings present, so the stored type numbers may be renumbered)."""
    meta = path.with_suffix(".json")
    if not meta.exists():
        return None, None
    settings = (json.loads(meta.read_text()).get("settings") or {})
    kinds = settings.get("types")
    if not kinds or len(np.unique(types)) != len(kinds):
        return None, None
    return np.array([kinds[t]["brightness"] for t in types], float), float(settings.get("brightness_spread", 0.0))


def padded_predict(model, volume, device, grey):
    pad = [(0, -n % 16) for n in volume.shape]
    maps = predict(model, np.pad(volume, pad, mode="edge"), device, grey)
    return maps[(slice(None), *(slice(0, n) for n in volume.shape))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="write one row per matched fiber here")
    parser.add_argument("--smooth", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0],
                        help="also measure bend and curvature on the traced lines smoothed over these many diameters")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load(args.checkpoint, device)
    rows = []
    for path in args.files:
        data = dict(np.load(path))
        volume = data["volume"].astype(np.float32)
        grey = levels(volume)
        lines, radii, types = true_lines(data)
        if not lines:
            continue
        maps = padded_predict(model, volume, device, grey)
        traced, traced_r = trace(maps, min_length=6.0 * float(radii.min()))
        traced = [np.asarray(line, float) for line in traced]
        if not traced:
            print(f"{path.stem}: nothing traced", flush=True)
            continue
        true_m = measures(lines, radii, volume, grey)
        found_m = measures(traced, traced_r, volume, grey)
        raw_true, raw_found = measures(lines, radii, volume, (0.0, 1.0)), measures(traced, traced_r, volume, (0.0, 1.0))
        smooth_m = {w: measures(traced, traced_r, smooth=w) for w in args.smooth if w}
        bright, spread = type_brightness(path, types)
        points = np.concatenate(lines)
        owner = np.concatenate([np.full(len(line), g) for g, line in enumerate(lines)])
        tree = cKDTree(points)
        for k, line in enumerate(traced):
            d, i = tree.query(line)
            near = d <= np.maximum(radii[owner[i]], 1.0)
            if near.sum() < 0.5 * len(line):
                continue
            g = int(np.bincount(owner[i][near]).argmax())
            rows.append({"scan": path.stem, "true_diameter": 2 * radii[g], "found_diameter": 2 * traced_r[k],
                         "true_bend": true_m["bend"][g], "found_bend": found_m["bend"][k],
                         "true_curvature": true_m["curvature"][g], "found_curvature": found_m["curvature"][k],
                         **{f"found_bend_s{w:g}": m["bend"][k] for w, m in smooth_m.items()},
                         **{f"found_curvature_s{w:g}": m["curvature"][k] for w, m in smooth_m.items()},
                         "true_axis_grey": true_m["grey"][g], "found_axis_grey": found_m["grey"][k],
                         "true_raw_grey": raw_true["grey"][g], "found_raw_grey": raw_found["grey"][k],
                         "type_brightness": np.nan if bright is None else bright[g],
                         "brightness_spread": np.nan if spread is None else spread})
        print(f"{path.stem}: {len(lines)} true, {len(traced)} traced, {sum(r['scan'] == path.stem for r in rows)} "
              "matched", flush=True)
    if not rows:
        raise SystemExit("no matched fibers")
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report({k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "scan"},
           np.array([r["scan"] for r in rows]))


def report(c, scans):
    n = len(c["true_bend"])
    print(f"\n{n} matched fibers")
    for name, floor in (("bend", 0.005), ("curvature", 0.002)):
        t = np.log(c[f"true_{name}"] + floor)
        for key in [f"found_{name}"] + sorted(k for k in c if k.startswith(f"found_{name}_s")):
            f = np.log(c[key] + floor)
            err = f - t
            label = name + (f", smoothed over {key.rsplit('_s', 1)[1]} diameters" if "_s" in key[6:] else ", as traced")
            print(f"{label} (log, + {floor}): r = {np.corrcoef(t, f)[0, 1]:.3f}, error median {np.median(err):+.3f}, "
                  f"IQR {np.percentile(err, 25):+.3f}..{np.percentile(err, 75):+.3f}")
    ok = np.isfinite(c["true_axis_grey"]) & np.isfinite(c["found_axis_grey"])
    t, f = c["true_axis_grey"][ok], c["found_axis_grey"][ok]
    print(f"axis grey, traced vs true axis: r = {np.corrcoef(t, f)[0, 1]:.3f}, error median {np.median(f - t):+.3f}")
    has = ok & np.isfinite(c["type_brightness"]) & np.isfinite(c["found_raw_grey"])
    if has.sum() > 10 and np.ptp(c["type_brightness"][has]) > 0:
        # in the scan's own grey: the scaling's void level (the median) sits at fiber grey on dense scans
        ratio = c["found_raw_grey"] / c["type_brightness"]
        for scan in np.unique(scans[has]):  # each scan has its own grey scale: relative to its own median
            m = has & (scans == scan)
            ratio[m] /= np.median(ratio[m])
        ratio = ratio[has]
        d = c["true_diameter"][has]
        print(f"raw axis grey / type brightness, relative to each scan's median ({has.sum()} fibers with known "
              "brightness), by true diameter:")
        for low, high in ((0, 5), (5, 7), (7, 10), (10, 14), (14, 20), (20, 40)):
            m = (d >= low) & (d < high)
            if m.sum() >= 5:
                print(f"  {low:>2}-{high:<2} voxels, n {m.sum():4d}: median {np.median(ratio[m]):.3f}, "
                      f"IQR {np.percentile(ratio[m], 25):.3f}..{np.percentile(ratio[m], 75):.3f}")
        within = []  # the correlation inside each scan whose types differ in brightness (grey scales differ)
        for scan in np.unique(scans[has]):
            m = has & (scans == scan)
            if m.sum() >= 10 and np.ptp(c["type_brightness"][m]) > 0:
                within.append(np.corrcoef(c["found_raw_grey"][m], c["type_brightness"][m])[0, 1])
        if within:
            print(f"  r(raw axis grey, type brightness) within each scan with brighter and dimmer types: median "
                  f"{np.median(within):.3f} over {len(within)} scans (range {min(within):.3f}..{max(within):.3f})")
    elif has.sum():
        print("type brightness: every fiber here has the same, so grey against brightness can't be checked")


if __name__ == "__main__":
    sys.exit(main())
