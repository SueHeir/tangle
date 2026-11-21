"""Per-scan audit of a training set: stored settings, measured image quality and orientation, a network's loss.

    python audit_training.py NETWORK.pt DATA_ROOT OUT_DIR [--sets data_mixed,data_stiff,...] [--f1-share 0.2]
                             [--exclude DATA_ROOT/exclude_overlaps.txt] [--shard 0/4] [--binder-positive 19] [--min-age 60]
    python audit_training.py --report OUT_DIR          (figures and report only, from the csv files there)

Reads DATA_ROOT/<set>/*.npz (make_data scans) and the .json beside each (its settings); every folder of DATA_ROOT
whose name starts with "data" is a set unless --sets names them. Writes OUT_DIR/train_audit[_shardK].csv (one row
per scan, appended, so a stopped run picks up where it left off), then (with one shard; with several, run
--report once all are done) OUT_DIR/train_audit.csv merged, OUT_DIR/audit_figures/<set>.png and OUT_DIR/train_audit_report.txt.
Nothing in DATA_ROOT is changed.

Per scan:
- settings (from the json; varied scans store them flat, mixed / bends / empty scans under "scanner"; pairs and
  hardpairs store none): orientation, tilt, blur (the "resolution" setting, voxels), propagation, photons, noise
  blur, number of types, thinnest and thickest diameter, blur / thinnest diameter, per-type brightness.
- measured from the volume and the truth: fiber count, fiber and binder volume fractions, the contrast-to-noise
  ratio per fiber type (median grey on its axes minus the median void grey, over the void's noise sd; void = more
  than 3 voxels from fiber and binder), the share of fiber voxels darker than the void's 95th percentile (fibers
  invisible in the scan), and the eigenvalues of the length-weighted orientation tensor of the true centerlines
  (all fibers, and the largest one per type).
- the network's loss on the center 128^3 crop as validation computes it, every term apart, with no hints and with
  every hint the truth allows. A 3-way network ("classes") gets train.py's terms; an older bond-point network
  (r23) gets its own terms (fiber and binder apart, binder positives weighted --binder-positive as it was trained,
  19 for r23) without the bond-point term, whose targets this code no longer makes.
- centerline F1 of the network's traces against the truth (score_trace style, no hints) for a seeded share of the
  scans (--f1-share; the same scans every run).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TERMS = ["heat", "offset", "direction", "radius", "fiber", "binder", "classes"]
COLS = (["set", "file", "index", "excluded", "family", "orientation", "tilt", "blur", "propagation", "photons",
         "noise_blur", "n_types", "d_min", "d_max", "blur_over_dmin", "type_brightness", "binder_brightness",
         "fibers", "fiber_vf", "binder_vf", "cnr_min", "cnr_per_type", "invisible_share",
         "eig1", "eig2", "eig3", "eig1_per_type"]
        + [f"loss_none_{t}" for t in TERMS] + ["loss_none"] + [f"loss_all_{t}" for t in TERMS] + ["loss_all"]
        + ["f1", "traced", "true", "seconds", "error"])
CUTS = {"blur > 0.7 x thinnest diameter": lambda r: num(r, "blur_over_dmin") > 0.7,
        "CNR < 2": lambda r: num(r, "cnr_min") < 2,
        "invisible fiber share > 30%": lambda r: num(r, "invisible_share") > 0.3,
        "binder > 15%": lambda r: num(r, "binder_vf") > 0.15}


class Loaded(dict):
    """A scan's npz read once (NpzFile decompresses an array again on every access), with NpzFile's .files."""

    def __init__(self, npz):
        super().__init__({k: npz[k] for k in npz.files})
        self.files = list(npz.files)


def settings(path: Path) -> dict:
    j = path.with_suffix(".json")
    if not j.exists():
        return {}
    s = json.loads(j.read_text()).get("settings", {})
    sc = s.get("scanner", s)
    types = s.get("types") or []
    ds = [t["diameter"] for t in types if "diameter" in t]
    out = dict(family=s.get("family", "mixed" if "scanner" in s else "varied" if "resolution" in s else ""),
               orientation=s.get("orientation", ""), tilt=s.get("tilt", ""), blur=sc.get("resolution", ""),
               propagation=sc.get("propagation", ""), photons=sc.get("photons", ""),
               noise_blur=sc.get("noise_blur", ""), n_types=len(types) or "",
               d_min=min(ds) if ds else "", d_max=max(ds) if ds else "",
               type_brightness=";".join(str(t.get("brightness", "")) for t in types))
    if ds and sc.get("resolution") not in (None, ""):
        out["blur_over_dmin"] = round(float(sc["resolution"]) / min(ds), 3)
    b = s.get("binder") or s.get("bonds") or {}
    out["binder_brightness"] = b.get("brightness", "") if isinstance(b, dict) else ""
    return out


def measured(d) -> dict:
    from scipy import ndimage

    vol = d["volume"].astype(np.float32)
    labels = d["labels"]
    fiber = labels > 0
    binder = next(((d[k] > 0) for k in ("binder_mask", "bond_labels") if k in d), np.zeros(labels.shape, bool))
    binder = binder & ~fiber
    far = ndimage.distance_transform_edt(~(fiber | binder)) > 3.0
    void = vol[far] if far.any() else vol[~(fiber | binder)]
    vmed = float(np.median(void))
    vsd = 1.4826 * float(np.median(np.abs(void - vmed)))
    v95 = float(np.percentile(void, 95))
    out = dict(fibers=len(np.unique(d["p_fid"])), fiber_vf=round(float(fiber.mean()), 4),
               binder_vf=round(float(binder.mean()), 4),
               invisible_share=round(float((vol[fiber] < v95).mean()), 4) if fiber.any() else "")
    pos, typ, tan = d["p_pos"], d["p_typ"], d["p_tan"]
    if len(pos):
        ijk = np.clip(np.floor(pos).astype(int), 0, np.array(vol.shape[::-1]) - 1)
        axis = vol[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        cnr = [(float(np.median(axis[typ == t])) - vmed) / max(vsd, 1e-6) for t in np.unique(typ)]
        out.update(cnr_per_type=";".join(f"{c:.2f}" for c in cnr), cnr_min=round(min(cnr), 3))

        def eig(m):  # the point tables are evenly spaced along each line, so a mean over points is length weighted
            t = tan[m] / np.maximum(np.linalg.norm(tan[m], axis=1, keepdims=True), 1e-9)
            return np.sort(np.linalg.eigvalsh(t.T @ t / max(len(t), 1)))[::-1]

        e = eig(np.ones(len(tan), bool))
        out.update(eig1=round(float(e[0]), 4), eig2=round(float(e[1]), 4), eig3=round(float(e[2]), 4))
        if len(np.unique(typ)) > 1:
            out["eig1_per_type"] = ";".join(f"{eig(typ == t)[0]:.3f}" for t in np.unique(typ))
    return out


def losses(model, d, device, binder_positive: float) -> dict:
    import torch
    import torch.nn.functional as F

    import maps as M
    from train import WEIGHTS

    shape = d["volume"].shape
    window = tuple(slice((s - 128) // 2, (s - 128) // 2 + 128) for s in shape)
    sample = M.targets(d, window)
    grey = tuple(float(g) for g in d["grey"]) if "grey" in d else None
    sample["image"] = M.normalize(d["volume"], grey)[window][None]
    width = model.film[0].in_features if getattr(model, "condition", False) else None
    out = {}
    for tag, said in (("none", {}), ("all", M.scan_hints(d))):
        code = M.size_code(**said)[:width] if width else None
        batch = {k: torch.from_numpy(np.ascontiguousarray(v))[None].to(device) for k, v in sample.items()}
        with torch.no_grad():
            o = (model(batch["image"], torch.from_numpy(code)[None].to(device)) if width else model(batch["image"])).float()
            if model.layout == "classes":
                terms = {k: float(v) for k, v in M.loss_terms(o, batch).items()}
                total = sum(WEIGHTS[k] * v for k, v in terms.items())
            else:  # a bond-point network's own terms, less the bond points
                terms = {k: float(v) for k, v in M.loss_terms(o, batch).items() if k != "classes"}
                fib, bnd = batch["fiber"], batch["binder"]
                terms["fiber"] = float((F.binary_cross_entropy_with_logits(o[:, M.FIBER], fib, reduction="none")
                                        * (1.0 + 2.0 * fib)).mean())
                terms["binder"] = float((F.binary_cross_entropy_with_logits(o[:, M.BINDER], bnd, reduction="none")
                                         * (1.0 + binder_positive * bnd)).mean())
                total = sum(terms[k] for k in ("heat", "offset", "direction", "radius", "binder")) + 0.5 * terms["fiber"]
        for k in TERMS:
            out[f"loss_{tag}_{k}"] = round(terms[k], 5) if k in terms else ""
        out[f"loss_{tag}"] = round(total, 5)
    return out


def f1(model, path: Path, device) -> dict:
    from maps import predict
    from score_trace import from_npz
    from tangle.ct._evaluate import _samples_inside, centerline_agreement
    from trace_maps import trace

    volume, lines_true, radii_true, _, data = from_npz(str(path))
    if not len(lines_true):
        return {}
    grey = tuple(float(g) for g in data["grey"]) if "grey" in data else None
    maps = predict(model, volume, device, grey)
    shortest = 6.0 * float(radii_true.min())
    lines, _ = trace(maps, min_length=shortest, min_votes=3, tidy=True)
    shape = volume.shape[::-1]
    stubs = {k for k, line in enumerate(lines_true) if 0.5 * len(_samples_inside(line, shape)) < shortest}
    a = centerline_agreement(lines, lines_true, radii_true, shape, skip=stubs)
    return dict(f1=round(a["f1"], 4), traced=len(lines), true=len(lines_true) - len(stubs))


def audit(args):
    import torch

    from maps import load

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load(args.network, device).eval()
    root = Path(args.data_root)
    sets = args.sets.split(",") if args.sets else sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("data"))
    excluded = set()
    if args.exclude:
        excluded = {Path(line.strip()).name for line in open(args.exclude) if line.strip()}
    k, n = (int(x) for x in args.shard.split("/"))
    now = time.time()
    files = [(s, p) for s in sets for p in sorted((root / s).glob("*.npz")) if now - p.stat().st_mtime > args.min_age]
    # a scan's shard follows from its name, so it stays put while a folder is still filling
    files = [(s, p) for s, p in files if zlib.crc32(p.name.encode()) % n == k]
    out = Path(args.out_dir) / (f"train_audit_shard{k}.csv" if n > 1 else "train_audit_rows.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["set"], r["file"]) for r in csv.DictReader(open(out))} if out.exists() else set()
    print(f"{len(files)} scans in {len(sets)} sets for this shard, {len(done)} done before", flush=True)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        if not done:
            w.writeheader()
        started = time.time()
        for i, (s, path) in enumerate(files):
            if (s, path.name) in done:
                continue
            t = time.time()
            row = {"set": s, "file": path.name, "index": "".join(c for c in path.stem if c.isdigit()),
                   "excluded": int(path.stem in excluded)}
            try:
                d = Loaded(np.load(path))
                row.update(settings(path))
                row.update(measured(d))
                row.update(losses(model, d, device, args.binder_positive))
                if (zlib.crc32(path.name.encode()) % 1000) / 1000 < args.f1_share:
                    row.update(f1(model, path, device))
            except Exception as e:  # a broken scan is a finding too
                row["error"] = f"{type(e).__name__}: {e}"[:200]
            row["seconds"] = round(time.time() - t, 2)
            w.writerow(row)
            fh.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(files)} {s}/{path.name} loss {row.get('loss_none', '')} "
                      f"({(time.time() - started) / (i + 1):.1f} s per scan)", flush=True)


def num(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return np.nan


def report(out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for p in sorted(out_dir.glob("train_audit_*.csv")) for r in csv.DictReader(open(p))]
    with open(out_dir / "train_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(rows)
    sets = defaultdict(list)
    for r in rows:
        sets[r["set"]].append(r)
    lines = [f"{len(rows)} scans in {len(sets)} sets; errors {sum(bool(r['error']) for r in rows)}; "
             f"flagged by the exclude list {sum(r['excluded'] == '1' for r in rows)}"]
    lines.append("\nper set: scans | past each cut-off (" + " | ".join(CUTS) + ") | orientation setting counts | "
                 "median loss no hints, all hints | mean F1 (scans)")
    for s, rs in sorted(sets.items()):
        orient = defaultdict(int)
        for r in rs:
            orient[r["orientation"] or "not stored"] += 1
        f1s = [num(r, "f1") for r in rs if r["f1"]]
        lines.append(f"{s}: {len(rs)} | " + " | ".join(str(sum(f(r) for r in rs)) for f in CUTS.values())
                     + f" | {dict(orient)} | {np.nanmedian([num(r, 'loss_none') for r in rs]):.3f}, "
                       f"{np.nanmedian([num(r, 'loss_all') for r in rs]):.3f} | "
                       f"{np.mean(f1s) if f1s else float('nan'):.3f} ({len(f1s)})")

    def mean(rs, k):
        v = [num(r, k) for r in rs if r.get(k) not in ("", None)]
        return (np.nanmean(v) if v else np.nan), len(v)

    lines.append("\nthe network on the scans past each cut-off vs the rest (all sets): mean loss no hints | mean F1 (n)")
    for name, f in CUTS.items():
        a = [r for r in rows if f(r)]
        b = [r for r in rows if not f(r)]
        lines.append(f"{name}: past {len(a)} loss {mean(a, 'loss_none')[0]:.3f} F1 {mean(a, 'f1')[0]:.3f} "
                     f"({mean(a, 'f1')[1]}) | rest {len(b)} loss {mean(b, 'loss_none')[0]:.3f} "
                     f"F1 {mean(b, 'f1')[0]:.3f} ({mean(b, 'f1')[1]})")
    keys = ["file", "loss_none", "loss_all", "f1", "error", "family", "orientation", "blur", "photons", "d_min",
            "d_max", "blur_over_dmin", "cnr_min", "invisible_share", "binder_vf", "fibers", "excluded"]
    lines.append("\n20 highest-loss scans per set (no hints)")
    figures = out_dir / "audit_figures"
    figures.mkdir(exist_ok=True)
    for s, rs in sorted(sets.items()):
        lines.append(f"-- {s}")
        for r in sorted(rs, key=lambda r: -np.nan_to_num(num(r, "loss_none"), nan=-1.0))[:20]:
            lines.append("  " + " ".join(f"{k}={r.get(k, '')}" for k in keys if r.get(k, "") != ""))
        panels = [("blur_over_dmin", "blur / thinnest diameter"), ("cnr_min", "CNR, dimmest type"),
                  ("invisible_share", "invisible fiber share"), ("f1", "F1 (sampled scans)"),
                  ("loss_none", "loss, no hints"), ("eig1", "largest orientation eigenvalue")]
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4))
        for ax, (k, label) in zip(axes, panels):
            v = np.array([num(r, k) for r in rs])
            v = v[np.isfinite(v)]
            if len(v):
                ax.hist(v, bins=30, color="#4C72B0")
            ax.set_title(f"{label} (n {len(v)})", fontsize=9)
        fig.suptitle(f"{s}: {len(rs)} scans")
        fig.tight_layout()
        fig.savefig(figures / f"{s}.png", dpi=110)
        plt.close(fig)
    (out_dir / "train_audit_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: 4 + len(sets) + len(CUTS) + 3]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("network", nargs="?")
    p.add_argument("data_root", nargs="?")
    p.add_argument("out_dir", nargs="?")
    p.add_argument("--report", metavar="OUT_DIR", help="only merge the csv files there and make the figures and report")
    p.add_argument("--sets", help="comma-separated set folders (default: every data* folder)")
    p.add_argument("--exclude", help="a list of scan names (one per line) to flag as excluded from training")
    p.add_argument("--f1-share", type=float, default=0.2)
    p.add_argument("--shard", default="0/1", help="K/N: audit the scans whose name hashes to K of N (run N at once)")
    p.add_argument("--min-age", type=float, default=60.0,
                   help="seconds: leave out scans written more recently (a generator may still be writing them); "
                        "run again later for the rest of a folder still filling")
    p.add_argument("--binder-positive", type=float, default=19.0,
                   help="binder positive weight of an older bond-point network's loss (r23: 19)")
    args = p.parse_args()
    if args.report:
        return report(Path(args.report))
    if not (args.network and args.data_root and args.out_dir):
        p.error("NETWORK DATA_ROOT OUT_DIR are needed (or --report OUT_DIR)")
    audit(args)
    out = Path(args.out_dir)
    k, n = (int(x) for x in args.shard.split("/"))
    if n == 1:
        report(out)
    else:
        print(f"shard {k} done; when every shard is, run: python audit_training.py --report {out}")


if __name__ == "__main__":
    main()
