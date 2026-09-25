"""How well each redraw judge's keep-or-revert decisions track the truth.

Fits ``ct_examples`` examples with redraws and, through the fitter's study
hook (``tangle.ct._fit._REDRAW_PROBE``), records every group of redrawn
regions: the fit before and after the redraw and every judge's gain for
the group ("nats", "grey", "mask", "confidence"). Each group is then scored
against the true fibers inside its boxes: the centerline samples that are
wrong before and after (true centerline not traced by the fitted fiber
that belongs to it, plus fitted centerline off its own fiber, in half-voxel
samples; see ``ct._evaluate.centerline_agreement``). A redraw is truly
better when that count falls.

For every judge it reports, over the same groups: how often it would keep a
truly better redraw and revert a truly worse one, and the truth change its
choices would add up to. So the judges are compared on identical proposals.

Usage::

    python ct_redraw_study.py [--output DIR] [--judge nats] [--set key=value ...] example ...

``--judge`` picks the judge the fit itself follows (later passes depend on
it); ``--set`` overrides other ``FitSettings`` (numbers, true/false, or
strings). Writes ``<output>/redraw_study.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import tangle.ct as ct
import tangle.ct._fit as fit_module
from tangle.ct._geometry import resample

import ct_examples

THRESHOLDS = {"nats": 1.0, "grey": 1.0, "mask": 1e-3, "confidence": 1e-3}


def _inside(points: np.ndarray, boxes) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for low, high in boxes:
        inside |= np.all((points >= low) & (points <= high), axis=1)
    return inside


def wrong_samples(fit_lines, truth_lines, truth_radii, shape, boxes, tolerance_radii: float = 0.5) -> int:
    """Wrong centerline samples inside ``boxes``: true not traced by its own fit, plus fitted off its own fiber."""
    upper = np.array(shape[::-1], dtype=np.float64)

    def samples(line):
        dense = resample(np.asarray(line, dtype=np.float64), 0.5) if len(line) > 1 else np.zeros((0, 3))
        return dense[np.all((dense >= 0) & (dense < upper), axis=1)]

    radii = np.asarray(truth_radii, dtype=np.float64)
    reference = [resample(np.asarray(line, dtype=np.float64), 0.25) for line in truth_lines]
    all_truth = np.vstack(reference)
    truth_id = np.concatenate([np.full(len(r), g) for g, r in enumerate(reference)])
    tree = cKDTree(all_truth)
    fit_samples = [samples(line) for line in fit_lines]
    owner = np.full(len(fit_lines), -1)
    wrong = 0
    for f, points in enumerate(fit_samples):
        if not len(points):
            continue
        distance, index = tree.query(points)
        nearest = truth_id[index]
        on = distance <= tolerance_radii * radii[nearest]
        if on.any():
            owner[f] = int(np.bincount(nearest[on]).argmax())
        own = on & (nearest == owner[f])
        wrong += int((~own & _inside(points, boxes)).sum())
    for g, line in enumerate(truth_lines):
        points = samples(line)
        points = points[_inside(points, boxes)]
        if not len(points):
            continue
        mine = [fit_samples[f] for f in np.flatnonzero(owner == g) if len(fit_samples[f])]
        if not mine:
            wrong += len(points)
            continue
        distance, _ = cKDTree(np.vstack(mine)).query(points)
        wrong += int((distance > tolerance_radii * radii[g]).sum())
    return wrong


def study(name: str, output: Path, judge: str, overrides: dict) -> dict:
    example = ct_examples.EXAMPLES[name](output / ".cache" / f"{name}.json")
    scan = example.scan
    volume, spec, _ = ct_examples.fit_input(scan, example.spec, ct_examples.INPUT or example.input)
    events: list[dict] = []
    fit_module._REDRAW_PROBE = events.append
    try:
        settings = ct.FitSettings(backend=ct_examples.BACKEND, redraw_score=judge, **overrides)
        fit = ct.fit_fibers(volume, scan.voxel_size, spec, settings)
    finally:
        fit_module._REDRAW_PROBE = None
    truth_lines = [np.asarray(line, dtype=np.float64) for line in scan.centerlines]
    truth_radii = np.asarray(scan.radii, dtype=np.float64)
    shape = scan.volume.shape
    groups = []
    for event in events:
        old_lines, _ = event["old"]
        new_lines, _ = event["new"]
        for c, boxes in enumerate(event["groups"]):
            before = wrong_samples(old_lines, truth_lines, truth_radii, shape, boxes)
            after = wrong_samples(new_lines, truth_lines, truth_radii, shape, boxes)
            gains = {key: (None if value is None else float(value[c])) for key, value in event["gains"].items()}
            groups.append(
                {"pass": event["pass"], "truth_gain": before - after, "accepted": bool(event["accepted"][c]), "gains": gains}
            )
    report = {"example": name, "judge": judge, "overrides": overrides, "groups": len(groups), "judges": {}}
    for key, threshold in THRESHOLDS.items():
        rows = [g for g in groups if g["gains"][key] is not None]
        if not rows:
            continue
        keep = np.array([g["gains"][key] > threshold for g in rows])
        truth = np.array([g["truth_gain"] for g in rows])
        report["judges"][key] = {
            "keeps_better": int((keep & (truth > 0)).sum()),
            "keeps_worse": int((keep & (truth < 0)).sum()),
            "reverts_better": int((~keep & (truth > 0)).sum()),
            "reverts_worse": int((~keep & (truth < 0)).sum()),
            "neutral": int((truth == 0).sum()),
            "truth_change_of_its_choices": int(truth[keep].sum()),
            "best_possible": int(truth[truth > 0].sum()),
        }
    report["score"] = {k: v for k, v in ct.score(fit, scan).items() if k in (
        "centerline_f1", "split", "merged_fibers", "missed", "false_fibers", "recovered"
    )}
    report["group_details"] = groups
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--judge", default="nats")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        if value.lower() in ("true", "false"):
            overrides[key] = value.lower() == "true"
        else:
            try:
                overrides[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                overrides[key] = value
    output = args.output or Path(os.environ.get("TANGLE_CT_OUTPUT", Path(__file__).with_name("output") / "ct"))
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.names:
        report = study(name, output, args.judge, overrides)
        reports.append(report)
        print(f"{name} (fit follows {args.judge}; {report['groups']} groups; score {report['score']}):")
        print("  judge       keeps better/worse  reverts better/worse  neutral  truth change / best")
        for key, row in report["judges"].items():
            print(
                f"  {key:<10}  {row['keeps_better']:>6} / {row['keeps_worse']:<6}      {row['reverts_better']:>6} / "
                f"{row['reverts_worse']:<6}   {row['neutral']:>6}  {row['truth_change_of_its_choices']:>7} / {row['best_possible']}"
            )
    (output / "redraw_study.json").write_text(json.dumps(reports, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
