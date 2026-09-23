"""One small scan per fitting step: what each step does, and why.

Each scenario is a handful of 12 µm fibers placed by hand in a 150 µm box,
rendered with ``ct.synthetic_ct`` (1.5 µm voxels, 8 voxels across a fiber),
and fitted three ways, all on Tangle's solver:

* ``mask``: from a generous binary mask (``scan.fiber_mask(level=0.35)``,
  thresholded 35% of the way from void to fiber, so fibers look thicker),
  no fiber-length prior;
* ``mask_prior``: the same mask with ``FiberSpec(length=400 µm)``;
* ``grey_prior``: from the grey-level scan with the prior. Grey levels are
  estimated from the histogram, which fails when fibers are a small part
  of the scan (the sparse scenarios here).

| scenario | step it exercises | right answer |
|---|---|---|
| ``single_straight`` | seeding, tracing | one fit end to end |
| ``interior_ends`` | end growth and trimming | fit ends at the true ends |
| ``gap_2r`` / ``gap_6r`` / ``gap_12r`` | joins | see below |
| ``crossing_90`` / ``crossing_30`` | tracing through a crossing, kink splits | two fits, none merged |
| ``parallel_touching`` | one fiber or two (side by side) | two fits |
| ``short_piece`` | removal | no fit (a 2-diameter piece is below the minimum length) |
| ``bent_near_limit`` | kink splits | one fit, not cut (it bends at 1.25× the bend radius) |

The gap scenarios are two collinear pieces along the box diagonal with a gap
of 2, 6 or 12 radii.
The scan cannot tell a real break from a stretch the scanner missed; only
the length prior decides. Without it, gaps under 4 radii are joined; with
it, a join is made when the image evidence against it is smaller than the
cost of the two ends it removes. Scoring counts a join as merging two true
fibers.

Per scenario and fit, the table shows: fitted fibers, recovered / split /
missed / merged, centerline error, the distance from each true interior end
to the nearest fit end, the splits, merges and births the fit made, and its
geometry (overlapping pairs, fibers over the bend limit, measured at 1.25
diameters). Overlays go to ``<output>/<scenario>/<fit>/``.

Needs NumPy, SciPy and a Tangle build with ``ImageRelaxer``; tifffile and
matplotlib add TIFF and PNG outputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import tangle
import tangle.ct as ct
from tangle.units import um

OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("output") / "ct_fit_scenarios")
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None

BOX = 150 * um
DIAMETER = 12 * um
R = DIAMETER / 2
MIN_BEND_RADIUS = 4 * DIAMETER
VOXEL = 1.5 * um
LENGTH = 400 * um
MID = BOX / 2


def line(start, end, n: int = 12) -> list[list[float]]:
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return [list(start + t * (end - start)) for t in np.linspace(0.0, 1.0, n)]


def arc(radius: float, sweep: float, center, n: int = 24) -> list[list[float]]:
    """A planar arc in z = center[2], bowing toward +y."""
    angles = np.linspace(-0.5 * sweep, 0.5 * sweep, n)
    cx, cy, cz = center
    return [[cx + radius * np.sin(a), cy - radius * np.cos(a) + radius, cz] for a in angles]


def gap(width: float) -> list[list[list[float]]]:
    """Two collinear pieces along the box's xy diagonal with a gap between them."""
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    start, end = np.array([5 * um, 5 * um, MID]), np.array([BOX - 5 * um, BOX - 5 * um, MID])
    center = 0.5 * (start + end)
    half = 0.5 * width * direction
    return [line(start, center - half, 8), line(center + half, end, 8)]


def crossing(angle_degrees: float) -> list[list[list[float]]]:
    a = np.radians(angle_degrees)
    half = 0.5 * BOX - 8 * um
    direction = np.array([np.cos(a), np.sin(a), 0.0])
    center = np.array([MID, MID, MID + 0.5 * DIAMETER + 0.1 * um])  # touching the first fiber
    return [
        line([5 * um, MID, MID - 0.5 * DIAMETER - 0.1 * um], [BOX - 5 * um, MID, MID - 0.5 * DIAMETER - 0.1 * um]),
        line(center - half * direction, center + half * direction),
    ]


SCENARIOS = {
    "single_straight": [line([5 * um, 60 * um, 70 * um], [BOX - 5 * um, 90 * um, 80 * um])],
    "interior_ends": [line([30 * um, MID, MID], [120 * um, 80 * um, MID], 10)],
    "gap_2r": gap(2 * R),
    "gap_6r": gap(6 * R),
    "gap_12r": gap(12 * R),
    "crossing_90": crossing(90.0),
    "crossing_30": crossing(30.0),
    "parallel_touching": [
        line([5 * um, MID - 0.5 * DIAMETER - 0.1 * um, MID], [BOX - 5 * um, MID - 0.5 * DIAMETER - 0.1 * um, MID]),
        line([5 * um, MID + 0.5 * DIAMETER + 0.1 * um, MID], [BOX - 5 * um, MID + 0.5 * DIAMETER + 0.1 * um, MID]),
    ],
    "short_piece": [line([MID - DIAMETER, MID, MID], [MID + DIAMETER, MID, MID], 3)],
    "bent_near_limit": [arc(1.25 * MIN_BEND_RADIUS, 1.6, [MID, 45 * um, MID])],
}


MASK_LEVEL = 0.35


def fits(scan: ct.SyntheticScan) -> dict[str, tuple[np.ndarray, ct.FiberSpec]]:
    base = ct.FiberSpec(diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    prior = base.replace(length=LENGTH)
    mask = scan.fiber_mask(level=MASK_LEVEL)
    return {"mask": (mask, base), "mask_prior": (mask, prior), "grey_prior": (scan.volume, prior)}


def scan_of(centerlines) -> ct.SyntheticScan:
    material = tangle.Material("fiber", diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
    assembly = tangle.Assembly(tangle.Cell([BOX] * 3))
    assembly.insert(tangle.FiberCollection.from_centerlines(centerlines, material), name="scenario")
    return ct.synthetic_ct(assembly, VOXEL, seed=5)


def end_errors(fit, scan) -> list[float]:
    """Distance (voxels) from each true end inside the scan to the nearest fit end."""
    from tangle.ct._ends import interior_end_mask

    tips = [line[e] for line in fit.centerlines for e in (0, -1) if len(line)]
    mask = interior_end_mask(scan.centerlines, scan.radii, scan.volume.shape)
    errors = []
    for i, line in enumerate(scan.centerlines):
        for e, inside in zip((0, -1), mask[i]):
            if inside:
                errors.append(min((float(np.linalg.norm(t - line[e])) for t in tips), default=float("inf")))
    return errors


def moves(fit) -> dict[str, int]:
    total = {"splits": 0, "merges": 0, "births": 0}
    for entry in fit.history:
        for key in ("splits", "merges"):
            total[key] += int(entry.get(key, 0))
        total["births"] += int(entry.get("born", 0))
    return total


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, centerlines in SCENARIOS.items():
        if ONLY and name not in ONLY:
            continue
        scan = scan_of(centerlines)
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
        ct.save_overlay_figure(OUTPUT / name / "truth_overlay.png", scan.volume, scan.labels, title=f"{name}: truth")
        settings = ct.FitSettings(backend=BACKEND)
        for label, (volume, spec) in fits(scan).items():
            started = time.perf_counter()
            fit = ct.fit_fibers(volume, VOXEL, spec, settings)
            seconds = time.perf_counter() - started
            report = ct.score(fit, scan)
            geometry = ct.geometry_report(
                fit.centerlines, fit.radii, MIN_BEND_RADIUS / VOXEL, spacing=1.25 * DIAMETER / VOXEL
            ) if fit.centerlines else {}
            ends = end_errors(fit, scan)
            row = {
                "scenario": name, "fit": label, "true": len(scan.centerlines), "fitted": fit.fiber_count,
                "rec/split/miss/merged": f"{report['recovered']}/{report['split']}/{report['missed']}/{report['merged_fibers']}",
                "line_err": report["centerline_error_voxels"],
                "diam_bias_um": report["diameter_bias_m"] / um if report["diameter_bias_m"] is not None else None,
                "end_err_max": max(ends) if ends else None,
                **moves(fit),
                "overlaps": geometry.get("overlapping_pairs"),
                "over_bend": geometry.get("fibers_over_bend_limit"),
                "seconds": seconds,
            }
            rows.append(row)
            if fit.fiber_count:
                fit.write(OUTPUT / name / label, volume=scan.volume)
            (OUTPUT / name / label).mkdir(parents=True, exist_ok=True)
            (OUTPUT / name / label / "scenario.json").write_text(json.dumps({**row, "score": report}, indent=1, default=str) + "\n")
            print(row)
    (OUTPUT / "scenarios.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")

    columns = list(rows[0]) if rows else []
    print("\n" + " | ".join(columns))
    for row in rows:
        print(" | ".join(f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]) for c in columns))


if __name__ == "__main__":
    main()
