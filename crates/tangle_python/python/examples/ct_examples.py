"""Every ``tangle.ct`` example, fitted the same way, one result folder each.

Each example is a synthetic scan rendered from a Tangle structure whose true
fibers are known. A generous binary fiber mask is thresholded from the scan
(``scan.fiber_mask(level=MASK_LEVEL)``: fibers look a little thicker, as
with a real threshold), and the fibers are fitted from the mask with
Tangle's solver on the GPU. Every example writes the same files, and only
these, to ``<output>/<example>/`` (the folder is emptied first, so a re-run
replaces the old result):

* ``raw.tif``: the rendered scan (uint16);
* ``mask.tif``: the fiber mask the fit starts from (0/255);
* ``true.tif``: the true fibers, one color per fiber, over the scan (RGB);
* ``segment.tif``: the fitted fibers, one color per fiber, over the scan (RGB);
* ``diff.tif``: where the segmentation and the truth disagree, over the dimmed
  scan (RGB): red = true fiber the fit left empty (missed), blue = fit where
  there is no fiber (extra), yellow = fiber given to the wrong fiber;
* ``fit.json``: the fit (reload with ``ct.load_fit``);
* ``score.json``: the score against the truth, the geometry report and the
  run time.

``<output>/summary.md`` gets one row per example. Relaxed truth structures
are cached in ``<output>/.cache/``.

Usage::

    python ct_examples.py [--output DIR] [--list] [example ...]

Without names every example runs. The output folder defaults to
``$TANGLE_CT_OUTPUT``, else ``examples/output/ct``. ``TANGLE_BACKEND`` picks
the solver backend (``wgpu``, the GPU, by default).

To add an example, write a function that returns an :class:`Example` and
register it in ``EXAMPLES``; the runner does the rest, so every example
keeps the same layout.

Examples:

* ``single_type``: 40 wavy planar 12 µm fibers in a closed 240 µm cell.
* ``long_fibers``: 300-450 µm fibers in a 480 µm cell periodic in the fiber
  plane, cropped to the central 240 µm, so most fibers cross the scan
  boundary (the case for the fiber-length prior).
* ``two_types``: 7 µm solid fibers and 19 µm fibers with a bright rim and a
  dim core (the core comes out as a hole in the mask, which is filled);
  each fit's type is chosen by its thickness.
* ``scenario_*``: one small scan per fitting step, a few 12 µm fibers
  placed by hand in a 150 µm box:

  | scenario | step it exercises | right answer |
  |---|---|---|
  | ``single_straight`` | seeding, tracing | one fit end to end |
  | ``interior_ends`` | end growth and trimming | fit ends at the true ends |
  | ``gap_2r`` / ``gap_6r`` / ``gap_12r`` | joins | two collinear pieces; joined or not |
  | ``crossing_90`` / ``crossing_30`` | tracing through a crossing | two fits, none merged |
  | ``parallel_touching`` | one fiber or two | two fits |
  | ``short_piece`` | removal | no fit (2 diameters, below the minimum length) |
  | ``bent_near_limit`` | kink splits | one fit, not cut (bends at 1.25x the bend radius) |
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import tangle
import tangle.ct as ct
from tangle.ct._fit import _write_stack
from tangle.units import um

BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")
MASK_LEVEL = 0.35  # of the way from void to fiber grey level: a generous threshold
FILES = ("raw.tif", "mask.tif", "true.tif", "segment.tif", "diff.tif", "fit.json", "score.json")
DIFF_COLORS = {"missed": (230, 50, 50), "extra": (60, 120, 255), "wrong_fiber": (255, 200, 0)}


@dataclass
class Example:
    scan: ct.SyntheticScan
    spec: ct.FiberSpec | list[ct.FiberSpec]
    min_bend_radius: float  # of the smallest type, for the geometry report (m)
    extra: Callable[[ct.FitResult, ct.SyntheticScan], dict] | None = None


# -- shared helpers -----------------------------------------------------------


def relaxed_truth(cache: Path, cell: tangle.Cell, populations: list[tangle.FiberPopulation]) -> tangle.Assembly:
    """Relax the populations once and cache the centerlines (the slow step)."""
    if not cache.exists():
        recipe = tangle.Recipe(cell)
        for index, population in enumerate(populations):
            recipe.insert(tangle.generate_fiber_population(cell, population), name=f"truth {index}")
        started = time.perf_counter()
        run = recipe.run(
            tangle.RelaxationSettings(
                backend=BACKEND, max_iterations=12_000, max_step=1.0 * um, penetration_tolerance=0.1 * um
            )
        )
        print(f"  truth relaxed: {run} ({time.perf_counter() - started:.0f} s)")
        cache.parent.mkdir(parents=True, exist_ok=True)
        counts = [population.count for population in populations]
        cache.write_text(json.dumps({"counts": counts, "centerlines": run.centerlines()}) + "\n")
    data = json.loads(cache.read_text())
    data.setdefault("counts", [len(data["centerlines"])])  # single-population caches from older examples
    assembly = tangle.Assembly(cell)
    start = 0
    for population, count in zip(populations, data["counts"]):
        lines = data["centerlines"][start : start + count]
        start += count
        # float32 GPU relaxation can end a hair past the bend limit, which
        # export_puma's validator rejects; render with a 1% looser limit.
        material = population.material
        looser = tangle.Material(
            material.name, diameter=material.diameter, min_bend_radius=0.99 * material.min_bend_radius
        )
        assembly.insert(tangle.FiberCollection.from_centerlines(lines, looser), name=material.name)
    return assembly


def planar_population(material, count, seed, length, segments) -> tangle.FiberPopulation:
    return tangle.FiberPopulation(
        material=material,
        count=count,
        segments_per_fiber=segments,  # keeps segments longer than a diameter
        seed=seed,
        length=length,
        curvature_amplitude=(2 * um, 8 * um),
        orientation=tangle.PlanarOrientation(max_tilt=0.35),
    )


def end_errors(fit: ct.FitResult, scan: ct.SyntheticScan) -> list[float]:
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


def end_error_report(fit: ct.FitResult, scan: ct.SyntheticScan) -> dict:
    errors = end_errors(fit, scan)
    return {"end_error_max_voxels": max(errors) if errors else None, "end_errors_voxels": errors}


# -- the examples -------------------------------------------------------------


def single_type(cache: Path) -> Example:
    diameter, bend = 12 * um, 48 * um
    material = tangle.Material("fiber_12um", diameter=diameter, min_bend_radius=bend)
    cell = tangle.Cell([240 * um] * 3)
    truth = relaxed_truth(cache, cell, [planar_population(material, 40, 7, (150 * um, 210 * um), 12)])
    scan = ct.synthetic_ct(truth, 1.5 * um, seed=7)
    spec = ct.FiberSpec(diameter=diameter, min_bend_radius=bend, length=180 * um, name="fiber_12um")
    return Example(scan, spec, bend, end_error_report)


def long_fibers(cache: Path) -> Example:
    diameter, bend, voxel = 12 * um, 48 * um, 1.5 * um
    cell_side, crop = 480 * um, 240 * um
    material = tangle.Material("fiber_12um", diameter=diameter, min_bend_radius=bend)
    cell = tangle.Cell([cell_side] * 3, periodic="xy")
    truth = relaxed_truth(cache, cell, [planar_population(material, 154, 11, (300 * um, 450 * um), 24)])
    full = ct.synthetic_ct(truth, voxel, seed=11)
    low = int(round((cell_side - crop) / 2 / voxel))
    scan = full.crop((low,) * 3, (low + int(round(crop / voxel)),) * 3)
    spec = ct.FiberSpec(diameter=diameter, min_bend_radius=bend, length=375 * um, name="fiber_12um")
    return Example(scan, spec, bend, end_error_report)


def two_types(cache: Path) -> Example:
    voxel, cell_side, crop, length = 1.25 * um, 320 * um, 200 * um, (300 * um, 500 * um)
    fine = tangle.Material("fine_7um", diameter=7 * um, min_bend_radius=35 * um)
    coarse = tangle.Material("coarse_19um", diameter=19 * um, min_bend_radius=95 * um)
    cell = tangle.Cell([cell_side] * 3, periodic="xy")
    truth = relaxed_truth(
        cache, cell, [planar_population(fine, 106, 21, length, 16), planar_population(coarse, 14, 22, length, 16)]
    )
    # Grey levels as in the scans this imitates: small fibers brightest (1),
    # large fibers a rim at 0.75 around a core at 0.25.
    profiles = [
        (7 * um, ct.CrossSection()),
        (19 * um, ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3)),
    ]
    full = ct.synthetic_ct(truth, voxel, seed=21, profiles=profiles)
    low = int(round((cell_side - crop) / 2 / voxel))
    scan = full.crop((low,) * 3, (low + int(round(crop / voxel)),) * 3)
    specs = [
        ct.FiberSpec(diameter=7 * um, min_bend_radius=35 * um, length=400 * um, name="fine_7um"),
        ct.FiberSpec(diameter=19 * um, min_bend_radius=95 * um, length=400 * um, name="coarse_19um"),
    ]
    return Example(scan, specs, 35 * um, lambda fit, scan: {"per_type": ct.score(fit, scan)["per_type"]})


BOX = 150 * um
DIAMETER = 12 * um
R = DIAMETER / 2
MIN_BEND_RADIUS = 4 * DIAMETER
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


def scenario(centerlines) -> Callable[[Path], Example]:
    def build(cache: Path) -> Example:
        material = tangle.Material("fiber_12um", diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS)
        assembly = tangle.Assembly(tangle.Cell([BOX] * 3))
        assembly.insert(tangle.FiberCollection.from_centerlines(centerlines, material), name="scenario")
        scan = ct.synthetic_ct(assembly, 1.5 * um, seed=5)
        spec = ct.FiberSpec(diameter=DIAMETER, min_bend_radius=MIN_BEND_RADIUS, length=400 * um, name="fiber_12um")
        return Example(scan, spec, MIN_BEND_RADIUS, end_error_report)

    return build


EXAMPLES: dict[str, Callable[[Path], Example]] = {
    "single_type": single_type,
    "long_fibers": long_fibers,
    "two_types": two_types,
    **{f"scenario_{name}": scenario(lines) for name, lines in SCENARIOS.items()},
}


# -- the runner -----------------------------------------------------------------


def label_diff(scan: ct.SyntheticScan, fit_labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """RGB stack of where the fit and the truth disagree, and voxel counts.

    Each fitted fiber is matched to the true fiber it overlaps most; a voxel
    both call fiber is "wrong_fiber" when its fit is matched to another one.
    """
    truth = np.asarray(scan.labels)
    both = (truth > 0) & (fit_labels > 0)
    mapping = np.zeros(int(fit_labels.max()) + 1, dtype=truth.dtype)
    if both.any():
        pairs = fit_labels[both].astype(np.int64) * (int(truth.max()) + 1) + truth[both]
        values, counts = np.unique(pairs, return_counts=True)
        fits, trues = np.divmod(values, int(truth.max()) + 1)
        order = np.lexsort((counts, fits))  # by fit, then count ascending
        mapping[fits[order]] = trues[order]  # the last (largest count) write wins per fit
    classes = {
        "missed": (truth > 0) & (fit_labels == 0),
        "extra": (truth == 0) & (fit_labels > 0),
        "wrong_fiber": both & (mapping[fit_labels] != truth),
    }
    volume = np.asarray(scan.volume, dtype=np.float32)
    low, high = np.percentile(volume[:: max(1, volume.shape[0] // 32)], [0.5, 99.5])
    grey = (np.clip((volume - low) / max(high - low, 1e-6), 0.0, 1.0) * 110).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=-1)
    for name, where in classes.items():
        rgb[where] = DIFF_COLORS[name]
    fiber = max(int((truth > 0).sum()), 1)
    counts = {f"{name}_voxels": int(where.sum()) for name, where in classes.items()}
    counts.update({f"{name}_fraction_of_true_fiber": int(where.sum()) / fiber for name, where in classes.items()})
    return rgb, counts


def run(name: str, output: Path) -> dict:
    print(f"{name}:")
    example = EXAMPLES[name](output / ".cache" / f"{name}.json")
    scan = example.scan
    h = scan.voxel_size
    mask = scan.fiber_mask(level=MASK_LEVEL)
    started = time.perf_counter()
    fit = ct.fit_fibers(mask, h, example.spec, ct.FitSettings(backend=BACKEND))
    seconds = time.perf_counter() - started

    report = ct.score(fit, scan)
    smallest = min(s.diameter for s in (example.spec if isinstance(example.spec, list) else [example.spec]))
    geometry = (
        ct.geometry_report(fit.centerlines, fit.radii, example.min_bend_radius / h, spacing=1.25 * smallest / h)
        if fit.centerlines
        else {}
    )
    extra = example.extra(fit, scan) if example.extra else {}

    folder = output / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    _write_stack(folder / "raw", scan.volume, h)
    _write_stack(folder / "mask", mask.astype(np.uint8) * 255, h)
    _write_stack(folder / "true", ct.overlay_volume(scan.volume, scan.labels), h, rgb=True)
    fit_labels = fit.label_volume()
    _write_stack(folder / "segment", ct.overlay_volume(scan.volume, fit_labels), h, rgb=True)
    diff, diff_counts = label_diff(scan, fit_labels)
    _write_stack(folder / "diff", diff, h, rgb=True)
    (folder / "fit.json").write_text(json.dumps(fit.to_dict(), indent=1) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("per_true_fiber", "per_type")}
    (folder / "score.json").write_text(
        json.dumps(
            {"seconds": seconds, "score": report, "geometry": geometry, "diff": diff_counts, **extra},
            indent=1, default=str,
        ) + "\n"
    )
    row = {
        "example": name,
        "true": summary["true_fibers_in_volume"],
        "fitted": summary["fitted_fibers"],
        "recovered": summary["recovered"],
        "split": summary["split"],
        "missed": summary["missed"],
        "false": summary["false_fibers"],
        "merged": summary["merged_fibers"],
        "line error (vox)": summary["centerline_error_voxels"],
        "diameter bias (um)": summary["diameter_bias_m"] / um if summary["diameter_bias_m"] is not None else None,
        "label accuracy": summary["voxel_label_accuracy"],
        "missed / extra / wrong (% of fiber)": "/".join(
            f"{100 * diff_counts[f'{k}_fraction_of_true_fiber']:.1f}" for k in ("missed", "extra", "wrong_fiber")
        ),
        "end error max (vox)": extra.get("end_error_max_voxels"),
        "overlaps": geometry.get("overlapping_pairs"),
        "over bend limit": geometry.get("fibers_over_bend_limit"),
        "seconds": seconds,
    }
    print("  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return row


def write_summary(output: Path, rows: list[dict]) -> None:
    """Merge ``rows`` into ``summary.md`` (one row per example, in registry order)."""
    store = output / ".cache" / "summary.json"
    known = json.loads(store.read_text()) if store.exists() else {}
    known.update({row["example"]: row for row in rows})
    known = {name: known[name] for name in EXAMPLES if name in known}
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(known, indent=1, default=str) + "\n")
    columns = list(next(iter(known.values())))

    def cell(value) -> str:
        if value is None:
            return "-"
        return f"{value:.3g}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for row in known.values()]
    header = (
        "# tangle.ct examples\n\n"
        f"Fitted from binary masks thresholded at {MASK_LEVEL} of the way from void to fiber. "
        "Each example's folder holds raw.tif, mask.tif, true.tif, segment.tif, diff.tif, fit.json and score.json. "
        "diff.tif: red = missed, blue = extra, yellow = wrong fiber.\n\n"
    )
    (output / "summary.md").write_text(header + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="examples to run (default: all)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="list the examples and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(EXAMPLES))
        return
    unknown = [name for name in args.names if name not in EXAMPLES]
    if unknown:
        parser.error(f"unknown examples: {', '.join(unknown)} (see --list)")
    output = args.output or Path(os.environ.get("TANGLE_CT_OUTPUT", Path(__file__).with_name("output") / "ct"))
    output.mkdir(parents=True, exist_ok=True)
    rows = [run(name, output) for name in (args.names or list(EXAMPLES))]
    write_summary(output, rows)
    print(f"summary: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
