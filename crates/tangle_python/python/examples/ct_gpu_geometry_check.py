"""Does Tangle's solver, run with the scan as an extra force, actually make the
fitted fibers valid fibers? A check to run before trusting the GPU fitter.

It uses the single-type synthetic scan of ``ct_fit_synthetic.py`` (40 fibers
of 12 µm, bend radius 48 µm, 1.5 µm voxels), whose true fibers are known,
and runs three experiments through ``tangle.ImageRelaxer``:

1. **Geometry only.** Start from the CPU fit and relax with the image force
   off. Bend-limit violations and overlaps should go to about zero while
   the fibers barely move.
2. **Damaged truth.** Take the true fibers, add sharp random kinks (well past
   the bend limit, with overlaps where fibers touch) and relax with the image
   force on. The fibers should come back to the truth: valid geometry and a
   small centerline error against the true fibers.
3. **CPU fit, polished.** Start from the CPU fit and relax with the image
   force on. Geometry should become valid without the fit to the scan
   getting worse (recovery, voxel accuracy, centerline error).

For each run it prints ``ct.geometry_report`` before and after (curvature
ratio against the bend limit, deepest overlap, segment lengths) and writes
before/after overlays. Fibers are resampled to segments at least one
diameter long first: Tangle's contact treats non-adjacent segments of the
same fiber as colliding, so shorter segments push a fiber apart by itself.
Each fiber rests straight (see ``straightened``).

Usage: ``python ct_gpu_geometry_check.py [output_dir] [fit.json]``; without a
fit.json the CPU fit is run first (about a minute). ``CT_TRUTH_DIR`` names the
folder holding ``ct_fit_synthetic.py``'s cached ``truth_centerlines.json``.
Needs a Tangle build with ``ImageRelaxer`` and a working per-fiber curvature
solve; without it the bend limit stalls just above 1 and experiment 1 fails
for a reason unrelated to the image force.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import tangle
import tangle.ct as ct
from tangle.ct._geometry import rasterize, resample
from tangle.ct._image import normalize

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ct_fit_synthetic as synthetic  # noqa: E402  (same scan, same cached truth)

# Where ct_fit_synthetic.py cached truth_centerlines.json.
synthetic.OUTPUT = Path(os.environ.get("CT_TRUTH_DIR", HERE / "output" / "ct_fit_synthetic"))
OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "output" / "ct_gpu_geometry_check")
FIT_JSON = Path(sys.argv[2]) if len(sys.argv) > 2 else None
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")

VOXEL = synthetic.VOXEL
RADIUS = 0.5 * synthetic.DIAMETER / VOXEL  # voxels
BEND = synthetic.MIN_BEND_RADIUS / VOXEL  # voxels
ITERATIONS = int(os.environ.get("CHECK_ITERATIONS", "2000"))
IMAGE_RATE = float(os.environ.get("CHECK_IMAGE_RATE", "0.3"))
KINK = 0.35  # random sideways kick per node, in radii


def relax(image: np.ndarray, lines: list[np.ndarray], radii: np.ndarray, rate: float) -> tuple[list[np.ndarray], float]:
    """Tangle relaxation of voxel-unit fibers with the scan pulling at ``rate``.

    The one place this script touches ImageRelaxer; adjust here if its
    signature changes.
    """
    h = VOXEL
    cell = tangle.Cell([n * h for n in image.shape[::-1]])
    materials: dict[int, tangle.Material] = {}
    collection = tangle.FiberCollection("ct fit")
    for line, radius in zip(lines, radii):
        key = int(round(2.0 * radius * h / 1e-9))
        if key not in materials:
            materials[key] = tangle.Material(f"ct {key}nm", diameter=key * 1e-9, min_bend_radius=BEND * h)
        line = np.asarray(line, dtype=np.float64)
        collection.add_fiber((line * h).tolist(), materials[key], rest_centerline=(straightened(line) * h).tolist())
    assembly = tangle.Assembly(cell)
    assembly.insert(collection, name="ct fit")
    settings = tangle.RelaxationSettings(
        backend=BACKEND, max_iterations=ITERATIONS, max_step=0.25 * RADIUS * h, penetration_tolerance=0.02 * RADIUS * h,
        neighbor_capacity=192,  # fits start overlapping; 48 slots overflow into the slow fallback
    )
    started = time.perf_counter()
    relaxer = tangle.ImageRelaxer(
        assembly, settings, np.ascontiguousarray(image, dtype="<f4").tobytes(), tuple(int(n) for n in image.shape), h
    )
    relaxer.set_image_force(rate=rate, reach_radii=1.4)
    status = relaxer.run(ITERATIONS)
    print(f"  solver: {status}")
    out = [np.asarray(line, dtype=np.float64) / h for line in relaxer.centerlines()]
    return out, time.perf_counter() - started


def straightened(line: np.ndarray) -> np.ndarray:
    """A straight rest shape with the same segment lengths.

    Tangle takes a fiber's rest (intrinsic) shape from the centerline it is
    placed with and rejects one that bends past the bend limit; a kinked fit
    would also keep its kinks as the shape the bending force returns to. The
    fibers here are straight-spun, so they rest straight: bending then
    resists every curve, and the scan (and contact) supply the waviness.
    """
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    direction = line[-1] - line[0]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return line[0] + np.concatenate([[0.0], np.cumsum(lengths)])[:, None] * direction


def spaced(lines: list[np.ndarray]) -> list[np.ndarray]:
    """Segments of about 1.25 diameters, never shorter than one diameter."""
    return [resample(np.asarray(line, dtype=np.float64), 2.5 * RADIUS) for line in lines]


def kinked(lines: list[np.ndarray], seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for line in lines:
        line = line.copy()
        kick = rng.normal(0.0, KINK * RADIUS, size=line.shape)
        kick[[0, -1]] = 0.0
        out.append(line + kick)
    return out


def error_to(lines: list[np.ndarray], reference: list[np.ndarray]) -> float:
    """Mean distance from each fiber to the same-index reference fiber (voxels)."""
    from scipy.spatial import cKDTree

    distances = []
    for line, ref in zip(lines, reference):
        distances.append(cKDTree(resample(ref, 0.5)).query(resample(line, 0.5))[0])
    return float(np.mean(np.concatenate(distances)))


def show(title: str, report: dict) -> None:
    keys = ("curvature_ratio_max", "curvature_ratio_p95", "fibers_over_bend_limit", "max_penetration_radii",
            "overlapping_pairs", "min_segment_diameters", "total_length")
    print(f"  {title:>7}: " + ", ".join(f"{k} {report[k]:.3g}" if isinstance(report[k], float) else f"{k} {report[k]}" for k in keys))


def overlay(name: str, volume: np.ndarray, lines: list[np.ndarray], radii: np.ndarray) -> None:
    labels, _, _ = rasterize(volume.shape, lines, radii, signed=True)
    ct.save_overlay_figure(OUTPUT / f"{name}.png", volume, labels, title=name.replace("_", " "))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    synthetic.OUTPUT.mkdir(parents=True, exist_ok=True)
    truth_assembly = synthetic.load_or_make_truth()
    scan = ct.synthetic_ct(truth_assembly, VOXEL, seed=7)
    spec = ct.FiberSpec(diameter=synthetic.DIAMETER, min_bend_radius=synthetic.MIN_BEND_RADIUS)
    if FIT_JSON and FIT_JSON.exists():
        fit = ct.load_fit(FIT_JSON)
    else:
        fit = ct.fit_fibers(scan.volume, VOXEL, spec)
    image, _ = normalize(scan.volume, denoise_sigma=0.7, levels=fit.levels)
    results: dict[str, dict] = {}

    def run(name: str, start: list[np.ndarray], radii: np.ndarray, rate: float, reference=None) -> list[np.ndarray]:
        print(f"{name} (image rate {rate}, {ITERATIONS} iterations)")
        before = ct.geometry_report(start, radii, BEND)
        after_lines, seconds = relax(image, start, radii, rate)
        after = ct.geometry_report(after_lines, radii, BEND)
        show("before", before)
        show("after", after)
        entry = {"before": before, "after": after, "seconds": seconds,
                 "moved_voxels": error_to(after_lines, start)}
        print(f"  mean movement {entry['moved_voxels']:.2f} voxels, {seconds:.1f} s")
        if reference is not None:
            entry["error_to_truth_before"] = error_to(start, reference)
            entry["error_to_truth_after"] = error_to(after_lines, reference)
            print(f"  distance to the true fibers {entry['error_to_truth_before']:.2f} -> {entry['error_to_truth_after']:.2f} voxels")
        results[name] = entry
        return after_lines

    fit_lines = spaced(fit.centerlines)
    run("1_geometry_only", fit_lines, fit.radii, 0.0)

    truth_lines = spaced(scan.centerlines)
    damaged = kinked(truth_lines)
    overlay("2_damaged_truth_before", scan.volume, damaged, scan.radii)
    repaired = run("2_damaged_truth", damaged, scan.radii, IMAGE_RATE, reference=truth_lines)
    overlay("2_damaged_truth_after", scan.volume, repaired, scan.radii)

    overlay("3_cpu_fit_before", scan.volume, fit_lines, fit.radii)
    polished = run("3_cpu_fit_polished", fit_lines, fit.radii, IMAGE_RATE)
    overlay("3_cpu_fit_after", scan.volume, polished, fit.radii)
    for label, lines in (("before", fit_lines), ("after", polished)):
        report = ct.score(replace(fit, centerlines=lines, support=np.ones(len(lines))), scan)
        results["3_cpu_fit_polished"][f"score_{label}"] = {
            k: report[k] for k in ("recovered", "split", "missed", "voxel_label_accuracy", "centerline_error_voxels")
        }
        print(f"  score {label}: {results['3_cpu_fit_polished'][f'score_{label}']}")
    ct.save_overlay_figure(OUTPUT / "truth_overlay.png", scan.volume, scan.labels, title="ground truth")
    (OUTPUT / "geometry_check.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
