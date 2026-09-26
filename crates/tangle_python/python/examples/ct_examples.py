"""Every ``tangle.ct`` example, fitted the same way, one result folder each.

Each example is a synthetic scan rendered from a Tangle structure whose true
fibers are known. By default the raw scan is fitted, with a grey range per
fiber type (``FiberSpec.profile``: its grey at 9 radii from the axis to
the surface, measured around the true fibers as one would on a few fibers
of a real scan). With ``--input mask`` a generous binary
mask is fitted instead (``scan.fiber_mask(level=MASK_LEVEL)``: fibers look
a little thicker, as with a real threshold). With ``--input plain`` the raw
scan is fitted from the diameters alone, with no grey information, as a
first fit of a new scan would be. Without ``--input`` each example uses its
own input: grey profiles, except ``noisy_two_types`` (plain). The summary
records each example's input. The fibers are fitted with
Tangle's solver on the GPU. Every example writes the same files, and only
these, to ``<output>/<example>/`` (the folder is emptied first, so a re-run
replaces the old result):

* ``raw.tif``: the rendered scan (uint16);
* ``input.tif``: what the fit sees, 0 (void) to 255 (fiber): the grey
  profiles' fiber fraction per voxel, or the mask;
* ``true.tif``: the true fibers, one color per fiber, over the scan (RGB);
* ``segment.tif``: the fitted fibers, one color per fiber, over the scan (RGB);
* ``diff.tif``: where the segmentation and the truth disagree, over the dimmed
  scan (RGB): red = true fiber the fit left empty (missed), blue = fit where
  there is no fiber (extra), orange = fiber given to the wrong fiber;
* ``confidence.tif``: the fit's own confidence in each fitted voxel, over the
  dimmed scan (RGB): green = sure, through yellow, to red = unsure. It uses
  no ground truth; ``score.json`` records how well it predicts the errors
  in ``diff.tif``;
* ``fit.json``: the fit (reload with ``ct.load_fit``);
* ``score.json``: the score against the truth, the geometry report and the
  run time.

``<output>/summary.md`` gets one row per example. Relaxed truth structures
are cached in ``<output>/.cache/``.

Usage::

    python ct_examples.py [--output DIR] [--list] [--varied] [example ...]

Without names every example but the ``varied_*`` ones runs (``--varied``
adds them). The output folder defaults to
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
  dim core (the core falls below the grey range, or out of the mask, and
  the hole is filled);
  each fit's type is chosen by its thickness.
* ``noisy_two_types``: two_types' fibers in a low-contrast, noisy scan:
  the coarse fibers solid and dim (0.45 of the fine fibers'
  contrast), and the noise correlated over about a
  voxel, so the light denoise
  removes little of it. A threshold of such a scan leaves the coarse
  fibers full of holes. Fitted from the grey alone (``plain``) by default.
* ``halo_two_types``: two_types' fibers and grey levels with a
  phase-contrast halo (``synthetic_ct(phase_contrast=...)``): a shallow dark
  band outside every surface that noise breaks into spots, darkest in the
  gaps between touching fibers.
* ``scanner_two_types``: two_types' fibers scanned by a simulated scanner
  (``ct.Scanner``: projections, propagation phase contrast, detector blur,
  photon noise, filtered back-projection), so the noise texture, blur and
  edge fringes come from the acquisition, as in a real scan: 2 µm
  resolution, and each fiber moving about 1 µm during the scan. Both types
  are solid, the coarse ones a lighter material a bit over half as dense
  as the fine ones.
* ``bundled_two_types``: the fine fibers packed in bundles of nineteen, with
  staggered ends, and the coarse ones loose, scanned as scanner_two_types
  with the noise blotchy (correlated by the scintillator's blur).
* ``varied_1`` … ``varied_8``: fresh structures drawn from seeds, for
  checking the fitter on structures it was not tuned on: 8-16 µm fibers
  at 2.5-4.5 voxels radius, planar, aligned, biaxial and isotropic (two
  each), solid fraction 0.04-0.14 (single_type is about 0.06), varied waviness, bend limit, scan noise and blur,
  160 voxels a side. ``score.json`` records each one's settings
  (``varied_settings``). Not in the default run.
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
  | ``missed_fiber`` | a fiber in single_type's densest bundle, touching 11 others at 4-88° | 12 fits, the center one whole (single_type split it in two) |
  | ``dense_crossing`` | varied_3's worst merges: 67 thin (3-voxel radius) fibers in cross-ply layers, noisy, in a 64-voxel cube | each fiber its own fit (the first fit merged 31 of them) |
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

import tangle
import tangle.ct as ct
from tangle.ct._fit import _write_stack
from tangle.units import um

BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")
MASK_LEVEL = 0.35  # of the way from void to fiber grey level: a generous threshold
# "grey" (raw scan + profiles), "plain" (raw scan alone) or "mask" for every example; None: each example's own (grey by default)
INPUT = os.environ.get("TANGLE_CT_INPUT")
BLUR = None  # scan blur (PSF sigma, voxels) for every example; None keeps each example's own
DIFF_COLORS = {"missed": (230, 50, 50), "extra": (60, 120, 255), "wrong_fiber": (255, 200, 0)}
FILES = ("raw.tif", "input.tif", "true.tif", "segment.tif", "diff.tif", "confidence.tif", "fit.json", "score.json")


def render_scan(*args, **kwargs) -> ct.SyntheticScan:
    """``ct.synthetic_ct``, with the scan blur set by ``--blur`` if given."""
    if BLUR is not None:
        kwargs["psf_sigma_voxels"] = BLUR
    return ct.synthetic_ct(*args, **kwargs)


@dataclass
class Example:
    scan: ct.SyntheticScan
    spec: ct.FiberSpec | list[ct.FiberSpec]
    min_bend_radius: float  # of the smallest type, for the geometry report (m)
    extra: Callable[[ct.FitResult, ct.SyntheticScan], dict] | None = None
    input: str = "grey"  # what it is fitted from, unless --input says otherwise


# -- shared helpers -----------------------------------------------------------


def relaxed_truth(cache: Path, cell: tangle.Cell, populations: list) -> tangle.Assembly:
    """Relax the populations once and cache the centerlines (the slow step).

    Each population is a ``tangle.FiberPopulation`` or a ready-made
    ``(FiberCollection, Material)`` pair (see bundles).
    """
    if not cache.exists():
        recipe = tangle.Recipe(cell)
        for index, population in enumerate(populations):
            if isinstance(population, tangle.FiberPopulation):
                fibers = tangle.generate_fiber_population(cell, population)
            else:
                fibers = population[0]
            recipe.insert(fibers, name=f"truth {index}")
        started = time.perf_counter()
        run = recipe.run(
            tangle.RelaxationSettings(
                backend=BACKEND, max_iterations=12_000, max_step=1.0 * um, penetration_tolerance=0.1 * um
            )
        )
        print(f"  truth relaxed: {run} ({time.perf_counter() - started:.0f} s)")
        cache.parent.mkdir(parents=True, exist_ok=True)
        counts = [p.count if isinstance(p, tangle.FiberPopulation) else len(p[0]) for p in populations]
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
        material = population.material if isinstance(population, tangle.FiberPopulation) else population[1]
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
    scan = render_scan(truth, 1.5 * um, seed=7)
    spec = ct.FiberSpec(diameter=diameter, min_bend_radius=bend, length=180 * um, name="fiber_12um")
    return Example(scan, spec, bend, end_error_report)


def long_fibers(cache: Path) -> Example:
    diameter, bend, voxel = 12 * um, 48 * um, 1.5 * um
    cell_side, crop = 480 * um, 240 * um
    material = tangle.Material("fiber_12um", diameter=diameter, min_bend_radius=bend)
    cell = tangle.Cell([cell_side] * 3, periodic="xy")
    truth = relaxed_truth(cache, cell, [planar_population(material, 154, 11, (300 * um, 450 * um), 24)])
    full = render_scan(truth, voxel, seed=11)
    low = int(round((cell_side - crop) / 2 / voxel))
    scan = full.crop((low,) * 3, (low + int(round(crop / voxel)),) * 3)
    spec = ct.FiberSpec(diameter=diameter, min_bend_radius=bend, length=375 * um, name="fiber_12um")
    return Example(scan, spec, bend, end_error_report)


# A simulated scanner (ct.Scanner): photon noise, detector blur and filtered
# back-projection, with a weakly absorbing, phase-shifting sample and a short
# propagation distance, so the edges show phase fringes; a 2 µm resolution
# softens the fibers, and each fiber moves about 1 µm during the scan. Both
# fiber types are solid; the coarse fibers are a lighter material, a bit over
# half as dense as the fine ones, so the fine fibers are the brightest thing
# in the scan, each with a dark band just outside it, and the coarse ones
# dim with a faint rim.
SCANNER = ct.Scanner(
    photons=1300,
    fiber_attenuation=0.005,
    resolution=2 * um,
    delta_beta=(12.0, 9.0),
    propagation=4.0,
    fiber_motion=1 * um,
)
SCANNER_PROFILES = [(7 * um, ct.CrossSection()), (19 * um, ct.CrossSection(brightness=0.55))]


def two_types(cache: Path, *, noisy: bool = False, halo: bool = False, scanner: bool = False) -> Example:
    voxel, cell_side, crop, length = 1.25 * um, 320 * um, 200 * um, (300 * um, 500 * um)
    fine = tangle.Material("fine_7um", diameter=7 * um, min_bend_radius=35 * um)
    coarse = tangle.Material("coarse_19um", diameter=19 * um, min_bend_radius=95 * um)
    cell = tangle.Cell([cell_side] * 3, periodic="xy")
    truth = relaxed_truth(
        cache, cell, [planar_population(fine, 106, 21, length, 16), planar_population(coarse, 14, 22, length, 16)]
    )
    if noisy:
        # Solid coarse fibers at 0.45 of the fine fibers' contrast, and noise
        # correlated over about a voxel.
        profiles = [(7 * um, ct.CrossSection()), (19 * um, ct.CrossSection(brightness=0.45))]
        full = render_scan(truth, voxel, seed=21, profiles=profiles, noise=0.19, noise_correlation=0.9)
    elif scanner:
        # two_types' fibers scanned by a simulated scanner (see SCANNER).
        full = render_scan(truth, voxel, seed=21, profiles=SCANNER_PROFILES, scanner=SCANNER)
    elif halo:
        # two_types' scan with a phase-contrast halo at unit strength (a
        # dark band outside every surface, deeper where surfaces face each
        # other).
        profiles = [
            (7 * um, ct.CrossSection()),
            (19 * um, ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3)),
        ]
        full = render_scan(truth, voxel, seed=21, profiles=profiles, phase_contrast=1.0)
    else:
        # Grey levels as in the scans this imitates: small fibers brightest (1),
        # large fibers a rim at 0.75 around a core at 0.25.
        profiles = [
            (7 * um, ct.CrossSection()),
            (19 * um, ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3)),
        ]
        full = render_scan(truth, voxel, seed=21, profiles=profiles)
    low = int(round((cell_side - crop) / 2 / voxel))
    scan = full.crop((low,) * 3, (low + int(round(crop / voxel)),) * 3)
    specs = [
        ct.FiberSpec(diameter=7 * um, min_bend_radius=35 * um, length=400 * um, name="fine_7um"),
        ct.FiberSpec(diameter=19 * um, min_bend_radius=95 * um, length=400 * um, name="coarse_19um"),
    ]
    return Example(scan, specs, 35 * um, lambda fit, scan: {"per_type": ct.score(fit, scan)["per_type"]})


def noisy_two_types(cache: Path) -> Example:
    example = two_types(cache.with_name("two_types.json"), noisy=True)  # the same fibers as two_types
    # Fitted from the grey alone, the case the noise breaks: a threshold punches
    # holes into the dim coarse fibers (see _fit._Fitter.classify).
    example.input = "plain"
    return example


def halo_two_types(cache: Path) -> Example:
    return two_types(cache.with_name("two_types.json"), halo=True)  # the same fibers as two_types


def scanner_two_types(cache: Path) -> Example:
    return two_types(cache.with_name("two_types.json"), scanner=True)  # the same fibers as two_types


def bundles(cell: tangle.Cell, leaders: tangle.FiberPopulation, per_bundle: int):
    """(FiberCollection, material) for relaxed_truth: a bundle of ``per_bundle`` fibers along each of ``leaders``.

    Each bundle follows one fiber drawn from ``leaders``, its members
    hexagonally packed around it (a center and a ring of six, then the next
    ring), a hair apart, with their ends staggered by up to a fifth of the
    length, as the filaments of a yarn or tow lie.
    """
    material, seed = leaders.material, leaders.seed
    leaders = tangle.generate_fiber_population(cell, leaders)
    rng = np.random.default_rng(seed)
    pitch = 1.02 * material.diameter
    slots = [(0.0, 0.0)]
    ring = 1
    while len(slots) < per_bundle:
        corners = [ring * np.array([np.cos(a), np.sin(a)]) for a in np.arange(6) * np.pi / 3]
        for k in range(6):
            for step in range(ring):
                slots.append(tuple(corners[k] + (corners[(k + 1) % 6] - corners[k]) * step / ring))
        ring += 1
    bottom, top = 0.5 * material.diameter, cell.lengths[2] - 0.5 * material.diameter
    lines = []
    for leader in leaders.centerlines():
        path = np.asarray(leader)
        tangent = np.gradient(path, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        # Across the bundle: one axis toward z (out of the fibers' plane), one in it.
        up = np.array([0.0, 0.0, 1.0]) - tangent[:, 2:3] * tangent
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        side = np.cross(tangent, up)
        for a, b in slots[:per_bundle]:
            member = path + pitch * (a * side + b * up)
            member[:, 2] = np.clip(member[:, 2], bottom, top)
            cut = rng.integers(0, len(member) // 10 + 1, size=2)
            lines.append(member[cut[0] : len(member) - cut[1]].tolist())
    return tangle.FiberCollection.from_centerlines(lines, material), material


def bundled_two_types(cache: Path) -> Example:
    """two_types with the fine fibers in bundles of nineteen, scanned by SCANNER with blotchy noise."""
    voxel, cell_side, crop, length = 1.25 * um, 320 * um, 200 * um, (300 * um, 500 * um)
    fine = tangle.Material("fine_7um", diameter=7 * um, min_bend_radius=35 * um)
    coarse = tangle.Material("coarse_19um", diameter=19 * um, min_bend_radius=95 * um)
    cell = tangle.Cell([cell_side] * 3, periodic="xy")
    truth = relaxed_truth(
        cache, cell, [bundles(cell, planar_population(fine, 6, 23, length, 16), 19), planar_population(coarse, 14, 22, length, 16)]
    )
    # The scintillator spreads each counted photon over about a pixel, so the
    # noise comes out blotchy instead of pixel to pixel; that blur also
    # averages the noise down, so fewer photons keep the same contrast to noise.
    scanner = replace(SCANNER, noise_blur=1.0, photons=56)
    full = render_scan(truth, voxel, seed=23, profiles=SCANNER_PROFILES, scanner=scanner)
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


# single_type's densest bundle: true fiber 35 (first) and the 11 fibers touching it,
# in µm in a 150 µm cube centered on it (single_type fit split fiber 35 in two): the relaxed
# truth's own vertices, each fiber cut to its longest run inside the cube (resampling sharpened
# the bends past the limit).
MISSED_FIBER_BUNDLE_UM = [
    [[143.33, 103.44, 105.3], [131.77, 94.24, 97.81], [118.73, 88.52, 89.23], [104.42, 84.76, 81.56], [88.66, 80.09, 79.13], [72.03, 77.48, 80.57], [56.12, 73.75, 76.24], [43.3, 67.73, 67.49], [34.11, 60.55, 55.57], [22.0, 53.15, 46.82], [8.75, 43.83, 43.31]],  # 35
    [[125.61, 145.92, 35.48], [118.86, 130.72, 40.09], [110.1, 115.79, 41.72], [99.63, 102.02, 43.58], [88.08, 90.18, 48.57], [75.21, 80.94, 56.71], [64.89, 70.57, 66.9], [57.47, 56.32, 73.35], [51.37, 39.99, 74.04], [42.87, 25.59, 78.97], [31.14, 12.86, 78.87]],  # 4
    [[90.55, 87.06, 95.78], [99.44, 76.55, 90.81], [109.61, 66.33, 89.31], [118.73, 55.29, 86.94], [126.95, 43.98, 83.02], [134.82, 32.1, 80.39], [141.57, 20.08, 75.5], [148.53, 8.94, 69.05]],  # 5
    [[138.58, 117.79, 115.22], [128.95, 105.9, 107.4], [118.12, 93.7, 102.92], [107.71, 80.32, 104.29], [94.15, 70.35, 101.81], [80.67, 63.5, 94.23], [66.56, 56.68, 87.12], [51.12, 49.25, 85.22], [38.21, 38.62, 82.67], [27.57, 27.95, 74.81], [14.7, 18.86, 68.53], [3.7, 7.3, 63.04]],  # 7
    [[139.08, 110.36, 130.38], [129.65, 102.09, 120.69], [117.38, 93.83, 115.35], [103.03, 87.36, 114.08], [90.03, 80.98, 107.8], [77.28, 76.1, 100.0], [63.53, 70.61, 94.24], [50.97, 65.88, 85.74], [36.94, 60.18, 81.37], [22.09, 54.85, 81.58], [8.53, 47.24, 79.04]],  # 11
    [[22.18, 41.24, 79.2], [35.11, 48.43, 73.88], [46.4, 53.29, 64.32], [60.17, 58.4, 59.01], [74.91, 63.57, 58.68], [87.58, 72.21, 61.41], [101.27, 79.7, 59.34], [115.09, 85.51, 54.6], [128.86, 92.19, 52.0], [142.37, 99.9, 51.25]],  # 17
    [[149.22, 105.38, 92.89], [133.06, 107.85, 92.49], [117.04, 109.22, 95.2], [100.75, 110.4, 97.09], [84.89, 111.88, 101.19], [69.22, 115.92, 103.26], [53.0, 117.92, 102.97], [36.83, 119.39, 104.74], [20.76, 122.02, 105.35], [4.34, 121.51, 105.42]],  # 23
    [[98.02, 3.89, 37.57], [88.28, 13.16, 43.76], [79.11, 24.58, 45.96], [71.15, 36.69, 43.89], [61.53, 47.41, 41.09], [51.16, 57.67, 42.69], [41.02, 67.04, 47.48], [29.29, 75.53, 50.31]],  # 26
    [[7.56, 64.15, 60.98], [13.52, 50.67, 59.29], [18.04, 37.93, 53.33], [24.93, 27.9, 44.42], [34.26, 17.13, 39.5], [42.39, 4.86, 38.55]],  # 28
    [[93.1, 2.49, 87.9], [100.7, 13.58, 92.66], [109.49, 24.56, 94.49], [119.08, 35.38, 94.37], [128.82, 45.98, 95.75], [137.54, 57.18, 95.91], [145.29, 68.46, 99.91]],  # 29
    [[56.07, 5.45, 67.27], [48.17, 17.15, 66.71], [39.42, 28.28, 66.74], [32.27, 40.02, 63.63], [25.52, 52.71, 63.35], [22.65, 66.81, 63.33], [20.01, 80.09, 59.47], [15.4, 93.1, 56.4], [8.36, 105.34, 55.81]],  # 32
    [[143.16, 128.25, 109.31], [128.81, 126.77, 107.01], [114.61, 122.71, 105.11], [99.97, 122.47, 102.4], [86.28, 123.84, 97.48], [71.9, 124.18, 94.58], [57.37, 126.12, 94.19], [42.76, 126.13, 93.76], [28.45, 122.12, 93.1]],  # 36
]


# varied_3's worst merged region: a 64-voxel cube (106.176 um) at voxel [50, 59, 60] of varied_3,
# holding 37% of that fit's wrong-fiber voxels. The relaxed truth's own vertices, every periodic
# image cut to its longest run inside the cube, in um from the cube's corner; "merged" marks the
# 31 true fibers a merged or false fit joined there. The number is varied_3's truth label.
DENSE_CROSSING_UM = [
    [[95.72, 21.73, 101.72], [85.39, 30.71, 104.41], [74.03, 38.8, 104.36], [61.45, 44.47, 102.13]],  # 2 merged
    [[104.6, 29.08, 80.12], [97.38, 16.92, 75.17], [88.86, 4.8, 72.53]],  # 4
    [[6.2, 43.15, 72.75], [21.93, 39.26, 69.02], [38.03, 35.59, 68.29], [52.78, 28.1, 67.11], [64.61, 17.15, 63.24], [73.12, 4.77, 55.88]],  # 18 merged
    [[90.92, 103.09, 10.05], [82.5, 90.51, 16.14], [73.95, 79.02, 23.9], [65.87, 65.33, 27.56], [58.52, 50.83, 26.86], [52.39, 35.76, 26.08]],  # 19 merged
    [[103.72, 92.61, 10.25], [94.72, 78.98, 6.94], [82.26, 67.33, 7.41], [68.57, 59.05, 12.06], [58.13, 46.57, 15.92], [50.82, 31.65, 16.15], [46.57, 15.84, 11.98]],  # 22 merged
    [[40.57, 36.51, 36.77], [51.11, 27.94, 32.45], [62.36, 19.3, 31.97], [73.71, 11.42, 35.32], [84.5, 5.2, 42.16], [96.7, 2.37, 49.05]],  # 23 merged
    [[97.77, 10.84, 7.8], [102.98, 23.47, 1.43]],  # 24
    [[33.95, 20.58, 62.33], [30.45, 6.8, 63.08]],  # 25
    [[62.43, 99.75, 56.78], [75.97, 91.67, 58.31], [89.06, 82.42, 56.72], [100.38, 73.17, 50.59]],  # 27
    [[86.35, 73.46, 27.56], [80.59, 58.82, 34.87], [74.84, 42.65, 36.79], [66.89, 27.46, 39.05], [57.64, 14.4, 45.54], [48.87, 0.35, 50.79]],  # 28
    [[62.57, 55.65, 37.03], [68.32, 71.09, 44.07], [77.61, 85.61, 46.78], [90.97, 97.12, 46.83]],  # 29 merged
    [[104.95, 44.49, 68.7], [98.28, 31.42, 64.51], [92.19, 17.64, 62.12], [87.62, 3, 62.62]],  # 30
    [[97.89, 5.29, 101.58], [83, 13.23, 100.73], [68.57, 21.46, 97.67], [54.88, 28.74, 90.81], [42.65, 37.63, 83.29], [28.17, 44.01, 77.3], [13.08, 51.33, 75.23]],  # 35 merged
    [[71.31, 83.44, 102.21], [85.9, 76.66, 101.27], [99.6, 69, 105.21]],  # 37
    [[69.75, 8.15, 99.94], [57.14, 18.25, 100.77], [42, 23.99, 101.13]],  # 40 merged
    [[64.01, 93.54, 47.82], [60.96, 79.71, 49.17], [57.56, 66.01, 46.64], [56.8, 51.76, 47.01], [54.71, 37.89, 51.19], [54.9, 24.19, 55.13], [58.33, 10.37, 56.8]],  # 42 merged
    [[93.28, 58.07, 105.21], [79.75, 62.94, 99.17], [65.56, 68.76, 96.43], [52.1, 76.55, 95.57], [39.37, 85.39, 96.94], [26.86, 94.69, 97.99], [14.52, 103.73, 94.95]],  # 45
    [[30.07, 97.53, 50.65], [22.25, 84.87, 46.76], [15.16, 71.29, 47.19], [7.09, 57.95, 47.99]],  # 46 merged
    [[33.2, 29.43, 92.14], [28.91, 16.16, 85.78], [22.8, 2.48, 83.5]],  # 52
    [[18.63, 80.15, 1.03], [31.94, 76.53, 0.44], [45.7, 75.8, 1.79], [58.63, 75.68, 6.68], [70.17, 73.23, 14.15], [81.56, 67.64, 19.61], [92.17, 59.31, 22.72], [104.15, 53.02, 25.43]],  # 54
    [[44.11, 42.65, 25.14], [38.06, 26.86, 26.53], [35.15, 10.47, 29.68]],  # 55
    [[44.37, 99.07, 38.42], [34.36, 86.15, 42.32], [27.08, 72.67, 49.4], [17.74, 59.83, 54.8], [7.15, 48.01, 60.29]],  # 57
    [[26.74, 40.12, 99.95], [16.61, 28.05, 97.71], [6.92, 15.72, 100.32]],  # 63
    [[102.42, 75.75, 93.98], [95.53, 62.28, 92.81], [87.48, 50.11, 95.65], [80.05, 37.14, 94.61], [73.45, 24.83, 89.38], [63.96, 15.12, 82.21], [52.99, 5.5, 79.09]],  # 68
    [[73.58, 100.37, 69.56], [66.07, 89.42, 66.21], [58.85, 79.86, 59.54], [52, 68.87, 55.09], [47.65, 56.3, 51.56], [43.06, 43.59, 49.44], [35.82, 32.36, 46.4], [28.1, 22.41, 41.05], [22.47, 11.07, 35.73]],  # 69 merged
    [[6.77, 18.34, 13.34], [17.67, 10.4, 6.03], [30.84, 4.26, 2.14], [45.32, 0.08, 1.93]],  # 72 merged
    [[44.29, 5.05, 90.42], [32.35, 13.72, 95.26], [22.19, 21.6, 104.29]],  # 77
    [[98.99, 10.28, 84.79], [88.07, 18.55, 84.2], [78.45, 27.81, 80.77], [67.76, 36.45, 80.8], [56.3, 43.75, 84.22], [46.19, 50.19, 90.92], [35.48, 55.95, 97.37], [23.69, 58.58, 103.87]],  # 80
    [[15.05, 22.75, 4.06], [30.2, 15.55, 5.35], [43.11, 6.23, 11.01], [56.8, 0.56, 19.01]],  # 82 merged
    [[8.87, 102.82, 104.18], [0.32, 89.73, 97.08]],  # 84
    [[48.72, 87.34, 7.25], [40.03, 74.6, 13.65], [32.63, 60.33, 18.43], [24.11, 48.46, 26.59], [17.22, 35.1, 34.42], [10.35, 20.08, 37.26], [3.9, 4.79, 34.78]],  # 86
    [[48.95, 73.95, 68.42], [65, 70.28, 65.93], [80.18, 69.05, 59.97], [94.09, 63.12, 53.42]],  # 91
    [[15.79, 1.89, 63.3], [22.3, 13.87, 59.78], [27.29, 26.78, 55.63], [31.64, 40.14, 54.49], [35.56, 52.91, 49.86], [41.82, 64.14, 44.19], [45.65, 77.32, 40.16]],  # 95 merged
    [[7.23, 62.29, 67.26], [21.75, 57.74, 72.06], [35.54, 50.27, 73.22], [48.63, 42.12, 69.69], [60.21, 35.01, 61.7], [70.11, 30.42, 49.83], [82.77, 28.76, 40.6], [97.6, 26.4, 35.62]],  # 102 merged
    [[15.26, 91.55, 53.61], [5.79, 77.21, 57.25]],  # 104 merged
    [[69.35, 102.2, 1.73], [61.67, 91.37, 10.67], [53.32, 78.59, 15.18], [45.75, 63.9, 15.08], [38.11, 50.6, 10.84], [31.65, 35.97, 11.2], [27.9, 21.06, 15.03], [21.26, 6.21, 15.04]],  # 106 merged
    [[61.52, 47.24, 2.15], [73.33, 43.83, 14.43], [87.38, 37.5, 22.23], [102.28, 27.94, 25.02]],  # 111
    [[45.91, 100.24, 83.03], [39.33, 85.68, 82.98], [30.71, 72.29, 80.31], [21.53, 59.31, 81.93], [13.87, 46.74, 88.24], [8.87, 35.55, 98.48]],  # 113 merged
    [[92.72, 63.6, 41.06], [92.46, 51.08, 47.03], [89.47, 38.03, 50.55], [84.09, 25.29, 51.24], [76.66, 13.8, 49.11], [67.23, 4.54, 44.76]],  # 115
    [[92.82, 25.78, 3.47], [81.78, 12.79, 7.66]],  # 119
    [[101.38, 77.11, 18.9], [94.84, 62.63, 13.45], [91.89, 48.11, 6.15], [85.36, 33.5, 1.85], [76.24, 19.77, 2.07], [68.83, 4.92, 0.62]],  # 126
    [[76.49, 99.65, 15.64], [67.81, 87.65, 17.98], [60.71, 74.66, 20.76], [54.74, 60.93, 19.41], [47.55, 48.13, 15.28], [41.09, 34.55, 15.04], [37.83, 20.07, 17.63], [32.13, 6.33, 19.6]],  # 130 merged
    [[82.8, 1.11, 18.03], [81.11, 15.8, 17.56], [78.98, 29.76, 12.92], [77.29, 41.85, 4.52]],  # 132 merged
    [[88.03, 80.96, 46], [77.38, 69.53, 50.29], [64.29, 61.62, 55.51], [53.09, 50.2, 58.03], [45.33, 36, 58.1], [42.01, 20.22, 56.25], [39.18, 4.35, 57.26]],  # 133 merged
    [[12.01, 45.07, 38.2], [4.54, 31.03, 38.14]],  # 138 merged
    [[45.93, 99.59, 19.91], [60.05, 93.15, 25.52], [75.62, 89.24, 29.46], [91.27, 84.06, 30.04], [105.93, 76.37, 29.27]],  # 142 merged
    [[21.67, 101.41, 85.88], [14.1, 89.31, 87.58], [5.06, 78.93, 91.88]],  # 144
    [[5.9, 17.51, 26.59], [18.1, 28.35, 25.92], [29.17, 39.45, 21.95], [39.37, 52.04, 22.47], [49.21, 63.8, 27.52], [61.05, 73.82, 32.36]],  # 146 merged
    [[93.81, 41.54, 59.98], [84.79, 27.96, 62.37], [79.83, 12.53, 64.2]],  # 148
    [[4.47, 102.21, 21], [1.85, 88.77, 22.54]],  # 149
    [[56.84, 5.27, 99.65], [48.46, 16.4, 90.37], [36.76, 23.27, 80.36], [22.02, 28.37, 74.5], [5.19, 31.51, 73.38]],  # 150
    [[56.28, 28.94, 76.54], [65.54, 38.11, 70.18], [75.04, 48.56, 67.12], [84.97, 59.08, 67.24], [95.58, 68.55, 64.67], [105.91, 77.03, 58.93]],  # 153 merged
    [[2.84, 18.35, 49.96], [14.52, 9.7, 51.66]],  # 154
    [[103.15, 46.71, 84.04], [95.52, 33.38, 76.24], [86.44, 19.15, 73.01], [77.2, 4.87, 75.41]],  # 156
    [[100.44, 25.16, 46.78], [97.65, 10.93, 38.44]],  # 168
    [[23.08, 1.88, 103.48], [9.45, 0.02, 94.33]],  # 169
    [[49.35, 92.59, 70.69], [34.28, 96.21, 75.24], [19.22, 101.21, 76.03]],  # 170
    [[67.84, 30.97, 1.81], [51.05, 35.84, 1.8], [35.72, 43.61, 0.09], [21.27, 53.03, 2.03], [9.5, 64.64, 7.1]],  # 172
    [[97.23, 57.45, 62.07], [83.94, 50.32, 57.93], [71.98, 44.56, 49.62], [61.13, 34.99, 43.7], [50.82, 23.29, 41.98]],  # 177 merged
    [[4.25, 106.16, 66.75], [18.82, 98.57, 65.22], [32.32, 91.54, 58.78], [43.87, 86.61, 48.06], [57.87, 84.44, 39.41], [73.71, 81.61, 35.69], [89.47, 76.73, 36.83], [103.48, 68.24, 34.76]],  # 180 merged
    [[102.3, 35.43, 94.61], [90.34, 43.6, 86.35], [77.1, 53.23, 82.93], [63.23, 62.38, 84], [47.46, 68.19, 84.53]],  # 182
    [[68.81, 30.21, 103.53], [53.94, 33.83, 99.48], [40.98, 40.54, 93.95], [28.1, 49.39, 92.44], [16.54, 59.52, 95.31], [6.19, 69.93, 101.66]],  # 184 merged
    [[47.98, 102.42, 7.29], [39.69, 91.51, 10.76], [31.42, 80.03, 12.09], [26.2, 66.87, 11.96], [22.03, 53.04, 12.52], [19.91, 39.04, 12.19], [18.34, 25.3, 15.51], [14.88, 12.71, 20.93]],  # 186 merged
    [[44.54, 3.65, 26.32], [51.06, 17.48, 22.85], [60.32, 30.43, 21.64]],  # 187 merged
    [[73.54, 50.35, 102.31], [60.36, 55.69, 101.87], [47.86, 61.23, 98.42], [35.34, 66.94, 95.11], [22.69, 70.11, 89.67], [10.9, 74.71, 83.02]],  # 191
    [[22.98, 8.21, 94.3], [11.49, 19.43, 89.18]],  # 193
    [[0.75, 40.7, 12.24], [10.32, 52.52, 14.02], [16.8, 65.92, 17.3], [22.15, 80.36, 17.1]],  # 199 merged
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
    "missed_fiber": [[[v * um for v in point] for point in line] for line in MISSED_FIBER_BUNDLE_UM],
}
SCENARIO_LENGTH = {"missed_fiber": 180 * um}  # as single_type; the rest use 400 µm


def scenario(
    centerlines,
    length: float = 400 * um,
    *,
    diameter: float = DIAMETER,
    min_bend_radius: float = MIN_BEND_RADIUS,
    box: float = BOX,
    voxel: float = 1.5 * um,
    seed: int = 5,
    **render,
) -> Callable[[Path], Example]:
    def build(cache: Path) -> Example:
        # 0.99: as for the relaxed truths, which sit a hair past the limit by Tangle's measure.
        name = f"fiber_{diameter / um:g}um"
        material = tangle.Material(name, diameter=diameter, min_bend_radius=0.99 * min_bend_radius)
        assembly = tangle.Assembly(tangle.Cell([box] * 3))
        assembly.insert(tangle.FiberCollection.from_centerlines(centerlines, material), name="scenario")
        scan = render_scan(assembly, voxel, seed=seed, **render)
        spec = ct.FiberSpec(diameter=diameter, min_bend_radius=min_bend_radius, length=length, name=name)
        return Example(scan, spec, min_bend_radius, end_error_report)

    return build


# -- varied structures ------------------------------------------------------------
#
# Fresh structures the fitter was not tuned on (Liz's test loop: generate,
# fit, find what fails, cut it into a scenario, fix, re-check everything).
# ``varied_<n>`` draws its settings from seed n, so each name is always the
# same structure. They are not in the default run; name them, or pass
# ``--varied``. Each keeps the scan at 160 voxels a side, so a fit costs
# about what single_type does.

VARIED_COUNT = 8
VARIED_VOXELS = 160


def varied_settings(index: int) -> dict:
    """The structure and scan settings ``varied_<index>`` uses (lengths in µm)."""
    rng = np.random.default_rng(1000 + index)
    diameter = float(rng.choice([8.0, 10.0, 12.0, 14.0, 16.0]))
    voxel = round(diameter / rng.uniform(5.0, 9.0), 3)  # fiber radius 2.5-4.5 voxels
    side = VARIED_VOXELS * voxel
    orientation = ["planar", "aligned", "biaxial", "isotropic"][(index - 1) % 4]  # each twice in 8
    long = orientation == "planar"
    shortest = rng.uniform(0.45, 0.65 if long else 0.45) * side
    longest = min(shortest * rng.uniform(1.1, 1.4), (0.9 if long else 0.6) * side)
    fraction = rng.uniform(0.04, 0.14 if orientation != "isotropic" else 0.09)  # single_type is ~0.06
    count = int(round(fraction * side**3 / (np.pi * (diameter / 2) ** 2 * 0.5 * (shortest + longest))))
    return {
        "diameter_um": diameter,
        "min_bend_radius_um": round(float(rng.uniform(3.0, 6.0)) * diameter, 1),
        "voxel_um": voxel,
        "side_um": round(side, 1),
        "orientation": orientation,
        "tilt": round(float(rng.uniform(0.15, 0.6)), 2),
        "length_um": [round(shortest, 1), round(longest, 1)],
        "waviness_um": [round(0.15 * diameter, 1), round(float(rng.uniform(0.3, 1.2)) * diameter, 1)],
        "solid_fraction": round(float(fraction), 3),
        "count": min(max(count, 8), 200),
        "noise": round(float(rng.uniform(0.08, 0.18)), 3),
        "psf_sigma_voxels": round(float(rng.uniform(0.7, 1.2)), 2),
        "seed": 1000 + index,
    }


def varied(index: int) -> Callable[[Path], Example]:
    def build(cache: Path) -> Example:
        v = varied_settings(index)
        diameter, bend = v["diameter_um"] * um, v["min_bend_radius_um"] * um
        side = VARIED_VOXELS * v["voxel_um"] * um  # exact: "side_um" is rounded, and the voxels must tile the cell
        material = tangle.Material(f"fiber_{v['diameter_um']:g}um", diameter=diameter, min_bend_radius=bend)
        orientation = {
            "planar": lambda: tangle.PlanarOrientation(max_tilt=v["tilt"]),
            "aligned": lambda: tangle.AlignedOrientation("x", max_angle=v["tilt"]),
            "biaxial": lambda: tangle.LayeredBiaxialOrientation(max_tilt=v["tilt"], seed=v["seed"]),
            "isotropic": tangle.IsotropicOrientation,
        }[v["orientation"]]()
        shortest, longest = (x * um for x in v["length_um"])
        population = tangle.FiberPopulation(
            material=material,
            count=v["count"],
            segments_per_fiber=max(4, int(shortest / (1.25 * diameter))),  # segments longer than a diameter
            seed=v["seed"],
            length=(shortest, longest),
            curvature_amplitude=tuple(x * um for x in v["waviness_um"]),
            orientation=orientation,
            # A biaxial orientation picks a direction per layer: layers about 4 diameters apart.
            position=(
                tangle.LayeredPosition(max(3, int(side / (4 * diameter))), jitter_fraction=0.25)
                if v["orientation"] == "biaxial"
                else tangle.UniformPosition()
            ),
        )
        # The cache is per name; a settings change needs a new one.
        key = hashlib.sha1(json.dumps(v, sort_keys=True).encode()).hexdigest()[:8]
        truth = relaxed_truth(cache.with_name(f"{cache.stem}-{key}.json"), tangle.Cell([side] * 3), [population])
        scan = render_scan(
            truth, v["voxel_um"] * um, seed=v["seed"], noise=v["noise"], psf_sigma_voxels=v["psf_sigma_voxels"]
        )
        spec = ct.FiberSpec(
            diameter=diameter, min_bend_radius=bend, length=0.5 * (shortest + longest), name=material.name
        )
        # A --blur run reports the blur it used, not the structure's own.
        used = v if BLUR is None else {**v, "psf_sigma_voxels": BLUR}
        return Example(scan, spec, bend, lambda fit, scan: {"settings": used, **end_error_report(fit, scan)})

    return build


EXAMPLES: dict[str, Callable[[Path], Example]] = {
    "single_type": single_type,
    "long_fibers": long_fibers,
    "two_types": two_types,
    "noisy_two_types": noisy_two_types,
    "halo_two_types": halo_two_types,
    "scanner_two_types": scanner_two_types,
    "bundled_two_types": bundled_two_types,
    **{
        f"scenario_{name}": scenario(lines, SCENARIO_LENGTH.get(name, 400 * um))
        for name, lines in SCENARIOS.items()
    },
}
# varied_3's settings (see varied_settings(3)), in its own 64-voxel cube.
EXAMPLES["scenario_dense_crossing"] = scenario(
    [[[v * um for v in point] for point in line] for line in DENSE_CROSSING_UM],
    137.25 * um,
    diameter=10 * um,
    min_bend_radius=51.7 * um,
    box=64 * 1.659 * um,
    voxel=1.659 * um,
    seed=1003,
    noise=0.177,
    psf_sigma_voxels=1.05,
)
VARIED = [f"varied_{index}" for index in range(1, VARIED_COUNT + 1)]
EXAMPLES.update({name: varied(index) for index, name in enumerate(VARIED, start=1)})



# -- scanned structures -----------------------------------------------------------
#
# The test loop on simulated scans: ``scanned_<n>`` draws a structure and a
# scanner from seed n (one or two fiber types, loose or in bundles, planar,
# aligned or biaxial, and the scanner's photons, resolution, motion, phase and
# noise grain around SCANNER's), scans it with ct.Scanner and fits it. Tune on
# ``scanned_1`` … ``scanned_16`` and check on ``scanned_101`` … (``--scanned``),
# so a fix that only suits the tuning set shows up. 160 voxels a side.

SCANNED_VOXELS = 160


def scanned_settings(index: int) -> dict:
    """The structure and scanner ``scanned_<index>`` uses (lengths in µm)."""
    rng = np.random.default_rng(5000 + index)
    voxel = round(float(rng.uniform(1.0, 1.5)), 3)
    side = SCANNED_VOXELS * voxel
    fine = round(float(rng.uniform(5.0, 9.0)), 1)
    coarse = round(float(rng.uniform(14.0, 22.0)), 1) if rng.random() < 0.7 else None
    orientation = ["planar", "planar", "aligned", "biaxial"][int(rng.integers(4))]
    blur = float(rng.choice([0.0, 0.5, 1.0]))
    return {
        "voxel_um": voxel,
        "side_um": round(side, 1),
        "orientation": orientation,
        "tilt": round(float(rng.uniform(0.15, 0.45)), 2),
        "fine_um": fine,
        "fine_fraction": round(float(rng.uniform(0.04, 0.12)), 3),
        "per_bundle": int(rng.choice([1, 7, 19])),
        "coarse_um": coarse,
        "coarse_fraction": round(float(rng.uniform(0.02, 0.06)), 3) if coarse else 0.0,
        "coarse_brightness": round(float(rng.uniform(0.4, 0.7)), 2),
        "length_fraction": [round(float(rng.uniform(0.5, 0.7)), 2), round(float(rng.uniform(0.75, 0.95)), 2)],
        # The noise blur averages the noise down (by about 1 / (2 sqrt(pi) sigma)),
        # so the photons drop with it to keep the contrast to noise.
        "photons": round(SCANNER.photons * float(rng.uniform(0.6, 1.6)) / max(1.0, 4 * np.pi * blur**2)),
        "noise_blur": blur,
        "resolution_um": round(float(rng.uniform(1.5, 3.0)), 2),
        "fiber_motion_um": round(float(rng.uniform(0.0, 1.5)), 2),
        "delta_beta": [round(float(rng.uniform(6.0, 16.0)), 1), round(float(rng.uniform(5.0, 12.0)), 1)],
        "seed": 5000 + index,
    }


def scanned(index: int) -> Callable[[Path], Example]:
    def build(cache: Path) -> Example:
        v = scanned_settings(index)
        side = SCANNED_VOXELS * v["voxel_um"] * um
        cell = tangle.Cell([side] * 3)
        length = tuple(f * side for f in v["length_fraction"])
        orientation = {
            "planar": lambda: tangle.PlanarOrientation(max_tilt=v["tilt"]),
            "aligned": lambda: tangle.AlignedOrientation("x", max_angle=v["tilt"]),
            "biaxial": lambda: tangle.LayeredBiaxialOrientation(max_tilt=v["tilt"], seed=v["seed"]),
        }[v["orientation"]]
        types = [("fine", v["fine_um"], v["fine_fraction"], v["per_bundle"])]
        if v["coarse_um"]:
            types.append(("coarse", v["coarse_um"], v["coarse_fraction"], 1))
        populations, specs, profiles = [], [], []
        for offset, (kind, diameter_um, fraction, per_bundle) in enumerate(types):
            diameter = diameter_um * um
            bend = 5 * diameter
            material = tangle.Material(f"{kind}_{diameter_um:g}um", diameter=diameter, min_bend_radius=bend)
            mean_length = 0.5 * sum(length)
            count = max(int(round(fraction * side**3 / (np.pi * (diameter / 2) ** 2 * mean_length))), 4)
            population = tangle.FiberPopulation(
                material=material,
                count=max(count // per_bundle, 1),
                segments_per_fiber=max(4, int(length[0] / (1.25 * diameter))),
                seed=v["seed"] + offset,
                length=length,
                curvature_amplitude=(0.15 * diameter, 0.6 * diameter),
                orientation=orientation(),
                position=(
                    tangle.LayeredPosition(max(3, int(side / (4 * diameter))), jitter_fraction=0.25)
                    if v["orientation"] == "biaxial"
                    else tangle.UniformPosition()
                ),
            )
            populations.append(bundles(cell, population, per_bundle) if per_bundle > 1 else population)
            specs.append(ct.FiberSpec(diameter=diameter, min_bend_radius=bend, length=mean_length, name=material.name))
            profiles.append((diameter, ct.CrossSection(brightness=1.0 if kind == "fine" else v["coarse_brightness"])))
        key = hashlib.sha1(json.dumps(v, sort_keys=True).encode()).hexdigest()[:8]
        truth = relaxed_truth(cache.with_name(f"{cache.stem}-{key}.json"), cell, populations)
        scanner = replace(
            SCANNER,
            photons=v["photons"],
            noise_blur=v["noise_blur"],
            resolution=v["resolution_um"] * um,
            fiber_motion=v["fiber_motion_um"] * um,
            delta_beta=tuple(v["delta_beta"][: len(types)]),
        )
        scan = render_scan(truth, v["voxel_um"] * um, seed=v["seed"], profiles=profiles, scanner=scanner)
        smallest = min(s.min_bend_radius for s in specs)
        return Example(
            scan,
            specs if len(specs) > 1 else specs[0],
            smallest,
            lambda fit, scan: {"settings": v, "per_type": ct.score(fit, scan)["per_type"]},
        )

    return build


SCANNED_TUNE = range(1, 17)
SCANNED_CHECK = range(101, 109)
EXAMPLES.update({f"scanned_{index}": scanned(index) for index in [*SCANNED_TUNE, *SCANNED_CHECK]})

# -- the runner -----------------------------------------------------------------


def _truth_mapping(truth: np.ndarray, fit_labels: np.ndarray) -> np.ndarray:
    """For every fit label, the true fiber it overlaps most (0 = none)."""
    both = (truth > 0) & (fit_labels > 0)
    mapping = np.zeros(int(fit_labels.max()) + 1, dtype=truth.dtype)
    if both.any():
        pairs = fit_labels[both].astype(np.int64) * (int(truth.max()) + 1) + truth[both]
        values, counts = np.unique(pairs, return_counts=True)
        fits, trues = np.divmod(values, int(truth.max()) + 1)
        order = np.lexsort((counts, fits))  # by fit, then count ascending
        mapping[fits[order]] = trues[order]  # the last (largest count) write wins per fit
    return mapping


def label_diff(scan: ct.SyntheticScan, fit_labels: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """RGB stack of where the fit and the truth disagree, voxel counts, and the class masks.

    Each fitted fiber is matched to the true fiber it overlaps most; a voxel
    both call fiber is "wrong_fiber" when its fit is matched to another one.
    """
    truth = np.asarray(scan.labels)
    both = (truth > 0) & (fit_labels > 0)
    mapping = _truth_mapping(truth, fit_labels)
    classes = {
        "missed": (truth > 0) & (fit_labels == 0),
        "extra": (truth == 0) & (fit_labels > 0),
        "wrong_fiber": both & (mapping[fit_labels] != truth),
    }
    rgb = _dim_scan(scan)
    for name, where in classes.items():
        rgb[where] = DIFF_COLORS[name]
    fiber = max(int((truth > 0).sum()), 1)
    counts = {f"{name}_voxels": int(where.sum()) for name, where in classes.items()}
    counts.update({f"{name}_fraction_of_true_fiber": int(where.sum()) / fiber for name, where in classes.items()})
    return rgb, counts, classes


def _dim_scan(scan: ct.SyntheticScan) -> np.ndarray:
    volume = np.asarray(scan.volume, dtype=np.float32)
    low, high = np.percentile(volume[:: max(1, volume.shape[0] // 32)], [0.5, 99.5])
    grey = (np.clip((volume - low) / max(high - low, 1e-6), 0.0, 1.0) * 110).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


def confidence_image(scan: ct.SyntheticScan, confidence: np.ndarray) -> np.ndarray:
    """RGB stack: fitted voxels from red (confidence 0) through yellow to green (1)."""
    rgb = _dim_scan(scan)
    fitted = ~np.isnan(confidence)
    c = np.clip(confidence[fitted], 0.0, 1.0)
    rgb[fitted] = np.stack(
        [np.where(c < 0.5, 230, 230 * (1 - c) * 2), np.where(c < 0.5, 400 * c, 200), 40 * np.ones_like(c)], axis=1
    ).astype(np.uint8)
    return rgb


def _auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Chance that a positive scores above a negative (ties count half)."""
    from scipy.stats import rankdata

    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def confidence_check(fit: ct.FitResult, report: dict, classes: dict, confidence: np.ndarray) -> dict:
    """How well the fit's own confidence picks out its errors.

    Voxels: among fitted voxels, the wrong ones (extra or wrong fiber).
    Fibers: among fits, the merged and false ones (by mean node confidence).
    """
    fitted = ~np.isnan(confidence)
    wrong = (classes["extra"] | classes["wrong_fiber"])[fitted]
    unsure = 1.0 - confidence[fitted]
    states = [entry["state"] for entry in report["per_fitted_fiber"]]
    fiber_confidence = np.array([float(np.mean(c)) if len(c) else 0.0 for c in fit.confidence or []])
    bad = np.array([state != "matched" for state in states], dtype=bool)
    result = {
        "voxel_auc": _auc(unsure, wrong),
        "voxel_mean_confidence_right": float(confidence[fitted][~wrong].mean()) if (~wrong).any() else None,
        "voxel_mean_confidence_wrong": float(confidence[fitted][wrong].mean()) if wrong.any() else None,
        "fiber_auc": None,
    }
    if len(fiber_confidence) == len(bad) and len(bad):
        result["fiber_auc"] = _auc(1.0 - fiber_confidence, bad)
        result["fiber_mean_confidence_matched"] = float(fiber_confidence[~bad].mean()) if (~bad).any() else None
        result["fiber_mean_confidence_merged_or_false"] = float(fiber_confidence[bad].mean()) if bad.any() else None
    # Missed fiber: how much of it lies next to (within 2 voxels of) a low-confidence fitted voxel.
    missed = classes["missed"]
    if missed.any() and fitted.any():
        from scipy.ndimage import grey_dilation

        low = np.where(fitted, 1.0 - np.nan_to_num(confidence, nan=1.0), 0.0).astype(np.float32)
        near = grey_dilation(low, size=5)
        result["missed_next_to_unsure_fit"] = float((near[missed] > 0.5).mean())
        result["missed_next_to_any_fit"] = float((grey_dilation(fitted.astype(np.uint8), size=5)[missed] > 0).mean())
    return result


def grey_profiles(scan: ct.SyntheticScan, count: int, sigma: float = 0.7) -> list[tuple[float, ...]]:
    """Each type's grey profile, measured on the denoised scan around the true fibers.

    As one would measure it on a few fibers of a real scan: the median grey
    at 9 radii from the axis to the surface (``tangle.ct._grey.measure_profiles``).
    """
    from scipy.ndimage import gaussian_filter
    from tangle.ct._grey import measure_profiles

    grey = gaussian_filter(np.asarray(scan.volume, dtype=np.float32), sigma)
    types = np.asarray(scan.types) if scan.types is not None else np.zeros(len(scan.centerlines), dtype=int)
    profiles = measure_profiles(grey, scan.centerlines, scan.radii, types, count)
    return [tuple(round(float(v), 2) for v in profile) for profile in profiles]


def fit_input(scan: ct.SyntheticScan, spec, input: str = "grey") -> tuple[np.ndarray, object, np.ndarray]:
    """What to fit (the raw scan or a mask), the specs to fit it with, and the 0-1 image the fit sees."""
    from scipy.ndimage import gaussian_filter
    from tangle.ct._grey import profile_levels
    from tangle.ct._ranges import range_image

    if input == "mask":
        mask = scan.fiber_mask(level=MASK_LEVEL)
        return mask, spec, mask.astype(np.float32)
    if input == "plain":
        from tangle.ct._image import normalize

        seen, _ = normalize(scan.volume, denoise_sigma=0.7)
        return scan.volume, spec, seen
    specs = spec if isinstance(spec, list) else [spec]
    profiles = grey_profiles(scan, len(specs))
    specs = [item.replace(profile=profile) for item, profile in zip(specs, profiles)]
    grey = gaussian_filter(np.asarray(scan.volume, dtype=np.float32), 0.7)
    _, _, ranges = profile_levels(grey, [np.asarray(profile) for profile in profiles])
    seen, _, _ = range_image(scan.volume, ranges, denoise_sigma=0.7)
    return scan.volume, specs if isinstance(spec, list) else specs[0], seen


def run(name: str, output: Path) -> dict:
    print(f"{name}:")
    example = EXAMPLES[name](output / ".cache" / f"{name}.json")
    scan = example.scan
    h = scan.voxel_size
    source = INPUT or example.input
    volume, spec, seen = fit_input(scan, example.spec, source)
    started = time.perf_counter()
    fit = ct.fit_fibers(volume, h, spec, ct.FitSettings(backend=BACKEND))
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
    _write_stack(folder / "input", (np.clip(seen, 0.0, 1.0) * 255).astype(np.uint8), h)
    _write_stack(folder / "true", ct.overlay_volume(scan.volume, scan.labels), h, rgb=True)
    fit_labels = fit.label_volume()
    _write_stack(folder / "segment", ct.overlay_volume(scan.volume, fit_labels), h, rgb=True)
    diff, diff_counts, diff_classes = label_diff(scan, fit_labels)
    _write_stack(folder / "diff", diff, h, rgb=True)
    confidence = fit.confidence_volume()
    _write_stack(folder / "confidence", confidence_image(scan, confidence), h, rgb=True)
    check = confidence_check(fit, report, diff_classes, confidence)
    (folder / "fit.json").write_text(json.dumps(fit.to_dict(), indent=1) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("per_true_fiber", "per_type", "per_fitted_fiber")}
    (folder / "score.json").write_text(
        json.dumps(
            {"seconds": seconds, "score": report, "geometry": geometry, "diff": diff_counts, "confidence": check, **extra},
            indent=1, default=str,
        ) + "\n"
    )
    row = {
        "example": name,
        "input": source,
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
        "centerline recall / precision / F1": "/".join(
            "-" if summary[k] is None else f"{summary[k]:.3f}"
            for k in ("centerline_recall", "centerline_precision", "centerline_f1")
        ),
        "missed / extra / wrong (% of fiber)": "/".join(
            f"{100 * diff_counts[f'{k}_fraction_of_true_fiber']:.1f}" for k in ("missed", "extra", "wrong_fiber")
        ),
        "confidence AUC (voxel/fiber)": "/".join(
            "-" if check[k] is None else f"{check[k]:.2f}" for k in ("voxel_auc", "fiber_auc")
        ),
        "sure coverage": _sure_coverage(fit.history),
        "redraws kept": sum(1 for e in fit.history if e["stage"].startswith("redraw") and e.get("kept")),
        "end error max (vox)": extra.get("end_error_max_voxels"),
        "overlaps": geometry.get("overlapping_pairs"),
        "over bend limit": geometry.get("fibers_over_bend_limit"),
        "seconds": seconds,
    }
    print("  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return row


def _sure_coverage(history: list[dict]) -> float | None:
    """The fit's final sure coverage: the last kept redraw's, else the first confidence's."""
    value = None
    for entry in history:
        if entry["stage"] == "confidence" or (entry["stage"].startswith("redraw") and entry.get("kept")):
            value = entry.get("sure_coverage")
    return value


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
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in known.values()]
    header = (
        "# tangle.ct examples\n\n"
        + "Input: grey = the raw scan with a grey profile per fiber type, measured around the true fibers; "
        "plain = the raw scan and the fiber diameters alone, with no grey information; "
        f"mask = a binary mask thresholded at {MASK_LEVEL} of the way from void to fiber. "
        + (f"Every scan rendered with a blur of {BLUR:g} voxels (PSF sigma). " if BLUR is not None else "")
        + "Each example's folder holds raw.tif, input.tif, true.tif, segment.tif, diff.tif, confidence.tif, fit.json and score.json. "
        "diff.tif: red = missed, blue = extra, orange = wrong fiber. confidence.tif: green = sure, red = unsure. "
        "Confidence AUC: how well low confidence picks out the wrong voxels (extra or wrong fiber) among fitted "
        "voxels, and the merged or false fits among all fits; 0.5 is chance, 1 is perfect.\n\n"
    )
    (output / "summary.md").write_text(header + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="examples to run (default: all)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="list the examples and exit")
    parser.add_argument("--varied", action="store_true", help="also run the varied_* structures")
    parser.add_argument(
        "--scanned", choices=("tune", "check", "all"), default=None,
        help="run the scanned_* structures: the tuning set (1-16), the check set (101-108) or both",
    )
    parser.add_argument(
        "--input", choices=("grey", "plain", "mask"), default=None,
        help="fit every example from the raw scan with grey profiles, the raw scan alone, or a generous mask "
        "(default: each example's own; grey, and plain for noisy_two_types)",
    )
    parser.add_argument(
        "--blur", type=float, default=None,
        help="render every scan with this blur (PSF sigma, voxels); default: each example's own (0.9; 0.7-1.2 for varied_*)",
    )
    args = parser.parse_args()
    global INPUT, BLUR
    if args.input:
        INPUT = args.input
    BLUR = args.blur
    if args.list:
        print("\n".join(EXAMPLES))
        return
    unknown = [name for name in args.names if name not in EXAMPLES]
    if unknown:
        parser.error(f"unknown examples: {', '.join(unknown)} (see --list)")
    output = args.output or Path(os.environ.get("TANGLE_CT_OUTPUT", Path(__file__).with_name("output") / "ct"))
    output.mkdir(parents=True, exist_ok=True)
    scanned_names = [name for name in EXAMPLES if name.startswith("scanned_")]
    if args.scanned:
        chosen = {"tune": SCANNED_TUNE, "check": SCANNED_CHECK, "all": [*SCANNED_TUNE, *SCANNED_CHECK]}[args.scanned]
        names = args.names + [f"scanned_{index}" for index in chosen]
    else:
        names = args.names or [name for name in EXAMPLES if name not in VARIED and name not in scanned_names]
    if args.varied:
        names += [name for name in VARIED if name not in names]
    rows = [run(name, output) for name in names]
    write_summary(output, rows)
    for label, chosen in (("tune", SCANNED_TUNE), ("check", SCANNED_CHECK)):
        f1 = [row["centerline recall / precision / F1"].split("/")[2] for row in rows if row["example"] in
              {f"scanned_{index}" for index in chosen}]
        f1 = [float(value) for value in f1 if value != "-"]
        if f1:
            print(f"scanned {label}: mean F1 {np.mean(f1):.3f} over {len(f1)}")
    print(f"summary: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
