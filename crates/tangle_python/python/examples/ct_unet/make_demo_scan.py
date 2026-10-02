"""A small made-up scan with known fibers, to try find_fibers.py on before your own scan.

    python make_demo_scan.py [FOLDER] [--size 160] [--seed 1]

Writes FOLDER/demo_scan.tif (default FOLDER: demo): a 160-voxel cube of 1.5 um voxels holding two types of
gently curved fibers, 10.5 and 21 um across (7 and 14 voxels), lying mostly flat as in a nonwoven, close but
never overlapping, never bending tighter than Tangle's default limit (5 diameters), blurred and noisy like a
CT scan. FOLDER/truth/ holds the true fibers in the files
find_fibers.py writes (fibers.csv, centerlines.csv), for compare_fibers.py.

Needs only NumPy, SciPy and tifffile (no Tangle build). It is far simpler than a real scan, and than the
simulated scans the network was trained on: take it as a check that everything runs, not as a test of the
method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

import fiber_outputs
from fiber_outputs import Fibers

VOXEL_UM = 1.5
KINDS = ((7.0, 2000.0, 0.65), (14.0, 2600.0, 0.35))  # diameter (voxels), grey above the void, share of fibers
VOID, NOISE, BLUR = 1000.0, 150.0, 0.8
GAP = 0.5  # voxels kept between fiber surfaces
BEND = 5.0  # no fiber bends tighter than this many diameters (Tangle's default limit)


def fibers_in_box(size: int, rng: np.random.Generator, fraction: float = 0.12, attempts: int = 5000):
    """Centerlines (x, y, z voxels, 1 apart), radii and types of non-overlapping fibers filling ``fraction``."""
    lines, radii, kinds = [], [], []
    points, owner_radius = np.zeros((0, 3)), np.zeros(0)
    tree = None
    filled, wanted = 0.0, fraction * size**3
    reach = max(d for d, _, _ in KINDS) / 2 + GAP
    for _ in range(attempts):
        if filled >= wanted:
            break
        kind = 0 if rng.random() < KINDS[0][2] else 1
        r = KINDS[kind][0] / 2
        phi = rng.uniform(0.0, np.pi)
        tilt = np.radians(np.clip(rng.normal(0.0, 8.0), -20.0, 20.0))
        u = np.array([np.cos(phi) * np.cos(tilt), np.sin(phi) * np.cos(tilt), np.sin(tilt)])
        side = np.cross(u, [0.0, 0.0, 1.0])
        side /= np.linalg.norm(side)
        up = np.cross(u, side)
        s = np.arange(-1.0 * size, 1.0 * size + 0.5, 1.0)
        wave, amplitude = rng.uniform(50.0, 140.0), rng.uniform(0.0, 1.2) * r
        # the two waves below bend the fiber by at most A (2 pi / wave)^2 (1 + 0.3 / 1.7^2): keep that 10% under
        # the bend limit, 1 / (BEND * 2 r)
        amplitude = min(amplitude, 0.9 * (wave / (2 * np.pi)) ** 2 / ((1 + 0.3 / 1.7**2) * BEND * 2 * r))
        phase = rng.uniform(0.0, 2 * np.pi, 2)
        line = (rng.uniform(0.0, size, 3) + s[:, None] * u
                + (amplitude * np.sin(2 * np.pi * s / wave + phase[0]))[:, None] * side
                + (0.3 * amplitude * np.sin(2 * np.pi * s / (1.7 * wave) + phase[1]))[:, None] * up)
        inside = np.all((line > -r) & (line < size + r), axis=1)
        if inside.sum() < 6 * r:
            continue
        line = line[_longest_run(inside)]
        if tree is not None:
            near = cKDTree(line).sparse_distance_matrix(tree, r + reach, output_type="ndarray")
            if len(near) and np.any(near["v"] < r + owner_radius[near["j"]] + GAP):
                continue
        lines.append(line)
        radii.append(r)
        kinds.append(kind)
        points = np.vstack([points, line])
        owner_radius = np.concatenate([owner_radius, np.full(len(line), r)])
        tree = cKDTree(points)
        filled += np.pi * r * r * np.all((line >= 0) & (line <= size), axis=1).sum()
    return lines, np.array(radii), np.array(kinds, int)


def _longest_run(mask: np.ndarray) -> slice:
    best, start = (0, 0), None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return slice(*best)


def render(lines, radii, kinds, size: int, rng: np.random.Generator) -> np.ndarray:
    """The scan: each fiber's grey over the void, edges partial-volume, then blur and noise (uint16)."""
    fiber, depth = fiber_outputs.draw(fiber_outputs.segments(lines, radii), (0, size, 0, size, 0, size), margin=1.0)
    occupancy = np.clip(0.5 - depth, 0.0, 1.0)
    contrast = np.r_[0.0, [KINDS[k][1] for k in kinds]][fiber]
    volume = gaussian_filter(VOID + occupancy * contrast, BLUR)
    volume += rng.normal(0.0, NOISE, volume.shape)
    return np.clip(np.rint(volume), 0, 65535).astype(np.uint16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="make_demo_scan.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", nargs="?", default="demo", help="where to write it (default: demo)")
    parser.add_argument("--size", type=int, default=160, help="voxels along each side (default 160)")
    parser.add_argument("--seed", type=int, default=1, help="another seed gives other fibers")
    args = parser.parse_args(argv)
    try:
        import tifffile
    except ImportError:
        print("make_demo_scan: writing TIFF needs tifffile: python -m pip install tifffile", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    folder = Path(args.folder)
    (folder / "truth").mkdir(parents=True, exist_ok=True)
    lines, radii, kinds = fibers_in_box(args.size, rng)
    volume = render(lines, radii, kinds, args.size, rng)
    path = folder / "demo_scan.tif"
    tifffile.imwrite(path, volume, imagej=True, resolution=(1.0 / VOXEL_UM, 1.0 / VOXEL_UM),
                     metadata={"spacing": VOXEL_UM, "unit": "um", "axes": "ZYX"})
    # the truth, cut to the box, in find_fibers.py's own files
    kept = [(line[_longest_run(np.all((line >= 0) & (line <= args.size), axis=1))], r, k)
            for line, r, k in zip(lines, radii, kinds)]
    kept = [(line, r, k) for line, r, k in kept if len(line) >= 2]
    truth = Fibers([line for line, _, _ in kept], np.array([r for _, r, _ in kept]), np.array([k for _, _, k in kept]),
                   np.ones(len(kept)), [], VOXEL_UM, (args.size,) * 3, [f"{d * VOXEL_UM:g} um" for d, _, _ in KINDS])
    fiber_outputs.write_tables(truth, folder / "truth")
    sizes = ",".join(f"{d * VOXEL_UM:g}um" for d, _, _ in KINDS)
    print(f"Wrote {path}: {args.size} x {args.size} x {args.size} voxels of {VOXEL_UM} um, {len(kept)} fibers "
          f"({int((truth.types == 0).sum())} of {KINDS[0][0] * VOXEL_UM:g} um, {int((truth.types == 1).sum())} of "
          f"{KINDS[1][0] * VOXEL_UM:g} um), and the true fibers in {folder / 'truth'}")
    print("Next:")
    print(f"  python find_fibers.py {path} --weights NETWORK.pt --diameters {sizes} --out {folder / 'found'}")
    print(f"  python compare_fibers.py {folder / 'found'} {folder / 'truth'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
