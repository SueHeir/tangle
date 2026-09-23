"""Continuous fit on Tangle's own solver (the GPU by default).

The NumPy loop in ``_refine`` imitates a fiber solver: a Laplacian bending
step and a pairwise push-apart. Here Tangle's relaxation does that part for
real (contact, stretch, bending and the bend limit, through CubeCL), with the
scan as one more force: every vertex moves sideways toward the intensity
centroid of the cross-section samples it owns under the
nearest-capsule-surface rule (``tangle.ImageRelaxer``).

One batch is: upload the fibers, relax with the image force, read back each
vertex's owned intensity (for the radii), relax a little more without the
image so the fibers come back admissible, then grow or trim the ends on the
host. Radii and ends change the fibers, so every batch uploads again;
topology moves (splits, joins, births) still happen once per round in
``_fit``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _refine
from ._geometry import rasterize, resample


def available() -> bool:
    """Whether this build of Tangle has the solver image force."""
    try:
        import tangle
    except ImportError:
        return False
    return hasattr(tangle, "ImageRelaxer")


def _straight(line: np.ndarray) -> np.ndarray:
    """A straight rest shape with the same segment lengths.

    Tangle takes a fiber's rest shape from the centerline it is placed with,
    rejects one that bends past the bend limit, and bends back toward it, so
    a fitted fiber would keep its kinks. Fitted fibers rest straight: bending
    resists every curve, and the scan and contacts supply the waviness.
    """
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    direction = line[-1] - line[0]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return line[0] + np.concatenate([[0.0], np.cumsum(lengths)])[:, None] * direction


def _relaxer(image, payload, lines, radii, *, voxel_size, bend, pad, settings) -> Any:
    import tangle

    h = voxel_size
    cell = tangle.Cell([(n + 2.0 * pad) * h for n in image.shape[::-1]])
    materials: dict[int, Any] = {}
    collection = tangle.FiberCollection("ct fit")
    for line, radius in zip(lines, radii):
        key = int(round(2.0 * float(radius) * h / 1e-9))
        if key not in materials:
            materials[key] = tangle.Material(f"ct {key}nm", diameter=key * 1e-9, min_bend_radius=bend * h)
        shifted = line + pad
        collection.add_fiber((shifted * h).tolist(), materials[key], rest_centerline=(_straight(shifted) * h).tolist())
    assembly = tangle.Assembly(cell)
    assembly.insert(collection, name="ct fit")
    return tangle.ImageRelaxer(
        assembly, settings, payload, tuple(int(n) for n in image.shape), h, (pad * h, pad * h, pad * h)
    )


def refine(
    image: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    *,
    voxel_size: float,
    radius: float,
    tolerance: float,
    prior_weight: float,
    bend: float,
    spacing: float,
    rate: float,
    reach_radii: float,
    batches: int,
    iterations: int,
    settle: int,
    backend: str | None,
    profile=None,
    log=None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """``batches`` rounds of (solver with the image force → radii → ends).

    ``lines`` and the result are in voxels at the fitter's node ``spacing``;
    on the device, fibers use segments of 1.25 diameters, because Tangle's
    contact treats non-adjacent segments of one fiber as colliding.
    """
    import tangle

    h = voxel_size
    payload = np.ascontiguousarray(image, dtype="<f4").tobytes()
    device_spacing = max(2.5 * radius, spacing)
    pad = 3.0 * radius  # room around the scan so fibers that leave it are not squeezed by walls
    options: dict[str, Any] = {
        "max_iterations": iterations,
        "max_step": 0.25 * radius * h,
        "penetration_tolerance": 0.02 * radius * h,
        # Fits start overlapping; the default 48 neighbor slots overflow
        # into a much slower fallback.
        "neighbor_capacity": 192,
    }
    if backend:
        options["backend"] = backend
    settings = tangle.RelaxationSettings(**options)
    radii = np.asarray(radii, dtype=np.float64)
    for _ in range(batches):
        if not lines:
            break
        coarse = [resample(line, device_spacing) for line in lines]
        relaxer = _relaxer(image, payload, coarse, radii, voxel_size=h, bend=bend, pad=pad, settings=settings)
        relaxer.set_image_force(rate, reach_radii=reach_radii)
        status = relaxer.run(iterations)
        stats = relaxer.vertex_image_stats()
        if settle:
            relaxer.set_image_force(0.0)
            status = relaxer.run(settle)
        if log is not None:
            log("solver", coarse, **{k: status[k] for k in ("converged", "max_curvature_ratio") if k in status})
        lines = [np.asarray(line, dtype=np.float64) / h - pad for line in relaxer.centerlines()]

        # Owned intensity per vertex is area-weighted over its cross-section
        # (voxels²); ends only own half a disc, so interior vertices size the fiber.
        area = np.array([
            np.mean([mass for mass, _ in fiber[1:-1]] if len(fiber) > 2 else [mass for mass, _ in fiber])
            if fiber else 0.0
            for fiber in stats
        ])
        if profile is None:
            measured = np.sqrt(np.maximum(area, 0.0) / np.pi)
        else:
            measured = profile.radius_from_area(area, h, 0.25 * radius, 2.0 * radius)
        blended = (measured + prior_weight * radius) / (1.0 + prior_weight)
        radii = np.clip(blended, radius * (1 - tolerance), radius * (1 + tolerance))

        lines = _refine.respace(lines, spacing)
        occupied, _, _ = rasterize(image.shape, lines, radii, signed=True)
        lines = _refine.end_step(image, lines, radii, step=spacing, occupied=occupied)
        lines = _refine.respace(lines, spacing)
    return lines, radii
