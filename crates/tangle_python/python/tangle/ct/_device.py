"""The fit's continuous part, on Tangle's own solver (the GPU by default).

Tangle's relaxation moves the fibers (contact, stretch, bending and the bend
limit, through CubeCL), with the scan as one more force: every vertex moves
sideways toward the intensity centroid of the cross-section samples it owns
under the nearest-capsule-surface rule (``tangle.ImageRelaxer``).

One batch (:func:`relax`) is: grow or trim the fiber ends on the host,
upload the fibers, relax with the image force, then relax a little more
without it so the fibers come back admissible. Ends change before the
solve, so the solver also cleans up what end growth did. Topology moves
(splits, joins, births) and fiber types happen between batches in ``_fit``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _refine
from ._geometry import resample


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


def _relaxer(image, payload, lines, radii, bends, *, voxel_size, pad, settings) -> Any:
    import tangle

    h = voxel_size
    cell = tangle.Cell([(n + 2.0 * pad) * h for n in image.shape[::-1]])
    materials: dict[tuple[int, int], Any] = {}
    collection = tangle.FiberCollection("ct fit")
    for line, radius, bend in zip(lines, radii, bends):
        key = (int(round(2.0 * float(radius) * h / 1e-9)), int(round(float(bend) * h / 1e-9)))
        if key not in materials:
            materials[key] = tangle.Material(f"ct {key[0]}nm", diameter=key[0] * 1e-9, min_bend_radius=key[1] * 1e-9)
        shifted = line + pad
        collection.add_fiber(
            (shifted * h).tolist(), materials[key], rest_centerline=(_straight(shifted) * h).tolist()
        )
    assembly = tangle.Assembly(cell)
    assembly.insert(collection, name="ct fit")
    return tangle.ImageRelaxer(
        assembly, settings, payload, tuple(int(n) for n in image.shape), h, (pad * h, pad * h, pad * h)
    )


def relax(
    image: np.ndarray,
    lines: list[np.ndarray],
    radii: np.ndarray,
    bends: np.ndarray,
    *,
    voxel_size: float,
    spacing: float,
    rate: float,
    reach_radii: float,
    iterations: int,
    settle: int,
    backend: str | None,
    reach: np.ndarray | None = None,
    anchors: list[np.ndarray] | None = None,
    anchor_tolerance: float = 0.0,
    log=None,
) -> list[np.ndarray]:
    """One batch: ends, then the solver with the image force, then a settle.

    ``lines``, ``radii`` and ``bends`` (each fiber's bend limit) are in
    voxels. On the device, fibers use segments of 1.25 diameters, because
    Tangle's contact treats non-adjacent segments of one fiber as colliding.
    ``reach`` is how far past each fiber's last node the foreground ends
    (its radius plus the mask margin; the radius by default), for the ends.
    The solver's centerlines come back as they are (voxels), so the caller
    gets exactly the admissible state the solver reached.

    Device nodes within ``anchor_tolerance`` of an ``anchors`` polyline are
    pinned during the image run: they do not move, and the other nodes fit
    around them. The settle after it runs unpinned, so pinned fibers that
    touch can still be pushed apart. This needs a Tangle build whose
    ``ImageRelaxer`` has ``set_pinned``; without it the anchors are ignored
    (the log says so).
    """
    import tangle

    if not lines:
        return []
    h = voxel_size
    radii = np.asarray(radii, dtype=np.float64)
    bends = np.asarray(bends, dtype=np.float64)
    smallest, largest = float(radii.min()), float(radii.max())
    pad = 3.0 * largest  # room around the scan so fibers that leave it are not squeezed by walls
    options: dict[str, Any] = {
        "max_iterations": iterations,
        "max_step": 0.25 * smallest * h,
        "penetration_tolerance": 0.02 * smallest * h,
        # Fits start overlapping; the default 48 neighbor slots overflow
        # into a much slower fallback.
        "neighbor_capacity": 192,
    }
    if backend:
        options["backend"] = backend
    settings = tangle.RelaxationSettings(**options)

    lines = _refine.respace(lines, spacing)
    lines = _refine.end_step(image, lines, radii, step=spacing, reach=reach)
    coarse = [resample(line, max(2.5 * float(r), spacing)) for line, r in zip(lines, radii)]
    payload = np.ascontiguousarray(image, dtype="<f4").tobytes()
    relaxer = _relaxer(image, payload, coarse, radii, bends, voxel_size=h, pad=pad, settings=settings)
    pinned = 0
    if anchors:
        from ._regrow import pinned_flags

        flags = pinned_flags(coarse, anchors, anchor_tolerance)
        pinned = int(sum(int(f.sum()) for f in flags))
        if pinned and hasattr(relaxer, "set_pinned"):
            relaxer.set_pinned([f.tolist() for f in flags])
        elif pinned:
            pinned = -pinned  # asked for, but this build cannot pin
    relaxer.set_image_force(rate, reach_radii=reach_radii)
    status = relaxer.run(iterations)
    if settle:
        relaxer.set_image_force(0.0)
        if pinned > 0:
            # Two pinned stretches that touch cannot push each other apart,
            # so the settle runs unpinned: with the image force off, it only
            # separates fibers in contact, and moves the rest little.
            relaxer.set_pinned([[False] * len(line) for line in coarse])
        status = relaxer.run(settle)
    if log is not None:
        extra = {k: status[k] for k in ("converged", "max_curvature_ratio") if k in status}
        if anchors:
            extra["pinned_nodes"] = pinned if pinned >= 0 else f"{-pinned} (not supported by this build)"
        log("solver", coarse, **extra)
    return [np.asarray(line, dtype=np.float64) / h - pad for line in relaxer.centerlines()]
