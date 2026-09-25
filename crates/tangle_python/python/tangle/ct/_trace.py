"""Initial fiber centerlines: seed on the distance-transform ridge, then trace (in Rust).

Each trace steps along the local tube axis (Hessian eigenvector blended with
momentum), re-centers in the cross-sectional plane on the image intensity,
limits the turn per step by the admissible bend radius, and stops when the
core intensity drops below the fiber/void midpoint. Voxels claimed by earlier
traces are down-weighted, so a trace keeps its direction through a crossing
instead of turning onto the other fiber. The work happens in the
``tangle_ct`` crate (``trace.rs``); this module keeps the Python signatures.
"""

from __future__ import annotations

import numpy as np

from . import _native
from ._image import HessianField


class Tracer:
    """Traces through ``image`` (void 0, fiber 1) with one fiber type's settings.

    ``claimed`` (int32, the image's shape) is read at every call, so the
    caller may paint it between calls."""

    def __init__(
        self,
        image: np.ndarray,
        hessian: HessianField,
        *,
        radius: float,
        min_bend_radius: float,
        step: float,
        claimed: np.ndarray,
    ) -> None:
        self.image = np.ascontiguousarray(image, dtype=np.float32)
        self.hessian = hessian
        self.radius = float(radius)
        self.min_bend_radius = float(min_bend_radius)
        self.step = float(step)
        self.claimed = claimed

    def trace_one_way(
        self, start: np.ndarray, direction: np.ndarray, max_steps: int, own_label: int = 0
    ) -> list[np.ndarray]:
        """Step from ``start`` along ``direction`` until the core leaves the fiber.

        Voxels of ``claimed`` labelled ``own_label`` are not down-weighted
        (the fiber being extended).
        """
        points = _native.trace_one_way(
            self.image, self.hessian.native, self.claimed, radius=self.radius,
            min_bend_radius=self.min_bend_radius, step=self.step, start=start, direction=direction,
            max_steps=max_steps, own_label=own_label,
        )
        return list(points)


def foreground_depth(foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The foreground's distance transform (float32) and its 3×3×3 maximum."""
    return _native.foreground_depth(foreground)


def trace_fibers(
    image: np.ndarray,
    hessian: HessianField,
    *,
    radius: float,
    min_bend_radius: float,
    min_length: float,
    node_spacing: float,
    claimed: np.ndarray | None = None,
    foreground: np.ndarray | None = None,
    label_offset: int = 0,
    max_fibers: int | None = None,
    seed_depth_radii: float = 0.5,
    depth: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Trace fibers from ridge seeds that are not yet explained by ``claimed``.

    Seeds are distance-transform ridge voxels, deepest first, at least
    ``seed_depth_radii`` radii from the void, so a large fiber type is only
    seeded where the foreground is that thick. ``depth`` is the foreground's
    distance transform and its 3×3×3 maximum, when the caller has them."""
    if claimed is None:
        claimed = np.zeros(image.shape, dtype=np.int32)
    else:
        claimed = np.array(claimed, dtype=np.int32, order="C")  # a copy: painted as fibers are found
    if depth is None:
        depth = foreground_depth(image > 0.5 if foreground is None else foreground)
    edt, peak = depth
    return _native.trace_fibers(
        image, hessian.native, claimed, edt, peak, radius=radius, min_bend_radius=min_bend_radius,
        step=max(0.75, 0.5 * radius), min_length=min_length, node_spacing=node_spacing,
        label_offset=label_offset, max_fibers=max_fibers, seed_depth_radii=seed_depth_radii,
    )
