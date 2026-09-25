"""Discrete topology moves: remove duplicates and unsupported fibers, merge
fragments that continue each other, and add fibers where the image is not
yet explained (births are done by re-tracing, see ``_fit``).

The work happens in the ``tangle_ct`` crate (``moves.rs``, ``render.rs``);
this module keeps the Python signatures.
"""

from __future__ import annotations

import numpy as np

from . import _native

EDGE = 1.2  # soft edge half-width of rendered capsules, voxels


def trim_duplicates(
    centerlines: list[np.ndarray], radii: np.ndarray, *, min_length: float, closeness: float = 0.8
) -> tuple[list[np.ndarray], np.ndarray]:
    """Remove fibers mostly lying inside another fiber, and trim end runs that do.

    Two distinct fibers never have centerlines closer than the sum of their
    radii, so nodes within ``closeness * radius`` of another fiber's nodes are
    re-traces of that fiber.
    """
    return _native.trim_duplicates(centerlines, radii, min_length=min_length, closeness=closeness)


def remove_unsupported(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_length: float,
    min_support: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Remove fibers shorter than ``min_length`` or with mean image value below ``min_support``."""
    return _native.remove_unsupported(image, centerlines, radii, min_length=min_length, min_support=min_support)


def merge_fragments(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    max_gap: float,
    max_angle_degrees: float = 35.0,
    min_bridge_support: float = 0.45,
    min_bend_radius: float | None = None,
    kink_threshold: float = 2.0,
    end_cost: float = 0.0,
    scale: float = 1.0,
    max_prior_gap: float | None = None,
    max_prior_angle_degrees: float = 45.0,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Join pairs of ends that continue each other across a short gap.

    With ``min_bend_radius``, a join that would create a kink the split step
    would cut again is not made.

    With an ``end_cost`` (the fiber-length prior, see ``_ends``), the fixed
    gap, angle and bridge tests are replaced by a joint choice: every pair of
    ends within ``max_prior_gap`` (default four ``max_gap``) whose directions
    (and the gap between them) agree within ``max_prior_angle_degrees`` is a
    candidate; the ends may also overlap by up to two radii. A candidate is
    scored in nats: the drop in squared residual when the neighborhood is
    rendered with the joined fiber instead of the two pieces, divided by
    ``scale``, plus ``2 * end_cost`` for the two ends the join removes.
    Candidates are taken best first, each end joins at most once, and joins
    that would close a loop are skipped, so pieces meeting at a crossing are
    paired the way the scan supports best.
    """
    return _native.merge_fragments(
        image, centerlines, radii, max_gap=max_gap, max_angle_degrees=max_angle_degrees,
        min_bridge_support=min_bridge_support, min_bend_radius=min_bend_radius, kink_threshold=kink_threshold,
        end_cost=end_cost, scale=scale, max_prior_gap=max_prior_gap, max_prior_angle_degrees=max_prior_angle_degrees,
    )


def split_kinks(
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_bend_radius: float,
    min_length: float,
    max_length: float | None = None,
    threshold: float = 2.0,
    min_angle_degrees: float = 35.0,
    image: np.ndarray | None = None,
    end_cost: float = 0.0,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Split fibers at kinks sharper than the bend limit, and overlong fibers.

    A trace that runs from one fiber onto another develops a kink once the fit
    pulls each part onto its own fiber. The turn is measured over two node
    spacings on each side to ignore node-scale noise; ``threshold`` is the
    allowed multiple of the admissible curvature. Fibers longer than
    ``max_length`` are cut where the image support along them is weakest.

    With an ``end_cost`` (the fiber-length prior, see ``_ends``), a kink is
    first smoothed out locally; the fiber is only cut when the smoothed
    fiber matches ``image`` worse than the kinked one by more than the cost of
    the two new ends. A trace that jumped onto another fiber cannot be smoothed
    without leaving both fibers, so it is still cut.
    """
    return _native.split_kinks(
        centerlines, radii, min_bend_radius=min_bend_radius, min_length=min_length, max_length=max_length,
        threshold=threshold, min_angle_degrees=min_angle_degrees, image=image, end_cost=end_cost, scale=scale,
    )


def render_occupancy(
    box_low: np.ndarray, box_high: np.ndarray, lines: list[np.ndarray], radii: np.ndarray, edge: float = EDGE
) -> np.ndarray:
    """Soft union occupancy of capsules over the voxel box ``[low, high)`` (x, y, z)."""
    return _native.render_occupancy(box_low, box_high, lines, radii, edge)


def resolve_side_by_side(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    min_length: float,
    reach: float = 2.3,
    end_cost: float = 0.0,
    scale: float = 1.0,
    max_angle_degrees: float = 30.0,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Decide, from the image, whether two adjacent parallel fits are one fiber.

    When two fits land on one fiber, contact pushes them apart until each
    sits about a radius off the true axis, where they look like two touching
    fibers. For every pair running side by side (at least 3 nodes of one fit
    within ``reach`` radii of the other, with tangents within
    ``max_angle_degrees``), the image in their neighborhood is compared with
    two renderings: both fibers as they are, and one fiber along their
    midline (the weaker fit is removed there). The rendering with the
    smaller squared residual wins. With an ``end_cost`` (the fiber-length
    prior), each fiber end a rendering adds or removes is charged or
    credited ``end_cost`` nats, the residual being measured in units of
    ``scale``. Fits that only cross are not tested.
    """
    return _native.resolve_side_by_side(
        image, centerlines, radii, min_length=min_length, reach=reach, end_cost=end_cost, scale=scale,
        max_angle_degrees=max_angle_degrees,
    )
