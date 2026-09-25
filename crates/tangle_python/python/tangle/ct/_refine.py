"""Host-side steps of the continuous fit: fiber ends, node spacing, support (in Rust).

The fibers themselves move in Tangle's solver (``_device``). The work
happens in the ``tangle_ct`` crate (``refine.rs``); this module keeps the
Python signatures.
"""
from __future__ import annotations

import numpy as np

from . import _native


def end_step(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    step: float,
    max_moves: int = 3,
    reach: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Grow or trim fiber ends by up to ``max_moves`` steps of length ``step``.

    A fiber's capsule reaches a radius past its last node, so the scan's
    foreground ends about ``reach`` (the radius, plus any margin by which the
    foreground over-reaches) beyond the true end of the centerline. An end
    grows while the scan is still fiber half a step past that, and is trimmed
    while it is void half a step short of it, which leaves the tip within
    half a step of the true end.

    An end does not grow into a voxel another of ``centerlines`` owns (the
    nearest capsule surface within its radius, as :class:`_geometry.OwnerLookup`).
    """
    reach = np.asarray(radii if reach is None else reach, dtype=np.float64)
    return _native.end_step(image, centerlines, radii, reach, step=step, max_moves=max_moves)


def cut_void(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    *,
    level: float = 0.3,
    min_gap_radii: float = 2.0,
    bridge_level: float = 0.7,
    bridge_offset_radii: float = 1.0,
    directions=None,
    aligned_level: float = 0.5,
    aligned_angle_degrees: float = 15.0,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, int]]:
    """Cut every fit where its centerline sits in void (image below ``level``).

    A fit that has drifted off its fiber into empty space (the solver's image
    force only pulls toward fiber, so void never pushes back) keeps a tail
    there, and another fit can take the place it left. Void nodes at an end
    are trimmed back to the first supported node. An interior void stretch
    shorter than ``min_gap_radii`` radii is kept (a dip, not a drift); a
    longer one is bridged when it is a bow, reaching at least
    ``bridge_offset_radii`` radii off the straight line between its
    supported neighbors, and that line reads at least ``bridge_level`` all
    the way (the fit bowed off its fiber and came back); otherwise it splits
    the fit. A dim stretch that barely leaves the line is where a fit hops
    from one fiber to another at a crossing: its chord crosses the gap
    between touching fibers and reads only 0.4 to 0.6, and bridging it kept
    the merge. With ``directions`` (``directions(fit, points)``: the scan's
    local fiber axis at each point), a stretch whose line reads at least
    ``aligned_level`` and lies within ``aligned_angle_degrees`` of the scan's
    axis at both supported neighbors is bridged too: the fiber visibly runs
    on along it. Nodes outside the scan are left alone. :func:`end_step` regrows an end where the scan does
    continue.

    Returns ``(pieces, source, counts)``: the pieces, the index of the fit
    each came from, and the numbers of nodes dropped (``"trimmed"``), of
    splits and of stretches bridged (``"bridged"``).
    """
    pieces, source, (trimmed, splits, bridged) = _native.cut_void(
        image, centerlines, np.asarray(radii, dtype=np.float64), level=level, min_gap_radii=min_gap_radii,
        bridge_level=bridge_level, bridge_offset_radii=bridge_offset_radii, aligned_level=aligned_level,
        aligned_angle_degrees=aligned_angle_degrees, directions=directions,
    )
    return pieces, source, {"trimmed": trimmed, "splits": splits, "bridged": bridged}


def respace(centerlines: list[np.ndarray], spacing: float) -> list[np.ndarray]:
    return _native.resample(centerlines, spacing)


def support(image: np.ndarray, centerlines: list[np.ndarray]) -> np.ndarray:
    """Mean normalized intensity along each centerline (about 1 on a real fiber)."""
    return _native.support(image, centerlines)


def curvature_ratio(centerlines: list[np.ndarray], min_bend_radius: float) -> np.ndarray:
    """Largest discrete curvature times the admissible bend radius, per fiber."""
    return _native.curvature_ratio(centerlines, min_bend_radius)
