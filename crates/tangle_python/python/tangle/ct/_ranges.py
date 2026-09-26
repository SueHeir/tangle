"""Raw grey scans with a grey range per fiber type.

Each fiber type says which grey values its voxels take in the scan
(``FiberSpec.intensity = (low, high)``, in the scan's own units, after the
light denoise). That decides what is fiber:

* a voxel inside a type's range is fiber of that type;
* a voxel between the void grey and a range's low end is partly fiber
  (a fiber edge cuts it), in proportion: ``(v - void) / (low - void)``;
* a voxel brighter than a range's high end fades out over one range width,
  so noise inside a fiber still counts but a much brighter inclusion does
  not;
* anything else is void.

The fit image is the largest of these over the types (0 void … 1 fiber),
so the solver sees sub-voxel edges rather than a hard mask boundary. Which
range a fiber's core voxels fall in is kept per voxel (``types``, one bit
per type) and helps decide each fiber's type.

A range is wide enough that the dimmer grey where two fibers touch is in
it too, so the fit image is flat across touching fibers: a packed bundle
reads as one blob at 1. ``graded_image`` keeps that dip, dimming each
voxel by how far its grey falls below the brightest grey near it, so a
fiber's axis is brighter than the contact beside it, for tracing and
relaxing onto single fibers.
"""

from __future__ import annotations

import numpy as np


def range_image(
    volume: np.ndarray,
    ranges: list[tuple[float, float]],
    *,
    denoise_sigma: float,
    exclude: np.ndarray | None = None,
    fill_holes_area: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The 0…1 fit image, the per-voxel type bits, and the void grey.

    The void grey is the median of the voxels below every range. With
    ``fill_holes_area``, slice holes no larger than that (a dim fiber core
    below its range) are filled as fiber, and a hole enclosed by one type's
    grey gets that type's bit.
    """
    from . import _native

    from ._fit import _core_holes

    grey = np.asarray(volume, dtype=np.float32)
    if denoise_sigma > 0:
        grey = _native.gaussian(grey, denoise_sigma)
    lows = np.array([low for low, _ in ranges], dtype=np.float64)
    highs = np.array([high for _, high in ranges], dtype=np.float64)
    if np.any(highs <= lows):
        raise ValueError("every intensity range needs low < high")
    below = grey < lows.min()
    if not below.any():
        raise ValueError(
            "no voxel is darker than every intensity range, so there is no void to compare with"
        )
    void = float(np.median(grey[below]))
    if void >= lows.min():
        raise ValueError("the void grey is not below the intensity ranges")

    image = np.zeros(grey.shape, dtype=np.float32)
    types = np.zeros(grey.shape, dtype=np.uint8)
    for k, (low, high) in enumerate(zip(lows, highs)):
        inside = (grey >= low) & (grey <= high)
        types |= inside.astype(np.uint8) << k
        rise = np.clip((grey - void) / (low - void), 0.0, 1.0)
        fall = np.clip(1.0 - (grey - high) / (high - low), 0.0, 1.0)
        np.maximum(
            image,
            np.where(grey < low, rise, np.where(grey > high, fall, 1.0)).astype(
                np.float32
            ),
            out=image,
        )
    if fill_holes_area:
        holes = _core_holes(image > 0.5, fill_holes_area)
        image[holes] = 1.0
        for k in range(len(ranges)):
            # A dim core inside a ring of this type's grey is this type too.
            plane = ((types >> k) & 1).astype(bool)
            types |= _core_holes(plane, fill_holes_area).astype(np.uint8) << k
    if exclude is not None:
        excluded = np.asarray(exclude, dtype=bool)
        image[excluded] = 0.0
        types[excluded] = 0
    return image, types, void


def graded_image(
    volume: np.ndarray,
    ranges: list[tuple[float, float]],
    void: float,
    flat: np.ndarray,
    *,
    reach: float,
    denoise_sigma: float,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """``flat`` (``range_image``'s image) dimmed where the grey dips below its surroundings.

    Each voxel's grey above void over the brightest grey above void within
    ``reach`` voxels (a cube), up to 1, times ``flat``: a fiber's axis keeps
    ``flat``, and the contact between two touching fibers, dimmer than both
    axes beside it, reads that much lower. Grading each type against its own
    level instead would not do: the contact between two bright fibers is as
    bright as a dimmer type's core. ``reach`` is about the smallest fiber
    radius. Voxels ``range_image`` filled as dim cores (fiber there though
    darker than every range) stay 1.
    """
    from . import _native

    grey = np.asarray(volume, dtype=np.float32)
    if denoise_sigma > 0:
        grey = _native.gaussian(grey, denoise_sigma)
    above = np.maximum(grey - np.float32(void), np.float32(0.0))
    local = above.copy()
    steps = max(int(np.ceil(reach)), 1)
    for axis in range(3):
        for _ in range(steps):  # a running 3-wide maximum, ``steps`` times: a (2 steps + 1)-wide one
            ahead = np.take(local, range(1, local.shape[axis]), axis=axis)
            behind = np.take(local, range(0, local.shape[axis] - 1), axis=axis)
            head = [slice(None)] * 3
            tail = [slice(None)] * 3
            head[axis] = slice(0, -1)
            tail[axis] = slice(1, None)
            grown = local.copy()
            np.maximum(grown[tuple(head)], ahead, out=grown[tuple(head)])
            np.maximum(grown[tuple(tail)], behind, out=grown[tuple(tail)])
            local = grown
    relative = np.clip(above / np.maximum(local, np.float32(1e-12)), 0.0, 1.0)
    image = (np.asarray(flat, dtype=np.float32) * relative).astype(np.float32)
    filled = (flat >= 1.0) & (grey < min(low for low, _ in ranges))
    image[filled] = 1.0
    if exclude is not None:
        image[np.asarray(exclude, dtype=bool)] = 0.0
    return image


def type_fractions(types: np.ndarray, points: np.ndarray, count: int) -> np.ndarray:
    """For points along a centerline, the fraction whose voxel is in each type's range."""
    from ._geometry import sample_labels

    bits = sample_labels(types, points)
    if len(bits) == 0:
        return np.zeros(count)
    return np.array([float(((bits >> k) & 1).mean()) for k in range(count)])
