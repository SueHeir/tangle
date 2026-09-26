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
reads as one blob at 1. ``graded_image`` keeps that dip: each type's grey
over its range's typical (median) value, so a fiber's axis is brighter
than the contact beside it, for tracing and relaxing onto single fibers.
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
    *,
    denoise_sigma: float,
    flat: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """The 0…1 grey that keeps the dips between touching fibers.

    Per type, ``(v - void) / (core - void)`` up to 1, where ``core`` is the
    median grey of the voxels inside its range, fading out above the range
    as in ``range_image``; the largest over the types. With ``flat``
    (``range_image``'s image), the dim cores it filled read 1 here too.
    """
    from . import _native

    grey = np.asarray(volume, dtype=np.float32)
    if denoise_sigma > 0:
        grey = _native.gaussian(grey, denoise_sigma)
    image = np.zeros(grey.shape, dtype=np.float32)
    for low, high in ranges:
        inside = (grey >= low) & (grey <= high)
        core = float(np.median(grey[inside])) if inside.any() else float(high)
        rise = np.clip((grey - void) / max(core - void, 1e-12), 0.0, 1.0)
        fall = np.clip(1.0 - (grey - high) / (high - low), 0.0, 1.0)
        np.maximum(image, np.where(grey > high, fall, rise).astype(np.float32), out=image)
    if flat is not None:
        # Filled: fiber in the range image though darker than every range.
        image[(flat >= 1.0) & (grey < min(low for low, _ in ranges))] = 1.0
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
