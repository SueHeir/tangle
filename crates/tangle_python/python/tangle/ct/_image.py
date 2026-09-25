"""Intensity normalization and local tube direction from the Hessian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    low, high = np.percentile(values, [0.1, 99.9])
    histogram, edges = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(histogram).astype(np.float64)
    total = weight[-1]
    mean = np.cumsum(histogram * centers)
    between = (mean[-1] * weight / total - mean) ** 2 / np.maximum(weight * (total - weight), 1e-12)
    return float(centers[np.argmax(between[:-1])])


@dataclass(frozen=True)
class Levels:
    """Grey levels used to map the scan to 0 (void) … 1 (fiber)."""

    void: float
    fiber: float
    threshold: float


def normalize(volume: np.ndarray, *, denoise_sigma: float, levels: Levels | None = None) -> tuple[np.ndarray, Levels]:
    """Return a float32 copy scaled so void is about 0 and fiber about 1."""
    from scipy.ndimage import gaussian_filter

    image = np.asarray(volume, dtype=np.float32)
    if denoise_sigma > 0:
        image = gaussian_filter(image, denoise_sigma)
    if levels is None:
        threshold = otsu_threshold(image[:: max(1, image.shape[0] // 64)])
        void = float(np.median(image[image < threshold]))
        fiber = float(np.median(image[image >= threshold]))
        levels = Levels(void=void, fiber=fiber, threshold=threshold)
    scale = levels.fiber - levels.void
    if abs(scale) < 1e-12:
        raise ValueError("the scan has no contrast between void and fiber levels")
    return (image - levels.void) / scale, levels


class HessianField:
    """Gaussian-scale Hessian of the normalized image, queried at arbitrary points."""

    def __init__(self, image: np.ndarray, sigma: float) -> None:
        from scipy.ndimage import gaussian_filter

        self.sigma = sigma
        # array axes are (z, y, x); store components in (x, y, z) order
        orders = {
            (0, 0): (0, 0, 2),
            (1, 1): (0, 2, 0),
            (2, 2): (2, 0, 0),
            (0, 1): (0, 1, 1),
            (0, 2): (1, 0, 1),
            (1, 2): (1, 1, 0),
        }
        scale = sigma**2
        self.components = {
            key: (scale * gaussian_filter(image, sigma, order=order)).astype(np.float32)
            for key, order in orders.items()
        }

    def at(self, points: np.ndarray) -> np.ndarray:
        from ._geometry import sample_image

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        h = np.zeros((len(points), 3, 3))
        for (i, j), component in self.components.items():
            values = sample_image(component, points)
            h[:, i, j] = values
            h[:, j, i] = values
        return h

    def directions(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Axis direction (smallest-magnitude eigenvector) and a tubularity score.

        The score is ``-(λ2 + λ3) / 2`` for eigenvalues ordered by magnitude,
        positive for a bright tube and near zero off-fiber.
        """
        values, vectors = np.linalg.eigh(self.at(points))
        order = np.argsort(np.abs(values), axis=1)
        rows = np.arange(len(values))[:, None]
        values = values[rows, order]
        vectors = np.take_along_axis(vectors, order[:, None, :], axis=2)
        axis = vectors[:, :, 0]
        tubularity = -0.5 * (values[:, 1] + values[:, 2])
        return axis, tubularity


def half_radius(profiles: np.ndarray, distances: np.ndarray, peak_reach: float) -> np.ndarray:
    """Per row of ``profiles`` (image values at ``distances`` from an axis):
    where it falls below half its peak, walking outward from the peak (found
    within ``peak_reach``), so a bright rim around a dim core reads its outer
    edge. The last distance when it never falls; 0 when the peak is not
    above void (0)."""
    profiles = np.atleast_2d(np.asarray(profiles, dtype=np.float64))
    rows = np.arange(len(profiles))
    limit = max(int(np.searchsorted(distances, peak_reach, side="right")), 1)
    top = np.argmax(profiles[:, :limit], axis=1)
    half = 0.5 * profiles[rows, top]
    index = np.arange(profiles.shape[1])
    below = (profiles < half[:, None]) & (index[None, :] > top[:, None])
    k = np.argmax(below, axis=1)
    a = profiles[rows, np.maximum(k - 1, 0)]
    b = profiles[rows, k]
    step = distances[1] - distances[0] if len(distances) > 1 else 0.0
    # Interpolate between the last sample above half and the first below.
    out = distances[np.maximum(k - 1, 0)] + step * (a - half) / np.maximum(a - b, 1e-12)
    out = np.where(below.any(axis=1), out, distances[-1])
    return np.where(half > 0.0, out, 0.0)


def half_widths(
    image: np.ndarray,
    lines: list[np.ndarray],
    reach: float,
    *,
    nodes: int = 24,
    step: float = 0.25,
) -> np.ndarray:
    """Each fiber's radius read from its mean cross-section in ``image`` (void 0), in voxels.

    At up to ``nodes`` interior nodes the image is sampled along four
    directions across the fiber, out to ``reach``; the median over nodes and
    directions at each distance is the fiber's cross-section, and its radius
    is where it falls below half its peak (:func:`half_radius`, the peak
    looked for within ``reach / 2``). NaN for a line of fewer than 3 nodes.

    This is a thickness that survives noise: the medians pool some hundred
    samples per distance, whereas the foreground's depth reads one
    thresholded voxel at a time. In a scan whose fibers sit a few noise
    sigma above void, a threshold that keeps a dim fiber's core also keeps
    void speckle, and one that drops the speckle punches holes into the
    core, so the depth there reads a fraction of the true radius.
    """
    from ._geometry import cross_frames, sample_image

    out = np.full(len(lines), np.nan)
    distances = np.arange(0.0, reach + 0.5 * step, step)
    frames = [cross_frames(line, nodes) for line in lines]
    used = [i for i, frame in enumerate(frames) if frame is not None]
    if not used:
        return out
    points = np.concatenate(
        [(frames[i][0][:, None, None, :] + frames[i][1][:, :, None, :] * distances[None, None, :, None]).reshape(-1, 3) for i in used]
    )
    values = sample_image(image, points).reshape(-1, len(distances))
    sizes = [4 * len(frames[i][0]) for i in used]
    profiles = np.stack([np.median(part, axis=0) for part in np.split(values, np.cumsum(sizes)[:-1])])
    out[used] = half_radius(profiles, distances, 0.5 * reach)
    return out
