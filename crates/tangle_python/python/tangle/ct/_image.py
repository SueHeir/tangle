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
