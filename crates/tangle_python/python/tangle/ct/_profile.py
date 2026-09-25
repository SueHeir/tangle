"""Cross-section brightness profiles: solid fibers, and fibers with a bright
rim around a dimmer core.

Brightness is on the normalized scale: void is 0 and 1 is the reference
fiber level (the brightest fiber type). A solid profile has ``brightness``
everywhere inside the fiber. A rimmed profile has ``brightness`` in the outer
``rim`` of the radius and ``core × brightness`` inside it, as in skin-core
fibers or fibers whose centers are dimmer in the scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossSection:
    """How bright a fiber is across its cross-section.

    ``brightness``
        Brightness of the fiber (of its rim, if it has one) on a scale where
        void is 0 and the brightest fiber type is 1.
    ``rim``
        Rim thickness in meters; ``None`` for a solid fiber.
    ``core``
        Brightness inside the rim, as a fraction of the rim brightness.
    """

    brightness: float = 1.0
    rim: float | None = None
    core: float = 1.0

    @property
    def solid(self) -> bool:
        return self.rim is None or self.core >= 1.0

    def inner_radius(self, radius: float, voxel_size: float) -> float:
        """Radius of the dim core in voxels (0 for a solid fiber)."""
        if self.solid:
            return 0.0
        return max(radius - self.rim / voxel_size, 0.0)

    def level(self, rho: np.ndarray, radius: float, voxel_size: float) -> np.ndarray:
        """Brightness at distance ``rho`` (voxels) from the axis, before blur."""
        rho = np.asarray(rho, dtype=np.float64)
        inside = rho <= radius
        if self.solid:
            return self.brightness * inside
        inner = self.inner_radius(radius, voxel_size)
        return self.brightness * np.where(rho < inner, self.core, 1.0) * inside

    def area(self, radius: float, voxel_size: float) -> float:
        """Integrated brightness of the cross-section (voxels²)."""
        inner = self.inner_radius(radius, voxel_size)
        return float(self.brightness * np.pi * (radius * radius - (1.0 - self.core) * inner * inner))

    def radius_from_area(self, area: np.ndarray, voxel_size: float, low: float, high: float) -> np.ndarray:
        """Invert :meth:`area`: the radius whose profile integrates to ``area``."""
        area = np.asarray(area, dtype=np.float64)
        if self.solid:
            return np.sqrt(np.maximum(area, 0.0) / (np.pi * self.brightness))
        grid = np.linspace(max(low, 1e-3), high, 256)
        values = np.array([self.area(r, voxel_size) for r in grid])
        return np.interp(area, values, grid)

    def center_response(self, radius: float, voxel_size: float, sigma: float) -> float:
        """Brightness at the axis after a Gaussian blur of ``sigma`` voxels.

        Used to rescale the per-type detection image so this type reads about
        1 on its axis.
        """
        if sigma <= 0:
            return float(self.level(np.array([0.0]), radius, voxel_size)[0])
        rho = np.linspace(0.0, radius, 400)
        weights = rho * np.exp(-rho * rho / (2.0 * sigma * sigma))
        full = 2.0 * np.pi * sigma * sigma  # the 2D Gaussian's integral over the plane
        return float(2.0 * np.pi * np.sum(weights * self.level(rho, radius, voxel_size)) * (rho[1] - rho[0]) / full)
