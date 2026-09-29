"""Fiber diameter along each centerline, from the network's maps (optional step).

Every voxel the network calls fiber points at its own axis (the offset map). A voxel belongs to the node of a
traced fiber that its vote (voxel + offset) lands nearest, within ``CATCH`` voxels, so touching fibers keep
their own voxels. Per node:

* **area** = the voxels it owns per unit length of axis, smoothed along the fiber over ``SMOOTH`` voxels;
  diameter = 2 sqrt(area / pi) (the equal-area diameter).
* **oval**: the spread of those voxels in the plane across the axis. For a filled ellipse the variance along
  a semi-axis a is a^2 / 4, so the long and short diameters are 4 sqrt of the two eigenvalues, and the long
  axis is the leading eigenvector.

Nodes whose cross-section is cut by the volume's faces are left NaN.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from maps import OFFSET, TYPE

CATCH = 3.0  # voxels: how far a vote may land from the node it is counted for
SMOOTH = 7  # nodes (1 voxel apart) averaged along the fiber


def _resampled(line: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    segment = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(segment)]
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


class Accumulator:
    """Collects vote sums per node; feed it maps tile by tile, then call ``result``."""

    def __init__(self, lines, shape):
        self.shape = shape  # (z, y, x) of the whole volume
        self.nodes = [_resampled(np.asarray(l, float)) for l in lines]
        self.owner = np.concatenate([np.full(len(n), f) for f, n in enumerate(self.nodes)])
        self.all = np.concatenate(self.nodes)
        self.tree = cKDTree(self.all)
        n = len(self.all)
        self.count = np.zeros(n)
        self.first = np.zeros((n, 3))  # sum of voxel positions relative to the node
        self.second = np.zeros((n, 3, 3))

    def add(self, maps: np.ndarray, origin=(0, 0, 0), window=None):
        window = window or tuple(slice(0, s) for s in maps.shape[1:])
        kind = np.argmax(maps[TYPE][(slice(None), *window)], axis=0)
        z, y, x = np.nonzero(kind > 0)
        z, y, x = z + window[0].start, y + window[1].start, x + window[2].start
        voxel = np.stack([x + origin[2], y + origin[1], z + origin[0]], axis=1) + 0.5
        vote = voxel + maps[OFFSET][:, z, y, x].T
        d, i = self.tree.query(vote, distance_upper_bound=CATCH)
        ok = np.isfinite(d)
        i, rel = i[ok], voxel[ok] - self.all[i[ok]]
        np.add.at(self.count, i, 1.0)
        np.add.at(self.first, i, rel)
        np.add.at(self.second, i, rel[:, :, None] * rel[:, None, :])

    def result(self, spacing_um: float = 1.0):
        """Per fiber: nodes (x, y, z voxels, 1 apart), equal-area diameter, long and short diameter, long axis."""
        out = []
        start = 0
        bounds = np.array(self.shape[::-1], float)
        for nodes in self.nodes:
            k = slice(start, start + len(nodes))
            start += len(nodes)
            count = uniform_filter1d(self.count[k], SMOOTH, mode="nearest")
            diameter = 2.0 * np.sqrt(count / np.pi)
            tangent = np.gradient(nodes, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            n = np.maximum(self.count[k], 1.0)[:, None]
            mean = self.first[k] / n
            cov = self.second[k] / n[:, :, None] - mean[:, :, None] * mean[:, None, :]
            cov = uniform_filter1d(cov, SMOOTH, axis=0, mode="nearest")
            # project onto the plane across the axis
            p = np.eye(3)[None] - tangent[:, :, None] * tangent[:, None, :]
            plane = p @ cov @ p
            w, v = np.linalg.eigh(plane)
            long_d, short_d = 4.0 * np.sqrt(np.maximum(w[:, 2], 0)), 4.0 * np.sqrt(np.maximum(w[:, 1], 0))
            # a cross-section cut by a face of the volume reads too small
            reach = 0.5 * np.maximum(long_d, diameter) + 1.0
            inside = np.all((nodes - reach[:, None] >= 0) & (nodes + reach[:, None] <= bounds), axis=1)
            for a in (diameter, long_d, short_d):
                a[~inside] = np.nan
            out.append({"nodes": nodes, "diameter": diameter * spacing_um, "long": long_d * spacing_um,
                        "short": short_d * spacing_um, "long_axis": v[:, :, 2]})
        return out


def measure(maps: np.ndarray, lines, spacing_um: float = 1.0):
    acc = Accumulator(lines, maps.shape[1:])
    acc.add(maps)
    return acc.result(spacing_um)
