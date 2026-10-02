"""``FitResult.coarse_grained`` and ``FitResult.relax(adaptive=True)``: fewer nodes, then Tangle's contact solve,
and a relaxed fit that Tangle accepts as an assembly again (so it can be meshed or exported)."""

import os

import numpy as np
import pytest

import tangle.ct as ct
from tangle.ct._geometry import coarse_nodes, within_bend_limit

UM = 1e-6
# The solve runs on the GPU; CI runners have none.
GPU = not (os.environ.get("CI") and "TANGLE_BACKEND" not in os.environ)


def _distance_to(polyline: np.ndarray, point: np.ndarray) -> float:
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    t = np.clip(((point - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1)))


def _curvature(points: np.ndarray) -> float:
    """The largest turn, measured as Tangle's validation does: 2 sin(angle / 2) over the mean segment length."""
    u, v = points[1:-1] - points[:-2], points[2:] - points[1:-1]
    lu, lv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    cosine = np.clip((u * v).sum(1) / (lu * lv), -1.0, 1.0)
    return float(np.max(2.0 * np.sin(0.5 * np.arccos(cosine)) / (0.5 * (lu + lv))))


def test_coarse_nodes_keep_the_shape_on_few_nodes():
    t = np.linspace(0.0, 1.0, 201)
    wave = np.stack([100.0 * t, 5.0 * np.sin(2.0 * np.pi * t), 0.0 * t], axis=1)
    keep = coarse_nodes(wave, 0.5, 30.0)
    assert keep[0] == 0 and keep[-1] == 200 and len(keep) < 20
    assert max(_distance_to(wave[keep], p) for p in wave) <= 0.5
    straight = np.stack([100.0 * t, 0.0 * t, 0.0 * t], axis=1)
    keep = coarse_nodes(straight, 0.5, 30.0)
    assert len(keep) == 5 and np.all(np.diff(straight[keep][:, 0]) <= 30.0)


def test_within_bend_limit_moves_only_what_bends_too_tightly():
    straight = np.stack([np.arange(20.0), np.zeros(20), np.zeros(20)], axis=1)
    same, moved = within_bend_limit(straight, bend=10.0)
    assert not moved and np.array_equal(same, straight)
    corner = np.concatenate([np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], 1),
                             np.stack([np.full(10, 9.0), np.arange(1.0, 11.0), np.zeros(10)], 1)])
    smooth, moved = within_bend_limit(corner, bend=10.0)
    assert moved and _curvature(smooth) <= 1.0 / 10.0
    assert np.array_equal(smooth[[0, -1]], corner[[0, -1]])


def _fit(lines) -> "ct.FitResult":
    """Fibers 6 um across in a 64 um cube of 1 um voxels, nodes every voxel."""
    lines = [np.asarray(line, float) for line in lines]
    return ct.FitResult(shape=(64, 64, 64), voxel_size=1 * UM, spec=ct.FiberSpec(diameter=6 * UM),
                        centerlines=lines, radii=np.full(len(lines), 3.0), support=np.ones(len(lines)),
                        levels=ct.Levels(0.0, 1.0, 0.5))


def _crossing():
    """Two straight fibers crossing at mid-height, their axes 4 um apart: they overlap by 2 um."""
    s = np.arange(4.0, 61.0)
    return [np.stack([s, np.full_like(s, 32.0), np.full_like(s, 30.0)], 1),
            np.stack([np.full_like(s, 32.0), s, np.full_like(s, 34.0)], 1)]


def test_coarse_grained_keeps_the_ends_and_drops_straight_nodes():
    fit = _fit(_crossing())
    coarse = fit.coarse_grained()
    assert sum(map(len, coarse.centerlines)) < sum(map(len, fit.centerlines)) // 5
    for before, after in zip(fit.centerlines, coarse.centerlines):
        assert np.array_equal(after[[0, -1]], before[[0, -1]])
        assert np.all(np.linalg.norm(np.diff(after, axis=0), axis=1) <= 10 * 6.0 + 1e-9)  # 10 diameters at most
    assert coarse.history[-1]["stage"] == "coarse-grained"


@pytest.mark.skipif(not GPU, reason="Tangle's solve runs on the GPU")
def test_adaptive_relax_separates_fibers_and_stays_valid_tangle():
    relaxed, run = _fit(_crossing()).coarse_grained().relax(adaptive=True)
    assert run.max_penetration < 0.1 * 6 * UM
    assert relaxed.history[-1]["adaptive"]
    relaxed.to_assembly()  # Tangle validates every fiber's shape against its bend limit here
