"""``FitResult.coarse_grained`` and ``FitResult.relax(adaptive=True)``: fewer nodes, then Tangle's contact solve,
and a relaxed fit that Tangle accepts as an assembly again (so it can be meshed or exported); and
``add_bpm_bonds``, which bonds the fibers of its bonded-particle file where the network found binder."""

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


def _read_bpm(path) -> tuple[dict, dict]:
    """A LAMMPS data file's header counts ({"bonds": 2, ...}) and its sections' rows, as numbers."""
    counts, sections, name = {}, {}, None
    for line in path.read_text().splitlines()[1:]:
        words = line.split("#")[0].split()
        if not words:
            continue
        if words[0][0].isalpha():
            name = words[0]
            sections[name] = []
        elif name is None:
            if not words[-1].endswith("hi"):  # not the box
                counts[" ".join(words[1:])] = int(words[0])
        else:
            sections[name].append([float(word) for word in words])
    return counts, sections


# Two fibers as export_bpm writes them, one capsule per segment: fiber 0 (molecule 1) along x at y 32, z 30, a long
# capsule over x 0-40 and a short one over 40-60; fiber 1 (molecule 2) along y at x 32, z 34, over y 0-32 and 32-64.
BPM = """TANGLE adaptive-capsule DEM-BPM fiber export
# atom type 1 = TANGLE material 0 "fiber" (4 capsules)

4 atoms
2 bonds
1 atom types
1 bond types

0.0 64.0 xlo xhi
0.0 64.0 ylo yhi
0.0 64.0 zlo zhi

Atoms # bpm/sphere

# id molecule type diameter density x y z
1 1 1 6.0 1000.0 20.0 32.0 30.0
2 1 1 6.0 1000.0 50.0 32.0 30.0
3 2 1 6.0 1000.0 32.0 16.0 34.0
4 2 1 6.0 1000.0 32.0 48.0 34.0

Capsules

# atom-id half-length axis-x axis-y axis-z
1 20.0 1.0 0.0 0.0
2 10.0 1.0 0.0 0.0
3 16.0 0.0 1.0 0.0
4 16.0 0.0 1.0 0.0

Bonds

# id type atom1 atom2
1 1 1 2
2 1 3 4
"""


@pytest.mark.parametrize("capsules", [True, False])
def test_add_bpm_bonds_bonds_each_pair_once_at_its_nearest_particles(tmp_path, capsules):
    path = tmp_path / "fibers_bpm.data"
    text = BPM if capsules else BPM[: BPM.index("Capsules")] + BPM[BPM.index("Bonds\n"):]  # spheres have no Capsules
    path.write_text(text)
    bonds = [{"fibers": (0, 1), "position": np.array([37.0, 30.0, 32.0])},  # voxels of 1 (the file's length unit)
             {"fibers": (1, 0), "position": np.array([10.0, 30.0, 32.0])},  # the same pair again: left out
             {"fibers": (0, 5), "position": np.array([10.0, 30.0, 32.0])}]  # no fiber 5 in the file: left out
    assert ct.add_bpm_bonds(path, bonds, voxel_size=1.0) == 1
    counts, sections = _read_bpm(path)
    assert counts == {"atoms": 4, "bonds": 3, "atom types": 1, "bond types": 2}
    assert len(sections["Bonds"]) == 3 and sections["Bonds"][:2] == [[1, 1, 1, 2], [2, 1, 3, 4]]
    # (37, 30, 32) lies along capsule 1 (x 0-40), though capsule 2's middle (x 50) is nearer than capsule 1's (x 20)
    assert sections["Bonds"][2] == ([3, 2, 1, 3] if capsules else [3, 2, 2, 3])
    assert ct.add_bpm_bonds(path, [], voxel_size=1.0) == 0 and _read_bpm(path)[0]["bonds"] == 3


@pytest.mark.skipif(not GPU, reason="Tangle's solve runs on the GPU")
def test_binder_bonds_join_the_relaxed_fibers_where_they_cross(tmp_path):
    relaxed, run = _fit(_crossing()).coarse_grained().relax(adaptive=True)
    path = tmp_path / "fibers_bpm.data"
    _, along = run.export_bpm(path)
    crossing = np.array([32.0, 32.0, 32.0])  # voxels: where the two fibers cross
    assert ct.add_bpm_bonds(path, [{"fibers": (0, 1), "position": crossing}], relaxed.voxel_size) == 1
    counts, sections = _read_bpm(path)
    assert counts["bonds"] == along + 1 == len(sections["Bonds"]) and counts["bond types"] == 2
    _, kind, a, b = sections["Bonds"][-1]
    atoms = {row[0]: row for row in sections["Atoms"]}  # id molecule type diameter density x y z
    assert kind == 2 and {atoms[a][1], atoms[b][1]} == {1, 2}
    capsules = {row[0]: row for row in sections.get("Capsules", [])}  # id half-length axis
    for atom in (a, b):  # each bonded particle reaches the crossing: within a radius and a bit of it
        center = np.array(atoms[atom][5:8])
        half, axis = (capsules[atom][1], np.array(capsules[atom][2:5])) if atom in capsules else (0.0, np.zeros(3))
        point = crossing * relaxed.voxel_size
        nearest = center + np.clip((point - center) @ axis, -half, half) * axis
        assert np.linalg.norm(nearest - point) < 4.5 * UM
