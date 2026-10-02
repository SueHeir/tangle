"""The binder in bonded-particle (DEM) files: ``add_bpm_binder_bonds``, ``add_bpm_binder_spheres`` and
``write_bpm_ovito``, on hand-written files in ``export_bpm``'s layout and, with a GPU, on Tangle's own export."""

import os

import numpy as np
import pytest

import tangle.ct as ct

UM = 1e-6
# The solve runs on the GPU; CI runners have none.
GPU = not (os.environ.get("CI") and "TANGLE_BACKEND" not in os.environ)


def read_bpm(path) -> tuple[dict, dict, list[str]]:
    """A data file's header counts ({"bonds": 2, ...}), its sections' rows as numbers, and its lines from the first
    section on."""
    lines = path.read_text().splitlines()
    first = next(index for index in range(1, len(lines)) if lines[index][:1].isalpha())
    counts, sections, name = {}, {}, None
    for index, line in enumerate(lines[1:], 1):
        words = line.split("#")[0].split()
        if not words:
            continue
        if index >= first and words[0][0].isalpha():
            name = words[0]
            sections[name] = []
        elif name is None:
            if not words[-1].endswith("hi"):  # not the box
                counts[" ".join(words[1:])] = int(words[0])
        else:
            sections[name].append([float(word) for word in words])
    return counts, sections, lines[first:]


def particles(sections) -> dict:
    """id: (molecule, type, radius, density, center, half-length, axis), from the Atoms and Capsules rows."""
    capsules = {row[0]: (row[1], np.array(row[2:5])) for row in sections.get("Capsules", [])}
    return {row[0]: (row[1], row[2], 0.5 * row[3], row[4], np.array(row[5:8]), *capsules.get(row[0], (0.0, np.zeros(3))))
            for row in sections["Atoms"]}


def segment_gap(a, b) -> float:
    """Between the surfaces of two particles (negative where they overlap): their axis segments' closest distance
    minus both radii (Ericson, Real-Time Collision Detection 5.1.9)."""
    p1, d1 = a[4] - a[5] * a[6], 2 * a[5] * a[6]
    p2, d2 = b[4] - b[5] * b[6], 2 * b[5] * b[6]
    r = p1 - p2
    aa, ee, ff = d1 @ d1, d2 @ d2, d2 @ r
    if aa < 1e-30 and ee < 1e-30:
        s = t = 0.0
    elif aa < 1e-30:
        s, t = 0.0, np.clip(ff / ee, 0, 1)
    else:
        c = d1 @ r
        if ee < 1e-30:
            s, t = np.clip(-c / aa, 0, 1), 0.0
        else:
            bb = d1 @ d2
            denominator = aa * ee - bb * bb
            s = np.clip((bb * ff - c * ee) / denominator, 0, 1) if denominator > 1e-30 else 0.0
            t = (bb * s + ff) / ee
            if t < 0:
                s, t = np.clip(-c / aa, 0, 1), 0.0
            elif t > 1:
                s, t = np.clip((bb - c) / aa, 0, 1), 1.0
    return float(np.linalg.norm(p1 + s * d1 - p2 - t * d2)) - a[2] - b[2]


def unbonded_overlaps(path, binder_type) -> list:
    """Pairs of a binder sphere and another particle that overlap with no bond between them."""
    _, sections, _ = read_bpm(path)
    every = particles(sections)
    bonded = {frozenset(row[2:4]) for row in sections["Bonds"]}
    spheres = [atom for atom, value in every.items() if value[1] == binder_type]
    return [(i, j) for i in spheres for j in every if j != i and every[j][0] != every[i][0]
            and frozenset((i, j)) not in bonded and segment_gap(every[i], every[j]) < -1e-12]


# Two fibers 6 across in a 64-wide box as export_bpm wrote them before the column legends moved to the header (so
# the old layout is read too), one capsule per segment: fiber 0 (molecule 1) along x at y 32, z 28, a long capsule
# over x 0-40 and a short one over 40-60; fiber 1 (molecule 2) along y at x 32, z 36, over y 0-32 and 32-64. Where
# they cross, 2 apart.
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
1 1 1 6.0 1000.0 20.0 32.0 28.0
2 1 1 6.0 1000.0 50.0 32.0 28.0
3 2 1 6.0 1000.0 32.0 16.0 36.0
4 2 1 6.0 1000.0 32.0 48.0 36.0

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


def write_bpm(tmp_path, capsules=True):
    path = tmp_path / "fibers_bpm.data"
    path.write_text(BPM if capsules else BPM[: BPM.index("Capsules")] + BPM[BPM.index("Bonds\n"):])
    return path


def binder_film() -> np.ndarray:
    """A (z, y, x) mask of the 64-wide box in voxels of 1: a film 2 thick filling the gap where the fibers cross,
    and a 6-wide blob clear of both."""
    binder = np.zeros((64, 64, 64), dtype=bool)
    binder[31:33, 24:40, 24:40] = True
    binder[50:56, 50:56, 8:14] = True
    return binder


@pytest.mark.parametrize("capsules", [True, False])
def test_binder_bonds_bond_each_pair_once_at_its_nearest_particles(tmp_path, capsules):
    path = write_bpm(tmp_path, capsules)
    bonds = [{"fibers": (0, 1), "position": np.array([37.0, 30.0, 32.0])},  # voxels of 1 (the file's length unit)
             {"fibers": (1, 0), "position": np.array([10.0, 30.0, 32.0])},  # the same pair again: left out
             {"fibers": (0, 5), "position": np.array([10.0, 30.0, 32.0])}]  # no fiber 5 in the file: left out
    assert ct.add_bpm_binder_bonds(path, bonds, voxel_size=1.0) == 1
    counts, sections, body = read_bpm(path)
    assert counts == {"atoms": 4, "bonds": 3, "atom types": 1, "bond types": 2}
    assert sections["Bonds"][:2] == [[1, 1, 1, 2], [2, 1, 3, 4]]
    # (37, 30, 32) lies along capsule 1 (x 0-40), though capsule 2's middle (x 50) is nearer than capsule 1's (x 20)
    assert sections["Bonds"][2] == ([3, 2, 1, 3] if capsules else [3, 2, 2, 3])
    assert not any(line.startswith("#") for line in body)  # the old layout's legends are gone: LAMMPS reads rows
    assert ct.add_bpm_binder_bonds(path, [], voxel_size=1.0) == 0 and read_bpm(path)[0]["bonds"] == 3


def test_binder_spheres_fill_the_binder_bonded_to_the_fibers_and_each_other(tmp_path):
    path = write_bpm(tmp_path)
    binder = binder_film()
    added = ct.add_bpm_binder_spheres(path, binder, voxel_size=1.0)
    counts, sections, body = read_bpm(path)
    every = particles(sections)
    spheres = {atom: value for atom, value in every.items() if value[1] == 2}
    assert added.spheres == len(spheres) >= 5 and added.diameter == 6.0  # the fibers' diameter
    assert counts == {"atoms": 4 + added.spheres, "bonds": 2 + added.fiber_bonds + added.sphere_bonds,
                      "atom types": 2, "bond types": 3}
    assert len(sections["Atoms"]) == counts["atoms"] and len(sections["Bonds"]) == counts["bonds"]
    assert len(sections["Capsules"]) == 4  # the fibers' capsules; a sphere has no Capsules row
    assert not any(line.startswith("#") for line in body)
    assert len({value[0] for value in spheres.values()}) == len(spheres)  # a molecule each
    # The spheres don't overlap each other, nor any particle they aren't bonded to.
    for i in spheres:
        for j in spheres:
            assert i == j or segment_gap(spheres[i], spheres[j]) >= -1e-9
    assert unbonded_overlaps(path, binder_type=2) == []
    # Each sphere carries the mass of its binder: in all, the binder outside the fibers, at the fibers' density.
    z, y, x = np.nonzero(binder)
    voxels = np.stack([x, y, z], axis=1) + 0.5
    inside = np.zeros(len(voxels), dtype=bool)
    for value in every.values():
        if value[1] == 1:
            inside |= np.array([segment_gap(value, (0, 0, 0.0, 0, v, 0.0, np.zeros(3))) <= 0 for v in voxels])
    mass = sum(value[3] * 4 / 3 * np.pi * value[2] ** 3 for value in spheres.values())
    assert mass == pytest.approx(1000.0 * (~inside).sum(), rel=1e-9)
    # The blob's one sphere is on its own; the film's spheres hang together and hold both fibers.
    bonds = [(int(row[2]), int(row[3])) for row in sections["Bonds"] if row[1] in (2, 3)]
    blob = [atom for atom, value in spheres.items() if value[4][2] > 45]
    assert len(blob) == 1 and not any(blob[0] in pair for pair in bonds)
    linked = {atom: {atom} for atom in every}
    for a, b in bonds:
        joined = linked[a] | linked[b]
        for atom in joined:
            linked[atom] = joined
    film = next(atom for atom in spheres if atom not in blob)
    assert {every[atom][0] for atom in linked[film]} >= {1, 2}  # both fibers
    assert all(linked[atom] == linked[film] for atom in spheres if atom not in blob)


def test_binder_spheres_shrink_clear_of_a_fiber_their_binder_misses(tmp_path):
    path = write_bpm(tmp_path)
    binder = np.zeros((64, 64, 64), dtype=bool)
    binder[34:39, 8:14, 36:38] = True  # x 36-38 beside fiber 1 (surface at x 35), more than a voxel clear of it
    added = ct.add_bpm_binder_spheres(path, binder, voxel_size=1.0)
    assert added.spheres >= 1 and added.full_size < added.spheres and added.fiber_bonds == 0
    assert unbonded_overlaps(path, binder_type=2) == []
    smaller = ct.add_bpm_binder_spheres(write_bpm(tmp_path), binder_film(), voxel_size=1.0, diameter=3.0)
    assert smaller.spheres > ct.add_bpm_binder_spheres(write_bpm(tmp_path), binder_film(), voxel_size=1.0).spheres


def test_binder_spheres_need_the_binder_on_the_files_grid(tmp_path):
    with pytest.raises(ValueError, match="binder covers"):
        ct.add_bpm_binder_spheres(write_bpm(tmp_path), np.zeros((64, 64, 32), dtype=bool), voxel_size=1.0)


def test_the_ovito_dump_holds_particles_and_the_bonds_between_them(tmp_path):
    path = write_bpm(tmp_path)
    added = ct.add_bpm_binder_spheres(path, binder_film(), voxel_size=1.0)
    dump = ct.write_bpm_ovito(path)
    lines = dump.read_text().splitlines()
    header = lines[8].split()[2:]
    rows = np.array([[float(value) for value in line.split()] for line in lines[9:]])
    assert header[:3] == ["id", "mol", "type"] and len(rows[0]) == len(header)
    kind = rows[:, header.index("type")]
    assert (kind == 1).sum() == 4 and (kind == 2).sum() == added.spheres
    assert (kind == 3).sum() == added.fiber_bonds + added.sphere_bonds  # the two bonds along the fibers left out
    capsule = rows[kind == 1][0]
    assert capsule[header.index("AsphericalShape.Z")] == pytest.approx(40.0 * 1e6)  # micrometers: 40 long
    assert int(float(lines[3])) == len(rows)


@pytest.mark.skipif(not GPU, reason="Tangle's solve runs on the GPU")
@pytest.mark.parametrize("mode", ["spherocylinders_exact", "spheres_exact"])
def test_tangles_own_export_takes_both_binder_styles(tmp_path, mode):
    s = np.arange(4.0, 61.0)
    lines = [np.stack([s, np.full_like(s, 32.0), np.full_like(s, 30.0)], 1),
             np.stack([np.full_like(s, 32.0), s, np.full_like(s, 34.0)], 1)]
    fit = ct.FitResult(shape=(64, 64, 64), voxel_size=1 * UM, spec=ct.FiberSpec(diameter=6 * UM), centerlines=lines,
                       radii=np.full(2, 3.0), support=np.ones(2), levels=ct.Levels(0.0, 1.0, 0.5))
    relaxed, run = fit.coarse_grained().relax(adaptive=True)
    binder = np.zeros((64, 64, 64), dtype=bool)
    binder[26:38, 26:38, 26:38] = True  # around the crossing
    bonds_file, spheres_file = tmp_path / "bonds.data", tmp_path / "spheres.data"
    _, along = run.export_bpm(bonds_file, mode=mode)
    assert ct.add_bpm_binder_bonds(bonds_file, [{"fibers": (0, 1), "position": np.full(3, 32.0)}], fit.voxel_size) == 1
    run.export_bpm(spheres_file, mode=mode)
    added = ct.add_bpm_binder_spheres(spheres_file, binder, fit.voxel_size)
    assert added.spheres > 0 and added.fiber_bonds > 0
    for path in (bonds_file, spheres_file):
        counts, sections, body = read_bpm(path)
        # As LAMMPS reads it: a blank line after each section's title, then only rows.
        assert not any(line.startswith("#") for line in body)
        for name in sections:
            title = next(index for index, line in enumerate(body) if line.split()[:1] == [name])
            assert body[title + 1] == ""
        assert len(sections["Atoms"]) == counts["atoms"] and len(sections["Bonds"]) == counts["bonds"]
    assert unbonded_overlaps(spheres_file, binder_type=2) == []
