"""The binder in a bonded-particle (DEM) model of the fibers, two ways, and an OVITO view of the model.

``RunResult.export_bpm`` writes the relaxed fibers as a LAMMPS data file: particles along each fiber (one capsule
per relaxed segment by default, or spheres with ``mode="spheres_exact"``), bonded along each fiber. The binder that
:func:`find_fibers` finds goes into that file in one of two ways:

    >>> run.export_bpm("binder_bonds.data")
    >>> ct.add_bpm_binder_bonds("binder_bonds.data", fit.bonds, fit.voxel_size)        # bonds from fiber to fiber
    >>> run.export_bpm("binder_spheres.data")
    >>> ct.add_bpm_binder_spheres("binder_spheres.data", fit.binder, fit.voxel_size)   # the binder as spheres
    >>> ct.write_bpm_ovito("binder_spheres.data")                                      # binder_spheres.dump

Both keep the file's particles and bonds and add the binder with the next free atom and bond types, noted in the
file's header comments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class _DataFile:
    """A LAMMPS data file: the header (title, comments, counts, box) and the sections, each a keyword line and its
    data rows, written back in the layout ``export_bpm`` uses. Comment lines inside a section are dropped: LAMMPS
    would read them as rows."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        lines = self.path.read_text().splitlines()
        starts = [index for index in range(1, len(lines)) if lines[index][:1].isalpha()]  # line 0 is the title
        self.header = lines[: starts[0]] if starts else lines
        while self.header and not self.header[-1].strip():
            self.header.pop()
        self.sections: dict[str, list] = {}  # name: [keyword line, data rows]
        for start, stop in zip(starts, starts[1:] + [len(lines)]):
            rows = [line for line in lines[start + 1 : stop] if line.strip() and not line.lstrip().startswith("#")]
            self.sections[lines[start].split()[0]] = [lines[start], rows]
        if "Atoms" not in self.sections:
            raise ValueError(f"{self.path} has no Atoms section; is it a file RunResult.export_bpm wrote?")

    def rows(self, name: str) -> list[list[str]]:
        return [row.split() for row in self.sections.get(name, ["", []])[1]]

    def add_rows(self, name: str, rows: list[str]) -> None:
        self.sections.setdefault(name, [name, []])[1].extend(rows)

    def count(self, what: str) -> int:
        """A header count such as ``"bond types"``; 0 if the header doesn't give it."""
        for line in self.header:
            words = line.split("#")[0].split()
            if len(words) > 1 and words[1:] == what.split() and words[0].isdigit():
                return int(words[0])
        return 0

    def set_count(self, what: str, value: int) -> None:
        counts = []
        for index, line in enumerate(self.header):
            words = line.split("#")[0].split()
            if len(words) > 1 and words[0].isdigit() and words[-1] not in ("xhi", "yhi", "zhi"):
                counts.append(index)
                if words[1:] == what.split():
                    self.header[index] = f"{value} {what}"
                    return
        self.header.insert(counts[-1] + 1 if counts else len(self.header), f"{value} {what}")

    def note(self, text: str) -> None:
        """A comment line after the header's other leading comments."""
        index = 1
        while index < len(self.header) and self.header[index].startswith("#"):
            index += 1
        self.header.insert(index, f"# {text}")

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = np.zeros(3), np.zeros(3)
        for line in self.header:
            words = line.split()
            for axis, name in enumerate("xyz"):
                if words[2:4] == [f"{name}lo", f"{name}hi"]:
                    low[axis], high[axis] = float(words[0]), float(words[1])
        return low, high

    def write(self) -> None:
        out = list(self.header)
        for keyword, rows in self.sections.values():
            out += ["", keyword, ""] + rows
        self.path.write_text("\n".join(out) + "\n")


@dataclass
class _Particles:
    """The particles of a data file: spheres, or capsules (a segment ``center ± half * axis`` swept by ``radius``)."""

    ids: np.ndarray
    molecules: np.ndarray
    types: np.ndarray
    radii: np.ndarray
    densities: np.ndarray
    centers: np.ndarray
    half: np.ndarray  # 0 for a sphere
    axes: np.ndarray  # unit vectors; 0 for a sphere

    @classmethod
    def read(cls, data: _DataFile) -> "_Particles":
        values = np.array([[float(value) for value in row[:8]] for row in data.rows("Atoms")]).reshape(-1, 8)
        ids = values[:, 0].astype(np.int64)  # columns: id molecule type diameter density x y z
        half, axes = np.zeros(len(ids)), np.zeros((len(ids), 3))
        if "Capsules" in data.sections:  # atom-id half-length axis-x axis-y axis-z
            row_of = {atom: index for index, atom in enumerate(ids.tolist())}
            for row in data.rows("Capsules"):
                half[row_of[int(row[0])]] = float(row[1])
                axes[row_of[int(row[0])]] = [float(value) for value in row[2:5]]
        return cls(ids, values[:, 1].astype(np.int64), values[:, 2].astype(np.int64), 0.5 * values[:, 3],
                   values[:, 4], values[:, 5:8], half, axes)

    def in_voxels(self, corner: np.ndarray, voxel_size: float) -> "_Particles":
        """The same particles with lengths in voxels and positions measured from ``corner``."""
        return _Particles(self.ids, self.molecules, self.types, self.radii / voxel_size, self.densities,
                          (self.centers - corner) / voxel_size, self.half / voxel_size, self.axes)

    def distance(self, chosen: np.ndarray, point: np.ndarray) -> np.ndarray:
        """From ``point`` to the axis segment of each chosen particle."""
        center, axis, half = self.centers[chosen], self.axes[chosen], self.half[chosen]
        along = np.clip(((point - center) * axis).sum(1), -half, half)
        return np.linalg.norm(center + along[:, None] * axis - point, axis=1)


def _segment_distance(points: np.ndarray, center: np.ndarray, axis: np.ndarray, half: float) -> np.ndarray:
    """From each of ``points`` to the segment ``center ± half * axis``."""
    along = np.clip((points - center) @ axis, -half, half)
    return np.linalg.norm(points - (center + along[:, None] * axis), axis=1)


def _next_type(data: _DataFile, what: str, used) -> int:
    return max([data.count(what), *(int(value) for value in used)]) + 1


def add_bpm_binder_bonds(path: str | os.PathLike, bonds, voxel_size: float, *, bond_type: int | None = None) -> int:
    """Put the binder into a bonded-particle (DEM) file that ``RunResult.export_bpm`` wrote as bonds from fiber to
    fiber: where the network found binder or a bond point holding two fibers together, the model holds them
    together too.

    ``bonds`` are as :func:`find_fibers` gives them (``fit.bonds``): the two fibers' indices and a position in
    voxels of ``voxel_size``, measured from the corner of the file's box. Each pair of fibers gets one bond of
    ``bond_type`` (default: the next free bond type), at its first bond in the list, between the two fibers'
    particles nearest that position (nearest the axis, for capsules). The fibers keep their order through
    ``coarse_grained``, ``relax`` and ``export_bpm``, so fiber ``k`` is the file's molecule ``k + 1``. Returns how
    many bonds were added.
    """
    data = _DataFile(path)
    particles = _Particles.read(data)
    low, _ = data.box()
    by_molecule = np.argsort(particles.molecules, kind="stable")
    sorted_molecules = particles.molecules[by_molecule]

    def nearest(molecule: int, point: np.ndarray) -> int | None:
        """The id of the molecule's particle nearest ``point``, or None if it has none."""
        start, stop = np.searchsorted(sorted_molecules, [molecule, molecule + 1])
        mine = by_molecule[start:stop]
        return int(particles.ids[mine[np.argmin(particles.distance(mine, point))]]) if len(mine) else None

    pairs, new = set(), []
    for bond in bonds:
        i, j = sorted(int(fiber) for fiber in bond["fibers"])
        if i == j or (i, j) in pairs:
            continue
        point = low + np.asarray(bond["position"], dtype=float) * voxel_size
        a, b = nearest(i + 1, point), nearest(j + 1, point)
        if a is None or b is None:  # a fiber the file doesn't have
            continue
        pairs.add((i, j))
        new.append((a, b))
    if not new:
        return 0
    existing = data.rows("Bonds")
    kind = bond_type or _next_type(data, "bond types", (row[1] for row in existing))
    first = max((int(row[0]) for row in existing), default=0) + 1
    data.add_rows("Bonds", [f"{first + k} {kind} {a} {b}" for k, (a, b) in enumerate(new)])
    data.set_count("bonds", len(existing) + len(new))
    data.set_count("bond types", max(data.count("bond types"), kind))
    data.note(f"bond type {kind} = binder: fiber to fiber where the CT network found them bonded ({len(new)} bonds)")
    data.write()
    return len(new)


@dataclass(frozen=True)
class BinderSpheres:
    """What :func:`add_bpm_binder_spheres` added: ``spheres`` binder spheres, ``fiber_bonds`` bonds from a sphere
    to a fiber particle and ``sphere_bonds`` between spheres; ``full_size`` of the spheres have the full
    ``diameter`` (meters), the rest were made smaller so they don't reach into a fiber their binder doesn't touch."""

    spheres: int
    fiber_bonds: int
    sphere_bonds: int
    full_size: int
    diameter: float


def add_bpm_binder_spheres(path: str | os.PathLike, binder, voxel_size: float, diameter: float | None = None, *,
                           density: float | None = None, atom_type: int | None = None,
                           bond_types: tuple[int, int] | None = None) -> BinderSpheres:
    """Put the binder into a bonded-particle (DEM) file that ``RunResult.export_bpm`` wrote as spheres of its own,
    bonded to the fibers and to each other.

    ``binder`` is a boolean ``(z, y, x)`` mask over the file's box in voxels of ``voxel_size``, such as the binder
    :func:`find_fibers` finds (``fit.binder``). Binder inside a fiber particle is left out (the fibers win).

    - Where: spheres of ``diameter`` (default: the median diameter of the file's particles, so about a fiber across)
      go into the binder deepest first, each at least a diameter from the others, so they don't overlap. Every
      binder voxel then belongs to its nearest sphere: the binder that sphere stands for.
    - Size: a sphere is made smaller where it would otherwise reach into a fiber its binder doesn't touch, so the
      model starts with no overlap that no bond holds.
    - Mass: each sphere has the mass of its binder at ``density`` (default: the median density of the file's
      particles). LAMMPS takes mass as a density, so each sphere's density is ``density`` times its binder's volume
      over its own: it varies from sphere to sphere.
    - Bonds: each sphere is bonded (``bond_types[0]``) to every particle it overlaps of each fiber its binder touches
      (within a voxel), or to that fiber's nearest particle if it overlaps none; and (``bond_types[1]``) to each
      sphere whose binder meets its own.

    Each sphere is a molecule of its own, of ``atom_type``, with no ``Capsules`` row in a capsule file (so it stays
    a sphere); the types default to the next free ones. Returns what was added, as :class:`BinderSpheres`. Binder
    thinner than the spheres (a film coating a fiber, say) still gets spheres, which then reach past it; give a
    smaller ``diameter`` to follow thin binder more closely.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    data = _DataFile(path)
    particles = _Particles.read(data)
    mask = np.asarray(binder, dtype=bool)
    low, high = data.box()
    if not np.allclose(high - low, np.array(mask.shape[::-1]) * voxel_size, rtol=1e-6):
        raise ValueError(f"binder covers {mask.shape[::-1]} voxels (x, y, z) of {voxel_size} m, but {data.path} "
                         f"spans {(high - low).tolist()} m")
    size = (diameter if diameter is not None else 2.0 * float(np.median(particles.radii))) / voxel_size
    # Below, lengths are in voxels and positions are measured from the box's corner: voxel (i, j, k) of the mask
    # spans x i..i+1, y j..j+1, z k..k+1.
    fibers = particles.in_voxels(low, voxel_size)

    # 1. The binder inside each fiber particle, and the binder within a voxel of its surface (touching it).
    covered = np.zeros(mask.shape, dtype=bool)
    touch_voxel, touch_particle = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    upper = np.array(mask.shape[::-1])
    for k in range(len(fibers.ids)):
        ends = fibers.centers[k] + np.outer([-fibers.half[k], fibers.half[k]], fibers.axes[k])
        start = np.clip(np.floor(ends.min(0) - fibers.radii[k] - 1.0).astype(int), 0, upper)
        stop = np.clip(np.ceil(ends.max(0) + fibers.radii[k] + 1.0).astype(int) + 1, 0, upper)
        z, y, x = np.nonzero(mask[start[2]:stop[2], start[1]:stop[1], start[0]:stop[0]])
        if not len(z):
            continue
        z, y, x = z + start[2], y + start[1], x + start[0]
        gap = _segment_distance(np.stack([x, y, z], 1) + 0.5, fibers.centers[k], fibers.axes[k],
                                fibers.half[k]) - fibers.radii[k]
        flat = np.ravel_multi_index((z, y, x), mask.shape)
        covered.flat[flat[gap <= 0.0]] = True
        touch_voxel.append(flat[gap <= 1.0])
        touch_particle.append(np.full(int((gap <= 1.0).sum()), k))
    free = mask & ~covered
    if not free.any():
        return BinderSpheres(0, 0, 0, 0, size * voxel_size)

    # 2. The spheres: deepest binder first, each at least a diameter from the ones placed before it.
    depth = ndimage.distance_transform_edt(free)
    candidates = np.flatnonzero(free)
    candidates = candidates[np.argsort(-depth.ravel()[candidates], kind="stable")]
    reach = int(np.ceil(size))
    offsets = np.argwhere(np.ones((2 * reach + 1,) * 3, dtype=bool)) - reach
    offsets = offsets[(offsets ** 2).sum(1) < size ** 2]
    blocked = np.zeros(mask.shape, dtype=bool)
    blocked_flat = blocked.ravel()  # a view: what is blocked below shows up here
    shape = np.array(mask.shape)
    chosen = []
    for index in candidates:
        if blocked_flat[index]:
            continue
        chosen.append(index)
        cells = np.array(np.unravel_index(index, mask.shape)) + offsets
        cells = cells[np.all((cells >= 0) & (cells < shape), axis=1)]
        blocked[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    z, y, x = np.unravel_index(np.array(chosen), mask.shape)
    at = np.stack([x, y, z], axis=1) + 0.5  # sphere centers
    n = len(at)

    # 3. Each binder voxel belongs to its nearest sphere.
    voxels = np.flatnonzero(free)
    z, y, x = np.unravel_index(voxels, mask.shape)
    owner = cKDTree(at).query(np.stack([x, y, z], axis=1) + 0.5)[1]
    label = np.full(mask.shape, -1, dtype=np.int32)
    label.ravel()[voxels] = owner
    binder_voxels = np.bincount(owner, minlength=n)

    # 4. Spheres whose binder meets: neighbouring voxels (sharing a face) that belong to two spheres.
    meets = []
    for axis in range(3):
        a = np.moveaxis(label, axis, 0)[:-1].ravel()
        b = np.moveaxis(label, axis, 0)[1:].ravel()
        meet = (a >= 0) & (b >= 0) & (a != b)
        meets.append(np.stack([np.minimum(a[meet], b[meet]), np.maximum(a[meet], b[meet])], axis=1))
    sphere_pairs = np.unique(np.concatenate(meets), axis=0)

    # 5. The fiber particles each sphere's binder touches.
    voxel, particle = np.concatenate(touch_voxel), np.concatenate(touch_particle)
    sphere = label.ravel()[voxel]
    touched = np.unique(np.stack([sphere[sphere >= 0], particle[sphere >= 0]], axis=1), axis=0)
    touched_by = np.split(touched[:, 1], np.searchsorted(touched[:, 0], np.arange(1, n)))

    # 6. Each sphere's size and its bonds to fibers. A fiber is in contact if the sphere's binder touches it, or if
    #    its surface comes within half a voxel of the sphere's center; the sphere shrinks to clear the others.
    nearby = cKDTree(fibers.centers).query_ball_point(at, 0.5 * size + float(np.max(fibers.radii + fibers.half)) + 1.0)
    sphere_radius = np.full(n, 0.5 * size)
    fiber_bonds = []
    for i in range(n):
        near = np.union1d(np.asarray(nearby[i], dtype=np.int64), touched_by[i])
        if not len(near):
            continue
        gap = fibers.distance(near, at[i]) - fibers.radii[near]
        molecule = fibers.molecules[near]
        contact = np.union1d(fibers.molecules[touched_by[i]], molecule[gap < 0.5])
        in_contact = np.isin(molecule, contact)
        if (~in_contact).any():  # a hair inside the nearest other fiber's surface, so they don't overlap
            sphere_radius[i] = min(0.5 * size, float(gap[~in_contact].min()) - 1e-6)
        overlap = in_contact & (gap < sphere_radius[i])
        bonded = list(near[overlap])
        for fiber in np.setdiff1d(contact, molecule[overlap]):  # touched, not overlapped: its nearest particle
            mine = molecule == fiber
            bonded.append(near[mine][np.argmin(gap[mine])])
        fiber_bonds += [(i, int(k)) for k in bonded]

    # 7. Write the spheres and their bonds.
    rho = density if density is not None else float(np.median(particles.densities))
    sphere_volume = 4.0 / 3.0 * np.pi * (sphere_radius * voxel_size) ** 3
    sphere_density = rho * binder_voxels * voxel_size ** 3 / sphere_volume
    kind = atom_type or _next_type(data, "atom types", particles.types)
    existing = data.rows("Bonds")
    used = [row[1] for row in existing]
    to_fiber, between = bond_types or (_next_type(data, "bond types", used),
                                       _next_type(data, "bond types", used) + 1)
    first_id, first_molecule = int(particles.ids.max()) + 1, int(particles.molecules.max()) + 1
    ids = first_id + np.arange(n)
    position = low + at * voxel_size
    data.add_rows("Atoms", [f"{ids[i]} {first_molecule + i} {kind} {2.0 * sphere_radius[i] * voxel_size:.17e} "
                            f"{sphere_density[i]:.17e} {position[i, 0]:.17e} {position[i, 1]:.17e} "
                            f"{position[i, 2]:.17e}" for i in range(n)])
    # No Capsules rows: an atom without one is a sphere (DIRT turns down a capsule of no length).
    first_bond = max((int(row[0]) for row in existing), default=0) + 1
    bond_rows = [f"{ids[i]} {particles.ids[k]}" for i, k in fiber_bonds]
    bond_rows = [f"{to_fiber} {row}" for row in bond_rows] + [f"{between} {ids[a]} {ids[b]}" for a, b in sphere_pairs]
    data.add_rows("Bonds", [f"{first_bond + k} {row}" for k, row in enumerate(bond_rows)])
    data.set_count("atoms", len(particles.ids) + n)
    data.set_count("bonds", len(existing) + len(bond_rows))
    data.set_count("atom types", max(data.count("atom types"), kind))
    data.set_count("bond types", max(data.count("bond types"), to_fiber, between))
    data.note(f"atom type {kind} = binder spheres ({n}, at most {size * voxel_size:.6g} m across; each one's density "
              f"gives it the mass of its binder at {rho:.6g})")
    data.note(f"bond type {to_fiber} = binder sphere to fiber particle ({len(fiber_bonds)} bonds)")
    data.note(f"bond type {between} = binder sphere to binder sphere ({len(sphere_pairs)} bonds)")
    data.write()
    full = int(np.sum(sphere_radius >= 0.5 * size))
    return BinderSpheres(n, len(fiber_bonds), len(sphere_pairs), full, size * voxel_size)


def _z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternions (i, j, k, w) turning the z axis onto each unit row of ``direction`` (none for a zero row)."""
    q = np.stack([-direction[:, 1], direction[:, 0], np.zeros(len(direction)), 1.0 + direction[:, 2]], axis=1)
    q[direction[:, 2] < -1.0 + 1e-12] = [1.0, 0.0, 0.0, 0.0]
    q[np.linalg.norm(direction, axis=1) < 1e-12] = [0.0, 0.0, 0.0, 1.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def write_bpm_ovito(path: str | os.PathLike, dump: str | os.PathLike | None = None) -> Path:
    """Write a bonded-particle file as a LAMMPS dump for OVITO (``dump``, default: ``path`` with ``.dump``), in
    micrometers: the fibers' particles as particle type 1, binder spheres as type 2, and each bond between two
    fibers, or between a binder sphere and anything, as type 3, a thin capsule from one particle's center to the
    other's (bonds along a fiber lie inside it and are left out). The ``bpm_type`` column is each particle's atom
    type, or each bond's bond type, in the file. Load the dump in OVITO and set the particle shape to Spherocylinder.
    Returns the dump's path.
    """
    path = Path(path)
    dump = Path(dump) if dump is not None else path.with_suffix(".dump")
    data = _DataFile(path)
    particles = _Particles.read(data)
    fiber_types = {int(line.split()[3]) for line in data.header if line.startswith("# atom type") and
                   "TANGLE material" in line and line.split()[3].isdigit()}
    is_fiber = np.isin(particles.types, list(fiber_types)) if fiber_types else np.ones(len(particles.ids), bool)
    um = 1e6
    n = len(particles.ids)
    table = np.column_stack([particles.molecules, np.where(is_fiber, 1, 2), particles.radii * um, particles.radii * um,
                             2.0 * particles.half * um, _z_to(particles.axes), particles.centers * um, particles.types])
    index_of = {atom: index for index, atom in enumerate(particles.ids.tolist())}
    sticks = []
    for row in data.rows("Bonds"):
        a, b = index_of[int(row[2])], index_of[int(row[3])]
        if particles.molecules[a] != particles.molecules[b]:
            sticks.append((a, b, int(row[1])))
    if sticks:
        a, b, kind = (np.array(column) for column in zip(*sticks))
        span = particles.centers[b] - particles.centers[a]
        length = np.linalg.norm(span, axis=1)
        unit = span / np.maximum(length, 1e-30)[:, None]
        r = 0.2 * float(np.median(particles.radii[is_fiber] if is_fiber.any() else particles.radii)) * um
        m = len(sticks)
        table = np.vstack([table, np.column_stack([np.zeros(m), np.full(m, 3), np.full(m, r), np.full(m, r),
                                                   length * um, _z_to(unit),
                                                   0.5 * (particles.centers[a] + particles.centers[b]) * um, kind])])
    low, high = data.box()
    with dump.open("w") as handle:
        handle.write(f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(table)}\nITEM: BOX BOUNDS ff ff ff\n")
        for axis in range(3):
            handle.write(f"{low[axis] * um:.6g} {high[axis] * um:.6g}\n")
        handle.write("ITEM: ATOMS id mol type AsphericalShape.X AsphericalShape.Y AsphericalShape.Z "
                     "quati quatj quatk quatw x y z bpm_type\n")
        ids = np.concatenate([particles.ids, particles.ids.max() + 1 + np.arange(len(table) - n)])
        np.savetxt(handle, np.column_stack([ids, table]),
                   fmt=["%d", "%d", "%d"] + ["%.6g"] * 3 + ["%.6f"] * 4 + ["%.6f"] * 3 + ["%d"])
    return dump
