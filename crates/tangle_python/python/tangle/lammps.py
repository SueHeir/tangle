"""LAMMPS bonded-particle (BPM) data files in Tangle: read the fibers, relax them, write the file back.

    >>> from tangle import lammps
    >>> data = lammps.read_data("fibers.data", periodic="p p p")  # the input script's boundary
    >>> run = data.relax()                                         # Tangle pushes the fibers apart (GPU)
    >>> data.write_data("fibers_relaxed.data", run)                # the same file, with the relaxed positions

A fiber is a chain of spheres in one molecule, each bonded to the next, as LAMMPS's ``bpm/sphere`` atom style holds
them (``atom-ID molecule-ID atom-type diameter density x y z``). The sphere centers in chain order are the fiber's
centerline and its rest shape, so the relaxation pushes the fibers apart while keeping each one's length and bends;
the fiber is as wide as its largest sphere. ``write_data`` copies the file and changes only the positions (and image
flags) of the spheres that moved: IDs, types, bonds, coefficients and every other section stay as they were.
LAMMPS's BPM bond styles take each bond's reference length and orientation from the positions when a run starts,
not from the data file, so the relaxed model starts unstressed.

Atoms in no chain (binder spheres, grains) stay where they are, and Tangle doesn't see them, so a fiber can be pushed
into one. Bonds between fibers are kept, but don't hold the fibers together while they relax. ``LammpsData.notes``
lists what was left alone. See ``docs/lammps_bpm.md``.
"""

from __future__ import annotations

import bisect
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ._tangle import Cell, FiberCollection, Material, Recipe

__all__ = ["LammpsData", "read_data"]

# The Atoms columns of the atom styles read here, as LAMMPS's read_data lists them. A row may end with three image
# flags.
_STYLES = {
    "atomic": ("id", "type", "x", "y", "z"),
    "charge": ("id", "type", "q", "x", "y", "z"),
    "bond": ("id", "molecule", "type", "x", "y", "z"),
    "angle": ("id", "molecule", "type", "x", "y", "z"),
    "molecular": ("id", "molecule", "type", "x", "y", "z"),
    "full": ("id", "molecule", "type", "q", "x", "y", "z"),
    "sphere": ("id", "type", "diameter", "density", "x", "y", "z"),
    "bpm/sphere": ("id", "molecule", "type", "diameter", "density", "x", "y", "z"),
}
# LAMMPS's other atom styles, so that an "Atoms # ellipsoid" comment is taken as a style this module doesn't read.
_OTHER_STYLES = {
    "amoeba", "bacillus", "body", "dielectric", "dipole", "dpd", "edpd", "electron", "ellipsoid", "line", "mdpd",
    "oxdna", "peri", "rheo", "rheo/thermal", "smd", "sph", "spin", "tdpd", "template", "tri", "wavepacket",
}
# In a file without bonds, a sphere farther than this many diameters from the next one ends its fiber.
_UNBONDED_GAP = 1.5


class _Atom:
    """One Atoms row: its line, molecule (None without a molecule column), type (as written: a number or a label),
    diameter (None where the file and ``diameter=`` give none), position and image flags (None if the row has
    none)."""

    __slots__ = ("row", "molecule", "type", "diameter", "position", "image")


class LammpsData:
    """The fibers of a LAMMPS data file, from :func:`read_data`, to relax and write back.

    - ``cell``: the ``tangle.Cell`` the fibers relax in: the file's box, periodic where ``read_data`` was told. On
      the other axes it is widened where a sphere sticks out of the box.
    - ``fibers``: a ``tangle.FiberCollection``, one fiber per chain of spheres, by molecule ID and then atom ID. A
      fiber's material is named after its spheres' atom type (``"type 1"``).
    - ``fiber_atoms``: each fiber's atom IDs in chain order: point ``i`` of fiber ``k`` is atom ``fiber_atoms[k][i]``.
    - ``other_atoms``: the IDs of the atoms in no fiber, which stay where they are.
    - ``notes``: what was left alone, and why.
    """

    def __init__(self, path: Path, atom_style: str, cell: Any, fibers: Any, fiber_atoms: list[list[int]],
                 other_atoms: list[int], notes: list[str], *, lines: list[str], newline: str, final_newline: bool,
                 atoms: dict[int, _Atom], columns: tuple[str, ...], points: list[list[list[float]]],
                 low: list[float], lengths: list[float], periodic: list[bool]):
        self.path = path
        self.atom_style = atom_style
        self.cell = cell
        self.fibers = fibers
        self.fiber_atoms = fiber_atoms
        self.other_atoms = other_atoms
        self.notes = notes
        self._lines = lines
        self._newline = newline
        self._final_newline = final_newline
        self._atoms = atoms
        self._x = columns.index("x")
        self._width = len(columns)
        self._points = points
        self._low = low
        self._lengths = lengths
        self._periodic = periodic

    def __repr__(self) -> str:
        return (f"LammpsData({str(self.path)!r}, atom_style={self.atom_style!r}, fibers={len(self.fiber_atoms)}, "
                f"fiber_atoms={sum(map(len, self.fiber_atoms))}, other_atoms={len(self.other_atoms)})")

    def relax(self, settings: Any | None = None) -> Any:
        """Relax the fibers in Tangle and return its ``RunResult``: the fibers pushed apart until none overlap, each
        keeping its length and bends as closely as it can.

        ``settings`` is a ``tangle.RelaxationSettings``; the default runs on the GPU, allows overlaps of 1% of the
        thinnest fiber's diameter (``penetration_tolerance``) and stops after ``max_iterations=2000``. Check
        ``run.converged`` and ``run.max_penetration`` before writing. With ``adaptive_segmentation`` left off (the
        default), every sphere goes exactly where its point on the fiber went; with it on, :meth:`write_data` places
        each sphere at its share of the relaxed fiber's length.
        """
        if not self.fiber_atoms:
            why = " ".join(self.notes) or "it has no chains of bonded spheres."
            raise ValueError(f"{self.path} has no fibers to relax: {why}")
        recipe = Recipe(self.cell)
        recipe.insert(self.fibers, name="LAMMPS fibers")
        return recipe.run(settings)

    def write_data(self, path: str | os.PathLike, relaxed: Any) -> int:
        """Write the data file again with every fiber sphere moved to its place on the relaxed fiber, and return how
        many atoms moved.

        ``relaxed`` is the ``RunResult`` from :meth:`relax` (or anything with these fibers, in this order, and a
        ``centerlines()`` method), or the relaxed centerlines themselves. Every line stays as it was except the
        Atoms rows of spheres that moved, where only x, y, z and the image flags change: on a periodic axis each
        sphere is put back inside the box, and its image flag, where the row has them, counts the box lengths it
        moved, so the fibers stay whole. A sphere that moved less than a millionth of the box keeps its row.
        """
        centerlines = relaxed.centerlines() if hasattr(relaxed, "centerlines") else relaxed
        centerlines = [[[float(value) for value in point] for point in line] for line in centerlines]
        if len(centerlines) != len(self.fiber_atoms):
            raise ValueError(f"{len(centerlines)} relaxed fibers, but {self.path} has {len(self.fiber_atoms)}")
        lines = list(self._lines)
        moved = 0
        for atom_ids, before, after in zip(self.fiber_atoms, self._points, centerlines):
            if len(after) != len(atom_ids):  # adaptive segmentation changed the points: by share of the length
                after = _along(after, _fractions(before))
            for atom_id, old, new in zip(atom_ids, before, after):
                if all(abs(n - o) <= 1e-6 * max(abs(o), length) for n, o, length in zip(new, old, self._lengths)):
                    continue
                lines[self._atoms[atom_id].row] = self._moved_row(atom_id, new)
                moved += 1
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            handle.write(self._newline.join(lines) + (self._newline if self._final_newline else ""))
        return moved

    def _moved_row(self, atom_id: int, point: Sequence[float]) -> str:
        """The atom's row with ``point`` (unwrapped) in its x, y and z, and its image flags to match."""
        atom = self._atoms[atom_id]
        data, mark, comment = self._lines[atom.row].partition("#")
        indent = data[: len(data) - len(data.lstrip())]
        words = data.split()
        for axis in range(3):
            value = point[axis]
            if self._periodic[axis]:
                image = math.floor((value - self._low[axis]) / self._lengths[axis])
                value -= image * self._lengths[axis]
                if atom.image is not None:
                    words[self._width + axis] = str(image)
            words[self._x + axis] = repr(value)
        return indent + " ".join(words) + (" " + mark + comment if mark else "")


def read_data(path: str | os.PathLike, *, periodic: str | bool | Sequence[bool] = "xyz",
              atom_style: str | None = None, diameter: float | Mapping[Any, float] | None = None,
              fiber_bond_types: Iterable[Any] | None = None) -> LammpsData:
    """Read the fibers of a LAMMPS data file, to relax them in Tangle and write the file back.

    - ``periodic``: the periodic axes, as letters (``"xy"``) or as the input script's ``boundary`` (``"p p f"``).
      The data file doesn't say; the default, ``"xyz"``, is LAMMPS's default ``boundary p p p``. On the other axes
      Tangle's walls keep the fibers' surfaces inside the box, or no farther out than a sphere already sticks out.
    - ``atom_style``: the style of the Atoms rows (default: the ``Atoms # style`` comment, as LAMMPS's ``write_data``
      writes it, else ``"bpm/sphere"``). Also ``sphere``, ``bond``, ``angle``, ``molecular``, ``full``, ``charge``,
      ``atomic``, and hybrids of them such as ``"hybrid sphere bond"``. Rows may end with image flags.
    - ``diameter``: the spheres' diameter (one value, or ``{atom type: value}``) in place of the file's; needed for
      a style without a diameter column, such as ``bond``.
    - ``fiber_bond_types``: the bond types that run along the fibers (default: all). Needed only when bonds between
      fibers join atoms of one molecule, as when every atom is molecule 0.

    A fiber is a chain of spheres of one molecule (or, without molecule IDs, of spheres joined by bonds), each bonded
    to the next: in bond order, or in atom-ID order when each sphere is bonded to the next ID even if also to others
    (as ``create_bonds many`` bonds them). Atoms in no such chain stay where they are. In a file with no bonds at all
    (bonds made in the input script), each run of consecutive atom IDs in one molecule whose spheres are within
    1.5 diameters of the next is a fiber.
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as handle:
        text = handle.read()
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    final_newline = lines[-1] == ""
    if final_newline:
        lines.pop()

    # The header runs from the title line to the first section's keyword, the first line starting with a letter.
    first = next((index for index in range(1, len(lines)) if _data(lines[index]).strip()[:1].isalpha()), len(lines))
    counts, low, high = _header(lines[1:first], path)
    lengths = [high[axis] - low[axis] for axis in range(3)]
    if any(not length > 0.0 for length in lengths):
        raise ValueError(f"{path}: the box has a side of length {min(lengths)}")
    if _find(lines, "Capsules", first) is not None or _find(lines, "Bacilli", first) is not None:
        raise ValueError(f"{path} holds capsules (Tangle's spherocylinder export); tangle.lammps reads sphere files")

    keyword = _find(lines, "Atoms", first)
    if keyword is None:
        raise ValueError(f"{path} has no Atoms section")
    if atom_style is None:
        named = lines[keyword].partition("#")[2].split()
        known = named[:1] and (named[0] in _STYLES or named[0] in _OTHER_STYLES or named[0] == "hybrid")
        atom_style = " ".join(named) if known else "bpm/sphere"
    columns = _columns(atom_style)
    atoms = _read_atoms(lines, _rows(lines, keyword, counts.get("atoms", 0), "Atoms", path), columns, atom_style,
                        _diameters(diameter), path)

    bonds = []
    keyword = _find(lines, "Bonds", first)
    if keyword is None and counts.get("bonds", 0):
        raise ValueError(f"{path} has {counts['bonds']} bonds but no Bonds section")
    bond_rows = [] if keyword is None else _rows(lines, keyword, counts.get("bonds", 0), "Bonds", path)
    for row in bond_rows:
        words = _data(lines[row]).split()
        if len(words) < 4:
            raise ValueError(f"{path}, line {row + 1}: a Bonds row is bond-ID bond-type atom1 atom2")
        ends = (_int(words[2], path, row), _int(words[3], path, row))
        for atom_id in ends:
            if atom_id not in atoms:
                raise ValueError(f"{path}, line {row + 1}: a bond to atom {atom_id}, which isn't in Atoms")
        bonds.append((words[1], *ends))

    flags = _periodic(periodic)
    along = None if fiber_bond_types is None else {str(kind) for kind in fiber_bond_types}
    chains, left, notes = _chains(atoms, bonds, along, "molecule" in columns, lengths, flags, path)
    fiber_of = {atom_id: index for index, chain in enumerate(chains) for atom_id in chain}

    fibers = FiberCollection("LAMMPS fibers")
    materials: dict[tuple[str, float], Any] = {}
    points, widths, mixed = [], [], 0
    for chain in chains:
        line = _unwrap([atoms[atom_id] for atom_id in chain], chain, lengths, flags, path)
        sizes = [atoms[atom_id].diameter for atom_id in chain]
        for atom_id, size in zip(chain, sizes):
            if size is None or not math.isfinite(size) or size <= 0.0:
                raise ValueError(f"{path}: atom {atom_id} has no diameter"
                                 + (f" (atom style {atom_style!r} has no diameter column)"
                                    if "diameter" not in columns else f" ({size})")
                                 + "; pass diameter= (one value, or one per atom type)")
        width = max(sizes)
        mixed += max(sizes) - min(sizes) > 1e-9 * width
        kind = Counter(atoms[atom_id].type for atom_id in chain).most_common(1)[0][0]
        if (kind, width) not in materials:
            materials[kind, width] = Material(f"type {kind}", width)
        molecule = atoms[chain[0]].molecule
        tags = {"source": "LAMMPS"} if molecule is None else {"source": "LAMMPS", "molecule": str(molecule)}
        fibers.add_fiber(line, materials[kind, width], tags=tags)
        points.append(line)
        widths.append(width)

    # Tangle's walls keep each fiber's surface inside the cell. On an axis that isn't periodic the cell is the box,
    # widened where a sphere sticks out of it, so that no sphere moves just because it is near a face.
    cell_low, cell_high = list(low), list(high)
    for line, width in zip(points, widths):
        for axis in range(3):
            if not flags[axis]:
                cell_low[axis] = min(cell_low[axis], min(point[axis] for point in line) - 0.5 * width)
                cell_high[axis] = max(cell_high[axis], max(point[axis] for point in line) + 0.5 * width)
    cell = Cell([cell_high[axis] - cell_low[axis] for axis in range(3)], flags, cell_low)

    between = to_other = 0
    for _, a, b in bonds:
        fa, fb = fiber_of.get(a), fiber_of.get(b)
        if fa is not None and fb is not None and fa != fb:
            between += 1
        elif (fa is None) != (fb is None):
            to_other += 1
    if left:
        reasons = ", ".join(f"{_count(count, 'atom')} {reason}" for reason, count in left.most_common())
        notes.append(f"Left where they are, and not seen by the fibers: {reasons}.")
    if between:
        notes.append(f"Kept, but not holding the fibers together while they relax: {_count(between, 'bond')} "
                     "between fibers.")
    if to_other:
        notes.append(f"Kept: {_count(to_other, 'bond')} from a fiber to an atom left where it is.")
    if mixed:
        notes.append(f"As wide as their largest sphere: {_count(mixed, 'fiber')} with spheres of different sizes.")
    return LammpsData(path, atom_style, cell, fibers, chains,
                      sorted(atom_id for atom_id in atoms if atom_id not in fiber_of), notes, lines=lines,
                      newline=newline, final_newline=final_newline, atoms=atoms, columns=columns, points=points,
                      low=low, lengths=lengths, periodic=flags)


def _data(line: str) -> str:
    """A line without its trailing comment."""
    return line.split("#", 1)[0]


def _is_number(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


def _int(word: str, path: Path, row: int) -> int:
    try:
        return int(word)
    except ValueError:
        raise ValueError(f"{path}, line {row + 1}: {word!r} isn't an integer ID") from None


def _columns(style: str) -> tuple[str, ...]:
    """The Atoms columns of ``style``. A hybrid's rows hold id, type, x, y and z, then each sub-style's other values
    in order, each value once."""
    words = [word for word in style.split() if not _is_number(word)]  # "sphere 1": a style's own arguments
    if not words:
        raise ValueError("the atom style is empty")
    if words[0] == "hybrid":
        if len(words) < 2:
            raise ValueError("a hybrid atom style needs sub-styles, as in 'hybrid sphere bond'")
        columns = ["id", "type", "x", "y", "z"]
        for sub in words[1:]:
            columns += [name for name in _style(sub) if name not in columns]
        return tuple(columns)
    if len(words) > 1:
        raise ValueError(f"atom style {style!r} names more than one style without 'hybrid'")
    return _style(words[0])


def _style(name: str) -> tuple[str, ...]:
    if name not in _STYLES:
        raise ValueError(f"atom style {name!r} isn't one tangle.lammps reads; it reads {', '.join(_STYLES)} and "
                         "hybrids of them")
    return _STYLES[name]


def _periodic(value: Any) -> list[bool]:
    """``periodic`` as three flags: axis letters (``"xy"``), the input script's boundary (``"p p f"``), one bool, or
    three."""
    if value is None:
        return [False] * 3
    if isinstance(value, str):
        words = value.lower().split()
        if words[:1] == ["boundary"]:
            words = words[1:]
        if len(words) == 3 and all(set(word) <= set("pfsm") for word in words):
            return [word == "p" for word in words]
        letters = "".join(words)
        if set(letters) <= set("xyz"):
            return [axis in letters for axis in "xyz"]
        raise ValueError(f"periodic={value!r}: give the periodic axes ('xy') or the input script's boundary "
                         "('p p f')")
    if isinstance(value, bool):
        return [value] * 3
    flags = [bool(flag) for flag in value]
    if len(flags) != 3:
        raise ValueError(f"periodic needs three flags, got {len(flags)}")
    return flags


def _diameters(diameter: Any) -> Any:
    """``diameter=`` as a function of the atom type: a value for it, or None."""
    if diameter is None:
        return None
    if isinstance(diameter, Mapping):
        table = {str(kind): float(value) for kind, value in diameter.items()}
        return table.get
    value = float(diameter)
    return lambda kind: value


def _header(lines: list[str], path: Path) -> tuple[dict[str, int], list[float], list[float]]:
    """The header's counts (``{"atoms": 10, "bond types": 2, ...}``) and the box's low and high corners."""
    counts: dict[str, int] = {}
    low: list[float | None] = [None] * 3
    high: list[float | None] = [None] * 3
    for line in lines:
        words = _data(line).split()
        if not words:
            continue
        if len(words) == 4 and words[2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            axis = "xyz".index(words[2][0])
            low[axis], high[axis] = float(words[0]), float(words[1])
        elif words[3:] == ["xy", "xz", "yz"]:
            if any(float(word) != 0.0 for word in words[:3]):
                raise ValueError(f"{path} has a triclinic box (xy xz yz); Tangle's cells are orthorhombic")
        elif words[-1] in ("avec", "bvec", "cvec") or words[-2:] == ["abc", "origin"]:
            raise ValueError(f"{path} has a general triclinic box; Tangle's cells are orthorhombic")
        elif words[0].isdigit():
            counts[" ".join(words[1:])] = int(words[0])
    for axis, name in enumerate("xyz"):
        if low[axis] is None:
            raise ValueError(f"{path} has no '{name}lo {name}hi' line")
    return counts, low, high  # type: ignore[return-value]


def _find(lines: list[str], name: str, start: int) -> int | None:
    """The line of section ``name``'s keyword, or None."""
    return next((index for index in range(start, len(lines)) if _data(lines[index]).strip() == name), None)


def _rows(lines: list[str], keyword: int, count: int, name: str, path: Path) -> list[int]:
    """The lines of the ``count`` rows after the section keyword on line ``keyword``."""
    rows = []
    for index in range(keyword + 1, len(lines)):
        if len(rows) == count:
            break
        if _data(lines[index]).strip():
            rows.append(index)
    if len(rows) < count:
        raise ValueError(f"{path}: the header gives {count} {name.lower()}, but the {name} section has {len(rows)}")
    return rows


def _read_atoms(lines: list[str], rows: list[int], columns: tuple[str, ...], style: str, diameters: Any,
                path: Path) -> dict[int, _Atom]:
    width, x = len(columns), columns.index("x")
    type_column = columns.index("type")
    molecule = columns.index("molecule") if "molecule" in columns else None
    size = columns.index("diameter") if "diameter" in columns else None
    atoms: dict[int, _Atom] = {}
    for row in rows:
        words = _data(lines[row]).split()
        if len(words) not in (width, width + 3):
            raise ValueError(f"{path}, line {row + 1}: {len(words)} values, but atom style {style!r} has {width} "
                             f"({' '.join(columns)}), or {width + 3} with image flags; is atom_style= right?")
        atom = _Atom()
        atom.row = row
        atom.molecule = _int(words[molecule], path, row) if molecule is not None else None
        atom.type = words[type_column]
        atom.position = [float(word) for word in words[x : x + 3]]
        atom.image = [_int(word, path, row) for word in words[width:]] if len(words) > width else None
        given = diameters(atom.type) if diameters is not None else None
        atom.diameter = given if given is not None else (float(words[size]) if size is not None else None)
        atom_id = _int(words[0], path, row)
        if atom_id in atoms:
            raise ValueError(f"{path}, line {row + 1}: atom {atom_id} is in Atoms twice")
        atoms[atom_id] = atom
    return atoms


def _chains(atoms: dict[int, _Atom], bonds: list[tuple[str, int, int]], along: set[str] | None, molecules: bool,
            lengths: list[float], periodic: list[bool], path: Path) -> tuple[list[list[int]], Counter, list[str]]:
    """The fibers' atom IDs in chain order, the atoms in no fiber counted by why, and notes on how they were found."""
    groups: dict[Any, list[int]] = defaultdict(list)
    for atom_id in sorted(atoms):
        groups[atoms[atom_id].molecule].append(atom_id)
    order = sorted(groups, key=lambda molecule: -math.inf if molecule is None else molecule)
    chains: list[list[int]] = []
    left: Counter = Counter()
    notes: list[str] = []

    if not bonds:  # bonds made in the input script: runs of consecutive IDs whose spheres are close
        notes.append("The file has no bonds, so each fiber is a run of consecutive atom IDs"
                     + (" in one molecule" if molecules else "")
                     + f" whose spheres are within {_UNBONDED_GAP:g} diameters of the next.")
        for molecule in order:
            run: list[int] = []
            for atom_id in groups[molecule]:
                if run and _too_far(atoms[run[-1]], atoms[atom_id], lengths, periodic, path):
                    _close(run, chains, left)
                    run = []
                run.append(atom_id)
            _close(run, chains, left)
        return chains, left, notes

    links: dict[int, set[int]] = defaultdict(set)
    for kind, a, b in bonds:
        if a != b and (along is None or kind in along) and atoms[a].molecule == atoms[b].molecule:
            links[a].add(b)
            links[b].add(a)
    seen: set[int] = set()
    for molecule in order:
        for start in groups[molecule]:
            if start in seen:
                continue
            component, queue = {start}, [start]
            while queue:
                for other in links[queue.pop()]:
                    if other not in component:
                        component.add(other)
                        queue.append(other)
            seen |= component
            if len(component) == 1:
                left["not bonded"] += 1
                continue
            chain = _path(component, links)
            if chain is None:
                left["whose bonds branch or close a loop"] += len(component)
            else:
                chains.append(chain)
    return chains, left, notes


def _path(component: set[int], links: dict[int, set[int]]) -> list[int] | None:
    """``component``'s atoms in chain order: along its bonds if they form one chain, else in ID order if each atom is
    bonded to the next; None if neither."""
    degree = {atom_id: len(links[atom_id]) for atom_id in component}
    if max(degree.values()) <= 2 and sum(degree.values()) == 2 * (len(component) - 1):
        chain = [min(atom_id for atom_id in component if degree[atom_id] == 1)]
        previous = None
        while len(chain) < len(component):
            step = next(other for other in links[chain[-1]] if other != previous)
            previous = chain[-1]
            chain.append(step)
        return chain
    ordered = sorted(component)
    if all(b in links[a] for a, b in zip(ordered, ordered[1:])):
        return ordered
    return None


def _close(run: list[int], chains: list[list[int]], left: Counter) -> None:
    if len(run) > 1:
        chains.append(run)
    elif run:
        left["with no sphere close enough"] += 1


def _step(a: _Atom, b: _Atom, lengths: list[float], periodic: list[bool]) -> list[float]:
    """From ``a`` to ``b``, across a periodic face where that is shorter."""
    step = [q - p for p, q in zip(a.position, b.position)]
    for axis in range(3):
        if periodic[axis]:
            step[axis] -= lengths[axis] * round(step[axis] / lengths[axis])
    return step


def _too_far(a: _Atom, b: _Atom, lengths: list[float], periodic: list[bool], path: Path) -> bool:
    sizes = [size for size in (a.diameter, b.diameter) if size is not None and size > 0.0]
    if not sizes:
        raise ValueError(f"{path}: the atoms have no diameter to find fibers by; pass diameter=")
    return math.hypot(*_step(a, b, lengths, periodic)) > _UNBONDED_GAP * max(sizes)


def _unwrap(chain: list[_Atom], ids: list[int], lengths: list[float], periodic: list[bool],
            path: Path) -> list[list[float]]:
    """The chain's sphere centers as one unbroken line: each sphere placed from the one before it, across a periodic
    face where that is shorter, starting from the first sphere's unwrapped position."""
    first = chain[0]
    point = [first.position[axis] + (first.image[axis] * lengths[axis] if first.image and periodic[axis] else 0.0)
             for axis in range(3)]
    line = [point]
    for index in range(1, len(chain)):
        step = _step(chain[index - 1], chain[index], lengths, periodic)
        if not any(step):
            raise ValueError(f"{path}: atoms {ids[index - 1]} and {ids[index]} of one fiber are at the same point")
        point = [value + delta for value, delta in zip(point, step)]
        line.append(point)
    return line


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _fractions(points: list[list[float]]) -> list[float]:
    """Each point's share of the line's length up to it."""
    total, shares = 0.0, [0.0]
    for a, b in zip(points, points[1:]):
        total += math.dist(a, b)
        shares.append(total)
    return [share / total for share in shares]


def _along(points: list[list[float]], fractions: list[float]) -> list[list[float]]:
    """The points at ``fractions`` of the line's length."""
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(a, b))
    result = []
    for fraction in fractions:
        target = fraction * cumulative[-1]
        index = min(max(bisect.bisect_right(cumulative, target) - 1, 0), len(points) - 2)
        span = cumulative[index + 1] - cumulative[index]
        t = (target - cumulative[index]) / span if span > 0.0 else 0.0
        result.append([p + t * (q - p) for p, q in zip(points[index], points[index + 1])])
    return result
