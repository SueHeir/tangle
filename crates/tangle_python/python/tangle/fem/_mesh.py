"""The mesh type shared by both meshers, and its Nastran and Abaqus writers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

import numpy as np

ELEMENT_NODE_COUNTS = {"hex8": 8, "tet4": 4, "tet10": 10}
NASTRAN_ELEMENTS = {"hex8": "CHEXA", "tet4": "CTETRA", "tet10": "CTETRA"}
ABAQUS_ELEMENTS = {"hex8": "C3D8", "tet4": "C3D4", "tet10": "C3D10"}
# The largest id an 8-column Nastran field holds.
MAX_NASTRAN_ID = 99_999_999
# Six tetrahedra around the 0-6 diagonal that fill a hexahedron in Nastran
# and Abaqus node order, each positively oriented.
HEX_TETRAHEDRA = np.array(
    [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
)


@dataclass(frozen=True)
class ElasticMaterial:
    """Isotropic linear-elastic properties, written as Nastran ``MAT1`` or
    Abaqus ``*ELASTIC`` and ``*DENSITY``.

    Use units that match the written mesh: pascals and kg/m^3 for the default
    meters, or MPa and t/mm^3 after ``scale=1e3`` (millimeters).
    """

    youngs_modulus: float
    poisson_ratio: float
    density: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.youngs_modulus) and self.youngs_modulus > 0.0):
            raise ValueError("youngs_modulus must be finite and positive")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be greater than -1 and less than 0.5")
        if self.density is not None and not (math.isfinite(self.density) and self.density > 0.0):
            raise ValueError("density must be finite and positive")


MaterialChoice = Union[ElasticMaterial, Mapping[str, ElasticMaterial], None]


@dataclass(frozen=True, eq=False)
class FemMesh:
    """A solid mesh of the fibers in a cell, ready to write for Nastran or
    Abaqus. Build one with :func:`tangle.fem.hex_mesh` or
    :func:`tangle.fem.tet_mesh`.

    Coordinates are in meters, like the rest of TANGLE; the writers can scale
    them. Node ``i`` is written with id ``i + 1`` and element ``j`` with id
    ``j + 1``.

    Attributes:
        nodes: ``(n, 3)`` node coordinates.
        elements: ``(m, k)`` zero-based node indices of each element, in
            Nastran and Abaqus node order (``k`` is 8, 4 or 10).
        element_type: ``"hex8"``, ``"tet4"`` or ``"tet10"``.
        element_fibers: ``(m,)`` TANGLE fiber id of each element, or 0 for
            binder.
        element_materials: ``(m,)`` index into ``materials`` of each element.
        materials: material names: the fibers' TANGLE materials, then
            ``"binder"`` when bonds were meshed.
        cell_origin: the cell's corner.
        cell_lengths: the cell's edge lengths. Nodes on its six faces form
            the ``XMIN`` to ``ZMAX`` node sets.
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_type: str
    element_fibers: np.ndarray
    element_materials: np.ndarray
    materials: tuple[str, ...]
    cell_origin: tuple[float, float, float]
    cell_lengths: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.element_type not in ELEMENT_NODE_COUNTS:
            raise ValueError(f"element_type must be one of {sorted(ELEMENT_NODE_COUNTS)}")
        count = len(self.elements)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (n, 3)")
        if self.elements.shape != (count, ELEMENT_NODE_COUNTS[self.element_type]):
            raise ValueError(f"{self.element_type} elements must have shape (m, {ELEMENT_NODE_COUNTS[self.element_type]})")
        if self.element_fibers.shape != (count,) or self.element_materials.shape != (count,):
            raise ValueError("element_fibers and element_materials need one entry per element")
        if count and (self.elements.min() < 0 or self.elements.max() >= len(self.nodes)):
            raise ValueError("elements refer to nodes that do not exist")
        if count and (self.element_materials.min() < 0 or self.element_materials.max() >= len(self.materials)):
            raise ValueError("element_materials refer to materials that do not exist")

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def element_count(self) -> int:
        return len(self.elements)

    @property
    def fiber_ids(self) -> list[int]:
        """TANGLE ids of the fibers that have elements, in increasing order."""
        return [int(fiber) for fiber in np.unique(self.element_fibers) if fiber != 0]

    def element_volumes(self) -> np.ndarray:
        """Volume of every element from its corner nodes (straight edges and
        flat faces)."""
        if self.element_type == "hex8":
            corners = self.elements[:, HEX_TETRAHEDRA]
        else:
            corners = self.elements[:, None, :4]
        points = self.nodes[corners]
        edges = points[..., 1:, :] - points[..., :1, :]
        determinants = np.einsum("...i,...i", edges[..., 0, :], np.cross(edges[..., 1, :], edges[..., 2, :]))
        return determinants.sum(axis=-1) / 6.0

    @property
    def volume(self) -> float:
        """Total element volume, in cubic meters."""
        return float(self.element_volumes().sum())

    def node_sets(self) -> dict[str, np.ndarray]:
        """Zero-based indices of the nodes on each cell face, keyed ``XMIN``,
        ``XMAX``, ``YMIN``, ``YMAX``, ``ZMIN`` and ``ZMAX``."""
        tolerance = 1.0e-9 * max(self.cell_lengths)
        sets = {}
        for axis, name in enumerate("XYZ"):
            low = self.cell_origin[axis]
            high = low + self.cell_lengths[axis]
            sets[f"{name}MIN"] = np.flatnonzero(np.abs(self.nodes[:, axis] - low) <= tolerance)
            sets[f"{name}MAX"] = np.flatnonzero(np.abs(self.nodes[:, axis] - high) <= tolerance)
        return sets

    def __repr__(self) -> str:
        return (
            f"FemMesh({self.element_type}: {self.node_count} nodes, "
            f"{self.element_count} elements, {len(self.fiber_ids)} fibers)"
        )

    def write_nastran(self, path: str | Path, material: MaterialChoice = None, *, scale: float = 1.0) -> Path:
        """Write the mesh as Nastran bulk data: ``GRID``, ``CHEXA`` or
        ``CTETRA``, ``PSOLID`` and ``MAT1`` entries.

        The file holds bulk data only. ``INCLUDE`` it after ``BEGIN BULK`` in
        a deck that has your loads and constraints, or import it into a
        pre-processor such as Femap, Patran or HyperMesh. Each fiber gets its
        own ``PSOLID``, numbered by its TANGLE fiber id; binder gets the next
        id. ``MAT1`` ids follow ``materials`` from 1.

        Args:
            path: the file to write, usually ``.bdf``.
            material: one :class:`ElasticMaterial` for every element, or a
                mapping from material name to one. A material left out gets
                no ``MAT1``, so the solver stops until you add it.
            scale: multiplies every coordinate; ``1e3`` writes millimeters.

        Returns:
            The path written.
        """
        path = Path(path)
        properties = _resolve_materials(self.materials, material)
        nodes = _scaled(self.nodes, scale)
        fibers, binder_pid = self._property_ids()
        if max(len(nodes), len(self.elements), binder_pid) > MAX_NASTRAN_ID:
            raise ValueError(f"Nastran ids are limited to {MAX_NASTRAN_ID:,}; this mesh needs more")
        name = NASTRAN_ELEMENTS[self.element_type]
        lines = [
            "$ TANGLE fiber mesh written by tangle.fem: Nastran bulk data.",
            f"$ {len(self.fiber_ids)} fibers, {len(nodes)} nodes, {len(self.elements)} "
            f"{name} elements ({self.element_type}).",
            f"$ Coordinates are meters times {scale:g}; use matching material units.",
            "$ INCLUDE this file after BEGIN BULK in a solution deck, or import it",
            "$ into a pre-processor (Femap, Patran, HyperMesh).",
            "$ PSOLID ids are TANGLE fiber ids"
            + (f"; binder is PSOLID {binder_pid}." if binder_pid else "."),
        ]
        for index, (material_name, elastic) in enumerate(zip(self.materials, properties), start=1):
            if elastic is None:
                lines.append(f"$ MAT1 {index} = {material_name}: not defined. Add a MAT1 {index} before solving.")
            else:
                lines.append(f"$ MAT1 {index} = {material_name}")
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n$\n")
            handle.write("$ Materials\n")
            for index, elastic in enumerate(properties, start=1):
                if elastic is not None:
                    handle.write(f"MAT1*   {index:16d}{_e16(elastic.youngs_modulus)}{'':16s}{_e16(elastic.poisson_ratio)}\n")
                    if elastic.density is not None:
                        handle.write(f"*       {_e16(elastic.density)}\n")
            handle.write("$ Properties: one PSOLID per fiber\n")
            for pid, material_index in fibers:
                handle.write(f"PSOLID  {pid:8d}{material_index + 1:8d}\n")
            handle.write("$ Nodes\n")
            _write_rows(
                handle,
                "GRID*   %16d                %16.9E%16.9E\n*       %16.9E\n",
                zip(range(1, len(nodes) + 1), *nodes.T.tolist()),
            )
            handle.write("$ Elements\n")
            element_pids = self._element_property_ids(binder_pid)
            nodes_per_element = ELEMENT_NODE_COUNTS[self.element_type]
            first = min(nodes_per_element, 6)
            row = f"{name:<8s}" + "%8d" * (2 + first)
            if nodes_per_element > first:
                row += "\n        " + "%8d" * (nodes_per_element - first)
            _write_rows(
                handle,
                row + "\n",
                zip(
                    range(1, len(self.elements) + 1),
                    element_pids.tolist(),
                    *(self.elements + 1).T.tolist(),
                ),
            )
        return path

    def write_abaqus(
        self,
        path: str | Path,
        material: MaterialChoice = None,
        *,
        scale: float = 1.0,
        element_type: str | None = None,
    ) -> Path:
        """Write the mesh as an Abaqus input file: ``*NODE``, ``*ELEMENT`` with
        one element set per fiber (``FIBER_<id>``, and ``BINDER``), one
        ``*SOLID SECTION`` per material, the materials, and one ``*NSET`` of
        nodes per cell face (``XMIN`` to ``ZMAX``).

        Args:
            path: the file to write, usually ``.inp``.
            material: one :class:`ElasticMaterial` for every element, or a
                mapping from material name to one. A material left out is
                referenced but not defined, so Abaqus stops until you add it.
            scale: multiplies every coordinate; ``1e3`` writes millimeters.
            element_type: the Abaqus element name. The default is ``C3D8``,
                ``C3D4`` or ``C3D10``; ``C3D8I`` or ``C3D8R`` also suit the
                hexahedra.

        Returns:
            The path written.
        """
        path = Path(path)
        properties = _resolve_materials(self.materials, material)
        nodes = _scaled(self.nodes, scale)
        abaqus_type = element_type or ABAQUS_ELEMENTS[self.element_type]
        material_names = _abaqus_names(self.materials)
        order = np.lexsort((np.arange(len(self.elements)), self.element_fibers, self.element_materials))
        groups = _group_boundaries(self.element_fibers[order], self.element_materials[order])
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(
                "** TANGLE fiber mesh written by tangle.fem: Abaqus input.\n"
                f"** {len(self.fiber_ids)} fibers, {len(nodes)} nodes, {len(self.elements)} "
                f"{abaqus_type} elements.\n"
                f"** Coordinates are meters times {scale:g}; use matching material units.\n"
                "*HEADING\nTANGLE fiber mesh\n*NODE\n"
            )
            _write_rows(handle, "%d, %.12e, %.12e, %.12e\n", zip(range(1, len(nodes) + 1), *nodes.T.tolist()))
            element_row = "%d" + ", %d" * ELEMENT_NODE_COUNTS[self.element_type] + "\n"
            members: dict[int, list[str]] = {}
            for start, stop in groups:
                fiber = int(self.element_fibers[order[start]])
                material_index = int(self.element_materials[order[start]])
                set_name = f"FIBER_{fiber}" if fiber else "BINDER"
                members.setdefault(material_index, []).append(set_name)
                rows = order[start:stop]
                handle.write(f"*ELEMENT, TYPE={abaqus_type}, ELSET={set_name}\n")
                _write_rows(handle, element_row, zip((rows + 1).tolist(), *(self.elements[rows] + 1).T.tolist()))
            for material_index, set_names in members.items():
                section_set = f"MATERIAL_{material_names[material_index]}"
                handle.write(f"*ELSET, ELSET={section_set}\n")
                _write_list(handle, set_names)
                handle.write(f"*SOLID SECTION, ELSET={section_set}, MATERIAL={material_names[material_index]}\n,\n")
            for material_index, elastic in enumerate(properties):
                if material_index not in members:
                    continue
                if elastic is None:
                    handle.write(
                        f"** Material {material_names[material_index]} is not defined: add "
                        f"*MATERIAL, NAME={material_names[material_index]} before running.\n"
                    )
                    continue
                handle.write(
                    f"*MATERIAL, NAME={material_names[material_index]}\n*ELASTIC\n"
                    f"{elastic.youngs_modulus:.12e}, {elastic.poisson_ratio:.12e}\n"
                )
                if elastic.density is not None:
                    handle.write(f"*DENSITY\n{elastic.density:.12e}\n")
            for set_name, indices in self.node_sets().items():
                if len(indices):
                    handle.write(f"*NSET, NSET={set_name}\n")
                    _write_list(handle, (indices + 1).tolist())
        return path

    def _property_ids(self) -> tuple[list[tuple[int, int]], int]:
        """``(pid, material index)`` of every fiber, and the binder's pid (0
        without binder)."""
        fibers = {}
        for fiber, material_index in zip(self.element_fibers.tolist(), self.element_materials.tolist()):
            fibers.setdefault(fiber, material_index)
        binder = fibers.pop(0, None)
        binder_pid = max(fibers, default=0) + 1 if binder is not None else 0
        properties = sorted(fibers.items())
        if binder is not None:
            properties.append((binder_pid, binder))
        return properties, binder_pid

    def _element_property_ids(self, binder_pid: int) -> np.ndarray:
        return np.where(self.element_fibers == 0, binder_pid, self.element_fibers)


def _resolve_materials(names: tuple[str, ...], material: MaterialChoice) -> list[ElasticMaterial | None]:
    if material is None:
        return [None] * len(names)
    if isinstance(material, ElasticMaterial):
        return [material] * len(names)
    unknown = sorted(set(material) - set(names))
    if unknown:
        raise ValueError(f"the mesh has no material named {', '.join(unknown)}; it has {', '.join(names)}")
    return [material.get(name) for name in names]


def _scaled(nodes: np.ndarray, scale: float) -> np.ndarray:
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError("scale must be finite and positive")
    scaled = nodes * scale
    # Keep every coordinate inside a 16-column field.
    scaled[np.abs(scaled) < 1.0e-90] = 0.0
    return scaled


def _e16(value: float) -> str:
    """A real in one 16-column large-field Nastran field."""
    return f"{value:16.9E}"


def _write_rows(handle, row_format: str, rows, chunk: int = 50_000) -> None:
    batch = []
    for row in rows:
        batch.append(row_format % row)
        if len(batch) == chunk:
            handle.write("".join(batch))
            batch.clear()
    handle.write("".join(batch))


def _write_list(handle, items, per_line: int = 16) -> None:
    items = list(items)
    for start in range(0, len(items), per_line):
        handle.write(", ".join(str(item) for item in items[start : start + per_line]) + "\n")


def _group_boundaries(fibers: np.ndarray, materials: np.ndarray) -> list[tuple[int, int]]:
    if len(fibers) == 0:
        return []
    change = np.flatnonzero((fibers[1:] != fibers[:-1]) | (materials[1:] != materials[:-1])) + 1
    edges = [0, *change.tolist(), len(fibers)]
    return list(zip(edges[:-1], edges[1:]))


def _abaqus_names(names: tuple[str, ...]) -> list[str]:
    """Abaqus-safe, unique upper-case labels for the material names."""
    labels: list[str] = []
    for name in names:
        label = re.sub(r"[^A-Za-z0-9_]", "_", name).upper() or "MATERIAL"
        if not label[0].isalpha():
            label = "M_" + label
        base, suffix = label, 2
        while label in labels:
            label, suffix = f"{base}_{suffix}", suffix + 1
        labels.append(label)
    return labels
