"""A conformal mesh: fibers and binder bridges fragmented in one gmsh model,
so the binder shares its nodes with the fibers instead of being tied to them."""

from __future__ import annotations

import math
import time

import numpy as np

from ._mesh import FemMesh
from ._source import as_assembly, binder_name, material_table
from ._tet import GMSH_TET4, GMSH_TET10, GMSH_TO_NASTRAN_TET10, _fiber_solid, _perimeter, _pieces, _upright, \
    _straighten_distorted, _without_repeated_vertices


def conformal_mesh(
    assembly,
    bonds=None,
    *,
    bond_radius_ratio: float = 0.5,
    elements_around: int = 8,
    order: int = 1,
    region=None,
    progress: bool = True,
    verbose: bool = False,
) -> FemMesh:
    """Mesh the fibers and a smooth binder bridge at each bond as one body
    whose tetrahedra share nodes across every fiber-binder face.

    Each fiber is the same solid as in :func:`tet_mesh`. Each bond becomes a
    round bridge from one fiber's centerline to the other's, of radius
    ``bond_radius_ratio`` times the thinner fiber's smaller radius; gmsh
    fragments all of them at once, so where a bridge meets a fiber the two
    meshes share one triangulation. The part of a bridge inside a fiber is
    that fiber's; fibers that overlap each other share the overlap, which
    goes to the first. There are no ties: the mesh is one connected body
    wherever fibers are bonded or overlap.

    The whole region goes into one gmsh model, which is far slower than
    :func:`tet_mesh`'s fiber at a time and can fail where fibers only just
    touch, so start with a small ``region``.

    Args:
        assembly: a ``tangle.Assembly`` or ``RunResult``.
        bonds: ``(fiber_a, point_a, fiber_b, point_b)`` per bond, fibers as
            indices into ``assembly.centerlines()`` and points in meters on
            their centerlines. Defaults to the assembly's junctions; for a CT
            fit, see ``fit_bonds``.
        bond_radius_ratio: each bridge's radius over the thinner fiber's
            smaller radius.
        elements_around: about this many element edges around a fiber.
        order: 1 for 4-node, 2 for 10-node tetrahedra.
        region: ``(low, high)`` corners in meters of the box to mesh, or the
            whole cell. A box smaller than the cell is never periodic.
        progress: print what it is doing.
        verbose: show gmsh's own output.
    """
    import gmsh

    started = time.monotonic()

    def say(text):
        if progress:
            print(f"conformal_mesh [{time.monotonic() - started:6.1f} s] {text}", flush=True)

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if not 0.0 < bond_radius_ratio <= 1.0:
        raise ValueError("bond_radius_ratio must be in (0, 1]")
    assembly = as_assembly(assembly)
    centerlines = [np.asarray(line, dtype=float) for line in assembly.centerlines()]
    if not centerlines:
        raise ValueError("the assembly has no fibers")
    long_axes = [np.asarray(axes, dtype=float) for axes in assembly.long_axes()]
    semi_axes = assembly.section_semi_axes()
    fiber_ids = assembly.fiber_ids()
    materials, fiber_material = material_table(assembly)
    cell = assembly.cell
    origin = np.asarray(cell.origin, dtype=float)
    lengths = np.asarray(cell.lengths, dtype=float)
    periodic = list(cell.periodic)
    if region is not None:
        low_m, high_m = (np.asarray(corner, dtype=float) for corner in region)
        if np.any(high_m <= low_m):
            raise ValueError("region's high corner must be above its low corner")
        whole = np.allclose(low_m, origin) and np.allclose(high_m, origin + lengths)
        if not whole:
            periodic = [False, False, False]
        origin, lengths = low_m, high_m - low_m
    if bonds is None:
        bonds = assembly_bonds(assembly)

    unit = min(short for _, short in semi_axes)  # see tet_mesh: OpenCASCADE wants lengths near 1
    low, size = origin / unit, lengths / unit
    element_size = float(np.median([_perimeter(long, short) for long, short in semi_axes])) / elements_around / unit

    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("tangle_fem_conformal")
        occ = gmsh.model.occ
        fiber_volumes, owners = [], []  # one entry per fiber solid piece
        for fiber, (points, axes, (long, short)) in enumerate(zip(centerlines, long_axes, semi_axes)):
            points, axes = _without_repeated_vertices(points / unit, axes)
            pieces = list(_pieces(points, long / unit, low, size, periodic, True))
            if not pieces:
                continue
            for tag in _fiber_solid(occ, fiber_ids[fiber], points, axes, long / unit, short / unit, pieces,
                                    (low, size)):
                fiber_volumes.append((3, tag))
                owners.append(fiber)
        if not fiber_volumes:
            raise ValueError("no fiber reaches into the region")
        say(f"{len(fiber_volumes)} fiber solids")

        bridges = []
        reach = low - 1.0, low + size + 1.0
        for fiber_a, point_a, fiber_b, point_b in bonds:
            a, b = np.asarray(point_a, dtype=float) / unit, np.asarray(point_b, dtype=float) / unit
            if np.any(np.maximum(a, b) < reach[0]) or np.any(np.minimum(a, b) > reach[1]):
                continue
            step = b - a
            if np.linalg.norm(step) < 1.0e-6:
                continue
            radius = bond_radius_ratio * min(semi_axes[fiber_a][1], semi_axes[fiber_b][1]) / unit
            bridges.append((3, occ.addCylinder(*a, *step, radius)))
        if bridges:
            box = occ.addBox(*low, *size)
            bridges, _ = occ.intersect(bridges, [(3, box)], removeObject=True, removeTool=True)
            bridges = [(dim, tag) for dim, tag in bridges if dim == 3]
        say(f"{len(bridges)} binder bridges")

        occ.synchronize()
        pieces, children = occ.fragment(fiber_volumes, bridges)
        occ.synchronize()
        owner_of = {}  # volume tag -> fiber index, or -1 for binder
        for parent, made in enumerate(children):  # the fibers come first, so a fiber keeps what it shares
            owner = owners[parent] if parent < len(owners) else -1
            for dim, tag in made:
                if dim == 3 and tag not in owner_of:
                    owner_of[tag] = owner
        binder_volumes = sum(1 for owner in owner_of.values() if owner < 0)
        say(f"fragmented into {len(owner_of)} volumes, {binder_volumes} of them binder; meshing")

        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(3, elements_around // 2))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            gmsh.model.mesh.setOrder(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        node_index = np.full(node_tags.max() + 1, -1, dtype=np.int64)
        node_index[node_tags] = np.arange(len(node_tags))
        nodes = np.array(coordinates, dtype=float).reshape(-1, 3)
        wanted, width = (GMSH_TET4, 4) if order == 1 else (GMSH_TET10, 10)
        element_blocks, block_owners = [], []
        for tag, owner in owner_of.items():
            types, _, members = gmsh.model.mesh.getElements(3, tag)
            for element_type, block in zip(types, members):
                if element_type != wanted:
                    raise RuntimeError(f"gmsh made element type {element_type} instead of tetrahedra")
                block = node_index[np.asarray(block, dtype=np.int64)].reshape(-1, width)
                element_blocks.append(block)
                block_owners.append(np.full(len(block), owner, dtype=np.int64))
    finally:
        gmsh.finalize()

    elements = np.concatenate(element_blocks)
    owner = np.concatenate(block_owners)
    if order == 2:
        elements = elements[:, GMSH_TO_NASTRAN_TET10]
    used, elements = np.unique(elements, return_inverse=True)
    elements = elements.reshape(len(owner), -1)
    nodes = nodes[used]
    elements = _upright(nodes, elements)
    if order == 2:
        _straighten_distorted(nodes, elements)
    nodes = nodes * unit

    is_binder = owner < 0
    fiber_of = np.where(is_binder, 0, owner)
    element_fibers = np.where(is_binder, 0, np.asarray(fiber_ids, dtype=np.int64)[fiber_of])
    element_materials = fiber_material[fiber_of]
    if is_binder.any():
        materials = (*materials, binder_name(materials))
        element_materials = np.where(is_binder, len(materials) - 1, element_materials)
    sequence = np.lexsort((np.arange(len(elements)), element_fibers, element_materials))
    say(f"{len(nodes)} nodes, {len(elements)} tetrahedra, {int(is_binder.sum())} of them binder")
    return FemMesh(
        nodes=nodes,
        elements=elements[sequence],
        element_type="tet4" if order == 1 else "tet10",
        element_fibers=element_fibers[sequence],
        element_materials=element_materials[sequence],
        materials=materials,
        cell_origin=tuple(origin.tolist()),
        cell_lengths=tuple(lengths.tolist()),
    )


def assembly_bonds(assembly) -> list:
    """``(fiber_a, point_a, fiber_b, point_b)`` at each of the assembly's
    junctions, points in meters on the two centerlines."""
    from .. import _tangle

    centerlines = [np.asarray(line, dtype=float) for line in assembly.centerlines()]
    bonds = []
    for _, fiber_a, segment_a, coordinate_a, fiber_b, segment_b, coordinate_b in _tangle.fem_bond_anchors(assembly):
        line_a, line_b = centerlines[fiber_a], centerlines[fiber_b]
        point_a = line_a[segment_a] + coordinate_a * (line_a[segment_a + 1] - line_a[segment_a])
        point_b = line_b[segment_b] + coordinate_b * (line_b[segment_b + 1] - line_b[segment_b])
        bonds.append((fiber_a, point_a, fiber_b, point_b))
    return bonds


def fit_bonds(fit) -> list:
    """``(fiber_a, point_a, fiber_b, point_b)`` for each of a CT fit's
    ``bonds`` (``fit.bonds``, from ``find_fibers``), points in meters on the
    fit's centerlines, as ``fit.to_assembly()`` places them."""
    from ..ct._fit import _attachments

    bonds = []
    for bond in fit.bonds or []:
        fiber_a, fiber_b = (int(k) for k in bond["fibers"])
        point_a, point_b = _attachments(bond, fit.centerlines)
        bonds.append((fiber_a, np.asarray(point_a) * fit.voxel_size, fiber_b, np.asarray(point_b) * fit.voxel_size))
    return bonds
