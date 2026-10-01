"""Tetrahedral meshes: each fiber's round or oval surface, meshed by gmsh."""

from __future__ import annotations

import itertools
import math

import numpy as np

from ._mesh import FemMesh
from ._source import as_assembly, material_table

# gmsh element types for 4- and 10-node tetrahedra.
GMSH_TET4, GMSH_TET10 = 4, 11
# gmsh numbers the last two mid-edge nodes of a 10-node tetrahedron (3-2 and
# 3-1) the other way round from Nastran and Abaqus (1-3 and 2-3, zero-based).
GMSH_TO_NASTRAN_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
# Swapping corners 1 and 2 turns a tetrahedron inside out, and back.
FLIP_TET4 = [0, 2, 1, 3]
FLIP_TET10 = [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]


def tet_mesh(
    assembly,
    element_size: float | None = None,
    *,
    order: int = 1,
    elements_around: int = 16,
    clip_to_cell: bool = True,
    verbose: bool = False,
) -> FemMesh:
    """Mesh the fibers with tetrahedra that follow each fiber's surface.

    Each fiber becomes a solid swept through round or oval cross-sections at
    its centerline vertices, with the oval's long axis where TANGLE puts it,
    and gmsh fills the solid with tetrahedra. Every fiber is its own body:
    fibers that touch or slightly overlap do not share nodes, so connect
    them in the solver (contact, or glued contact for bonded fibers). Fiber
    ends are flat. Where a cell wall slices a fiber lengthwise, gmsh may
    need finer elements in that fiber to mesh the thin edge of the cut, so
    such fibers get more elements. ``tet_mesh`` starts and stops its own
    gmsh session, so call it while gmsh is not initialized.

    Args:
        assembly: an ``Assembly`` or a recipe's ``RunResult``.
        element_size: largest element edge in meters. By default
            ``elements_around`` alone sets the size.
        order: 1 for 4-node tetrahedra (``CTETRA``/``C3D4``), 2 for 10-node
            tetrahedra (``CTETRA``/``C3D10``) with straight edges.
        elements_around: element edges around each fiber's cross-section,
            which also sets the element size along that fiber. 16 keeps a
            round fiber's volume within about 2%; more follows the surface
            more closely, and fewer makes a lighter mesh.
        clip_to_cell: cut the fibers at the cell walls, with their periodic
            images on periodic axes, so the mesh fills the same box as
            :func:`hex_mesh`. A fiber that only touches a wall, such as one
            resting on the floor, is left whole, so it can stick out by up
            to 5% of its thickness. With ``False``, every fiber is meshed
            whole along its unwrapped centerline.
        verbose: let gmsh print its progress.

    Returns:
        A :class:`FemMesh` of ``"tet4"`` or ``"tet10"`` elements.
    """
    try:
        import gmsh
    except ImportError as error:  # gmsh is an optional dependency
        raise ImportError("tangle.fem.tet_mesh needs gmsh: pip install gmsh") from error
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
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
    if element_size is not None and not (math.isfinite(element_size) and element_size > 0.0):
        raise ValueError("element_size must be finite and positive")
    if elements_around < 3:
        raise ValueError("elements_around must be at least 3")
    # OpenCASCADE's tolerances suit lengths near 1, not micrometers written in
    # meters, so the geometry is built in units of the thinnest fiber's smaller
    # radius and scaled back at the end.
    unit = min(short for _, short in semi_axes)
    low, size = origin / unit, lengths / unit
    fiber_sizes = [_perimeter(long, short) / elements_around / unit for long, short in semi_axes]
    if element_size is not None:
        fiber_sizes = [min(fiber_size, element_size / unit) for fiber_size in fiber_sizes]

    if gmsh.isInitialized():
        raise RuntimeError("tet_mesh runs its own gmsh session; call gmsh.finalize() first")
    _start_gmsh(gmsh, verbose)
    node_blocks, element_blocks, block_fibers = [], [], []
    offset = 0
    try:
        for fiber, (points, axes, (long, short)) in enumerate(zip(centerlines, long_axes, semi_axes)):
            points, axes = _without_repeated_vertices(points / unit, axes)
            pieces = list(_pieces(points, long / unit, low, size, periodic, clip_to_cell))
            if not pieces:
                continue
            mesh = _mesh_fiber(
                gmsh, fiber_ids[fiber], points, axes, long / unit, short / unit,
                pieces, (low, size) if clip_to_cell else None, fiber_sizes[fiber], order, elements_around, verbose,
            )
            if mesh is None:
                continue
            nodes, elements = mesh
            node_blocks.append(nodes)
            element_blocks.append(elements + offset)
            block_fibers.append(np.full(len(elements), fiber, dtype=np.int64))
            offset += len(nodes)
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
    if not element_blocks:
        raise ValueError("no fiber reaches into the cell")

    elements = np.concatenate(element_blocks)
    fibers = np.concatenate(block_fibers)
    if order == 2:
        elements = elements[:, GMSH_TO_NASTRAN_TET10]
    # Keep only the nodes the tetrahedra use, numbered in order.
    used, elements = np.unique(elements, return_inverse=True)
    elements = elements.reshape(len(fibers), -1)
    nodes = np.concatenate(node_blocks)[used] * unit
    corners = nodes[elements[:, :4]]
    edges = corners[:, 1:] - corners[:, :1]
    inverted = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])) < 0.0
    elements[inverted] = elements[inverted][:, FLIP_TET4 if order == 1 else FLIP_TET10]

    element_fibers = np.asarray(fiber_ids, dtype=np.int64)[fibers]
    element_materials = fiber_material[fibers]
    sequence = np.lexsort((np.arange(len(fibers)), element_fibers, element_materials))
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


def _start_gmsh(gmsh, verbose: bool) -> None:
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)


def _mesh_fiber(gmsh, fiber_id, points, axes, long, short, pieces, box, size, order, around, verbose):
    """Nodes and gmsh-ordered tetrahedra of one fiber, in the scaled units, or
    None when nothing of it is left inside the cell.

    The fiber gets its own gmsh model. Where a wall cuts a fiber lengthwise
    at a grazing angle, the cut face meets the fiber's surface in a thin
    wedge that a uniform size can fail to mesh; the later tries refine with
    the surface curvature, which resolves the wedge at the cost of more
    elements in that fiber."""
    tries = (0, max(3, around // 2), around, 2 * around)
    for attempt, curvature in enumerate(tries):
        if attempt:  # after a failure gmsh meshes nothing more until it restarts
            gmsh.finalize()
            _start_gmsh(gmsh, verbose)
        gmsh.model.add(f"tangle_fem_fiber_{fiber_id}")
        try:
            volumes = _fiber_solid(gmsh.model.occ, fiber_id, points, axes, long, short, pieces, box)
            if not volumes:
                return None
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            try:
                gmsh.model.mesh.generate(3)
                node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
                if len(node_tags) == 0:
                    raise RuntimeError("gmsh made no nodes")
            except Exception as error:  # gmsh raises a plain Exception
                if attempt + 1 == len(tries):
                    raise RuntimeError(f"gmsh could not mesh fiber {fiber_id}: {error}") from error
                continue
            node_tags = np.asarray(node_tags, dtype=np.int64)
            node_index = np.full(node_tags.max() + 1, -1, dtype=np.int64)
            node_index[node_tags] = np.arange(len(node_tags))
            wanted = GMSH_TET4 if order == 1 else GMSH_TET10
            blocks = []
            for tag in volumes:
                types, _, element_nodes = gmsh.model.mesh.getElements(3, tag)
                for element_type, nodes in zip(types, element_nodes):
                    if element_type != wanted:
                        raise RuntimeError(f"gmsh made element type {element_type} instead of tetrahedra")
                    blocks.append(node_index[np.asarray(nodes, dtype=np.int64)].reshape(-1, 4 if order == 1 else 10))
            return np.asarray(coordinates, dtype=float).reshape(-1, 3), np.concatenate(blocks)
        finally:
            gmsh.model.remove()


def _fiber_solid(occ, fiber_id, points, axes, long, short, pieces, box):
    """Volume tags of one fiber's solid pieces, cut to ``box`` (``(low,
    size)``) when it is given.

    A wall within 5% of the fiber's thickness of its surface neither cuts it
    nor brings in a periodic image: OpenCASCADE cannot cut a solid that
    touches a wall, as a fiber resting on the floor of the cell does."""
    sliver = 1.0e-6 * math.pi * long * short * short
    margin = 0.05 * short
    volumes = []
    for vertices, shift in pieces:
        piece = points[vertices] + shift
        tool, deep = None, np.zeros(len(piece), dtype=bool)
        if box is not None:
            low, high = box[0], box[0] + box[1]
            reach_low, reach_high = piece.min(axis=0) - long, piece.max(axis=0) + long
            if np.any(reach_high < low + margin) or np.any(reach_low > high - margin):
                continue  # this image only grazes the cell
            cut = (reach_low < low - margin) | (reach_high > high + margin)
            if cut.any():
                tool_low = np.where(cut, low, np.minimum(low, reach_low) - long)
                tool_high = np.where(cut, high, np.maximum(high, reach_high) + long)
                tool = occ.addBox(*tool_low, *(tool_high - tool_low))
            # Cross-sections at these vertices lie wholly inside the cell.
            deep = np.all((piece > low + long) & (piece < high - long), axis=1)
        try:
            solids = _loft(occ, piece, axes[vertices], long, short)
            if tool is not None:
                solids, _ = occ.intersect(solids, [(3, tool)], removeObject=True, removeTool=True)
        except Exception as error:  # gmsh raises a plain Exception
            raise RuntimeError(f"gmsh could not build fiber {fiber_id}: {error}") from error
        kept = 0
        for dim, tag in solids:
            if dim != 3:
                continue
            if occ.getMass(dim, tag) < sliver:
                occ.remove([(dim, tag)], recursive=True)
            else:
                volumes.append(tag)
                kept += 1
        if deep.any() and not kept:
            raise RuntimeError(f"gmsh lost fiber {fiber_id} while cutting it at the cell walls")
    occ.synchronize()
    return volumes


def _without_repeated_vertices(points: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], steps > 1.0e-9 * max(steps.max(initial=0.0), 1.0e-300)])
    return points[keep], axes[keep]


def _pieces(points, pad, origin, lengths, periodic, clip):
    """``(vertex slice, shift)`` for every stretch of the fiber to loft: the
    whole fiber, or each stretch of each periodic image that reaches the
    cell, with one extra vertex at both ends so the walls cut cleanly."""
    if not clip:
        yield slice(0, len(points)), np.zeros(3)
        return
    pad = 1.01 * pad
    low, high = points.min(axis=0) - pad, points.max(axis=0) + pad
    ranges = []
    for axis in range(3):
        if periodic[axis]:
            first = math.ceil((origin[axis] - high[axis]) / lengths[axis])
            last = math.floor((origin[axis] + lengths[axis] - low[axis]) / lengths[axis])
            ranges.append(range(first, last + 1))
        else:
            ranges.append(range(0, 1))
    for image in itertools.product(*ranges):
        shift = np.asarray(image, dtype=float) * lengths
        moved = points + shift
        segment_low = np.minimum(moved[:-1], moved[1:]) - pad
        segment_high = np.maximum(moved[:-1], moved[1:]) + pad
        reaches = np.all((segment_high >= origin) & (segment_low <= origin + lengths), axis=1)
        keep = reaches.copy()
        keep[1:] |= reaches[:-1]
        keep[:-1] |= reaches[1:]
        vertex = np.zeros(len(points), dtype=bool)
        vertex[:-1] |= keep
        vertex[1:] |= keep
        for start, stop in _runs(vertex):
            if stop - start >= 2:
                yield slice(start, stop), shift


def _perimeter(long: float, short: float) -> float:
    """An ellipse's perimeter (Ramanujan's approximation; exact for a circle)."""
    return math.pi * (3.0 * (long + short) - math.sqrt((3.0 * long + short) * (long + 3.0 * short)))


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return zip(edges[0::2], edges[1::2])


def _loft(occ, points: np.ndarray, long_axes: np.ndarray, long: float, short: float) -> list[tuple[int, int]]:
    """A solid through a cross-section at every vertex: an ellipse with its
    long semi-axis along TANGLE's long axis, or a circle."""
    tangents = np.empty_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    oval = long > short * (1.0 + 1.0e-9)
    directions = _section_directions(tangents, long_axes if oval else None)
    wires = []
    for point, tangent, direction in zip(points, tangents, directions):
        if oval:
            curve = occ.addEllipse(*point, long, short, zAxis=tangent.tolist(), xAxis=direction.tolist())
        else:
            curve = occ.addCircle(*point, long, zAxis=tangent.tolist(), xAxis=direction.tolist())
        wires.append(occ.addWire([curve]))
    solid = occ.addThruSections(wires, makeSolid=True, makeRuled=False)
    occ.remove([(1, wire) for wire in wires], recursive=True)
    return [(dim, tag) for dim, tag in solid if dim == 3]


def _section_directions(tangents: np.ndarray, long_axes: np.ndarray | None) -> np.ndarray:
    """A unit direction across the fiber at every vertex, turning as little
    as possible between vertices: TANGLE's long axis for an oval fiber, or a
    transported one for a round fiber, whose seam then runs straight."""
    directions = np.empty_like(tangents)
    if long_axes is None:
        start = np.eye(3)[np.argmin(np.abs(tangents[0]))]
    else:
        start = long_axes[0]
    previous = start
    for index, tangent in enumerate(tangents):
        candidate = previous if long_axes is None else long_axes[index]
        candidate = candidate - np.dot(candidate, tangent) * tangent
        norm = np.linalg.norm(candidate)
        if norm < 1.0e-9:  # fall back to the previous direction
            candidate = previous - np.dot(previous, tangent) * tangent
            norm = np.linalg.norm(candidate)
        candidate = candidate / norm
        if index and np.dot(candidate, directions[index - 1]) < 0.0:
            candidate = -candidate
        directions[index] = candidate
        previous = candidate
    return directions
