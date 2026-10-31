"""A conformal mesh: fibers and binder bridges fragmented in one gmsh model,
so the binder shares its nodes with the fibers instead of being tied to them."""

from __future__ import annotations

import itertools
import math
import time
import warnings

import numpy as np

from ._mesh import FemMesh
from ._source import as_assembly, binder_name, material_table
from ._tet import GMSH_TET4, _fiber_solid, _perimeter, _pieces, _upright, _without_repeated_vertices


def conformal_mesh(
    assembly,
    bonds=None,
    *,
    bond_radius_ratio: float = 0.5,
    elements_around: int = 8,
    order: int = 1,
    region=None,
    min_quality: float = 0.02,
    binder=None,
    voxel_size: float | None = None,
    progress: bool = True,
    verbose: bool = False,
) -> FemMesh:
    """Mesh the fibers and their binder (a smooth bridge at each bond, or a
    binder mask) as one body whose tetrahedra share nodes across every
    fiber-binder face.

    Each fiber is the same solid as in :func:`tet_mesh`. Each bond becomes a
    round bridge from one fiber's centerline to the other's, of radius
    ``bond_radius_ratio`` times the thinner fiber's smaller radius; gmsh
    fragments all of them at once, so where a bridge meets a fiber the two
    meshes share one triangulation. The part of a bridge inside a fiber is
    that fiber's; fibers that overlap each other share the overlap, which
    goes to the first. There are no ties: the mesh is one connected body
    wherever fibers are bonded or overlap. Where two fibers touch, a bridge
    has a hole down its middle, so its binder never thins to a knife-edge.

    It aims for no tetrahedron flatter than 0.1 (:func:`tet_quality`):
    slivers gmsh leaves are mended (short edges collapsed, tetrahedra
    flipped, corners slid along their surfaces). It warns with their
    places if any stay under 0.1, and raises rather than return one
    flatter than ``min_quality``. (Meshing again finer around them helps
    as often as it hurts, and can crash gmsh's frontal mesher.)

    With a binder mask (``binder``, as for :func:`sweep_mesh`) the binder is
    the mask instead of bridges: gmsh meshes the fibers and the space
    between them together, the tetrahedra of that space whose middles lie in
    the mask (smoothed by a voxel, read at the level that keeps its volume)
    become binder and the rest are dropped, with any piece of binder that
    no fiber holds. The binder's outer nodes are then moved onto the
    smoothed mask's surface. So the binder shares its nodes with the fibers,
    with no voxel steps and no ties, at the fibers' coarseness: its outer
    surface is faceted at the element size. ``elements_around=5`` or ``6``
    keeps the mesh coarse; 8 and more can crash gmsh on a whole cell.

    The whole region goes into one gmsh model, which is far slower than
    :func:`tet_mesh`'s fiber at a time and can fail where fibers only just
    touch, so start with a small ``region``.

    Args:
        assembly: a ``tangle.Assembly`` or ``RunResult``.
        bonds: ``(fiber_a, point_a, fiber_b, point_b)`` per bond, fibers as
            indices into ``assembly.centerlines()`` and points in meters on
            their centerlines. Defaults to the assembly's junctions; for a CT
            fit, see ``fit_bonds``. Not used with ``binder``.
        bond_radius_ratio: each bridge's radius over the thinner fiber's
            smaller radius.
        elements_around: about this many element edges around a fiber; 5 or
            6 for a coarse mesh.
        order: 1 for 4-node, 2 for 10-node tetrahedra (with straight edges:
            the mid-edge nodes are not moved onto the fibers' surfaces).
        region: ``(low, high)`` corners in meters of the box to mesh, or the
            whole cell. A smaller box is cut from the periodic cell: the
            fibers' and bridges' periodic images that reach into it are in it.
        min_quality: the least signed inverse condition number
            (:func:`tet_quality`) a tetrahedron may have; 0.02 by default,
            where Nastran only refuses tetrahedra inside out or flat.
        binder: a boolean ``(z, y, x)`` binder mask over the whole cell in
            voxels of ``voxel_size`` (a CT fit's binder, or
            ``scan.binder_occupancy > 0.5`` for a synthetic scan), meshed
            as binder in place of bridges. Needs SciPy.
        voxel_size: the mask's voxel size in meters.
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
    if not 0.0 < bond_radius_ratio < 1.0:
        raise ValueError("bond_radius_ratio must be in (0, 1): a bridge as wide as the fiber is tangent to it")
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
    cell_low, cell_size = origin, lengths  # the periodic images come from the cell, the cut from the region
    if region is not None:
        low_m, high_m = (np.asarray(corner, dtype=float) for corner in region)
        if np.any(high_m <= low_m):
            raise ValueError("region's high corner must be above its low corner")
        origin, lengths = low_m, high_m - low_m
    if binder is not None:
        if voxel_size is None:
            raise ValueError("pass the binder mask's voxel_size with it")
        binder = np.asarray(binder, dtype=bool)
        if binder.ndim != 3 or not np.allclose(np.asarray(binder.shape[::-1]) * voxel_size, np.asarray(cell.lengths)):
            raise ValueError("binder must be a (z, y, x) mask over the whole cell in voxels of voxel_size")
        bonds = []  # the mask is the binder
    elif bonds is None:
        bonds = assembly_bonds(assembly)

    unit = min(short for _, short in semi_axes)  # see tet_mesh: OpenCASCADE wants lengths near 1
    low, size = origin / unit, lengths / unit
    cell_low, cell_size = cell_low / unit, cell_size / unit
    element_size = float(np.median([_perimeter(long, short) for long, short in semi_axes])) / elements_around / unit

    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("tangle_fem_conformal")
        occ = gmsh.model.occ
        fiber_volumes, owners = [], []  # one entry per fiber solid piece
        for fiber, (points, axes, (long, short)) in enumerate(zip(centerlines, long_axes, semi_axes)):
            points, axes = _without_repeated_vertices(points / unit, axes)
            pieces = list(_pieces(points, long / unit, cell_low, cell_size, periodic, True))
            if not pieces:
                continue
            for tag in _fiber_solid(occ, fiber_ids[fiber], points, axes, long / unit, short / unit, pieces,
                                    (low, size)):
                fiber_volumes.append((3, tag))
                owners.append(fiber)
        if not fiber_volumes:
            raise ValueError("no fiber reaches into the region")
        fiber_volumes, owners = _without_overlaps(occ, fiber_volumes, owners)
        say(f"{len(fiber_volumes)} fiber solids")

        bridges, bridged = [], 0
        images = [range(-2, 3) if wraps else range(0, 1) for wraps in periodic]
        for fiber_a, point_a, fiber_b, point_b in bonds:
            thinner = min(semi_axes[fiber_a][1], semi_axes[fiber_b][1]) / unit
            radius = bond_radius_ratio * thinner
            # A bridge ending at a fiber's flat end would graze its cap: start it a little way in instead.
            a = _off_the_end(centerlines[fiber_a] / unit, np.asarray(point_a, dtype=float) / unit, 1.5 * radius)
            b = _off_the_end(centerlines[fiber_b] / unit, np.asarray(point_b, dtype=float) / unit, 1.5 * radius)
            step = b - a
            for axis in range(3):  # a bond across a periodic wall joins a to b's nearest image
                if periodic[axis]:
                    step[axis] -= round(step[axis] / cell_size[axis]) * cell_size[axis]
            if np.linalg.norm(step) < 1.0e-6:
                continue
            gap = np.linalg.norm(step) - (semi_axes[fiber_a][1] + semi_axes[fiber_b][1]) / unit
            hole = _bridge_hole(gap, thinner, radius)
            for image in itertools.product(*images):
                start = a + np.asarray(image, dtype=float) * cell_size
                if np.any(np.maximum(start, start + step) + radius < low) or \
                        np.any(np.minimum(start, start + step) - radius > low + size):
                    continue
                bridge = [(3, occ.addCylinder(*start, *step, radius))]
                if hole > 0.0:
                    core = occ.addCylinder(*(start - 0.05 * step), *(1.1 * step), hole)
                    bridge, _ = occ.cut(bridge, [(3, core)], removeObject=True, removeTool=True)
                bridges.extend(volume for volume in bridge if volume[0] == 3)
                bridged += 1
        if bridges:
            box = occ.addBox(*low, *size)
            bridges, _ = occ.intersect(bridges, [(3, box)], removeObject=True, removeTool=True)
            bridges = [(dim, tag) for dim, tag in bridges if dim == 3]
        # Bridges that overlap one another come out of the intersection in pieces.
        say(f"{bridged} binder bridges, {len(bridges)} pieces, from {len(bonds)} bonds")

        if binder is not None:  # the space between the fibers, meshed with them; the mask picks the binder from it
            bridges = [(3, occ.addBox(*low, *size))]
        occ.synchronize()
        pieces, children = occ.fragment(fiber_volumes, bridges)
        occ.synchronize()
        owner_of = {}  # volume tag -> fiber index, or -1 for binder (-2 for the space between, with a mask)
        for parent, made in enumerate(children):  # the fibers come first, so a fiber keeps what it shares
            owner = owners[parent] if parent < len(owners) else (-1 if binder is None else -2)
            for dim, tag in made:
                if dim == 3 and tag not in owner_of:
                    owner_of[tag] = owner
        binder_volumes = sum(1 for owner in owner_of.values() if owner < 0)  # with a mask: the space between
        say(f"fragmented into {len(owner_of)} volumes, {binder_volumes} of them binder; meshing")

        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(3, elements_around // 2))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        # Delaunay's boundary recovery fails ("PLC Error") where fibers nearly touch; the frontal mesher copes.
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.mesh.generate(3)
        nodes, elements, owner, on = _linear_mesh(gmsh, owner_of)
        nodes, elements, owner, mended = _mend(gmsh, nodes, elements, owner, on, max(_AIM, min_quality))
        if binder is not None:
            nodes, elements, owner, dropped = _carve(nodes, elements, owner, binder, voxel_size / unit,
                                                     np.asarray(cell.origin, dtype=float) / unit)
            say(f"binder from the mask: {int((owner < 0).sum())} tetrahedra kept, {dropped} dropped")
        quality = tet_quality(nodes, elements)
        say(f"{len(elements)} tetrahedra, {mended} mended, {int((quality < _AIM).sum())} still under {_AIM}, "
            f"worst {quality.min():.3f}")
    finally:
        gmsh.finalize()

    flat = np.flatnonzero(quality < _AIM)
    if len(flat):
        worst = flat[np.argsort(quality[flat])]
        where = "; ".join(f"{quality[k]:.3f} at {np.round(nodes[elements[k]].mean(axis=0) * unit * 1e6, 2).tolist()} um"
                          for k in worst[:5])
        if quality.min() < min_quality:
            raise RuntimeError(f"conformal_mesh: {int((quality < min_quality).sum())} tetrahedra stay flatter than "
                               f"min_quality {min_quality} (worst: {where}); try more "
                               "elements_around, a smaller bond_radius_ratio or another region")
        warnings.warn(f"conformal_mesh: {len(flat)} of {len(elements)} tetrahedra are flatter than {_AIM} "
                      f"(worst: {where})", stacklevel=2)
    if order == 2:
        nodes, elements = _straight_tet10(nodes, elements)
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


def _off_the_end(line, point, keep):
    """``point`` on the polyline ``line``, moved along it to at least
    ``keep`` from either end where the line is long enough."""
    steps = np.diff(line, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    arcs = np.concatenate([[0.0], np.cumsum(lengths)])
    if arcs[-1] <= 2.0 * keep:
        return point
    along = np.clip(np.einsum("ij,ij->i", point - line[:-1], steps) / np.maximum(lengths**2, 1e-300), 0.0, 1.0)
    nearest = np.argmin(np.linalg.norm(line[:-1] + along[:, None] * steps - point, axis=1))
    arc = arcs[nearest] + along[nearest] * lengths[nearest]
    if keep <= arc <= arcs[-1] - keep:
        return point
    arc = min(max(arc, keep), arcs[-1] - keep)
    k = min(np.searchsorted(arcs, arc, side="right") - 1, len(steps) - 1)
    return line[k] + (arc - arcs[k]) / lengths[k] * steps[k]


_HOLE_GAP = 0.15  # see _bridge_hole


def _bridge_hole(gap, thinner, radius):
    """The radius of the hole down a bridge's axis, where its two fibers'
    surfaces are closer than ``_HOLE_GAP`` of the thinner fiber's radius.

    Round fibers that touch or nearly touch leave between them a gap that
    thins to nothing at the contact, and binder filling it right in would
    end in a knife-edge that only slivers can mesh. Two surfaces of radius
    ``thinner`` that are ``gap`` apart on the axis are about ``gap + s**2 /
    thinner`` apart at ``s`` from it (crossing fibers curve only one way
    each), so the binder stops where that reaches ``_HOLE_GAP * thinner``,
    and meets the fibers there at about 20 degrees or more. The fibers keep
    whatever they share at the contact itself."""
    reach = _HOLE_GAP * thinner - gap
    if reach <= 0.0:
        return 0.0
    return min(math.sqrt(reach * thinner), 0.8 * radius)


_AIM = 0.1  # the quality conformal_mesh mends towards, and warns below
# Edges of the regular tetrahedron of unit edge, as columns: the shape tet_quality measures against.
_REGULAR_INVERSE = np.linalg.inv(np.array([[1.0, 0.5, 0.5], [0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0) / 6.0],
                                           [0.0, 0.0, math.sqrt(2.0 / 3.0)]]))
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))  # the face opposite each corner
# Mid-edge nodes 5 to 10 of a Nastran CTETRA (Abaqus C3D10) lie on these edges, zero-based.
_TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def tet_quality(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Each tetrahedron's signed inverse condition number (gmsh's SICN):
    1 for a regular one, near 0 for a sliver, negative inside out. Only its
    four corners count."""
    corners = nodes[elements[:, :4]]
    jacobian = np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1)) @ _REGULAR_INVERSE
    columns = np.transpose(jacobian, (0, 2, 1))
    # The inverse's norm is the adjugate's over |det|, and the adjugate's rows are crosses of the columns.
    adjugate = np.stack([np.cross(columns[:, 1], columns[:, 2]), np.cross(columns[:, 2], columns[:, 0]),
                         np.cross(columns[:, 0], columns[:, 1])], axis=1)
    det = np.einsum("ni,ni->n", columns[:, 0], adjugate[:, 0])
    norms = np.sqrt(np.einsum("nij,nij->n", jacobian, jacobian) * np.einsum("nij,nij->n", adjugate, adjugate))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norms > 0.0, 3.0 * det / norms, 0.0)


def _linear_mesh(gmsh, owner_of):
    """The 4-node tetrahedra gmsh made, upright, with their owners (fiber
    index, or -1 for binder) and only the nodes they use."""
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_index = np.full(node_tags.max() + 1, -1, dtype=np.int64)
    node_index[node_tags] = np.arange(len(node_tags))
    nodes = np.array(coordinates, dtype=float).reshape(-1, 3)
    blocks, owners = [], []
    for tag, owner in owner_of.items():
        types, _, members = gmsh.model.mesh.getElements(3, tag)
        for element_type, block in zip(types, members):
            if element_type != GMSH_TET4:
                raise RuntimeError(f"gmsh made element type {element_type} instead of tetrahedra")
            block = node_index[np.asarray(block, dtype=np.int64)].reshape(-1, 4)
            blocks.append(block)
            owners.append(np.full(len(block), owner, dtype=np.int64))
    elements, owner = np.concatenate(blocks), np.concatenate(owners)
    # Where each node lies: on a CAD point (dim 0), curve (1) or surface (2), or inside a volume (3).
    on = np.zeros((len(nodes), 2), dtype=np.int64)
    for dim, tag in gmsh.model.getEntities():
        tags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
        on[node_index[np.asarray(tags, dtype=np.int64)]] = dim, tag
    used, elements = np.unique(elements, return_inverse=True)
    nodes = nodes[used]
    return nodes, _upright(nodes, elements.reshape(-1, 4)), owner, on[used]


def _mend(gmsh, nodes, elements, owner, on, min_quality):
    """Lift tetrahedra under ``min_quality`` without moving the fibers'
    surfaces: collapse a sliver's edge that is much shorter than the rest,
    left where two CAD edges or vertices fall a few nanometers apart, then
    flip tetrahedra of one owner into better shapes (2-3 flips across a
    face, and removing an edge from the ring around it), and drop slivers
    lying on the outer surface; and move their corners towards the middle
    of their neighbours, those on a CAD surface or curve back onto it, those
    at a CAD corner only a little. A change is kept only if every tetrahedron it
    touches ends better than the worst one before. Returns the new mesh and
    how many changes it made."""
    nodes = nodes.copy()
    changes = 0
    for _ in range(8):
        quality = tet_quality(nodes, elements)
        flat = np.flatnonzero(quality < min_quality)
        if not len(flat):
            break
        made = _collapse_short_edges(nodes, elements, on, quality, flat)
        if made is not None:
            elements, owner, count = made[0], owner[made[1]], made[2]
            changes += count
            quality = tet_quality(nodes, elements)
            flat = np.flatnonzero(quality < min_quality)
        elements, owner, count = _flip(nodes, elements, owner, quality, flat)
        changes += count
        quality = tet_quality(nodes, elements)
        smoothed = _smooth(gmsh, nodes, elements, on, np.flatnonzero(quality < min_quality))
        changes += smoothed
        quality = tet_quality(nodes, elements)
        elements, owner, dropped = _drop_caps(elements, owner, quality, np.flatnonzero(quality < min_quality))
        changes += dropped
        if made is None and not count and not dropped and not smoothed:
            break
    used, elements = np.unique(elements, return_inverse=True)
    return nodes[used], elements.reshape(-1, 4), owner, changes


def _corner_tets(elements, count):
    """For each node, the tetrahedra at it: (order, starts) as for a CSR matrix."""
    flat = elements.ravel()
    order = np.argsort(flat, kind="stable")
    return order // 4, np.searchsorted(flat[order], np.arange(count + 1))


def _collapse_short_edges(nodes, elements, on, quality, flat, ratio=0.25):
    """Collapse the shortest edge of each of ``flat`` that is under ``ratio``
    of its longest onto one end or its middle, moving ``nodes`` in place
    (and ``on``, where a node lies, with it).
    Returns (elements, which of the old ones they are, collapses), or None."""
    corners = nodes[elements[flat]]
    lengths = np.stack([np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a, b in _EDGES], axis=1)
    shortest = lengths.argmin(axis=1)
    short = lengths[np.arange(len(flat)), shortest] < ratio * lengths.max(axis=1)
    if not short.any():
        return None
    at, starts = _corner_tets(elements, len(nodes))
    alive = np.ones(len(elements), dtype=bool)
    touched = np.zeros(len(nodes), dtype=bool)
    collapses = 0
    for k in np.flatnonzero(short)[np.argsort(lengths[short, shortest[short]])]:
        a, b = (elements[flat[k], corner] for corner in _EDGES[shortest[k]])
        if touched[a] or touched[b]:
            continue
        ring = np.unique(np.concatenate([at[starts[a]:starts[a + 1]], at[starts[b]:starts[b + 1]]]))
        ring = ring[alive[ring]]
        both = (elements[ring] == a).any(axis=1) & (elements[ring] == b).any(axis=1)
        kept = ring[~both]
        moved = np.where(elements[kept] == a, b, elements[kept])
        before, saved, best = quality[ring].min(), nodes[b].copy(), None
        for target in (nodes[a], saved, 0.5 * (nodes[a] + saved)):
            nodes[b] = target
            after = tet_quality(nodes, moved).min() if len(moved) else np.inf
            if after > before and (best is None or after > best[0]):
                best = (after, target.copy())
        nodes[b] = saved if best is None else best[1]
        if best is None:
            continue
        if on[a, 0] < on[b, 0]:  # the kept node now lies where the more constrained one did
            on[b] = on[a]
        elements[kept] = moved
        alive[ring[both]] = False
        touched[np.unique(elements[ring])] = True
        collapses += 1
    if not collapses:
        return None
    keep = np.flatnonzero(alive)
    return elements[keep], keep, collapses


def _flip(nodes, elements, owner, quality, flat):
    """2-3 flips across a face, and removing an edge from the ring of 3 to 7
    tetrahedra around it, between
    tetrahedra of one owner, wherever they lift the worst of them; so the
    fiber-binder interfaces and the outer surface stay as they are."""
    faces = np.sort(elements[:, _FACES].reshape(-1, 3), axis=1)
    _, face_of = np.unique(faces, axis=0, return_inverse=True)
    face_of = face_of.ravel()
    face_order = np.argsort(face_of, kind="stable")
    face_starts = np.searchsorted(face_of[face_order], np.arange(face_of.max() + 2))
    at, starts = _corner_tets(elements, len(nodes))
    dead = np.zeros(len(elements), dtype=bool)
    made, made_owner = [], []
    for t in flat[np.argsort(quality[flat])]:
        if dead[t]:
            continue
        best = None
        for corner in range(4):  # 2-3 across the face opposite this corner
            f = face_of[4 * t + corner]
            other = [slot for slot in face_order[face_starts[f]:face_starts[f + 1]] if slot // 4 != t]
            if not other:
                continue
            u = other[0] // 4
            if dead[u] or owner[u] != owner[t]:
                continue
            trial = np.repeat(elements[t:t + 1], 3, axis=0)
            for row, c in enumerate(_FACES[corner]):
                trial[row, c] = elements[u, other[0] % 4]
            worst = tet_quality(nodes, trial).min()
            if worst > min(quality[t], quality[u]) and (best is None or worst > best[0]):
                best = (worst, [t, u], trial)
        for a, b in _EDGES:  # remove the edge from corner a to corner b
            i, j = elements[t, a], elements[t, b]
            ring = np.intersect1d(at[starts[i]:starts[i + 1]], at[starts[j]:starts[j + 1]])
            if not 3 <= len(ring) <= 7 or dead[ring].any() or np.any(owner[ring] != owner[t]):
                continue
            for trial in _edge_removals(nodes, elements[ring], i, j):
                worst = tet_quality(nodes, trial).min()
                if worst > quality[ring].min() and (best is None or worst > best[0]):
                    best = (worst, list(ring), trial)
        if best is not None:
            dead[best[1]] = True
            made.extend(best[2])
            made_owner.extend([owner[t]] * len(best[2]))
    if not made:
        return elements, owner, 0
    elements = np.concatenate([elements[~dead], np.array(made)])
    owner = np.concatenate([owner[~dead], np.array(made_owner, dtype=owner.dtype)])
    return elements, owner, len(made_owner) // 2


def _edge_removals(nodes, ring, i, j):
    """Each way of filling the ring of tetrahedra around edge (i, j) without
    that edge: a fan of triangles across the ring's rim, each joined to i
    and to j. None if the ring is open (the edge lies on a surface) or a way
    turns a tetrahedron inside out."""
    pairs = [tuple(tet[(tet != i) & (tet != j)]) for tet in ring]
    if any(len(pair) != 2 for pair in pairs):
        return
    rim = list(pairs[0])
    left = pairs[1:]
    while left:
        for k, (p, q) in enumerate(left):
            if rim[-1] in (p, q):
                rim.append(q if p == rim[-1] else p)
                del left[k]
                break
        else:
            return
    if rim[-1] != rim[0]:
        return  # open ring
    rim = rim[:-1]
    n = len(rim)
    if len(set(rim)) != n:
        return
    for apex in range(n):
        fan = [(rim[apex], rim[(apex + m) % n], rim[(apex + m + 1) % n]) for m in range(1, n - 1)]
        trial = np.array([[x, y, z, i] for x, y, z in fan] + [[x, z, y, j] for x, y, z in fan])
        quality = tet_quality(nodes, trial)
        if np.all(quality > 0.0):
            yield trial
        elif np.all(quality < 0.0):
            yield trial[:, [0, 2, 1, 3]]


def _smooth(gmsh, nodes, elements, on, flat):
    """Move each corner of ``flat`` towards the middle of its neighbours,
    back onto the CAD curve or surface it lies on (``on``: dim, tag per
    node; one at a CAD corner moves a quarter of its shortest edge at
    most), where that lifts the worst
    tetrahedron around it. Moves ``nodes`` in place."""
    at, starts = _corner_tets(elements, len(nodes))
    moved = 0
    for node in np.unique(elements[flat]):
        dim, tag = on[node]
        star = at[starts[node]:starts[node + 1]]
        around = np.setdiff1d(elements[star], [node])
        here, middle = nodes[node].copy(), nodes[around].mean(axis=0)
        if dim == 0:  # a CAD corner moves only a little, a quarter of its shortest edge at most
            reach = 0.25 * np.linalg.norm(nodes[around] - here, axis=1).min()
            step = np.linalg.norm(middle - here)
            middle = here + (middle - here) * min(1.0, reach / max(step, 1e-300))
        before, best = tet_quality(nodes, elements[star]).min(), None
        for share in (1.0, 0.5, 0.25):
            target = here + share * (middle - here)
            if 0 < dim < 3:
                target = np.asarray(gmsh.model.getClosestPoint(int(dim), int(tag), target)[0])
            nodes[node] = target
            after = tet_quality(nodes, elements[star]).min()
            if after > before and (best is None or after > best[0]):
                best = (after, target)
        nodes[node] = here if best is None else best[1]
        moved += best is not None
    return moved


def _drop_caps(elements, owner, quality, flat):
    """Remove each of ``flat`` that has two faces on the outer surface and
    is the only tetrahedron on the edge between them: a sliver lying on the
    surface, whose two other faces then become the surface instead."""
    faces = np.sort(elements[:, _FACES].reshape(-1, 3), axis=1)
    _, face_of, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    outer = counts[face_of.ravel()].reshape(-1, 4) == 1
    edges = np.sort(elements[:, _EDGES].reshape(-1, 2), axis=1)
    _, edge_of, edge_counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_count = edge_counts[edge_of.ravel()].reshape(-1, 6)
    drop = []
    for t in flat:
        sides = np.flatnonzero(outer[t])
        if len(sides) != 2:
            continue
        ridge = sorted(set(range(4)) - set(sides))  # the corners both outer faces hold
        if edge_count[t, _EDGES.index(tuple(ridge))] == 1:
            drop.append(t)
    if not drop:
        return elements, owner, 0
    keep = np.setdiff1d(np.arange(len(elements)), drop)
    return elements[keep], owner[keep], len(drop)


def _carve(nodes, elements, owner, mask, voxel, origin):
    """Keep the tetrahedra between the fibers (owner -2) whose middles lie
    in the binder ``mask`` (smoothed by a voxel, read at the level that
    keeps its volume) as
    binder (-1), and drop the rest, and any piece of binder that touches no
    fiber. Returns the mesh and how many tetrahedra were dropped."""
    from scipy import ndimage

    field = ndimage.gaussian_filter(mask.astype(float), 1.0)
    # Read the smoothed mask at the level that keeps the mask's volume (thin sheets would shrink at 0.5).
    field -= np.partition(field.ravel(), field.size - int(mask.sum()))[field.size - int(mask.sum())] - 0.5
    between = np.flatnonzero(owner == -2)
    middles = nodes[elements[between]].mean(axis=1)
    where = ((middles - origin) / voxel - 0.5)[:, ::-1].T  # (z, y, x) voxel coordinates
    inside = ndimage.map_coordinates(field, where, order=1, mode="nearest") >= 0.5
    owner = owner.copy()
    owner[between[inside]] = -1
    # Smooth the carving: a binder tetrahedron with three or four faces on empty space is a spike, and an empty
    # one with three or four faces on binder (or fibers) a dent.
    faces = np.sort(elements[:, _FACES].reshape(-1, 3), axis=1)
    _, face_of = np.unique(faces, axis=0, return_inverse=True)
    face_of = face_of.ravel()
    order = np.argsort(face_of, kind="stable")
    paired = face_of[order[1:]] == face_of[order[:-1]]
    neighbour = np.full(len(face_of), -1)
    neighbour[order[1:][paired]] = order[:-1][paired] // 4
    neighbour[order[:-1][paired]] = order[1:][paired] // 4
    neighbour = neighbour.reshape(-1, 4)
    for _ in range(3):
        kind = np.where(neighbour >= 0, owner[np.maximum(neighbour, 0)], -3)  # -3: the cell's walls
        empty_faces = ((kind == -2) | (kind == -3)).sum(axis=1)
        solid_faces = (kind >= -1).sum(axis=1)
        spikes = (owner == -1) & (empty_faces >= 3)
        dents = (owner == -2) & (solid_faces >= 3) & np.isin(np.arange(len(owner)), between)
        if not spikes.any() and not dents.any():
            break
        owner[spikes], owner[dents] = -2, -1
    keep = owner != -2
    # Binder pieces (joined through shared nodes) that no fiber holds.
    binder_tets = np.flatnonzero(keep & (owner == -1))
    if len(binder_tets):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        rows = np.repeat(np.arange(len(binder_tets)), 4)
        graph = coo_matrix((np.ones(len(rows)), (rows, elements[binder_tets].ravel())),
                           shape=(len(binder_tets), len(nodes))).tocsr()
        _, piece = connected_components(graph @ graph.T, directed=False)
        on_fiber = np.zeros(len(nodes), dtype=bool)
        on_fiber[elements[owner >= 0].ravel()] = True
        held = np.zeros(piece.max() + 1, dtype=bool)
        np.logical_or.at(held, piece, on_fiber[elements[binder_tets]].any(axis=1))
        keep[binder_tets[~held[piece]]] = False
    used, kept = np.unique(elements[keep], return_inverse=True)
    nodes, elements, owner = nodes[used], kept.reshape(-1, 4), owner[keep]
    _snap(nodes, elements, owner, field, voxel, origin)
    return nodes, elements, owner, int((~keep).sum())


def _snap(nodes, elements, owner, field, voxel, origin, aim=_AIM):
    """Move the nodes of the binder's free surface (faces of one binder
    tetrahedron, touching no fiber) onto the mask's 0.5 level along its
    gradient, a step at a time and only as far as leaves every tetrahedron
    around them better than ``aim`` (or no worse than it was), so that the
    carved binder is smooth instead of jagged. Moves ``nodes`` in place."""
    from scipy import ndimage

    faces = np.sort(elements[:, _FACES].reshape(-1, 3), axis=1)
    _, face_of, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    free = (counts[face_of.ravel()] == 1).reshape(-1, 4) & (owner == -1)[:, None]
    on_fiber = np.zeros(len(nodes), dtype=bool)
    on_fiber[elements[owner >= 0].ravel()] = True
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    movable = np.zeros(len(nodes), dtype=bool)
    movable[faces[free.ravel()].ravel()] = True
    movable &= ~on_fiber
    movable &= np.all((nodes > low + 1e-6 * voxel) & (nodes < high - 1e-6 * voxel), axis=1)  # not on the cell's walls
    gradient = np.gradient(field)  # along z, y, x

    def sample(values, points):
        where = ((points - origin) / voxel - 0.5)[:, ::-1].T
        return ndimage.map_coordinates(values, where, order=1, mode="nearest")

    at, starts = _corner_tets(elements, len(nodes))
    for _ in range(6):
        candidates = np.flatnonzero(movable)
        level = sample(field, nodes[candidates]) - 0.5
        grad = np.stack([sample(g, nodes[candidates]) for g in gradient[::-1]], axis=1) / voxel  # x, y, z
        norm2 = np.einsum("ij,ij->i", grad, grad)
        step = -(level / np.maximum(norm2, 1e-300))[:, None] * grad
        lengths = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, voxel / np.maximum(lengths, 1e-300))[:, None]  # a voxel at a time
        moved = 0
        for node, delta in zip(candidates, step):
            if not np.any(delta):
                continue
            star = at[starts[node]:starts[node + 1]]
            here = nodes[node].copy()
            floor = min(aim, tet_quality(nodes, elements[star]).min())
            for share in (1.0, 0.5, 0.25):
                nodes[node] = here + share * delta
                if tet_quality(nodes, elements[star]).min() >= floor:
                    moved += 1
                    break
            else:
                nodes[node] = here
        if not moved:
            break


def _straight_tet10(nodes, elements):
    """10-node tetrahedra with straight edges: a node at the middle of each
    edge, shared by the tetrahedra on it, in Nastran's order."""
    edges = np.sort(elements[:, _TET10_EDGES].reshape(-1, 2), axis=1)
    unique, middle_of = np.unique(edges, axis=0, return_inverse=True)
    middles = 0.5 * (nodes[unique[:, 0]] + nodes[unique[:, 1]])
    return (np.concatenate([nodes, middles]),
            np.concatenate([elements, len(nodes) + middle_of.reshape(-1, 6)], axis=1))


def _without_overlaps(occ, volumes, owners):
    """The fiber pieces cut by the other fibers' pieces before them, so that
    where two fibers overlap a little the overlap is the earlier fiber's.
    Left whole, it is a lens-thin piece whose two faces nearly coincide, and
    gmsh cannot mesh it ("overlapping facets")."""
    kept, kept_owners, boxes = [], [], []
    for volume, owner in zip(volumes, owners):
        box = np.asarray(occ.getBoundingBox(*volume))
        tools = [other for other, other_owner, other_box in zip(kept, kept_owners, boxes)
                 if other_owner != owner and np.all(other_box[:3] <= box[3:]) and np.all(box[:3] <= other_box[3:])]
        made = [volume]
        if tools:
            made, _ = occ.cut([volume], tools, removeObject=True, removeTool=False)
            made = [(dim, tag) for dim, tag in made if dim == 3]
        for piece in made:
            kept.append(piece)
            kept_owners.append(owner)
            boxes.append(np.asarray(occ.getBoundingBox(*piece)))
    return kept, kept_owners


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
