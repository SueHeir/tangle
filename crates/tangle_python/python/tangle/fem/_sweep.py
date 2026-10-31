"""Swept meshes: each fiber as a few wedges between hexagonal cross-sections,
and bonds as beams."""

from __future__ import annotations

import itertools
import math

import numpy as np

from .. import _tangle
from ._mesh import PENTA_FACES, PENTA_TETRAHEDRA, Bonds, FemMesh, Tie
from ._source import as_assembly, binder_name, material_table
from ._tet import _binder_mesh, _binder_voxel_size, _moved, _section_directions

# Two wedges that fill a voxel, as corners of hex_mesh's CORNERS: each
# triangle runs counterclockwise seen from the one above it.
VOXEL_PRISMS = np.array([[0, 1, 2, 4, 5, 6], [0, 2, 3, 4, 6, 7]])
# Turning a wedge's triangles round turns it inside out, and back.
FLIP_PENTA = [0, 2, 1, 3, 5, 4]
# The faces of a wedge (PENTA_FACES) on the fiber's surface: the side
# between two section edges, and the triangles that close the fiber's ends.
SIDE_FACE, START_FACE, END_FACE = 3, 0, 1
# A wall cuts a fiber's end flush with it unless the fiber runs within this
# cosine of along the wall; such an end stays square to the fiber.
WALL_COSINE = 0.25


def sweep_mesh(
    assembly,
    *,
    nodes_around: int = 6,
    max_length: float = 4.0,
    bend_tolerance: float = 0.25,
    clip_to_cell: bool = True,
    bond_radius_ratio: float | None = None,
    binder=None,
    voxel_size: float | None = None,
    bridges=None,
) -> FemMesh:
    """Mesh each fiber with few wedges: a cross-section of ``nodes_around``
    nodes on its surface and one in its middle every so often along the
    centerline, and six-node wedges (``CPENTA``/``C3D6``) between one
    section and the next.

    The section is a polygon, a hexagon by default, on the fiber's round or
    oval outline, scaled so that its area is the fiber's; a hexagon so also
    has a round fiber's bending stiffness within 1%. Sections lie along the
    centerline at its ends and at most ``max_length`` diameters apart, and
    closer only where the fiber bends: between two sections the centerline
    stays within ``bend_tolerance`` of a radius of the straight line from
    one to the other. Every fiber is its own body, as in :func:`tet_mesh`.

    Bonds at persistent junctions (``bond_radius_ratio``) become beams
    (:class:`Bonds`): a section lies at each bond's anchor on each fiber,
    and a round beam joins nodes at the middles of the two, each of which
    follows its section (``RBE3``, or an Abaqus distributing coupling). The
    bonds come from the fibers' current shape, so they follow a relaxed
    assembly. A binder mask (``binder``) becomes two wedges to each voxel
    of ``voxel_size``, tied to the fibers by a :class:`Tie`, which the
    Nastran file writes as ``MPC`` equations and the Abaqus file as a
    ``*TIE``.

    Bonds can instead be solid binder that shares the fibers' nodes
    (``bridges``): each becomes two binder wedges between the two fibers.
    Each fiber gets a section half a side on either side of the bond, turned
    so a flat side faces the other fiber, and the binder's two quadrilateral
    faces are those sides of the fibers' own wedges. The mesh then needs no
    ties and no beams, and stays as coarse as the fibers'.

    Args:
        assembly: an ``Assembly`` or a recipe's ``RunResult``.
        nodes_around: nodes on each section's outline; with the middle one,
            7 to a section by default.
        max_length: the longest wedge, in fiber diameters (the mean of an
            oval's two widths).
        bend_tolerance: how far the centerline may stray from a wedge's
            straight axis, as a fraction of the fiber's smaller radius.
        clip_to_cell: cut the fibers' centerlines at the cell walls, with
            their periodic images on periodic axes, so the mesh fills the
            cell; a cut end lies in the wall unless the fiber runs nearly
            along it. With ``False``, every fiber is meshed whole.
        bond_radius_ratio: also mesh the bonds of persistent junctions as
            beams with this radius as a fraction of the thinner fiber's
            radius, as the binder bridges of :func:`hex_mesh` have.
        binder: also mesh these voxels as binder, as for :func:`tet_mesh`:
            a boolean ``(z, y, x)`` array over the grid of ``voxel_size``
            voxels across the cell, or over a grid a whole number of times
            finer such as a CT scan's own. Voxels a fiber fills stay fiber.
        voxel_size: the edge in meters of the binder's voxels, which must
            divide every cell length. By default, a whole number of the
            mask's voxels as near as that allows to the fibers' smaller
            radius (their median).
        bridges: also mesh bonds as solid binder bridges: ``True`` for the
            assembly's persistent junctions, or ``(fiber_a, point_a,
            fiber_b, point_b)`` for each bond, fibers as indices into the
            assembly's fibers and points in meters on their centerlines (for
            a CT fit, ``fem.fit_bonds(fit)`` with ``fit.to_assembly()``).
            Not with ``bond_radius_ratio``. Bonds whose fibers' sides cannot
            take a bridge (they are crowded by other bonds, too near a
            fiber's end or a wall, or the sides overlap) are left out, with a
            warning.

    Returns:
        A :class:`FemMesh` of ``"penta6"`` elements, with :attr:`FemMesh.bonds`
        when there are bonds.
    """
    if nodes_around < 3:
        raise ValueError("nodes_around must be at least 3")
    for value, name in ((max_length, "max_length"), (bend_tolerance, "bend_tolerance")):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if bond_radius_ratio is not None and not (math.isfinite(bond_radius_ratio) and bond_radius_ratio > 0.0):
        raise ValueError("bond_radius_ratio must be finite and positive")
    if bridges is not None and bridges is not False and bond_radius_ratio is not None:
        raise ValueError("bridges and bond_radius_ratio both mesh the bonds; pass one of them")
    if voxel_size is not None and binder is None:
        raise ValueError("voxel_size only sets the binder's voxels; pass binder with it")
    if binder is not None and not clip_to_cell:
        raise ValueError("binder fills the cell's voxels, so it needs clip_to_cell=True")
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
    if binder is not None and voxel_size is None:
        voxel_size = _binder_voxel_size(np.shape(binder), lengths, float(np.median([short for _, short in semi_axes])))
    binder_mesh = None
    if binder is not None:  # before the fibers, so a binder mask that does not fit fails at once
        pairs = np.array([(corner, corner) for corner in range(8)])
        binder_mesh = _binder_mesh(assembly, voxel_size, None, binder, 1, split=(pairs, VOXEL_PRISMS, PENTA_FACES))
        if binder_mesh is None:
            raise ValueError("the binder mask has no voxel outside the fibers at this voxel_size")

    # Each bond's junction and its two anchors: (fiber index, arc length along its centerline).
    bonds = []
    if bond_radius_ratio is not None:
        found = _tangle.fem_bond_anchors(assembly)
        if not found:
            raise ValueError(
                "bond_radius_ratio bonds the assembly's junctions, and it has none; a CT fit's binder is a mask: "
                "pass binder=fit.binder"
            )
        for junction, *ends in found:
            anchors = []
            for fiber, segment, coordinate in (ends[:3], ends[3:]):
                arcs = _arc_lengths(centerlines[fiber])
                anchors.append((fiber, arcs[segment] + coordinate * (arcs[segment + 1] - arcs[segment])))
            bonds.append((junction, *anchors))
    forced = [[] for _ in centerlines]
    for _, *anchors in bonds:
        for fiber, arc in anchors:
            forced[fiber].append(arc)

    # Each bridge's two ends: (fiber, arc length of the bond on it), and the
    # sections half a side before and after it on each fiber.
    bridge_ends = _bridge_ends(assembly, centerlines, bridges) if bridges is not None and bridges is not False else []
    scale = math.sqrt(2.0 * math.pi / (nodes_around * math.sin(2.0 * math.pi / nodes_around)))
    half_sides = [scale * short * math.sin(math.pi / nodes_around) for _, short in semi_axes]
    facing_of = {}  # (fiber, low arc) -> (bridge, end, high arc, unit direction toward the other fiber)
    for bridge, ends in enumerate(bridge_ends):
        for end, (fiber, arc) in enumerate(ends):
            other_fiber, other_arc = ends[1 - end]
            here = _at(centerlines[fiber], _arc_lengths(centerlines[fiber]), arc)
            there = _at(centerlines[other_fiber], _arc_lengths(centerlines[other_fiber]), other_arc)
            toward = there - here
            for axis in range(3):
                if periodic[axis]:
                    toward[axis] -= round(toward[axis] / lengths[axis]) * lengths[axis]
            low, high = arc - half_sides[fiber], arc + half_sides[fiber]
            if low <= 0.0 or high >= _arc_lengths(centerlines[fiber])[-1] or np.linalg.norm(toward) <= 0.0:
                continue
            facing_of[(fiber, low)] = (bridge, end, high, toward / np.linalg.norm(toward))
            forced[fiber] += [low, high]
    bridge_images = {}  # (bridge, end) -> [(low section nodes, high section nodes, facing side, bond position)]

    node_blocks, element_blocks, block_fibers, surfaces = [], [], [], []
    stations_at = {}  # (fiber, arc) -> [(section nodes, middle position)], one per image in the cell
    node_count = element_count = 0
    for fiber, (points, axes, (long, short)) in enumerate(zip(centerlines, long_axes, semi_axes)):
        arcs = _arc_lengths(points)
        if arcs[-1] <= 0.0:
            continue
        spacing = max_length * (long + short)
        gap = 0.05 * (long + short)  # sections closer than this would make flat wedges
        anchors = sorted(set(forced[fiber]))
        for shift, start, end, start_wall, end_wall in _intervals(points, arcs, origin, lengths, periodic, clip_to_cell):
            if end - start < gap:
                continue
            kept = [arc for arc in anchors if start - 1e-12 <= arc <= end + 1e-12]
            # A slanted end slides its outline along the fiber into the wall, by up to `reach`: keep the
            # next section beyond that, or the end stays square when a bond's section or the other end is there.
            reach = [_wall_reach(points, arcs, arc, wall, long, nodes_around) for arc, wall in
                     ((start, start_wall), (end, end_wall))]
            if sum(reach) > end - start - gap:
                reach = [0.0, 0.0]
            if any(start < arc < start + reach[0] for arc in kept):
                reach[0] = 0.0
            if any(end - reach[1] < arc < end for arc in kept):
                reach[1] = 0.0
            stations = _stations(points, arcs, start, end, kept, spacing, bend_tolerance * short, gap, reach)
            facing, wanted = {}, []  # station index -> direction its flat side faces; the bridges here
            for (bridge_fiber, low), (bridge, bridge_end, high, toward) in facing_of.items():
                if bridge_fiber != fiber or not (start < low and high < end):
                    continue
                first, second = np.searchsorted(stations, [low - 1e-9 * spacing, high - 1e-9 * spacing])
                if (second != first + 1 or second >= len(stations) or abs(stations[first] - low) > 1e-9 * spacing
                        or abs(stations[second] - high) > 1e-9 * spacing or first in facing or second in facing):
                    continue  # another section lies between, or one of these already faces another bond
                facing[first] = facing[second] = toward
                wanted.append((bridge, bridge_end, first, (low + high) / 2.0))
            nodes, elements, outer, sides = _piece(points + shift, axes, arcs, stations, long, short, nodes_around,
                                                   start_wall if reach[0] else None, end_wall if reach[1] else None,
                                                   facing)
            for bridge, bridge_end, first, middle in wanted:
                low_section = node_count + first * (nodes_around + 1) + np.arange(nodes_around + 1)
                bridge_images.setdefault((bridge, bridge_end), []).append(
                    (low_section, low_section + nodes_around + 1, sides[first], _at(points, arcs, middle) + shift))
            node_blocks.append(nodes)
            element_blocks.append(elements + node_count)
            block_fibers.append(np.full(len(elements), fiber, dtype=np.int64))
            surfaces.append(np.stack([outer[:, 0] + element_count, outer[:, 1]], axis=1))
            for arc in kept:
                index = int(np.argmin(np.abs(stations - arc)))
                section = node_count + index * (nodes_around + 1) + np.arange(nodes_around + 1)
                stations_at.setdefault((fiber, arc), []).append((section, nodes[index * (nodes_around + 1)]))
            node_count += len(nodes)
            element_count += len(elements)
    if not element_blocks:
        raise ValueError("no fiber reaches into the cell")

    nodes = np.concatenate(node_blocks)
    elements = np.concatenate(element_blocks)
    fibers = np.concatenate(block_fibers)
    element_fibers = np.asarray(fiber_ids, dtype=np.int64)[fibers]
    element_materials = fiber_material[fibers]
    surfaces = np.concatenate(surfaces)

    binder_material = len(materials)  # the binder's, added after the fibers' when there is any
    tie = None
    if binder_mesh is not None:
        binder_nodes, binder_elements, next_to_fibers = binder_mesh
        if len(next_to_fibers):
            # A binder node next to a voxel a fiber fills, or fills part of, lies within that voxel's
            # diagonal, about 1.7 voxels, of the fiber's round surface; the section's flat sides lie up to
            # a tenth of a radius from it, and the centerline up to bend_tolerance from the wedges' axes.
            largest = max(long for long, _ in semi_axes)
            tie = (next_to_fibers + [len(elements), 0], surfaces, 2.0 * voxel_size + 2.0 * (0.1 + bend_tolerance) * largest)
        elements = np.concatenate([elements, binder_elements + len(nodes)])
        nodes = np.concatenate([nodes, binder_nodes])
        element_fibers = np.concatenate([element_fibers, np.zeros(len(binder_elements), dtype=np.int64)])
        element_materials = np.concatenate(
            [element_materials, np.full(len(binder_elements), binder_material, dtype=np.int64)]
        )
    if bridge_ends:
        binder_nodes = _bridge_wedges(nodes, bridge_ends, bridge_images, centerlines, semi_axes, nodes_around,
                                      lengths, periodic)
        if len(binder_nodes):
            elements = np.concatenate([elements, binder_nodes])
            element_fibers = np.concatenate([element_fibers, np.zeros(len(binder_nodes), dtype=np.int64)])
            element_materials = np.concatenate(
                [element_materials, np.full(len(binder_nodes), binder_material, dtype=np.int64)]
            )
    beams = None
    if bonds:
        nodes, beams = _bond_beams(nodes, bonds, stations_at, centerlines, semi_axes, bond_radius_ratio,
                                   origin, lengths, periodic, clip_to_cell, binder_material)
    if binder_mesh is not None or beams is not None or (element_materials == binder_material).any():
        materials = (*materials, binder_name(materials))

    sequence = np.lexsort((np.arange(len(elements)), element_fibers, element_materials))
    ties = ()
    if tie is not None:
        position = np.empty_like(sequence)
        position[sequence] = np.arange(len(sequence))
        secondary, main, tolerance = tie
        ties = (Tie("BINDER_TIE", _moved(secondary, position), _moved(main, position), tolerance),)
    return FemMesh(
        nodes=nodes,
        elements=elements[sequence],
        element_type="penta6",
        element_fibers=element_fibers[sequence],
        element_materials=element_materials[sequence],
        materials=materials,
        cell_origin=tuple(origin.tolist()),
        cell_lengths=tuple(lengths.tolist()),
        ties=ties,
        bonds=beams,
    )


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    """Arc length along the polyline at each of its vertices."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _at(points: np.ndarray, arcs: np.ndarray, arc: float) -> np.ndarray:
    """The point of the polyline at this arc length."""
    segment = int(np.clip(np.searchsorted(arcs, arc, side="right") - 1, 0, len(points) - 2))
    length = arcs[segment + 1] - arcs[segment]
    t = 0.0 if length <= 0.0 else float(np.clip((arc - arcs[segment]) / length, 0.0, 1.0))
    return points[segment] + t * (points[segment + 1] - points[segment])


def _intervals(points, arcs, origin, lengths, periodic, clip):
    """``(shift, start, end, start wall, end wall)`` for every stretch of the
    centerline, or of one of its periodic images, inside the cell: its arc
    lengths, and the ``(axis, coordinate)`` of the wall that cuts each end,
    or None at an end of the fiber."""
    if not clip:
        yield np.zeros(3), 0.0, float(arcs[-1]), None, None
        return
    high_corner = origin + lengths
    low, high = points.min(axis=0), points.max(axis=0)
    ranges = []
    for axis in range(3):
        if periodic[axis]:
            first = math.ceil((origin[axis] - high[axis]) / lengths[axis]) - 1
            last = math.floor((high_corner[axis] - low[axis]) / lengths[axis]) + 1
            ranges.append(range(first, last + 1))
        else:
            ranges.append(range(0, 1))
    for image in itertools.product(*ranges):
        shift = np.asarray(image, dtype=float) * lengths
        moved = points + shift
        current = None  # [start, end, start wall, end wall] of the stretch being followed
        for segment in range(len(points) - 1):
            a, step = moved[segment], moved[segment + 1] - moved[segment]
            t0, t1, wall0, wall1 = 0.0, 1.0, None, None
            for axis in range(3):
                for p, q, value in ((-step[axis], a[axis] - origin[axis], origin[axis]),
                                    (step[axis], high_corner[axis] - a[axis], high_corner[axis])):
                    if p == 0.0:
                        if q < 0.0:
                            t0, t1 = 1.0, 0.0
                    elif p < 0.0:
                        if q / p > t0:
                            t0, wall0 = q / p, (axis, value)
                    elif q / p < t1:
                        t1, wall1 = q / p, (axis, value)
            if t0 >= t1:
                continue
            length = arcs[segment + 1] - arcs[segment]
            start, end = arcs[segment] + t0 * length, arcs[segment] + t1 * length
            if current is not None and wall0 is None and abs(start - current[1]) <= 1e-12 * max(arcs[-1], 1e-300):
                current[1], current[3] = end, wall1
            else:
                if current is not None:
                    yield (shift, *current)
                current = [start, end, wall0, wall1]
            if wall1 is not None:
                yield (shift, *current)
                current = None
        if current is not None:
            yield (shift, *current)


def _wall_reach(points, arcs, arc, wall, long, around) -> float:
    """How far along the fiber the outline of a stretch's end at ``arc``
    slides to lie in its ``wall`` (with a margin), or 0 when there is no
    wall there or the fiber runs too nearly along it to slide."""
    if wall is None:
        return 0.0
    segment = int(np.clip(np.searchsorted(arcs, arc, side="right") - 1, 0, len(points) - 2))
    while segment > 0 and arcs[segment + 1] - arcs[segment] <= 0.0:
        segment -= 1
    step = points[segment + 1] - points[segment]
    cosine = abs(step[wall[0]]) / max(float(np.linalg.norm(step)), 1e-300)
    if cosine < WALL_COSINE:
        return 0.0
    scale = math.sqrt(2.0 * math.pi / (around * math.sin(2.0 * math.pi / around)))
    return 1.25 * scale * long * math.sqrt(max(1.0 - cosine * cosine, 0.0)) / cosine + 1e-9 * max(arcs[-1], 1e-300)


def _stations(points, arcs, start, end, forced, spacing, tolerance, gap, clear=(0.0, 0.0)):
    """Arc lengths of a stretch's sections: its ends and ``forced`` always
    (or the nearest of them, when they are closer than ``gap``), and as few
    others, at the polyline's vertices or evenly between them, as keep every
    wedge within ``spacing`` long and the centerline within ``tolerance`` of
    its axis. No other section lies within ``clear`` of the start and of the
    end, so a wedge there can be longer."""
    must = sorted({start, end, *forced})
    merged = [must[0]]
    for arc in must[1:]:  # forced sections closer than gap become one, the ends kept
        if arc - merged[-1] < gap:
            if arc == end:
                merged[-1] = end
            continue
        merged.append(arc)
    if merged[0] != start:
        merged[0] = start
    must = merged
    inner = [float(arc) for arc in arcs if start < arc < end]
    inner_arcs = np.asarray(sorted(set(inner)))
    inner_points = np.array([_at(points, arcs, arc) for arc in inner_arcs]).reshape(-1, 3)
    candidates = sorted(set(must) | set(inner))
    must_set = set(must)
    chosen = [candidates[0]]
    index = 0
    while index < len(candidates) - 1:
        here = candidates[index]
        origin = _at(points, arcs, here)
        best = None
        for next_index in range(index + 1, len(candidates)):
            there = candidates[next_index]
            if there - here > spacing * (1.0 + 1e-9):  # a hair over from rounding still fits
                break
            low, high = np.searchsorted(inner_arcs, here, side="right"), np.searchsorted(inner_arcs, there)
            if high > low:  # the vertices between here and there
                chord = _at(points, arcs, there) - origin
                offsets = inner_points[low:high] - origin
                along = np.clip(offsets @ chord / max(chord @ chord, 1e-300), 0.0, 1.0)
                if np.linalg.norm(offsets - along[:, None] * chord, axis=1).max() > tolerance:
                    break
            best = next_index
            if there in must_set:
                break
        if best is None:  # the next candidate is more than spacing away along a straight stretch
            there = candidates[index + 1]
            parts = math.ceil((there - here) / spacing - 1e-9)
            candidates.insert(index + 1, here + (there - here) / parts)
            best = index + 1
        chosen.append(candidates[best])
        index = best
    # A section too close to the next becomes one with it, unless both are forced.
    stations = [chosen[0]]
    for arc in chosen[1:]:
        if arc - stations[-1] < gap:
            if stations[-1] in must_set and arc not in must_set:
                continue
            if stations[-1] not in must_set:
                stations[-1] = arc
                continue
        stations.append(arc)
    stations = [arc for arc in stations if arc in must_set or start + clear[0] <= arc <= end - clear[1]]
    return np.asarray(stations)


def _piece(points, axes, arcs, stations, long, short, around, start_wall, end_wall, facing=None):
    """Nodes, wedges and ``(wedge, face)`` surface faces of one stretch of a
    fiber with sections at these arc lengths, and for each station of
    ``facing`` (station index -> a direction) the side ``k`` (between its
    outline nodes ``k`` and ``k + 1``) that faces that way. Each section is
    its middle node, then ``around`` nodes counterclockwise seen from the
    next one. A round fiber's facing sections turn, by at most half a side,
    so that a side faces the direction squarely."""
    facing = facing or {}
    centers = np.array([_at(points, arcs, arc) for arc in stations])
    steps = np.diff(centers, axis=0)
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    tangents = np.empty_like(centers)
    tangents[0], tangents[-1] = steps[0], steps[-1]
    tangents[1:-1] = steps[:-1] + steps[1:]
    norms = np.linalg.norm(tangents, axis=1)
    folded = norms < 1.0e-9  # a stretch that turns straight back: square to the one before
    tangents[folded], norms[folded] = steps[np.flatnonzero(folded) - 1], 1.0
    tangents /= norms[:, None]
    oval = long > short * (1.0 + 1.0e-9)
    long_axes = None
    if oval:
        long_axes = np.array([_at(axes, arcs, arc) for arc in stations])
    directions = _section_directions(tangents, long_axes)
    across = np.cross(tangents, directions)
    angles = 2.0 * math.pi * np.arange(around) / around
    sides = {}
    for index, toward in sorted(facing.items()):
        angle = math.atan2(float(toward @ across[index]), float(toward @ directions[index]))
        same = index - 1 in sides and facing[index - 1] is toward  # the other section of the same bond
        if oval:  # an oval's outline keeps its long axis: the side nearest the direction
            middles = angles + math.pi / around
            sides[index] = sides[index - 1] if same else int(np.argmax(np.cos(middles - angle)))
        else:  # turn the section by at most half a side so that a side faces it squarely
            side = round((angle - math.pi / around) / (2.0 * math.pi / around))
            if same:  # the side the bond's other section turned to, whichever way this one rounds
                step = (sides[index - 1] - side) % around
                side += step - around if step > around / 2 else step
            turn = angle - 2.0 * math.pi * (side + 0.5) / around
            turned = math.cos(turn) * directions[index] + math.sin(turn) * across[index]
            directions[index], across[index] = turned, np.cross(tangents[index], turned)
            sides[index] = side % around
    scale = math.sqrt(2.0 * math.pi / (around * math.sin(2.0 * math.pi / around)))  # the polygon's area is the outline's
    outline = scale * np.stack([long * np.cos(angles), short * np.sin(angles)], axis=1)
    rim = (centers[:, None, :] + outline[None, :, :1] * directions[:, None, :]
           + outline[None, :, 1:] * across[:, None, :])
    square = rim.copy()
    for index, wall, step in ((0, start_wall, steps[0]), (-1, end_wall, steps[-1])):
        if wall is None:
            continue
        axis, value = wall
        if abs(step[axis]) >= WALL_COSINE:  # slide the end's nodes along the fiber into the wall
            rim[index] += ((value - rim[index][:, axis]) / step[axis])[:, None] * step
            rim[index][:, axis] = value
            centers[index][axis] = value

    per = around + 1
    segment = np.arange(len(stations) - 1)[:, None]
    k = np.arange(around)[None, :]
    first, second = 1 + k, 1 + (k + 1) % around
    elements = np.stack(
        [segment * per + 0 * k, segment * per + first, segment * per + second,
         (segment + 1) * per + 0 * k, (segment + 1) * per + first, (segment + 1) * per + second],
        axis=2,
    ).reshape(-1, 6)
    for _ in range(2):
        nodes = np.concatenate([centers[:, None, :], rim], axis=1).reshape(-1, 3)
        corners = nodes[elements[:, PENTA_TETRAHEDRA]]  # (wedge, tetrahedron, corner, xyz)
        edges = corners[:, :, 1:] - corners[:, :, :1]
        volumes = np.einsum("...i,...i", edges[..., 0, :], np.cross(edges[..., 1, :], edges[..., 2, :]))
        folded = (volumes <= 0.0).any(axis=1).reshape(len(stations) - 1, around).any(axis=1)
        if not (folded[0] or folded[-1]):
            break
        for index in ([0] if folded[0] else []) + ([-1] if folded[-1] else []):
            rim[index] = square[index]  # a slid end that folds its wedges stays square
    inverted = volumes.sum(axis=1) < 0.0
    elements[inverted] = elements[inverted][:, FLIP_PENTA]
    wedges = np.arange(len(elements))
    outer = [np.stack([wedges, np.full(len(wedges), SIDE_FACE)], axis=1),
             np.stack([wedges[:around], np.full(around, START_FACE)], axis=1),
             np.stack([wedges[-around:], np.full(around, END_FACE)], axis=1)]
    return nodes, elements, np.concatenate(outer), sides


def _bridge_ends(assembly, centerlines, bridges) -> list:
    """``((fiber, arc), (fiber, arc))`` for each bond: where it holds each of
    its two fibers, as an arc length along the centerline."""
    ends = []
    if bridges is True:
        for _, *anchors in _tangle.fem_bond_anchors(assembly):
            pair = []
            for fiber, segment, coordinate in (anchors[:3], anchors[3:]):
                arcs = _arc_lengths(centerlines[fiber])
                pair.append((fiber, arcs[segment] + coordinate * (arcs[segment + 1] - arcs[segment])))
            ends.append(tuple(pair))
        return ends
    for fiber_a, point_a, fiber_b, point_b in bridges:
        pair = []
        for fiber, point in ((int(fiber_a), point_a), (int(fiber_b), point_b)):
            if not 0 <= fiber < len(centerlines):
                raise ValueError(f"a bridge names fiber {fiber}, and the assembly has {len(centerlines)}")
            pair.append((fiber, _nearest_arc(centerlines[fiber], np.asarray(point, dtype=float))))
        ends.append(tuple(pair))
    return ends


def _nearest_arc(points: np.ndarray, point: np.ndarray) -> float:
    """The arc length of the point of the polyline nearest ``point``."""
    start, step = points[:-1], np.diff(points, axis=0)
    t = np.clip(((point - start) * step).sum(axis=1) / np.maximum((step * step).sum(axis=1), 1e-300), 0.0, 1.0)
    k = int(np.argmin(np.linalg.norm(start + t[:, None] * step - point, axis=1)))
    return float(_arc_lengths(points)[k] + t[k] * np.linalg.norm(step[k]))


def _bridge_wedges(nodes, bridge_ends, images, centerlines, semi_axes, around, lengths, periodic) -> np.ndarray:
    """Two binder wedges for each bridge whose two ends both have their
    sides in the mesh: a hexahedron from the side of one fiber's wedge that
    faces the other fiber to the side of the other's that faces back, cut
    into two wedges that keep those sides whole, so the binder shares its
    nodes with both fibers. Bridges that cannot take one are counted and
    left out with a warning."""
    import warnings

    wedges, missing = [], 0
    scale = float(lengths.max())
    for bridge, ((fiber_a, arc_a), (fiber_b, arc_b)) in enumerate(bridge_ends):
        firsts, seconds = images.get((bridge, 0), []), images.get((bridge, 1), [])
        if not firsts or not seconds:
            missing += 1
            continue
        delta = (_at(centerlines[fiber_b], _arc_lengths(centerlines[fiber_b]), arc_b)
                 - _at(centerlines[fiber_a], _arc_lengths(centerlines[fiber_a]), arc_a))
        for axis in range(3):
            if periodic[axis]:
                delta[axis] -= round(delta[axis] / lengths[axis]) * lengths[axis]
        made = False
        for low_a, high_a, side_a, here in firsts:
            # The image of the other end nearest where the bond reaches: the two fibers' middles of the bond
            # are at most a little apart, not a cell length.
            low_b, high_b, side_b, there = min(seconds, key=lambda image: np.linalg.norm(image[3] - here - delta))
            if np.linalg.norm(there - here - delta) > 0.25 * scale:
                continue
            a = np.array([low_a[1 + side_a], high_a[1 + side_a], high_a[1 + (side_a + 1) % around],
                          low_a[1 + (side_a + 1) % around]])
            b = np.array([low_b[1 + side_b], high_b[1 + side_b], high_b[1 + (side_b + 1) % around],
                          low_b[1 + (side_b + 1) % around]])
            best = _bridge_split(nodes, a, b)
            if best is None:
                continue
            wedges += best
            made = True
        missing += not made
    if missing:
        warnings.warn(f"sweep_mesh left out {missing} of {len(bridge_ends)} bridges: their fibers' sides could not "
                      "take one (crowded by other bonds, near a fiber's end or a wall, or overlapping)", stacklevel=3)
    return np.asarray(wedges, dtype=np.int64).reshape(-1, 6)


def _bridge_split(nodes, a, b):
    """Two wedges filling the hexahedron from quadrilateral ``a`` to ``b``
    (each a fiber's side, in order around it), with ``b``'s corners matched
    to ``a``'s across the gap, or None when no way of cutting it leaves both
    wedges right side out with some thickness."""
    corners_a = nodes[a]
    best, best_score = None, 0.0
    for order in (b, b[::-1]):
        for turn in range(4):
            matched = np.roll(order, turn)
            # Matching corners straight across the gap twists the hexahedron least.
            twist = np.linalg.norm(nodes[matched] - corners_a, axis=1).sum()
            for cut in range(4):  # the cut runs from a's edge (cut, cut + 1) to b's opposite edge
                i = [(cut + k) % 4 for k in range(4)]
                a0, a1, a2, a3 = a[i]
                b0, b1, b2, b3 = matched[i]
                pair = [[a0, a3, b3, a1, a2, b2], [a0, b0, b3, a1, b1, b2]]
                volumes = []
                for wedge in pair:
                    corners = nodes[np.asarray(wedge)[PENTA_TETRAHEDRA]]
                    edges = corners[:, 1:] - corners[:, :1]
                    volumes.append(np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])))
                volumes = np.asarray(volumes)
                signs = np.sign(volumes.sum(axis=1))
                if (signs == 0).any():
                    continue
                volumes *= signs[:, None]
                size = np.linalg.norm(corners_a[1] - corners_a[0]) * np.linalg.norm(corners_a[3] - corners_a[0])
                score = volumes.min() / max(size, 1e-300) ** 1.5 / (1.0 + twist / max(size, 1e-300) ** 0.5)
                if volumes.min() > 0.0 and score > best_score:
                    best_score = score
                    best = [list(wedge) if sign > 0 else [wedge[k] for k in FLIP_PENTA]
                            for wedge, sign in zip(pair, signs)]
    return best


def _bond_beams(nodes, bonds, stations_at, centerlines, semi_axes, ratio, origin, lengths, periodic, clip,
                material):
    """The mesh's nodes with one more at the middle of every section a beam
    ends at, and the :class:`Bonds`.

    A bond's beam runs from each image of one anchor in the cell to the
    image of the other as near as the periodic cell allows. When that image
    is outside the cell, the beam stops where it leaves the cell, at a free
    node on the wall, as a fiber does; without ``clip`` it is left out."""
    extra, ends, sections = [], [], []
    end_of = {}  # the first node of a section -> its beam end node
    scale = float(lengths.max())

    def end_node(section, position):
        key = int(section[0])
        if key not in end_of:
            end_of[key] = len(nodes) + len(extra)
            extra.append(position)
            ends.append(end_of[key])
            sections.append(section)
        return end_of[key]

    def wall_point(position, toward):
        """Where the line from ``position`` to ``toward`` leaves the cell, or
        None when that is too near ``position`` for a beam."""
        step = toward - position
        exit = 1.0
        for axis in range(3):
            if step[axis] > 0.0:
                exit = min(exit, (origin[axis] + lengths[axis] - position[axis]) / step[axis])
            elif step[axis] < 0.0:
                exit = min(exit, (origin[axis] - position[axis]) / step[axis])
        return position + exit * step if exit >= 0.05 else None  # on a wall itself: its image's beam is whole

    def wall_node(point):
        extra.append(point)
        return len(nodes) + len(extra) - 1

    beams, radii, orientations, junctions = [], [], [], []
    for junction, first_anchor, second_anchor in bonds:
        (first_fiber, first_arc), (second_fiber, second_arc) = first_anchor, second_anchor
        first = _at(centerlines[first_fiber], _arc_lengths(centerlines[first_fiber]), first_arc)
        second = _at(centerlines[second_fiber], _arc_lengths(centerlines[second_fiber]), second_arc)
        delta = second - first
        for axis in range(3):
            if periodic[axis]:
                delta[axis] -= round(delta[axis] / lengths[axis]) * lengths[axis]
        length = float(np.linalg.norm(delta))
        if length <= 1e-12 * scale:
            continue
        radius = ratio * min(semi_axes[first_fiber][1], semi_axes[second_fiber][1])
        orientation = np.eye(3)[int(np.argmin(np.abs(delta)))]  # the axis most across the beam
        firsts = stations_at.get(first_anchor, [])
        seconds = stations_at.get(second_anchor, [])
        pairs = []
        for section, position in firsts:  # from each image of the first anchor...
            match = [index for index, (_, other) in enumerate(seconds)
                     if np.linalg.norm(other - (position + delta)) <= 1e-9 * scale]
            if match:
                other_section, other_position = seconds[match[0]]
                pairs.append((end_node(section, position), end_node(other_section, other_position)))
            elif clip and (point := wall_point(position, position + delta)) is not None:
                pairs.append((end_node(section, position), wall_node(point)))
        for section, position in seconds:  # ...and from each image of the second whose first is outside
            if clip and not any(np.linalg.norm(other - (position - delta)) <= 1e-9 * scale for _, other in firsts):
                if (point := wall_point(position, position - delta)) is not None:
                    pairs.append((end_node(section, position), wall_node(point)))
        for pair in pairs:
            beams.append(pair)
            radii.append(radius)
            orientations.append(orientation)
            junctions.append(junction)
    if not beams:
        return nodes, None
    return np.concatenate([nodes, np.asarray(extra)]), Bonds(
        beams=np.asarray(beams, dtype=np.int64),
        radii=np.asarray(radii, dtype=float),
        orientations=np.asarray(orientations, dtype=float),
        junction_ids=np.asarray(junctions, dtype=np.int64),
        material=material,
        ends=np.asarray(ends, dtype=np.int64),
        sections=np.asarray(sections, dtype=np.int64).reshape(len(ends), -1),
    )
