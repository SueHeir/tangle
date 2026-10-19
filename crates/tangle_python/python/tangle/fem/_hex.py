"""Hexahedral meshes: one 8-node brick per voxel the fibers fill."""

from __future__ import annotations

import numpy as np

from .. import _tangle
from ._mesh import FemMesh
from ._source import as_assembly, binder_name, material_table

# Corner offsets of a voxel in Nastran and Abaqus hexahedron order: the
# bottom face counterclockwise seen from above, then the top face.
CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)


def hex_mesh(assembly, voxel_size: float, *, bond_radius_ratio: float | None = None, binder=None) -> FemMesh:
    """Mesh the fibers with one 8-node hexahedron per voxel they fill.

    The voxels are exactly those of ``export_puma``: a grid that spans the
    cell, where a voxel belongs to the fiber whose surface is closest to its
    center. Fibers that touch share the nodes between their voxels, so the
    mesh is one connected body wherever voxels meet. The surface steps from
    voxel to voxel; a smaller ``voxel_size`` follows it more closely, at the
    cost of eight times the elements for every halving.

    Args:
        assembly: an ``Assembly`` or a recipe's ``RunResult``.
        voxel_size: voxel edge in meters. It must divide every cell length;
            the error suggests one that does.
        bond_radius_ratio: also mesh binder bridges at persistent junctions,
            with this radius as a fraction of the thinner fiber's radius
            (as ``export_puma`` does). Binder elements get fiber id 0 and
            the material ``"binder"``.
        binder: also mesh these voxels as binder: a boolean ``(z, y, x)``
            array over the same voxel grid (``nz, ny, nx`` voxels of
            ``voxel_size`` across the cell), such as a CT scan's binder
            segmented on that grid. It can also be over a grid a whole number
            of times finer, such as the scan's own when ``voxel_size`` is a
            few scan voxels: then the voxels where binder fills the most of
            the room the fibers leave (on the finer grid) become binder, as
            many as hold about as much room as the mask's binder fills, so
            binder too thin to fill half a voxel gathers where it is densest
            rather than dropping out. Voxels a fiber fills stay fiber; the
            rest become binder elements that share nodes with their
            neighbors, so they bond the fibers they touch.

    Returns:
        A :class:`FemMesh` of ``"hex8"`` elements.
    """
    assembly = as_assembly(assembly)
    counts, phase, owner, _ = voxel_labels(assembly, voxel_size, bond_radius_ratio, binder)
    nx, ny, nz = counts
    filled = np.flatnonzero(phase)
    if filled.size == 0:
        raise ValueError("no fiber fills a voxel of the cell; check the voxel size and the fibers")

    # Voxel (i, j, k) has its low corner at grid point (i, j, k) of an
    # (nx + 1) x (ny + 1) x (nz + 1) point grid.
    voxel = np.stack([filled % nx, (filled // nx) % ny, filled // (nx * ny)], axis=1)
    corners = voxel[:, None, :] + CORNERS[None, :, :]
    points = corners[..., 0] + (nx + 1) * (corners[..., 1] + (ny + 1) * corners[..., 2])
    used = np.zeros((nx + 1) * (ny + 1) * (nz + 1), dtype=bool)
    used[points.ravel()] = True
    grid_points = np.flatnonzero(used)
    node_of_point = np.cumsum(used) - 1
    elements = node_of_point[points]

    origin = np.asarray(assembly.cell.origin, dtype=float)
    lengths = np.asarray(assembly.cell.lengths, dtype=float)
    spacing = lengths / np.asarray(counts, dtype=float)
    indices = np.stack(
        [grid_points % (nx + 1), (grid_points // (nx + 1)) % (ny + 1), grid_points // ((nx + 1) * (ny + 1))],
        axis=1,
    )
    nodes = origin + indices * spacing

    fiber_ids = np.asarray(assembly.fiber_ids(), dtype=np.int64)
    materials, fiber_material = material_table(assembly)
    owners = owner[filled].astype(np.int64)
    is_binder = owners == 0
    element_fibers = np.where(is_binder, 0, fiber_ids[np.maximum(owners - 1, 0)])
    element_materials = fiber_material[np.maximum(owners - 1, 0)]
    if is_binder.any():
        materials = (*materials, binder_name(materials))
        element_materials = np.where(is_binder, len(materials) - 1, element_materials)

    # Group the elements by material, then fiber, keeping voxel order inside.
    order = np.lexsort((np.arange(len(filled)), element_fibers, element_materials))
    return FemMesh(
        nodes=nodes,
        elements=elements[order],
        element_type="hex8",
        element_fibers=element_fibers[order],
        element_materials=element_materials[order],
        materials=materials,
        cell_origin=tuple(origin.tolist()),
        cell_lengths=tuple(lengths.tolist()),
    )


def voxel_labels(assembly, voxel_size: float, bond_radius_ratio: float | None, binder):
    """The ``export_puma`` voxels of the cell, with binder: the voxel counts
    ``(nx, ny, nz)``, then for every voxel (x fastest) whether it is filled
    (``phase``, nonzero), by what (``owner``: a fiber's index plus 1, or 0
    for binder), and whether a fiber fills any of it (``fibrous``, which on
    a finer mask's grid includes voxels a fiber fills only part of). Binder
    is the junction bridges of ``bond_radius_ratio`` and the voxels of the
    ``binder`` mask that no fiber fills, with a mask over a finer grid first
    brought to this one by :func:`_coarsened`."""
    counts, phase, owner = _labels(assembly, voxel_size, bond_radius_ratio)
    nx, ny, nz = counts
    fibrous = (phase != 0) & (owner != 0)
    if binder is not None:
        mask = np.asarray(binder, dtype=bool)
        factor = mask.shape[0] // nz if mask.ndim == 3 else 0
        if factor < 1 or mask.shape != (factor * nz, factor * ny, factor * nx):
            raise ValueError(
                f"binder must be a (z, y, x) mask of the {nz} x {ny} x {nx} voxel grid, or of a grid a whole "
                f"number of times finer; got {mask.shape}"
            )
        if factor > 1:
            mask, partly = _coarsened(assembly, voxel_size / factor, bond_radius_ratio, mask, factor, phase, owner)
            fibrous = fibrous | partly
        extra = mask.ravel() & (phase == 0)  # the fibers win: only voxels no fiber fills
        if extra.any():
            phase, owner = phase.copy(), owner.copy()
            phase[extra], owner[extra] = 1, 0  # filled, and owned by no fiber: binder
    return counts, phase, owner, fibrous


def _labels(assembly, voxel_size: float, bond_radius_ratio: float | None):
    """TANGLE's own labels of the voxels of :func:`voxel_labels`, before
    any mask."""
    counts, phase, owner, _ = _tangle.fem_voxel_labels(assembly, voxel_size, bond_radius_ratio)
    return counts, np.frombuffer(phase, dtype="<u2"), np.frombuffer(owner, dtype="<u4")


def _coarsened(assembly, fine_size: float, bond_radius_ratio: float | None, mask: np.ndarray, factor: int,
               phase: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A binder ``mask`` over voxels of ``fine_size`` brought to the grid
    ``factor`` times coarser, whose own labels are ``phase`` and ``owner``,
    and which voxels of that grid a fiber fills any of on the finer one,
    both for every voxel, x fastest.

    Of the coarse voxels that nothing fills, those where binder fills the
    largest share of the room the fibers leave become binder, the fuller of
    equal shares first, until they hold about as much room as the mask's
    binder fills (with any junction bridges). Binder too thin to fill half
    a coarse voxel so gathers where it is densest rather than dropping out,
    and the binder keeps the mask's volume outside the fibers."""
    _, fine_phase, fine_owner = _labels(assembly, fine_size, bond_radius_ratio)
    fiber = ((fine_phase != 0) & (fine_owner != 0)).reshape(mask.shape)
    binder = (mask | (fine_phase != 0).reshape(mask.shape)) & ~fiber  # with any junction bridges
    nz, ny, nx = (count // factor for count in mask.shape)

    def per_voxel(fine: np.ndarray) -> np.ndarray:
        return fine.reshape(nz, factor, ny, factor, nx, factor).sum(axis=(1, 3, 5)).ravel()

    filled, fibers = per_voxel(binder), per_voxel(fiber)
    room = factor**3 - fibers  # the finer voxels no fiber fills
    bridged = (phase != 0) & (owner == 0)  # junction bridges, binder already
    free = np.flatnonzero((phase == 0) & (filled > 0))
    share = filled[free] / room[free]
    ranked = free[np.lexsort((free, -filled[free], -share))]
    held = np.concatenate([[0], np.cumsum(room[ranked])])
    count = int(np.argmin(np.abs(held - (filled.sum() - room[bridged].sum()))))
    coarse = np.zeros(nx * ny * nz, dtype=bool)
    coarse[ranked[:count]] = True
    return coarse, fibers > 0
