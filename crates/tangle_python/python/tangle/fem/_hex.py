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
            few scan voxels: then a voxel is binder when binder fills at
            least as much of it as is left empty, leaving out what the
            fibers fill (on the finer grid). Voxels a fiber fills stay fiber;
            the rest become binder elements that share nodes with their
            neighbors, so they bond the fibers they touch.

    Returns:
        A :class:`FemMesh` of ``"hex8"`` elements.
    """
    assembly = as_assembly(assembly)
    counts, phase, owner = voxel_labels(assembly, voxel_size, bond_radius_ratio, binder)
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
    (``phase``, nonzero) and by what (``owner``: a fiber's index plus 1, or 0
    for binder). Binder is the junction bridges of ``bond_radius_ratio`` and
    the voxels of the ``binder`` mask that no fiber fills, with a mask over a
    finer grid first brought to this one."""
    counts, phase, owner = _labels(assembly, voxel_size, bond_radius_ratio)
    nx, ny, nz = counts
    if binder is not None:
        mask = np.asarray(binder, dtype=bool)
        factor = mask.shape[0] // nz if mask.ndim == 3 else 0
        if factor < 1 or mask.shape != (factor * nz, factor * ny, factor * nx):
            raise ValueError(
                f"binder must be a (z, y, x) mask of the {nz} x {ny} x {nx} voxel grid, or of a grid a whole "
                f"number of times finer; got {mask.shape}"
            )
        if factor > 1:
            mask = _coarsened(assembly, voxel_size / factor, bond_radius_ratio, mask, factor)
        extra = mask.ravel() & (phase == 0)  # the fibers win: only voxels no fiber fills
        if extra.any():
            phase, owner = phase.copy(), owner.copy()
            phase[extra], owner[extra] = 1, 0  # filled, and owned by no fiber: binder
    return counts, phase, owner


def _labels(assembly, voxel_size: float, bond_radius_ratio: float | None):
    """TANGLE's own labels of the voxels of :func:`voxel_labels`, before
    any mask."""
    counts, phase, owner, _ = _tangle.fem_voxel_labels(assembly, voxel_size, bond_radius_ratio)
    return counts, np.frombuffer(phase, dtype="<u2"), np.frombuffer(owner, dtype="<u4")


def _coarsened(assembly, fine_size: float, bond_radius_ratio: float | None, mask: np.ndarray, factor: int) -> np.ndarray:
    """A binder ``mask`` over voxels of ``fine_size`` on the grid ``factor``
    times coarser: a coarse voxel is binder when its fine binder voxels are
    at least as many as its empty ones, leaving out those a fiber fills (so
    binder that fills the gap beside a fiber stays binder), and there is
    some."""
    _, phase, owner = _labels(assembly, fine_size, bond_radius_ratio)
    fiber = ((phase != 0) & (owner != 0)).reshape(mask.shape)
    binder = (mask | (phase != 0).reshape(mask.shape)) & ~fiber  # with any junction bridges
    empty = ~(binder | fiber)
    nz, ny, nx = (count // factor for count in mask.shape)

    def per_voxel(fine: np.ndarray) -> np.ndarray:
        return fine.reshape(nz, factor, ny, factor, nx, factor).sum(axis=(1, 3, 5))

    filled, left = per_voxel(binder), per_voxel(empty)
    return (filled > 0) & (filled >= left)
