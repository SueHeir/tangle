"""Finite-element meshes of TANGLE fibers for Nastran and Abaqus.

Three meshers return the same :class:`FemMesh`:

``hex_mesh(assembly, voxel_size)``
    One 8-node hexahedron per voxel the fibers fill, on the ``export_puma``
    grid. Fibers that touch share nodes. ``binder=`` adds binder voxels.
``tet_mesh(assembly, element_size)``
    Tetrahedra that follow each fiber's round or oval surface, meshed with
    gmsh (``pip install gmsh``). Each fiber is its own body. ``binder=``
    adds binder voxels as tetrahedra, tied to the fibers by a :class:`Tie`.
``sweep_mesh(assembly)``
    The lightest: six wedges between hexagonal cross-sections up to four
    diameters apart. ``bond_radius_ratio=`` adds each junction's bond as a
    beam (:class:`Bonds`); ``binder=`` adds binder voxels as wedges, tied to
    the fibers.

Write a mesh with ``mesh.write_nastran("fibers.bdf")`` (bulk data: ``GRID``,
``CHEXA``, ``CTETRA`` or ``CPENTA``, ``PSOLID``, ``MAT1``, and ``MPC`` ties
and bond beams) or
``mesh.write_abaqus("fibers.inp")``. Pass an :class:`ElasticMaterial` to fill
in the material. ``mesh.write_vtu("fibers.vtu")`` writes a copy to look at in
ParaView. The module needs NumPy; ``tet_mesh`` also needs gmsh.
See ``docs/fem_export.md``.
"""

from ._hex import hex_mesh
from ._mesh import Bonds, ElasticMaterial, FemMesh, Tie
from ._sweep import sweep_mesh
from ._tet import tet_mesh

__all__ = ["Bonds", "ElasticMaterial", "FemMesh", "Tie", "hex_mesh", "sweep_mesh", "tet_mesh"]
