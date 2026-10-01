"""Finite-element meshes of TANGLE fibers for Nastran and Abaqus.

Two meshers return the same :class:`FemMesh`:

``hex_mesh(assembly, voxel_size)``
    One 8-node hexahedron per voxel the fibers fill, on the ``export_puma``
    grid. Fibers that touch share nodes.
``tet_mesh(assembly, element_size)``
    Tetrahedra that follow each fiber's round or oval surface, meshed with
    gmsh (``pip install gmsh``). Each fiber is its own body.

Write a mesh with ``mesh.write_nastran("fibers.bdf")`` (bulk data: ``GRID``,
``CHEXA`` or ``CTETRA``, ``PSOLID``, ``MAT1``) or
``mesh.write_abaqus("fibers.inp")``. Pass an :class:`ElasticMaterial` to fill
in the material. The module needs NumPy; ``tet_mesh`` also needs gmsh.
See ``docs/fem_export.md``.
"""

from ._hex import hex_mesh
from ._mesh import ElasticMaterial, FemMesh
from ._tet import tet_mesh

__all__ = ["ElasticMaterial", "FemMesh", "hex_mesh", "tet_mesh"]
