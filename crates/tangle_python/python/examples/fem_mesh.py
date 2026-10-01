"""Mesh a few fibers for Nastran and Abaqus: hexahedra from voxels and
tetrahedra from gmsh.

Builds a small cell, periodic in x and y, where two wavy round fibers cross
each other and a flat (oval) fiber lies on top, all touching. Then it writes
four files:

- fibers_hex.bdf and fibers_hex.inp: one 8-node brick per 2 um voxel, with
  touching fibers sharing nodes;
- fibers_tet.bdf and fibers_tet.inp: 10-node tetrahedra that follow each
  fiber's surface, one body per fiber.

The material is PET in SI units. Needs NumPy and gmsh (pip install numpy
gmsh) and runs in seconds on a CPU. See docs/fem_export.md.

Usage: python fem_mesh.py [output_directory]
"""

import math
import sys
from pathlib import Path

import tangle
from tangle import fem
from tangle.units import um

CELL = [200 * um, 200 * um, 80 * um]
POINTS = 21


def along_x(y: float, z: float, wave: float) -> list[list[float]]:
    """A round fiber across the cell in x, waving in y; its ends meet
    across the periodic wall."""
    return [
        [CELL[0] * t, y + wave * math.sin(2 * math.pi * t), z]
        for t in (i / (POINTS - 1) for i in range(POINTS))
    ]


def along_y(x: float, z: float, wave: float) -> list[list[float]]:
    return [
        [x + wave * math.sin(4 * math.pi * t), CELL[1] * t, z]
        for t in (i / (POINTS - 1) for i in range(POINTS))
    ]


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fem_mesh_output")
    output.mkdir(parents=True, exist_ok=True)

    assembly = tangle.Assembly(tangle.Cell(CELL, periodic=[True, True, False]))
    # Round fibers 20 um across: one at z = 30 um under one at z = 50 um.
    assembly.insert(
        tangle.FiberCollection.from_centerlines(
            [along_x(60 * um, 30 * um, 10 * um), along_y(120 * um, 50 * um, 8 * um)],
            tangle.Material("pet", diameter=20 * um),
        )
    )
    # A flat fiber 30 x 12 um lying on the upper round one, corner to corner.
    assembly.insert(
        tangle.FiberCollection.from_centerlines(
            [[[0.0, 0.0, 66 * um], [CELL[0], CELL[1], 66 * um]]],
            tangle.Material("flat pet", diameter=30 * um, thickness=12 * um),
        )
    )
    pet = fem.ElasticMaterial(youngs_modulus=2.5e9, poisson_ratio=0.35, density=1380.0)
    cell_volume = math.prod(CELL)

    hexes = fem.hex_mesh(assembly, voxel_size=2 * um)
    hexes.write_nastran(output / "fibers_hex.bdf", pet)
    hexes.write_abaqus(output / "fibers_hex.inp", pet)
    print(f"{hexes}: solid fraction {hexes.volume / cell_volume:.3f}")

    tets = fem.tet_mesh(assembly, order=2)
    tets.write_nastran(output / "fibers_tet.bdf", pet)
    tets.write_abaqus(output / "fibers_tet.inp", pet)
    print(f"{tets}: solid fraction {tets.volume / cell_volume:.3f}")
    print(f"wrote 4 files to {output}")


if __name__ == "__main__":
    main()
