# FEM meshes for Nastran and Abaqus

`tangle.fem` turns an assembly into a solid finite-element mesh and writes it
as Nastran bulk data (`.bdf`) or as an Abaqus input file (`.inp`). There are
two meshers, and both return the same `FemMesh`:

| | `hex_mesh` | `tet_mesh` |
|---|---|---|
| Elements | 8-node hexahedra (`CHEXA`, `C3D8`) | 4- or 10-node tetrahedra (`CTETRA`, `C3D4` or `C3D10`) |
| Fiber surface | voxel steps, like a CT scan | follows the round or oval surface |
| Fibers that touch | share nodes, so they are bonded | separate bodies, so the solver needs contact or glue |
| Binder at junctions | yes, with `bond_radius_ratio` | no |
| Needs | NumPy | NumPy and [gmsh](https://gmsh.info) |

Install the extras with `pip install numpy gmsh`, or `pip install "tangle[fem]"`.
gmsh is GPL software that TANGLE calls but does not ship; `hex_mesh` and the
writers work without it.

## Quick start

```python
import tangle
from tangle import fem
from tangle.units import um

assembly = result.assembly  # a recipe's result, or any tangle.Assembly
pet = fem.ElasticMaterial(youngs_modulus=2.5e9, poisson_ratio=0.35, density=1380.0)

hexes = fem.hex_mesh(assembly, voxel_size=2 * um)
hexes.write_nastran("felt_hex.bdf", pet)
hexes.write_abaqus("felt_hex.inp", pet)

tets = fem.tet_mesh(assembly, order=2)
tets.write_nastran("felt_tet.bdf", pet)
tets.write_abaqus("felt_tet.inp", pet)
```

[`examples/fem_mesh.py`](../crates/tangle_python/python/examples/fem_mesh.py)
builds a small assembly and writes all four files in a few seconds, without a
GPU.

## Hexahedra from voxels

`hex_mesh(assembly, voxel_size, *, bond_radius_ratio=None)` puts one 8-node
brick in every voxel the fibers fill. The voxels are exactly those of
`export_puma`: the grid spans the cell, a voxel is filled when its center lies
inside a fiber, and it belongs to the fiber whose surface is closest. So the
voxel size must divide every cell length, and the error suggests one that
does.

- Neighboring bricks share nodes, including bricks of two fibers that touch.
  The mesh is one body, bonded wherever the voxels of two fibers meet. Fibers
  closer than about one voxel can end up bonded too.
- Bricks that meet only along an edge or at a corner form hinges, as in any
  voxel mesh. They are common where fibers cross at a grazing angle and
  shrink as the voxel size does.
- The surface steps from voxel to voxel. Aim for at least 6 to 8 voxels across
  the thinnest fiber; every halving of the voxel size gives eight times the
  elements.
- `bond_radius_ratio` also meshes binder bridges at persistent junctions (see
  [fiber bonds](fiber_bonds.md)). Binder bricks get fiber id 0 and the
  material `"binder"`.

## Tetrahedra from gmsh

`tet_mesh(assembly, element_size=None, *, order=1, elements_around=16,
clip_to_cell=True, verbose=False)` builds each fiber as a smooth solid swept
through a cross-section at every centerline vertex, then lets gmsh fill it
with tetrahedra. Oval fibers keep their long axis where TANGLE puts it. Fiber
ends are flat.

- Every fiber is its own body. Fibers that touch, or overlap by TANGLE's small
  allowed penetration, do not share nodes. Connect them in the solver: contact
  for loose fibers, glued contact for bonded ones.
- `elements_around` sets each fiber's element size: the perimeter of its
  cross-section divided by this many edges, so thin and flat fibers get
  smaller elements than thick ones. The default of 16 keeps a round fiber's
  volume within about 2%. `element_size`, when given, caps the element edge in
  every fiber.
- `order=2` writes 10-node tetrahedra with straight edges. They avoid the
  stiffness of 4-node tetrahedra in bending, which matters for thin fibers.
- With `clip_to_cell=True` the fibers are cut at the cell walls, with their
  periodic images on periodic axes, so the mesh fills the same box as
  `hex_mesh`. A fiber that only touches a wall, such as one resting on the
  floor of the cell, is not cut there (OpenCASCADE cannot cut a solid that
  touches the cutting face), so it can stick out by up to 5% of its
  thickness. With `False` each fiber is meshed whole along its unwrapped
  centerline.
- Where a cell wall slices a fiber lengthwise, the cut leaves a thin edge
  that the default size cannot always mesh. gmsh then retries that fiber with
  elements refined by the surface curvature, so such fibers get several times
  more elements.
- A fiber that bends tighter than its own radius can make the solid fail; the
  error names the fiber.
- `tet_mesh` starts and stops its own gmsh session, so call it while gmsh is
  not initialized.

## Writing the files

`write_nastran(path, material=None, *, scale=1.0)` and
`write_abaqus(path, material=None, *, scale=1.0, element_type=None)` take one
`ElasticMaterial` for everything, or a dict from material name to
`ElasticMaterial` (the names are in `mesh.materials`). A material left out is
not written, so the solver stops until you define it rather than running with
a made-up value. `scale` multiplies the coordinates; `1e3` writes millimeters.

### Nastran

The `.bdf` holds bulk data only:

- `GRID*` nodes in large-field format (10 significant digits);
- `CHEXA` or `CTETRA` elements;
- one `PSOLID` per fiber, numbered by its TANGLE fiber id, and the next id for
  binder;
- one `MAT1` per material, numbered from 1 in the order of `mesh.materials`.

Import it into a pre-processor (Femap, Patran, HyperMesh) or include it in a
deck that has your loads and constraints:

```
SOL 101
CEND
SUBCASE 1
  SPC = 1
  LOAD = 2
BEGIN BULK
INCLUDE 'felt_hex.bdf'
$ SPC1, FORCE, ... entries for your boundary conditions
ENDDATA
```

### Abaqus

[Abaqus](https://www.3ds.com/products/simulia/abaqus) is a commercial
finite-element solver from Dassault Systèmes; the open-source CalculiX reads
the same `.inp` format. The file is a flat input file without parts, which
Abaqus/CAE imports as an orphan mesh:

- `*NODE`, then `*ELEMENT` blocks with one element set per fiber (`FIBER_<id>`)
  and `BINDER`;
- `MATERIAL_<NAME>` element sets, each with a `*SOLID SECTION`, and the
  `*MATERIAL` definitions;
- `*NSET` node sets `XMIN`, `XMAX`, `YMIN`, `YMAX`, `ZMIN` and `ZMAX` for the
  nodes on each cell face, ready for boundary conditions.

The elements are `C3D8`, `C3D4` or `C3D10`. For bending of hexahedral fibers
`element_type="C3D8I"` is usually better than `C3D8`.

## Units

TANGLE works in meters. Give material values in the unit system the file is
written in:

| `scale` | Length | Force | Stress, `youngs_modulus` | `density` |
|---|---|---|---|---|
| `1.0` (default) | m | N | Pa | kg/m³ |
| `1e3` | mm | N | MPa | t/mm³ (1380 kg/m³ is `1.38e-9`) |

## Checking a mesh

`mesh.volume` and `mesh.element_volumes()` give element volumes, and
`mesh.node_sets()` the face node sets. Dividing `mesh.volume` by the cell
volume gives the solid fraction, which should be close to the
`voxel_volume_fraction` of `export_puma` at a fine voxel size. To look at a
mesh in ParaView, convert it with [meshio](https://github.com/nschloe/meshio)
(not a TANGLE dependency): `meshio.read("felt_hex.inp").write("felt_hex.vtu")`.

## Limits

- Orthorhombic cells only, as for `export_puma`.
- `tet_mesh` does not mesh binder bridges.
- Element and node ids must fit Nastran's 8-column fields, so a mesh can have
  at most 99,999,999 of each.
