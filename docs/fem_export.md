# FEM meshes for Nastran and Abaqus

`tangle.fem` turns an assembly into a solid finite-element mesh and writes it
as Nastran bulk data (`.bdf`) or as an Abaqus input file (`.inp`). There are
three meshers, and all return the same `FemMesh`:

| | `hex_mesh` | `tet_mesh` | `sweep_mesh` |
|---|---|---|---|
| Elements | 8-node hexahedra (`CHEXA`, `C3D8`) | 4- or 10-node tetrahedra (`CTETRA`, `C3D4` or `C3D10`) | 6-node wedges (`CPENTA`, `C3D6`) |
| Fiber surface | voxel steps, like a CT scan | follows the round or oval surface (10-node elements curve with it) | a hexagon of the fiber's area, swept along the centerline |
| Size | many elements | many elements | 6 wedges every few diameters |
| Fibers that touch | share nodes, so they are bonded | separate bodies, so the solver needs contact or glue | separate bodies |
| Bonds at junctions (`bond_radius_ratio`) | binder voxels | binder voxels, tied to the fibers | one beam per bond, tied to the two fibers' cross-sections |
| Binder voxels (`binder`) | share nodes with the fibers | tetrahedra tied to the fibers | wedges tied to the fibers |
| Needs | NumPy | NumPy and [gmsh](https://gmsh.info) | NumPy |

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
hexes.write_vtu("felt_hex.vtu")  # to look at in ParaView

tets = fem.tet_mesh(assembly, order=2)
tets.write_nastran("felt_tet.bdf", pet)
tets.write_abaqus("felt_tet.inp", pet)
```

[`examples/fem_mesh.py`](../crates/tangle_python/python/examples/fem_mesh.py)
builds a small assembly and writes all four files in a few seconds, without a
GPU.

## Hexahedra from voxels

`hex_mesh(assembly, voxel_size, *, bond_radius_ratio=None, binder=None)` puts one 8-node
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
- `binder` meshes other voxels as binder too: a boolean `(z, y, x)` array over
  the same grid (`nz, ny, nx` voxels across the cell), such as the binder that
  `tangle.ct.find_fibers` finds in a CT scan (`fit.binder`, with the cell of
  `fit.to_assembly()` and `voxel_size=fit.voxel_size`). Voxels a fiber fills
  stay fiber; the others become binder bricks, which share nodes with the
  fibers they touch and so bond them.
- The mask can also be over a grid a whole number of times finer than the
  mesh's, such as the scan's own with `voxel_size=4 * fit.voxel_size`. Of the
  mesh voxels no fiber fills, those where binder fills the largest share of
  the room the fibers leave (on the finer grid) then become binder, as many
  as hold about as much room as the mask's binder fills. The binder so keeps
  its volume outside the fibers, and binder too thin to fill half a mesh
  voxel gathers where it is densest instead of dropping out.

## Tetrahedra from gmsh

`tet_mesh(assembly, element_size=None, *, order=1, elements_around=None,
clip_to_cell=True, voxel_size=None, bond_radius_ratio=None, binder=None,
workers=None, fiber_time_limit=None, progress=True, verbose=False)` builds each fiber as a solid through a round or oval
cross-section at every centerline vertex, straight from one vertex to the
next as TANGLE's fibers are, then lets gmsh fill it with tetrahedra. At a
bend the section halves the angle between the two stretches, so they meet as
mitered pipes do. Oval fibers keep their long axis where TANGLE puts it.
Fiber ends are flat.

- Every fiber is its own body. Fibers that touch, or overlap by TANGLE's small
  allowed penetration, do not share nodes. Connect them in the solver: contact
  for loose fibers, glued contact for bonded ones.
- `elements_around` sets each fiber's element size: the perimeter of its
  cross-section divided by this many edges, so thin and flat fibers get
  smaller elements than thick ones. The default is 16 for 4-node tetrahedra,
  whose flat faces then keep a round fiber's volume within about 2%, and 6 for
  10-node tetrahedra. Halving it gives about a seventh of the elements.
  `element_size`, when given, caps the element edge in every fiber.
- `order=2` writes 10-node tetrahedra. They avoid the stiffness of 4-node
  tetrahedra in bending, which matters for thin fibers. Their mid-edge nodes
  on a fiber's surface sit on that surface, so their faces there curve with
  the fiber: the default of 6 around keeps a round fiber's volume within
  about 1%, and 8 around within about 0.5%, closer than 16 flat faces do with
  a small share of their elements. Where curving would fold an element (its
  Jacobian dropping anywhere below a tenth of its straight-sided self's), its
  edges bow less, or not at all, so every element is valid for the solver.
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
- The fibers are meshed side by side, each in one of `workers` gmsh
  processes (by default one per CPU core), and put together in fiber order,
  so the mesh is the same however many there are. The processes are plain
  Python processes that `tet_mesh` starts itself, so a script that calls it
  needs no `if __name__ == "__main__":` guard. With `workers=1` the fibers
  are meshed one after another in your own process, in a gmsh session
  `tet_mesh` starts and stops, so call it while gmsh is not initialized.
- Once a mesh has taken 10 s, `tet_mesh` prints how far it has got, and
  again every 30 s, naming the slowest fiber still being meshed once that
  fiber has taken 30 s; `progress=False` keeps it quiet. `fiber_time_limit`
  stops the mesh with an error naming any fiber that takes longer than that
  many seconds.

### Binder in a tetrahedral mesh

`binder`, `bond_radius_ratio` and `voxel_size` work as in `hex_mesh`: the
binder is the same voxels, those of the mask or of the junction bridges that
no fiber fills, on the grid of `voxel_size`. Each binder voxel becomes six
tetrahedra (10-node ones with `order=2`) with fiber id 0 and the material
`"binder"`, so the binder's surface steps from voxel to voxel while the
fibers stay round. With a mask, `voxel_size` can be left out: the binder's
voxels are then a whole number of the mask's, as near as that allows to the
fibers' element size (their median), so the binder is about as coarse as the
fibers.

- The binder is one body of its own: its tetrahedra share nodes with each
  other but not with the fibers. `mesh.ties` holds one `Tie`, `BINDER_TIE`,
  from the binder's faces next to a voxel a fiber fills (or fills part of, on
  a finer mask's grid) to the faces of the fibers' surfaces, and
  `write_abaqus` writes it as a `*TIE` that glues them. Its position
  tolerance is twice the voxel size plus twice the depth a flat element face
  on a fiber's surface can have, since a binder face next to such a voxel
  lies within its diagonal, about 1.7 voxels, of the fiber's round surface.
  Binder nodes farther away stay free.
- The Nastran file glues the binder with `MPC` equations (see below).
- The voxels a fiber fills are those of `hex_mesh`, whose fibers end in half
  spheres. `tet_mesh` ends them flat, so binder next to a fiber's end can
  stop short of it: within the tolerance it is tied across the gap, and
  beyond it stays free.
- `voxel_size` must divide every cell length, and binder needs
  `clip_to_cell=True`, since it fills the cell.
- Six tetrahedra to a voxel add up: with `order=2`, binder has 8 to 12 nodes
  per voxel, the more the thinner it is, so binder at a CT scan's voxel size
  can have more nodes than all the fibers. The fibers do not depend on
  `voxel_size`, so pass the scan's mask as it is and leave `voxel_size` out:

  ```python
  tets = fem.tet_mesh(fit.to_assembly(), order=2, binder=fit.binder)
  ```

  A larger `voxel_size`, a whole number of scan voxels, makes the binder
  coarser still: each doubling gives about an eighth of its elements. Binder
  beside a fiber fills whole voxels, so coarse binder reaches into the
  fibers, by up to about half a voxel. Compare `mesh.element_volumes()` of
  the binder (fiber id 0) at two sizes to see what a coarser one changes.

## Swept wedges

`sweep_mesh(assembly, *, nodes_around=6, max_length=4.0, bend_tolerance=0.25,
clip_to_cell=True, bond_radius_ratio=None, binder=None, voxel_size=None)` is
the lightest mesh, for large cells in Nastran. Each fiber gets a cross-section
of `nodes_around` nodes on its outline and one in its middle, 7 by default,
at both ends and every so often along its centerline, and six-node wedges
between one section and the next: 6 wedges and 7 nodes per section.

- The section is a polygon on the fiber's round or oval outline, scaled so its
  area is the fiber's. A hexagon so has a round fiber's bending stiffness
  within 1%, and a straight fiber's volume is exact.
- Sections are at most `max_length` fiber diameters apart (the mean of an
  oval's two widths), and closer only where the fiber bends: between two
  sections the centerline stays within `bend_tolerance` of a radius of the
  straight line from one to the other. A straight fiber 40 diameters long is
  60 wedges.
- A single linear wedge across the fiber is stiffer in bending than the fiber
  itself, the more so the longer the wedges. Compare a bending case against
  a finer `max_length` or against `tet_mesh(order=2)` before trusting
  bending results.
- With `clip_to_cell=True` the centerlines are cut at the cell walls, with
  their periodic images on periodic axes, and a cut end lies in the wall, so
  its nodes are in the `XMIN` to `ZMAX` sets. The next section stays beyond
  how far the cut end's outline slides along the fiber, so the end wedge of
  a steep cut can be longer than `max_length`. A fiber that crosses a wall
  within about 15 degrees of along it, or whose cut end has a bond's section
  or the stretch's other end too close, ends square to itself instead.
- Every fiber is its own body, as in `tet_mesh`.

### Bonds as beams

`bond_radius_ratio` turns each bond of a persistent junction (see
[fiber bonds](fiber_bonds.md)) into one round beam, with that radius as a
fraction of the thinner fiber's radius. A section lies at the bond's anchor
on each fiber, and the beam joins two nodes at the middles of those
sections. Each end node belongs to no wedge; it follows its section's 7 nodes
as their mean (Nastran `RBE3`, Abaqus distributing `*COUPLING`), so the beam
carries force and moment into the fiber without stiffening it. The beams use
the material `"binder"`.

The anchors come from the assembly's current fiber shape, so meshing a
relaxed result puts the bonds where the relaxed fibers are. A bond that
crosses a cell wall stops at a free node on the wall on each side, as the
fibers do (unless its anchor is on the wall itself, where the image on the
other side has the whole beam); with `clip_to_cell=False` such bonds are left
out. `mesh.bonds`
(a `Bonds`) has the beams, their radii and the junction of each.

### Binder voxels

`binder` and `voxel_size` work as in `tet_mesh`, except that each binder voxel
becomes two wedges. Without `voxel_size`, the binder's voxels are a whole
number of the mask's, as near as that allows to the fibers' smaller radius,
so a CT scan's binder is meshed about as coarsely as the fibers. The binder
shares no nodes with the fibers; `mesh.ties` glues it to them, as `MPC`
equations in Nastran and a `*TIE` in Abaqus.

```python
fit = ...  # a tangle.ct fit with its binder mask
mesh = fem.sweep_mesh(fit.to_assembly(), binder=fit.binder)
mesh.write_nastran("scan.bdf")  # add MAT1 1 (fiber) and 2 (binder), or pass them
```

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
- `CHEXA`, `CTETRA` or `CPENTA` elements;
- one `PSOLID` per fiber, numbered by its TANGLE fiber id, and the next id for
  binder;
- one `MAT1` per material, numbered from 1 in the order of `mesh.materials`;
- with a tie (binder in `tet_mesh` or `sweep_mesh`), `MPC` equations in set 1:
  each binder node next to a fiber follows the closest point of the closest
  fiber face, as the weighted mean of that face's corners, in x, y and z.
  `mesh.tie_constraints(tie)` gives the same equations as arrays. Select them
  with `MPC = 1` in the case control, and keep `SPC`s off the tied binder
  nodes, which move with the fibers. Binder nodes beyond the tolerance stay
  free;
- with bonds (`sweep_mesh`), a `PBAR` per bond radius (a round bar's area,
  bending and torsion constants), a `CBAR` per bond, and an `RBE3` per beam
  end that makes it follow its fiber's cross-section.

Import it into a pre-processor (Femap, Patran, HyperMesh) or include it in a
deck that has your loads and constraints:

```
SOL 101
CEND
SUBCASE 1
  SPC = 1
  MPC = 1
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
  nodes on each cell face, ready for boundary conditions;
- for a tetrahedral mesh with binder, the surfaces `BINDER_TIE_SECONDARY` (the
  binder's faces next to fibers) and `BINDER_TIE_MAIN` (the fibers' surfaces)
  and a `*TIE` named `BINDER_TIE` that glues the first to the second, with
  `ADJUST=NO` so no node moves.

The elements are `C3D8`, `C3D4`, `C3D10` or `C3D6`. Bonds of a swept mesh are
`B31` beams in sets `BONDS_<n>` (all in `BONDS`), each set with a round
`*BEAM SECTION`, and a distributing `*COUPLING` per beam end ties it to the
nodes of its fiber's cross-section (node sets `BOND_SECTION_<n>`). For bending of hexahedral fibers
`element_type="C3D8I"` is usually better than `C3D8`.

### ParaView

`write_vtu(path, *, scale=1.0)` writes the mesh as a VTK unstructured grid
(`.vtu`) that ParaView opens directly, with each element's TANGLE fiber id
(`fiber_id`, 0 for binder) and material (`material`, an index into
`mesh.materials`) as cell data; bond beams are lines with fiber id 0. Color by `fiber_id` to see the fibers, or
apply a Threshold on it from 0 to 0 to see only the binder. ParaView cannot
read the `.bdf` itself: its Nastran reader does not read large-field `GRID*`
nodes.

## Units

TANGLE works in meters. Give material values in the unit system the file is
written in:

| `scale` | Length | Force | Stress, `youngs_modulus` | `density` |
|---|---|---|---|---|
| `1.0` (default) | m | N | Pa | kg/m³ |
| `1e3` | mm | N | MPa | t/mm³ (1380 kg/m³ is `1.38e-9`) |

## Checking a mesh

`mesh.volume` and `mesh.element_volumes()` give element volumes, following the
curved faces of 10-node tetrahedra, and `mesh.node_sets()` the face node sets. Dividing `mesh.volume` by the cell
volume gives the solid fraction, which should be close to the
`voxel_volume_fraction` of `export_puma` at a fine voxel size. To look at a
mesh, write it with `write_vtu` and open it in ParaView.

## Limits

- Orthorhombic cells only, as for `export_puma`.
- The `MPC` ties glue a binder node to the flat corner triangle of a 10-node
  tetrahedron's face, not to its curved face.
- Element and node ids must fit Nastran's 8-column fields, so a mesh can have
  at most 99,999,999 of each.
