# FEM meshes for Nastran and Abaqus

`tangle.fem` turns an assembly into a solid finite-element mesh and writes it
as Nastran bulk data (`.bdf`) or as an Abaqus input file (`.inp`). There are
four meshers, and all return the same `FemMesh`:

| | `hex_mesh` | `tet_mesh` | `sweep_mesh` | `conformal_mesh` |
|---|---|---|---|---|
| Elements | 8-node hexahedra (`CHEXA`, `C3D8`) | 4- or 10-node tetrahedra (`CTETRA`, `C3D4` or `C3D10`) | 6-node wedges (`CPENTA`, `C3D6`) | 4- or 10-node tetrahedra |
| Fiber surface | voxel steps, like a CT scan | follows the round or oval surface (10-node elements curve with it) | a hexagon of the fiber's area, swept along the centerline | follows the surface, faceted at the element size |
| Size | many elements | many elements | 6 wedges every few diameters | coarse with `elements_around=5` or `6` |
| Fibers that touch | share nodes, so they are bonded | separate bodies, so the solver needs contact or glue | separate bodies | share nodes where they overlap |
| Bonds at junctions | binder voxels (`bond_radius_ratio`) | binder voxels, tied to the fibers | one beam per bond (`bond_radius_ratio`), or two binder wedges on the fibers' own sides (`bridges`) | a smooth bridge per bond, sharing nodes with the fibers |
| Binder mask (`binder`) | shares nodes with the fibers | tetrahedra tied to the fibers | wedges tied to the fibers | smooth tetrahedra sharing nodes with the fibers |
| Needs | NumPy | NumPy and [gmsh](https://gmsh.info) | NumPy | NumPy, gmsh and SciPy |

For a Nastran model of a CT scan with its binder, `conformal_mesh` with the
scan's binder mask is the one to start with: the binder and the fibers share
nodes, so the `.bdf` needs no `MPC` ties, and the mesh is coarse. See
[Conformal tetrahedra with binder](#conformal-tetrahedra-with-binder).

Install the extras with `pip install numpy gmsh scipy`, or
`pip install "tangle[fem]"`. gmsh is GPL software that TANGLE calls but does
not ship; `hex_mesh`, `sweep_mesh` and the writers work without it.

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

### Bonds as binder wedges

`bridges` makes each bond solid binder that shares the fibers' nodes, instead
of a beam: `bridges=True` for the assembly's persistent junctions, or a list
of `(fiber_a, point_a, fiber_b, point_b)` (fiber indices, points in meters on
the centerlines), such as `fem.fit_bonds(fit)` for a CT fit's bonds. Each
fiber gets a section half a side on either side of the bond, turned so a flat
side faces the other fiber, and two binder wedges fill the space between the
two facing sides. The binder's quadrilateral faces are the fibers' own wedge
faces, so there are no ties and no beams, and the mesh stays as coarse as the
fibers'.

```python
mesh = fem.sweep_mesh(assembly, bridges=True)
```

Each bridge is one side of the hexagon wide, and only as thick as the gap
between the two hexagons' flat sides, about a tenth of a radius where fibers
touch. It bonds the fibers, but it is not the shape of real binder; for a
scan's webbed binder use `conformal_mesh` with the mask. Bonds close together
along a fiber share its pair of sections, each on its own side, and a bond
near a fiber's end moves inward until its sides fit. Bonds that still cannot
take a bridge (their side is taken, a periodic wall cuts it, or the two
fibers' sides overlap) are left out with a warning that counts them. Not with
`bond_radius_ratio`.

## Conformal tetrahedra with binder

`conformal_mesh(assembly, bonds=None, *, binder=None, voxel_size=None,
elements_around=8, order=1, region=None, min_quality=0.02,
bond_radius_ratio=0.5)` meshes the fibers and their binder as one body in one
gmsh model, so every fiber-binder face is shared: the binder's tetrahedra use
the fibers' nodes, and the `.bdf` has no `MPC` ties.

### From a binder mask

With a binder mask, the binder is the mask's own shape. gmsh meshes the
fibers and the space between them together; the tetrahedra of that space
whose middles lie in the mask (smoothed by a voxel, read at the level that
keeps its volume) become binder, and the rest are dropped, as is any piece of
binder that no fiber holds. The binder's outer nodes then move onto the
smoothed mask's surface. There are no voxel steps; the binder's outer surface
is faceted at the element size.

```python
import tifffile
from tangle import fem
from tangle.ct import load_fit

fit = load_fit("fit/fit.json")
binder = tifffile.imread("fit/binder.tif") > 0  # or fit.binder from find_fibers with NetworkSettings(binder=True)
mesh = fem.conformal_mesh(fit.to_assembly(), binder=binder, voxel_size=fit.voxel_size, elements_around=6)
print(mesh)
fiber = fem.ElasticMaterial(youngs_modulus=2.5e9, poisson_ratio=0.35, density=1380.0)
glue = fem.ElasticMaterial(youngs_modulus=3.0e9, poisson_ratio=0.35, density=1200.0)
mesh.write_nastran("scan.bdf", {name: glue if name.startswith("binder") else fiber for name in mesh.materials})
mesh.write_vtu("scan.vtu")  # binder is fiber id 0 and its own material
```

- `binder` is a boolean `(z, y, x)` array over the whole cell, with voxels of
  `voxel_size` meters: a CT fit's `binder.tif` or `fit.binder`, or
  `scan.binder_occupancy > 0.5` for a synthetic scan. The fibers can be the
  fit's, coarse-grained or relaxed: the binder stays where the scan shows it,
  and wherever a relaxed fiber has moved into the mask, the fiber wins.
- `elements_around=5` or `6` keeps the mesh coarse. On a synthetic webbed
  scan of 160 um (24 fibers, 7.7% binder), `elements_around=6` gives about
  5,000 nodes and 18,700 tetrahedra, 7,000 of them binder, in 8 seconds, and
  keeps 96% of the mask's binder volume; `sweep_mesh(binder=...)` of the same
  cell has 407,000 nodes. At 8 and above gmsh's frontal mesher can crash on a
  whole cell.
- The fibers' volume comes out a few percent under their true volume at
  these settings, since their faceted surfaces cut the round ones.

### From bonds

Without a mask, each bond becomes a round bridge from one fiber's centerline
to the other's, of radius `bond_radius_ratio` times the thinner fiber's
smaller radius (below 1). `bonds` defaults to the assembly's junctions; for a
CT fit pass `fem.fit_bonds(fit)`. Bonds across periodic walls go to the
nearest image of the other fiber. Where two fibers touch, a bridge has a hole
down its middle, so its binder never thins to a knife-edge; the fibers share
nodes there instead.

```python
mesh = fem.conformal_mesh(result.assembly, bond_radius_ratio=0.9, elements_around=6)
```

### Quality

`conformal_mesh` aims for no tetrahedron flatter than 0.1 in signed inverse
condition number (`fem.tet_quality(mesh)`, 1 for a regular tetrahedron).
Slivers gmsh leaves where fibers nearly touch are mended: short edges
collapsed, tetrahedra flipped, corners slid along their surfaces. It warns
with their places if any stay under 0.1, and raises rather than return one
flatter than `min_quality` (0.02 by default; Nastran itself only refuses
tetrahedra that are inside out or flat). A handful between the two is
normal on a large cell and does not spoil a linear solve; lower
`min_quality` to 0.005 if one run stops on a single tetrahedron.

The whole region goes into one gmsh model, so this is slower than
`tet_mesh`, which meshes a fiber at a time. Try a small `region=(low, high)`
(corners in meters) first on a new scan.

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
- `conformal_mesh` binder is only as fine as the elements: a binder sheet
  much thinner than an element is lost or thickened, and the binder volume
  should be checked against the mask's.
- The `MPC` ties glue a binder node to the flat corner triangle of a 10-node
  tetrahedron's face, not to its curved face.
- Element and node ids must fit Nastran's 8-column fields, so a mesh can have
  at most 99,999,999 of each.
