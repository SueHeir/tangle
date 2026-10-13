# LAMMPS BPM files in and out

`tangle.lammps` reads the fibers of a LAMMPS data file for the bonded-particle
model (BPM), relaxes them in TANGLE so that no two fibers overlap, and writes
the data file again with the spheres where the relaxation put them. Only their
positions change: atom IDs, types, bonds, coefficients and every other section
are copied as they were, so the input script that read the old file reads the
new one. It is plain Python and needs no NumPy.

## Quick start

```python
from tangle import lammps

data = lammps.read_data("fibers.data", periodic="p p f")  # the input script's boundary
print(data)        # LammpsData('fibers.data', atom_style='bpm/sphere', fibers=120, ...)
print(data.notes)  # what was left alone, and why

run = data.relax()  # on the GPU; tangle.RelaxationSettings(backend="cpu") runs without one
print(run.converged, run.max_penetration)
moved = data.write_data("fibers_relaxed.data", run)  # how many spheres moved
```

Then point the input script at the new file:

```
atom_style bpm/sphere
boundary p p f
read_data fibers_relaxed.data
```

## What it reads

- **Atom styles.** `bpm/sphere` (`atom-ID molecule-ID atom-type diameter
  density x y z`), which `bond_style bpm/rotational` needs, and `bond`
  (`atom-ID molecule-ID atom-type x y z`), which `bpm/spring` uses. Also
  `sphere`, `angle`, `molecular`, `full`, `charge`, `atomic` and hybrids of
  them, such as `hybrid sphere bond`. The style comes from `atom_style=`, else
  from the comment after the section keyword (`Atoms # bpm/sphere`, as
  `write_data` writes it), else it is `bpm/sphere`. Rows may end with image
  flags.
- **Diameters.** From the file's diameter column, or from `diameter=`, which
  takes its place: one value, or one per atom type (`{1: 1e-5, 2: 2e-5}`, or
  by type label). A style without a diameter column, such as `bond`, needs it.
- **The box.** Orthorhombic: a triclinic box is refused. The data file doesn't
  say which axes are periodic, so `periodic=` takes the input script's
  `boundary` (`"p p f"`) or the periodic axes (`"xy"`). The default, `"xyz"`,
  is LAMMPS's default `boundary p p p`.

## How the fibers are found

A fiber is a chain of spheres in one molecule, each bonded to the next:

- The spheres are in the order of their bonds. Where spheres are also bonded
  past their neighbors, as `create_bonds many` bonds a fiber when its cutoff
  reaches beyond the next sphere, they are in atom-ID order, as long as each
  atom is bonded to the next ID.
- When fibers share a molecule ID (every atom in molecule 0, say) and bonds
  join them, `fiber_bond_types=` names the bond types that run along the
  fibers, so that the others don't merge them.
- In a file without bonds, because the input script makes them with
  `create_bonds` after `read_data`, each run of consecutive atom IDs in one
  molecule whose spheres are within 1.5 diameters of the next is a fiber.

Atoms in no chain stay where they are: single spheres, clusters whose bonds
branch or close a loop, binder spheres. TANGLE doesn't see them, so a fiber
can be pushed into one. Bonds between fibers stay in the file, but don't hold
the fibers together while they relax. `data.notes` counts all of these.

| Attribute | |
|---|---|
| `data.fibers` | a `tangle.FiberCollection`, one fiber per chain, by molecule ID and then atom ID |
| `data.fiber_atoms` | each fiber's atom IDs in chain order: point `i` of fiber `k` is atom `fiber_atoms[k][i]` |
| `data.other_atoms` | the IDs of the atoms in no fiber |
| `data.cell` | the `tangle.Cell` the fibers relax in |
| `data.notes` | what was left alone, and why |

A fiber's centerline runs through its sphere centers and is also its rest
shape, so the relaxation keeps each fiber's length and bends as closely as it
can while it pushes the fibers apart. The fiber is a smooth tube as wide as
its largest sphere, so neighboring spheres of one fiber may overlap, as they
often do in BPM fibers. Its material is named after its spheres' atom type
(`"type 1"`).

## Relaxing

`data.relax(settings=None)` relaxes the fibers in `data.cell` and returns the
`RunResult`. `settings` is a `tangle.RelaxationSettings`. By default the run
is on the GPU (`backend="wgpu"`), allows overlaps of up to 1% of the thinnest
fiber's diameter (`penetration_tolerance`), and stops after 2000 iterations
(`max_iterations`). Check `run.converged` and `run.max_penetration` before
writing the file.

On an axis that isn't periodic, TANGLE's walls keep the fibers' surfaces
inside the box. Where a sphere already sticks out of the box, as spheres
centered on a shrink-wrapped face do, the wall on that face is moved out to
the sphere's surface, so that no sphere moves just because it is near a face.

With `adaptive_segmentation` left off, as it is by default, every sphere goes
exactly where its point on the fiber went. With it on, a relaxed fiber can
have other points than it started with, and `write_data` places each sphere
at its share of the fiber's length.

A recipe of your own works too: `tangle.Recipe(data.cell)`, insert
`data.fibers`, add steps, and pass the run to `write_data`, as long as the
steps leave the cell as it is, since the file keeps its box.

## Writing the file

`data.write_data(path, run)` copies the file, changing only the Atoms rows of
the spheres that moved, and returns how many moved. In those rows only x, y, z
and the image flags change. The header, coefficients, velocities, bonds, other
sections, comments and line endings stay as they were. `run` can also be the
relaxed centerlines themselves, one list of points per fiber.

- On a periodic axis, a sphere that moved is put back inside the box, and its
  image flags, where the rows have them, count the box lengths it moved, so
  that a fiber that crosses a face stays whole.
- The GPU works in single precision, so spheres that didn't move come back a
  little off. A sphere that moved less than a millionth of the box length (or
  of its coordinate, if that is larger) keeps its row as it was.

## The bonds' rest state

LAMMPS's BPM bond styles store each bond's reference state (its length, and
for `bpm/rotational` its orientations) when the bond is first computed at the
start of a run. It is written to restart files but not to data files, so a
simulation that reads the relaxed file defines it again from the relaxed
positions: the bonds start unstressed, at the fibers' relaxed lengths and
bends. A bond between two fibers that the relaxation moved apart rests at its
new length.

## Limits

- Orthorhombic boxes only, as for TANGLE's cells.
- Round fibers of spheres. Files with capsules, such as TANGLE's
  `spherocylinders_*` exports for DIRT, are refused; files from its
  `spheres_*` exports read back.
- The relaxation sees only the fibers. Check the spheres in no fiber, which
  stay where they were, for overlaps with the fibers that moved.
- To build a BPM model from a TANGLE network instead, see
  [BPM export](../crates/tangle_python/README.md#bpm-export).
