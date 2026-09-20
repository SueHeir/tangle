# Fake needled 2.0

This experiment moves the needled-layer recipe into physical SI units and a
two-material population:

- a `2.5 mm × 2.5 mm` footprint with periodic `x` and `y` faces;
- a 4.8 mm temporary staging height that keeps all six plies separate and leaves clearance below the deposited stack for
  deep needle pulls;
- six bounded deposition layers through thickness;
- 188 nominally 7 µm diameter local fiber windows with a 500 µm minimum bend radius;
- 188 nominally 19 µm diameter local fiber windows with a 60 µm minimum bend radius;
- 3–4 mm simulated centerline windows representing nominal 50.8 mm (2 inch)
  manufacturing fibers;
- a final nominal volume-fraction target of 13%, representative of a nominal
  `0.20 g/cc` MERINO felt if the fully dense carbon-phenolic blend is
  approximately `1.5 g/cc`.

Each deposited layer receives a reproducibly random in-plane reference
direction. Approximately 40% of its fibers lie within 10 degrees of that
direction, 40% lie within 10 degrees of the perpendicular direction, and the
remaining 20% are uniformly random in-plane. All groups retain at most 0.1
degree of out-of-plane tilt before relaxation and lowering, so ply loft arises
from crossings and contact instead of a large imposed initial slope.

This uses one sixteenth of the former 10 mm-square footprint area. Because the
simulated local windows are about half as long, it uses one eighth of the old
fiber count to preserve approximately the same centerline length per unit area.
Equal fiber counts do not mean equal solid volume: the 19 µm population
contributes most of the nominal volume because cross-sectional area scales with
diameter squared. This is an intentional first trial and is easy to change in
`src/main.rs`.

The periodic generator samples fiber centers over the complete in-plane cell.
Centerlines may cross an `x` or `y` face and interact through the GPU solver's
minimum-image contact rules, avoiding artificial free-edge depletion.

Run the normal centerline experiment:

```console
cargo run --release -p fake_needled_2
```

The example reports completed formation operations, relaxation residuals every
250 GPU iterations, and every accepted compaction increment. The compaction
lines include current volume fraction, cell thickness, accepted penetration,
and penalty energy, so a long or jammed run can be diagnosed while it runs.

Every deposition, approach, and needling command is followed by an explicit
admissibility gate. A newly activated layer first relaxes in its physically
separated staging plane, then moves alone through two approach steps toward the
previously deposited stack. The recipe cannot advance until capsule
penetration is at or below 0.30 µm and every bend ratio is at or below one. If
a gate exhausts its local iteration budget, formation is rejected and no final
OVITO state is written. Once a temporary layer target is released, contact
transmits its load through the free stack.

The 1,504 initially active segments are a deliberately coarse, four-segment
root discretization. Adaptive segmentation reserves dyadic refinement trees on
the GPU and activates midpoint children only near persistent unresolved
contacts. A segment is eligible to split while it is longer than four fiber
diameters, never below two diameters, with at most six subdivision levels.
Adaptation is checked every 32 solver iterations independently of the GRASS
batch size, and contact must persist through three checks before the next split.
Two quiet sibling segments may merge after eight contact-free checks (256
iterations), but only when their midpoint is within 0.1 fiber diameter of the
parent chord and below 25% of the admissible bend limit. Pinned, actively
needled, significantly bent, and junction-bearing topology is not coarsened.

A compact final-state-only OVITO file is useful for checking in-plane coverage
without recording the full relaxation trajectory:

```console
cargo run --release -p fake_needled_2 -- --final-ovito
```

For long experiments, enable an atomic rolling restart checkpoint. It is saved
every 500 GPU iterations and once more at normal completion:

```console
cargo run --release -p fake_needled_2 -- --checkpoint
```

Resume from `output/fake_needled_2.restart` with the complete formation and GPU
state restored. `--resume` continues updating the same checkpoint file:

```console
cargo run --release -p fake_needled_2 -- --resume
```

For collision debugging, the companion audit binary compares every active
cross-fiber capsule pair on the host with the GPU cell-list contact capture:

```console
cargo run --release -p fake_needled_2 --bin collision_audit -- \
  examples/fake_needled_2/output/fake_needled_2.restart
```

This intentionally performs an exhaustive host-side check and is not part of
the resident GPU relaxation loop. It reports the independently measured
maximum penetration, the stored solver residual, GPU contact count, and any
pair missed by the broad phase. It is especially useful before trusting a
long-run checkpoint or final OVITO export.

During final closed-loop compaction, only converged increments are accepted.
An inadmissible trial is restored from its last accepted device checkpoint and
retried with a smaller strain increment; its penetrated geometry is never used
as the starting point for the next trial.

From the third deposited layer onward, a circular needle footprint is placed at
a reproducibly random position in the periodic plane. It selects at most one
internal vertex from every 19 um bendy fiber that has a vertex inside the
footprint; the 7 um stiff population is not needled. The deliberately visible
teaching footprint is 150 um in diameter. Needling begins after the third ply
has been deposited, and selected vertices are pulled downward by seven 50 um
layer spacings (350 um) into the free space below the bottom ply. The needle
advances by at most 0.5 um per solver iteration—well below the 7 um obstacle
diameter—and remains active for at least 1,400 iterations. A hard contact and
bend convergence gate must then pass while the target is still held. Only that
admissible threaded configuration may release the needle; the free stack must
pass a second hard gate afterward. This quasi-static ordering prevents a thick
fiber from jumping through a thin one between discrete solver states.

Detailed OVITO output is optional because refinement can grow well beyond the 1,504 initial
centerline segments. It streams a frame every 100 solver iterations and also
captures exact formation milestones such as layer insertion, completed
approaches, converged releases, needle pulls, and accepted compaction steps:

```console
cargo run --release -p fake_needled_2 -- --debug-ovito
```

For the compact educational trajectory, record only geometry-changing
formation steps and accepted compaction steps:

```console
cargo run --release -p fake_needled_2 -- --debug-ovito-every-step
```

An explicit periodic cadence can be selected with, for example,
`--debug-ovito-interval=250`. `--debug-ovito-every-iteration` is available for
tiny tests, but it forces a complete GPU geometry readback on every solver
iteration and can produce terabyte-scale output for this example.

Run the generated `relaxation_view.py` with OVITO's Python runner (or open the
generated `relaxation.ovito` session) instead of viewing the raw dump with
default display settings. Each centerline segment is one native OVITO
spherocylinder. Its imported `Aspherical Shape.Z` value is the distance between
the two segment vertices (the cylindrical body length); OVITO adds
hemispherical caps of one fiber radius at both ends. The dump uses explicit
`AsphericalShape.X/Y/Z` column names rather than LAMMPS's
`shapex/shapey/shapez`, which OVITO automatically divides by two as diameter
values. Neighboring segments therefore meet through their caps at their shared
centerline vertex.

The fully resolved DEM-BPM bead chains contain millions of particles at these
fiber aspect ratios and are therefore opt-in:

```console
cargo run --release -p fake_needled_2 -- --export-dem
```

This remains a hypothetical material. The diameters and contrasting bend
limits are explicit; the population ratio, length distributions, needling
density, and constitutive stiffnesses are not yet calibrated to an experiment.
