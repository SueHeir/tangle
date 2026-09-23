# Two-fiber needled preform

Start with the **[Python notebook](../../crates/tangle_python/python/examples/native/needled_preform_two_fiber.ipynb)**
or [editable script](../../crates/tangle_python/python/examples/native/needled_preform_two_fiber.py).
See the [installation guide](../../crates/tangle_python/README.md) first.
The commands and output paths below describe the native Rust version; Python
uses the same solver but can have different export/debug defaults and paths.

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
- a final nominal volume-fraction target of 13%, an illustrative value obtained
  from assumed bulk/solid densities of `0.20 / 1.5 g/cc`; this is not a measured
  or validated MERINO specification.

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
cargo run --release -p needled_preform_two_fiber
```

The example reports completed formation operations, relaxation residuals every
250 GPU iterations, and every accepted compaction increment. The compaction
lines include current volume fraction, cell thickness, accepted penetration,
and penalty energy, so a long or jammed run can be diagnosed while it runs.

Every deposition, approach, and needling command is followed by an explicit
admissibility gate. A newly activated layer first relaxes in its physically
separated staging plane, then moves alone through two approach steps toward the
previously deposited stack. Formation gates require capsule penetration at or
below 0.30 µm and use a soft bend-ratio acceptance target of 1.05. On budget
exhaustion, a soft gate may continue if its hard limits pass. Final cleanup
uses hard penetration and bend-ratio limits (0.30 µm and 1.001); failure rejects
the final state. Layer targets are released after approach and a 100-iteration
dwell, allowing contacts to settle in a free stack. These are staged solve
policies, not a claim that every intermediate bend meets its final hard limit.

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
cargo run --release -p needled_preform_two_fiber -- --final-ovito
```

For long experiments, enable an atomic rolling restart checkpoint. It is saved
every 500 GPU iterations and once more at normal completion:

```console
cargo run --release -p needled_preform_two_fiber -- --checkpoint
```

Resume from `output/needled_preform_two_fiber.restart` with the complete formation and GPU
state restored. `--resume` continues updating the same checkpoint file:

```console
cargo run --release -p needled_preform_two_fiber -- --resume
```

For collision debugging, the companion audit binary compares every active
cross-fiber capsule pair on the host with the GPU cell-list contact capture:

```console
cargo run --release -p needled_preform_two_fiber --bin collision_audit -- \
  examples/needled_preform_two_fiber/output/needled_preform_two_fiber.restart
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
internal vertex from every `coarse_19um` fiber that has a vertex inside the
footprint; the `fine_7um` population is not needled. The deliberately visible
teaching footprint is 150 um in diameter. Needling begins after the third ply
has been deposited, and selected vertices are pulled downward by seven 50 um
layer spacings (350 um) into the free space below the bottom ply. The needle
advances by at most 1 um per solver iteration, also capped relative to fiber
diameter. Target-error convergence ends the advance, followed by a 150-iteration
dwell, release, and a free-stack formation gate. Small increments help contacts
respond but do not constitute a continuous-collision/topology guarantee.

Detailed OVITO output is optional because refinement can grow well beyond the 1,504 initial
centerline segments. It streams a frame every 100 solver iterations and also
captures exact formation milestones such as layer insertion, completed
approaches, converged releases, needle pulls, and accepted compaction steps:

```console
cargo run --release -p needled_preform_two_fiber -- --debug-ovito
```

For the compact educational trajectory, record only geometry-changing
formation steps and accepted compaction steps:

```console
cargo run --release -p needled_preform_two_fiber -- --debug-ovito-every-step
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
cargo run --release -p needled_preform_two_fiber -- --export-dem
```

This remains a hypothetical material. The diameters and contrasting bend
limits are explicit; the population ratio, length distributions, needling
density, and constitutive stiffnesses are not yet calibrated to an experiment.
