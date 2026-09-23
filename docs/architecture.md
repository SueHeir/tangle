# Solver and architecture notes

For normal use, begin with the [Python guide](../crates/tangle_python/README.md).
Python builds collections and recipes; public Rust APIs implement validation,
geometry, characterization, relaxation, and file formats. Python does not have
a separate numerical implementation.

## Data and scheduling

`FiberAssembly` holds intrinsic (rest), placed, and optional assembled-reference
centerlines, material/section tables, periodic cell, provenance, and junctions.
Material-coordinate `FiberAnchor`s resolve onto the current discretization.

A recipe is an ordered approximation of a manufacturing process. Collections
are prepared on the host, packed into the device world, and activated at their
scheduled insertion steps. Dormant fibers do not participate in contacts.
GRASS schedules bounded batches and operation transitions; it can change layer
targets, activate populations, and compact the cell between batches without
re-uploading all geometry. The stack axis (the direction layers stack along)
comes from the cell: its single non-periodic axis, else z. `Recipe(cell,
stack_axis="x")` overrides it.

## Quasi-static relaxation, not a dynamics solver

The current method iteratively applies capsule-contact separation, stretch
constraints, rest-chord bending corrections, boundary corrections, and a
three-point admissible-curvature projection. It is a constraint/penalty-based
geometric relaxation, not a proof of a global energy minimum or a calibrated
force–displacement solver. Formation penalty energies and directional pressure
proxies help control compaction; they are not automatically physical stresses.

Rigid mode translates every vertex of a fiber together. Flexible mode lets
vertices move independently subject to stretch, rest-shape, and bend-limit
constraints. Pinned vertices have zero inverse mass in curvature projection.
Natural curvature and admissible curvature are distinct: the first is a
preference, the second limits allowed bending.

Hard acceptance requires both penetration and bend utilization to meet their
configured tolerances. Recipes may use softer intermediate gates and temporary
overrides. Inspect final residuals and events; reaching an iteration budget or
passing a soft stage is not final mechanical acceptance.

## Device world and adaptation

CubeCL kernels run via WGPU or the explicit native CPU backend. A compact
count/scan/scatter cell list builds per-segment Verlet neighbor lists holding
every capsule within a skin of touching; exact capsule checks resolve contacts
from those lists. Lists are rebuilt on the device once any vertex has moved
half the skin (`CellListConfig::neighbor_skin_scale`, a multiple of the largest
radius), or after activation, refinement, or compaction changes the geometry. Orthorhombic periodic axes use wrapped neighborhoods/minimum-image
contacts; bounded axes have planar walls. Normal batches read scalar status;
debug trajectories and checkpoints explicitly read geometry.

Adaptive centerlines reserve dyadic trees and alter activity masks. Persistent
contact triggers refinement; quiet, sufficiently straight siblings can coarsen.
Pinned, targeted, strongly bent, and junction-bearing topology is protected.
Refinement/coarsening cadence uses solver iterations independently of GRASS
batch size. Reserved memory and inactive-node work mean fewer active segments
do not guarantee proportional speedups.

## Compaction and junctions

Compaction moves the cell incrementally, relaxes, and accepts or rolls back a
trial. Targets include nominal volume fraction, dimensions, and penalty-based
pressure/energy measures. Axis selection and opposing-face balancing can adapt
to relaxation cost. A jam or exhausted guard is a reported outcome, not a
license to export a penetrated trial as converged.

Junction capture filters contact candidates by material pair, gap, angle,
probability, multiplicity, and anchor spacing. Captured junctions become
persistent topology and can be exported as inter-fiber bonds. **They currently
do not enforce mechanical bonding during subsequent TANGLE relaxation.**
Ordinary contacts are neither permanent bonds nor junctions. Late capture is
appropriate when bonds represent binder added after formation.

## Reproducibility and restart

Seeded geometry generation is deterministic; parallel accumulation and atomic
scatter ordering mean relaxation is not promised bitwise deterministic.
Compare geometry and residuals within explicit tolerances across runs/devices.

Atomic checkpoints contain assembly, recipe cursor/partial operation, counters,
active topology, adaptive history, cell bounds, and manufacturing targets.
Scratch contact data is rebuilt on resume. Checkpoint schemas are experimental;
cross-revision compatibility is not guaranteed. Archive the generating revision,
settings, backend, and a final geometry export, and branch output paths when
trying a new cleanup strategy.

## Crate ownership

| Crate | Responsibility |
| --- | --- |
| `tangle_python` | Primary user API, PyO3 bindings, scripts, notebooks |
| `tangle_core` | Canonical assembly and validation |
| `tangle_app` | GRASS stages, phases, plugin contracts |
| `tangle_generate` | Seeded generators and formation operations |
| `tangle_contact` | Cell list and capsule contact kernels |
| `tangle_relax` | Persistent device world and mechanics corrections |
| `tangle_checkpoint` | Atomic restart serialization |
| `tangle_characterize` | Native centerline/material metrics |
| `tangle_export` | BPM, OVITO, and PuMA-compatible outputs |
