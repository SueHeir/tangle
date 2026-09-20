# Adaptive crossed fibers

This example separates two useful adaptive-segmentation behaviors:

- `orthogonal_stop_early` starts with one segment per fiber at a 90-degree
  crossing. The contact is point-like, so refinement should stop as soon as the
  newly introduced midpoint supplies enough freedom.
- `shallow_adaptive` starts with one segment per fiber at a deep 12-degree
  crossing. Its longer, denser contact zone should remain active through
  multiple refinement decisions and create a locally graded centerline.
- `shallow_uniform_reference` resolves that same shallow crossing with eight
  uniform segments per fiber.

All endpoints are pinned. The adaptive target length is `2 * diameter`, with a
hard reserved minimum of `1 * diameter`. Contact must persist through two
adaptation checks before a segment splits. At most one tree generation can
become active in a refinement epoch.

Refinement is checked every three lifetime solver iterations. GRASS/debug
batches end every four iterations, deliberately demonstrating that the
refinement clock is independent of application scheduling. Geometry and
topology remain resident in the CubeCL device world between both kinds of
boundary.

Refinement is reversible. A sibling pair may merge after eight contact-free
checks, provided its midpoint is unpinned, below 25% of the admissible-curvature
limit, and within 0.1 fiber diameter of the parent chord. Separate persistence
windows prevent rapid split/merge oscillation.

Run it from the TANGLE workspace:

```sh
cargo run --release -p adaptive_crossed_fibers
```

Each case writes a LAMMPS trajectory, an OVITO viewing script, an OVITO session
target, and a final DEM-BPM data file below its `output/` directory. Run a
generated `relaxation_view.py` with `ovitos` to create the session. Capsules are
colored by subdivision level.

This remains an educational mechanics case, not the long performance
benchmark. The current adaptive layout reserves a complete binary tree and
skips inactive nodes; a compact active-segment frontier is still needed before
claiming production performance gains.
