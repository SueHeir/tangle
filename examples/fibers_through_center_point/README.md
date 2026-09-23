# Fibers through center point

Start with the **[Python notebook](../../crates/tangle_python/python/examples/native/fibers_through_center_point.ipynb)**
or [editable script](../../crates/tangle_python/python/examples/native/fibers_through_center_point.py).
See the [installation guide](../../crates/tangle_python/README.md) first.
The commands and output paths below describe the native Rust version; Python
uses the same solver but can have different export/debug defaults and paths.

[![Eight rigid fibers separating from a shared center point](../../docs/media/center-point.png)](../../docs/media/center-point.mp4)

*Click the preview to play the OVITO rendering.*

This example deliberately starts eight straight circular fibers with all
centerlines passing through the center of a unit box. It then:

1. uploads flat centerline and topology buffers into a persistent CubeCL GPU
   world;
2. rebuilds a uniform cell list and detects capsule overlaps on the GPU;
3. applies frictionless, translation-only GPU projection until the overlap
   tolerance is reached, moving every vertex of each fiber together so its
   shape and orientation remain exact;
4. samples each relaxed centerline into DEM spheres;
5. bonds consecutive spheres belonging to the same source fiber;
6. writes the resulting bonded-particle model as a LAMMPS data file.

Run it from the TANGLE repository root:

    cargo run -p fibers_through_center_point

The program follows the physical workflow in code order: define and generate
the assembly, relax it on the GPU, optionally observe it, and export it. The
GRASS plugin schedule remains explicit; only repetitive output paths and
configuration defaults are delegated to small helpers.

Generated files are placed in the output directory. Cross-fiber bonds are
intentionally not created: different fibers are separated geometrically, while
each fiber's own sphere chain receives explicit BPM bonds.

To record the relaxation for inspection in OVITO:

    cargo run -p fibers_through_center_point -- --debug-ovito

This opt-in mode explicitly downloads debug snapshots and writes
`output/relaxation.dump`, a multi-frame LAMMPS dump
containing one spherocylinder per centerline segment. For this straight-fiber
example that means eight capsules per frame instead of the denser sphere-chain
BPM discretization. Each capsule carries its stable segment ID, source-fiber
ID, radius, length, orientation, local segment index, and fiber point count.

It also writes `output/relaxation_view.py`, a portable viewing recipe that
loads the trajectory, selects spherocylinder rendering, colors by fiber ID, and
saves `output/relaxation.ovito`. Run it on a machine with OVITO Pro or the OVITO
Python package:

    ovitos output/relaxation_view.py

OVITO Basic users can open `relaxation.dump`, select the Spherocylinder particle
shape, and use **File → Save Session State** to create the same `.ovito` file.

The exported data file uses `spheres-dynamic` mode with one-radius target
spacing, so source centerline endpoints remain explicit. It contains both the
`bpm/sphere` Atoms section and the Bonds section. DIRT or another compatible
LAMMPS BPM workflow can consume the model without TANGLE owning its runtime
configuration.

Useful first experiments are changing `FIBER_COUNT`, the generator radius, or
the penetration tolerance. Increasing the radius makes the initial collision
more severe, while selecting `RelaxationConfig::flexible()` allows individual
centerline vertices to move.
