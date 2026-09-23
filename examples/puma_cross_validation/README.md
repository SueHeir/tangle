# PuMA cross-validation fixture

Start with the [Python notebook](../../crates/tangle_python/python/examples/puma_cross_validation.ipynb)
or run `python crates/tangle_python/python/examples/puma_cross_validation.py`
from the repository root after [installing TANGLE](../../crates/tangle_python/README.md).
PuMA is optional; without it the script writes the bundle and skips comparison.
The Rust command below is the equivalent native fixture.

This example creates two equal, orthogonal fibers with a small physical gap,
runs TANGLE's exact centerline characterization, and writes a directly
importable VTI/JSON bundle.

```bash
cargo run --release -p puma_cross_validation
```

The default output is `output/crossed.puma/`. The first command-line argument
may select a different directory. The matching Python example is
`crates/tangle_python/python/examples/puma_cross_validation.py`; that script
imports `pumapy` directly when it is installed.

For the documented 2 µm grid, the checked fixture produces a PuMA workspace of
`50 × 50 × 50`, voxel solid fraction `0.011136`, and orientation diagonal
`[0.5, 0.5, 0.0]`. TANGLE's nominal swept-volume fraction is `0.012566`; the
difference is expected because the voxel result is center-sampled and includes
the exporter's capsule end convention. Rust and Python produce byte-identical
VTI and native-analysis files for this fixture.

## Saved needled-felt section report

With `examples/felt_20ply_control_vs_needled/output/needled_polish_capsules.data`
present (written by `--specimen needled --stage polish`), analyze a central
240 µm cube of the saved 1 mm specimen:

This source file is a local archived result, not included in a fresh clone.
The reconstruction binary is a specimen-specific utility, not a general
capsule importer in the Python API. It assumes the archived two-material data
layout and x/y-periodic boundaries.

```bash
cargo run -p puma_cross_validation --bin needled_section
python examples/puma_cross_validation/analyze_needled_section.py
```

The Python environment needs `pumapy`, NumPy, matplotlib, and Markdown. It uses
PuMA directly, with no TANGLE adapter. Results are under
`output/needled_section/`: `report.html` (embedded figures), `report.md`, raw
`comparison.json`, and interoperable voxel bundles at 3, 2, and 1 µm resolution.
The HTML report can be viewed on its own; retain the adjacent JSON for its raw-data link.

This is a geometric comparison using the final capsule export, not a resumed
simulation or a verification of mechanical convergence. The report distinguishes
nominal centerline volume from voxel union volume and documents crop boundaries,
surface-area normalization, and independent PuMA orientation estimation.
