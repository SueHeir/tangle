# Native example equivalents

Each manufacturing workflow below has an importable Python script and a matching Jupyter
notebook. The Python versions call the same native Rust generators and the
same CubeCL relaxation engine; they do not launch the Rust binaries as
subprocesses.
Python is the recommended starting point; use the [installation guide](../../../README.md)
first. The PuMA fixture is separately available in [the parent examples folder](../puma_cross_validation.ipynb).
Scientific configurations are mirrored, but CLI flags, checkpoint case IDs,
debug cadence, and export representations can differ from the native binaries.

| Rust workflow | Python script and notebook | Configurations represented |
| --- | --- | --- |
| [`fibers_through_center_point`](../../../../../examples/fibers_through_center_point) | `fibers_through_center_point` | Eight rigid fibers through one point |
| [`multisegment_flexible_relaxation`](../../../../../examples/multisegment_flexible_relaxation) | `multisegment_flexible_relaxation` | All four rest/placed-shape cases |
| [`adaptive_crossed_fibers`](../../../../../examples/adaptive_crossed_fibers) | `adaptive_crossed_fibers` | Orthogonal early-stop, shallow uniform, shallow adaptive |
| [`biased_fiber_box`](../../../../../examples/biased_fiber_box) and its `biased_fiber_stress` binary | `biased_fiber_box` | Isotropic, planar layered, aligned-x, and configurable high-density size |
| [`tps_preform_formation`](../../../../../examples/tps_preform_formation) | `tps_preform_formation` | Planar consolidation, through-thickness insertion, junction capture, compaction |
| [`needled_felt_toy`](../../../../../examples/needled_felt_toy) | `needled_felt_toy` | Unitless six-layer random-fraction needling |
| [`needled_preform_two_fiber`](../../../../../examples/needled_preform_two_fiber) | `needled_preform_two_fiber` | Physical-scale two-material needling and restart |
| [`felt_20ply_control_vs_needled`](../../../../../examples/felt_20ply_control_vs_needled) | `felt_20ply_control_vs_needled` | `--specimen {control,needled}` at `--stage {form,cleanup,polish}`; cleanup resumes from form, polish from cleanup |

Launch the notebooks from this directory so their sibling script imports and
case-scoped output paths remain obvious:

```console
cd crates/tangle_python/python/examples/native
jupyter lab
```

The small crossing and shape notebooks execute directly. The manufacturing
notebooks contain an explicit `RUN_FULL_RECIPE` or `RUN_FULL_SCALE` guard.
Those configurations intentionally preserve the native fiber counts and
iteration budgets and may therefore take minutes to hours. Checkpoint-enabled
examples should be resumed instead of restarted after interruption.
Use a matching code revision for restart files; cross-revision compatibility
is not guaranteed. Scripts may start expensive solves without notebook guards.
