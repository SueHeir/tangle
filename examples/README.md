# Example progression (Python first)

Install the extension using the [Python guide](../crates/tangle_python/README.md),
then begin with [tutorial 01](../crates/tangle_python/python/tutorials/01_high_level_overview.ipynb).
The [topic notebooks](../crates/tangle_python/python/tutorials/README.md) explain
individual settings. Each script below has a matching notebook beside it.

| Concept | Python script | Native notes |
| --- | --- | --- |
| Minimal crossing | [crossed_fibers](../crates/tangle_python/python/examples/crossed_fibers.py) | Small complete workflow |
| Rigid separation | [fibers_through_center_point](../crates/tangle_python/python/examples/native/fibers_through_center_point.py) | [README](fibers_through_center_point/README.md) |
| Rest shape and bend limits | [multisegment_flexible_relaxation](../crates/tangle_python/python/examples/native/multisegment_flexible_relaxation.py) | [README](multisegment_flexible_relaxation/README.md) |
| Adaptive versus uniform resolution | [adaptive_crossed_fibers](../crates/tangle_python/python/examples/native/adaptive_crossed_fibers.py) | [README](adaptive_crossed_fibers/README.md) |
| Biased populations | [biased_fiber_box](../crates/tangle_python/python/examples/native/biased_fiber_box.py) | [README](biased_fiber_box/README.md) |
| Staged manufacturing | [tps_preform_formation](../crates/tangle_python/python/examples/native/tps_preform_formation.py) | [README](tps_preform_formation/README.md) |
| Six-layer needling | [needled_felt_toy](../crates/tangle_python/python/examples/native/needled_felt_toy.py) | [README](needled_felt_toy/README.md) |
| Two-material needling | [needled_preform_two_fiber](../crates/tangle_python/python/examples/native/needled_preform_two_fiber.py) | [README](needled_preform_two_fiber/README.md) |
| Twenty-ply control and needled felt | [felt_20ply_control_vs_needled](../crates/tangle_python/python/examples/native/felt_20ply_control_vs_needled.py) | [README](felt_20ply_control_vs_needled/README.md) |
| Native versus voxel analysis | [puma_cross_validation](../crates/tangle_python/python/examples/puma_cross_validation.py) | [PuMA](puma_cross_validation/README.md) |
| Periodic domain size for DIRT | [periodic_domain_sweep](../crates/tangle_python/python/examples/periodic_domain_sweep.py) | Python only; side 10 to 2 at fixed fiber length |

For another small staged-insertion example, see
[layered_recipe](../crates/tangle_python/python/examples/layered_recipe.ipynb).

Run scripts from the repository root, for example:

```console
python crates/tangle_python/python/examples/crossed_fibers.py
```

The `native/` Python scripts reproduce scientific configurations using
Rust-backed generators and the same solver, not subprocesses. They are not
byte-identical ports of native CLI output: paths, export modes, debug defaults,
and checkpoint case IDs can differ. Inspect `build()`/`run()` or `--help` where
provided before starting a large calculation.

Small examples use normalized cell units; physical-scale felt uses meters.
Convert geometry and all length settings together when changing units.
WGPU is the default. In editable Python code, select `settings.backend = "cpu"`
before `recipe.run(settings)` if needed; large recipes can be slow on CPU.

Manufacturing notebooks guard full solves explicitly; the corresponding scripts
may start them immediately. They can take minutes to hours. Long-run data is
not bundled with the source. Outputs are ignored by Git, and re-running a case
can overwrite its paths—choose a new directory for each study.

Existing movies are archived demonstrations, not guaranteed output from the
current revision. Check acceptance reports rather than inferring convergence
from a visually plausible OVITO frame.
