# Python API migration (0.1 alpha)

The Python API was renamed in one pass to make it consistent and harder to
misuse. There are no deprecated aliases: old names raise `AttributeError` or
`TypeError`. This page maps each old spelling to its replacement.

## Conventions

- `max_` and `min_` prefixes everywhere; `maximum_*` and `minimum_*` are gone.
- Diameters, never radii, at the Python boundary.
- "Curvature ratio" is the only name for curvature over allowed curvature
  (no more "bend ratio").
- Axes accept `"x"`, `"y"`, `"z"` or `0`, `1`, `2`. Axis sets accept a string
  such as `"xy"`, one axis, or a bool triple.
- `*Settings` are run-wide, a `*Policy` is a named rule for one step, and
  `*Overrides` are temporary per-step deltas.
- Every configuration class takes keyword arguments and has
  `replace(**changes)`. Unknown keywords raise `TypeError`; bad option strings
  raise `ValueError` on the line that set them.
- Recipe failures raise `tangle.RecipeError` (a `RuntimeError`) with
  `operation_index`, `operation`, `iteration` and `reason`.
- `tangle.units` provides `um`, `mm` and `nm` in meters.

## Cells, materials and collections

| Old | New |
|---|---|
| `Cell(lengths, periodic=[True, True, False])` | `Cell(lengths, periodic="xy")` (bool lists still work) |
| `Recipe(cell, layer_axis=2)` | `Cell(...)` infers `stack_axis` from its single bounded axis; override with `Cell(..., stack_axis="z")` or `Recipe(cell, stack_axis="z")` |
| `Material(name, diameter, minimum_bend_radius=r)` | `Material(name, diameter, min_bend_radius=r)` |
| `FiberCollection.layers()` | `FiberCollection.layer_ids()` |
| `a.extend(b)` for a new collection | `a + b` (`extend` still mutates in place) |
| `AnalysisReport.maximum_curvature`, `maximum_curvature_ratio`, `bend_limit_violations` | `max_curvature`, `max_curvature_ratio`, `curvature_limit_violations` |

## Generation

`FiberPopulationSettings` is now `FiberPopulation`, built with keywords.

| Old `FiberPopulationSettings` field | New `FiberPopulation` argument |
|---|---|
| `material_name`, `minimum_bend_radius` | `material=Material(...)` |
| `length_minimum`, `length_maximum` | `length=(min, max)` or a single value |
| `radius_minimum`, `radius_maximum` | `diameter=(min, max)`, a single value, or `None` for the material's diameter |
| `curvature_amplitude_minimum`, `curvature_amplitude_maximum` | `curvature_amplitude=(min, max)` |
| `orientation="isotropic"` | `orientation=IsotropicOrientation()` |
| `orientation="planar"`, `orientation_axis`, `maximum_tilt` | `PlanarOrientation(normal=None, max_tilt=...)` (normal defaults to the stack axis) |
| `orientation="layered_biaxial"`, `primary_fraction`, `cross_fraction`, `maximum_in_plane_deviation`, `layer_orientation_seed` | `LayeredBiaxialOrientation(primary_fraction=, cross_fraction=, max_in_plane_deviation=, max_tilt=, seed=)` |
| `orientation="aligned"`, `orientation_axis`, `maximum_angle` | `AlignedOrientation(axis, max_angle=)` |
| `position="uniform"` | `position=UniformPosition()` |
| `position="layered"`, `position_axis`, `layers`, `jitter_fraction` | `LayeredPosition(layer_count, axis=None, jitter_fraction=)` |
| `position="density_gradient"`, `density_exponent`, `density_toward_high` | `DensityGradientPosition(axis=None, exponent=, toward_high=)` |

The generator functions take `material=Material(...)` in place of
`radius=`, `material_name=` and `minimum_bend_radius=`.

## Relaxation settings, policies and overrides

| Old | New |
|---|---|
| `CellListSettings`, `RelaxationSettings.cell_list` | `RelaxationSettings.cell_size_scale` |
| `AdaptiveSegmentationSettings.minimum_length_over_diameter`, `maximum_refinement_levels` | `min_length_over_diameter`, `max_refinement_levels` |
| hand-written cleanup overrides | `RelaxationOverrides.preset("contact_first" \| "curvature_cleanup" \| "contact_cleanup", **changes)` |
| `SolvePolicy.solver_penetration`, `solver_curvature_ratio` | `target_penetration`, `target_curvature_ratio` |
| `SolvePolicy.acceptance_penetration`, `acceptance_curvature_ratio` | `max_penetration`, `max_curvature_ratio` (default to the targets) |
| `penetration_enforcement`, `curvature_enforcement` = `"hard"`/`"soft"` | `hard_penetration`, `hard_curvature` booleans |
| `on_exhaustion = "reject"` / `"continue_if_hard_limits_satisfied"` | `on_budget_exhausted = "fail"` / `"continue_if_hard_ok"` |
| `SolvePolicy.maximum_iterations` | `max_iterations` |
| `JunctionPolicy.maximum_surface_gap`, `minimum_crossing_angle`, `maximum_crossing_angle`, `maximum_per_fiber_pair`, `minimum_anchor_separation` | `max_surface_gap`, `min_crossing_angle`, `max_crossing_angle`, `max_per_fiber_pair`, `min_anchor_separation` |

## Compaction

| Old `CompactionSettings` | New |
|---|---|
| `target_type`, `target_value`, `target_values` | `target=VolumeFractionTarget(v)`, `CellVolumeTarget`, `CellLengthsTarget`, `MeanPressureTarget`, `DirectionalPressureTarget`, `PenaltyEnergyTarget` |
| `path="axis_weights"`, `axis_weights` | `path=AxisWeightsPath("z")` or `AxisWeightsPath([0, 0, 1])`; `AxisWeightsPath()` follows the stack axis |
| `path="equal_pressure"`, `active_axes`, `pressure_floor` | `EqualPressurePath("xy", pressure_floor=)` |
| `path="stress_ratio"`, `stress_ratio` | `StressRatioPath(ratio, pressure_floor=)` |
| `path="minimum_work"`, `active_axes` | `MinimumWorkPath("xy")` |
| `volume_fraction(v, axis_weights=[0, 0, 1])` | `volume_fraction(v)` compresses along the stack axis; any field can follow as a keyword |
| `minimum_log_strain`, `maximum_log_strain`, `maximum_steps`, `maximum_relax_windows`, `maximum_pressure`, `maximum_penalty_energy`, `maximum_penetration` | `min_log_strain`, `max_log_strain`, `max_steps`, `max_relax_windows`, `max_pressure`, `max_penalty_energy`, `max_penetration` |
| `maximum_bend_ratio` | `max_curvature_ratio` |
| `maximum_shortening_over_minimum_diameter` | `max_shortening_over_min_diameter` |

## Recipe verbs

| Old | New |
|---|---|
| `relax(maximum_iterations=)` | `relax_until_converged(max_iterations=)` |
| `relax_until_targets_reached(tolerance, maximum_iterations)` | `settle_targets(tolerance=, max_iterations=)` |
| `relax_with_policy(policy, overrides)` | `solve(policy, overrides=None)` |
| `set_material_bend_radius(name, r)` | `set_min_bend_radius(material_or_name, r)`; unknown names raise `ValueError` |
| `move_layers(spacing_scale)` | `scale_layer_spacing(factor)` |
| `release_layer_targets()` | `release_layer_placement()` |
| `needle_layer_circular(layer, center, diameter, depth, ...)` | `needle_layer(layer, footprint=CircularFootprint(center, diameter=), depth=)` |
| hand-rolled `splitmix64` needle centers | `CircularFootprint.random(diameter=, seed=)` (the layer is mixed in automatically) |
| `needle_layer_random(layer, fraction, seed, depth, ...)` | `needle_layer(layer, footprint=RandomFiberFraction(fraction, seed=), depth=)` |
| `minimum_fiber_diameter`, `maximum_translation_over_fiber_diameter` | `min_fiber_diameter`, `max_translation_over_diameter` |
| `relax_and_capture(iterations, every, policy)` | `relax_and_capture(iterations=, capture_every=, policy=)` |
| paired place/release calls | `with recipe.place_layer_above(...):` or `with recipe.needle_layer(...):` releases at the end of the block |
| `run()` wrote the result back into the input `Assembly` | `run()` leaves the input alone; use `RunResult.assembly` |
| `export_bpm(mode="spherocylinders-exact")` | `mode="spherocylinders_exact"` (hyphens still accepted) |

## Examples

| Old folder | New folder |
|---|---|
| `fake_needled` | `needled_felt_toy` |
| `fake_needled_2` | `needled_preform_two_fiber` |
| `fake_felted` | `felt_20ply_control_vs_needled` |
| `fake_tps_formation` | `tps_preform_formation` |
| `fibers_through_center_point_dem_bpm`, native `center_point` | `fibers_through_center_point` |
| `biased_fiber_box_dem_bpm` | `biased_fiber_box` |
| `multisegment_flexible_relaxation_dem_bpm`, native `multisegment_shapes` | `multisegment_flexible_relaxation` |

The felt example's six `--mode` values became `--specimen {control,needled}
--stage {form,cleanup,polish}`, and its materials are now `fine_7um` and
`coarse_19um`. Checkpoints written before the rename carry the old material
names, so re-run formation instead of resuming from them.
