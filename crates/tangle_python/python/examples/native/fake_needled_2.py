"""Physical-scale two-material needled preform matching the Rust configuration."""

import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "fake_needled_2"
FOOTPRINT = 0.0025
INITIAL_THICKNESS = 0.0048
LAYER_COUNT = 6
LAYER_GAP = 50.0e-6


def material_settings(name: str, seed: int, diameter: float) -> tangle.FiberPopulationSettings:
    settings = tangle.FiberPopulationSettings()
    settings.count = 188
    settings.segments_per_fiber = 4
    settings.seed = seed
    settings.nominal_parent_length = 0.0508
    settings.length_minimum, settings.length_maximum = 0.003, 0.004
    settings.radius_minimum = settings.radius_maximum = 0.5 * diameter
    settings.orientation = "layered_biaxial"
    settings.orientation_axis = [0.0, 0.0, 1.0]
    settings.primary_fraction = 0.40
    settings.cross_fraction = 0.40
    settings.maximum_in_plane_deviation = math.pi / 18.0
    settings.maximum_tilt = math.pi / 1_800.0
    settings.layer_orientation_seed = 20_260_941
    settings.position = "layered"
    settings.position_axis = 2
    settings.layers = LAYER_COUNT
    settings.jitter_fraction = 0.25
    settings.max_attempts_per_fiber = 2_048
    settings.material_name = name
    if diameter < 10.0e-6:
        settings.curvature_amplitude_minimum = 0.0
        settings.curvature_amplitude_maximum = 2.0e-6
        settings.minimum_bend_radius = 500.0e-6
    else:
        settings.curvature_amplitude_minimum = 0.0
        settings.curvature_amplitude_maximum = 8.0e-6
        settings.minimum_bend_radius = 60.0e-6
    return settings


def formation_policy(final: bool = False) -> tangle.SolvePolicy:
    policy = tangle.SolvePolicy(
        "final hard relaxation" if final else "formation relaxation",
        solver_penetration=0.30e-6,
        solver_curvature_ratio=1.00001,
        acceptance_penetration=0.30e-6,
        penetration_enforcement="hard",
        acceptance_curvature_ratio=1.001 if final else 1.05,
        curvature_enforcement="hard" if final else "soft",
        maximum_iterations=100_000 if final else 15_000,
        on_exhaustion="reject" if final else "continue_if_hard_limits_satisfied",
    )
    return policy


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def needle_center(layer: int) -> list[float]:
    seed = 20_260_940
    unit = lambda bits: (bits >> 40) / float(1 << 24)
    return [
        FOOTPRINT * unit(splitmix64(seed ^ (2 * layer))),
        FOOTPRINT * unit(splitmix64(seed ^ (2 * layer + 1))),
    ]


def final_compaction() -> tangle.CompactionSettings:
    settings = tangle.CompactionSettings.volume_fraction(0.13, axis_weights=[0.0, 0.0, 1.0])
    settings.kinematics = "moving_walls"
    settings.cell_anchor = [0.0, 0.0, 0.0]
    settings.initial_log_strain = 0.01
    settings.minimum_log_strain = 0.001
    settings.maximum_log_strain = 0.02
    settings.growth_factor = 1.2
    settings.shrink_factor = 0.5
    settings.relax_iterations = 1_000
    settings.maximum_shortening_over_minimum_diameter = 0.5
    settings.maximum_penetration = 0.301e-6
    settings.maximum_bend_ratio = 1.001
    settings.maximum_steps = 256
    settings.maximum_relax_windows = 8
    return settings


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell(
        [FOOTPRINT, FOOTPRINT, INITIAL_THICKNESS], periodic=[True, True, False]
    )
    small = tangle.generate_fiber_population(
        cell, material_settings("7 um stiff fiber", 20_260_938, 7.0e-6), name="small"
    )
    large = tangle.generate_fiber_population(
        cell, material_settings("19 um bendy fiber", 20_260_939, 19.0e-6), name="large"
    )
    recipe = tangle.Recipe(cell, layer_axis=2)
    for layer in range(LAYER_COUNT):
        ply = small.select_layer(layer, name=f"mixed ply {layer}")
        ply.extend(large.select_layer(layer))
        recipe.insert(ply)
        recipe.relax_with_policy(formation_policy())
        if layer == 0:
            continue
        for gap in (65.0e-6, LAYER_GAP):
            recipe.place_layer_above(layer, gap, stiffness=1.0, max_translation=2.0e-6)
            recipe.relax_until_targets_reached(0.5e-6, 3_000)
            recipe.relax_for(100)
            recipe.release_layer_targets()
            recipe.relax_with_policy(formation_policy())
        if layer >= 2:
            recipe.needle_layer_circular(
                layer,
                needle_center(layer),
                150.0e-6,
                7.0 * LAYER_GAP,
                minimum_fiber_diameter=0.99 * 19.0e-6,
                stiffness=0.5,
                max_translation=1.0e-6,
                maximum_translation_over_fiber_diameter=0.25,
            )
            recipe.relax_until_targets_reached(0.5e-6, 3_000)
            recipe.relax_for(150)
            recipe.release_needles()
            recipe.relax_with_policy(formation_policy())
    recipe.release_needles()
    recipe.release_layer_targets()
    recipe.relax_with_policy(formation_policy(final=True))
    recipe.compact(final_compaction())
    recipe.relax_with_policy(formation_policy(final=True))

    settings = tangle.RelaxationSettings()
    settings.constraint_iterations = 8
    settings.curvature_limit_safety_margin = 0.06
    settings.curvature_cleanup_sweeps = 4
    settings.max_iterations = 500_000
    settings.iterations_per_batch = 10
    settings.penetration_tolerance = 0.30e-6
    settings.correction_fraction = 0.35
    settings.max_step = 2.0e-6
    adaptive = tangle.AdaptiveSegmentationSettings()
    adaptive.contact_length_over_diameter = 4.0
    adaptive.minimum_length_over_diameter = 2.0
    adaptive.maximum_refinement_levels = 6
    adaptive.refinement_interval = 32
    adaptive.refinement_persistence = 3
    adaptive.coarsening_persistence = 8
    settings.adaptive_segmentation = adaptive
    return recipe, settings


def run(resume: bool = False) -> tangle.RunResult:
    recipe, settings = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = tangle.CheckpointSettings(
        "fake_needled_2-v19-staged-solve-policy",
        OUTPUT / "fake_needled_2.restart",
        interval_iterations=500,
        resume=resume,
    )
    result = recipe.run(settings, checkpoint=checkpoint)
    result.write_ovito(
        OUTPUT / "final.dump",
        view_script_path=OUTPUT / "view.py",
        session_path=OUTPUT / "final.ovito",
        coloring="curvature_ratio",
    )
    return result


if __name__ == "__main__":
    print(run())
