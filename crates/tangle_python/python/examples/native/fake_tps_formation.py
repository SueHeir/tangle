"""Python form of the modular insert/move/relax fake-TPS recipe."""

import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "fake_tps_formation"


def in_plane_settings() -> tangle.FiberPopulationSettings:
    settings = tangle.FiberPopulationSettings()
    settings.count = 96
    settings.segments_per_fiber = 6
    settings.seed = 20_260_919
    settings.length_minimum, settings.length_maximum = 0.28, 0.42
    settings.radius_minimum, settings.radius_maximum = 0.006, 0.009
    settings.curvature_amplitude_minimum = 0.0
    settings.curvature_amplitude_maximum = 0.008
    settings.orientation = "planar"
    settings.orientation_axis = [0.0, 0.0, 1.0]
    settings.maximum_tilt = math.radians(8.0)
    settings.position = "layered"
    settings.position_axis = 2
    settings.layers = 6
    settings.jitter_fraction = 0.18
    settings.minimum_bend_radius = 0.06
    settings.max_attempts_per_fiber = 512
    settings.material_name = "in-plane felt fiber"
    return settings


def through_thickness_settings() -> tangle.FiberPopulationSettings:
    settings = tangle.FiberPopulationSettings()
    settings.count = 24
    settings.segments_per_fiber = 6
    settings.seed = 20_260_920
    settings.length_minimum, settings.length_maximum = 0.50, 0.72
    settings.radius_minimum, settings.radius_maximum = 0.005, 0.007
    settings.curvature_amplitude_minimum = 0.0
    settings.curvature_amplitude_maximum = 0.006
    settings.orientation = "aligned"
    settings.orientation_axis = [0.0, 0.0, 1.0]
    settings.maximum_angle = math.radians(10.0)
    settings.position = "uniform"
    settings.minimum_bend_radius = 0.06
    settings.max_attempts_per_fiber = 512
    settings.material_name = "through-thickness fiber"
    return settings


def planar_bonding() -> tangle.JunctionPolicy:
    policy = tangle.JunctionPolicy("planar consolidation", "felt contact bond")
    policy.maximum_surface_gap = 2.0e-4
    policy.minimum_crossing_angle = math.radians(12.0)
    policy.probability = 0.35
    policy.seed = 20_260_921
    policy.material_pairs = [("in-plane felt fiber", "in-plane felt fiber")]
    return policy


def through_thickness_bonding() -> tangle.JunctionPolicy:
    policy = tangle.JunctionPolicy("through-thickness tying", "through-thickness tie")
    policy.parameter_set = 1
    policy.maximum_surface_gap = 2.0e-4
    policy.minimum_crossing_angle = math.radians(20.0)
    policy.probability = 0.8
    policy.seed = 20_260_922
    policy.material_pairs = [("in-plane felt fiber", "through-thickness fiber")]
    return policy


def final_compaction() -> tangle.CompactionSettings:
    settings = tangle.CompactionSettings.volume_fraction(0.01, axis_weights=[0.0, 0.0, 1.0])
    settings.kinematics = "moving_walls"
    settings.cell_anchor = [0.5, 0.5, 0.5]
    settings.initial_log_strain = 0.03
    settings.minimum_log_strain = 0.002
    settings.maximum_log_strain = 0.06
    settings.growth_factor = 1.25
    settings.shrink_factor = 0.5
    settings.relax_iterations = 40
    settings.maximum_shortening_over_minimum_diameter = 0.5
    settings.maximum_penetration = 1.5e-4
    settings.maximum_bend_ratio = 1.001
    settings.maximum_steps = 32
    return settings


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 1.0])
    planar = tangle.generate_fiber_population(cell, in_plane_settings(), name="planar layers")
    through = tangle.generate_fiber_population(
        cell, through_thickness_settings(), name="through-thickness fibers"
    )
    recipe = tangle.Recipe(cell, layer_axis=2)
    recipe.insert(planar)
    recipe.relax_for(80)
    for scale, iterations in [(0.60, 60), (0.30, 60), (0.16, 80), (0.09, 120)]:
        recipe.move_layers(scale, stiffness=1.0, max_translation=0.08)
        recipe.relax_for(iterations)
    recipe.capture_junctions(planar_bonding())
    recipe.insert(through)
    recipe.relax_and_capture(160, 40, through_thickness_bonding())
    recipe.release_layer_targets()
    recipe.compact(final_compaction())

    relaxation = tangle.RelaxationSettings()
    relaxation.max_iterations = 1_500
    relaxation.iterations_per_batch = 20
    return recipe, relaxation


def run() -> tangle.RunResult:
    recipe, settings = build()
    result = recipe.run(settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.write_ovito(
        OUTPUT / "final.dump",
        view_script_path=OUTPUT / "view.py",
        session_path=OUTPUT / "final.ovito",
        coloring="curvature_ratio",
    )
    result.export_bpm(OUTPUT / "capsules.data", density=1_800.0)
    return result


if __name__ == "__main__":
    print(run())
