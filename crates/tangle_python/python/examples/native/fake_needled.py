"""Python form of the six-layer educational needled-felt recipe."""

import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "fake_needled"
LAYER_COUNT = 6
LAYER_GAP = 0.040


def population_settings() -> tangle.FiberPopulationSettings:
    settings = tangle.FiberPopulationSettings()
    settings.count = 180
    settings.segments_per_fiber = 10
    settings.seed = 20_260_929
    settings.length_minimum, settings.length_maximum = 0.36, 0.48
    settings.radius_minimum, settings.radius_maximum = 0.016, 0.021
    settings.curvature_amplitude_minimum = 0.0
    settings.curvature_amplitude_maximum = 0.006
    settings.orientation = "planar"
    settings.orientation_axis = [0.0, 0.0, 1.0]
    settings.maximum_tilt = math.radians(6.0)
    settings.position = "layered"
    settings.position_axis = 2
    settings.layers = LAYER_COUNT
    settings.jitter_fraction = 0.12
    settings.minimum_bend_radius = 0.050
    settings.max_attempts_per_fiber = 1_024
    settings.material_name = "needled felt fiber"
    return settings


def final_compaction() -> tangle.CompactionSettings:
    settings = tangle.CompactionSettings.volume_fraction(0.40, axis_weights=[0.0, 0.0, 1.0])
    settings.kinematics = "moving_walls"
    settings.cell_anchor = [0.0, 0.0, 0.0]
    settings.initial_log_strain = 0.02
    settings.minimum_log_strain = 0.005
    settings.maximum_log_strain = 0.04
    settings.growth_factor = 1.2
    settings.shrink_factor = 0.5
    settings.relax_iterations = 60
    settings.maximum_shortening_over_minimum_diameter = 0.5
    settings.maximum_penetration = 1.0e-3
    settings.maximum_bend_ratio = 1.001
    settings.maximum_steps = 128
    settings.maximum_relax_windows = 20
    return settings


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 0.32])
    population = tangle.generate_fiber_population(cell, population_settings(), name="batting")
    recipe = tangle.Recipe(cell, layer_axis=2)

    recipe.insert(population.select_layer(0))
    recipe.insert(population.select_layer(1))
    recipe.relax_for(80)
    recipe.place_layer_above(1, LAYER_GAP, stiffness=1.0, max_translation=0.08)
    recipe.relax_for(120)
    for layer in range(2, LAYER_COUNT):
        recipe.insert(population.select_layer(layer))
        recipe.place_layer_above(layer, LAYER_GAP, stiffness=1.0, max_translation=0.08)
        recipe.relax_for(100)
        recipe.release_layer_targets()
        recipe.needle_layer_random(
            layer,
            0.30,
            2.0 * LAYER_GAP,
            seed=20_260_930 + layer,
            stiffness=1.0,
            max_translation=0.012,
            maximum_translation_over_fiber_diameter=0.25,
        )
        recipe.relax_for(160)
        recipe.release_needles()
        recipe.relax_for(40)
    recipe.release_needles()
    recipe.release_layer_targets()
    recipe.compact(final_compaction())

    relaxation = tangle.RelaxationSettings()
    relaxation.constraint_iterations = 4
    relaxation.max_iterations = 24_000
    relaxation.iterations_per_batch = 20
    relaxation.penetration_tolerance = 5.0e-4
    relaxation.max_step = 0.005
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
