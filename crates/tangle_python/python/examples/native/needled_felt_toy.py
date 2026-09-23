"""Python form of the six-layer educational needled-felt recipe.

Python equivalent of ``examples/needled_felt_toy``. Lengths are unitless.
"""

import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "needled_felt_toy"
LAYER_COUNT = 6
LAYER_GAP = 0.040
FIBER = tangle.Material("needled felt fiber", diameter=0.037, min_bend_radius=0.050)


def fiber_population() -> tangle.FiberPopulation:
    return tangle.FiberPopulation(
        material=FIBER,
        count=180,
        segments_per_fiber=10,
        seed=20_260_929,
        length=(0.36, 0.48),
        diameter=(0.032, 0.042),
        curvature_amplitude=(0.0, 0.006),
        orientation=tangle.PlanarOrientation(max_tilt=math.radians(6.0)),
        position=tangle.LayeredPosition(LAYER_COUNT, jitter_fraction=0.12),
        max_attempts_per_fiber=1_024,
    )


def final_compaction() -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        0.40,
        kinematics="moving_walls",
        cell_anchor=[0.0, 0.0, 0.0],
        initial_log_strain=0.02,
        min_log_strain=0.005,
        max_log_strain=0.04,
        growth_factor=1.2,
        shrink_factor=0.5,
        relax_iterations=60,
        max_shortening_over_min_diameter=0.5,
        max_penetration=1.0e-3,
        max_curvature_ratio=1.001,
        max_steps=128,
        max_relax_windows=20,
    )


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 0.32])
    population = tangle.generate_fiber_population(cell, fiber_population(), name="batting")
    recipe = tangle.Recipe(cell)

    recipe.insert(population.select_layer(0))
    recipe.insert(population.select_layer(1))
    recipe.relax_for(80)
    # Layer 1 stays held while layer 2 arrives; the first placement block in
    # the loop releases every held layer when it ends.
    recipe.place_layer_above(1, gap=LAYER_GAP, stiffness=1.0, max_translation=0.08)
    recipe.relax_for(120)
    for layer in range(2, LAYER_COUNT):
        recipe.insert(population.select_layer(layer))
        with recipe.place_layer_above(
            layer, gap=LAYER_GAP, stiffness=1.0, max_translation=0.08
        ):
            recipe.relax_for(100)
        with recipe.needle_layer(
            layer,
            footprint=tangle.RandomFiberFraction(0.30, seed=20_260_930 + layer),
            depth=2.0 * LAYER_GAP,
            stiffness=1.0,
            max_translation=0.012,
            max_translation_over_diameter=0.25,
        ):
            recipe.relax_for(160)
        recipe.relax_for(40)
    recipe.compact(final_compaction())

    relaxation = tangle.RelaxationSettings(
        constraint_iterations=4,
        max_iterations=24_000,
        iterations_per_batch=20,
        penetration_tolerance=5.0e-4,
        max_step=0.005,
    )
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
