"""Physical-scale two-material needled preform matching the Rust configuration.

Python equivalent of ``examples/needled_preform_two_fiber``.
"""

import math
from pathlib import Path

import tangle
from tangle.units import mm, um


OUTPUT = Path(__file__).parent / "output" / "needled_preform_two_fiber"
FOOTPRINT = 2.5 * mm
INITIAL_THICKNESS = 4.8 * mm
LAYER_COUNT = 6
LAYER_GAP = 50 * um

FINE = tangle.Material("fine_7um", diameter=7.0e-6, min_bend_radius=500.0e-6)
COARSE = tangle.Material("coarse_19um", diameter=19.0e-6, min_bend_radius=60.0e-6)

# Formation steps may leave curvature slightly above the limit and carry on
# when their budget runs out, as long as the hard contact limit holds.
FORMATION = tangle.SolvePolicy(
    "formation relaxation",
    target_penetration=0.30 * um,
    target_curvature_ratio=1.00001,
    max_curvature_ratio=1.05,
    hard_curvature=False,
    max_iterations=15_000,
    on_budget_exhausted="continue_if_hard_ok",
)
FINAL = tangle.SolvePolicy(
    "final hard relaxation",
    target_penetration=0.30 * um,
    target_curvature_ratio=1.00001,
    max_curvature_ratio=1.001,
    max_iterations=100_000,
)


def ply_population(
    material: tangle.Material, seed: int, max_curvature_amplitude: float
) -> tangle.FiberPopulation:
    return tangle.FiberPopulation(
        material=material,
        count=188,
        segments_per_fiber=4,
        seed=seed,
        nominal_parent_length=0.0508,
        length=(3 * mm, 4 * mm),
        curvature_amplitude=(0.0, max_curvature_amplitude),
        orientation=tangle.LayeredBiaxialOrientation(
            primary_fraction=0.40,
            cross_fraction=0.40,
            max_in_plane_deviation=math.pi / 18.0,
            max_tilt=math.pi / 1_800.0,
            seed=20_260_941,
        ),
        position=tangle.LayeredPosition(LAYER_COUNT, jitter_fraction=0.25),
        max_attempts_per_fiber=2_048,
    )


def final_compaction() -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        0.13,
        kinematics="moving_walls",
        cell_anchor=[0.0, 0.0, 0.0],
        initial_log_strain=0.01,
        min_log_strain=0.001,
        max_log_strain=0.02,
        growth_factor=1.2,
        shrink_factor=0.5,
        relax_iterations=1_000,
        max_shortening_over_min_diameter=0.5,
        max_penetration=0.301 * um,
        max_curvature_ratio=1.001,
        max_steps=256,
        max_relax_windows=8,
    )


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([FOOTPRINT, FOOTPRINT, INITIAL_THICKNESS], periodic="xy")
    fine = tangle.generate_fiber_population(
        cell, ply_population(FINE, 20_260_938, 2 * um), name="fine"
    )
    coarse = tangle.generate_fiber_population(
        cell, ply_population(COARSE, 20_260_939, 8 * um), name="coarse"
    )
    recipe = tangle.Recipe(cell)
    for layer in range(LAYER_COUNT):
        recipe.insert(
            fine.select_layer(layer, name=f"mixed ply {layer}")
            + coarse.select_layer(layer)
        )
        recipe.solve(FORMATION)
        if layer == 0:
            continue
        # Approach in two steps: first to 65 um, then to the final gap.
        for gap in (65 * um, LAYER_GAP):
            with recipe.place_layer_above(
                layer, gap=gap, stiffness=1.0, max_translation=2 * um
            ):
                recipe.settle_targets(tolerance=0.5 * um, max_iterations=3_000)
                recipe.relax_for(100)
            recipe.solve(FORMATION)
        if layer >= 2:
            with recipe.needle_layer(
                layer,
                footprint=tangle.CircularFootprint.random(
                    diameter=150 * um, seed=20_260_940
                ),
                depth=7.0 * LAYER_GAP,
                min_fiber_diameter=0.99 * COARSE.diameter,
                stiffness=0.5,
                max_translation=1 * um,
                max_translation_over_diameter=0.25,
            ):
                recipe.settle_targets(tolerance=0.5 * um, max_iterations=3_000)
                recipe.relax_for(150)
            recipe.solve(FORMATION)
    recipe.solve(FINAL)
    recipe.compact(final_compaction())
    recipe.solve(FINAL)

    settings = tangle.RelaxationSettings(
        constraint_iterations=8,
        curvature_limit_safety_margin=0.06,
        curvature_cleanup_sweeps=4,
        max_iterations=500_000,
        iterations_per_batch=10,
        penetration_tolerance=0.30e-6,
        correction_fraction=0.35,
        max_step=2.0e-6,
        adaptive_segmentation=tangle.AdaptiveSegmentationSettings(
            contact_length_over_diameter=4.0,
            min_length_over_diameter=2.0,
            max_refinement_levels=6,
            refinement_interval=32,
            refinement_persistence=3,
            coarsening_persistence=8,
        ),
    )
    return recipe, settings


def run(resume: bool = False) -> tangle.RunResult:
    recipe, settings = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = tangle.CheckpointSettings(
        "needled_preform_two_fiber-v19-staged-solve-policy",
        OUTPUT / "needled_preform_two_fiber.restart",
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
