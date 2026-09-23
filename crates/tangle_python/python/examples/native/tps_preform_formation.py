"""Python form of the modular insert/move/relax TPS preform recipe.

Python equivalent of ``examples/tps_preform_formation``. Lengths are unitless.
"""

import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "tps_preform_formation"
IN_PLANE = tangle.Material("in-plane felt fiber", diameter=0.015, min_bend_radius=0.06)
THROUGH_THICKNESS = tangle.Material(
    "through-thickness fiber", diameter=0.012, min_bend_radius=0.06
)


def in_plane_population() -> tangle.FiberPopulation:
    return tangle.FiberPopulation(
        material=IN_PLANE,
        count=96,
        segments_per_fiber=6,
        seed=20_260_919,
        length=(0.28, 0.42),
        diameter=(0.012, 0.018),
        curvature_amplitude=(0.0, 0.008),
        orientation=tangle.PlanarOrientation(max_tilt=math.radians(8.0)),
        position=tangle.LayeredPosition(6, jitter_fraction=0.18),
        max_attempts_per_fiber=512,
    )


def through_thickness_population() -> tangle.FiberPopulation:
    return tangle.FiberPopulation(
        material=THROUGH_THICKNESS,
        count=24,
        segments_per_fiber=6,
        seed=20_260_920,
        length=(0.50, 0.72),
        diameter=(0.010, 0.014),
        curvature_amplitude=(0.0, 0.006),
        orientation=tangle.AlignedOrientation("z", max_angle=math.radians(10.0)),
        position=tangle.UniformPosition(),
        max_attempts_per_fiber=512,
    )


def planar_bonding() -> tangle.JunctionPolicy:
    return tangle.JunctionPolicy(
        "planar consolidation",
        "felt contact bond",
        max_surface_gap=2.0e-4,
        min_crossing_angle=math.radians(12.0),
        probability=0.35,
        seed=20_260_921,
        material_pairs=[(IN_PLANE.name, IN_PLANE.name)],
    )


def through_thickness_bonding() -> tangle.JunctionPolicy:
    return tangle.JunctionPolicy(
        "through-thickness tying",
        "through-thickness tie",
        parameter_set=1,
        max_surface_gap=2.0e-4,
        min_crossing_angle=math.radians(20.0),
        probability=0.8,
        seed=20_260_922,
        material_pairs=[(IN_PLANE.name, THROUGH_THICKNESS.name)],
    )


def final_compaction() -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        0.01,
        kinematics="moving_walls",
        cell_anchor=[0.5, 0.5, 0.5],
        initial_log_strain=0.03,
        min_log_strain=0.002,
        max_log_strain=0.06,
        growth_factor=1.25,
        shrink_factor=0.5,
        relax_iterations=40,
        max_shortening_over_min_diameter=0.5,
        max_penetration=1.5e-4,
        max_curvature_ratio=1.001,
        max_steps=32,
    )


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 1.0])
    planar = tangle.generate_fiber_population(cell, in_plane_population(), name="planar layers")
    through = tangle.generate_fiber_population(
        cell, through_thickness_population(), name="through-thickness fibers"
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(planar)
    recipe.relax_for(80)
    # Each call retargets the held layers; they stay held through junction
    # capture and the through-thickness insertion, then release before compaction.
    for factor, iterations in [(0.60, 60), (0.30, 60), (0.16, 80), (0.09, 120)]:
        recipe.scale_layer_spacing(factor, stiffness=1.0, max_translation=0.08)
        recipe.relax_for(iterations)
    recipe.capture_junctions(planar_bonding())
    recipe.insert(through)
    recipe.relax_and_capture(
        iterations=160, capture_every=40, policy=through_thickness_bonding()
    )
    recipe.release_layer_placement()
    recipe.compact(final_compaction())

    relaxation = tangle.RelaxationSettings(max_iterations=1_500, iterations_per_batch=20)
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
