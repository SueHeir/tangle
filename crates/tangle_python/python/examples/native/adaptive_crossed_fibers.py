"""All three adaptive-crossing configurations from the Rust example.

Python equivalent of ``examples/adaptive_crossed_fibers``.
"""

from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "adaptive_crossed_fibers"
CASES = {
    "orthogonal_stop_early": dict(angle=90.0, separation_ratio=0.8, segments=1, adaptive=True),
    "shallow_uniform_reference": dict(angle=12.0, separation_ratio=0.1, segments=8, adaptive=False),
    "shallow_adaptive": dict(angle=12.0, separation_ratio=0.1, segments=1, adaptive=True),
}
FIBER = tangle.Material("fiber", diameter=0.05, min_bend_radius=0.1)


def build(case_name: str) -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    case = CASES[case_name]
    cell = tangle.Cell([1.0, 1.0, 1.0])
    fibers = tangle.generate_fiber_pair_crossing(
        cell,
        material=FIBER,
        segments_per_fiber=case["segments"],
        length=0.8,
        axis_separation=case["separation_ratio"] * FIBER.diameter,
        crossing_angle_degrees=case["angle"],
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(fibers)
    recipe.relax_until_converged(max_iterations=1_000)

    settings = tangle.RelaxationSettings(
        correction_fraction=1.0,
        stretch_stiffness=0.5,
        bend_stiffness=0.15,
        curvature_limit_stiffness=1.0,
        curvature_ratio_tolerance=1.0e-4,
        max_step=0.004,
        pin_fiber_ends=True,
        max_iterations=1_000,
        iterations_per_batch=17,
        constraint_iterations=4,
    )
    if case["adaptive"]:
        settings.adaptive_segmentation = tangle.AdaptiveSegmentationSettings(
            contact_length_over_diameter=2.0,
            min_length_over_diameter=1.0,
            max_refinement_levels=6,
            refinement_interval=3,
            refinement_persistence=2,
            coarsening_persistence=8,
        )
    return recipe, settings


def run_case(case_name: str) -> tangle.RunResult:
    recipe, settings = build(case_name)
    result = recipe.run(settings)
    directory = OUTPUT / case_name
    directory.mkdir(parents=True, exist_ok=True)
    result.write_ovito(
        directory / "relaxation.dump",
        view_script_path=directory / "view.py",
        session_path=directory / "relaxation.ovito",
        coloring="refinement_level",
    )
    result.export_bpm(directory / "capsules.data", density=1_800.0)
    return result


if __name__ == "__main__":
    for name in CASES:
        result = run_case(name)
        print(
            f"{name}: {result.active_segments} segments, "
            f"{result.segment_splits} splits, {result.segment_merges} merges, "
            f"{result.iterations} iterations"
        )
