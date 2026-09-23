"""The three biased-population cases and configurable stress-size variant."""

import argparse
import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "biased_fiber_box"
CASES = ("isotropic_3d", "planar_layered", "aligned_x")


def population_settings(case_name: str, count: int = 160, segments: int = 8):
    settings = tangle.FiberPopulationSettings()
    settings.count = count
    settings.segments_per_fiber = segments
    settings.seed = 20_260_918
    settings.length_minimum = 0.4
    settings.length_maximum = 0.58
    settings.radius_minimum = 0.012
    settings.radius_maximum = 0.017
    settings.curvature_amplitude_minimum = 0.0
    settings.curvature_amplitude_maximum = 0.014
    settings.minimum_bend_radius = 0.08
    settings.max_attempts_per_fiber = 256
    if case_name == "isotropic_3d":
        settings.orientation = "isotropic_3d"
        settings.position = "uniform"
    elif case_name == "planar_layered":
        settings.orientation = "planar"
        settings.orientation_axis = [0.0, 0.0, 1.0]
        settings.maximum_tilt = math.radians(10.0)
        settings.position = "layered"
        settings.position_axis = 2
        settings.layers = 4
        settings.jitter_fraction = 0.25
    elif case_name == "aligned_x":
        settings.orientation = "aligned"
        settings.orientation_axis = [1.0, 0.0, 0.0]
        settings.maximum_angle = math.radians(15.0)
        settings.position = "uniform"
    else:
        raise ValueError(f"unknown case {case_name!r}")
    return settings


def build(
    case_name: str,
    count: int = 160,
    segments: int = 8,
    layered_formation: bool = True,
    max_iterations: int = 2_000,
    batch_iterations: int = 128,
    initial_layer_iterations: int = 128,
    compaction_steps: int = 4,
    layer_iterations_per_step: int = 128,
    final_layer_spacing_scale: float = 0.8,
) -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 1.0])
    population = tangle.generate_fiber_population(
        cell, population_settings(case_name, count, segments), name=case_name
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(population)
    if case_name == "planar_layered" and layered_formation:
        recipe.relax_for(initial_layer_iterations)
        for step in range(1, compaction_steps + 1):
            scale = 1.0 + (final_layer_spacing_scale - 1.0) * step / compaction_steps
            recipe.move_layers(scale, stiffness=0.5, max_translation=0.01)
            recipe.relax_for(layer_iterations_per_step)
        recipe.release_layer_targets()
    recipe.relax(maximum_iterations=max_iterations)
    settings = tangle.RelaxationSettings()
    settings.max_iterations = max_iterations
    settings.iterations_per_batch = batch_iterations
    return recipe, settings


def run_case(case_name: str, count: int = 160, segments: int = 8, **options):
    recipe, settings = build(case_name, count, segments, **options)
    result = recipe.run(settings)
    directory = OUTPUT / case_name
    directory.mkdir(parents=True, exist_ok=True)
    result.write_ovito(
        directory / "relaxation.dump",
        view_script_path=directory / "view.py",
        session_path=directory / "relaxation.ovito",
        coloring="curvature_ratio",
    )
    result.export_bpm(directory / "capsules.data", density=1_800.0)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--fiber-count", type=int, default=160)
    parser.add_argument("--segments-per-fiber", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--batch-iterations", type=int, default=128)
    parser.add_argument("--no-layered-formation", action="store_true")
    parser.add_argument("--initial-layer-iterations", type=int, default=128)
    parser.add_argument("--compaction-steps", type=int, default=4)
    parser.add_argument("--layer-iterations-per-step", type=int, default=128)
    parser.add_argument("--final-layer-spacing-scale", type=float, default=0.8)
    args = parser.parse_args()
    for name in (args.case,) if args.case else CASES:
        print(
            name,
            run_case(
                name,
                args.fiber_count,
                args.segments_per_fiber,
                layered_formation=not args.no_layered_formation,
                max_iterations=args.max_iterations,
                batch_iterations=args.batch_iterations,
                initial_layer_iterations=args.initial_layer_iterations,
                compaction_steps=args.compaction_steps,
                layer_iterations_per_step=args.layer_iterations_per_step,
                final_layer_spacing_scale=args.final_layer_spacing_scale,
            ),
        )
