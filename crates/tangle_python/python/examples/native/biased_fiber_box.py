"""The three biased-population cases and configurable stress-size variant.

Python equivalent of ``examples/biased_fiber_box`` and its
``biased_fiber_stress`` companion binary.
"""

import argparse
import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "biased_fiber_box"
CASES = ("isotropic_3d", "planar_layered", "aligned_x")
# Each fiber draws its own diameter from the population's range.
FIBER = tangle.Material("fiber", diameter=0.029, min_bend_radius=0.08)


def fiber_population(
    case_name: str, count: int = 160, segments: int = 8
) -> tangle.FiberPopulation:
    population = tangle.FiberPopulation(
        material=FIBER,
        count=count,
        segments_per_fiber=segments,
        seed=20_260_918,
        length=(0.4, 0.58),
        diameter=(0.024, 0.034),
        curvature_amplitude=(0.0, 0.014),
        max_attempts_per_fiber=256,
    )
    if case_name == "isotropic_3d":
        return population.replace(
            orientation=tangle.IsotropicOrientation(),
            position=tangle.UniformPosition(),
        )
    if case_name == "planar_layered":
        return population.replace(
            orientation=tangle.PlanarOrientation(max_tilt=math.radians(10.0)),
            position=tangle.LayeredPosition(4, jitter_fraction=0.25),
        )
    if case_name == "aligned_x":
        return population.replace(
            orientation=tangle.AlignedOrientation("x", max_angle=math.radians(15.0)),
            position=tangle.UniformPosition(),
        )
    raise ValueError(f"unknown case {case_name!r}")


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
        cell, fiber_population(case_name, count, segments), name=case_name
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(population)
    if case_name == "planar_layered" and layered_formation:
        recipe.relax_for(initial_layer_iterations)
        for step in range(1, compaction_steps + 1):
            scale = 1.0 + (final_layer_spacing_scale - 1.0) * step / compaction_steps
            recipe.scale_layer_spacing(scale, stiffness=0.5, max_translation=0.01)
            recipe.relax_for(layer_iterations_per_step)
        recipe.release_layer_placement()
    recipe.relax_until_converged(max_iterations=max_iterations)
    settings = tangle.RelaxationSettings(
        max_iterations=max_iterations, iterations_per_batch=batch_iterations
    )
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
