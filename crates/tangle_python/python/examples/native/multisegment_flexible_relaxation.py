"""All four intrinsic-versus-placed configurations from the Rust tutorial.

Python equivalent of ``examples/multisegment_flexible_relaxation``.
"""

from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "multisegment_flexible_relaxation"
FIBER = tangle.Material("fiber", diameter=0.036, min_bend_radius=0.12)

CASES = {
    "straight_intrinsic_straight_placed": {
        "description": "straight now; wants to be straight",
        "rest_shape": "straight",
        "rest_amplitude": 0.0,
        "placed_shape": "straight",
        "placed_amplitude": 0.0,
        "placed_chord_fraction": 1.0,
    },
    "straight_intrinsic_curved_placed": {
        "description": "curved now; wants to become straight",
        "rest_shape": "straight",
        "rest_amplitude": 0.0,
        "placed_shape": "curved",
        "placed_amplitude": 0.04,
        "placed_chord_fraction": 0.9,
    },
    "curved_intrinsic_curved_placed": {
        "description": "curved now; wants to retain its natural curve",
        "rest_shape": "curved",
        "rest_amplitude": 0.04,
        "placed_shape": "curved",
        "placed_amplitude": 0.04,
        "placed_chord_fraction": 1.0,
    },
    "straight_intrinsic_overbent_placed": {
        "description": "starts beyond its bend limit; relaxation must smooth it",
        "rest_shape": "straight",
        "rest_amplitude": 0.0,
        "placed_shape": "curved",
        "placed_amplitude": 0.12,
        "placed_chord_fraction": 0.7,
    },
}


def build(case_name: str) -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    case = CASES[case_name]
    cell = tangle.Cell([1.0, 1.0, 1.0])
    fibers = tangle.generate_multisegment_crossing(
        cell,
        material=FIBER,
        count=8,
        segments_per_fiber=8,
        length=0.7,
        placed_chord_fraction=case["placed_chord_fraction"],
        rest_shape=case["rest_shape"],
        rest_amplitude=case["rest_amplitude"],
        placed_shape=case["placed_shape"],
        placed_amplitude=case["placed_amplitude"],
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(fibers)
    recipe.relax_until_converged(max_iterations=5_000)

    settings = tangle.RelaxationSettings(
        correction_fraction=1.0,
        stretch_stiffness=0.35,
        bend_stiffness=0.03,
        curvature_limit_stiffness=1.0,
        curvature_ratio_tolerance=1.0e-4,
        max_step=0.008,
        max_iterations=5_000,
        iterations_per_batch=128,
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
        coloring="curvature_ratio",
    )
    result.export_bpm(directory / "capsules.data", density=1_800.0)
    return result


if __name__ == "__main__":
    for name, case in CASES.items():
        print(f"\n{name}: {case['description']}")
        print(run_case(name))
