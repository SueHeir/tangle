"""Python equivalent of the Rust ``examples/fibers_through_center_point`` example."""

from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "fibers_through_center_point"


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 1.0])
    fibers = tangle.generate_point_crossing(
        cell,
        material=tangle.Material("fiber", diameter=0.05),
        count=8,
        length=0.7,
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(fibers)
    recipe.relax_until_converged(max_iterations=2_000)

    settings = tangle.RelaxationSettings(
        motion_model="rigid_translation",
        penetration_tolerance=1.0e-6,
        correction_fraction=0.75,
        max_step=0.01,
    )
    return recipe, settings


def run() -> tangle.RunResult:
    recipe, settings = build()
    result = recipe.run(settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.write_ovito(
        OUTPUT / "fibers_through_center_point.dump",
        view_script_path=OUTPUT / "fibers_through_center_point_view.py",
        session_path=OUTPUT / "fibers_through_center_point.ovito",
    )
    result.export_bpm(
        OUTPUT / "fibers_through_center_point_capsules.data", density=1_800.0
    )
    return result


if __name__ == "__main__":
    print(run())
