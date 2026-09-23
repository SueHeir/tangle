"""Python equivalent of the Rust fibers-through-center-point example."""

from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "center_point"


def build() -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    cell = tangle.Cell([1.0, 1.0, 1.0])
    fibers = tangle.generate_point_crossing(
        cell,
        count=8,
        length=0.7,
        radius=0.025,
        material_name="fiber",
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(fibers)
    recipe.relax(maximum_iterations=2_000)

    settings = tangle.RelaxationSettings()
    settings.motion_model = "rigid_translation"
    settings.penetration_tolerance = 1.0e-6
    settings.correction_fraction = 0.75
    settings.max_step = 0.01
    return recipe, settings


def run() -> tangle.RunResult:
    recipe, settings = build()
    result = recipe.run(settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.write_ovito(
        OUTPUT / "center_point.dump",
        view_script_path=OUTPUT / "center_point_view.py",
        session_path=OUTPUT / "center_point.ovito",
    )
    result.export_bpm(OUTPUT / "center_point_capsules.data", density=1_800.0)
    return result


if __name__ == "__main__":
    print(run())
