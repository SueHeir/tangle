"""Smallest complete Python-to-CubeCL TANGLE workflow."""

from pathlib import Path

import tangle


fiber = tangle.Material("fiber", diameter=0.05)
crossing = tangle.FiberCollection("orthogonal crossing")
crossing.add_fiber(
    [[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]],
    fiber,
    tags={"family": "x"},
)
crossing.add_fiber(
    [[0.0, -0.35, 0.0], [0.0, 0.35, 0.0]],
    fiber,
    tags={"family": "y"},
)

recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
inserted = recipe.insert(crossing, translation=[0.5, 0.5, 0.5])
recipe.relax(maximum_iterations=2_000)

settings = tangle.RelaxationSettings()
settings.motion_model = "rigid_translation"
settings.penetration_tolerance = 1.0e-6
settings.max_iterations = 4_000
settings.max_step = 0.01

result = recipe.run(settings)
print(inserted)
print(result)

output = Path(__file__).parent / "output"
result.write_ovito(
    output / "crossed_fibers.dump",
    view_script_path=output / "crossed_fibers_view.py",
    session_path=output / "crossed_fibers.ovito",
)
result.export_bpm(output / "crossed_fibers_capsules.data", density=1_800.0)
