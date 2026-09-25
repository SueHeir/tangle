"""Stack plies of flat oval fibers, relax them, and export the result.

Each fiber is 30 um wide and 20 um thick (aspect ratio 1.5). Ovals start with
their long axis lying flat in the ply; contact during relaxation can tip
them about their own axis.
"""

import math
import random
from pathlib import Path

import tangle
from tangle.units import mm, um


def oval_ply(name: str, layer: int, count: int, seed: int) -> tangle.FiberCollection:
    rng = random.Random(seed)
    material = tangle.Material("oval fiber", diameter=30 * um, thickness=20 * um)
    collection = tangle.FiberCollection(name)
    for _ in range(count):
        angle = rng.uniform(0.0, math.tau)
        direction = [math.cos(angle), math.sin(angle), 0.0]
        center = [rng.uniform(0.2 * mm, 0.8 * mm), rng.uniform(0.2 * mm, 0.8 * mm), 0.0]
        half_length = 0.35 * mm
        points = [
            [center[i] + t * half_length * direction[i] for i in range(3)]
            for t in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        collection.add_fiber(points, material, formation_layer=layer)
    return collection


recipe = tangle.Recipe(tangle.Cell([1 * mm, 1 * mm, 1 * mm], periodic="xy"))
for layer in range(3):
    recipe.insert(
        oval_ply(f"ply_{layer}", layer, count=12, seed=200 + layer),
        translation=[0.0, 0.0, 0.3 * mm + layer * 15 * um],
    )
recipe.relax_until_converged(max_iterations=4_000)

result = recipe.run(
    tangle.RelaxationSettings(
        max_iterations=10_000,
        penetration_tolerance=0.1 * um,
        max_step=2 * um,
    )
)
print(result)

# How far the long axes tipped out of the ply plane during relaxation.
tilts = [
    math.degrees(math.asin(min(1.0, abs(axis[2]))))
    for fiber in result.assembly.long_axes()
    for axis in fiber
]
print(f"long-axis tilt: mean {sum(tilts) / len(tilts):.2f} deg, max {max(tilts):.2f} deg")

output = Path(__file__).parent / "output"
result.write_ovito(
    output / "oval_fibers.dump",
    view_script_path=output / "oval_fibers_view.py",
    session_path=output / "oval_fibers.ovito",
)
result.export_puma(output / "oval_fibers_puma", voxel_size=5 * um)
