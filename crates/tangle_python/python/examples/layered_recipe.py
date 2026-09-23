"""Build collections in Python and insert them as separate recipe stages."""

import math
import random

import tangle


def planar_layer(name: str, layer: int, count: int, seed: int) -> tangle.FiberCollection:
    rng = random.Random(seed)
    material = tangle.Material("felt fiber", diameter=19.0e-6)
    collection = tangle.FiberCollection(name)
    for _ in range(count):
        angle = rng.uniform(0.0, math.tau)
        direction = [math.cos(angle), math.sin(angle), 0.0]
        center = [rng.uniform(0.2e-3, 0.8e-3), rng.uniform(0.2e-3, 0.8e-3), 0.0]
        half_length = 0.35e-3
        collection.add_fiber(
            [
                [center[i] - half_length * direction[i] for i in range(3)],
                [center[i] + half_length * direction[i] for i in range(3)],
            ],
            material,
            formation_layer=layer,
            tags={"ply": str(layer)},
        )
    return collection


# The plies stack along z, the cell's default stack axis.
recipe = tangle.Recipe(tangle.Cell([1.0e-3, 1.0e-3, 2.0e-3]))
for layer in range(3):
    recipe.insert(
        planar_layer(f"ply_{layer}", layer, count=12, seed=100 + layer),
        translation=[0.0, 0.0, (layer + 1) * 0.4e-3],
    )
    recipe.relax_until_converged(max_iterations=2_000)

settings = tangle.RelaxationSettings(
    max_iterations=10_000,
    penetration_tolerance=0.1e-6,
    max_step=2.0e-6,
    adaptive_segmentation=tangle.AdaptiveSegmentationSettings.profile("fast"),
)

print("Recipe operations:")
for operation in recipe.operations():
    print(" -", operation)

result = recipe.run(settings)
print(result)
