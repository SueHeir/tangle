"""Bond a FiberForm-like felt at its fiber crossings, then export the bonds.

Carbon-bonded carbon fiber insulation (FiberForm, CBCF) is chopped carbon
fiber, about 11 um across and a few hundred um long, laid mostly in-plane and
joined by carbonized phenolic binder where fibers cross. This example stacks
a few plies of such fibers, relaxes them, captures touching crossings as
junctions (a fraction of them, as the binder is uneven), and writes:

- an OVITO frame in which every junction is a grey binder bond;
- a PuMA bundle with binder as its own phase, plus bond_ids.vti and
  binder_interface.vti (see docs/fiber_bonds.md).

The bond radius is half the fiber radius. No published bond size exists,
so treat it as a setting to vary.

Usage: python bonded_fibers.py [output_directory]
"""

import math
import random
import sys
from pathlib import Path

import tangle
from tangle.units import um

BOX = 300 * um
DIAMETER = 11 * um
LENGTH = 250 * um
BOND_RADIUS_RATIO = 0.5


def ply(name: str, layer: int, count: int, seed: int) -> tangle.FiberCollection:
    rng = random.Random(seed)
    material = tangle.Material("carbon fiber", diameter=DIAMETER)
    collection = tangle.FiberCollection(name)
    for _ in range(count):
        angle = rng.uniform(0.0, math.pi)
        tilt = math.radians(rng.uniform(-5.0, 5.0))
        direction = [
            math.cos(angle) * math.cos(tilt),
            math.sin(angle) * math.cos(tilt),
            math.sin(tilt),
        ]
        center = [rng.uniform(0.0, BOX), rng.uniform(0.0, BOX), 0.0]
        points = [
            [center[i] + t * 0.5 * LENGTH * direction[i] for i in range(3)]
            for t in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        collection.add_fiber(points, material, formation_layer=layer)
    return collection


output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output"

recipe = tangle.Recipe(tangle.Cell([BOX, BOX, BOX], periodic="xy"))
for layer in range(6):
    recipe.insert(
        ply(f"ply_{layer}", layer, count=14, seed=300 + layer),
        translation=[0.0, 0.0, 0.35 * BOX + layer * 1.2 * DIAMETER],
    )
recipe.relax_until_converged(max_iterations=4_000)
# Binder joins fibers that touch after the felt has settled; only some
# crossings carry enough binder to bond.
recipe.capture_junctions(
    tangle.JunctionPolicy(
        "binder at crossings",
        "bond",
        max_surface_gap=0.5 * um,
        min_crossing_angle=math.radians(10),
        probability=0.7,
        seed=5,
        max_per_fiber_pair=1,
    )
)

result = recipe.run(
    tangle.RelaxationSettings(
        max_iterations=10_000,
        penetration_tolerance=0.1 * um,
        max_step=2 * um,
    )
)
print(result)
print(f"junctions (bonds): {result.junction_count}")

result.write_ovito(
    output / "bonded_fibers.dump",
    view_script_path=output / "bonded_fibers_view.py",
    session_path=output / "bonded_fibers.ovito",
    bond_radius_ratio=BOND_RADIUS_RATIO,
)
bundle = result.export_puma(
    output / "bonded_fibers_puma",
    voxel_size=1 * um,
    bond_radius_ratio=BOND_RADIUS_RATIO,
)
print(bundle)
print(
    f"bonds {bundle.bonds}, binder voxels {bundle.binder_voxels} "
    f"({bundle.binder_voxels / max(bundle.occupied_voxels, 1):.2%} of the solid)"
)
