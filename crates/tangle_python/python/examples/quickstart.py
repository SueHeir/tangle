"""Quick start: build a small fiber network, then fit it back from a simulated CT scan.

Run from the repository root after installing the package:

    python crates/tangle_python/python/examples/quickstart.py

Without a GPU, pass ``tangle.RelaxationSettings(backend="cpu")`` to ``run``.
"""

from pathlib import Path

import tangle
import tangle.ct as ct
from tangle.units import um

output = Path(__file__).parent / "output" / "quickstart"

# 1. A network: 40 random 10 µm fibers in a periodic 120 µm box, relaxed so
#    that no two fibers overlap.
cell = tangle.Cell([120 * um] * 3, periodic="xyz")
fiber = tangle.Material("fiber", diameter=10 * um, min_bend_radius=50 * um)
population = tangle.FiberPopulation(material=fiber, count=40, length=(60 * um, 100 * um))

recipe = tangle.Recipe(cell)
recipe.insert(tangle.generate_fiber_population(cell, population))
recipe.relax_until_converged()
result = recipe.run()  # solver lengths default to fractions of the fiber diameter
print(f"relaxed {result.fiber_count} fibers in {result.iterations} iterations, converged={result.converged}")

result.write_ovito(output / "network.dump")

# 2. A CT scan of that network (blur and noise), and the fibers fitted back.
scan = ct.synthetic_ct(result, voxel_size=1.5 * um, seed=1)
fit = ct.fit_fibers(scan.volume, scan.voxel_size, ct.FiberSpec(diameter=10 * um))
print(f"fitted {len(fit.centerlines)} fibers")
score = ct.score(fit, scan)
print(f"recovered {score['recovered']} of {score['true_fibers_in_volume']} true fibers, F1 {score['f1']:.2f}")

fit.write(output / "fit", volume=scan.volume)
