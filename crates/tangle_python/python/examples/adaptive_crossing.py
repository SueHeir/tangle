"""Two initially two-point fibers with adaptive refinement and coarsening."""

import tangle


fiber = tangle.Material(
    "flexible fiber",
    diameter=20.0e-6,
    min_bend_radius=80.0e-6,
)
crossing = tangle.FiberCollection("adaptive crossing")
crossing.add_fiber([[-0.4e-3, 0.0, 0.0], [0.4e-3, 0.0, 0.0]], fiber)
crossing.add_fiber([[0.0, -0.4e-3, 0.0], [0.0, 0.4e-3, 0.0]], fiber)

recipe = tangle.Recipe(tangle.Cell([1.0e-3] * 3))
recipe.insert(crossing, translation=[0.5e-3] * 3)
recipe.relax_until_converged(max_iterations=10_000)

settings = tangle.RelaxationSettings(
    max_iterations=12_000,
    max_step=2.0e-6,
    penetration_tolerance=0.1e-6,
    correction_fraction=0.2,
    adaptive_segmentation=tangle.AdaptiveSegmentationSettings(
        refinement_interval=1,
        refinement_persistence=1,
    ),
)

result = recipe.run(settings)
print(result)
print(
    f"adaptive topology: {result.active_segments} active segments, "
    f"{result.segment_splits} splits, {result.segment_merges} merges"
)
