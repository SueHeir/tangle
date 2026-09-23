"""Configuration checks for the periodic domain-size sweep example."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import periodic_domain_sweep as sweep  # noqa: E402


class PeriodicDomainSweepTests(unittest.TestCase):
    def test_every_side_builds_within_budget_and_image_clearance(self):
        for side in sweep.DOMAIN_SIDES:
            with self.subTest(side=side):
                recipe, settings, config = sweep.build(side)
                self.assertLessEqual(config.segment_count, 5_000)
                self.assertLess(
                    config.segment_length + config.diameter, 0.5 * config.side
                )
                self.assertIsNone(settings.adaptive_segmentation)
                self.assertAlmostEqual(
                    config.fiber_count / side**2, sweep.FIBERS_PER_AREA
                )
                self.assertTrue(recipe.operations()[0].startswith("insert"))

    def test_fiber_length_and_layers_are_preserved(self):
        config = sweep.SweepConfig(side=2.0)
        collection = sweep.straight_layer_fibers(
            config, layer=1, count=3, rng=sweep.random.Random(1)
        )
        self.assertEqual(collection.layer_ids(), [1])
        for centerline in collection.centerlines():
            self.assertEqual(len(centerline), config.segments_per_fiber + 1)
            chord = sum(
                (a - b) ** 2 for a, b in zip(centerline[-1], centerline[0])
            ) ** 0.5
            self.assertAlmostEqual(chord, config.fiber_length, places=6)

    def test_cells_too_small_for_the_diameter_are_rejected(self):
        with self.assertRaises(ValueError):
            sweep.build(0.5)


if __name__ == "__main__":
    unittest.main()
