"""Configuration checks for the periodic domain-size sweep example."""

import math
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
                self.assertGreaterEqual(
                    config.segment_length, config.diameter * (1 - 1e-9)
                )
                self.assertIsNone(settings.adaptive_segmentation)
                self.assertAlmostEqual(
                    config.fiber_count / side**2, sweep.FIBERS_PER_AREA
                )
                self.assertTrue(recipe.operations()[0].startswith("insert"))

    def test_fiber_length_and_layers_are_preserved(self):
        for side in (1.0, 2.0):
            config = sweep.SweepConfig(side=side)
            layers = sweep.sample_layers(config, sweep.random.Random(1))
            collection = sweep.layer_fibers(config, 1, layers[1], z=1.0)
            with self.subTest(side=side):
                self.assertEqual(collection.layer_ids(), [1])
                for centerline in collection.centerlines():
                    self.assertEqual(len(centerline), config.segments_per_fiber + 1)
                    arc = sum(
                        math.dist(a, b) for a, b in zip(centerline, centerline[1:])
                    )
                    # Centerlines are stored in single precision.
                    self.assertAlmostEqual(arc, config.fiber_length, places=4)

    def test_wavy_fibers_undulate_through_about_one_layer(self):
        config = sweep.SweepConfig(side=10.0)
        layers = sweep.sample_layers(config, sweep.random.Random(2))
        flat = [p for layer in layers for p in layer if not p.slope]
        self.assertTrue(flat)
        amplitude = config.wave_amplitude * config.diameter
        collection = sweep.layer_fibers(config, 0, flat[:5], z=1.0)
        for placement, centerline in zip(flat, collection.centerlines()):
            heights = [point[2] for point in centerline]
            with self.subTest(phase=placement.phase):
                self.assertAlmostEqual(placement.amplitude, amplitude)
                self.assertGreater(max(heights) - min(heights), 1.8 * amplitude)
                self.assertLessEqual(
                    max(heights) - min(heights), placement.rise(config) + 1e-4
                )
                self.assertGreaterEqual(min(heights), 1.0 - 1e-4)
                for a, b in zip(centerline, centerline[1:]):
                    self.assertAlmostEqual(
                        math.dist(a, b), config.segment_length, places=4
                    )

    def test_straight_option_reproduces_flat_layers(self):
        config = sweep.SweepConfig(side=10.0, wave_amplitude=0.0)
        layers = sweep.sample_layers(config, sweep.random.Random(2))
        (centerline,) = sweep.layer_fibers(
            config, 0, [p for p in layers[0] if not p.slope][:1], z=1.0
        ).centerlines()
        heights = [point[2] for point in centerline]
        self.assertLess(max(heights) - min(heights), 0.01 * config.diameter)

    def test_layer_counts_cover_every_fiber(self):
        for side in sweep.DOMAIN_SIDES:
            config = sweep.SweepConfig(side=side)
            counts = sweep.layer_counts(config)
            with self.subTest(side=side):
                self.assertEqual(sum(counts), config.fiber_count)
                self.assertLessEqual(max(counts) - min(counts), 1)
                self.assertGreaterEqual(min(counts), 1)

    def test_only_self_touching_fibers_ramp(self):
        side_one = sweep.SweepConfig(side=1.0)
        side_ten = sweep.SweepConfig(side=10.0)
        for angle in (0.1, 0.5, 1.0, 1.4, 2.5):
            with self.subTest(angle=angle):
                self.assertGreater(sweep.ramp_slope(side_one, angle), 0.0)
                self.assertEqual(sweep.ramp_slope(side_ten, angle), 0.0)
        # Along a lattice direction a length-10 fiber meets its image on a
        # side-2 cell after one wrap.
        side_two = sweep.SweepConfig(side=2.0)
        self.assertAlmostEqual(sweep.self_contact_arc(side_two, 0.0), 2.0)

    def test_ramped_strands_clear_each_other(self):
        config = sweep.SweepConfig(side=1.0)
        rng = sweep.random.Random(3)
        for layer in sweep.sample_layers(config, rng):
            (centerline,) = sweep.layer_fibers(
                config, 0, layer, z=1.0
            ).centerlines()
            points = []
            for a, b in zip(centerline, centerline[1:]):
                for fraction in (0.0, 0.25, 0.5, 0.75):
                    points.append([a[k] + fraction * (b[k] - a[k]) for k in range(3)])
            spacing = config.segment_length / 4
            closest = math.inf
            for i, p in enumerate(points):
                for j in range(i + 1, len(points)):
                    if (j - i) * spacing < 2.0 * config.diameter:
                        continue
                    q = points[j]
                    delta = [p[k] - q[k] for k in range(3)]
                    for k in range(2):
                        delta[k] -= round(delta[k] / config.side) * config.side
                    closest = min(closest, math.hypot(*delta))
            self.assertGreater(closest, config.diameter)

    def test_cells_too_small_for_the_diameter_are_rejected(self):
        with self.assertRaises(ValueError):
            sweep.build(0.5)
        with self.assertRaises(ValueError):
            sweep.build(1.0, diameter=0.25)


if __name__ == "__main__":
    unittest.main()
