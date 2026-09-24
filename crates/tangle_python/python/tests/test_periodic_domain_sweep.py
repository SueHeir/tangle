"""Configuration checks for the periodic domain-size sweep example."""

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import periodic_domain_sweep as sweep  # noqa: E402


class PeriodicDomainSweepTests(unittest.TestCase):
    def test_every_side_builds_within_budget_and_image_clearance(self):
        for side in (*sweep.DOMAIN_SIDES, 1.0):
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

    def test_fibers_keep_length_and_fit_the_placement_cell(self):
        for side in (1.0, 2.0, 10.0):
            config = sweep.SweepConfig(side=side)
            placements = sweep.sample_fibers(config, sweep.random.Random(1))
            collection = sweep.fiber_collection(config, placements)
            radius = 0.5 * config.diameter
            with self.subTest(side=side):
                self.assertEqual(len(placements), config.fiber_count)
                for centerline in collection.centerlines():
                    self.assertEqual(len(centerline), config.segments_per_fiber + 1)
                    for a, b in zip(centerline, centerline[1:]):
                        # Centerlines are stored in single precision.
                        self.assertAlmostEqual(
                            math.dist(a, b), config.segment_length, places=5
                        )
                    heights = [point[2] for point in centerline]
                    self.assertGreaterEqual(min(heights), radius - 1e-5)
                    self.assertLessEqual(
                        max(heights), config.placement_thickness - radius + 1e-5
                    )

    def test_tilt_stays_within_the_bias_unless_a_fiber_meets_itself(self):
        config = sweep.SweepConfig(side=10.0)
        placements = sweep.sample_fibers(config, sweep.random.Random(4))
        limit = math.radians(config.max_tilt)
        for placement in placements:
            floor = sweep.min_tilt(config, placement.angle)
            with self.subTest(angle=placement.angle):
                self.assertLessEqual(abs(placement.tilt), max(limit, floor) + 1e-12)
                self.assertGreaterEqual(abs(placement.tilt), floor - 1e-12)
        self.assertGreater(max(abs(p.tilt) for p in placements), 0.5 * limit)

    def test_only_self_touching_fibers_need_a_minimum_tilt(self):
        side_one = sweep.SweepConfig(side=1.0)
        side_ten = sweep.SweepConfig(side=10.0)
        for angle in (0.1, 0.5, 1.0, 1.4, 2.5):
            with self.subTest(angle=angle):
                self.assertGreater(sweep.min_tilt(side_one, angle), 0.0)
                self.assertEqual(sweep.min_tilt(side_ten, angle), 0.0)
        # Along a lattice direction a length-10 fiber meets its image on a
        # side-2 cell after one wrap.
        side_two = sweep.SweepConfig(side=2.0)
        self.assertAlmostEqual(sweep.self_contact_arc(side_two, 0.0), 2.0)
        # Just past one fiber length, the end meets the start of its image.
        odd = sweep.SweepConfig(side=1.97)
        self.assertGreater(sweep.min_tilt(odd, math.atan2(5.0, -1.0)), 0.0)

    def test_self_touching_strands_clear_each_other(self):
        config = sweep.SweepConfig(side=1.0)
        placements = sweep.sample_fibers(config, sweep.random.Random(3))
        for centerline in sweep.fiber_collection(config, placements).centerlines():
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

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            sweep.build(0.5)
        with self.assertRaises(ValueError):
            sweep.build(1.0, diameter=0.25)
        with self.assertRaises(ValueError):
            sweep.build(10.0, placement_volume_fraction=0.3)
        with self.assertRaises(ValueError):
            sweep.build(10.0, max_tilt=90.0)
        # Too steep for the placement cell, whatever the seed.
        with self.assertRaises(ValueError):
            sweep.build(2.0, max_tilt=38.0)
        with self.assertRaises(ValueError):
            sweep.build(2.0, fibers_per_area=0.5, placement_volume_fraction=0.2)


if __name__ == "__main__":
    unittest.main()
