import tempfile
import unittest
from pathlib import Path

import tangle
from tangle.units import um

try:
    import numpy as np
    import scipy  # noqa: F401

    import tangle.ct as ct
except ImportError:  # the CT fitter needs NumPy and SciPy
    ct = None


DIAMETER = 10 * um
VOXEL = 1.25 * um


def crossing_scan():
    material = tangle.Material("fiber", diameter=DIAMETER)
    arc = [[(10 + 70 * t) * um, (20 + 4 * np.sin(np.pi * t)) * um, (20 + 6 * np.sin(np.pi * t)) * um] for t in np.linspace(0, 1, 15)]
    fibers = tangle.FiberCollection.from_centerlines(
        [
            [[8 * um, 40 * um, 40 * um], [82 * um, 44 * um, 40 * um]],
            [[45 * um, 8 * um, 52 * um], [47 * um, 82 * um, 52 * um]],
            arc,
        ],
        material,
    )
    assembly = tangle.Assembly(tangle.Cell([90 * um] * 3))
    assembly.insert(fibers)
    return ct.synthetic_ct(assembly, VOXEL, seed=3)


@unittest.skipIf(ct is None, "tangle.ct needs NumPy and SciPy")
class CtFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = crossing_scan()
        cls.fit = ct.fit_fibers(cls.scan.volume, VOXEL, ct.FiberSpec(diameter=DIAMETER))

    def test_synthetic_scan_has_ground_truth(self):
        self.assertEqual(self.scan.volume.shape, (72, 72, 72))
        self.assertEqual(int(self.scan.labels.max()), 3)
        self.assertEqual(len(self.scan.centerlines), 3)

    def test_fit_recovers_every_fiber(self):
        report = ct.score(self.fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        self.assertLess(report["centerline_error_voxels"], 0.5)
        self.assertLess(abs(report["diameter_bias_m"]), 0.05 * DIAMETER)
        self.assertGreater(report["voxel_label_accuracy"], 0.95)

    def test_outputs_round_trip_into_tangle(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fit.write(tmp, volume=self.scan.volume)
            self.assertTrue(Path(paths["config"]).is_file())
            self.assertTrue(Path(paths["labels"]).is_file())
            self.assertTrue(Path(paths["overlay_stack"]).is_file())
            reloaded = ct.load_fit(paths["config"])
        self.assertEqual(reloaded.fiber_count, self.fit.fiber_count)
        assembly = reloaded.to_assembly()
        self.assertEqual(assembly.fiber_count, self.fit.fiber_count)
        population = self.fit.suggested_population(count=5)
        self.assertEqual(population.count, 5)

    def test_overlay_colors_each_fiber(self):
        labels = self.fit.label_volume()
        z = labels.shape[0] // 2
        rgb = ct.overlay_slice(self.scan.volume[z], labels[z])
        self.assertEqual(rgb.shape, labels[z].shape + (3,))
        stack = ct.overlay_volume(self.scan.volume, labels)
        self.assertEqual(stack.dtype, np.uint8)
        self.assertEqual(stack.shape, labels.shape + (3,))

    def test_thin_fibers_are_rejected(self):
        with self.assertRaises(ValueError):
            ct.fit_fibers(self.scan.volume, VOXEL, ct.FiberSpec(diameter=1 * um))


if __name__ == "__main__":
    unittest.main()
