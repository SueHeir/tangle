import os
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
SOLVER = hasattr(tangle, "ImageRelaxer")


def fast_settings(**changes):
    """Short solver runs on the CPU backend (CI has no GPU adapter)."""
    return ct.FitSettings(
        backend=os.environ.get("TANGLE_BACKEND", "cpu"), rounds=2, solver_batches=1, solver_iterations=150,
        solver_settle_iterations=50, **changes,
    )


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


def two_type_scan():
    """Two 10 µm fibers and one 20 µm fiber, apart from each other."""
    small = tangle.Material("small", diameter=DIAMETER)
    large = tangle.Material("large", diameter=2 * DIAMETER)
    fibers = tangle.FiberCollection("two types")
    fibers.add_fiber([[8 * um, 25 * um, 30 * um], [82 * um, 28 * um, 30 * um]], small)
    fibers.add_fiber([[8 * um, 65 * um, 60 * um], [82 * um, 62 * um, 60 * um]], small)
    fibers.add_fiber([[45 * um, 8 * um, 62 * um], [47 * um, 82 * um, 20 * um]], large)
    assembly = tangle.Assembly(tangle.Cell([90 * um] * 3))
    assembly.insert(fibers)
    profiles = [(DIAMETER, ct.CrossSection()), (2 * DIAMETER, ct.CrossSection())]
    return ct.synthetic_ct(assembly, VOXEL, seed=4, profiles=profiles)


@unittest.skipIf(ct is None, "tangle.ct needs NumPy and SciPy")
@unittest.skipUnless(SOLVER, "tangle.ct needs a Tangle build with ImageRelaxer")
class CtFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = crossing_scan()
        cls.fit = ct.fit_fibers(cls.scan.volume, VOXEL, ct.FiberSpec(diameter=DIAMETER), fast_settings())

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

    def test_length_prior_keeps_every_fiber(self):
        spec = ct.FiberSpec(diameter=DIAMETER, length=200 * um)
        fit = ct.fit_fibers(self.scan.volume, VOXEL, spec, fast_settings())
        report = ct.score(fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        summary = fit.population_summary()
        self.assertIn("interior_ends", summary)
        self.assertGreater(summary["expected_interior_ends"], 0.0)

    def test_length_prior_joins_across_a_long_gap(self):
        from tangle.ct import _ends, _moves
        from tangle.ct._geometry import resample
        from tangle.ct._image import normalize

        image, _ = normalize(self.scan.volume, denoise_sigma=0.7)
        line = resample(self.scan.centerlines[0], 4.0)
        radius = 0.5 * DIAMETER / VOXEL
        gap = int(np.ceil(6 * radius / 4.0))  # a 6-radius break, beyond the fixed 4-radius limit
        middle = len(line) // 2
        pieces = [line[: middle - gap // 2], line[middle + gap - gap // 2 :]]
        radii = np.full(2, radius)
        _, _, without = _moves.merge_fragments(image, pieces, radii, max_gap=4 * radius)
        self.assertEqual(without, 0)
        cost = _ends.end_cost(200 * um, DIAMETER)
        scale = _ends.evidence_scale(image, pieces, radii, radius)
        joined, _, merges = _moves.merge_fragments(
            image, pieces, radii, max_gap=4 * radius, end_cost=cost, scale=scale, max_prior_gap=16 * radius
        )
        self.assertEqual(merges, 1)
        self.assertEqual(len(joined), 1)

    def test_end_statistics_ignore_boundary_ends(self):
        from tangle.ct._ends import end_statistics

        shape = (40, 40, 40)
        through = np.array([[0.0, 20.0, 20.0], [40.0, 20.0, 20.0]])
        inside = np.array([[10.0, 10.0, 10.0], [30.0, 10.0, 10.0]])
        stats = end_statistics([through, inside], np.array([2.0, 2.0]), shape, length=60.0)
        self.assertEqual(stats["interior_ends"], 2)
        self.assertAlmostEqual(stats["implied_length"], 60.0)
        self.assertAlmostEqual(stats["expected_interior_ends"], 2.0)

    def test_crop_clips_truth_to_the_window(self):
        crop = self.scan.crop((0, 0, 0), (36, 72, 72))
        self.assertEqual(crop.volume.shape, (72, 72, 36))
        for line in crop.centerlines:
            self.assertTrue(np.all(line[:, 0] < 36))
        self.assertEqual(int(crop.labels.max()), len(crop.centerlines))

    def test_cross_section_area_round_trip(self):
        rimmed = ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3)
        radius = 7.0
        area = rimmed.area(radius, VOXEL)
        self.assertLess(area, 0.75 * np.pi * radius * radius)
        recovered = rimmed.radius_from_area(np.array([area]), VOXEL, 2.0, 14.0)[0]
        self.assertAlmostEqual(recovered, radius, places=2)
        solid = ct.CrossSection(brightness=0.5)
        self.assertAlmostEqual(solid.radius_from_area(np.array([solid.area(3.0, VOXEL)]), VOXEL, 1.0, 6.0)[0], 3.0)
        self.assertAlmostEqual(solid.center_response(3.0, VOXEL, 0.0), 0.5)
        self.assertLess(rimmed.center_response(radius, VOXEL, 0.5 * radius), 0.75)

    def test_fit_json_keeps_fiber_types(self):
        small = ct.FiberSpec(diameter=DIAMETER, name="small")
        large = ct.FiberSpec(diameter=2 * DIAMETER, min_bend_radius=30 * DIAMETER, name="large")
        typed = ct.FitResult(
            shape=self.fit.shape, voxel_size=VOXEL, spec=large, centerlines=self.fit.centerlines,
            radii=self.fit.radii, support=self.fit.support, levels=self.fit.levels,
            specs=[small, large], types=np.arange(self.fit.fiber_count) % 2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            reloaded = ct.load_fit(typed.write(tmp)["config"])
        self.assertEqual(reloaded.specs[1].min_bend_radius, 30 * DIAMETER)
        self.assertEqual(list(reloaded.types), list(typed.types))
        self.assertEqual(len(typed.suggested_population()), 2)

    def test_geometry_report_finds_overlaps_and_kinks(self):
        straight = np.stack([np.linspace(0, 40, 9), np.zeros(9), np.zeros(9)], axis=1)
        beside = straight + np.array([0.0, 3.0, 0.0])  # radii 2: 1 voxel deep, half a radius
        report = ct.geometry_report([straight, beside], np.array([2.0, 2.0]), min_bend_radius=20.0)
        self.assertAlmostEqual(report["max_penetration_radii"], 0.5, places=6)
        self.assertEqual(report["overlapping_pairs"], 1)
        self.assertEqual(report["fibers_over_bend_limit"], 0)
        bent = straight.copy()
        bent[4, 1] = 4.0  # a sharp kink in the middle
        report = ct.geometry_report([bent, straight + np.array([0.0, 20.0, 0.0])], np.array([2.0, 2.0]), 20.0)
        self.assertEqual(report["fibers_over_bend_limit"], 1)
        self.assertEqual(report["overlapping_pairs"], 0)
        self.assertAlmostEqual(report["min_segment_diameters"], 1.25, places=6)

    def test_fit_is_valid_tangle_geometry(self):
        geometry = ct.geometry_report(
            self.fit.centerlines, self.fit.radii, 5 * DIAMETER / VOXEL, spacing=1.25 * DIAMETER / VOXEL
        )
        self.assertEqual(geometry["overlapping_pairs"], 0, geometry)
        self.assertEqual(geometry["fibers_over_bend_limit"], 0, geometry)
        self.assertTrue(any(entry["stage"] == "solver" for entry in self.fit.history))

    def test_fit_from_a_generous_mask(self):
        mask = self.scan.fiber_mask(level=0.35)  # over-reaches, like a generous threshold
        fit = ct.fit_fibers(mask, VOXEL, ct.FiberSpec(diameter=DIAMETER), fast_settings())
        report = ct.score(fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        self.assertTrue(fit.history[0]["mask"])
        # Fibers look thicker in the mask; the estimated margin takes that off.
        self.assertLess(abs(report["diameter_bias_m"]), 0.1 * DIAMETER)

    def test_exclude_mask_hides_a_fiber(self):
        exclude = np.zeros(self.scan.volume.shape, dtype=bool)
        # A slab across the middle, wider than the 4-radius join gap.
        exclude[:, :, 26:46] = True
        fit = ct.fit_fibers(self.scan.fiber_mask(), VOXEL, ct.FiberSpec(diameter=DIAMETER), fast_settings(), exclude=exclude)
        for line in fit.centerlines:
            inside = (line[:, 0] > 27) & (line[:, 0] < 45)
            self.assertFalse(inside.any())

    def test_types_are_chosen_by_size(self):
        scan = two_type_scan()
        specs = [ct.FiberSpec(diameter=DIAMETER, name="small"), ct.FiberSpec(diameter=2 * DIAMETER, name="large")]
        fit = ct.fit_fibers(scan.fiber_mask(level=0.35), VOXEL, specs, fast_settings())
        report = ct.score(fit, scan)
        self.assertEqual(report["recovered"], 3, report)
        for kind in (0, 1):
            self.assertEqual(report["per_type"][kind]["fitted_as_this_type"], report["per_type"][kind]["fitted"], report["per_type"])
        geometry = ct.geometry_report(fit.centerlines, fit.radii, 5 * DIAMETER / VOXEL, spacing=1.25 * DIAMETER / VOXEL)
        self.assertEqual(geometry["overlapping_pairs"], 0, geometry)

    def test_thin_fibers_are_rejected(self):
        with self.assertRaises(ValueError):
            ct.fit_fibers(self.scan.volume, VOXEL, ct.FiberSpec(diameter=1 * um))


if __name__ == "__main__":
    unittest.main()
