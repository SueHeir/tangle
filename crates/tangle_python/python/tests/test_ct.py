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
# Fits run on the GPU. CI runners have none (and the CPU runtime is far too
# slow), so fits are skipped on CI unless TANGLE_BACKEND names a backend.
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")
GPU_FITS = hasattr(tangle, "ImageRelaxer") and not (os.environ.get("CI") and "TANGLE_BACKEND" not in os.environ)


def fit_settings(**changes):
    """Default settings with two rounds of two solver batches."""
    return ct.FitSettings(backend=BACKEND, rounds=2, solver_batches=2, **changes)


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
class CtToolTests(unittest.TestCase):
    """Everything but the fit itself; runs anywhere."""

    @classmethod
    def setUpClass(cls):
        cls.scan = crossing_scan()

    def test_synthetic_scan_has_ground_truth(self):
        self.assertEqual(self.scan.volume.shape, (72, 72, 72))
        self.assertEqual(int(self.scan.labels.max()), 3)
        self.assertEqual(len(self.scan.centerlines), 3)

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

    def test_centerline_agreement_ignores_capsule_edges(self):
        from tangle.ct._evaluate import centerline_agreement

        shape = (40, 40, 60)  # (z, y, x)
        truth = [np.array([[0.0, 10.0, 20.0], [60.0, 10.0, 20.0]]), np.array([[0.0, 30.0, 20.0], [60.0, 30.0, 20.0]])]
        radii = np.array([4.0, 4.0])
        # Fiber 0 traced 1.5 voxels off its axis (inside half a radius), fiber 1
        # only over its first half; a false fit sits in the void between them.
        fits = [
            np.array([[0.0, 11.5, 20.0], [60.0, 11.5, 20.0]]),
            np.array([[0.0, 30.0, 20.0], [30.0, 30.0, 20.0]]),
            np.array([[10.0, 20.0, 20.0], [30.0, 20.0, 20.0]]),
        ]
        result = centerline_agreement(fits, truth, radii, shape)
        self.assertAlmostEqual(result["recall"], (60.0 + 32.0) / 120.0, delta=0.02)  # 2 past its end
        self.assertAlmostEqual(result["precision"], 90.0 / 110.0, delta=0.02)
        # A fit that follows fiber 0 and then jumps to fiber 1 keeps only the
        # length along fiber 0.
        hop = [np.array([[0.0, 10.0, 20.0], [40.0, 10.0, 20.0], [42.0, 30.0, 20.0], [60.0, 30.0, 20.0]])]
        result = centerline_agreement(hop, truth, radii, shape)
        self.assertAlmostEqual(result["recall"], 42.0 / 120.0, delta=0.03)

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

    def test_scan_noise_can_be_correlated(self):
        assembly = tangle.Assembly(tangle.Cell([90 * um] * 3))
        white = ct.synthetic_ct(assembly, VOXEL, seed=3, drift=0.0).volume.astype(np.float64)
        blurred = ct.synthetic_ct(assembly, VOXEL, seed=3, drift=0.0, noise_correlation=0.9).volume.astype(np.float64)

        def lag_one(volume):
            v = volume - volume.mean()
            return float((v[:, :, 1:] * v[:, :, :-1]).mean() / (v * v).mean())

        self.assertLess(lag_one(white), 0.1)
        self.assertGreater(lag_one(blurred), 0.6)

    def test_cross_section_width_survives_noise_that_breaks_the_depth(self):
        from scipy.ndimage import gaussian_filter

        from tangle.ct._geometry import paint, sample_image
        from tangle.ct._image import half_radius, half_widths
        from tangle.ct._trace import foreground_depth

        occupied = np.zeros((40, 40, 64), dtype=np.int32)
        axis = np.array([[4.0, 20.0, 20.0], [60.0, 20.0, 20.0]])
        paint(occupied, axis, 7.0, 1)  # a dim fiber of radius 7 along x
        rng = np.random.default_rng(1)
        noise = gaussian_filter(rng.normal(size=occupied.shape), 0.9)
        image = (0.6 * gaussian_filter((occupied > 0).astype(np.float64), 0.9) + 0.2 * noise / noise.std()).astype(np.float32)
        line = np.stack([np.arange(8.0, 57.0, 2.0), np.full(25, 20.0), np.full(25, 20.0)], axis=1)
        _, depth = foreground_depth(image > 0.3)  # a threshold half way to the fiber's grey
        self.assertLess(float(np.median(sample_image(depth, line))), 0.6 * 7.0)
        self.assertAlmostEqual(float(half_widths(image, [line], reach=12.0)[0]), 7.0, delta=1.0)
        self.assertTrue(np.isnan(half_widths(image, [line[:2]], reach=12.0)[0]))  # too short for a cross-section
        # A bright rim around a dim core reads its outer edge.
        distances = np.arange(0.0, 10.0, 0.5)
        rim = np.where(distances < 4.0, 0.3, np.where(distances < 6.0, 1.0, 0.0))
        self.assertAlmostEqual(float(half_radius(rim[None], distances, 6.0)[0]), 5.75, delta=0.3)

    def test_mask_input_fills_cores_and_applies_exclude(self):
        from tangle.ct._fit import _is_mask, _mask_image

        z, y, _ = np.indices((30, 30, 30), dtype=np.float64) + 0.5
        rho = np.hypot(y - 15.0, z - 15.0)
        hollow = (rho <= 6.0) & (rho >= 3.0)  # a tube along x, as a threshold misses a dim core
        self.assertTrue(_is_mask(hollow.astype(np.uint8) * 255))
        self.assertFalse(_is_mask(rho))
        settings = ct.FitSettings(denoise_sigma_voxels=0.0)
        image, levels = _mask_image(hollow, None, settings, largest_radius=6.0)
        self.assertEqual(float(image[15, 15, 15]), 1.0)  # core filled
        self.assertEqual((levels.void, levels.fiber), (0.0, 1.0))
        exclude = np.zeros(hollow.shape, dtype=bool)
        exclude[:, :, :10] = True
        image, _ = _mask_image(hollow, exclude, settings, largest_radius=6.0)
        self.assertEqual(float(image[:, :, :10].max()), 0.0)
        # A void region bigger than a fiber core stays void, even when enclosed in a slice.
        ring = (rho <= 14.0) & (rho >= 12.0)
        image, _ = _mask_image(ring, None, settings, largest_radius=3.0)
        self.assertEqual(float(image[15, 15, 15]), 0.0)

    def test_side_by_side_merges_two_fits_on_one_fiber_only(self):
        from tangle.ct._moves import render_occupancy, resolve_side_by_side

        low, high = np.zeros(3, dtype=int), np.array([48, 40, 40])
        x = np.linspace(4.0, 44.0, 11)

        def along(y):
            return np.stack([x, np.full_like(x, y), np.full_like(x, 20.0)], axis=1)

        radii = np.array([4.0, 4.0])
        one_fiber = render_occupancy(low, high, [along(20.0)], radii[:1])
        # Two fits pushed a radius off either side of one fiber's axis: one fiber.
        kept, _, changed = resolve_side_by_side(one_fiber, [along(16.0), along(24.0)], radii, min_length=12.0)
        self.assertEqual((changed, len(kept)), (1, 1))
        # Two real touching fibers: kept.
        two_fibers = render_occupancy(low, high, [along(16.0), along(24.0)], radii)
        kept, _, changed = resolve_side_by_side(two_fibers, [along(16.0), along(24.0)], radii, min_length=12.0)
        self.assertEqual((changed, len(kept)), (0, 2))
        # Fits that only cross are not tested.
        across = np.stack([np.full_like(x, 24.0), x - 4.0, np.full_like(x, 20.0)], axis=1)
        crossing = render_occupancy(low, high, [along(20.0), across], radii)
        kept, _, changed = resolve_side_by_side(crossing, [along(20.0), across], radii, min_length=12.0)
        self.assertEqual((changed, len(kept)), (0, 2))

    def test_confidence_flags_a_fit_between_two_fibers(self):
        from scipy.ndimage import distance_transform_edt, maximum_filter

        from tangle.ct._confidence import node_confidence
        from tangle.ct._geometry import paint

        x = np.linspace(8.0, 48.0, 11)

        def along(y):
            return np.stack([x, np.full_like(x, y), np.full_like(x, 20.0)], axis=1)

        occupied = np.zeros((40, 40, 56), dtype=np.int32)
        for label, y in ((1, 16.0), (2, 24.0)):  # two touching fibers of radius 4
            paint(occupied, along(y), 4.0, label)
        image = (occupied > 0).astype(np.float32)
        depth = maximum_filter(distance_transform_edt(image > 0.5), size=3).astype(np.float32)

        def mean(lines, **options):
            nodes, summary = node_confidence(image, depth, lines, np.full(len(lines), 4.0), spacing=4.0, **options)
            self.assertEqual([len(c) for c in nodes], [len(line) for line in lines])
            self.assertTrue(all(((c >= 0) & (c <= 1)).all() for c in nodes))
            return summary["mean"]

        right = mean([along(16.0), along(24.0)])
        self.assertGreater(right, 0.7)
        # One fit along the contact line of both: the wrong cross-section, and
        # fiber on both sides that no fit explains.
        self.assertLess(mean([along(20.0)]), 0.3)
        # Fits that just moved three voxels are less sure than settled ones.
        moved = mean([along(16.0), along(24.0)], previous=[along(13.0), along(27.0)])
        self.assertLess(moved, 0.5 * right)

    def test_redraw_cuts_unsure_stretches_and_grows_the_sure_ends_back(self):
        from tangle.ct._geometry import paint
        from tangle.ct._image import HessianField
        from tangle.ct._regrow import cut_unsure, grow_cut_ends, pinned_flags
        from tangle.ct._trace import Tracer

        occupied = np.zeros((40, 40, 64), dtype=np.int32)
        axis = np.array([[4.0, 20.0, 20.0], [60.0, 20.0, 20.0]])
        paint(occupied, axis, 4.0, 1)  # one straight fiber, radius 4, along x
        image = occupied.astype(np.float32)
        line = np.stack([np.arange(8.0, 57.0, 4.0), np.full(13, 20.0), np.full(13, 20.0)], axis=1)
        confidence = np.where((line[:, 0] > 22) & (line[:, 0] < 42), 0.2, 0.9)

        cut = cut_unsure([line], [confidence], np.array([4.0]), threshold=0.5, spacing=4.0)
        self.assertEqual(len(cut.pieces), 2)
        self.assertEqual(sorted(cut.cut_ends), [(0, -1), (1, 0)])
        self.assertLessEqual(cut.pieces[0][:, 0].max(), 22.0)
        self.assertGreaterEqual(cut.pieces[1][:, 0].min(), 42.0)
        # Anchors leave two radii free at each cut end.
        self.assertLessEqual(max(a[:, 0].max() for a in cut.anchors if a[0, 0] < 30), 22.0 - 8.0 + 1e-6)
        self.assertIsNone(cut_unsure([line], [np.full(13, 0.9)], np.array([4.0]), threshold=0.5, spacing=4.0))

        hessian = HessianField(image, sigma=2.4)

        def tracer_for(index, claimed):
            return Tracer(image, hessian, radius=4.0, min_bend_radius=40.0, step=2.0, claimed=claimed)

        pieces, grown = grow_cut_ends(
            cut.pieces, cut.cut_ends, np.array([4.0, 4.0]), tracer_for=tracer_for, shape=image.shape,
            spacing=4.0, max_length=80.0,
        )
        self.assertGreater(grown, 4.0)
        first, second = sorted(pieces, key=lambda p: p[:, 0].min())
        # The ends grew toward each other and stopped short of overlapping.
        self.assertGreater(first[:, 0].max(), 26.0)
        self.assertLess(first[:, 0].max(), second[:, 0].min())
        self.assertLess(second[:, 0].min() - first[:, 0].max(), 16.0)  # within the join gap
        self.assertLess(np.abs(first[:, 1:] - 20.0).max(), 1.0)  # stayed on the fiber axis
        # The second try at a region re-traces it from fresh seeds instead of growing.
        unchanged, none = grow_cut_ends(
            cut.pieces, cut.cut_ends, np.array([4.0, 4.0]), tracer_for=tracer_for, shape=image.shape,
            spacing=4.0, max_length=80.0, attempt=lambda index, end: 1,
        )
        self.assertEqual(none, 0.0)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(unchanged, cut.pieces)))
        flags = pinned_flags([first], cut.anchors, 0.5)[0]
        self.assertTrue(flags[0])
        self.assertFalse(flags[-1])

    def test_redraw_regions_are_kept_or_reverted_whole(self):
        from tangle.ct._regrow import changed_regions, choose, cut_unsure, region_components

        x = np.arange(8.0, 57.0, 4.0)

        def along(y):
            return np.stack([x, np.full_like(x, y), np.full_like(x, 20.0)], axis=1)

        bridge = np.stack([np.full(9, 30.0), np.linspace(8.0, 42.0, 9), np.full(9, 21.0)], axis=1)
        # Two fibers 30 voxels apart, unsure in the middle, and a sure fiber across both.
        lines = [along(10.0), along(40.0), bridge]
        confidence = [np.where((x > 22) & (x < 42), 0.2, 0.9)] * 2 + [np.full(9, 0.9)]
        radii = np.array([4.0, 4.0, 4.0])
        cut = cut_unsure(lines, confidence, radii, threshold=0.5, spacing=4.0)
        self.assertEqual(cut.removed_nodes, 10)
        self.assertEqual(len(cut.regions), 2)  # one per fiber's unsure stretch
        self.assertEqual(cut.fiber_regions, [{0}, {1}, set()])
        # A failed region is cut wider next time; one given up on is not cut.
        first = (np.zeros(3), np.array([64.0, 20.0, 40.0]))
        wider = cut_unsure(lines, confidence, radii, threshold=0.5, spacing=4.0, widen=[(*first, 8.0)])
        self.assertGreater(wider.removed_nodes, cut.removed_nodes)
        skipped = cut_unsure(lines, confidence, radii, threshold=0.5, spacing=4.0, skip=[first])
        self.assertEqual((skipped.removed_nodes, len(skipped.regions)), (5, 1))

        # The redraw moved the two fibers and left the sure one as it was: the
        # sure fiber passing through both regions does not tie them together.
        redrawn = [along(10.5), along(40.5), bridge]
        touch, boxes = changed_regions(redrawn, cut.anchors, 1.0, cut.regions, 4.0)
        self.assertEqual(touch, [{0}, {1}, set()])
        self.assertEqual(len(boxes), 2)
        component = region_components(2, cut.fiber_regions, touch)
        self.assertNotEqual(component[0], component[1])
        # Keep the first region's redraw, revert the second.
        accepted = np.zeros(2, dtype=bool)
        accepted[component[0]] = True
        keep_old, keep_new = choose(cut.fiber_regions, touch, component, accepted)
        self.assertEqual((keep_old, keep_new), ([1], [0, 2]))
        # A redrawn stretch reaching both regions ties them together.
        moved = bridge + np.array([2.0, 0.0, 0.0])
        touch, _ = changed_regions([*redrawn[:2], moved], cut.anchors, 1.0, cut.regions, 4.0)
        self.assertEqual(touch[2], {0, 1})
        component = region_components(2, cut.fiber_regions, touch)
        self.assertEqual(component[0], component[1])

    def test_redraw_failures_are_counted_per_region(self):
        from tangle.ct._fit import _Fitter

        failures = []
        box = (np.zeros(3), np.full(3, 10.0))
        far = (np.full(3, 50.0), np.full(3, 60.0))
        _Fitter._record(failures, *far, False)
        _Fitter._record(failures, *box, False)
        _Fitter._record(failures, *box, False)  # the second record overlaps: counted twice
        self.assertEqual(sorted(f[2] for f in failures), [1, 2])
        _Fitter._record(failures, *box, True)  # kept: forgotten
        self.assertEqual(len(failures), 1)
        self.assertTrue(np.array_equal(failures[0][0], far[0]))

    def test_fiber_ends_are_priced_by_length(self):
        from tangle.ct._ends import end_cost, length_end_cost, length_join_cost

        mean, diameter = 400.0, 10.0
        # Exponential lengths: every end costs ln(L / D), and joins are free.
        for so_far in (20.0, 400.0, 1200.0):
            self.assertAlmostEqual(length_end_cost(so_far, mean, diameter, 1.0), end_cost(mean, diameter), places=6)
        self.assertAlmostEqual(length_join_cost(300.0, 300.0, 0.0, mean, 1.0), 0.0, places=6)
        # Peaked lengths: ending a short fiber is dear, a long one cheap ...
        short, typical, long = (length_end_cost(x, mean, diameter, 3.0) for x in (40.0, 400.0, 1200.0))
        self.assertGreater(short, typical)
        self.assertGreater(typical, long)
        self.assertGreaterEqual(long, 1.0)
        # ... and a join that makes a fiber far longer than L costs more.
        into_typical = length_join_cost(150.0, 150.0, 10.0, mean, 3.0)
        into_long = length_join_cost(600.0, 600.0, 10.0, mean, 3.0)
        self.assertGreaterEqual(into_typical, 0.0)
        self.assertGreater(into_long, into_typical + 1.0)
        self.assertEqual(length_end_cost(40.0, None, diameter, 3.0), 1.0)

    def test_crossing_ends_are_joined_straight_through(self):
        from tangle.ct._geometry import paint
        from tangle.ct._junctions import allowed_pairs, assemble, rank_plans, region_ports

        occupied = np.zeros((48, 48, 48), dtype=np.int32)
        paint(occupied, np.array([[2.0, 24.0, 24.0], [46.0, 24.0, 24.0]]), 3.0, 1)  # along x
        paint(occupied, np.array([[24.0, 2.0, 24.0], [24.0, 46.0, 24.0]]), 3.0, 2)  # along y
        image = (occupied > 0).astype(np.float32)

        def piece(start, stop, along_x):
            t = np.linspace(start, stop, 6)
            other = np.full(6, 24.0)
            return np.stack([t, other, other] if along_x else [other, t, other], axis=1)

        # The crossing was cut out of both fibers: four loose ends around it.
        pieces = [piece(4, 14, True), piece(34, 44, True), piece(4, 14, False), piece(34, 44, False)]
        cut_ends = [(0, -1), (1, 0), (2, -1), (3, 0)]
        radii = np.full(4, 3.0)
        box = (np.array([10.0, 10.0, 18.0]), np.array([38.0, 38.0, 30.0]))
        ports = region_ports(pieces, cut_ends, np.zeros(4, dtype=int), radii, [box])[0]
        self.assertEqual(len(ports), 4)
        pairs = allowed_pairs(ports, np.array([30.0]), 2.0, 60.0)
        # Only straight through: a 90 degree turn breaks the bend limit.
        self.assertEqual(sorted(pairs), [(0, 1), (2, 3)])
        plans = rank_plans(
            image, box, ports, pairs, [np.zeros((0, 3))] * 4, pieces, radii,
            margin=0.0, scale=3.0, end_costs=np.ones(4), interior=[True] * 4,
        )
        self.assertEqual(len(plans), 4)
        self.assertEqual(sorted(plans[0].pairs), [(0, 1), (2, 3)])
        self.assertEqual(plans[0].ends, 0)
        connections = [
            (ports[i].piece, ports[i].end, ports[j].piece, ports[j].end, pairs[(i, j)]) for i, j in plans[0].pairs
        ]
        fibers, first = assemble(pieces, connections, {}, 2.0)
        self.assertEqual(len(fibers), 2)
        for fiber in fibers:
            span = fiber.max(axis=0) - fiber.min(axis=0)
            self.assertGreater(span.max(), 39.0)
            self.assertLess(np.sort(span)[1], 0.5)  # straight

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

    def test_fits_are_cut_where_they_sit_in_void(self):
        from tangle.ct import _refine

        image = np.zeros((20, 20, 60), dtype=np.float32)
        image[8:12, 8:12, 5:40] = 1.0  # a fiber along x over voxels 5-39
        image[8:12, 8:12, 45:] = 1.0  # and another from 45 to the scan's edge
        # Follows the fiber to x = 37, then drifts off sideways into void.
        drifted = np.array(
            [[x, 10.0, 10.0] for x in np.arange(6.0, 38.0)] + [[38.0 + k, 11.5 + 1.5 * k, 10.0] for k in range(1, 5)]
        )
        across = np.array([[x, 10.0, 10.0] for x in np.arange(6.0, 57.0)])  # runs over the gap (x 41-44 in void)
        dip = np.array([[x, 10.0, 10.0] for x in np.arange(30.0, 50.0)])  # the same gap, radius 4: too short to split
        outside = np.array([[x, 10.0, 10.0] for x in np.arange(50.0, 67.0)])  # leaves the scan: not trimmed there
        # Bows off the fiber into void over x 31-34 and comes back: the straight
        # line under the bow is fiber, so it is bridged, not split.
        bowed = np.array(
            [[x, 10.0, 10.0] for x in np.arange(6.0, 31.0)]
            + [[31.0, 14.0, 10.0], [32.0, 15.0, 10.0], [33.0, 15.0, 10.0], [34.0, 14.0, 10.0]]
            + [[x, 10.0, 10.0] for x in np.arange(35.0, 40.0)]
        )
        # The cases lie on the same fiber, so each is cut on its own (another
        # fit's core under a bow's bridge blocks it).
        cut = [_refine.cut_void(image, [line], np.array([r])) for line, r in (
            (drifted, 2.0), (across, 2.0), (dip, 4.0), (outside, 2.0), (bowed, 1.0)
        )]
        pieces = [piece for result in cut for piece in result[0]]
        self.assertEqual([len(result[0]) for result in cut], [1, 2, 1, 1, 1])
        self.assertEqual(len(pieces[0]), 32)  # the drifted tail is gone
        self.assertEqual(pieces[1][-1, 0], 40.0)
        self.assertEqual(pieces[2][0, 0], 45.0)
        self.assertEqual(len(pieces[3]), len(dip))
        self.assertEqual(len(pieces[4]), len(outside))
        np.testing.assert_allclose(pieces[5][:, 1], 10.0)  # bridged along the fiber
        self.assertEqual(pieces[5][0, 0], 6.0)
        self.assertEqual(pieces[5][-1, 0], 39.0)
        totals = {key: sum(result[2][key] for result in cut) for key in cut[0][2]}
        self.assertEqual(totals, {"trimmed": 4 + 4, "splits": 1, "bridged": 1})
        # A dim node that barely leaves a straight fit (where a fit hops to
        # another fiber at a crossing) splits it, though the line under it is
        # fiber: only a real bow is bridged.
        hop = np.array([[10.0, 10.0, 10.0], [15.0, 10.0, 10.0], [20.0, 13.5, 10.0], [25.0, 10.0, 10.0], [30.0, 10.0, 10.0]])
        pieces, source, counts = _refine.cut_void(image, [hop], np.array([4.0]))
        np.testing.assert_array_equal(source, [0, 0])
        self.assertEqual((counts["splits"], counts["bridged"]), (1, 0))
        # Unless the scan's fiber axis runs along the line at both ends.
        along_x = lambda index, points: np.tile([1.0, 0.0, 0.0], (len(points), 1))  # noqa: E731
        across_y = lambda index, points: np.tile([0.0, 1.0, 0.0], (len(points), 1))  # noqa: E731
        _, source, counts = _refine.cut_void(image, [hop], np.array([4.0]), directions=along_x)
        self.assertEqual((len(source), counts["bridged"]), (1, 1))
        _, source, counts = _refine.cut_void(image, [hop], np.array([4.0]), directions=across_y)
        self.assertEqual((len(source), counts["bridged"]), (2, 0))

    def test_batched_capsule_drawing_matches_segment_by_segment(self):
        from tangle.ct import _geometry, _grey, _moves

        rng = np.random.default_rng(3)
        shape = (18, 22, 26)
        lines = [np.cumsum(rng.normal(0.0, 1.6, (6, 3)), axis=0) + [13.0, 11.0, 9.0] for _ in range(5)]
        lines.append(np.array([[4.0, 4.0, 4.0]]))  # one node: no segments
        lines.append(np.array([[20.0, 5.0, 5.0], [20.0, 5.0, 5.0], [22.0, 6.0, 5.0]]))  # a zero-length segment
        radii = np.array([1.5, 2.0, 2.5, 1.0, 3.0, 2.0, 1.2])
        centers = np.stack(np.meshgrid(*[np.arange(n) + 0.5 for n in shape[::-1]], indexing="ij"), axis=-1)
        centers = centers.transpose(2, 1, 0, 3).reshape(-1, 3)  # (z, y, x) order, points (x, y, z)
        distances = []  # per segment, every voxel's distance, drawn one segment at a time
        for line in lines:
            distances.append([_geometry._segment_distances(centers, a, b) for a, b in zip(line[:-1], line[1:])])

        def nearest(reach, signed):
            best = np.full(len(centers), np.inf)
            labels = np.zeros(len(centers), dtype=int)
            segments = np.full(len(centers), -1)
            index = 0
            for f, per_segment in enumerate(distances):
                for d in per_segment:
                    value = d - radii[f] if signed else d
                    better = (d <= reach[f]) & (value < best)
                    best[better], labels[better], segments[better] = value[better], f + 1, index
                    index += 1
            return best, labels, segments

        for signed in (False, True):
            reach = radii + 1.0
            labels, best, segments = _geometry.rasterize(shape, lines, radii, reach=reach, signed=signed)
            want_best, want_labels, want_segments = nearest(reach, signed)
            np.testing.assert_array_equal(labels.ravel(), want_labels)
            np.testing.assert_array_equal(segments.ravel() >= 0, want_segments >= 0)
            np.testing.assert_allclose(best.ravel(), want_best.astype(np.float32), rtol=1e-6)
            # Where two segments of a line meet, both are equally near up to
            # rounding, so either may own a voxel; the owner must be nearest.
            flat = [d for per_segment in distances for d in per_segment]
            owned = np.nonzero(segments.ravel() >= 0)[0]
            chosen = np.array([flat[k][v] for v, k in zip(owned, segments.ravel()[owned])])
            shift = radii[labels.ravel()[owned] - 1] if signed else 0.0
            np.testing.assert_allclose(chosen - shift, want_best[owned], atol=1e-9)

        low, high = np.array([3, 2, 1]), np.array([24, 20, 15])  # a sub-box, (x, y, z)
        inside = np.all((centers >= low) & (centers < high), axis=1)
        occupancy = np.zeros(len(centers))
        for f, per_segment in enumerate(distances):
            for d in per_segment:
                occupancy = np.maximum(occupancy, np.clip(0.5 - (d - radii[f]) / 2.4, 0.0, 1.0))
        got = _moves.render_occupancy(low, high, lines, radii)
        np.testing.assert_allclose(got.ravel(), occupancy[inside].reshape(got.shape).ravel(), atol=1e-12)

        profiles = [np.array([1.0, 0.9, 0.6]), np.array([0.4, 0.8])] * 3 + [np.array([0.7, 0.7, 0.7])]
        void = 0.1
        surface, _, owner = nearest(radii + 2.4, True)
        want = np.full(len(centers), void)
        segment_line = np.repeat(np.arange(len(lines)), [max(len(line) - 1, 0) for line in lines])
        for v in np.nonzero((owner >= 0) & (surface < 2.4))[0]:
            f = segment_line[owner[v]]
            profile = profiles[f]
            fraction = min((surface[v] + radii[f]) / radii[f], 1.0)
            value = np.interp(fraction, np.linspace(0.0, 1.0, len(profile)), profile)
            want[v] = void + np.clip(1.0 - surface[v] / 2.4, 0.0, 1.0) * (value - void)
        got = _grey.render_grey(low, high, lines, radii, profiles, void)
        np.testing.assert_allclose(got.ravel(), want[inside], atol=1e-9)

    def test_grey_ranges_decide_what_is_fiber(self):
        from tangle.ct._ranges import range_image, type_fractions

        volume = np.zeros((10, 10, 10), dtype=np.float32)
        volume[:, :, 1:4] = 1.0  # type 0: grey 0.9-1.1
        volume[:, :, 6:9] = 0.5  # type 1: grey 0.45-0.55
        volume[5, 5, 5] = 0.25  # an edge voxel, half of type 1's grey
        volume[5, 5, 2] = 3.0  # a bright inclusion inside type 0
        volume[5, 2, 2] = 1.15  # noise just above type 0's range
        ranges = [(0.9, 1.1), (0.45, 0.55)]
        image, types, void = range_image(volume, ranges, denoise_sigma=0.0)
        self.assertEqual(void, 0.0)
        self.assertEqual(float(image[0, 0, 2]), 1.0)
        self.assertEqual(float(image[0, 0, 7]), 1.0)
        self.assertEqual(float(image[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(image[5, 5, 5]), 0.25 / 0.45, places=5)
        self.assertEqual(float(image[5, 5, 2]), 0.0)
        self.assertAlmostEqual(float(image[5, 2, 2]), 0.75, places=5)
        self.assertEqual((int(types[0, 0, 2]), int(types[0, 0, 7]), int(types[0, 0, 0])), (1, 2, 0))
        line = np.array([[2.5, 0.5, z + 0.5] for z in range(10)])
        np.testing.assert_allclose(type_fractions(types, line, 2), [1.0, 0.0])
        exclude = np.zeros(volume.shape, dtype=bool)
        exclude[:, :, 7] = True
        image, types, _ = range_image(volume, ranges, denoise_sigma=0.0, exclude=exclude)
        self.assertEqual((float(image[0, 0, 7]), int(types[0, 0, 7])), (0.0, 0))
        with self.assertRaises(ValueError):
            range_image(volume, [(0.0, 0.5)], denoise_sigma=0.0)  # nothing darker: no void

    def test_grey_profiles_draw_measure_and_type_fibers(self):
        from tangle.ct import _grey

        shape = (32, 40, 40)
        void = 0.1
        solid = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 3-voxel radius, bright
        rimmed = np.array([0.3, 0.3, 0.3, 0.8, 0.8])  # 5-voxel radius, dim core, brighter rim
        lines = [
            np.array([[10.0, 10.0, z] for z in np.linspace(0.0, 32.0, 9)]),
            np.array([[28.0, 28.0, z] for z in np.linspace(0.0, 32.0, 9)]),
        ]
        radii = np.array([3.0, 5.0])
        grey = _grey.render_grey(np.zeros(3, dtype=int), np.array(shape[::-1]), lines, radii, [solid, rimmed], void)
        self.assertAlmostEqual(float(grey[16, 10, 10]), 1.0, places=6)  # axis of the solid fiber
        self.assertAlmostEqual(float(grey[16, 28, 28]), 0.3, places=6)  # dim core
        self.assertAlmostEqual(float(grey[16, 2, 2]), void, places=6)  # far from both
        np.testing.assert_allclose(_grey.squared_residual_map(grey, lines, radii, [solid, rimmed], void), 0.0, atol=1e-9)
        measured = _grey.measure_profiles(grey, lines, radii, np.array([0, 1]), 2, samples=5)
        np.testing.assert_allclose(measured[0], solid, atol=0.02)
        np.testing.assert_allclose(measured[1][:2], rimmed[:2], atol=0.05)
        self.assertGreater(measured[1][4], 0.7)
        found_void, noise, ranges = _grey.profile_levels(grey, [solid, rimmed])
        self.assertAlmostEqual(found_void, void, places=6)
        # The range is the bright body (the rim); the dim core falls below it.
        self.assertGreater(ranges[1][0], 0.3)
        self.assertLess(ranges[1][0], 0.8)
        self.assertGreater(ranges[1][1], 0.8)
        # A thin fiber's blurred profile peaks only on the axis; its range
        # reaches down to the grey 1.5 voxels out (0.7 at radius 2), less the
        # noise margin. On a thick fiber that grey is still near the peak.
        blurred = np.array([1.0, 0.95, 0.85, 0.7, 0.5])
        _, _, (thin, thick) = _grey.profile_levels(grey, [blurred, blurred], np.array([2.0, 8.0]))
        self.assertLess(thin[0], 0.7)
        self.assertGreater(thin[0], 0.6)
        self.assertGreater(thick[0], 0.9)
        types = _grey.profile_types(grey, lines, radii, [solid, rimmed], void)
        np.testing.assert_array_equal(types, [0, 1])

    def test_thin_fibers_are_rejected(self):
        with self.assertRaises(ValueError):
            ct.fit_fibers(self.scan.volume, VOXEL, ct.FiberSpec(diameter=1 * um))


@unittest.skipIf(ct is None, "tangle.ct needs NumPy and SciPy")
@unittest.skipUnless(GPU_FITS, "fits run on the GPU; set TANGLE_BACKEND to run them here")
class CtFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = crossing_scan()
        # One grey-scan fit, with the length prior, serves most tests: the
        # solver is slow on the CPU backend CI uses.
        cls.fit = ct.fit_fibers(cls.scan.volume, VOXEL, ct.FiberSpec(diameter=DIAMETER, length=200 * um), fit_settings())

    def test_fit_recovers_every_fiber(self):
        report = ct.score(self.fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        self.assertLess(report["centerline_error_voxels"], 0.5)
        self.assertLess(abs(report["diameter_bias_m"]), 0.05 * DIAMETER)
        self.assertGreater(report["voxel_label_accuracy"], 0.95)

    def test_fit_from_grey_ranges(self):
        from scipy.ndimage import gaussian_filter

        from tangle.ct._image import otsu_threshold

        grey = gaussian_filter(self.scan.volume.astype(np.float32), 0.7)
        values = grey[self.scan.labels > 0]
        low, high = np.percentile(values[values >= otsu_threshold(values)], [1, 99])
        spec = ct.FiberSpec(diameter=DIAMETER, length=200 * um, intensity=(float(low), float(high)))
        fit = ct.fit_fibers(self.scan.volume, VOXEL, spec, fit_settings())
        self.assertEqual(fit.history[0]["source"], "grey ranges")
        report = ct.score(fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        self.assertGreater(report["voxel_label_accuracy"], 0.9)
        with self.assertRaises(ValueError):
            ct.fit_fibers(self.scan.volume, VOXEL, [spec, ct.FiberSpec(diameter=2 * DIAMETER)], fit_settings())

    def test_fit_from_grey_profiles(self):
        from scipy.ndimage import gaussian_filter

        from tangle.ct._grey import measure_profiles

        grey = gaussian_filter(self.scan.volume.astype(np.float32), 0.7)
        kinds = np.zeros(len(self.scan.centerlines), dtype=int)
        profile = measure_profiles(grey, self.scan.centerlines, self.scan.radii, kinds, 1)[0]
        spec = ct.FiberSpec(diameter=DIAMETER, length=200 * um, profile=tuple(float(v) for v in profile))
        fit = ct.fit_fibers(self.scan.volume, VOXEL, spec, fit_settings())
        self.assertEqual(fit.history[0]["stage"], "profiles")
        self.assertEqual(next(e for e in fit.history if e["stage"] == "input")["source"], "grey profiles")
        report = ct.score(fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        self.assertGreater(report["voxel_label_accuracy"], 0.9)
        with tempfile.TemporaryDirectory() as tmp:
            reloaded = ct.load_fit(fit.write(tmp)["config"])
        self.assertEqual(reloaded.spec.profile, spec.profile)  # one type: kept on .spec

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

    def test_fit_reports_confidence_per_node(self):
        confidence = self.fit.confidence
        self.assertEqual([len(c) for c in confidence], [len(line) for line in self.fit.centerlines])
        values = np.concatenate(confidence)
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        # Three well separated, correctly fitted fibers: mostly sure.
        self.assertGreater(float(values.mean()), 0.6)
        stages = [entry["stage"] for entry in self.fit.history]
        self.assertIn("confidence", stages)
        coverage = next(e for e in self.fit.history if e["stage"] == "confidence")["sure_coverage"]
        self.assertGreater(coverage, 0.5)
        volume = self.fit.confidence_volume()
        fitted = self.fit.label_volume() > 0
        self.assertTrue(np.isnan(volume[~fitted]).all())
        self.assertFalse(np.isnan(volume[fitted]).any())
        with tempfile.TemporaryDirectory() as tmp:
            reloaded = ct.load_fit(self.fit.write(tmp)["config"])
        np.testing.assert_allclose(np.concatenate(reloaded.confidence), values, atol=1e-3)

    def test_overlay_colors_each_fiber(self):
        labels = self.fit.label_volume()
        z = labels.shape[0] // 2
        rgb = ct.overlay_slice(self.scan.volume[z], labels[z])
        self.assertEqual(rgb.shape, labels[z].shape + (3,))
        stack = ct.overlay_volume(self.scan.volume, labels)
        self.assertEqual(stack.dtype, np.uint8)
        self.assertEqual(stack.shape, labels.shape + (3,))

    def test_length_prior_summary(self):
        summary = self.fit.population_summary()
        self.assertIn("interior_ends", summary)
        self.assertGreater(summary["expected_interior_ends"], 0.0)

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

    def test_fit_is_valid_tangle_geometry(self):
        geometry = ct.geometry_report(
            self.fit.centerlines, self.fit.radii, 5 * DIAMETER / VOXEL, spacing=1.25 * DIAMETER / VOXEL
        )
        self.assertEqual(geometry["overlapping_pairs"], 0, geometry)
        self.assertEqual(geometry["fibers_over_bend_limit"], 0, geometry)
        self.assertTrue(any(entry["stage"] == "solver" for entry in self.fit.history))

    def test_fit_from_a_generous_mask(self):
        mask = self.scan.fiber_mask(level=0.35)  # over-reaches, like a generous threshold
        fit = ct.fit_fibers(mask, VOXEL, ct.FiberSpec(diameter=DIAMETER), fit_settings())
        report = ct.score(fit, self.scan)
        self.assertEqual(report["recovered"], 3, report)
        self.assertEqual(report["false_fibers"], 0, report)
        self.assertTrue(fit.history[0]["mask"])
        # Fibers look thicker in the mask; the estimated margin takes that off.
        self.assertLess(abs(report["diameter_bias_m"]), 0.1 * DIAMETER)

    def test_types_are_chosen_by_size(self):
        scan = two_type_scan()
        specs = [ct.FiberSpec(diameter=DIAMETER, name="small"), ct.FiberSpec(diameter=2 * DIAMETER, name="large")]
        fit = ct.fit_fibers(scan.fiber_mask(level=0.35), VOXEL, specs, fit_settings())
        report = ct.score(fit, scan)
        self.assertEqual(report["recovered"], 3, report)
        for kind in (0, 1):
            self.assertEqual(report["per_type"][kind]["fitted_as_this_type"], report["per_type"][kind]["fitted"], report["per_type"])
        geometry = ct.geometry_report(fit.centerlines, fit.radii, 5 * DIAMETER / VOXEL, spacing=1.25 * DIAMETER / VOXEL)
        self.assertEqual(geometry["overlapping_pairs"], 0, geometry)


if __name__ == "__main__":
    unittest.main()
