"""Unit tests of the fit's final passes (``tangle.ct._join``) and the settings that switch them on.

Everything here runs on the CPU in well under a second per test: tiny volumes with Gaussian tubes drawn in them
and stand-in fitters carrying only the attributes each pass reads.
"""

import dataclasses
import inspect
import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import scipy  # noqa: F401

    import tangle.ct as ct
except ImportError:  # the CT fitter needs NumPy and SciPy
    ct = None


SHAPE = (64, 64, 64)  # (z, y, x)


def tubes(*paths, shape=SHAPE, sigma=1.5):
    """A ``(z, y, x)`` float32 volume of Gaussian tubes. Each path is ``(polyline, amplitude)`` or a polyline
    (amplitude 1) in voxel coordinates ``(x, y, z)``; within a path the nearest segment counts, paths add up."""
    z, y, x = np.indices(shape, dtype=np.float64) + 0.5  # voxel (k, j, i) is read at (i + 0.5, j + 0.5, k + 0.5)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    volume = np.zeros(len(points))
    for path in paths:
        polyline, amplitude = path if isinstance(path, tuple) else (path, 1.0)
        polyline = np.asarray(polyline, dtype=np.float64)
        nearest = np.full(len(points), np.inf)
        for a, b in zip(polyline[:-1], polyline[1:]):
            ab = b - a
            t = np.clip((points - a) @ ab / float(ab @ ab), 0.0, 1.0)
            nearest = np.minimum(nearest, np.linalg.norm(points - a - t[:, None] * ab, axis=1))
        volume += amplitude * np.exp(-0.5 * (nearest / sigma) ** 2)
    return volume.reshape(shape).astype(np.float32)


def line(a, b, step=2.0):
    """A straight polyline from ``a`` to ``b`` with nodes about ``step`` apart."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = max(int(round(np.linalg.norm(b - a) / step)), 1)
    return a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)


def bend_radii(points):
    """Radius of the circle through each three consecutive nodes (inf where straight)."""
    a, b, c = points[:-2], points[1:-1], points[2:]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    sides = np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1)
    return np.where(area2 > 1e-12, sides / (2.0 * np.maximum(area2, 1e-300)), np.inf)


def monotonic(values, strict=True):
    d = np.diff(values)
    if strict:
        return bool(np.all(d > 0) or np.all(d < 0))
    return bool(np.all(d >= 0) or np.all(d <= 0))


def zigzag(amplitude):
    """Nodes 2 voxels apart along x, alternately ``amplitude`` above and below y = 32."""
    x = np.arange(4.0, 61.0, 2.0)
    return np.stack([x, 32.0 + amplitude * (-1.0) ** np.arange(len(x)), np.full(len(x), 32.0)], axis=1)


@unittest.skipIf(ct is None, "tangle.ct needs NumPy and SciPy")
class CtFinishTests(unittest.TestCase):
    """The final passes of ``_join`` on synthetic tubes; no fit, no GPU."""

    # -- straight_join --------------------------------------------------------------------------------------

    def test_straight_join_joins_collinear_pieces_across_a_gap(self):
        from tangle.ct._join import straight_join

        fitter = SimpleNamespace(radius=np.array([2.0]), evidence=tubes([[2, 32, 32], [62, 32, 32]]))
        a = line([5, 32, 32], [25, 32, 32])
        b = line([35, 32, 32], [58, 32, 32])[::-1]  # stored the other way round
        lines, radii, types, joins = straight_join(fitter, [a, b], [2.0, 2.0], [0, 0], {0}, 30.0, 15.0, 1.0, 20.0, 0.5)
        self.assertEqual(joins, 1)
        self.assertEqual(len(lines), 1)
        joined = lines[0]
        self.assertTrue(monotonic(joined[:, 0]))
        self.assertAlmostEqual(float(joined[:, 0].min()), 5.0)
        self.assertAlmostEqual(float(joined[:, 0].max()), 58.0)
        np.testing.assert_allclose(joined[:, 1:], 32.0)
        self.assertEqual(types.tolist(), [0])
        self.assertAlmostEqual(float(radii[0]), 2.0)

    def test_straight_join_needs_a_bright_bridge_when_support_is_set(self):
        from tangle.ct._join import straight_join

        a = line([5, 32, 32], [25, 32, 32])
        b = line([37, 32, 32], [58, 32, 32])
        fitter = SimpleNamespace(radius=np.array([2.0]), evidence=tubes(a, b))  # void across the gap
        args = ([a, b], [2.0, 2.0], [0, 0], {0}, 30.0, 15.0, 1.0, 20.0)
        self.assertEqual(straight_join(fitter, *args, 0.5)[3], 0)
        self.assertEqual(straight_join(fitter, *args, None)[3], 1)

    def test_straight_join_refuses_a_kink(self):
        from tangle.ct._join import straight_join

        a = line([5, 32, 32], [25, 32, 32])
        turn = np.radians(30.0)
        start = np.array([28.0, 32.0, 32.0])  # a short gap, so both extended axes still pass the other's end
        b = line(start, start + 25.0 * np.array([np.cos(turn), np.sin(turn), 0.0]))
        fitter = SimpleNamespace(radius=np.array([2.0]), evidence=tubes(a, np.stack([a[-1], start]), b))
        args = ([a, b], [2.0, 2.0], [0, 0], {0}, 30.0)
        lines, _, _, joins = straight_join(fitter, *args, 15.0, 1.0, 20.0, 0.5)
        self.assertEqual(joins, 0)
        self.assertEqual(len(lines), 2)
        # the same pair within a 35 degree limit: the angle is what refused it
        self.assertEqual(straight_join(fitter, *args, 35.0, 1.0, 20.0, 0.5)[3], 1)

    def test_straight_join_refuses_a_sideways_offset(self):
        from tangle.ct._join import straight_join

        a = line([5, 32, 32], [25, 32, 32])
        b = line([35, 38, 32], [58, 38, 32])  # parallel, 6 voxels (3 radii) to the side
        fitter = SimpleNamespace(radius=np.array([2.0]), evidence=tubes(a, b))
        args = ([a, b], [2.0, 2.0], [0, 0], {0}, 30.0, 15.0)
        lines, _, _, joins = straight_join(fitter, *args, 1.0, 20.0, None)
        self.assertEqual(joins, 0)
        self.assertEqual(len(lines), 2)
        self.assertEqual(straight_join(fitter, *args, 4.0, 20.0, None)[3], 1)  # the offset is what refused it

    def test_straight_join_drops_the_overlap_of_pieces_running_past_each_other(self):
        from tangle.ct._join import straight_join

        a = line([5, 32, 32], [31, 32, 32])
        b = line([25, 32, 32], [58, 32, 32])
        fitter = SimpleNamespace(radius=np.array([2.0]), evidence=tubes([[2, 32, 32], [62, 32, 32]]))
        lines, _, _, joins = straight_join(fitter, [a, b], [2.0, 2.0], [0, 0], {0}, 30.0, 15.0, 1.0, 20.0, 0.5)
        self.assertEqual(joins, 1)
        self.assertEqual(len(lines), 1)
        self.assertTrue(monotonic(lines[0][:, 0]))  # the second piece's start behind the first's end is gone

    def test_straight_join_leaves_other_types_alone(self):
        from tangle.ct._join import straight_join

        fitter = SimpleNamespace(radius=np.array([2.0, 4.0]), evidence=tubes([[2, 32, 32], [62, 32, 32]]))
        a = line([5, 32, 32], [25, 32, 32])
        b = line([35, 32, 32], [58, 32, 32])
        lines, _, types, joins = straight_join(fitter, [a, b], [4.0, 4.0], [1, 1], {0}, 30.0, 15.0, 1.0, 20.0, 0.5)
        self.assertEqual(joins, 0)
        self.assertEqual(len(lines), 2)
        self.assertEqual(types.tolist(), [1, 1])

    # -- coarse_join ----------------------------------------------------------------------------------------

    def coarse_pair(self, band):
        """Two fits of the larger type running past each other side by side, one on each rim of a dim-cored oval
        fiber (the two rims 8 voxels apart, 2 short radii); with ``band``, a bright fiber between the two."""
        from tangle.ct._join import coarse_join

        rims = [([[0, 28, 32], [64, 28, 32]], 1.0), ([[0, 36, 32], [64, 36, 32]], 1.0)]
        if band:
            rims.append(([[0, 32, 32], [64, 32, 32]], 1.5))
        grey = tubes(*rims)
        fitter = SimpleNamespace(radius=np.array([2.0, 4.0]), axis_grey=grey, axis_range=(0.0, 1.0), evidence=grey)
        a = line([5, 28, 32], [35, 28, 32])
        b = line([25, 36, 32], [58, 36, 32])
        fine = line([5, 50, 20], [58, 50, 20])  # the smallest type: never touched
        return coarse_join(fitter, [a, b, fine], [4.0, 4.0, 2.0], [1, 1, 0])

    def test_coarse_join_joins_a_side_by_side_pair_on_one_fiber(self):
        lines, radii, types, info = self.coarse_pair(band=False)
        self.assertEqual(info, {"gap_joins": 0, "side_joins": 1})
        self.assertEqual(sorted(types.tolist()), [0, 1])
        joined = lines[types.tolist().index(1)]
        self.assertTrue(monotonic(joined[:, 0], strict=False))  # the midline's nodes of both fits, sorted along
        self.assertAlmostEqual(float(joined[:, 0].min()), 5.0)
        self.assertAlmostEqual(float(joined[:, 0].max()), 58.0)
        overlap = (joined[:, 0] > 26.0) & (joined[:, 0] < 34.0)
        self.assertTrue(overlap.any())
        np.testing.assert_allclose(joined[overlap, 1], 32.0, atol=0.5)  # the midline replaces the overlap
        np.testing.assert_allclose(joined[joined[:, 0] < 24.0, 1], 28.0)
        np.testing.assert_allclose(joined[joined[:, 0] > 36.0, 1], 36.0)
        self.assertAlmostEqual(float(radii[types.tolist().index(1)]), 4.0)

    def test_coarse_join_keeps_a_pair_with_a_bright_band_between_apart(self):
        lines, _, types, info = self.coarse_pair(band=True)
        self.assertEqual(info, {"gap_joins": 0, "side_joins": 0})
        self.assertEqual(sorted(types.tolist()), [0, 1, 1])
        self.assertEqual(len(lines), 3)

    def test_coarse_join_makes_no_side_join_without_the_axis_grey(self):
        from tangle.ct._join import coarse_join

        # the pair coarse_pair joins, but with no axis grey to show there is no bright band between them
        grey = tubes(([[0, 28, 32], [64, 28, 32]], 1.0), ([[0, 36, 32], [64, 36, 32]], 1.0))
        fitter = SimpleNamespace(radius=np.array([2.0, 4.0]), axis_grey=None, axis_range=(0.0, 1.0), evidence=grey)
        a = line([5, 28, 32], [35, 28, 32])
        b = line([25, 36, 32], [58, 36, 32])
        lines, _, _, info = coarse_join(fitter, [a, b], [4.0, 4.0], [1, 1])
        self.assertEqual(info, {"gap_joins": 0, "side_joins": 0})
        self.assertEqual(len(lines), 2)

    # -- bend_finish ----------------------------------------------------------------------------------------

    def bend_fitter(self, bend=40.0, min_length=10.0):
        return SimpleNamespace(spacing=2.0, bend=np.array([bend]), min_length=np.array([min_length]), peak_void=0.0)

    def test_bend_finish_straightens_a_zigzag(self):
        from tangle.ct._join import bend_finish

        rough = zigzag(1.5)
        self.assertLess(float(bend_radii(rough).min()), 3.0)
        lines, radii, types, info = bend_finish(self.bend_fitter(), [rough], [2.0], [0], {0}, None, None)
        self.assertEqual(len(lines), 1)
        np.testing.assert_allclose(lines[0][[0, -1]], rough[[0, -1]], atol=1e-9)  # the ends stay
        self.assertLess(float(np.abs(lines[0][4:-4, 1] - 32.0).max()), 0.1)
        self.assertGreater(info["moved_voxels_mean"], 0.5)
        self.assertEqual(info["pieces_dropped"], 0)
        self.assertEqual(types.tolist(), [0])
        self.assertAlmostEqual(float(radii[0]), 2.0)

    def test_bend_finish_keeps_a_gentle_zigzag_to_the_bend_limit(self):
        from tangle.ct._join import bend_finish

        gentle = zigzag(0.3)
        self.assertLess(float(bend_radii(gentle).min()), 4.0)
        lines, _, _, info = bend_finish(self.bend_fitter(), [gentle], [2.0], [0], {0}, None, None)
        self.assertGreaterEqual(float(bend_radii(lines[0]).min()), 0.9 * 40.0)
        self.assertGreater(info["moved_voxels_mean"], 0.0)

    @unittest.expectedFailure
    def test_bend_finish_keeps_a_rough_zigzag_to_the_bend_limit(self):
        # The sagitta limit s**2 / (2 R) is set once from the mean segment length s before smoothing. Smoothing a
        # rough trace shortens its segments (here from 1.7 to 1.13 voxels), so the limit it converges to is
        # R (1.13 / 1.7)**2, about 0.44 R: the output bends at 17.8 voxels under a 40 voxel limit.
        from tangle.ct._join import bend_finish

        lines, _, _, _ = bend_finish(self.bend_fitter(), [zigzag(1.5)], [2.0], [0], {0}, None, None)
        self.assertGreaterEqual(float(bend_radii(lines[0]).min()), 0.9 * 40.0)

    def test_bend_finish_leaves_a_straight_line_and_a_gentle_arc_alone(self):
        from tangle.ct._join import bend_finish

        straight = line([4, 20, 32], [60, 44, 32])
        lines, _, _, info = bend_finish(self.bend_fitter(), [straight], [2.0], [0], {0}, None, None)
        self.assertEqual(info["moved_voxels_mean"], 0.0)
        np.testing.assert_allclose(lines[0][[0, -1]], straight[[0, -1]], atol=1e-9)
        direction = (straight[-1] - straight[0]) / np.linalg.norm(straight[-1] - straight[0])
        off = lines[0] - straight[0]
        np.testing.assert_allclose(off - (off @ direction)[:, None] * direction, 0.0, atol=1e-9)
        # an arc of radius 80 keeps to a 40 voxel limit and is left as it is ...
        angle = np.linspace(-0.35, 0.35, 29)
        arc = np.stack([32 + 80 * np.sin(angle), 32 + 80 * (1 - np.cos(angle)) - 3.0, np.full(len(angle), 32.0)], axis=1)
        _, _, _, info = bend_finish(self.bend_fitter(), [arc], [2.0], [0], {0}, None, None)
        self.assertEqual(info["moved_voxels_mean"], 0.0)
        # ... but not under a 400 voxel limit given per type (bends=, the bend_finish_diameters path). A long
        # gentle bend needs about as many sweeps as its node count squared, so check the limit with enough of them.
        lines, _, _, info = bend_finish(self.bend_fitter(), [arc], [2.0], [0], {0}, None, None, bends=np.array([400.0]))
        self.assertGreater(info["moved_voxels_mean"], 0.0)
        self.assertGreater(float(bend_radii(lines[0]).min()), 1.5 * 80.0)
        lines, _, _, _ = bend_finish(self.bend_fitter(), [arc], [2.0], [0], {0}, None, None, iterations=4000,
                                     bends=np.array([400.0]))
        self.assertGreaterEqual(float(bend_radii(lines[0]).min()), 0.9 * 400.0)

    def test_bend_finish_leaves_other_types_alone(self):
        from tangle.ct._join import bend_finish

        rough = zigzag(1.5)
        fitter = SimpleNamespace(spacing=2.0, bend=np.array([40.0, 80.0]), min_length=np.array([10.0, 20.0]))
        lines, _, _, info = bend_finish(fitter, [rough], [4.0], [1], {0}, None, None)
        np.testing.assert_array_equal(lines[0], rough)
        self.assertEqual(info["moved_voxels_mean"], 0.0)

    def test_bend_finish_cuts_a_stretch_over_void_with_grey_min(self):
        from tangle.ct._join import bend_finish

        fit = line([4, 32, 32], [60, 32, 32])
        grey = tubes([[4, 32, 32], [20, 32, 32]], [[32, 32, 32], [60, 32, 32]])  # void from x = 20 to 32
        lines, _, _, info = bend_finish(self.bend_fitter(), [fit], [2.0], [0], {0}, grey, None)
        self.assertEqual(len(lines), 1)  # no cut without grey_min
        self.assertEqual(info["cut_voxels"], 0.0)
        lines, _, types, info = bend_finish(self.bend_fitter(), [fit], [2.0], [0], {0}, grey, 0.6)
        self.assertEqual(len(lines), 2)
        self.assertEqual(types.tolist(), [0, 0])
        self.assertGreater(info["cut_voxels"], 6.0)
        ends = sorted((float(p[:, 0].min()), float(p[:, 0].max())) for p in lines)
        self.assertLess(ends[0][1], 23.0)
        self.assertGreater(ends[1][0], 29.0)
        # a piece the cut leaves shorter than the type's minimum length is dropped
        lines, _, _, info = bend_finish(self.bend_fitter(min_length=20.0), [fit], [2.0], [0], {0}, grey, 0.6)
        self.assertEqual(len(lines), 1)
        self.assertEqual(info["pieces_dropped"], 1)
        self.assertGreater(float(lines[0][:, 0].min()), 29.0)

    # -- clean_finish ---------------------------------------------------------------------------------------

    def test_clean_finish_trims_a_doubled_stretch(self):
        from tangle.ct._join import clean_finish

        grey = 200.0 * tubes([[2, 32, 32], [62, 32, 32]])  # grey in scan units
        fitter = SimpleNamespace(radius=np.array([2.0]), spacing=2.0)
        long = line([4, 32, 32], [50, 32, 32])
        short = line([40, 33, 32], [60, 33, 32])  # one voxel off the long fit, on the same fiber, for x <= 50
        lines, _, _, info = clean_finish(fitter, [long, short], [2.0, 2.0], [0, 0], {0}, grey, 1.25, None)
        self.assertGreater(info["doubled_voxels"], 0)
        self.assertEqual(len(lines), 2)
        np.testing.assert_array_equal(lines[0], long)  # the longer fit keeps its whole length
        self.assertGreater(float(lines[1][:, 0].min()), 51.0)
        self.assertAlmostEqual(float(lines[1][:, 0].max()), 60.0)
        # with a minimum length, the trimmed remnant goes too
        lines, _, _, info = clean_finish(fitter, [long, short], [2.0, 2.0], [0, 0], {0}, grey, 1.25, 10.0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(info["short_dropped"], 1)

    def two_touching(self, scale):
        """Two thin fibers 2.2 voxels apart (inside the 1.25 radius reach), with a dip at their contact."""
        from tangle.ct._join import clean_finish

        grey = scale * tubes([[0, 31.0, 32], [64, 31.0, 32]], [[0, 33.2, 32], [64, 33.2, 32]], sigma=0.6)
        fitter = SimpleNamespace(radius=np.array([2.0]), spacing=2.0)
        a = line([4, 31.0, 32], [60, 31.0, 32])
        b = line([10, 33.2, 32], [50, 33.2, 32])
        return clean_finish(fitter, [a, b], [2.0, 2.0], [0, 0], {0}, grey, 1.25, None), (a, b)

    def test_clean_finish_keeps_two_fits_with_a_dark_gap_between(self):
        (lines, _, _, info), (a, b) = self.two_touching(200.0)
        self.assertEqual(info["doubled_voxels"], 0)
        self.assertEqual(len(lines), 2)
        np.testing.assert_array_equal(lines[0], a)
        np.testing.assert_array_equal(lines[1], b)

    @unittest.expectedFailure
    def test_clean_finish_dark_gap_test_does_not_depend_on_the_grey_units(self):
        # The "no darker gap" test reads midpoint >= dimmer axis - 1.0 in absolute grey units: on a volume in
        # [0, 1] every pair within reach counts as doubled, and the shorter of two touching fibers is cut.
        (lines, _, _, info), _ = self.two_touching(1.0)
        self.assertEqual(info["doubled_voxels"], 0)
        self.assertEqual(len(lines), 2)

    # -- merge_short ----------------------------------------------------------------------------------------

    def test_merge_short_attaches_pieces_on_the_extended_axis(self):
        from tangle.ct._join import merge_short

        fitter = SimpleNamespace(radius=np.array([2.0]))
        long = line([20, 32, 32], [44, 32, 32])
        ahead = line([50, 32.5, 32], [56, 32.5, 32])
        behind = line([14, 32, 32], [8, 32, 32])  # stored pointing away from the long fit
        lines, _, types, merged = merge_short(fitter, [ahead, long, behind], [2.0] * 3, [0] * 3, {0}, 10.0, 30.0, 1.5)
        self.assertEqual(merged, 2)
        self.assertEqual(len(lines), 1)
        self.assertTrue(monotonic(lines[0][:, 0]))
        self.assertAlmostEqual(float(lines[0][:, 0].min()), 8.0)
        self.assertAlmostEqual(float(lines[0][:, 0].max()), 56.0)
        self.assertEqual(len(lines[0]), len(long) + len(ahead) + len(behind))

    def test_merge_short_leaves_a_piece_off_the_axis(self):
        from tangle.ct._join import merge_short

        fitter = SimpleNamespace(radius=np.array([2.0]))
        long = line([20, 32, 32], [44, 32, 32])
        aside = line([50, 37, 32], [56, 37, 32])  # 5 voxels (2.5 radii) off the axis
        far = line([80, 32, 32], [86, 32, 32])  # on the axis, beyond the gap
        lines, _, _, merged = merge_short(fitter, [long, aside, far], [2.0] * 3, [0] * 3, {0}, 10.0, 30.0, 1.5)
        self.assertEqual(merged, 0)
        self.assertEqual(len(lines), 3)
        np.testing.assert_array_equal(lines[0], long)

    # -- extend_through_ridge -------------------------------------------------------------------------------

    def test_extend_through_ridge_walks_to_the_faces_along_a_tube(self):
        from tangle.ct._join import extend_through_ridge

        fitter = SimpleNamespace(radius=np.array([2.0]))
        # axis on voxel centres: between two voxel centres the trilinear grey is flat, and the walk may drift there
        grey = tubes([[-2, 32.5, 32.5], [66, 32.5, 32.5]])
        fit = line([20, 32.5, 32.5], [40, 32.5, 32.5])
        lines, info = extend_through_ridge(fitter, [fit], [2.0], [0], {0}, grey, 0.0, 1.0, 0.3, 5, 50.0)
        self.assertEqual(info["ends_reached_face"], 2)
        self.assertEqual(info["stop_face"], 2)
        grown = lines[0]
        self.assertTrue(monotonic(grown[:, 0]))
        self.assertLessEqual(float(grown[:, 0].min()), 5.0)
        self.assertGreaterEqual(float(grown[:, 0].max()), 58.0)
        np.testing.assert_allclose(grown[:, 1:], 32.5, atol=0.1)

    def test_extend_through_ridge_follows_a_bending_tube(self):
        from tangle.ct._join import extend_through_ridge

        turn = np.radians(20.0)
        corner = np.array([40.0, 24.0, 32.0])
        far = corner + 30.0 * np.array([np.cos(turn), np.sin(turn), 0.0])
        grey = tubes([[-2, 24, 32], corner, far])
        fitter = SimpleNamespace(radius=np.array([2.0]))
        fit = line([15, 24, 32], [35, 24, 32])
        lines, info = extend_through_ridge(fitter, [fit], [2.0], [0], {0}, grey, 0.0, 1.0, 0.3, 5, 10.0)
        self.assertEqual(info["ends_reached_face"], 2)
        end = lines[0][np.argmax(lines[0][:, 0])]
        self.assertGreaterEqual(float(end[0]), 58.0)
        on_tube = corner + ((end - corner) @ (far - corner)) / float((far - corner) @ (far - corner)) * (far - corner)
        self.assertLess(float(np.linalg.norm(end - on_tube)), 1.5)

    def test_extend_through_ridge_stops_at_void(self):
        from tangle.ct._join import extend_through_ridge

        fitter = SimpleNamespace(radius=np.array([2.0]))
        grey = tubes([[12, 32, 32], [52, 32, 32]])
        fit = line([20, 32, 32], [40, 32, 32])
        lines, info = extend_through_ridge(fitter, [fit], [2.0], [0], {0}, grey, 0.0, 1.0, 0.3, 5, 50.0)
        self.assertEqual(info["stop_dim"], 2)
        self.assertEqual(info["ends_reached_face"], 0)
        x = lines[0][:, 0]
        self.assertTrue(8.0 <= float(x.min()) <= 12.0)  # the tube's cap reads bright a voxel or two past its end
        self.assertTrue(52.0 <= float(x.max()) <= 56.0)

    def test_extend_through_ridge_stops_at_a_fit_on_its_way_and_passes_a_crossing(self):
        from tangle.ct._join import extend_through_ridge

        fitter = SimpleNamespace(radius=np.array([2.0]))
        grey = tubes([[-2, 32, 32], [66, 32, 32]], [[48, -2, 32], [48, 66, 32]])
        fit = line([20, 32, 32], [36, 32, 32])
        crossing = line([48, 6, 32], [48, 58, 32])
        lines, info = extend_through_ridge(fitter, [fit, crossing], [2.0, 2.0], [0, 0], {0}, grey, 0.0, 1.0, 0.3, 5,
                                           50.0)
        self.assertEqual(info["stop_blocked"], 0)
        self.assertGreaterEqual(float(lines[0][:, 0].max()), 58.0)
        ahead = line([46, 32, 32], [60, 32, 32])  # a fit running on along the same fiber
        lines, info = extend_through_ridge(fitter, [fit, ahead], [2.0, 2.0], [0, 0], {0}, grey, 0.0, 1.0, 0.3, 5, 50.0)
        self.assertGreaterEqual(info["stop_blocked"], 1)
        self.assertLess(float(lines[0][:, 0].max()), 46.0)

    def test_extend_through_ridge_ends_on_a_closed_ridge(self):
        from tangle.ct._join import extend_through_ridge

        angle = np.linspace(0.0, 2.0 * np.pi, 49)
        ring = np.stack([32.5 + 20.0 * np.cos(angle), 32.5 + 20.0 * np.sin(angle), np.full(len(angle), 32.5)], axis=1)
        fitter = SimpleNamespace(radius=np.array([2.0]))
        fit = ring[:13]  # a quarter of a ring fiber inside the block: the walks can follow it round and round
        lines, info = extend_through_ridge(fitter, [fit], [2.0], [0], {0}, tubes(ring), 0.0, 1.0, 0.3, 5, 10.0)
        self.assertEqual(info["stop_loop"], 2)
        self.assertEqual(info["grown_voxels"], 0)
        np.testing.assert_array_equal(lines[0], fit)

    def test_extend_through_ridge_skips_an_end_with_no_direction(self):
        from tangle.ct._join import extend_through_ridge

        fitter = SimpleNamespace(radius=np.array([2.0]))
        grey = tubes([[-2, 32.5, 32.5], [66, 32.5, 32.5]])
        point = np.array([[30.0, 32.5, 32.5], [30.0, 32.5, 32.5]])  # two coincident nodes
        lines, info = extend_through_ridge(fitter, [point], [2.0], [0], {0}, grey, 0.0, 1.0, 0.3, 5, 50.0)
        self.assertEqual(info["grown_voxels"], 0)
        np.testing.assert_array_equal(lines[0], point)

    def test_extend_through_ridge_walks_toward_each_other_meet(self):
        from tangle.ct._join import extend_through_ridge

        fitter = SimpleNamespace(radius=np.array([2.0]))
        grey = tubes([[-2, 32.5, 32.5], [66, 32.5, 32.5]])
        a = line([8, 32.5, 32.5], [20, 32.5, 32.5])
        b = line([44, 32.5, 32.5], [56, 32.5, 32.5])  # one fiber, two pieces
        lines, info = extend_through_ridge(fitter, [a, b], [2.0, 2.0], [0, 0], {0}, grey, 0.0, 1.0, 0.3, 5, 50.0)
        self.assertEqual(info["stop_blocked"], 2)
        self.assertLess(float(lines[0][:, 0].max()), float(lines[1][:, 0].min()))  # the gap is filled once
        self.assertGreater(float(lines[0][:, 0].max()), 38.0)

    # -- through_block --------------------------------------------------------------------------------------

    def test_through_block_reports_through_share(self):
        from tangle.ct._join import through_block

        grey = tubes([[-2, 20, 20], [66, 20, 20]], [[44, 12, 44], [44, 52, 44]])  # through, and ending inside
        fitter = SimpleNamespace(radius=np.array([2.0]), spacing=2.0, min_length=np.array([10.0]))
        pieces = [line([10, 20, 20], [26, 20, 20]), line([34, 20, 20], [54, 20, 20])]
        inner = line([44, 16, 44], [44, 48, 44])
        lines, _, types, info = through_block(fitter, pieces + [inner], [2.0] * 3, [0] * 3, {0}, grey, grey, 0.0, 20.0)
        self.assertEqual(len(lines), 2)
        self.assertGreaterEqual(info["joins"], 1)
        self.assertEqual(info["ghost_voxels_cut"], 0)
        self.assertEqual(info["through_share"], 0.5)
        through = max(lines, key=lambda l: float(np.ptp(l[:, 0])))
        self.assertLessEqual(float(through[:, 0].min()), 5.0)
        self.assertGreaterEqual(float(through[:, 0].max()), 58.0)
        inside = min(lines, key=lambda l: float(np.ptp(l[:, 0])))
        self.assertGreater(float(inside[:, 1].min()), 8.0)
        self.assertLess(float(inside[:, 1].max()), 56.0)


@unittest.skipIf(ct is None, "tangle.ct needs NumPy and SciPy")
class CtFinishSettingsTests(unittest.TestCase):
    """The settings of the final passes and of the coarse-zone typing are off by default, so the default fit
    does not run them."""

    DEFAULTS = {
        # coarse zone
        "coarse_axis_grey_min": None, "coarse_axis_depth_min": 4.0, "coarse_axis_disc_max": 0.3,
        "coarse_disc_classify": False, "coarse_disc_trace": True, "coarse_disc_end": True,
        "birth_ridge_smallest_only": False, "coarse_birth_disc_max": None,
        # final passes
        "coarse_join": False,
        "straight_join": False, "straight_join_types": "smallest", "straight_join_gap": 30.0,
        "straight_join_angle": 15.0, "straight_join_offset": 1.0, "straight_join_overlap": 20.0,
        "straight_join_support": 0.5,
        "bend_finish": False, "bend_finish_types": "smallest", "bend_finish_grey_min": 0.6,
        "bend_finish_diameters": None,
        "clean_finish": False, "clean_finish_reach_radii": 1.25, "clean_finish_min_length": None,
        "merge_short": False, "merge_short_length": None, "merge_short_gap": 30.0, "merge_short_join_gap": 60.0,
        "through_block": False,
        # ridge finish
        "ridge_sigma_voxels": None, "ridge_min_length_scale": 1.0, "ridge_refine": False,
        "ridge_extend_reach": 0.5, "ridge_rim_share": None, "ridge_rim_reach": 1.35, "ridge_rim_grey": 0.58,
        # rim exemption and hole births
        "ridge_rim_sigma": None,
        "hole_births": False, "hole_birth_ridge_min": 0.5, "hole_birth_depth_radii": 0.6,
        "hole_birth_claim_radii": 1.2, "hole_birth_grey_min": None, "hole_birth_passes": 1,
        "hole_birth_grey_void": False, "hole_birth_rim_drop": False, "hole_birth_rim_share": 0.7,
    }

    def test_every_setting_has_its_documented_default(self):
        fields = {f.name for f in dataclasses.fields(ct.FitSettings)}
        self.assertEqual(sorted(set(self.DEFAULTS) - fields), [])
        settings = ct.FitSettings()
        for name, default in self.DEFAULTS.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(settings, name), default)

    def test_defaults_switch_every_pass_off(self):
        s = ct.FitSettings()
        for name in ("coarse_join", "straight_join", "bend_finish", "clean_finish", "merge_short", "through_block",
                     "hole_births", "ridge_refine", "coarse_disc_classify", "birth_ridge_smallest_only"):
            with self.subTest(name=name):
                self.assertIs(getattr(s, name), False)
        for name in ("coarse_axis_grey_min", "coarse_birth_disc_max", "ridge_sigma_voxels", "ridge_rim_share",
                     "ridge_rim_sigma"):
            with self.subTest(name=name):
                self.assertIsNone(getattr(s, name))
        # The settings that default to on only act under a switch that is off by default.
        self.assertIsNone(s.coarse_disc_max)  # coarse_disc_trace, coarse_disc_end
        self.assertFalse(s.ridge_finish)  # ridge_extend_reach, ridge_min_length_scale
        self.assertFalse(s.final_births)  # hole_births needs final_births

    def test_ridge_finish_defaults_reproduce_the_fixed_values(self):
        from tangle.ct import _ridge

        s = ct.FitSettings()
        self.assertEqual(inspect.signature(_ridge._extend).parameters["reach"].default, s.ridge_extend_reach)
        self.assertEqual(s.ridge_min_length_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
