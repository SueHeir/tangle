"""Unit tests of the grey-model pass (``tangle.ct._model``): the summed-grey drawing, the measured profile,
the model recenter and the residual births, on tiny volumes of touching tubes with a dark halo. CPU only."""

import unittest
from types import SimpleNamespace

try:
    import numpy as np
    from scipy.special import erfc

    import tangle.ct as ct
    from tangle.ct import _model
except ImportError:  # the CT fitter needs NumPy and SciPy
    ct = None


SHAPE = (48, 48, 64)  # (z, y, x)
R = 3.25  # fiber radius, voxels
VOID, BRIGHT = 100.0, 200.0


def profile(d):
    """A blurred disc with a dark halo just outside it, as a phase-contrast scan shows a fiber."""
    return 0.5 * erfc((d - R) / (np.sqrt(2.0) * 0.9)) - 0.3 * np.exp(-0.5 * ((d - R - 1.4) / 0.8) ** 2)


def straight(y, z=24.0):
    x = np.arange(2.0, 62.1, R)
    return np.stack([x, np.full(len(x), y), np.full(len(x), z)], axis=1)


def scan(ys, amplitudes=None, noise=0.0, seed=0):
    """Straight fibers along x at heights ``ys`` (z = 24), their profiles summed over the void."""
    z, y, x = np.indices(SHAPE, dtype=np.float64) + 0.5
    volume = np.full(SHAPE, VOID)
    for i, fy in enumerate(ys):
        a = BRIGHT - VOID if amplitudes is None else amplitudes[i]
        volume += a * profile(np.hypot(y - fy, z - 24.0))
    if noise:
        volume += np.random.default_rng(seed).normal(0.0, noise, SHAPE)
    return volume.astype(np.float32)


def fitter():
    """A stand-in carrying what ``residual_births`` reads: one fiber type."""
    return SimpleNamespace(
        radius=np.array([R]), bend=np.array([30.0]), min_length=np.array([20.0]), spacing=R,
        settings=SimpleNamespace(trace_claim_radii=1.1),
    )


@unittest.skipIf(ct is None, "tangle.ct needs NumPy and SciPy")
class CtModelTests(unittest.TestCase):
    YS = (17.5, 24.0, 30.5)  # three fibers touching side by side (2 radii apart)

    def test_draw_sums_touching_fibers(self):
        lines = [straight(y) for y in self.YS]
        rasters = _model._rasters(SHAPE, lines, np.full(3, R), None)
        profiles = np.array([profile(_model._BINS * R)])
        drawn = _model.draw(SHAPE, rasters, np.zeros(3, int), np.full(3, BRIGHT - VOID), profiles)
        expected = scan(self.YS) - VOID
        inner = (slice(None), slice(None), slice(8, 56))  # away from the fibers' ends
        self.assertLess(np.abs(drawn[inner] - expected[inner]).max(), 3.0)

    def test_measured_profile_has_the_halo(self):
        grey = scan(self.YS, noise=2.0)
        lines = [straight(y) for y in self.YS]
        model = _model.GreyModel(grey, 1)
        model.update(lines, np.full(3, R), np.zeros(3, int))
        self.assertAlmostEqual(model.void, VOID, delta=2.0)
        measured = model.profile(0, _model._BINS)
        self.assertAlmostEqual(measured[0], 1.0, delta=0.1)
        self.assertLess(measured[_model._BINS >= 1.3].min(), -0.15)  # the dark halo
        self.assertTrue(np.allclose(model.amplitudes, BRIGHT - VOID, rtol=0.1))

    def test_model_recenter_puts_an_offset_fit_back_on_its_axis(self):
        grey = scan(self.YS, noise=2.0)
        offset = 0.45 * R  # the middle fit sits toward its neighbour's axis
        lines = [straight(self.YS[0]), straight(self.YS[1] + offset), straight(self.YS[2])]
        moved, _ = _model.model_recenter(
            _model.GreyModel(grey, 1), lines, np.full(3, R), np.zeros(3, int), 0, sweeps=2, max_radii=0.5
        )
        middle = moved[1][3:-3]  # away from the ends
        self.assertLess(np.abs(middle[:, 1] - self.YS[1]).max(), 0.35 * R)
        for i in (0, 2):
            self.assertLess(np.abs(moved[i][3:-3, 1] - self.YS[i]).max(), 0.3 * R)

    def test_residual_births_find_the_fiber_no_fit_explains(self):
        grey = scan(self.YS, noise=2.0)
        lines = [straight(self.YS[0]), straight(self.YS[2])]
        born = _model.residual_births(
            fitter(), _model.GreyModel(grey, 1), lines, np.full(2, R), np.zeros(2, int), 0,
            level=0.4, median_min=0.45, near_share=0.5,
        )
        self.assertEqual(len(born), 1)
        self.assertLess(np.abs(born[0][:, 1] - self.YS[1]).max(), 0.5 * R)
        self.assertGreater(np.ptp(born[0][:, 0]), 40.0)

    def test_residual_births_add_nothing_when_every_fiber_is_fitted(self):
        grey = scan(self.YS, noise=2.0)
        lines = [straight(y) for y in self.YS]
        born = _model.residual_births(
            fitter(), _model.GreyModel(grey, 1), lines, np.full(3, R), np.zeros(3, int), 0,
            level=0.4, median_min=0.45, near_share=0.5,
        )
        self.assertEqual(born, [])

    def test_settings_default_off(self):
        s = ct.FitSettings()
        self.assertIs(s.model_recenter, False)
        self.assertIs(s.residual_births, False)
        self.assertEqual(
            (s.model_recenter_sweeps, s.model_recenter_max_radii, s.residual_birth_level, s.residual_birth_min,
             s.residual_birth_near_share),
            (2, 0.5, 0.4, 0.45, 0.5),
        )


if __name__ == "__main__":
    unittest.main()
