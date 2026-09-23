import os
import unittest

import tangle
from tangle.units import um

try:
    import numpy as np
except ImportError:  # the image is built with NumPy
    np = None


VOXEL = 1.0 * um
SIDE = 24
RADIUS = 2.0 * um
# The solver runs on the GPU. CI runners have none, so these tests are
# skipped on CI unless TANGLE_BACKEND names a backend.
BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")
GPU = not (os.environ.get("CI") and "TANGLE_BACKEND" not in os.environ)


def tube_image(centers_yz_voxels, radius_voxels):
    """Soft bright tubes along x, normalized to void 0 and fiber 1."""
    coordinates = np.arange(SIDE) + 0.5
    z, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    value = np.zeros((SIDE, SIDE))
    for cy, cz in centers_yz_voxels:
        distance = np.hypot(y - cy, z - cz)
        value = np.maximum(value, np.clip(radius_voxels + 0.5 - distance, 0.0, 1.0))
    return np.ascontiguousarray(np.broadcast_to(value[:, :, None], (SIDE, SIDE, SIDE)), dtype="<f4")


def straight_assembly(offsets_yz_voxels):
    material = tangle.Material("fiber", diameter=2 * RADIUS)
    # Segments longer than a diameter, so a fiber's own non-adjacent segments
    # never touch.
    lines = [
        [[(3.0 + 4.5 * n) * VOXEL, y * VOXEL, z * VOXEL] for n in range(5)]
        for y, z in offsets_yz_voxels
    ]
    assembly = tangle.Assembly(tangle.Cell([SIDE * VOXEL] * 3))
    assembly.insert(tangle.FiberCollection.from_centerlines(lines, material))
    return assembly


@unittest.skipIf(np is None, "the test image needs NumPy")
@unittest.skipUnless(GPU, "runs on the GPU; set TANGLE_BACKEND to run it here")
class ImageRelaxerTests(unittest.TestCase):
    def relaxer(self, offsets, centers):
        image = tube_image(centers, RADIUS / VOXEL)
        # Tolerances in the scene's own units: the default penetration
        # tolerance (1e-4 m) would accept any overlap between these fibers.
        settings = tangle.RelaxationSettings(
            backend=BACKEND, max_step=0.25 * VOXEL, penetration_tolerance=0.01 * VOXEL
        )
        return tangle.ImageRelaxer(
            straight_assembly(offsets), settings, image.tobytes(), image.shape, VOXEL
        )

    def test_offset_fiber_converges_onto_a_bright_tube(self):
        relaxer = self.relaxer([(13.0, 12.0)], [(12.0, 12.0)])
        self.assertFalse(relaxer.image_force_active)
        relaxer.set_image_force(0.5)
        self.assertTrue(relaxer.image_force_active)
        status = relaxer.run(60)
        self.assertEqual(status["iterations"], 60)
        self.assertTrue(status["converged"], status)
        (line,) = relaxer.centerlines()
        points = np.asarray(line) / VOXEL
        np.testing.assert_allclose(points[:, 1], 12.0, atol=0.05)
        np.testing.assert_allclose(points[:, 2], 12.0, atol=0.05)
        (stats,) = relaxer.vertex_image_stats()
        self.assertEqual(len(stats), len(line))
        for mass, support in stats:
            self.assertGreater(support, 0.95)
            self.assertTrue(10.0 < mass < 16.0, mass)

    def test_zero_rate_leaves_the_fiber_in_place(self):
        relaxer = self.relaxer([(13.0, 12.0)], [(12.0, 12.0)])
        before = relaxer.centerlines()
        relaxer.run(10)
        np.testing.assert_allclose(relaxer.centerlines(), before, atol=1e-9)

    def test_neighboring_fibers_each_keep_their_own_tube(self):
        relaxer = self.relaxer([(10.5, 12.0), (13.5, 12.0)], [(9.9, 12.0), (14.1, 12.0)])
        relaxer.set_image_force(0.5)
        relaxer.run(80)
        first, second = (np.asarray(line) / VOXEL for line in relaxer.centerlines())
        np.testing.assert_allclose(first[:, 1], 9.9, atol=0.15)
        np.testing.assert_allclose(second[:, 1], 14.1, atol=0.15)

    def test_pinned_fiber_stays_put_while_its_neighbor_moves(self):
        # The first fiber is pinned 0.6 voxels off its tube; the second is
        # drawn toward its tube but stays two radii from the pinned one.
        relaxer = self.relaxer([(10.5, 12.0), (13.5, 12.0)], [(9.9, 12.0), (14.1, 12.0)])
        first, second = relaxer.centerlines()
        relaxer.set_pinned([[True] * len(first), [False] * len(second)])
        self.assertEqual(relaxer.pinned_count, len(first))
        relaxer.set_image_force(0.5)
        relaxer.run(80)
        relaxer.set_image_force(0.0)
        relaxer.run(50)
        pinned, free = (np.asarray(line) / VOXEL for line in relaxer.centerlines())
        np.testing.assert_allclose(pinned, np.asarray(first) / VOXEL, atol=1e-6)
        self.assertTrue(np.all((free[:, 1] > 14.45) & (free[:, 1] < 14.6)), free[:, 1])
        with self.assertRaises(ValueError):
            relaxer.set_pinned([[True] * len(first)])
        relaxer.set_pinned([[False] * len(first), [False] * len(second)])
        self.assertEqual(relaxer.pinned_count, 0)

    def test_rejects_a_mismatched_image(self):
        image = tube_image([(12.0, 12.0)], 2.0)
        with self.assertRaises(ValueError):
            tangle.ImageRelaxer(
                straight_assembly([(12.0, 12.0)]), None, image.tobytes()[:-4], image.shape, VOXEL
            )


if __name__ == "__main__":
    unittest.main()
