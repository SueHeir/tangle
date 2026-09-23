import tempfile
import unittest
from pathlib import Path

import tangle


class CollectionTests(unittest.TestCase):
    def test_collection_preserves_placed_and_rest_centerlines(self):
        material = tangle.Material(
            "fiber", diameter=7.0e-6, minimum_bend_radius=35.0e-6
        )
        collection = tangle.FiberCollection("ply")
        collection.add_fiber(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]],
            material,
            rest_centerline=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            tags={"family": "machine"},
            formation_layer=3,
        )
        self.assertEqual(len(collection), 1)
        self.assertEqual(collection.centerlines()[0][1], [1.0, 0.2, 0.0])
        self.assertEqual(collection.rest_centerlines()[0][1], [1.0, 0.0, 0.0])

    def test_recipe_insert_returns_persistent_selection(self):
        material = tangle.Material("fiber", diameter=0.1)
        collection = tangle.FiberCollection("crossing")
        collection.add_fiber([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material)
        assembly = tangle.Assembly(tangle.Cell([2.0, 2.0, 2.0]))
        recipe = tangle.Recipe(assembly)
        selection = recipe.insert(collection, name="first", translation=[0.0, 0.5, 0.5])
        self.assertEqual(selection.name, "first")
        self.assertEqual(selection.fiber_ids, [1])
        self.assertEqual(selection.formation_step, 0)
        self.assertEqual(assembly.fiber_count, 1)
        self.assertIn("insert", recipe.operations()[0])

    def test_insert_accepts_a_rigid_rotation_matrix(self):
        material = tangle.Material("fiber", diameter=0.1)
        collection = tangle.FiberCollection("one")
        collection.add_fiber([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material)
        recipe = tangle.Recipe(tangle.Cell([2.0, 2.0, 2.0]))
        recipe.insert(
            collection,
            rotation=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(recipe.centerlines()[0][1], [0.0, 1.0, 0.0])

    def test_native_generators_return_reusable_collections(self):
        cell = tangle.Cell([1.0, 1.0, 1.0])
        crossing = tangle.generate_point_crossing(cell, count=4)
        self.assertEqual(len(crossing), 4)

        population = tangle.FiberPopulationSettings()
        population.count = 6
        population.segments_per_fiber = 2
        population.position = "layered"
        population.layers = 2
        population.jitter_fraction = 0.1
        generated = tangle.generate_fiber_population(cell, population)
        self.assertEqual(len(generated), 6)
        self.assertEqual(generated.layers(), [0, 1])
        first = generated.select_layer(0)
        second = generated.select_layer(1)
        first.extend(second)
        self.assertEqual(len(first), 6)

    def test_native_analysis_and_puma_bundle_share_the_rust_assembly(self):
        material = tangle.Material("fiber", diameter=0.2)
        collection = tangle.FiberCollection("one")
        collection.add_fiber(
            [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]], material
        )
        assembly = tangle.Assembly(tangle.Cell([1.0, 1.0, 1.0]))
        tangle.Recipe(assembly).insert(collection)

        analysis = assembly.characterize()
        self.assertEqual(analysis.fiber_count, 1)
        self.assertEqual(analysis.segment_count, 1)
        self.assertAlmostEqual(
            analysis.length_weighted_orientation_tensor[0][0], 1.0
        )
        self.assertEqual(analysis.to_dict()["schema_version"], 1)
        self.assertIn('"schema_version": 1', analysis.to_json())

        with tempfile.TemporaryDirectory() as directory:
            report = assembly.export_puma(Path(directory) / "case.puma", 0.1)
            self.assertEqual(report.voxel_counts, [10, 10, 10])
            self.assertGreater(report.occupied_voxels, 0)
            self.assertTrue(report.domain_path.is_file())
            self.assertTrue(report.manifest_path.is_file())
            self.assertTrue(report.analysis_path.is_file())

    def test_assembly_insert_adds_fibers_without_a_recipe(self):
        material = tangle.Material("fiber", diameter=0.1)
        collection = tangle.FiberCollection("scan")
        collection.add_fiber([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material)
        assembly = tangle.Assembly(tangle.Cell([2.0, 2.0, 2.0]))
        selection = assembly.insert(collection, translation=[0.5, 1.0, 1.0])
        self.assertEqual(selection.name, "scan")
        self.assertEqual(selection.fiber_ids, [1])
        self.assertEqual(assembly.fiber_count, 1)
        self.assertEqual(assembly.centerlines()[0][1], [1.5, 1.0, 1.0])
        with self.assertRaises(ValueError):
            assembly.insert(tangle.FiberCollection("empty"))

    def test_neighbor_analysis_counts_crossing_contacts(self):
        radius = 0.05
        material = tangle.Material("fiber", diameter=2 * radius)
        collection = tangle.FiberCollection.from_centerlines(
            [
                [[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]],
                [[1.0, 0.5, 1.0 + 2 * radius], [1.0, 1.5, 1.0 + 2 * radius]],
            ],
            material,
        )
        assembly = tangle.Assembly(tangle.Cell([2.0, 2.0, 2.0]))
        assembly.insert(collection)

        report = assembly.characterize_neighbors(0.01, sample_spacing=0.002)
        self.assertEqual(report.contact_count, 2)
        self.assertEqual(report.in_axis_contact_fraction, 0.0)
        self.assertAlmostEqual(report.median_crossing_angle_degrees, 90.0)
        self.assertAlmostEqual(report.median_excess_persistence, 1.0, delta=0.1)
        self.assertEqual(len(report.crossing_angles_degrees), 2)
        self.assertEqual(report.to_dict()["schema_version"], 1)
        self.assertIn('"contacts": 2', report.to_json())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "neighbors.json"
            report.write_json(path)
            self.assertTrue(path.is_file())
        with self.assertRaises(ValueError):
            assembly.characterize_neighbors(0.1, neighbor_gap=0.01)


class SettingsTests(unittest.TestCase):
    def test_rust_defaults_are_visible_and_editable(self):
        settings = tangle.RelaxationSettings()
        self.assertEqual(settings.backend, "wgpu")
        self.assertEqual(settings.motion_model, "flexible")
        self.assertEqual(settings.max_iterations, 2_000)
        self.assertIsNone(settings.adaptive_segmentation)
        settings.enable_adaptive_segmentation()
        self.assertEqual(settings.adaptive_segmentation.refinement_interval, 8)
        settings.adaptive_segmentation = tangle.AdaptiveSegmentationSettings.profile("fast")
        self.assertEqual(settings.adaptive_segmentation.refinement_interval, 16)

    def test_neighbor_list_settings_round_trip(self):
        cell_list = tangle.CellListSettings(neighbor_skin_scale=0.5, neighbor_capacity=16)
        self.assertEqual(cell_list.to_dict()["neighbor_capacity"], 16)
        settings = tangle.RelaxationSettings()
        self.assertEqual(settings.cell_list.neighbor_skin_scale, 2.0)
        settings.cell_list = cell_list
        self.assertEqual(settings.cell_list.neighbor_skin_scale, 0.5)
        self.assertEqual(settings.cell_list.neighbor_capacity, 16)

    def test_cpu_backend_is_selectable(self):
        settings = tangle.RelaxationSettings()
        settings.backend = "cpu"
        self.assertEqual(settings.to_dict()["backend"], "cpu")

    def test_advanced_recipe_settings_are_editable(self):
        compaction = tangle.CompactionSettings.volume_fraction(
            0.4, axis_weights=[0.0, 0.0, 1.0]
        )
        compaction.balance_opposing_faces = True
        compaction.maximum_steps = 250
        self.assertEqual(compaction.target_value, 0.4)
        self.assertTrue(compaction.balance_opposing_faces)

        policy = tangle.SolvePolicy("contact first")
        policy.curvature_enforcement = "soft"
        overrides = tangle.RelaxationOverrides()
        overrides.bend_stiffness = 0.0
        junctions = tangle.JunctionPolicy("spray bond", "bond")
        junctions.material_pairs = [("small", "large")]
        self.assertEqual(policy.curvature_enforcement, "soft")
        self.assertEqual(overrides.bend_stiffness, 0.0)
        self.assertEqual(junctions.material_pairs, [("small", "large")])

    def test_checkpoint_settings_are_editable(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tangle.CheckpointSettings(
                "python-case",
                Path(directory) / "run.restart",
                interval_iterations=25,
            )
            checkpoint.resume = True
            checkpoint.fresh_formation_on_resume = True
            self.assertEqual(checkpoint.interval_iterations, 25)
            self.assertTrue(checkpoint.resume)
            self.assertTrue(checkpoint.fresh_formation_on_resume)


if __name__ == "__main__":
    unittest.main()
