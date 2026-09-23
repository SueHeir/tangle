import ast
import math
import random
import tempfile
import unittest
from pathlib import Path

import tangle
from tangle.units import um


API_STUB = Path(__file__).parents[1] / "tangle" / "_tangle.pyi"


def two_crossing_fibers():
    material = tangle.Material("fiber", diameter=0.1, min_bend_radius=0.5)
    return tangle.FiberCollection.from_centerlines(
        [
            [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]],
            [[0.5, 0.2, 0.5], [0.5, 0.8, 0.5]],
        ],
        material,
    )


class CollectionTests(unittest.TestCase):
    def test_collection_preserves_placed_and_rest_centerlines(self):
        material = tangle.Material("fiber", diameter=7 * um, min_bend_radius=35 * um)
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
        self.assertAlmostEqual(material.min_bend_radius, 35 * um)

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

        population = tangle.FiberPopulation(
            count=6,
            segments_per_fiber=2,
            position=tangle.LayeredPosition(2, jitter_fraction=0.1),
        )
        generated = tangle.generate_fiber_population(cell, population)
        self.assertEqual(len(generated), 6)
        self.assertEqual(generated.layer_ids(), [0, 1])
        first = generated.select_layer(0)
        second = generated.select_layer(1)
        self.assertEqual(len(first + second), 6)
        first.extend(second)
        self.assertEqual(len(first), 6)

    def test_generators_take_a_material(self):
        cell = tangle.Cell([1.0, 1.0, 1.0])
        material = tangle.Material("thin", diameter=0.02, min_bend_radius=0.2)
        crossing = tangle.generate_fiber_pair_crossing(cell, material=material)
        self.assertEqual(len(crossing), 2)
        population = tangle.FiberPopulation(material=material, count=3)
        self.assertEqual(population.material.name, "thin")
        self.assertIsNone(population.diameter)
        self.assertEqual(len(tangle.generate_fiber_population(cell, population)), 3)

    def test_native_analysis_and_puma_bundle_share_the_rust_assembly(self):
        material = tangle.Material("fiber", diameter=0.2)
        collection = tangle.FiberCollection("one")
        collection.add_fiber([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]], material)
        assembly = tangle.Assembly(tangle.Cell([1.0, 1.0, 1.0]))
        tangle.Recipe(assembly).insert(collection)

        analysis = assembly.characterize()
        self.assertEqual(analysis.fiber_count, 1)
        self.assertEqual(analysis.segment_count, 1)
        self.assertAlmostEqual(analysis.length_weighted_orientation_tensor[0][0], 1.0)
        self.assertEqual(analysis.curvature_limit_violations, 0)
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
        graph = report.contact_graph()
        self.assertEqual(graph["edges"], 1)
        self.assertEqual(graph["degrees"], [1, 1])
        self.assertEqual(graph["components"], 1)
        self.assertEqual(report.to_dict()["schema_version"], 1)
        self.assertIn('"contacts": 2', report.to_json())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "neighbors.json"
            report.write_json(path)
            self.assertTrue(path.is_file())
        with self.assertRaises(ValueError):
            assembly.characterize_neighbors(0.1, neighbor_gap=0.01)

    def test_entanglement_measures_contact_linking(self):
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

        report = assembly.characterize_entanglement(0.01, neighbor_sample_spacing=0.002)
        self.assertIsInstance(report, tangle.EntanglementReport)
        self.assertAlmostEqual(report.sample_spacing, 2 * radius)
        self.assertAlmostEqual(report.window, 20 * 2 * radius)
        linking = report.absolute_contact_linking
        self.assertEqual(linking["count"], 1)
        self.assertGreater(linking["mean"], 0.2)
        self.assertLess(linking["mean"], 0.5)
        self.assertEqual(len(report.fiber_writhes), 2)
        self.assertTrue(all(abs(value) < 1e-9 for value in report.fiber_writhes))
        self.assertEqual(report.to_dict()["schema_version"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "entanglement.json"
            report.write_json(path)
            self.assertTrue(path.is_file())

    def test_slice_analysis_measures_a_square_lattice(self):
        material = tangle.Material("fiber", diameter=0.1)
        collection = tangle.FiberCollection.from_centerlines(
            [
                [[i + 0.5, j + 0.5, 0.0], [i + 0.5, j + 0.5, 2.0]]
                for i in range(4)
                for j in range(4)
            ],
            material,
        )
        assembly = tangle.Assembly(tangle.Cell([4.0, 4.0, 2.0], periodic="xy"))
        assembly.insert(collection)

        report = assembly.characterize_slices(slice_count=4, bin_count=8)
        self.assertIsInstance(report, tangle.SliceReport)
        self.assertEqual(report.section_counts, [16, 16, 16, 16])
        self.assertAlmostEqual(report.sections_per_area, 1.0)
        nearest = report.nearest_neighbor_distance
        self.assertEqual(nearest["count"], 64)
        self.assertAlmostEqual(nearest["quantiles"][0], 1.0)
        self.assertAlmostEqual(nearest["quantiles"][-1], 1.0)
        self.assertGreater(report.clark_evans_ratio, 1.9)
        self.assertAlmostEqual(report.max_radius, 2.0)
        self.assertEqual(len(report.pair_correlation), 8)
        self.assertEqual(report.pair_correlation[0], 0.0)
        self.assertEqual(report.to_dict()["schema_version"], 1)
        with self.assertRaises(ValueError):
            assembly.characterize_slices(axis=3)

    def test_shape_analysis_measures_a_helix(self):
        a, c = 1.0, 0.5
        helix = [
            [5.0 + a * math.cos(t), 5.0 + a * math.sin(t), 2.0 + c * t]
            for t in (2 * math.pi * i / 4000 for i in range(8001))
        ]
        material = tangle.Material("fiber", diameter=0.02)
        assembly = tangle.Assembly(tangle.Cell([10.0, 10.0, 10.0]))
        assembly.insert(tangle.FiberCollection.from_centerlines([helix], material))

        report = assembly.characterize_shape(sample_spacing=0.05, quantile_count=11)
        self.assertEqual(report.fiber_count, 1)
        self.assertEqual(len(report.curvature["quantiles"]), 11)
        self.assertAlmostEqual(report.curvature["mean"], a / (a * a + c * c), delta=0.01)
        self.assertAlmostEqual(report.torsion["mean"], c / (a * a + c * c), delta=0.01)
        self.assertEqual(report.orientation_axis, [0.0, 0.0, 1.0])
        self.assertIsNotNone(report.schladitz_beta)
        self.assertEqual(report.to_dict()["schema_version"], 1)
        self.assertEqual(len(report.fiber_curl_indices), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "shape.json"
            report.write_json(path)
            self.assertTrue(path.is_file())
        with self.assertRaises(ValueError):
            assembly.characterize_shape(orientation_axis=[0.0, 0.0, 0.0])

    def test_scorecard_flags_a_wavier_structure(self):
        def slab(amplitude, seed):
            rng = random.Random(seed)
            lines = []
            for _ in range(40):
                x0, y0, z = 0.2 + 3.6 * rng.random(), 0.2 + 3.6 * rng.random(), 0.5
                angle, phase = 2 * math.pi * rng.random(), 2 * math.pi * rng.random()
                u, v = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
                line = []
                for i in range(61):
                    s = 1.2 * i / 60
                    w = amplitude * math.sin(2 * math.pi * s / 0.4 + phase)
                    line.append([x0 + s * u[0] + w * v[0], y0 + s * u[1] + w * v[1], z])
                lines.append(line)
            assembly = tangle.Assembly(tangle.Cell([4.0, 4.0, 1.0]))
            material = tangle.Material("fiber", diameter=0.02)
            assembly.insert(tangle.FiberCollection.from_centerlines(lines, material))
            return assembly

        reference = slab(0.02, 1)
        card = tangle.score_structure(slab(0.08, 2), reference, 0.005, subdivisions=[2, 2, 1])
        self.assertEqual(card.reference_subvolume_count, 4)
        self.assertEqual(card.subvolume_size, [2.0, 2.0, 1.0])
        self.assertIn("curvature", card.scores)
        self.assertGreater(card.scores["curvature"], 2.0)
        self.assertIn("curvature", card.table())
        self.assertEqual(len(card.rows), len(card.scores))
        self.assertEqual(card.to_dict()["schema_version"], 1)
        # The same assembly on both sides must not deadlock.
        tangle.score_structure(reference, reference, 0.005, subdivisions=[2, 2, 1])
        with self.assertRaises(ValueError):
            tangle.score_structure(reference, reference, 0.005, subdivisions=[1, 1, 1])


class CellTests(unittest.TestCase):
    def test_stack_axis_is_the_single_bounded_axis(self):
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0]).stack_axis, 2)
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], periodic="xy").stack_axis, 2)
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], periodic="yz").stack_axis, 0)
        self.assertEqual(
            tangle.Cell([1.0, 1.0, 1.0], periodic=[True, False, True]).stack_axis, 1
        )
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], stack_axis="y").stack_axis, 1)
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], periodic="xy").periodic, [True, True, False])

    def test_recipe_inherits_or_overrides_the_stack_axis(self):
        cell = tangle.Cell([1.0, 1.0, 1.0], periodic="yz")
        self.assertEqual(tangle.Recipe(cell).stack_axis, 0)
        self.assertEqual(tangle.Recipe(cell, stack_axis="z").stack_axis, 2)

    def test_axis_edge_cases(self):
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], periodic=True).periodic, [True] * 3)
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], periodic=False).periodic, [False] * 3)
        # z is periodic here, so the last bounded axis (y) is the stack axis.
        self.assertEqual(tangle.Cell([1.0, 1.0, 1.0], periodic="z").stack_axis, 1)
        with self.assertRaises(TypeError):
            tangle.Cell([1.0, 1.0, 1.0], stack_axis=True)
        with self.assertRaises(AttributeError):
            tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0])).stack_axis = 0

    def test_bad_axes_are_rejected(self):
        with self.assertRaises(ValueError):
            tangle.Cell([1.0, 1.0, 1.0], periodic="xw")
        with self.assertRaises(ValueError):
            tangle.LayeredPosition(2, axis=3)


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
        settings = tangle.RelaxationSettings()
        self.assertEqual(settings.neighbor_skin_scale, 2.0)
        tuned = settings.replace(neighbor_skin_scale=0.5, neighbor_capacity=16)
        self.assertEqual(tuned.to_dict()["neighbor_capacity"], 16)
        self.assertEqual(tuned.neighbor_skin_scale, 0.5)
        with self.assertRaises(ValueError):
            settings.replace(neighbor_capacity=0)

    def test_keyword_constructors_and_replace(self):
        settings = tangle.RelaxationSettings(backend="cpu", max_iterations=50)
        self.assertEqual(settings.to_dict()["backend"], "cpu")
        changed = settings.replace(max_iterations=10, cell_size_scale=2.0)
        self.assertEqual(settings.max_iterations, 50)
        self.assertEqual(changed.max_iterations, 10)
        self.assertEqual(changed.cell_size_scale, 2.0)
        self.assertEqual(changed.backend, "cpu")

        profile = tangle.AdaptiveSegmentationSettings.profile("strict", max_refinement_levels=2)
        self.assertEqual(profile.max_refinement_levels, 2)

    def test_typos_fail_loudly(self):
        with self.assertRaisesRegex(TypeError, "backnd"):
            tangle.RelaxationSettings(backnd="cpu")
        with self.assertRaisesRegex(TypeError, "max_iteration"):
            tangle.SolvePolicy("p").replace(max_iteration=5)
        settings = tangle.RelaxationSettings()
        with self.assertRaisesRegex(ValueError, "wgpu"):
            settings.backend = "gpu"
        with self.assertRaises(ValueError):
            settings.contact_aggregation = "deepest"
        with self.assertRaises(ValueError):
            tangle.SolvePolicy("p", on_budget_exhausted="ignore")

    def test_values_the_solver_would_reject_fail_early(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            tangle.SolvePolicy("p", target_curvature_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "shrink_factor"):
            tangle.CompactionSettings.volume_fraction(0.1, shrink_factor=1.5)
        with self.assertRaisesRegex(ValueError, "growth_factor"):
            tangle.CompactionSettings.volume_fraction(0.1, growth_factor=0.5)
        with self.assertRaises(ValueError):
            tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0])).place_layer_above(1, gap=1e-50)
        with self.assertRaisesRegex(TypeError, "copy"):
            tangle.RelaxationSettings(copy=1)

    def test_solve_policy_limits_default_to_targets(self):
        policy = tangle.SolvePolicy("contact first", target_penetration=0.05)
        self.assertEqual(policy.max_penetration, 0.05)
        loose = policy.replace(max_penetration=0.1)
        self.assertEqual(loose.target_penetration, 0.05)
        self.assertEqual(loose.max_penetration, 0.1)

    def test_override_presets(self):
        cleanup = tangle.RelaxationOverrides.preset("curvature_cleanup")
        self.assertEqual(cleanup.contact_aggregation, "deepest_only")
        self.assertEqual(cleanup.curvature_cleanup_sweeps, 16)
        self.assertEqual(cleanup.bend_stiffness, 0.0)
        tweaked = tangle.RelaxationOverrides.preset("contact_first", constraint_iterations=4)
        self.assertEqual(tweaked.constraint_iterations, 4)
        self.assertIsNone(tangle.RelaxationOverrides().motion_model)
        self.assertNotIn("None", repr(tangle.RelaxationOverrides(bend_stiffness=0.0)))

    def test_compaction_targets_and_paths_are_typed(self):
        compaction = tangle.CompactionSettings.volume_fraction(
            0.4, path=tangle.AxisWeightsPath("z"), max_steps=250
        )
        self.assertIsInstance(compaction.target, tangle.VolumeFractionTarget)
        self.assertEqual(compaction.target.value, 0.4)
        self.assertEqual(compaction.path.weights, [0.0, 0.0, 1.0])
        self.assertEqual(compaction.max_steps, 250)
        compaction.balance_opposing_faces = True
        self.assertTrue(compaction.balance_opposing_faces)

        pressure = tangle.CompactionSettings(
            tangle.MeanPressureTarget(1.0e3),
            path=tangle.EqualPressurePath("xy"),
        )
        self.assertEqual(pressure.path.axes, [True, True, False])
        with self.assertRaises(ValueError):
            tangle.VolumeFractionTarget(1.5)

    def test_junction_policy_is_editable(self):
        junctions = tangle.JunctionPolicy(
            "spray bond", "bond", material_pairs=[("small", "large")]
        )
        self.assertEqual(junctions.material_pairs, [("small", "large")])
        self.assertEqual(junctions.replace(probability=0.5).probability, 0.5)
        with self.assertRaises(ValueError):
            junctions.replace(probability=2.0)

    def test_checkpoint_settings_are_editable(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tangle.CheckpointSettings(
                "python-case",
                Path(directory) / "run.restart",
                interval_iterations=25,
            )
            resumed = checkpoint.replace(resume=True, fresh_formation_on_resume=True)
            self.assertEqual(resumed.interval_iterations, 25)
            self.assertTrue(resumed.resume)
            self.assertFalse(checkpoint.resume)

    def test_population_variants_are_typed(self):
        population = tangle.FiberPopulation(
            orientation=tangle.AlignedOrientation("x", max_angle=0.1),
            position=tangle.DensityGradientPosition(exponent=2.0),
            length=(0.2, 0.4),
        )
        self.assertEqual(population.orientation.max_angle, 0.1)
        self.assertEqual(population.length, (0.2, 0.4))
        self.assertEqual(population.replace(length=[0.1, 0.3]).length, (0.1, 0.3))
        planar = population.replace(orientation=tangle.PlanarOrientation(max_tilt=0.2))
        self.assertIsInstance(planar.orientation, tangle.PlanarOrientation)
        self.assertIsInstance(population.orientation, tangle.AlignedOrientation)
        with self.assertRaises(TypeError):
            population.orientation = "planar"
        with self.assertRaisesRegex(ValueError, "LayeredPosition"):
            tangle.FiberPopulation(orientation=tangle.LayeredBiaxialOrientation())


class RecipeTests(unittest.TestCase):
    def test_held_targets_release_when_the_block_ends(self):
        recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
        recipe.insert(two_crossing_fibers())
        footprint = tangle.CircularFootprint.random(diameter=0.2, seed=1)
        with recipe.needle_layer(0, footprint=footprint, depth=0.5) as held:
            self.assertIs(held, recipe)
            held.relax_for(1)
        operations = recipe.operations()
        self.assertIn("needle layer 0", operations[-3])
        self.assertEqual(operations[-1], "release needles")

    def test_bend_radius_names_are_checked_at_run(self):
        recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
        recipe.insert(two_crossing_fibers())
        # A material inserted later in the recipe is fine.
        recipe.set_min_bend_radius("late", 1.0)
        late = tangle.Material("late", diameter=0.05)
        recipe.insert(
            tangle.FiberCollection.from_centerlines([[[0.2, 0.2, 0.2], [0.8, 0.2, 0.2]]], late)
        )
        recipe.set_min_bend_radius("missing", 1.0)
        recipe.relax_for(1)
        with self.assertRaisesRegex(ValueError, "missing"):
            recipe.run(tangle.RelaxationSettings(backend="cpu"))

    def test_failed_operation_raises_recipe_error(self):
        recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
        recipe.insert(two_crossing_fibers())
        recipe.solve(
            tangle.SolvePolicy(
                "strict",
                target_penetration=1e-9,
                max_iterations=1,
                on_budget_exhausted="fail",
            )
        )
        with self.assertRaises(tangle.RecipeError) as caught:
            recipe.run(tangle.RelaxationSettings(backend="cpu"))
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertEqual(caught.exception.operation_index, 1)
        self.assertIn("strict", caught.exception.operation)
        self.assertIn("penetration", caught.exception.reason)

    def test_a_result_assembly_continues_in_a_new_recipe(self):
        recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
        fibers = two_crossing_fibers()
        recipe.insert(fibers)
        extra = tangle.FiberCollection.from_centerlines(
            [[[0.2, 0.2, 0.8], [0.8, 0.2, 0.8]]], tangle.Material("fiber", diameter=0.1)
        )
        recipe.insert(extra)
        recipe.relax_for(2)
        first = recipe.run(tangle.RelaxationSettings(backend="cpu"))
        self.assertEqual(first.assembly.fiber_count, 3)

        follow_on = tangle.Recipe(first.assembly)
        follow_on.insert(
            tangle.FiberCollection.from_centerlines(
                [[[0.2, 0.8, 0.2], [0.8, 0.8, 0.2]]], tangle.Material("fiber", diameter=0.1)
            )
        )
        follow_on.relax_for(2)
        second = follow_on.run(tangle.RelaxationSettings(backend="cpu"))
        self.assertEqual(second.fiber_count, 4)

    def test_result_assembly_is_the_same_object_state(self):
        recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
        recipe.insert(two_crossing_fibers())
        recipe.relax_for(1)
        result = recipe.run(tangle.RelaxationSettings(backend="cpu"))
        result.assembly.insert(
            tangle.FiberCollection.from_centerlines(
                [[[0.2, 0.8, 0.8], [0.8, 0.8, 0.8]]], tangle.Material("fiber", diameter=0.1)
            )
        )
        self.assertEqual(result.assembly.fiber_count, 3)
        self.assertEqual(result.fiber_count, 3)

    def test_run_returns_a_new_assembly(self):
        assembly = tangle.Assembly(tangle.Cell([1.0, 1.0, 1.0]))
        recipe = tangle.Recipe(assembly)
        recipe.insert(two_crossing_fibers())
        recipe.relax_for(2)
        before = assembly.centerlines()
        result = recipe.run(tangle.RelaxationSettings(backend="cpu"))
        self.assertEqual(result.assembly.fiber_count, 2)
        self.assertEqual(assembly.centerlines(), before)
        self.assertEqual(result.assembly.centerlines(), result.centerlines())


class StubTests(unittest.TestCase):
    def test_stub_matches_the_native_module(self):
        native = tangle._tangle
        tree = ast.parse(API_STUB.read_text())
        declared = set()
        missing = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                declared.add(node.name)
                if not hasattr(native, node.name):
                    missing.append(node.name)
            if not isinstance(node, ast.ClassDef):
                continue
            declared.add(node.name)
            runtime = getattr(native, node.name, None)
            if runtime is None:
                missing.append(node.name)
                continue
            if issubclass(runtime, BaseException):
                continue  # Exception attributes are set per instance.
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    name = item.name
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                else:
                    continue
                if not name.startswith("__") and not hasattr(runtime, name):
                    missing.append(f"{node.name}.{name}")
        self.assertEqual(missing, [])
        undeclared = {
            name for name in dir(native) if not name.startswith("_")
        } - declared
        self.assertEqual(undeclared, set())
        self.assertEqual(set(tangle.__all__) - {"units", "__version__"}, declared)


if __name__ == "__main__":
    unittest.main()
