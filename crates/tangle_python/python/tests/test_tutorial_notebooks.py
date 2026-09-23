import ast
import json
import unittest
from pathlib import Path


TUTORIALS = Path(__file__).parents[1] / "tutorials"
API_STUB = Path(__file__).parents[1] / "tangle" / "_tangle.pyi"
EXPECTED = {
    "00_installation_and_environment.ipynb",
    "01_high_level_overview.ipynb",
    "02_cells_and_boundaries.ipynb",
    "03_materials_and_centerlines.ipynb",
    "04_fiber_collections.ipynb",
    "05_fiber_generation.ipynb",
    "06_insertion_and_transforms.ipynb",
    "07_relaxation_settings.ipynb",
    "08_adaptive_refinement.ipynb",
    "09_solve_policies_and_overrides.ipynb",
    "10_layer_motion_and_needling.ipynb",
    "11_compaction.ipynb",
    "12_junction_capture.ipynb",
    "13_checkpoints_and_resume.ipynb",
    "14_results_and_exports.ipynb",
}


class TutorialNotebookTests(unittest.TestCase):
    def test_complete_tutorial_set_is_present_and_valid(self) -> None:
        actual = {path.name for path in TUTORIALS.glob("*.ipynb")}
        self.assertEqual(actual, EXPECTED)

        for path in sorted(TUTORIALS.glob("*.ipynb")):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text())
                self.assertEqual(notebook["nbformat"], 4)
                self.assertGreaterEqual(len(notebook["cells"]), 3)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        compile(cell["source"], f"{path.name}:cell-{index}", "exec")
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])
                        self.assertIn(
                            "#",
                            cell["source"],
                            f"{path.name}:cell-{index} needs an explanatory comment",
                        )

    def test_every_configuration_field_and_recipe_operation_is_documented(self) -> None:
        tree = ast.parse(API_STUB.read_text())
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        notebook_for_class = {
            "FiberPopulation": "05_fiber_generation.ipynb",
            "RelaxationSettings": "07_relaxation_settings.ipynb",
            "AdaptiveSegmentationSettings": "08_adaptive_refinement.ipynb",
            "RelaxationOverrides": "09_solve_policies_and_overrides.ipynb",
            "SolvePolicy": "09_solve_policies_and_overrides.ipynb",
            "CompactionSettings": "11_compaction.ipynb",
            "JunctionPolicy": "12_junction_capture.ipynb",
            "CheckpointSettings": "13_checkpoints_and_resume.ipynb",
        }

        for class_name, filename in notebook_for_class.items():
            notebook = json.loads((TUTORIALS / filename).read_text())
            documented = "\n".join(cell["source"] for cell in notebook["cells"])
            fields = {
                statement.target.id
                for statement in classes[class_name].body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
            with self.subTest(configuration=class_name):
                self.assertEqual(
                    {field for field in fields if field not in documented},
                    set(),
                )

        all_tutorial_text = "\n".join(
            cell["source"]
            for path in sorted(TUTORIALS.glob("*.ipynb"))
            for cell in json.loads(path.read_text())["cells"]
        )
        recipe_methods = {
            statement.name
            for statement in classes["Recipe"].body
            if isinstance(statement, ast.FunctionDef)
            and not statement.name.startswith("_")
        }
        self.assertEqual(
            {method for method in recipe_methods if method not in all_tutorial_text},
            set(),
        )


if __name__ == "__main__":
    unittest.main()
