"""Keep the Python-first entry points and notebook examples usable."""

import contextlib
import io
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[4]
PYTHON = ROOT / "crates" / "tangle_python" / "python"


class DocumentationTests(unittest.TestCase):
    def test_markdown_local_links_exist(self):
        documents = [ROOT / "README.md", *ROOT.glob("docs/*.md")]
        documents += list((ROOT / "examples").glob("*/README.md"))
        documents += [ROOT / "examples" / "README.md"]
        documents += list((ROOT / "crates").glob("*/README.md"))
        documents += list(PYTHON.glob("**/README.md"))
        for document in documents:
            # Fenced snippets can contain illustrative, non-navigation paths.
            prose = re.sub(r"```.*?```", "", document.read_text(), flags=re.S)
            for target in re.findall(r"\]\(([^\s)]+)\)", prose):
                parsed = urlsplit(target.strip("<>"))
                if parsed.scheme or not parsed.path:
                    continue
                with self.subTest(document=document.relative_to(ROOT), link=target):
                    self.assertTrue((document.parent / unquote(parsed.path)).exists())

    def test_topic_notebooks_execute_with_default_guards(self):
        # Installation is intentionally not executed by tests: it changes the
        # environment. All remaining tutorials default to configuration only.
        for path in sorted((PYTHON / "tutorials").glob("*.ipynb")):
            if path.name.startswith("00_"):
                continue
            scope = {"__name__": "__documentation_test__"}
            notebook = json.loads(path.read_text())
            with self.subTest(notebook=path.name), contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        source = cell["source"]
                        if isinstance(source, list):
                            source = "".join(source)
                        exec(compile(source, f"{path.name}:cell-{index}", "exec"), scope)

    def test_workflow_notebooks_are_clean_and_syntactically_valid(self):
        for path in sorted((PYTHON / "examples").rglob("*.ipynb")):
            if "output" in path.parts or ".ipynb_checkpoints" in path.parts:
                continue
            with self.subTest(notebook=path.relative_to(PYTHON)):
                for index, cell in enumerate(json.loads(path.read_text())["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = cell["source"]
                    if isinstance(source, list):
                        source = "".join(source)
                    compile(source, f"{path.name}:cell-{index}", "exec")
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
