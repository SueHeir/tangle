"""tangle.fem: hexahedral and tetrahedral fiber meshes and their Nastran and
Abaqus files."""

import math
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import tangle

try:
    import numpy as np
    import tangle.fem as fem
except ImportError:  # tangle.fem needs NumPy
    fem = None

try:
    import gmsh  # noqa: F401
except ImportError:  # tet_mesh needs gmsh
    gmsh = None


def assembly_of(lengths, centerlines, material, periodic=None):
    """An Assembly holding ``centerlines`` of one material."""
    assembly = tangle.Assembly(tangle.Cell(lengths, periodic))
    assembly.insert(tangle.FiberCollection.from_centerlines(centerlines, material))
    return assembly


def one_fiber():
    """The ``export_puma`` test fiber: diameter 0.2 along x in a unit cell."""
    return assembly_of([1.0, 1.0, 1.0], [[[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]]], tangle.Material("fiber", diameter=0.2))


def touching_cross():
    """Two round fibers of radius 0.1 crossing at right angles and touching."""
    return assembly_of(
        [1.0, 1.0, 1.0],
        [[[0.2, 0.5, 0.4], [0.8, 0.5, 0.4]], [[0.5, 0.2, 0.6], [0.5, 0.8, 0.6]]],
        tangle.Material("fiber", diameter=0.2),
    )


def read_nastran(path):
    """GRID*, CHEXA/CTETRA, PSOLID and MAT1* entries of a written deck."""
    grids, elements, properties, materials = {}, {}, {}, {}
    lines = Path(path).read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        name = line[:8].strip()
        if line.startswith("$"):
            index += 1
        elif name == "GRID*":
            grids[int(line[8:24])] = (float(line[40:56]), float(line[56:72]), float(lines[index + 1][8:24]))
            index += 2
        elif name in ("CHEXA", "CTETRA"):
            values = [int(line[start : start + 8]) for start in range(8, len(line), 8)]
            index += 1
            while index < len(lines) and lines[index].startswith(" " * 8):
                values += [int(lines[index][start : start + 8]) for start in range(8, len(lines[index]), 8)]
                index += 1
            elements[values[0]] = (name, values[1], values[2:])
        elif name == "PSOLID":
            properties[int(line[8:16])] = int(line[16:24])
            index += 1
        elif name == "MAT1*":
            density = None
            if index + 1 < len(lines) and lines[index + 1].startswith("*"):
                density = float(lines[index + 1][8:24])
            materials[int(line[8:24])] = (float(line[24:40]), float(line[56:72]), density)
            index += 2 if density is not None else 1
        else:
            raise AssertionError(f"unexpected line {line!r}")
    return grids, elements, properties, materials


def read_abaqus(path):
    """Keyword blocks of a written input file as ``(keyword line, data rows)``."""
    blocks = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("**"):
            continue
        if line.startswith("*"):
            blocks.append((line, []))
        else:
            blocks[-1][1].append([item.strip() for item in line.split(",")])
    return blocks


def read_vtu(path):
    """The ``Piece`` element of a written ``.vtu`` and its appended arrays, by name."""
    data = Path(path).read_bytes()
    start = data.index(b'<AppendedData encoding="raw">')
    base = data.index(b"_", start) + 1  # the arrays' offsets count from just after the "_"
    root = ElementTree.fromstring(data[:start].decode("ascii") + "</VTKFile>")
    dtypes = {"Float64": "<f8", "Int64": "<i8", "Int32": "<i4", "UInt8": "u1"}
    arrays = {}
    for array in root.iter("DataArray"):
        at = base + int(array.get("offset"))
        size = int(np.frombuffer(data[at : at + 8], dtype="<u8")[0])
        arrays[array.get("Name")] = np.frombuffer(data[at + 8 : at + 8 + size], dtype=dtypes[array.get("type")])
    return root.find("UnstructuredGrid/Piece"), arrays


@unittest.skipIf(fem is None, "tangle.fem needs NumPy")
class HexMeshTests(unittest.TestCase):
    def test_one_brick_per_voxel_of_the_puma_grid(self):
        assembly = one_fiber()
        mesh = fem.hex_mesh(assembly, 0.1)
        with tempfile.TemporaryDirectory() as directory:
            report = assembly.export_puma(Path(directory) / "case.puma", 0.1)
        self.assertEqual(mesh.element_type, "hex8")
        self.assertEqual(mesh.element_count, report.occupied_voxels)
        np.testing.assert_allclose(mesh.element_volumes(), 0.1**3)
        np.testing.assert_allclose(mesh.nodes / 0.1, np.round(mesh.nodes / 0.1), atol=1e-9)
        self.assertEqual(mesh.fiber_ids, [1])
        self.assertEqual(mesh.materials, ("fiber",))
        self.assertEqual(len(np.unique(mesh.elements)), mesh.node_count)

    def test_touching_fibers_share_nodes(self):
        mesh = fem.hex_mesh(touching_cross(), 0.05)
        self.assertEqual(mesh.fiber_ids, [1, 2])
        first = set(mesh.elements[mesh.element_fibers == 1].ravel().tolist())
        second = set(mesh.elements[mesh.element_fibers == 2].ravel().tolist())
        self.assertTrue(first & second)

    def test_a_voxel_size_that_does_not_tile_suggests_one(self):
        with self.assertRaisesRegex(ValueError, "try voxel_size="):
            fem.hex_mesh(one_fiber(), 0.3)

    def test_a_binder_mask_adds_binder_bricks_that_bond_the_fibers(self):
        assembly = touching_cross()
        fibers = fem.hex_mesh(assembly, 0.05)  # a 20 x 20 x 20 voxel grid
        binder = np.zeros((20, 20, 20), dtype=bool)  # (z, y, x)
        binder[9:11, 6:14, 8:12] = True  # a box around the crossing, partly inside both fibers
        mesh = fem.hex_mesh(assembly, 0.05, binder=binder)

        def voxels(mesh, chosen):
            """(z, y, x) of the chosen elements' voxels: node 0 is a brick's low corner."""
            low = np.round((mesh.nodes[mesh.elements[chosen, 0]] - mesh.cell_origin) / 0.05).astype(int)
            return {(z, y, x) for x, y, z in low.tolist()}

        filled = voxels(fibers, slice(None))
        expected = {voxel for voxel in zip(*np.nonzero(binder)) if voxel not in filled}  # the fibers win
        is_binder = mesh.element_fibers == 0
        self.assertTrue(expected and len(expected) < binder.sum())
        self.assertEqual(voxels(mesh, is_binder), {tuple(int(i) for i in voxel) for voxel in expected})
        self.assertEqual(mesh.element_count, fibers.element_count + len(expected))
        self.assertEqual(mesh.materials, ("fiber", "binder"))
        self.assertTrue((mesh.element_materials[is_binder] == 1).all())
        binder_nodes = set(mesh.elements[is_binder].ravel().tolist())
        for fiber in (1, 2):
            self.assertEqual((mesh.element_fibers == fiber).sum(), (fibers.element_fibers == fiber).sum())
            self.assertTrue(binder_nodes & set(mesh.elements[mesh.element_fibers == fiber].ravel().tolist()))

    def test_a_binder_mask_must_match_the_grid(self):
        with self.assertRaisesRegex(ValueError, "binder must be a"):
            fem.hex_mesh(touching_cross(), 0.05, binder=np.zeros((20, 20, 19), dtype=bool))

    def test_a_run_result_works_like_its_assembly(self):
        class Result:
            assembly = one_fiber()

        self.assertEqual(fem.hex_mesh(Result(), 0.1).element_count, fem.hex_mesh(Result.assembly, 0.1).element_count)


@unittest.skipIf(fem is None, "tangle.fem needs NumPy")
class WriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.mesh = fem.hex_mesh(touching_cross(), 0.05)
        self.material = fem.ElasticMaterial(2.5e9, 0.35, 1380.0)

    def tearDown(self):
        self.directory.cleanup()

    def test_nastran_deck_reads_back(self):
        grids, elements, properties, materials = read_nastran(
            self.mesh.write_nastran(self.path / "mesh.bdf", self.material, scale=1e3)
        )
        self.assertEqual(sorted(grids), list(range(1, self.mesh.node_count + 1)))
        np.testing.assert_allclose([grids[i + 1] for i in range(self.mesh.node_count)], self.mesh.nodes * 1e3, rtol=1e-9)
        self.assertEqual(sorted(elements), list(range(1, self.mesh.element_count + 1)))
        for index, (name, pid, nodes) in elements.items():
            self.assertEqual(name, "CHEXA")
            self.assertEqual(pid, self.mesh.element_fibers[index - 1])
            self.assertEqual(nodes, (self.mesh.elements[index - 1] + 1).tolist())
        self.assertEqual(properties, {1: 1, 2: 1})
        self.assertEqual(materials, {1: (2.5e9, 0.35, 1380.0)})

    def test_nastran_leaves_out_a_material_it_was_not_given(self):
        path = self.mesh.write_nastran(self.path / "mesh.bdf")
        _, _, properties, materials = read_nastran(path)
        self.assertEqual(properties, {1: 1, 2: 1})
        self.assertEqual(materials, {})
        self.assertIn("MAT1 1 = fiber: not defined", path.read_text())

    def test_unknown_material_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no material named glass"):
            self.mesh.write_nastran(self.path / "mesh.bdf", {"glass": self.material})

    def test_abaqus_input_reads_back(self):
        blocks = read_abaqus(self.mesh.write_abaqus(self.path / "mesh.inp", {"fiber": self.material}))
        keywords = {line: rows for line, rows in blocks}
        nodes = np.array([[float(value) for value in row[1:]] for row in keywords["*NODE"]])
        np.testing.assert_allclose(nodes, self.mesh.nodes, rtol=1e-11)
        elements = {}
        for line, rows in blocks:
            if line.startswith("*ELEMENT"):
                self.assertTrue(line.startswith("*ELEMENT, TYPE=C3D8, ELSET=FIBER_"))
                fiber = int(line.rsplit("_", 1)[1])
                for row in rows:
                    elements[int(row[0])] = (fiber, [int(value) for value in row[1:]])
        self.assertEqual(sorted(elements), list(range(1, self.mesh.element_count + 1)))
        for index, (fiber, nodes) in elements.items():
            self.assertEqual(fiber, self.mesh.element_fibers[index - 1])
            self.assertEqual(nodes, (self.mesh.elements[index - 1] + 1).tolist())
        self.assertEqual(keywords["*ELSET, ELSET=MATERIAL_FIBER"], [["FIBER_1", "FIBER_2"]])
        self.assertIn("*SOLID SECTION, ELSET=MATERIAL_FIBER, MATERIAL=FIBER", keywords)
        self.assertEqual([float(value) for value in keywords["*ELASTIC"][0]], [2.5e9, 0.35])
        self.assertEqual([float(value) for value in keywords["*DENSITY"][0]], [1380.0])
        for name, indices in self.mesh.node_sets().items():
            rows = keywords.get(f"*NSET, NSET={name}", [])
            self.assertEqual([int(value) for row in rows for value in row], (indices + 1).tolist())

    def test_vtu_reads_back(self):
        piece, arrays = read_vtu(self.mesh.write_vtu(self.path / "mesh.vtu", scale=1e3))
        self.assertEqual(int(piece.get("NumberOfPoints")), self.mesh.node_count)
        self.assertEqual(int(piece.get("NumberOfCells")), self.mesh.element_count)
        np.testing.assert_allclose(arrays["Points"].reshape(-1, 3), self.mesh.nodes * 1e3, rtol=1e-12)
        np.testing.assert_array_equal(arrays["connectivity"].reshape(-1, 8), self.mesh.elements)
        np.testing.assert_array_equal(arrays["offsets"], 8 * np.arange(1, self.mesh.element_count + 1))
        np.testing.assert_array_equal(arrays["types"], 12)  # VTK_HEXAHEDRON
        np.testing.assert_array_equal(arrays["fiber_id"], self.mesh.element_fibers)
        np.testing.assert_array_equal(arrays["material"], self.mesh.element_materials)

    def test_material_values_must_be_physical(self):
        with self.assertRaises(ValueError):
            fem.ElasticMaterial(2.5e9, 0.5)
        with self.assertRaises(ValueError):
            fem.ElasticMaterial(-1.0, 0.3)


@unittest.skipIf(fem is None or gmsh is None, "tangle.fem.tet_mesh needs NumPy and gmsh")
class TetMeshTests(unittest.TestCase):
    def test_round_fiber_fills_its_cylinder(self):
        radius, length = 10e-6, 200e-6
        assembly = assembly_of(
            [length, 100e-6, 100e-6],
            [[[0.0, 50e-6, 50e-6], [length / 2, 50e-6, 50e-6], [length, 50e-6, 50e-6]]],
            tangle.Material("fiber", diameter=2 * radius),
            periodic=[True, False, False],
        )
        mesh = fem.tet_mesh(assembly)
        self.assertEqual(mesh.element_type, "tet4")
        self.assertTrue((mesh.element_volumes() > 0).all())
        self.assertAlmostEqual(mesh.volume / (math.pi * radius**2 * length), 1.0, delta=0.03)
        self.assertEqual(mesh.fiber_ids, [1])

    def test_oval_fiber_keeps_its_long_axis(self):
        # A fiber along x puts its long axis along y by default.
        assembly = assembly_of(
            [200e-6, 100e-6, 100e-6],
            [[[20e-6, 50e-6, 50e-6], [180e-6, 50e-6, 50e-6]]],
            tangle.Material("flat", diameter=30e-6, thickness=12e-6),
        )
        mesh = fem.tet_mesh(assembly)
        extent = mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)
        np.testing.assert_allclose(extent, [160e-6, 30e-6, 12e-6], rtol=0.01)
        self.assertAlmostEqual(mesh.volume / (math.pi * 15e-6 * 6e-6 * 160e-6), 1.0, delta=0.03)

    def test_ten_node_tetrahedra_put_mid_edge_nodes_in_nastran_order(self):
        mesh = fem.tet_mesh(touching_cross(), 0.1, order=2)
        self.assertEqual(mesh.element_type, "tet10")
        points = mesh.nodes[mesh.elements]
        for middle, (first, second) in zip(range(4, 10), [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]):
            np.testing.assert_allclose(points[:, middle], (points[:, first] + points[:, second]) / 2, atol=1e-12)
        self.assertTrue((mesh.element_volumes() > 0).all())

    def test_fibers_are_separate_bodies(self):
        mesh = fem.tet_mesh(touching_cross(), 0.1)
        self.assertEqual(mesh.fiber_ids, [1, 2])
        first = set(mesh.elements[mesh.element_fibers == 1].ravel().tolist())
        second = set(mesh.elements[mesh.element_fibers == 2].ravel().tolist())
        self.assertFalse(first & second)

    def test_fibers_are_cut_at_the_cell_walls_unless_asked_not_to(self):
        assembly = assembly_of(
            [100e-6, 100e-6, 100e-6],
            [[[-30e-6, 50e-6, 50e-6], [130e-6, 50e-6, 50e-6]]],
            tangle.Material("fiber", diameter=20e-6),
        )
        clipped = fem.tet_mesh(assembly)
        self.assertAlmostEqual(clipped.nodes[:, 0].min(), 0.0, delta=1e-12)
        self.assertAlmostEqual(clipped.nodes[:, 0].max(), 100e-6, delta=1e-12)
        self.assertGreater(len(clipped.node_sets()["XMIN"]), 0)
        whole = fem.tet_mesh(assembly, clip_to_cell=False)
        self.assertAlmostEqual(whole.nodes[:, 0].min(), -30e-6, delta=1e-12)
        self.assertAlmostEqual(whole.nodes[:, 0].max(), 130e-6, delta=1e-12)

    def test_ten_node_deck_reads_back(self):
        mesh = fem.tet_mesh(touching_cross(), 0.1, order=2)
        with tempfile.TemporaryDirectory() as directory:
            _, elements, _, _ = read_nastran(mesh.write_nastran(Path(directory) / "mesh.bdf"))
            blocks = read_abaqus(mesh.write_abaqus(Path(directory) / "mesh.inp"))
            _, arrays = read_vtu(mesh.write_vtu(Path(directory) / "mesh.vtu"))
        self.assertEqual(elements[1], ("CTETRA", mesh.element_fibers[0], (mesh.elements[0] + 1).tolist()))
        self.assertTrue(any(line.startswith("*ELEMENT, TYPE=C3D10") for line, _ in blocks))
        np.testing.assert_array_equal(arrays["types"], 24)  # VTK_QUADRATIC_TETRA, in the same node order
        np.testing.assert_array_equal(arrays["connectivity"].reshape(-1, 10), mesh.elements)


if __name__ == "__main__":
    unittest.main()
