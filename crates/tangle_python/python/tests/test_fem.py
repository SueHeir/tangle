"""tangle.fem: hexahedral and tetrahedral fiber meshes and their Nastran and
Abaqus files."""

import contextlib
import dataclasses
import io
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

import tangle

try:
    import numpy as np
    import tangle.fem as fem
    import tangle.fem._mesh as mesh_module
    import tangle.fem._sweep as sweep_module
    import tangle.fem._tet as tet_module
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


# The centerlines of touching_cross(): fiber 1 along x, fiber 2 along y.
CROSS_AXES = (((0.2, 0.5, 0.4), (0.8, 0.5, 0.4)), ((0.5, 0.2, 0.6), (0.5, 0.8, 0.6)))
# The corners of a tetrahedron's faces S1 to S4, as Abaqus numbers them.
ABAQUS_FACES = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]
# The ends of the edges whose middles are nodes 4 to 9 of a 10-node tetrahedron.
TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def touching_cross():
    """Two round fibers of radius 0.1 crossing at right angles and touching."""
    return assembly_of(
        [1.0, 1.0, 1.0],
        [[list(start), list(end)] for start, end in CROSS_AXES],
        tangle.Material("fiber", diameter=0.2),
    )


def crossing_binder():
    """Binder voxels for touching_cross() on its 20 x 20 x 20 grid of 0.05
    voxels: a box around the crossing, partly inside both fibers."""
    binder = np.zeros((20, 20, 20), dtype=bool)  # (z, y, x)
    binder[9:11, 6:14, 8:12] = True
    return binder


def ten_node_tetrahedron(corners, shape=lambda points: points):
    """The nodes of one 10-node tetrahedron with these corners and its
    mid-edge nodes in the middles of its edges, all moved by ``shape``."""
    corners = np.asarray(corners, dtype=float)
    middles = [(corners[first] + corners[second]) / 2 for first, second in TET10_EDGES]
    return shape(np.concatenate([corners, middles])), np.arange(10)[None, :]


def voxels_of(mesh, chosen, size):
    """(z, y, x) of the voxels of a hexahedral mesh's chosen bricks: node 0
    is a brick's low corner."""
    low = np.round((mesh.nodes[mesh.elements[chosen, 0]] - mesh.cell_origin) / size).astype(int)
    return {(z, y, x) for x, y, z in low.tolist()}


def face_nodes(mesh, faces):
    """The corner nodes of each ``(element, face)`` of a tetrahedral mesh."""
    return mesh.elements[faces[:, :1], np.array(ABAQUS_FACES)[faces[:, 1]]]


def axis_distances(points):
    """Each point's distance from each centerline of touching_cross()."""
    distances = []
    for start, end in CROSS_AXES:
        start, step = np.array(start), np.subtract(end, start)
        along = np.clip((points - start) @ step / (step @ step), 0.0, 1.0)
        distances.append(np.linalg.norm(points - (start + along[:, None] * step), axis=1))
    return distances


def read_nastran(path):
    """GRID*, CHEXA/CTETRA/CPENTA, PSOLID and MAT1* entries of a written
    deck; :func:`nastran_entries` reads the others."""
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
        elif name in ("CHEXA", "CTETRA", "CPENTA"):
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
        elif name in ("MPC", "CBAR", "RBE3"):
            index += 1
            while index < len(lines) and lines[index].startswith(" " * 8):
                index += 1
        elif name == "PBAR*":
            index += 2
        else:
            raise AssertionError(f"unexpected line {line!r}")
    return grids, elements, properties, materials


def nastran_entries(path, name):
    """The fields 2 to 9 of every small-field ``name`` entry of a written
    deck, with those of its continuations after them."""
    entries, current = [], None
    for line in Path(path).read_text().splitlines():
        fields = [line[start : start + 8].strip() for start in range(8, 72, 8)]
        if current is not None and line.startswith(" " * 8):
            current += fields
            continue
        current = None
        if line[:8].strip() == name:
            current = fields
            entries.append(current)
    return entries


def mpc_equations(path):
    """The MPC equations of a written deck, each as its terms
    ``(node, direction, coefficient)``, the dependent node's first. Only
    field 9 of a line and field 2 of a continuation are blank."""
    equations = []
    for fields in nastran_entries(path, "MPC"):
        assert fields[0] == "1"  # the set
        values = [field for field in fields[1:] if field]
        equations.append([(int(values[at]), int(values[at + 1]), float(values[at + 2]))
                          for at in range(0, len(values), 3)])
    return equations


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
        binder = crossing_binder()
        mesh = fem.hex_mesh(assembly, 0.05, binder=binder)
        filled = voxels_of(fibers, slice(None), 0.05)
        expected = {voxel for voxel in zip(*np.nonzero(binder)) if voxel not in filled}  # the fibers win
        is_binder = mesh.element_fibers == 0
        self.assertTrue(expected and len(expected) < binder.sum())
        self.assertEqual(voxels_of(mesh, is_binder, 0.05), {tuple(int(i) for i in voxel) for voxel in expected})
        self.assertEqual(mesh.element_count, fibers.element_count + len(expected))
        self.assertEqual(mesh.materials, ("fiber", "binder"))
        self.assertTrue((mesh.element_materials[is_binder] == 1).all())
        binder_nodes = set(mesh.elements[is_binder].ravel().tolist())
        for fiber in (1, 2):
            self.assertEqual((mesh.element_fibers == fiber).sum(), (fibers.element_fibers == fiber).sum())
            self.assertTrue(binder_nodes & set(mesh.elements[mesh.element_fibers == fiber].ravel().tolist()))

    def test_a_binder_mask_must_match_the_grid(self):
        for shape in ((20, 20, 19), (40, 40, 60), (30, 30, 30)):
            with self.assertRaisesRegex(ValueError, "binder must be a"):
                fem.hex_mesh(touching_cross(), 0.05, binder=np.zeros(shape, dtype=bool))

    def test_a_binder_mask_twice_as_fine_comes_to_the_same_voxels(self):
        assembly = touching_cross()
        fine = crossing_binder().repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)  # 40 x 40 x 40
        mesh = fem.hex_mesh(assembly, 0.05, binder=fine)
        same = fem.hex_mesh(assembly, 0.05, binder=crossing_binder())
        np.testing.assert_array_equal(mesh.elements, same.elements)
        np.testing.assert_array_equal(mesh.element_fibers, same.element_fibers)

    def test_thin_binder_in_a_finer_mask_gathers_where_it_is_densest(self):
        assembly = touching_cross()
        # Four voxels far from the fibers, along x in the cell's corner, with eight finer voxels each.
        voxels = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
        # None is half binder; their binder fills one voxel's worth, or two: the fullest take it.
        for counts, binder in (((3, 2, 2, 1), {voxels[0]}), ((4, 4, 4, 4), set(voxels[:2]))):
            mask = np.zeros((40, 40, 40), dtype=bool)
            for voxel, count in zip(voxels, counts):
                fine = [tuple(2 * at + step for at, step in zip(voxel, steps))
                        for steps in itertools.product((0, 1), repeat=3)]
                for point in fine[:count]:
                    mask[point] = True
            mesh = fem.hex_mesh(assembly, 0.05, binder=mask)
            self.assertEqual(voxels_of(mesh, mesh.element_fibers == 0, 0.05), binder)

    def test_a_finer_mask_keeps_its_binder_volume_beside_the_fibers(self):
        assembly = touching_cross()
        fine_fibers = voxels_of(fem.hex_mesh(assembly, 0.025), slice(None), 0.025)
        mask = np.random.default_rng(7).random((40, 40, 40)) < 0.2
        mesh = fem.hex_mesh(assembly, 0.05, binder=mask)
        binder = voxels_of(mesh, mesh.element_fibers == 0, 0.05)
        self.assertTrue(binder)
        # The room the binder voxels leave beside the fibers, in finer voxels, is the mask's binder there,
        # to within half a voxel.
        room = sum(tuple(2 * at + step for at, step in zip(voxel, steps)) not in fine_fibers
                   for voxel in binder for steps in itertools.product((0, 1), repeat=3))
        wanted = sum(tuple(point) not in fine_fibers for point in np.argwhere(mask).tolist())
        self.assertLessEqual(abs(room - wanted), 4)

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

    def test_a_tie_glues_faces_that_share_no_nodes(self):
        # Two tetrahedra that share no nodes, with face S1 of each on the plane z = 0.
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, -1]],
                         dtype=float)
        tie = fem.Tie("GLUE", secondary=np.array([[1, 0]]), main=np.array([[0, 0]]), tolerance=1e-3)
        mesh = fem.FemMesh(
            nodes=nodes, elements=np.arange(8).reshape(2, 4), element_type="tet4",
            element_fibers=np.array([1, 0]), element_materials=np.array([0, 1]), materials=("fiber", "binder"),
            cell_origin=(0.0, 0.0, -1.0), cell_lengths=(1.0, 1.0, 2.0), ties=(tie,),
        )
        keywords = dict(read_abaqus(mesh.write_abaqus(self.path / "tie.inp", scale=1e3)))
        self.assertEqual(keywords["*ELSET, ELSET=GLUE_SECONDARY_S1"], [["2"]])
        self.assertEqual(keywords["*ELSET, ELSET=GLUE_MAIN_S1"], [["1"]])
        self.assertEqual(keywords["*SURFACE, TYPE=ELEMENT, NAME=GLUE_SECONDARY"], [["GLUE_SECONDARY_S1", "S1"]])
        self.assertEqual(keywords["*SURFACE, TYPE=ELEMENT, NAME=GLUE_MAIN"], [["GLUE_MAIN_S1", "S1"]])
        self.assertEqual(keywords["*TIE, NAME=GLUE, POSITION TOLERANCE=1.000000e+00, ADJUST=NO"],
                         [["GLUE_SECONDARY", "GLUE_MAIN"]])
        with self.assertRaisesRegex(ValueError, "faces that do not exist"):
            dataclasses.replace(mesh, ties=(fem.Tie("GLUE", np.array([[2, 0]]), np.array([[0, 0]]), 1e-3),))
        with self.assertRaisesRegex(ValueError, "tetrahedral and wedge meshes only"):
            dataclasses.replace(self.mesh, ties=(tie,))

    def test_a_curved_ten_node_tetrahedron_has_the_volume_of_its_shape(self):
        # (x, y, z + c x z) is quadratic, so ten nodes follow it exactly; it scales volume by 1 + c x.
        corners = [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]]  # volume 1, centroid at x = 0.5
        bent = lambda points: points + 0.4 * points[:, :1] * points[:, 2:] * [0, 0, 1]
        nodes, elements = ten_node_tetrahedron(corners, bent)
        mesh = fem.FemMesh(
            nodes=nodes, elements=elements, element_type="tet10", element_fibers=np.array([1]),
            element_materials=np.array([0]), materials=("fiber",), cell_origin=(0.0, 0.0, 0.0),
            cell_lengths=(2.0, 1.0, 3.0),
        )
        np.testing.assert_allclose(mesh.element_volumes(), [1.2], rtol=1e-12)
        lowest, mean = mesh_module.tet10_jacobians(nodes, elements)
        np.testing.assert_allclose([lowest[0], mean[0]], [6.0, 7.2], rtol=1e-12)  # 6 (1 + 0.4 x) at x = 0, and its mean

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
        # Ten-node tetrahedra curve with the surface, so 6 around do better than 16 flat ones, with far fewer
        # elements.
        curved = fem.tet_mesh(assembly, order=2)
        self.assertEqual(curved.element_type, "tet10")
        self.assertAlmostEqual(curved.volume / (math.pi * radius**2 * length), 1.0, delta=0.01)
        self.assertLess(5 * curved.element_count, mesh.element_count)

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

    def test_ten_node_tetrahedra_curve_onto_the_fiber_surface(self):
        mesh = fem.tet_mesh(touching_cross(), 0.1, order=2)
        self.assertEqual(mesh.element_type, "tet10")
        points = mesh.nodes[mesh.elements]
        gaps = np.min([np.abs(distance - 0.1) for distance in axis_distances(mesh.nodes)], axis=0)
        on_surface = (gaps < 1e-9)[mesh.elements]
        curved = np.zeros(len(mesh.elements), dtype=bool)
        for middle, (first, second) in zip(range(4, 10), TET10_EDGES):
            offset = np.linalg.norm(points[:, middle] - (points[:, first] + points[:, second]) / 2, axis=1)
            length = np.linalg.norm(points[:, second] - points[:, first], axis=1)
            bowed = offset > 1e-12
            # A middle node is its edge's middle, or that pushed out onto the round surface the edge lies on,
            # by less than a quarter of the edge: in Nastran order, it is no other edge's middle.
            self.assertTrue((on_surface[bowed][:, [first, second, middle]]).all())
            self.assertTrue((offset < 0.25 * length).all())
            curved |= bowed
        self.assertTrue(curved.any())
        lowest, _ = mesh_module.tet10_jacobians(mesh.nodes, mesh.elements)
        self.assertTrue((lowest > 0).all())
        self.assertTrue((mesh.element_volumes() > 0).all())

    def test_a_ten_node_tetrahedron_that_curving_folds_is_straightened_as_far_as_it_needs(self):
        corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        straight, elements = ten_node_tetrahedron(corners)
        # Pulling the middle of edge 1-2 toward corner 0 by p makes the Jacobian 1 - 4 p (barycentric 1 + 2)
        # times its straight value, lowest along that edge: a pull of 0.4 needs half of it gone, 2 all of it.
        for pull, kept in ((-0.05, 1.0), (0.4, 0.5), (1.0, 0.125), (2.0, 0.0)):
            nodes = straight.copy()
            nodes[5] -= pull
            self.assertEqual(tet_module._straighten_distorted(nodes, elements), int(kept < 1.0))
            expected = straight.copy()
            expected[5] -= kept * pull
            np.testing.assert_allclose(nodes, expected, atol=1e-12)

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

    def test_a_bent_fiber_runs_straight_from_vertex_to_vertex(self):
        radius, turn = 10e-6, math.radians(30)
        line = np.array([[20e-6, 50e-6, 50e-6], [100e-6, 50e-6, 50e-6],
                         [100e-6 + 80e-6 * math.cos(turn), 50e-6 + 80e-6 * math.sin(turn), 50e-6]])
        mesh = fem.tet_mesh(assembly_of([200e-6, 150e-6, 100e-6], [line.tolist()],
                                        tangle.Material("fiber", diameter=2 * radius)))
        # Every node lies within the radius of the centerline's two straight stretches.
        start, step = line[:-1], np.diff(line, axis=0)
        along = np.clip(np.einsum("nsk,sk->ns", mesh.nodes[:, None] - start, step) / (step**2).sum(axis=1), 0.0, 1.0)
        distance = np.linalg.norm(mesh.nodes[:, None] - (start + along[..., None] * step), axis=2).min(axis=1)
        self.assertLessEqual(distance.max(), radius * (1 + 1e-6))
        self.assertGreater(distance.max(), radius * (1 - 1e-6))
        # Square sections at the ends and a section halving the turn: about the volume of the two stretches.
        self.assertAlmostEqual(mesh.volume / (math.pi * radius**2 * 160e-6), 1.0, delta=0.05)
        # Ten-node tetrahedra curve onto the same surface, mid-edge nodes and all.
        curved = fem.tet_mesh(assembly_of([200e-6, 150e-6, 100e-6], [line.tolist()],
                                          tangle.Material("fiber", diameter=2 * radius)), order=2)
        along = np.clip(np.einsum("nsk,sk->ns", curved.nodes[:, None] - start, step) / (step**2).sum(axis=1), 0.0, 1.0)
        distance = np.linalg.norm(curved.nodes[:, None] - (start + along[..., None] * step), axis=2).min(axis=1)
        self.assertLessEqual(distance.max(), radius * (1 + 1e-6))
        self.assertAlmostEqual(curved.volume / (math.pi * radius**2 * 160e-6), 1.0, delta=0.05)

    def test_processes_mesh_exactly_as_this_process_does(self):
        here = fem.tet_mesh(touching_cross(), 0.1, workers=1)
        apart = fem.tet_mesh(touching_cross(), 0.1, workers=2)
        np.testing.assert_array_equal(apart.nodes, here.nodes)
        np.testing.assert_array_equal(apart.elements, here.elements)
        np.testing.assert_array_equal(apart.element_fibers, here.element_fibers)

    def test_a_fiber_over_the_time_limit_stops_the_mesh_and_is_named(self):
        with self.assertRaisesRegex(RuntimeError, r"longer than 0\.001 s on fiber 1\b"):
            fem.tet_mesh(one_fiber(), fiber_time_limit=0.001)

    def test_ten_node_binder_tetrahedra_put_mid_edge_nodes_in_nastran_order(self):
        mesh = fem.tet_mesh(touching_cross(), 0.1, order=2, voxel_size=0.05, binder=crossing_binder())
        is_binder = mesh.element_fibers == 0
        points = mesh.nodes[mesh.elements[is_binder]]
        for middle, (first, second) in zip(range(4, 10), [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]):
            np.testing.assert_allclose(points[:, middle], (points[:, first] + points[:, second]) / 2, atol=1e-12)
        self.assertTrue((mesh.element_volumes() > 0).all())
        # Neighboring binder tetrahedra share their nodes: one node at each point.
        used = np.unique(mesh.elements[is_binder])
        self.assertEqual(len(np.unique(np.round(mesh.nodes[used] / 0.025).astype(int), axis=0)), len(used))
        self.assertEqual([tie.name for tie in mesh.ties], ["BINDER_TIE"])

    def test_junction_bridges_need_a_voxel_size_and_binder_the_whole_cell(self):
        binder = crossing_binder()
        with self.assertRaisesRegex(ValueError, "bond_radius_ratio needs voxel_size"):
            fem.tet_mesh(touching_cross(), 0.1, bond_radius_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "not cubes; pass voxel_size"):
            fem.tet_mesh(touching_cross(), 0.1, binder=np.zeros((20, 20, 40), dtype=bool))
        with self.assertRaisesRegex(ValueError, "voxel_size only sets"):
            fem.tet_mesh(touching_cross(), 0.1, voxel_size=0.05)
        with self.assertRaisesRegex(ValueError, "needs clip_to_cell=True"):
            fem.tet_mesh(touching_cross(), 0.1, voxel_size=0.05, binder=binder, clip_to_cell=False)
        with self.assertRaisesRegex(ValueError, "binder must be a"):
            fem.tet_mesh(touching_cross(), 0.1, voxel_size=0.05, binder=binder[:, :, :19])
        plain = fem.tet_mesh(touching_cross(), 0.1, voxel_size=0.05, binder=np.zeros_like(binder))
        self.assertEqual((plain.materials, plain.ties), (("fiber",), ()))

    def test_a_long_mesh_reports_its_progress(self):
        with mock.patch.object(tet_module._Progress, "FIRST", 0.0), contextlib.redirect_stdout(io.StringIO()) as out:
            fem.tet_mesh(touching_cross(), 0.1)
        lines = out.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("tet_mesh: 0 of 2 fibers meshed, 0 tetrahedra"))
        self.assertTrue(lines[-1].startswith("tet_mesh: all 2 fibers meshed"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            fem.tet_mesh(touching_cross(), 0.1)
        self.assertEqual(out.getvalue(), "")


@unittest.skipIf(fem is None or gmsh is None, "tangle.fem.tet_mesh needs NumPy and gmsh")
class TetBinderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assembly = touching_cross()
        cls.fibers = fem.tet_mesh(assembly, 0.1)
        cls.mesh = fem.tet_mesh(assembly, 0.1, voxel_size=0.05, binder=crossing_binder())
        hexes = fem.hex_mesh(assembly, 0.05, binder=crossing_binder())
        cls.is_binder = cls.mesh.element_fibers == 0
        cls.binder_voxels = voxels_of(hexes, hexes.element_fibers == 0, 0.05)
        cls.fiber_voxels = voxels_of(hexes, hexes.element_fibers != 0, 0.05)
        (cls.tie,) = cls.mesh.ties

    def test_six_tetrahedra_fill_each_binder_voxel_of_the_hex_mesh(self):
        mesh = self.mesh
        self.assertEqual(self.is_binder.sum(), 6 * len(self.binder_voxels))
        volumes = mesh.element_volumes()
        self.assertTrue((volumes > 0).all())
        np.testing.assert_allclose(volumes[self.is_binder], 0.05**3 / 6)
        centers = mesh.nodes[mesh.elements[self.is_binder]].mean(axis=1)
        x, y, z = np.floor((centers - mesh.cell_origin) / 0.05).astype(int).T.tolist()
        self.assertEqual(set(zip(z, y, x)), self.binder_voxels)
        self.assertEqual(mesh.materials, ("fiber", "binder"))
        self.assertTrue((mesh.element_materials[self.is_binder] == 1).all())

    def test_the_binder_tetrahedra_fit_together(self):
        # The faces that only one binder tetrahedron has make up the surface of the binder's voxels.
        faces = np.sort(self.mesh.elements[self.is_binder][:, np.array(ABAQUS_FACES)].reshape(-1, 3), axis=1)
        unique, counts = np.unique(faces, axis=0, return_counts=True)
        self.assertLessEqual(counts.max(), 2)
        outer = self.mesh.nodes[unique[counts == 1]]
        area = 0.5 * np.linalg.norm(np.cross(outer[:, 1] - outer[:, 0], outer[:, 2] - outer[:, 0]), axis=1).sum()
        steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        sides = sum((z + dz, y + dy, x + dx) not in self.binder_voxels
                    for z, y, x in self.binder_voxels for dz, dy, dx in steps)
        self.assertAlmostEqual(area, sides * 0.05**2)

    def test_the_fibers_are_meshed_as_without_binder_and_share_no_nodes_with_it(self):
        count = self.fibers.node_count
        np.testing.assert_array_equal(self.mesh.nodes[:count], self.fibers.nodes)
        np.testing.assert_array_equal(self.mesh.elements[~self.is_binder], self.fibers.elements)
        self.assertTrue((self.mesh.elements[self.is_binder] >= count).all())

    def test_the_tie_joins_binder_faces_next_to_fibers_to_the_fiber_surfaces(self):
        mesh, tie = self.mesh, self.tie
        self.assertEqual(tie.name, "BINDER_TIE")
        self.assertTrue(self.is_binder[tie.secondary[:, 0]].all())
        self.assertFalse(self.is_binder[tie.main[:, 0]].any())
        for element, corners in zip(tie.secondary[:, 0].tolist(), face_nodes(mesh, tie.secondary)):
            # A voxel face, square to one axis, with a voxel a fiber fills beyond it.
            points = mesh.nodes[corners]
            (axis,) = np.flatnonzero(np.ptp(points, axis=0) < 1e-12)
            beyond = points.mean(axis=0)
            beyond[axis] += 0.025 * np.sign(beyond[axis] - mesh.nodes[mesh.elements[element]].mean(axis=0)[axis])
            x, y, z = np.floor((beyond - mesh.cell_origin) / 0.05).astype(int).tolist()
            self.assertIn((z, y, x), self.fiber_voxels)
        # Their nodes lie within 1.25 voxels of a fiber's surface, well inside the tolerance.
        tied = mesh.nodes[np.unique(face_nodes(mesh, tie.secondary))]
        gaps = np.min([np.abs(distance - 0.1) for distance in axis_distances(tied)], axis=0)
        self.assertLessEqual(gaps.max(), 1.25 * 0.05)
        self.assertGreater(tie.tolerance, 1.25 * 0.05)
        # The main faces are each fiber's whole surface: round, or a flat end.
        for fiber, (start, end) in zip((1, 2), CROSS_AXES):
            faces = tie.main[mesh.element_fibers[tie.main[:, 0]] == fiber]
            corners = face_nodes(mesh, faces)
            points = mesh.nodes[corners] - start
            along = int(np.argmax(np.abs(np.subtract(end, start))))
            radial = np.linalg.norm(np.delete(points, along, axis=-1), axis=-1)
            at_end = np.isclose(points[..., along], 0.0) | np.isclose(points[..., along], end[along] - start[along])
            self.assertTrue((np.isclose(radial, 0.1, rtol=1e-6) | at_end).all())
            # A closed surface of triangles has two faces for every corner, less four.
            self.assertEqual(len(faces), 2 * len(np.unique(corners)) - 4)

    def test_a_binder_mask_twice_as_fine_meshes_the_same_binder(self):
        fine = crossing_binder().repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        mesh = fem.tet_mesh(touching_cross(), 0.1, voxel_size=0.05, binder=fine)
        np.testing.assert_array_equal(mesh.nodes, self.mesh.nodes)
        np.testing.assert_array_equal(mesh.elements, self.mesh.elements)
        self.assertEqual(mesh.ties[0].tolerance, self.tie.tolerance)

    def test_a_mask_without_a_voxel_size_meshes_binder_about_as_coarse_as_the_fibers(self):
        fine = crossing_binder().repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)  # voxels of 0.025
        # 4-node fibers of radius 0.1 have edges of 2 pi 0.1 / 16 = 0.039, nearer 0.05 than 0.025.
        mesh = fem.tet_mesh(touching_cross(), 0.1, binder=fine)
        np.testing.assert_array_equal(mesh.nodes, self.mesh.nodes)
        np.testing.assert_array_equal(mesh.elements, self.mesh.elements)
        # 10-node ones, capped by element_size at 0.1, bring binder voxels of 0.1: four of the mask's.
        curved = fem.tet_mesh(touching_cross(), 0.1, order=2, binder=fine)
        is_binder = curved.element_fibers == 0
        self.assertTrue(is_binder.any())
        np.testing.assert_allclose(curved.element_volumes()[is_binder], 0.1**3 / 6)

    def test_a_binder_face_beside_a_voxel_a_fiber_fills_part_of_is_tied(self):
        assembly = touching_cross()
        fine_fibers = voxels_of(fem.hex_mesh(assembly, 0.025), slice(None), 0.025)

        def finer(voxel):
            return [tuple(2 * at + step for at, step in zip(voxel, steps))
                    for steps in itertools.product((0, 1), repeat=3)]

        # A voxel no fiber fills, beside one that a fiber fills part of, also no fiber's.
        for voxel, axis, step in itertools.product(itertools.product(range(1, 19), repeat=3), range(3), (-1, 1)):
            beside = list(voxel)
            beside[axis] += step
            beside = tuple(beside)
            partly = sum(point in fine_fibers for point in finer(beside))
            if (voxel not in self.fiber_voxels and beside not in self.fiber_voxels
                    and not any(point in fine_fibers for point in finer(voxel)) and 0 < partly < 8):
                break
        else:
            self.fail("no voxel beside a fiber's edge")
        mask = np.zeros((40, 40, 40), dtype=bool)
        for point in finer(voxel):
            mask[point] = True
        coarse = np.zeros((20, 20, 20), dtype=bool)
        coarse[voxel] = True
        # The side between them, square to x, y or z: tied when the finer mask shows the fiber in the next voxel.
        across = 2 - axis
        for binder, tied in ((mask, True), (coarse, False)):
            mesh = fem.tet_mesh(assembly, 0.1, voxel_size=0.05, binder=binder)
            self.assertEqual(voxels_of(fem.hex_mesh(assembly, 0.05, binder=binder),
                                       slice(None), 0.05) - self.fiber_voxels, {voxel})
            plane = mesh.cell_origin[across] + (voxel[axis] + (step > 0)) * 0.05
            corners = mesh.nodes[face_nodes(mesh, mesh.ties[0].secondary)] if mesh.ties else np.zeros((0, 3, 3))
            on_side = np.isclose(corners[..., across], plane, rtol=0.0, atol=1e-12).all(axis=1)
            self.assertEqual(on_side.any(), tied)

    def test_abaqus_writes_the_tie_and_nastran_says_to_glue_the_binder(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = read_abaqus(self.mesh.write_abaqus(Path(directory) / "mesh.inp", scale=1e6))
            deck = self.mesh.write_nastran(Path(directory) / "mesh.bdf").read_text()
        keywords = dict(blocks)
        for role, faces in (("SECONDARY", self.tie.secondary), ("MAIN", self.tie.main)):
            written = set()
            for element_set, side in keywords[f"*SURFACE, TYPE=ELEMENT, NAME=BINDER_TIE_{role}"]:
                written |= {(int(element) - 1, int(side[1:]) - 1)
                            for row in keywords[f"*ELSET, ELSET={element_set}"] for element in row}
            self.assertEqual(written, set(map(tuple, faces.tolist())))
        tolerance = f"{self.tie.tolerance * 1e6:.6e}"
        self.assertEqual(keywords[f"*TIE, NAME=BINDER_TIE, POSITION TOLERANCE={tolerance}, ADJUST=NO"],
                         [["BINDER_TIE_SECONDARY", "BINDER_TIE_MAIN"]])
        self.assertIn("The binder (PSOLID 3) shares no nodes with the fibers", deck)
        self.assertIn("MPC = 1", deck)

    def test_nastran_glues_each_tied_binder_node_to_the_closest_fiber_face_with_mpcs(self):
        mesh, tie = self.mesh, self.tie
        found, main, weights = mesh.tie_constraints(tie)
        tied = np.unique(face_nodes(mesh, tie.secondary))
        self.assertGreater(len(found), 0)
        self.assertTrue(set(found.tolist()) <= set(tied.tolist()))
        binder_nodes = set(mesh.elements[self.is_binder].ravel().tolist())
        self.assertTrue(set(found.tolist()) <= binder_nodes)
        self.assertFalse(set(main.ravel().tolist()) & binder_nodes)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        self.assertTrue((weights >= 0.0).all())
        # The point each node follows lies on a fiber's surface, within the tolerance of the node.
        followed = np.einsum("kc,kcd->kd", weights, mesh.nodes[main])
        self.assertLessEqual(np.linalg.norm(followed - mesh.nodes[found], axis=1).max(), tie.tolerance)
        gaps = np.min([np.abs(distance - 0.1) for distance in axis_distances(followed)], axis=0)
        ends = np.min([np.abs(followed[:, int(np.argmax(np.abs(np.subtract(end, start))))] - value)
                       for start, end in CROSS_AXES for value in (start[int(np.argmax(np.abs(np.subtract(end, start))))],
                                                                   end[int(np.argmax(np.abs(np.subtract(end, start))))])],
                      axis=0)
        self.assertTrue(((gaps < 0.01) | (ends < 1e-9)).all())
        with tempfile.TemporaryDirectory() as directory:
            equations = mpc_equations(mesh.write_nastran(Path(directory) / "mesh.bdf"))
        self.assertEqual(len(equations), 3 * len(found))
        for terms, node in zip(equations[::3], found.tolist()):
            self.assertEqual(terms[0], (node + 1, 1, 1.0))
            self.assertAlmostEqual(sum(value for _, _, value in terms[1:]), -1.0, places=5)
        for index, terms in enumerate(equations):
            self.assertEqual({direction for _, direction, _ in terms}, {index % 3 + 1})


def straight_fiber(lengths, start, end, diameter=0.2, periodic=None):
    """One round fiber from ``start`` to ``end`` in a cell of ``lengths``."""
    return assembly_of(lengths, [[list(start), list(end)]], tangle.Material("fiber", diameter=diameter), periodic)


@unittest.skipIf(fem is None, "tangle.fem needs NumPy")
class SweepMeshTests(unittest.TestCase):
    def test_a_short_straight_fiber_is_six_wedges_between_two_hexagons_of_its_area(self):
        mesh = fem.sweep_mesh(one_fiber())
        self.assertEqual(mesh.element_type, "penta6")
        self.assertEqual((mesh.node_count, mesh.element_count), (14, 6))
        self.assertTrue((mesh.element_volumes() > 0).all())
        self.assertAlmostEqual(mesh.volume, math.pi * 0.1**2 * 0.6)
        # Each section is its middle node and six around it, as far out as gives the circle's area.
        ends = mesh.nodes.reshape(2, 7, 3)
        np.testing.assert_allclose(ends[:, 0], [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
        radius = 0.1 * math.sqrt(2 * math.pi / (6 * math.sin(math.pi / 3)))
        np.testing.assert_allclose(np.linalg.norm(ends[:, 1:, 1:] - 0.5, axis=2), radius)
        self.assertIsNone(mesh.bonds)
        self.assertEqual(mesh.ties, ())

    def test_sections_are_at_most_max_length_diameters_apart(self):
        mesh = fem.sweep_mesh(straight_fiber([4.0, 1.0, 1.0], (0.1, 0.5, 0.5), (3.9, 0.5, 0.5)))
        middles = mesh.nodes[::7, 0]
        np.testing.assert_allclose(middles, np.linspace(0.1, 3.9, 6))  # 3.8 long, at most 0.8: five wedges
        self.assertEqual(mesh.element_count, 5 * 6)
        self.assertAlmostEqual(mesh.volume, math.pi * 0.01 * 3.8)
        finer = fem.sweep_mesh(straight_fiber([4.0, 1.0, 1.0], (0.1, 0.5, 0.5), (3.9, 0.5, 0.5)), max_length=1.0)
        self.assertEqual(finer.element_count, 19 * 6)
        around = fem.sweep_mesh(one_fiber(), nodes_around=8)
        self.assertEqual((around.node_count, around.element_count), (18, 8))
        self.assertAlmostEqual(around.volume, math.pi * 0.1**2 * 0.6)

    def test_a_bend_gets_sections_where_the_centerline_strays_from_the_wedges(self):
        angles = np.linspace(0.0, math.pi / 2, 31)
        arc = np.stack([0.1 + 0.8 * np.sin(angles), 0.9 - 0.8 * np.cos(angles), np.full(31, 0.5)], axis=1)
        assembly = assembly_of([1.0, 1.0, 1.0], [arc.tolist()], tangle.Material("fiber", diameter=0.05))
        loose = fem.sweep_mesh(assembly, bend_tolerance=1.0)
        tight = fem.sweep_mesh(assembly, bend_tolerance=0.1)
        self.assertLess(loose.element_count, tight.element_count)
        for mesh, tolerance in ((loose, 1.0), (tight, 0.1)):
            middles = mesh.nodes[::7]
            np.testing.assert_allclose(np.linalg.norm(middles[:, :2] - [0.1, 0.9], axis=1), 0.8, rtol=1e-9)
            # The arc's sagitta between two sections stays within the tolerance.
            chords = np.linalg.norm(np.diff(middles, axis=0), axis=1)
            self.assertLessEqual((0.8 - np.sqrt(0.64 - chords**2 / 4)).max(), tolerance * 0.025 + 0.8 * 0.0035)
            self.assertLessEqual(chords.max(), 4 * 0.05 + 1e-12)
            self.assertTrue((mesh.element_volumes() > 0).all())

    def test_fibers_are_cut_flush_with_the_cell_walls(self):
        assembly = straight_fiber([1.0, 1.0, 1.0], (0.7, 0.5, 0.5), (1.5, 0.5, 0.5), periodic=[True, False, False])
        mesh = fem.sweep_mesh(assembly)
        self.assertAlmostEqual(mesh.volume, math.pi * 0.01 * 0.8)
        sets = mesh.node_sets()
        self.assertEqual((len(sets["XMIN"]), len(sets["XMAX"])), (7, 7))
        whole = fem.sweep_mesh(assembly, clip_to_cell=False)
        self.assertAlmostEqual(whole.volume, math.pi * 0.01 * 0.8)
        self.assertGreater(whole.nodes[:, 0].max(), 1.4)
        # A fiber across a wall at an angle ends in the wall too.
        slanted = fem.sweep_mesh(straight_fiber([1.0, 1.0, 1.0], (0.6, 0.3, 0.5), (1.4, 0.7, 0.5),
                                                periodic=[True, False, False]))
        sets = slanted.node_sets()
        self.assertEqual((len(sets["XMIN"]), len(sets["XMAX"])), (7, 7))
        self.assertTrue((slanted.element_volumes() > 0).all())

    def test_a_steep_cut_end_keeps_the_next_section_beyond_its_slide(self):
        # Crossing the x wall at about 65 degrees from its normal, with short wedges: the cut ends slide
        # about three radii along the fiber, past where the next sections would be.
        line = np.linspace([0.7, 0.3, 0.5], [1.3, 1.6, 0.5], 30)
        assembly = assembly_of([1.0, 2.0, 1.0], [line.tolist()], tangle.Material("fiber", diameter=0.2),
                               periodic=[True, False, False])
        mesh = fem.sweep_mesh(assembly, max_length=0.5)
        corners = mesh.nodes[mesh.elements[:, mesh_module.PENTA_TETRAHEDRA]]
        edges = corners[:, :, 1:] - corners[:, :, :1]
        volumes = np.einsum("...i,...i", edges[..., 0, :], np.cross(edges[..., 1, :], edges[..., 2, :]))
        self.assertTrue((volumes > 0).all())
        sets = mesh.node_sets()
        self.assertEqual((len(sets["XMIN"]), len(sets["XMAX"])), (7, 7))
        self.assertAlmostEqual(mesh.volume, math.pi * 0.01 * float(np.linalg.norm([0.6, 1.3])), places=6)

    def test_an_oval_fiber_keeps_its_long_axis(self):
        oval = tangle.Material("ribbon", diameter=0.2, thickness=0.1)
        assembly = assembly_of([1.0, 1.0, 1.0], [[[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]]], oval)
        mesh = fem.sweep_mesh(assembly)
        self.assertAlmostEqual(mesh.volume, math.pi * 0.1 * 0.05 * 0.6)
        ((long, short),) = assembly.section_semi_axes()
        axis = np.asarray(assembly.long_axes()[0][0])
        offsets = mesh.nodes.reshape(2, 7, 3)[0, 1:] - [0.2, 0.5, 0.5]
        widths = np.abs(offsets @ axis)
        self.assertAlmostEqual(widths.max() / np.abs(offsets @ np.cross([1.0, 0.0, 0.0], axis)).max(),
                               long / short / math.sin(math.pi / 3), places=6)

    def test_a_binder_mask_is_two_wedges_to_a_voxel_glued_to_the_fibers(self):
        assembly = touching_cross()
        mesh = fem.sweep_mesh(assembly, binder=crossing_binder(), voxel_size=0.05)
        is_binder = mesh.element_fibers == 0
        hexes = fem.hex_mesh(assembly, 0.05, binder=crossing_binder())
        voxels = voxels_of(hexes, hexes.element_fibers == 0, 0.05)
        self.assertEqual(is_binder.sum(), 2 * len(voxels))
        np.testing.assert_allclose(mesh.element_volumes()[is_binder], 0.05**3 / 2)
        self.assertEqual(mesh.materials, ("fiber", "binder"))
        (tie,) = mesh.ties
        self.assertTrue(is_binder[tie.secondary[:, 0]].all())
        self.assertFalse(is_binder[tie.main[:, 0]].any())
        found, main, weights = mesh.tie_constraints(tie)
        self.assertGreater(len(found), 0)
        binder_nodes = set(mesh.elements[is_binder].ravel().tolist())
        self.assertTrue(set(found.tolist()) <= binder_nodes)
        self.assertFalse(set(main.ravel().tolist()) & binder_nodes)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            deck = mesh.write_nastran(path / "mesh.bdf")
            _, elements, properties, _ = read_nastran(deck)
            self.assertEqual({name for name, _, _ in elements.values()}, {"CPENTA"})
            self.assertEqual(properties, {1: 1, 2: 1, 3: 2})
            self.assertEqual(len(mpc_equations(deck)), 3 * len(found))
            keywords = dict(read_abaqus(mesh.write_abaqus(path / "mesh.inp")))
            self.assertIn("*ELEMENT, TYPE=C3D6, ELSET=BINDER", keywords)
            self.assertTrue(any(line.startswith("*TIE, NAME=BINDER_TIE") for line in keywords))
            piece, arrays = read_vtu(mesh.write_vtu(path / "mesh.vtu"))
            np.testing.assert_array_equal(arrays["types"], 13)  # VTK_WEDGE

    def test_bonds_become_beams_that_follow_the_fiber_sections(self):
        assembly = touching_cross()
        # One junction between the middles of the two fibers.
        with mock.patch.object(sweep_module._tangle, "fem_bond_anchors", return_value=[(7, 0, 0, 0.5, 1, 0, 0.5)]):
            mesh = fem.sweep_mesh(assembly, bond_radius_ratio=0.5)
        bonds = mesh.bonds
        self.assertEqual(bonds.count, 1)
        np.testing.assert_allclose(mesh.nodes[bonds.beams[0]], [[0.5, 0.5, 0.4], [0.5, 0.5, 0.6]], atol=1e-12)
        np.testing.assert_allclose(bonds.radii, [0.05])
        self.assertEqual(bonds.junction_ids.tolist(), [7])
        self.assertEqual(bonds.material, 1)
        self.assertEqual(mesh.materials, ("fiber", "binder"))
        self.assertEqual(sorted(bonds.ends.tolist()), sorted(bonds.beams[0].tolist()))
        # The beam ends are nodes of no wedge, each following a section of its fiber at the anchor.
        self.assertFalse(set(bonds.ends.tolist()) & set(mesh.elements.ravel().tolist()))
        for end, section in zip(bonds.ends.tolist(), bonds.sections):
            self.assertEqual(len(section), 7)
            np.testing.assert_allclose(mesh.nodes[section].mean(axis=0), mesh.nodes[end], atol=1e-12)
            self.assertEqual(len(set(mesh.element_fibers[np.isin(mesh.elements, section).any(axis=1)].tolist())), 1)
        np.testing.assert_allclose(np.abs(bonds.orientations @ [0.0, 0.0, 1.0]), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            deck = mesh.write_nastran(path / "mesh.bdf", fem.ElasticMaterial(1e9, 0.3))
            grids, elements, properties, materials = read_nastran(deck)
            self.assertEqual(len(grids), mesh.node_count)
            (bar,) = nastran_entries(deck, "CBAR")
            self.assertEqual([int(value) for value in bar[:4]],
                             [mesh.element_count + 1, 3, *(bonds.beams[0] + 1).tolist()])
            self.assertEqual(float(bar[6]) ** 2 + float(bar[4]) ** 2 + float(bar[5]) ** 2, 1.0)
            rbe3 = nastran_entries(deck, "RBE3")
            self.assertEqual(len(rbe3), 2)
            for fields, end, section in zip(rbe3, bonds.ends.tolist(), bonds.sections.tolist()):
                values = [field for field in fields if field]
                self.assertEqual(values[1:5], [str(end + 1), "123456", "1.0", "123"])
                self.assertEqual([int(value) for value in values[5:]], [node + 1 for node in section])
            text = deck.read_text()
            self.assertIn("PBAR*", text)
            pbar = text[text.index("PBAR*"):].splitlines()
            self.assertEqual(int(pbar[0][8:24]), 3)
            self.assertEqual(int(pbar[0][24:40]), 2)
            self.assertAlmostEqual(float(pbar[0][40:56]), math.pi * 0.05**2)
            self.assertAlmostEqual(float(pbar[1][24:40]), math.pi * 0.05**4 / 2)
            keywords = dict(read_abaqus(mesh.write_abaqus(path / "mesh.inp", fem.ElasticMaterial(1e9, 0.3))))
            self.assertEqual(keywords["*ELEMENT, TYPE=B31, ELSET=BONDS_1"],
                             [[str(mesh.element_count + 1), *map(str, bonds.beams[0] + 1)]])
            self.assertIn("*BEAM SECTION, ELSET=BONDS_1, MATERIAL=BINDER, SECTION=CIRC", keywords)
            self.assertIn("*MATERIAL, NAME=BINDER", keywords)
            couplings = [line for line in keywords if line.startswith("*COUPLING")]
            self.assertEqual(len(couplings), 2)
            piece, arrays = read_vtu(mesh.write_vtu(path / "mesh.vtu"))
            self.assertEqual(int(piece.get("NumberOfCells")), mesh.element_count + 1)
            self.assertEqual(arrays["types"][-1], 3)  # VTK_LINE
            np.testing.assert_array_equal(arrays["connectivity"][-2:], bonds.beams[0])

    def test_a_bond_across_a_periodic_wall_stops_at_the_wall_on_both_sides(self):
        # Fiber 1 along y at x = 0.05 and fiber 2 along z at x = 0.85: 0.2 apart through the x wall.
        assembly = assembly_of([1.0, 1.0, 1.0], [[[0.05, 0.2, 0.5], [0.05, 0.8, 0.5]], [[0.85, 0.5, 0.2], [0.85, 0.5, 0.8]]],
                               tangle.Material("fiber", diameter=0.2), periodic=[True, False, False])
        with mock.patch.object(sweep_module._tangle, "fem_bond_anchors", return_value=[(1, 0, 0, 0.5, 1, 0, 0.5)]):
            mesh = fem.sweep_mesh(assembly, bond_radius_ratio=1.0)
            whole = fem.sweep_mesh(assembly, bond_radius_ratio=1.0, clip_to_cell=False)
        bonds = mesh.bonds
        self.assertEqual(bonds.count, 2)
        ends = sorted(mesh.nodes[bonds.beams].tolist())
        np.testing.assert_allclose(ends, [[[0.05, 0.5, 0.5], [0.0, 0.5, 0.5]], [[0.85, 0.5, 0.5], [1.0, 0.5, 0.5]]],
                                   atol=1e-12)
        free = set(bonds.beams.ravel().tolist()) - set(bonds.ends.tolist())
        sets = mesh.node_sets()
        self.assertEqual(free, {int(sets["XMIN"][np.isin(sets["XMIN"], list(free))][0]),
                                int(sets["XMAX"][np.isin(sets["XMAX"], list(free))][0])})
        # Whole fibers have no images to bond across the wall.
        self.assertIsNone(whole.bonds)
        self.assertEqual(whole.materials, ("fiber",))

    def test_bonds_of_a_relaxed_run_sit_on_its_placed_centerlines(self):
        recipe = tangle.Recipe(tangle.Cell([1.0, 1.0, 1.0]))
        recipe.insert(tangle.FiberCollection.from_centerlines(
            [[list(start), list(end)] for start, end in CROSS_AXES], tangle.Material("fiber", diameter=0.2)))
        recipe.relax_for(5)
        recipe.capture_junctions(tangle.JunctionPolicy("binder at crossings", "bond", max_surface_gap=0.05))
        recipe.relax_for(5)
        result = recipe.run(tangle.RelaxationSettings(backend="cpu"))
        self.assertEqual(result.junction_count, 1)
        mesh = fem.sweep_mesh(result, bond_radius_ratio=0.5)
        self.assertEqual(mesh.bonds.count, 1)
        centerlines = [np.asarray(line) for line in result.assembly.centerlines()]
        on = []
        for end in mesh.bonds.beams[0]:  # each end lies on one fiber's relaxed centerline
            point = mesh.nodes[end]
            gaps = []
            for line in centerlines:
                steps = np.diff(line, axis=0)
                along = np.clip(np.einsum("ij,ij->i", point - line[:-1], steps) / (steps * steps).sum(1), 0.0, 1.0)
                gaps.append(np.linalg.norm(point - (line[:-1] + along[:, None] * steps), axis=1).min())
            self.assertLess(min(gaps), 1e-9)
            on.append(int(np.argmin(gaps)))
        self.assertEqual(sorted(on), [0, 1])
        self.assertAlmostEqual(float(np.linalg.norm(np.subtract(*mesh.nodes[mesh.bonds.beams[0]]))), 0.2, delta=0.02)

    def test_arguments_are_checked(self):
        with self.assertRaisesRegex(ValueError, "nodes_around"):
            fem.sweep_mesh(one_fiber(), nodes_around=2)
        with self.assertRaisesRegex(ValueError, "max_length"):
            fem.sweep_mesh(one_fiber(), max_length=0.0)
        with self.assertRaisesRegex(ValueError, "voxel_size only sets"):
            fem.sweep_mesh(one_fiber(), voxel_size=0.05)
        with self.assertRaisesRegex(ValueError, "clip_to_cell=True"):
            fem.sweep_mesh(touching_cross(), binder=crossing_binder(), clip_to_cell=False)

    def test_binder_that_would_come_out_empty_is_an_error(self):
        # An assembly without junctions, as a CT fit's is, has no bonds to mesh.
        with self.assertRaisesRegex(ValueError, "pass binder=fit.binder"):
            fem.sweep_mesh(touching_cross(), bond_radius_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "no voxel outside the fibers"):
            fem.sweep_mesh(touching_cross(), binder=np.zeros((20, 20, 20), dtype=bool), voxel_size=0.05)

    def test_a_fiber_a_hair_past_its_bend_limit_still_gets_binder(self):
        # Fitted fibers can end up just over their bend limit; the binder's voxels do not depend on it.
        bent = [[0.2, 0.5, 0.4], [0.5, 0.5, 0.4], [0.8, 0.53, 0.4]]
        step = np.diff(np.asarray(bent), axis=0)
        angle = math.acos(step[0] @ step[1] / np.linalg.norm(step[0]) / np.linalg.norm(step[1]))
        assembly = tangle.Assembly(tangle.Cell([1.0, 1.0, 1.0]))
        assembly.insert(tangle.FiberCollection.from_centerlines(
            [bent, [list(point) for point in CROSS_AXES[1]]],
            tangle.Material("fiber", diameter=0.2, min_bend_radius=0.3 / angle)))
        mesh = fem.sweep_mesh(assembly, binder=crossing_binder(), voxel_size=0.05)
        self.assertTrue((mesh.element_fibers == 0).any())
        self.assertEqual(len(mesh.ties), 1)


if __name__ == "__main__":
    unittest.main()
