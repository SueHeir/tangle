"""tangle.lammps: a LAMMPS bonded-particle data file's fibers read, relaxed in Tangle and written back."""

import math
import os
import tempfile
import unittest
from pathlib import Path

import tangle
from tangle import lammps

BACKEND = os.environ.get("TANGLE_BACKEND", "wgpu")
# The relaxation runs on the GPU; CI runners have none.
GPU = not (os.environ.get("CI") and "TANGLE_BACKEND" not in os.environ)

TITLE = "LAMMPS data file via write_data, version 29 Aug 2024, timestep = 0, units = si"

# A 10-wide periodic box as LAMMPS's write_data leaves it, the rows out of order. Fiber A (molecule 1): five
# touching spheres 1 across along x at y 5, z 5, from x 7 through the periodic face to x 11 (atoms 4 and 5 written
# at 0 and 1, one box on). Fiber B (molecule 2): five along y at x 0, z 5.5, so that its middle sphere overlaps
# A's fourth by half a diameter. Atom 11 is a sphere on its own, and bond 9 joins the fibers.
ATOMS = [  # id, molecule, type, diameter, x, y, z, x image flag
    (3, 1, 1, 1.0, 9, 5, 5, 0),
    (1, 1, 1, 1.0, 7, 5, 5, 0),
    (5, 1, 1, 1.0, 1, 5, 5, 1),
    (2, 1, 1, 1.0, 8, 5, 5, 0),
    (4, 1, 1, 1.0, 0, 5, 5, 1),
    (6, 2, 2, 1.0, 0, 3, 5.5, 0),
    (7, 2, 2, 1.0, 0, 4, 5.5, 0),
    (8, 2, 2, 1.0, 0, 5, 5.5, 0),
    (9, 2, 2, 1.0, 0, 6, 5.5, 0),
    (10, 2, 2, 1.0, 0, 7, 5.5, 0),
    (11, 3, 3, 0.5, 5, 1, 1, 0),
]
BONDS = [(1, 4, 5), (1, 2, 3), (1, 1, 2), (1, 3, 4), (1, 6, 7), (1, 7, 8), (1, 8, 9), (1, 9, 10), (2, 3, 8)]
FIBER_A = [[x, 5.0, 5.0] for x in (7.0, 8.0, 9.0, 10.0, 11.0)]
FIBER_B = [[0.0, y, 5.5] for y in (3.0, 4.0, 5.0, 6.0, 7.0)]
# Two fibers by the z faces: along x on the floor, out of the box by 0.3, and along y under the ceiling, out by 0.4.
BY_THE_WALLS = [(atom_id, 1, 1, 1.0, atom_id + 2, 5, 0.2, 0) for atom_id in (1, 2, 3)] + [
    (atom_id, 2, 1, 1.0, 5, atom_id - 1, 9.9, 0) for atom_id in (4, 5, 6)]
WALL_BONDS = [(1, 1, 2), (1, 2, 3), (1, 4, 5), (1, 5, 6)]


def bpm_rows(atoms=ATOMS, images=True):
    """``bpm/sphere`` Atoms rows (id molecule type diameter density x y z, then the image flags); atom 2's row
    ends with a comment."""
    rows = []
    for atom_id, molecule, kind, diameter, x, y, z, image in atoms:
        row = f"{atom_id} {molecule} {kind} {diameter:g} 1000 {x:g} {y:g} {z:g}"
        rows.append(row + (f" {image} 0 0" if images else "") + (" # second" if atom_id == 2 else ""))
    return rows


def write(directory, rows, bonds=(), *, style="bpm/sphere", header=(), sections=(), velocities=False,
          name="fibers.data"):
    """A data file with these Atoms rows and ``(type, atom1, atom2)`` bonds in a 10-wide box."""
    lines = [TITLE, "", f"{len(rows)} atoms", "3 atom types"]
    if bonds:
        lines += [f"{len(bonds)} bonds", "2 bond types"]
    lines += ["", "0 10 xlo xhi", "0 10 ylo yhi", "0 10 zlo zhi", *header, ""]
    for section in sections:
        lines += [*section, ""]
    lines += [f"Atoms # {style}" if style else "Atoms", "", *rows, ""]
    if velocities:
        # bpm/sphere Velocities rows: id, velocity, angular velocity
        lines += ["Velocities", "", *[f"{row.split()[0]} 0 0 0 0 0 0" for row in rows], ""]
    if bonds:
        lines += ["Bonds", "", *[f"{index} {kind} {a} {b}" for index, (kind, a, b) in enumerate(bonds, 1)]]
    path = Path(directory) / name
    path.write_text("\n".join(lines) + "\n")
    return path


def fibers_file(directory, images=True, name="fibers.data"):
    return write(directory, bpm_rows(images=images), BONDS, velocities=True, name=name)


def assembly_of(data):
    assembly = tangle.Assembly(data.cell)
    assembly.insert(data.fibers)
    return assembly


def row_values(text, atom_id):
    """The numbers of an atom's Atoms row, the only rows with 8 values or more."""
    for line in text.splitlines():
        words = line.split("#")[0].split()
        if len(words) >= 8 and words[0] == str(atom_id):
            return [float(word) for word in words]
    raise AssertionError(f"no row for atom {atom_id}")


class ReadDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_each_molecule_is_a_fiber_in_bond_order_and_unbroken_across_the_box(self):
        data = lammps.read_data(fibers_file(self.directory))
        self.assertEqual(data.atom_style, "bpm/sphere")
        self.assertEqual(data.fiber_atoms, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self.assertEqual(data.other_atoms, [11])
        self.assertEqual(data.fibers.centerlines(), [FIBER_A, FIBER_B])
        self.assertEqual(data.fibers.rest_centerlines(), [FIBER_A, FIBER_B])
        self.assertEqual(data.cell.lengths, [10.0] * 3)
        self.assertEqual(data.cell.periodic, [True] * 3)
        assembly = assembly_of(data)
        self.assertEqual(assembly.fiber_materials(), ["type 1", "type 2"])
        self.assertEqual(assembly.section_semi_axes(), [(0.5, 0.5), (0.5, 0.5)])
        notes = " ".join(data.notes)
        self.assertIn("1 atom not bonded", notes)
        self.assertIn("1 bond between fibers", notes)
        self.assertIn("fibers=2", repr(data))

    def test_rows_without_image_flags_are_joined_across_the_box_along_their_bonds(self):
        data = lammps.read_data(fibers_file(self.directory, images=False))
        self.assertEqual(data.fibers.centerlines(), [FIBER_A, FIBER_B])

    def test_bonds_that_branch_and_lone_atoms_stay_where_they_are(self):
        branch = [(12, 4, 1, 1.0, 2, 2, 2, 0), (13, 4, 1, 1.0, 3, 2, 2, 0), (14, 4, 1, 1.0, 2, 3, 2, 0),
                  (15, 4, 1, 1.0, 2, 2, 3, 0)]
        bonds = BONDS + [(1, 12, 13), (1, 12, 14), (1, 12, 15)]
        data = lammps.read_data(write(self.directory, bpm_rows(ATOMS + branch), bonds))
        self.assertEqual(data.fiber_atoms, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self.assertEqual(data.other_atoms, [11, 12, 13, 14, 15])
        self.assertIn("4 atoms whose bonds branch or close a loop, 1 atom not bonded", " ".join(data.notes))

    def test_spheres_bonded_to_more_than_the_next_one_follow_their_ids(self):
        # As create_bonds many leaves a chain of spheres closer than its cutoff: each bonded to the next two.
        atoms = [(atom_id, 1, 1, 1.0, atom_id, 5, 5, 0) for atom_id in range(1, 6)]
        bonds = [(1, a, b) for a in range(1, 6) for b in (a + 1, a + 2) if b <= 5]
        data = lammps.read_data(write(self.directory, bpm_rows(atoms), bonds))
        self.assertEqual(data.fiber_atoms, [[1, 2, 3, 4, 5]])

    def test_a_file_without_bonds_chains_consecutive_atoms_of_a_molecule(self):
        atoms = [(1, 1, 1, 1.0, 1, 5, 5, 0), (2, 1, 1, 1.0, 2, 5, 5, 0), (3, 1, 1, 1.0, 3, 5, 5, 0),
                 (4, 1, 1, 1.0, 7, 5, 5, 0), (5, 1, 1, 1.0, 8, 5, 5, 0), (6, 1, 1, 1.0, 9, 5, 5, 0),
                 (7, 2, 1, 1.0, 5, 8, 8, 0)]
        data = lammps.read_data(write(self.directory, bpm_rows(atoms)))
        self.assertEqual(data.fiber_atoms, [[1, 2, 3], [4, 5, 6]])  # 4 apart between atoms 3 and 4
        self.assertEqual(data.other_atoms, [7])
        notes = " ".join(data.notes)
        self.assertIn("The file has no bonds", notes)
        self.assertIn("1 atom with no sphere close enough", notes)

    def test_fiber_bond_types_tell_fibers_of_one_molecule_apart(self):
        atoms = [(atom_id, 0, *rest) for atom_id, _, *rest in ATOMS]  # every atom molecule 0
        path = write(self.directory, bpm_rows(atoms), BONDS)
        self.assertEqual(lammps.read_data(path).fiber_atoms, [])  # the bond between them makes one branched group
        data = lammps.read_data(path, fiber_bond_types=[1])
        self.assertEqual(data.fiber_atoms, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self.assertIn("1 bond between fibers", " ".join(data.notes))

    def test_other_atom_styles(self):
        # hybrid sphere bond: id type x y z diameter density molecule
        rows = [f"{a} {kind} {x:g} {y:g} {z:g} {d:g} 1000 {m}" for a, m, kind, d, x, y, z, _ in ATOMS]
        data = lammps.read_data(write(self.directory, rows, BONDS, style="hybrid sphere bond"))
        self.assertEqual(data.atom_style, "hybrid sphere bond")
        self.assertEqual(data.fibers.centerlines(), [FIBER_A, FIBER_B])

        # bond, with a type label: no diameter column, so diameter= gives it
        labels = [["Atom Type Labels", "", "1 fiber", "2 coarse", "3 grain"]]
        rows = [f"{a} {m} {['fiber', 'coarse', 'grain'][kind - 1]} {x:g} {y:g} {z:g}"
                for a, m, kind, d, x, y, z, _ in ATOMS]
        path = write(self.directory, rows, BONDS, style="bond", sections=labels, name="bond.data")
        with self.assertRaisesRegex(ValueError, "diameter="):
            lammps.read_data(path)
        data = lammps.read_data(path, diameter={"fiber": 1.0, "coarse": 2.0})
        assembly = assembly_of(data)
        self.assertEqual(assembly.fiber_materials(), ["type fiber", "type coarse"])
        self.assertEqual(assembly.section_semi_axes(), [(0.5, 0.5), (1.0, 1.0)])

        # Without the comment, rows are read as bpm/sphere, or as atom_style= says.
        rows = [row.split("#")[0].strip() for row in bpm_rows(images=False)]  # no image flags or comment
        path = write(self.directory, rows, BONDS, style=None, name="plain.data")
        self.assertEqual(lammps.read_data(path).atom_style, "bpm/sphere")
        with self.assertRaisesRegex(ValueError, "8 values, but atom style 'sphere' has 7"):
            lammps.read_data(path, atom_style="sphere")
        with self.assertRaisesRegex(ValueError, "isn't one tangle.lammps reads"):
            lammps.read_data(path, atom_style="ellipsoid")

    def test_periodic_takes_the_axes_or_the_input_scripts_boundary(self):
        path = fibers_file(self.directory)
        for periodic, flags in [("p p f", [True, True, False]), ("boundary p f p", [True, False, True]),
                                ("xy", [True, True, False]), ("", [False] * 3), (True, [True] * 3),
                                ([False, True, False], [False, True, False])]:
            with self.subTest(periodic=periodic):
                self.assertEqual(lammps.read_data(path, periodic=periodic).cell.periodic, flags)
        with self.assertRaisesRegex(ValueError, "boundary"):
            lammps.read_data(path, periodic="ppf")

    def test_walls_hold_spheres_that_stick_out_of_the_box_no_farther_out(self):
        path = write(self.directory, bpm_rows(BY_THE_WALLS), WALL_BONDS)
        cell = lammps.read_data(path, periodic="p p f").cell
        for got, want in zip(cell.origin + cell.lengths, [0.0, 0.0, -0.3, 10.0, 10.0, 10.7]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(lammps.read_data(path).cell.origin, [0.0] * 3)  # periodic: the box

    def test_triclinic_boxes_and_capsule_files_are_refused(self):
        rows = bpm_rows()
        self.assertEqual(len(lammps.read_data(write(self.directory, rows, BONDS, header=["0 0 0 xy xz yz"]))
                             .fiber_atoms), 2)
        with self.assertRaisesRegex(ValueError, "triclinic"):
            lammps.read_data(write(self.directory, rows, BONDS, header=["0.5 0 0 xy xz yz"]))
        capsules = [["Capsules", "", *[f"{atom[0]} 0.5 1 0 0" for atom in ATOMS]]]
        with self.assertRaisesRegex(ValueError, "capsules"):
            lammps.read_data(write(self.directory, rows, BONDS, sections=capsules))

    def test_relax_needs_fibers(self):
        data = lammps.read_data(write(self.directory, bpm_rows([ATOMS[-1]])))
        with self.assertRaisesRegex(ValueError, "no fibers to relax"):
            data.relax()


class WriteDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.path = fibers_file(self.directory)
        self.data = lammps.read_data(self.path)
        self.lines = self.data.fibers.centerlines()

    def tearDown(self):
        self.temporary.cleanup()

    def written(self, lines, *, data=None):
        """How many atoms moved, and the rows that changed, from their old text to their new."""
        data = data or self.data
        out = self.directory / "relaxed.data"
        moved = data.write_data(out, lines)
        before, after = data.path.read_text().splitlines(), out.read_text().splitlines()
        self.assertEqual(len(before), len(after))
        return moved, {old: new for old, new in zip(before, after) if old != new}

    def test_only_the_rows_of_spheres_that_moved_change(self):
        self.lines[0][1][2] += 0.25  # atom 2 up
        self.lines[0][3][0] -= 0.5  # atom 4 from x 10 (written at 0, one box on) to 9.5
        self.assertEqual(self.written(self.lines), (2, {
            "2 1 1 1 1000 8 5 5 0 0 0 # second": "2 1 1 1 1000 8.0 5.0 5.25 0 0 0 # second",
            "4 1 1 1 1000 0 5 5 1 0 0": "4 1 1 1 1000 9.5 5.0 5.0 0 0 0",
        }))
        self.assertEqual(self.written(self.data.fibers), (0, {}))  # unmoved fibers leave the file as it was

    def test_spheres_go_back_into_the_box_and_their_image_flags_count_the_boxes(self):
        self.lines[0][4][0] = 21.5  # atom 5: two boxes on, at 1.5
        self.lines[0][0][0] = -3.0  # atom 1: one box back, at 7
        self.assertEqual(self.written(self.lines)[1], {
            "5 1 1 1 1000 1 5 5 1 0 0": "5 1 1 1 1000 1.5 5.0 5.0 2 0 0",
            "1 1 1 1 1000 7 5 5 0 0 0": "1 1 1 1 1000 7.0 5.0 5.0 -1 0 0",
        })

    def test_rows_without_image_flags_get_none(self):
        data = lammps.read_data(fibers_file(self.directory, images=False, name="plain.data"))
        lines = data.fibers.centerlines()
        lines[0][4][0] = 21.5
        self.assertEqual(self.written(lines, data=data)[1], {"5 1 1 1 1000 1 5 5": "5 1 1 1 1000 1.5 5.0 5.0"})

    def test_spheres_go_by_their_share_of_the_length_when_the_points_changed(self):
        # As adaptive segmentation leaves a fiber: fewer points than spheres, here bent up to z 6 at its middle.
        self.lines[0] = [[7.0, 5.0, 5.0], [9.0, 5.0, 6.0], [11.0, 5.0, 5.0]]
        out = self.directory / "relaxed.data"
        self.assertEqual(self.data.write_data(out, self.lines), 3)  # the ends stay
        text = out.read_text()
        for atom_id, point in [(2, [8.0, 5.0, 5.5]), (3, [9.0, 5.0, 6.0]), (4, [10.0, 5.0, 5.5])]:
            for got, want in zip(row_values(text, atom_id)[5:8], point):
                self.assertAlmostEqual(math.remainder(got - want, 10.0), 0.0, places=12)  # atom 4 at 0 or 10

    def test_line_endings_and_the_wrong_number_of_fibers(self):
        crlf = self.directory / "crlf.data"
        crlf.write_bytes(self.path.read_bytes().replace(b"\n", b"\r\n"))
        data = lammps.read_data(crlf)
        self.lines[0][1][2] += 0.25
        out = self.directory / "relaxed.data"
        self.assertEqual(data.write_data(out, self.lines), 1)
        written = out.read_bytes()
        self.assertEqual(written.count(b"\n"), written.count(b"\r\n"))
        self.assertEqual(written.replace(b"5.25", b"5").replace(b"8.0 5.0", b"8 5"), crlf.read_bytes())
        with self.assertRaisesRegex(ValueError, "1 relaxed fibers, but"):
            data.write_data(out, self.lines[:1])


@unittest.skipUnless(GPU, "Tangle's relaxation runs on the GPU; set TANGLE_BACKEND to run it here")
class RoundTripTests(unittest.TestCase):
    def test_the_relaxed_file_has_no_overlapping_fibers_and_the_same_bonds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fibers_file(directory)
            data = lammps.read_data(path)
            run = data.relax(tangle.RelaxationSettings(backend=BACKEND))
            self.assertTrue(run.converged)
            out = Path(directory) / "relaxed.data"
            self.assertGreater(data.write_data(out, run), 0)

            relaxed = lammps.read_data(out)  # written whole: each fiber is still one chain across the box
            self.assertEqual(relaxed.fiber_atoms, data.fiber_atoms)
            a, b = relaxed.fibers.centerlines()
            nearest = min(math.hypot(*[(q - p) - 10.0 * round((q - p) / 10.0) for p, q in zip(i, j)])
                          for i in a for j in b)
            self.assertGreater(nearest, 0.98)  # spheres 1 across, from half overlapping; 1% overlap is allowed
            for line in (a, b):
                for i, j in zip(line, line[1:]):
                    self.assertAlmostEqual(math.dist(i, j), 1.0, delta=0.05)  # the chains keep their spacing
            before, after = path.read_text().splitlines(), out.read_text().splitlines()
            bonds = before.index("Bonds")
            self.assertEqual(after[bonds:], before[bonds:])
            self.assertIn("11 3 3 0.5 1000 5 1 1 0 0 0", after)  # the lone sphere stays

    def test_fibers_by_a_wall_that_overlap_nothing_stay_where_they_are(self):
        with tempfile.TemporaryDirectory() as directory:
            data = lammps.read_data(write(directory, bpm_rows(BY_THE_WALLS), WALL_BONDS), periodic="p p f")
            run = data.relax(tangle.RelaxationSettings(backend=BACKEND))
            self.assertTrue(run.converged)
            self.assertEqual(data.write_data(Path(directory) / "relaxed.data", run), 0)

    def test_a_box_of_millions_of_cells_relaxes(self):
        # Millions of contact cells: more cubes in one launch than WebGPU allows along an axis.
        fibers = tangle.FiberCollection("crossing")
        material = tangle.Material("type 1", 1.0)
        fibers.add_fiber([[x, 400.0, 400.0] for x in range(396, 405)], material)
        fibers.add_fiber([[400.0, y, 400.5] for y in range(396, 405)], material)
        recipe = tangle.Recipe(tangle.Cell([800.0] * 3, "xyz"))
        recipe.insert(fibers)
        run = recipe.run(tangle.RelaxationSettings(backend=BACKEND))
        self.assertTrue(run.converged)
        self.assertLess(run.max_penetration, 0.011)


if __name__ == "__main__":
    unittest.main()
