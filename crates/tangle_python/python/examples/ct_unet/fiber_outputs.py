"""Writing the fibers find_fibers.py finds: tables, Tangle's fit.json, ParaView and OVITO files, TIFF stacks.

Lengths are in micrometers and coordinates are measured from the corner of the analysed region (the crop, if
one was given): x across the image (columns), y down it (rows), z through the slices. A voxel's center sits at
(column + 0.5, row + 0.5, slice + 0.5) voxels, as in tangle.ct. Fiber and bond numbers start at 1, as do
fiber types (1 = the thinnest); fit.json keeps Tangle's own conventions (meters, types from 0).
"""

from __future__ import annotations

import colorsys
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STACK_LIMIT = 600_000_000  # voxels: bigger regions skip the overlay and label stacks unless asked for
SPECK = 20  # voxels: binder pieces smaller than this are dropped as noise
BINDER_PARTICLES = 1_500_000  # at most this many binder spheres in fibers.dump (bigger blocks past it)
BEND_DIAMETERS = 5.0  # Tangle's default bend limit (tangle.ct.FiberSpec.min_bend_radius), in fiber diameters
OUTPUTS = ("summary.txt", "run.json", "fit.json", "fibers.csv", "centerlines.csv", "bonds.csv", "diameters.csv",
           "fibers.vtk", "bonds.vtk", "fibers.dump", "view_in_ovito.py", "overlay.png", "overlay.tif", "labels.tif",
           "binder.tif", "overlay.npy", "labels.npy", "binder.npy", "overlay.partial.npy", "labels.partial.npy",
           "binder.partial.npy", "binder_uncleaned.npy")


def clear_outputs(out: Path) -> None:
    """Delete the files an earlier run wrote to ``out`` (not ``work/``), so none outlives the fibers it shows."""
    from scan_io import ScanError

    for name in OUTPUTS:
        try:
            (out / name).unlink(missing_ok=True)
        except OSError as error:
            raise ScanError(f"Can't replace {out / name} ({error.strerror}): close it in any program that has it "
                            "open, or write the results elsewhere with --out") from None


@dataclass
class Fibers:
    """Traced fibers in the analysed region's voxels; see the module notes for the coordinates."""

    lines: list  # (n, 3) arrays: x, y, z in voxels
    radii: np.ndarray  # voxels
    types: np.ndarray  # 0 = the thinnest type
    support: np.ndarray  # share of each trace backed by the network's axis votes
    bonds: list  # {"fibers": (i, j) indices into lines, "center": (x, y, z) voxels, "strength": peak height (NaN for
    #              a bond found through binder alone, which has "voxels": the binder voxels joining the pair)}
    voxel_um: float
    shape: tuple  # (z, y, x) voxels of the analysed region
    type_names: list  # one per type, e.g. "12 um"
    profiles: list | None = None  # diameters.Accumulator.result, in micrometers
    binder: np.ndarray | None = None  # (n, 3) voxels (x, y, z) the network calls binder, for a binder-aware network

    @property
    def count(self) -> int:
        return len(self.lines)

    def lengths_um(self) -> np.ndarray:
        return np.array([polyline_length(line) for line in self.lines]) * self.voxel_um

    def diameters_um(self) -> np.ndarray:
        return 2.0 * np.asarray(self.radii, float) * self.voxel_um


def polyline_length(line) -> float:
    line = np.asarray(line, float)
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) if len(line) > 1 else 0.0


def resample(line, spacing: float) -> np.ndarray:
    """Points evenly spaced along a polyline, at most ``spacing`` apart, ends kept."""
    line = np.asarray(line, float)
    if len(line) < 2:
        return line.copy()
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    n = max(int(np.ceil(s[-1] / spacing)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def within_bend_limit(points, bend: float, sweeps: int = 2000) -> tuple[np.ndarray, bool]:
    """A copy of the polyline ``points`` smoothed wherever it turns tighter than a bend radius of ``bend`` (in
    the same units), so that Tangle accepts it as a fiber's shape. The turn is measured as Tangle does:
    2 sin(angle / 2) over the mean of the two segments, at every inner point. Points over the limit move
    halfway to the middle of their neighbours, sweep after sweep; after ``sweeps``, every inner point does.
    The ends stay put. Returns the points and whether all of them are now within the limit."""
    p = np.array(points, dtype=float)
    if len(p) < 3:
        return p, True
    limit = 0.98 / bend  # a little inside the limit, so rounding to meters can't tip a point over it
    for sweep in range(11 * sweeps):
        before, point, after = p[:-2], p[1:-1], p[2:]
        u, v = point - before, after - point
        lu, lv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        cosine = np.clip((u * v).sum(1) / np.maximum(lu * lv, 1e-12), -1.0, 1.0)
        over = 2.0 * np.sin(0.5 * np.arccos(cosine)) > limit * 0.5 * (lu + lv)
        if not over.any():
            return p, True
        if sweep >= sweeps:  # a long stretch over the limit: smooth the whole line
            over[:] = True
        point[over] += 0.5 * (0.5 * (before + after)[over] - point[over])  # point is a view into p
    return p, False


def axis_of(line) -> np.ndarray:
    """A fiber's overall direction: the leading axis of its length-weighted tangent tensor (sign-free)."""
    d = np.diff(np.asarray(line, float), axis=0)
    length = np.linalg.norm(d, axis=1)
    if not len(d) or length.sum() == 0:
        return np.array([1.0, 0.0, 0.0])
    t = d / np.maximum(length[:, None], 1e-12)
    tensor = (length[:, None, None] * t[:, :, None] * t[:, None, :]).sum(0)
    return np.linalg.eigh(tensor)[1][:, -1]


def fiber_palette(count: int, seed: int = 0) -> np.ndarray:
    """``count + 1`` RGB colors in [0, 1]; row 0 (no fiber) is black. Golden-ratio hues, as tangle.ct uses,
    so fibers with neighbouring numbers differ strongly."""
    start = np.random.default_rng(seed).uniform()
    colors = [(0.0, 0.0, 0.0)]
    for i in range(count):
        hue = (start + 0.61803398875 * i) % 1.0
        colors.append(colorsys.hsv_to_rgb(hue, 0.65 + 0.3 * ((i * 7) % 3) / 2, 0.95 - 0.2 * ((i * 5) % 2)))
    return np.asarray(colors)


# -- tables --------------------------------------------------------------------------------------------------------


def fiber_rows(fibers: Fibers) -> tuple[list[str], list[list]]:
    """fibers.csv: one row per fiber."""
    h = fibers.voxel_um
    upper = np.array(fibers.shape[::-1], float)
    header = ["fiber", "type", "diameter_um", "length_um", "support", "straightness", "out_of_plane_deg",
              "in_plane_deg", "touches_edge", "start_x_um", "start_y_um", "start_z_um", "end_x_um", "end_y_um",
              "end_z_um"]
    if fibers.profiles is not None:
        header += ["diameter_median_um", "diameter_spread_um", "long_um", "short_um"]
    rows = []
    for k, line in enumerate(fibers.lines):
        line = np.asarray(line, float)
        length = polyline_length(line)
        chord = float(np.linalg.norm(line[-1] - line[0]))
        t = axis_of(line)
        margin = max(1.5 * float(fibers.radii[k]), 2.0)
        ends = line[[0, -1]]
        edge = bool(np.any(ends < margin) or np.any(ends > upper - margin))
        row = [k + 1, int(fibers.types[k]) + 1, _f(2 * fibers.radii[k] * h), _f(length * h),
               _f(fibers.support[k], 3), _f(chord / length if length > 0 else 1.0, 3),
               _f(np.degrees(np.arcsin(min(abs(t[2]), 1.0))), 1),
               _f(np.degrees(np.arctan2(t[1], t[0])) % 180.0, 1), int(edge),
               *[_f(v * h) for v in line[0]], *[_f(v * h) for v in line[-1]]]
        if fibers.profiles is not None:
            p = fibers.profiles[k]
            ok = np.isfinite(p["diameter"])
            stats = [np.median(p["diameter"][ok]), np.std(p["diameter"][ok]), np.median(p["long"][ok]),
                     np.median(p["short"][ok])] if ok.any() else [np.nan] * 4
            row += [_f(v) for v in stats]
        rows.append(row)
    return header, rows


def _f(value, digits: int = 4) -> str:
    value = float(value)
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else str(value)


def write_csv(path: Path, header: list[str], rows) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_tables(fibers: Fibers, out: Path) -> list[Path]:
    """fibers.csv, centerlines.csv, bonds.csv (when there are bonds), diameters.csv (with the profiles)."""
    h = fibers.voxel_um
    paths = [write_csv(out / "fibers.csv", *fiber_rows(fibers))]
    paths.append(_write_points(out / "centerlines.csv", "fiber,point,x_um,y_um,z_um",
                               [np.asarray(line, float) * h for line in fibers.lines]))
    if fibers.bonds:
        paths.append(write_csv(out / "bonds.csv", ["bond", "fiber_a", "fiber_b", "x_um", "y_um", "z_um", "strength",
                                                   "binder_voxels"],
                               ([n + 1, b["fibers"][0] + 1, b["fibers"][1] + 1, *[_f(v * h) for v in b["center"]],
                                 _f(b["strength"], 3) if np.isfinite(b["strength"]) else "", b.get("voxels", 0)]
                                for n, b in enumerate(fibers.bonds))))
    if fibers.profiles is not None:
        paths.append(_write_points(out / "diameters.csv", "fiber,node,x_um,y_um,z_um,diameter_um,long_um,short_um",
                                   [np.column_stack([p["nodes"] * h, p["diameter"], p["long"], p["short"]])
                                    for p in fibers.profiles]))
    return paths


def _write_points(path: Path, header: str, per_fiber: list) -> Path:
    """A CSV with one row per point: fiber number, point number along it, then the columns of ``per_fiber``."""
    rows = [np.column_stack([np.full(len(a), k + 1), np.arange(len(a)), a]) for k, a in enumerate(per_fiber) if len(a)]
    columns = len(header.split(","))
    table = np.concatenate(rows) if rows else np.zeros((0, columns))
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt=["%d", "%d"] + ["%.4f"] * (columns - 2))
    return path


def write_fit_json(fibers: Fibers, path: Path, levels: dict, history: list[dict],
                   given_um: list | None = None) -> tuple[Path, list[str]]:
    """The fibers in tangle.ct's fit.json format (meters), so ``tangle.ct.load_fit`` reads them back and
    ``FitResult.relax`` solves them. Returns the path and lines for the summary.

    Only the types that have fibers are listed (Tangle can't make a material or population from an empty
    type), numbered from 0 in order of size; each keeps its number from fibers.csv in its name. A type's
    diameter is the one given (``given_um``), else the median of its fibers.

    Tangle refuses a fiber whose shape bends tighter than its type's bend limit (5 diameters by default), and
    traces wiggle by a fraction of a voxel, so each centerline here is resampled one radius of its type apart
    (with finer nodes on thick fibers, Tangle's solve takes longer to settle) and smoothed where it bends
    tighter than the limit (``within_bend_limit``). The tables, ParaView and OVITO files and the stacks keep
    the traced lines."""
    h = fibers.voxel_um * 1e-6
    diameters = 2.0 * np.asarray(fibers.radii, float) * h
    present = [t for t in range(len(fibers.type_names)) if np.any(fibers.types == t)]
    specs = []
    for t in present:
        d = given_um[t] * 1e-6 if given_um and t < len(given_um) else float(np.median(diameters[fibers.types == t]))
        specs.append({"diameter": float(d), "name": f"type {t + 1} ({fibers.type_names[t]})"})
    if not specs:
        specs = [{"diameter": given_um[0] * 1e-6 if given_um else 1e-5, "name": "fiber"}]
    number = {t: k for k, t in enumerate(present)}
    spec_vox = np.array([s["diameter"] for s in specs]) / h
    lines, shift, moved, tight = [], 0.0, 0, []
    for k, (line, t) in enumerate(zip(fibers.lines, fibers.types)):
        d = float(spec_vox[number.get(int(t), 0)])
        start = resample(line, max(1.0, 0.5 * d))
        smooth, ok = within_bend_limit(start, BEND_DIAMETERS * d)
        step = float(np.max(np.linalg.norm(smooth - start, axis=1))) if len(start) else 0.0
        shift = max(shift, step)
        moved += int(step > 0.25)
        if not ok:
            tight.append(k + 1)
        lines.append(smooth)
    notes = [f"fit.json: centerlines smoothed where they bend tighter than Tangle's bend limit ({BEND_DIAMETERS:g} "
             f"diameters); {moved:,} fibers moved more than a quarter voxel, by at most {shift * fibers.voxel_um:.3g} um"]
    if tight:
        notes.append(f"  {len(tight)} fibers still bend tighter than the limit, so Tangle will refuse fit.json: "
                     + ", ".join(map(str, tight[:10])) + (", ..." if len(tight) > 10 else "") + " in fibers.csv")
    history = history + [{"stage": "smoothed to Tangle's bend limit", "bend_limit_diameters": BEND_DIAMETERS,
                          "node_spacing": "one radius of the fiber's type",
                          "largest_shift_um": round(shift * fibers.voxel_um, 4), "fibers_over_limit": tight}]
    data = {
        "schema": "tangle.ct.fit/1",
        "units": "meters",
        "source": "ct_unet/find_fibers.py",
        "voxel_size": h,
        "shape_zyx": [int(n) for n in fibers.shape],
        "cell_lengths": [int(n) * h for n in fibers.shape[::-1]],
        "spec": specs[0],
        "levels": levels,
        "specs": specs if len(specs) > 1 else None,
        "fibers": [{"id": k + 1, "diameter": float(d), "support": float(s), "type": number.get(int(t), 0),
                    "centerline": np.round(line * h, 10).tolist()}
                   for k, (line, d, s, t) in enumerate(zip(lines, diameters, fibers.support, fibers.types))],
        "bonds": [{"fibers": [b["fibers"][0] + 1, b["fibers"][1] + 1],
                   "center": np.round(np.asarray(b["center"], float) * h, 10).tolist(),
                   "strength": round(float(b["strength"]), 4) if np.isfinite(b["strength"]) else None,
                   "binder_voxels": int(b.get("voxels", 0))} for b in fibers.bonds],
        "history": history,
    }
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path, notes


# -- ParaView and OVITO ----------------------------------------------------------------------------------------------


def write_vtk(fibers: Fibers, out: Path) -> list[Path]:
    """fibers.vtk (polylines with per-fiber number, type and diameter, per-point radius) and bonds.vtk."""
    h = fibers.voxel_um
    lines = [np.asarray(line, float) for line in fibers.lines]
    points = np.concatenate(lines) * h if lines else np.zeros((0, 3))
    path = out / "fibers.vtk"
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nfibers found by find_fibers.py (micrometers)\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        np.savetxt(f, points, fmt="%.4f")
        f.write(f"LINES {len(lines)} {len(points) + len(lines)}\n")
        start = 0
        for line in lines:
            f.write(" ".join(map(str, [len(line), *range(start, start + len(line))])) + "\n")
            start += len(line)
        if lines:
            f.write(f"CELL_DATA {len(lines)}\n")
            for name, values, fmt in (("fiber", np.arange(1, len(lines) + 1), "%d"),
                                      ("type", np.asarray(fibers.types) + 1, "%d"),
                                      ("diameter_um", fibers.diameters_um(), "%.4f"),
                                      ("support", np.asarray(fibers.support), "%.3f")):
                f.write(f"SCALARS {name} {'int' if fmt == '%d' else 'float'} 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, values, fmt=fmt)
            f.write(f"POINT_DATA {len(points)}\n")
            each = [len(line) for line in lines]
            for name, values, fmt in (("radius_um", np.repeat(0.5 * fibers.diameters_um(), each), "%.4f"),
                                      ("fiber_id", np.repeat(np.arange(1, len(lines) + 1), each), "%d")):
                f.write(f"SCALARS {name} {'int' if fmt == '%d' else 'float'} 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, values, fmt=fmt)
    paths = [path]
    if fibers.bonds:
        centers = np.array([b["center"] for b in fibers.bonds], float) * h
        path = out / "bonds.vtk"
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0\nbonds found by find_fibers.py (micrometers)\nASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {len(centers)} float\n")
            np.savetxt(f, centers, fmt="%.4f")
            f.write(f"VERTICES {len(centers)} {2 * len(centers)}\n")
            np.savetxt(f, np.stack([np.ones(len(centers), int), np.arange(len(centers))], 1), fmt="%d")
            f.write(f"POINT_DATA {len(centers)}\n")
            for name, values, fmt in (("strength", [b["strength"] for b in fibers.bonds], "%.3f"),
                                      ("binder_voxels", [b.get("voxels", 0) for b in fibers.bonds], "%d"),
                                      ("fiber_a", [b["fibers"][0] + 1 for b in fibers.bonds], "%d"),
                                      ("fiber_b", [b["fibers"][1] + 1 for b in fibers.bonds], "%d")):
                f.write(f"SCALARS {name} {'int' if fmt == '%d' else 'float'} 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, np.asarray(values), fmt=fmt)
        paths.append(path)
    return paths


def _z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternions (i, j, k, w) turning the z axis onto each unit row of ``direction`` (as Tangle's OVITO export)."""
    q = np.stack([-direction[:, 1], direction[:, 0], np.zeros(len(direction)), 1.0 + direction[:, 2]], axis=1)
    flipped = direction[:, 2] < -1.0 + 1e-12
    q[flipped] = [1.0, 0.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


OVITO_SCRIPT = '''\
# Open the fibers in OVITO: run "ovitos view_in_ovito.py", then open fibers.ovito in OVITO.
# (Or load fibers.dump in OVITO directly and set the particle shape to Spherocylinder.)
# Each fiber is a chain of capsules (type 1); the binder is spheres of type 2 (orange; OVITO's Construct surface
# mesh on type 2 turns it into a surface) and bonds are type 3 (yellow). Lengths are micrometers.
from pathlib import Path

import ovito
from ovito.io import import_file
from ovito.modifiers import AssignColorModifier, ColorCodingModifier, ExpressionSelectionModifier
from ovito.vis import ParticlesVis

HERE = Path(__file__).resolve().parent
pipeline = import_file(str(HERE / "fibers.dump"), sort_particles=True)
pipeline.modifiers.append(ColorCodingModifier(property="Molecule Identifier"))  # one color per fiber
# pipeline.modifiers.append(ColorCodingModifier(property="fiber_type"))  # or one color per fiber type
{extras}pipeline.add_to_scene()
data = pipeline.compute()
data.particles.vis.shape = ParticlesVis.Shape.Spherocylinder
data.cell.vis.enabled = True
session = HERE / "fibers.ovito"
ovito.scene.save(str(session))
print(f"Saved {{session}}: open it in OVITO")
'''

OVITO_BINDER = '''\
pipeline.modifiers.append(ExpressionSelectionModifier(expression="ParticleType == 2"))  # binder in orange
pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.55, 0.0)))
'''

OVITO_BONDS = '''\
pipeline.modifiers.append(ExpressionSelectionModifier(expression="ParticleType == 3"))  # bonds in yellow
pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.85, 0.1)))
'''


def write_ovito(fibers: Fibers, out: Path) -> list[Path]:
    """fibers.dump (LAMMPS dump of capsules, one per centerline segment, binder spheres and bond spheres) and
    view_in_ovito.py. Each binder voxel is a sphere as big as its voxel; past ``BINDER_PARTICLES`` of them, one
    sphere stands for each k x k x k block holding binder, with the smallest k that keeps under it."""
    h = fibers.voxel_um
    rows = []
    for k, line in enumerate(fibers.lines):
        line = np.asarray(line, float) * h
        if len(line) < 2:
            continue
        a, b = line[:-1], line[1:]
        length = np.linalg.norm(b - a, axis=1)
        unit = (b - a) / np.maximum(length[:, None], 1e-12)
        r = float(fibers.radii[k]) * h
        n = len(a)
        rows.append(np.column_stack([np.full(n, k + 1), np.ones(n), np.full(n, r), np.full(n, r), length, _z_to(unit),
                                     0.5 * (a + b), np.full(n, int(fibers.types[k]) + 1)]))
    if fibers.binder is not None and len(fibers.binder):
        cells, k = np.asarray(fibers.binder, np.int64), 1
        while len(cells) > BINDER_PARTICLES:
            k += 1
            cells = np.unique(np.asarray(fibers.binder, np.int64) // k, axis=0)
        m, r = len(cells), 0.62 * k * h  # a sphere with the volume of the k-voxel cube it stands for
        rows.append(np.column_stack([np.zeros(m), np.full(m, 2), np.full(m, r), np.full(m, r), np.zeros((m, 4)),
                                     np.ones(m), (cells + 0.5) * k * h, np.zeros(m)]))
    if fibers.bonds:
        radii = np.asarray(fibers.radii, float)
        for b in fibers.bonds:
            r = 0.6 * h * min(radii[b["fibers"][0]], radii[b["fibers"][1]])
            rows.append(np.array([[0, 3, r, r, 0.0, 0, 0, 0, 1, *(np.asarray(b["center"], float) * h), 0]]))
    table = np.concatenate(rows) if rows else np.zeros((0, 13))
    upper = np.array(fibers.shape[::-1], float) * h
    path = out / "fibers.dump"
    with open(path, "w") as f:
        f.write(f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(table)}\nITEM: BOX BOUNDS ff ff ff\n")
        for axis in range(3):
            f.write(f"0 {upper[axis]:.6g}\n")
        f.write("ITEM: ATOMS id mol type AsphericalShape.X AsphericalShape.Y AsphericalShape.Z quati quatj quatk quatw "
                "x y z fiber_type\n")
        ids = np.arange(1, len(table) + 1)
        np.savetxt(f, np.column_stack([ids, table]),
                   fmt=["%d", "%d", "%d"] + ["%.5g"] * 3 + ["%.6f"] * 4 + ["%.5f"] * 3 + ["%d"])
    script = out / "view_in_ovito.py"
    binder = fibers.binder is not None and len(fibers.binder) > 0
    script.write_text(OVITO_SCRIPT.format(extras=(OVITO_BINDER if binder else "") + (OVITO_BONDS if fibers.bonds else "")))
    return [path, script]


# -- drawing the fibers into voxels ----------------------------------------------------------------------------------


def segments(lines, radii):
    """Every fiber as short capsules: (starts, ends) in x, y, z voxels, radii, and 1-based fiber numbers."""
    a, b, r, f = [], [], [], []
    for k, (line, radius) in enumerate(zip(lines, radii)):
        p = resample(line, float(np.clip(radius, 1.5, 6.0)))  # pieces about a radius long draw fastest
        if len(p) == 1:
            p = np.vstack([p, p])
        a.append(p[:-1])
        b.append(p[1:])
        r.append(np.full(len(p) - 1, float(radius)))
        f.append(np.full(len(p) - 1, k + 1, dtype=np.int32))
    if not a:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int32)
    return np.concatenate(a), np.concatenate(b), np.concatenate(r), np.concatenate(f)


def draw(pieces, box, margin: float = 0.0):
    """Which fiber each voxel of ``box`` = (z0, z1, y0, y1, x0, x1) belongs to.

    Returns (fiber, depth) as (z, y, x) arrays: ``depth`` is the distance from the voxel center to the nearest
    fiber axis minus that fiber's radius (negative inside a fiber), for the fiber the voxel is deepest inside;
    ``fiber`` is that fiber's 1-based number. Voxels further than ``margin`` voxels outside every fiber get 0
    and +inf. Where two drawn fibers overlap, each voxel goes to the one it is deeper inside."""
    a, b, r, f = pieces
    z0, z1, y0, y1, x0, x1 = (int(v) for v in box)
    nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
    depth = np.full(nz * ny * nx, np.inf, np.float32)
    fiber = np.zeros(nz * ny * nx, np.int32)
    if len(a):
        reach = (r + margin + 1.0)[:, None]
        lo_box, hi_box = np.array([x0, y0, z0]), np.array([x1, y1, z1])
        # the voxels within reach of each piece, clipped to the box (a thin slab computes only its own slices)
        low = np.maximum(np.floor(np.minimum(a, b) - reach).astype(np.int64), lo_box)
        high = np.minimum(np.ceil(np.maximum(a, b) + reach).astype(np.int64), hi_box - 1)
        near = np.flatnonzero(np.all(high >= low, axis=1))
        extent = high[near] - low[near] + 1
        extent = np.where(extent > 4, (extent + 3) // 4 * 4, extent)  # rounded up: fewer grid shapes to loop over
        shape_key = (extent[:, 0] * 1024 + extent[:, 1]) * 1024 + extent[:, 2]
        for key in np.unique(shape_key):
            group = shape_key == key
            members = near[group]
            sx, sy, sz = extent[group][0]
            grid = np.stack(np.meshgrid(np.arange(sx), np.arange(sy), np.arange(sz), indexing="ij"), -1).reshape(-1, 3)
            chunk = max(1, int(1_000_000 // len(grid)))
            for start in range(0, len(members), chunk):
                m = members[start:start + chunk]
                vox = low[m][:, None, :] + grid[None]  # (pieces, offsets, xyz) voxel indices
                inside = np.all(vox < hi_box, axis=2)  # a rounded-up grid can run past the box
                ab = (b[m] - a[m])[:, None, :]
                ap = vox + 0.5 - a[m][:, None, :]
                t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
                d = np.linalg.norm(ap - t[..., None] * ab, axis=-1) - r[m][:, None]
                keep = inside & (d <= margin)
                if not keep.any():
                    continue
                piece, offset = np.nonzero(keep)
                v = vox[piece, offset]
                flat = ((v[:, 2] - z0) * ny + (v[:, 1] - y0)) * nx + (v[:, 0] - x0)
                dd = d[piece, offset].astype(np.float32)
                ff = f[m][piece]
                order = np.lexsort((dd, flat))  # by voxel, deepest first
                flat, dd, ff = flat[order], dd[order], ff[order]
                first = np.r_[True, flat[1:] != flat[:-1]]
                flat, dd, ff = flat[first], dd[first], ff[first]
                better = dd < depth[flat]
                depth[flat[better]] = dd[better]
                fiber[flat[better]] = ff[better]
    return fiber.reshape(nz, ny, nx), depth.reshape(nz, ny, nx)


def _edges(labels: np.ndarray) -> np.ndarray:
    """Voxels whose in-plane (last two axes) neighbour has another label."""
    edge = np.zeros(labels.shape, bool)
    differ = labels[..., 1:, :] != labels[..., :-1, :]
    edge[..., 1:, :] |= differ
    edge[..., :-1, :] |= differ
    differ = labels[..., :, 1:] != labels[..., :, :-1]
    edge[..., :, 1:] |= differ
    edge[..., :, :-1] |= differ
    return edge


BOND_COLOR = np.array([255.0, 220.0, 40.0])
BINDER_COLOR = np.array([255.0, 140.0, 0.0])


def overlay_rgb(grey: np.ndarray, labels: np.ndarray, palette: np.ndarray, display, alpha: float = 0.35,
                bonds: np.ndarray | None = None, binder: np.ndarray | None = None) -> np.ndarray:
    """uint8 RGB of grey slices with each fiber tinted in its own color and outlined, binder voxels outside the
    fibers in orange and bond voxels in yellow."""
    low, high = display
    g = np.clip((np.asarray(grey, np.float32) - low) / max(high - low, 1e-12), 0.0, 1.0) * 255.0
    rgb = np.repeat(np.nan_to_num(g)[..., None], 3, axis=-1)
    inside = labels > 0
    color = palette[labels[inside]]
    rgb[inside] = (1.0 - alpha) * rgb[inside] + alpha * color
    edge = inside & _edges(labels)
    rgb[edge] = 0.75 * palette[labels[edge]] + 0.25 * 255.0
    if binder is not None:
        glue = binder & ~inside
        rgb[glue] = 0.3 * rgb[glue] + 0.7 * BINDER_COLOR
    if bonds is not None:
        rgb[bonds] = BOND_COLOR
    return (rgb + 0.5).astype(np.uint8)


def binder_mask(binder: np.ndarray | None, box) -> np.ndarray | None:
    """The voxels of ``box`` (z0, z1, y0, y1, x0, x1) among ``binder`` ((n, 3) voxels x, y, z), or None without
    binder."""
    if binder is None:
        return None
    z0, z1, y0, y1, x0, x1 = box
    mask = np.zeros((z1 - z0, y1 - y0, x1 - x0), bool)
    v = binder[(binder[:, 2] >= z0) & (binder[:, 2] < z1) & (binder[:, 1] >= y0) & (binder[:, 1] < y1)
               & (binder[:, 0] >= x0) & (binder[:, 0] < x1)].astype(int)
    mask[v[:, 2] - z0, v[:, 1] - y0, v[:, 0] - x0] = True
    return mask


def bond_mask(centers: np.ndarray, box, radius: float = 1.5) -> np.ndarray | None:
    """Voxels of ``box`` within ``radius`` of a bond center (x, y, z voxels), or None without bonds."""
    if centers is None or not len(centers):
        return None
    z0, z1, y0, y1, x0, x1 = box
    mask = np.zeros((z1 - z0, y1 - y0, x1 - x0), bool)
    near = centers[np.all((centers > np.array([x0, y0, z0]) - radius - 1) &
                          (centers < np.array([x1, y1, z1]) + radius + 1), axis=1)]
    span = np.arange(-int(np.ceil(radius)) - 1, int(np.ceil(radius)) + 2)
    for c in near:
        base = np.floor(c).astype(int)
        z, y, x = np.meshgrid(base[2] + span, base[1] + span, base[0] + span, indexing="ij")
        hit = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2 + (z + 0.5 - c[2]) ** 2 <= radius ** 2) & \
            (z >= z0) & (z < z1) & (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
        mask[z[hit] - z0, y[hit] - y0, x[hit] - x0] = True
    return mask


def open_stack(path: Path, shape, dtype, voxel_um: float, rgb: bool = False):
    """A TIFF stack to fill slab by slab, in ImageJ format with the voxel size so Fiji shows micrometers.
    Returns (array, path, finish); call ``finish()`` once filled. Without tifffile it is a .npy file."""
    try:
        import tifffile
    except ImportError:
        path = path.with_suffix(".npy")
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        return array, path, array.flush
    options = dict(imagej=True, photometric="rgb" if rgb else "minisblack", resolution=(1.0 / voxel_um, 1.0 / voxel_um),
                   metadata={"spacing": voxel_um, "unit": "um", "axes": "ZYXS" if rgb else "ZYX"})
    try:
        array = tifffile.memmap(path, shape=shape, dtype=dtype, **options)
        return array, path, array.flush
    except Exception:  # an older tifffile: fill a .npy, then write the TIFF from it
        scratch = path.with_suffix(".partial.npy")
        array = np.lib.format.open_memmap(scratch, mode="w+", dtype=dtype, shape=shape)

        def finish():
            array.flush()
            tifffile.imwrite(path, np.load(scratch, mmap_mode="r"), **options)
            scratch.unlink()

        return array, path, finish


def write_stacks(fibers: Fibers, out: Path, region, display, overlay: bool = True, labels: bool = True,
                 binder_level: float | None = None, say=print) -> tuple[list[Path], dict]:
    """overlay.tif (the scan with each traced fiber tinted, the network's binder in orange, bonds in yellow),
    labels.tif (each voxel's fiber number, 0 outside) and, given ``binder_level`` (the grey above which a voxel is
    solid), binder.tif: the network's binder (``fibers.binder``) outside every traced fiber or, for a network
    without binder, solid voxels outside every traced fiber with a 1-voxel margin so fiber edges don't count;
    either way pieces under ``SPECK`` voxels are dropped (fibers always win). Returns the paths and volume
    fractions."""
    nz, ny, nx = (int(n) for n in fibers.shape)
    n = fibers.count
    pieces = segments(fibers.lines, fibers.radii)
    palette = 255.0 * fiber_palette(n)
    centers = np.array([b["center"] for b in fibers.bonds], float) if fibers.bonds else None
    net = fibers.binder if fibers.binder is not None else None
    if net is not None:
        net = net[np.argsort(net[:, 2], kind="stable")]  # by slice, to pick out each slab's binder quickly
    margin = 1.0 if binder_level is not None and net is None else 0.0
    opened, paths = {}, []
    if overlay:
        opened["overlay"] = open_stack(out / "overlay.tif", (nz, ny, nx, 3), np.uint8, fibers.voxel_um, rgb=True)
    if labels:
        dtype = np.uint16 if n < 65535 else np.float32
        opened["labels"] = open_stack(out / "labels.tif", (nz, ny, nx), dtype, fibers.voxel_um)
    raw = None
    if binder_level is not None:
        raw_path = out / "binder_uncleaned.npy"
        raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint8, shape=(nz, ny, nx))
    slab = max(1, int(8_000_000 // max(ny * nx, 1)))
    inside_count, binder_count, last = 0, 0, time.time()
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        box = (z0, z1, 0, ny, 0, nx)
        fiber, depth = draw(pieces, box, margin)
        inside = depth <= 0
        number = np.where(inside, fiber, 0)
        inside_count += int(inside.sum())
        if "labels" in opened:
            opened["labels"][0][z0:z1] = number
        glue = None
        if net is not None:
            lo, hi = np.searchsorted(net[:, 2], [z0, z1])
            glue = binder_mask(net[lo:hi], box)
            binder_count += int((glue & ~inside).sum())
        grey = np.asarray(region[z0:z1], np.float32) if overlay or (raw is not None and net is None) else None
        if "overlay" in opened:
            opened["overlay"][0][z0:z1] = overlay_rgb(grey, number, palette, display, bonds=bond_mask(centers, box),
                                                      binder=glue)
        if raw is not None:
            raw[z0:z1] = (glue & ~inside) if net is not None else (grey > binder_level) & ~(depth <= margin)
        if time.time() - last > 10:
            last = time.time()
            say(f"  drawing the fibers into the stacks: {100 * z1 / nz:.0f}%")
    fractions = {"fibers": inside_count / float(nz * ny * nx)}
    if net is not None:
        fractions["binder"] = binder_count / float(nz * ny * nx)
    for name in ("overlay", "labels"):
        if name in opened:
            array, path, finish = opened.pop(name)
            finish()
            del array
            paths.append(path)
    if raw is not None:
        raw.flush()
        binder, path, finish = open_stack(out / "binder.tif", (nz, ny, nx), np.uint8, fibers.voxel_um)
        fractions["binder"] = remove_specks(raw, binder, say=say) / float(nz * ny * nx)
        finish()
        del binder, raw
        raw_path.unlink()
        paths.append(path)
    return paths, fractions


def remove_specks(raw, target, smallest: int = SPECK, block: int = 256, say=None) -> int:
    """Copy the 0/1 mask ``raw`` into ``target`` as 0/255 without its pieces under ``smallest`` voxels
    (6-connected); returns the voxels kept. Block by block, each read with ``smallest`` voxels more on every
    side: a piece that small can't reach further, and a bigger piece cut by the edge still shows more than
    ``smallest`` voxels, so every size is judged right."""
    from scipy import ndimage

    shape = raw.shape
    starts = [range(0, n, block) for n in shape]
    total, done, kept, last = len(starts[0]) * len(starts[1]) * len(starts[2]), 0, 0, time.time()
    for z0 in starts[0]:
        for y0 in starts[1]:
            for x0 in starts[2]:
                core = tuple(slice(s, min(s + block, n)) for s, n in zip((z0, y0, x0), shape))
                read = tuple(slice(max(0, c.start - smallest), min(n, c.stop + smallest)) for c, n in zip(core, shape))
                pieces, _ = ndimage.label(np.asarray(raw[read]) > 0)
                small = np.bincount(pieces.ravel()) < smallest
                small[0] = True
                keep = ~small[pieces[tuple(slice(c.start - r.start, c.stop - r.start) for c, r in zip(core, read))]]
                target[core] = keep.astype(np.uint8) * 255
                kept += int(keep.sum())
                done += 1
                if say and time.time() - last > 10:
                    last = time.time()
                    say(f"  cleaning up the binder: {100 * done / total:.0f}%")
    return kept


def write_preview(fibers: Fibers, path: Path, region, display, title: str) -> Path | None:
    """overlay.png: the middle slice in each direction, the scan above and the traced fibers below.
    None when matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    nz, ny, nx = (int(n) for n in fibers.shape)
    zc, yc, xc = nz // 2, ny // 2, nx // 2
    pieces = segments(fibers.lines, fibers.radii)
    palette = 255.0 * fiber_palette(fibers.count)
    centers = np.array([b["center"] for b in fibers.bonds], float) if fibers.bonds else None
    views = [
        (f"slice z = {zc} (x across, y down)", (zc, zc + 1, 0, ny, 0, nx), lambda a: a[0]),
        (f"row y = {yc} (x across, z down)", (0, nz, yc, yc + 1, 0, nx), lambda a: a[:, 0, :]),
        (f"column x = {xc} (y across, z down)", (0, nz, 0, ny, xc, xc + 1), lambda a: a[:, :, 0]),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9.5), squeeze=False)
    for col, (name, box, flat) in enumerate(views):
        z0, z1, y0, y1, x0, x1 = box
        grey = flat(np.asarray(region[z0:z1, y0:y1, x0:x1], np.float32))
        fiber, depth = draw(pieces, box)
        number = flat(np.where(depth <= 0, fiber, 0))
        marks = bond_mask(centers, box)
        glue = binder_mask(fibers.binder, box)
        image = overlay_rgb(grey, number, palette, display, bonds=None if marks is None else flat(marks),
                            binder=None if glue is None else flat(glue))
        low, high = display
        axes[0, col].imshow(np.clip((grey - low) / max(high - low, 1e-12), 0, 1), cmap="gray", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[1, col].imshow(image, interpolation="nearest")
        axes[0, col].set_title(name, fontsize=10)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


# -- summary -----------------------------------------------------------------------------------------------------------


def summary(fibers: Fibers, given_um: list | None, fractions: dict | None) -> list[str]:
    """Plain-language statistics of the traced fibers."""
    lines = []
    n = fibers.count
    if n == 0:
        return ["No fibers were found."]
    d = fibers.diameters_um()
    length = fibers.lengths_um()
    upper = np.array(fibers.shape[::-1], float)
    whole = np.array([not (np.any(np.asarray(l)[[0, -1]] < max(1.5 * r, 2)) or
                           np.any(np.asarray(l)[[0, -1]] > upper - max(1.5 * r, 2)))
                      for l, r in zip(fibers.lines, fibers.radii)])
    lines.append(f"Fibers: {n:,}, {length.sum() / 1000:.4g} mm in all")
    for t, name in enumerate(fibers.type_names):
        mine = fibers.types == t
        if not mine.any():
            lines.append(f"  type {t + 1} ({name}): none found")
            continue
        given = f", given {given_um[t]:.4g} um" if given_um and t < len(given_um) else ""
        lines.append(f"  type {t + 1}: {int(mine.sum()):,} fibers, median diameter {np.median(d[mine]):.4g} um{given}, "
                     f"{length[mine].sum() / 1000:.4g} mm")
    lines.append(f"Fibers cut by the region's faces: {int((~whole).sum()):,} (their lengths are lower bounds)")
    if whole.any():
        lines.append(f"Fibers wholly inside: {int(whole.sum()):,}, median length {np.median(length[whole]):.4g} um")
    # orientation, length-weighted, from every segment
    tensor, flat_share, total = np.zeros((3, 3)), 0.0, 0.0
    for line in fibers.lines:
        seg = np.diff(np.asarray(line, float), axis=0)
        size = np.linalg.norm(seg, axis=1)
        ok = size > 0
        t = seg[ok] / size[ok, None]
        tensor += (size[ok, None, None] * t[:, :, None] * t[:, None, :]).sum(0)
        flat_share += float(size[ok][np.abs(t[:, 2]) < np.sin(np.radians(20.0))].sum())
        total += float(size[ok].sum())
    if total > 0:
        values = np.sort(np.linalg.eigvalsh(tensor / total))[::-1]
        lines.append(f"Orientation: {100 * flat_share / total:.0f}% of the fiber length lies within 20 degrees of "
                     f"the slice (xy) plane; orientation tensor eigenvalues {values[0]:.2f}, {values[1]:.2f}, "
                     f"{values[2]:.2f} (1, 0, 0: all parallel; 0.5, 0.5, 0: random in a plane; 0.33 each: random in 3D)")
    weak = int((np.asarray(fibers.support) < 0.5).sum())
    if weak:
        lines.append(f"Traces less than half backed by the network: {weak} (support < 0.5 in fibers.csv; check them)")
    if fibers.bonds:
        points = sum(1 for b in fibers.bonds if np.isfinite(b["strength"]))
        lines.append(f"Bonds: {len(fibers.bonds):,} ({2 * len(fibers.bonds) / n:.2f} per fiber): {points:,} at the "
                     f"network's bond points, {len(fibers.bonds) - points:,} where its binder joins two fibers")
    if fractions:
        lines.append(f"Volume fraction of the drawn fibers: {100 * fractions['fibers']:.1f}%")
        if "binder" in fractions:
            what = "the network's binder outside the traced fibers" if fibers.binder is not None else \
                "solid outside the fibers"
            lines.append(f"Volume fraction of binder ({what}): {100 * fractions['binder']:.2f}%")
    return lines
