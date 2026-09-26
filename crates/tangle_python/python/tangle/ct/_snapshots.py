"""Snapshots of a fit at each of its steps, for OVITO (``fit_fibers(..., snapshots=folder)``).

Every step the fitter logs (trace, each solve, each round's cleanup and new
fibers, the final solve, each redraw pass and the steps inside it) becomes one
frame of ``fits.dump``, a multi-frame LAMMPS dump: one spherocylinder per
centerline segment, in micrometers, as Tangle's own OVITO trajectories draw
fibers. A fiber keeps its id (``mol``, so its color) from frame to frame while
it mostly retraces the same place. ``stages.txt`` lists each frame's step, and ``view_fits.py`` loads the
dump in OVITO (``ovitos view_fits.py``, or the ``ovito`` Python package),
colored by fiber (``--render`` also writes one labeled image per frame).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_COLUMNS = (
    "id mol type AsphericalShape.X AsphericalShape.Y AsphericalShape.Z quati quatj quatk quatw x y z "
    "fiber_type radius segment confidence settled surround frozen"
)
_VALUES = ("confidence", "settled", "surround", "frozen")


class Snapshots:
    """Writes the frames; ``write`` once per step, ``close`` at the end."""

    def __init__(self, folder, voxel_size: float, shape_zyx) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.scale = voxel_size * 1e6  # voxels to micrometers
        self.box = np.array(shape_zyx[::-1], dtype=np.float64) * self.scale
        self.dump = self.folder / "fits.dump"
        self.dump.write_text("")
        self.stages: list[dict] = []
        # Frames of alternatives (a redraw pass's plans) wait under a key
        # until ``keep`` picks the one the fit went on with.
        self.held: str | None = None
        self.pending: dict[str, list] = {}
        # Stable fiber ids: each fiber takes the id of the fiber it mostly retraces in the previous frame.
        self.previous: tuple[np.ndarray, np.ndarray] | None = None  # points (voxels) and their fiber ids
        self.next_id = 1

    def hold(self, key: str | None) -> None:
        """Hold the next frames under ``key`` (None: write them straight away)."""
        self.held = key

    def keep(self, key: str) -> None:
        """Write the frames held under ``key``, drop every other held frame, and stop holding."""
        frames = self.pending.get(key, [])
        self.pending, self.held = {}, None
        for frame in frames:
            self.write(*frame)

    def write(self, stage: str, lines, radii, types, values=None) -> None:
        """One frame: ``lines`` (voxel coordinates), each fiber's radius (voxels) and type, and
        optionally ``values``: per name in ``_VALUES``, one array per fiber with a value per node
        (a segment gets the lower of its two nodes'; -1 where not given)."""
        if self.held is not None:
            copy = [np.array(line, dtype=np.float64) for line in lines]
            self.pending.setdefault(self.held, []).append(
                (stage, copy, None if radii is None else np.array(radii), None if types is None else np.array(types),
                 values)
            )
            return
        rows = []
        lines = [np.asarray(line, dtype=np.float64) for line in lines]
        ids = self._stable_ids(lines)
        radii = np.asarray(radii, dtype=np.float64) if radii is not None else np.full(len(lines), 1.0)
        types = np.asarray(types, dtype=int) if types is not None else np.zeros(len(lines), dtype=int)
        for fiber, line in enumerate(lines):
            if len(line) < 2:
                continue
            r = float(radii[fiber]) * self.scale
            points = line * self.scale
            for k, (a, b) in enumerate(zip(points[:-1], points[1:])):
                axis = b - a
                length = float(np.linalg.norm(axis))
                if length <= 0.0:
                    continue
                q = _quaternion_from_z(axis / length)
                c = 0.5 * (a + b)
                extra = [
                    float(min(values[name][fiber][k], values[name][fiber][k + 1]))
                    if values is not None and name in values else -1.0
                    for name in _VALUES
                ]
                rows.append((ids[fiber], int(types[fiber]), r, length, q, c, k, extra))
        with self.dump.open("a") as out:
            out.write(f"ITEM: TIMESTEP\n{len(self.stages)}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n")
            out.write("ITEM: BOX BOUNDS ff ff ff\n")
            for high in self.box:
                out.write(f"0 {high:.6g}\n")
            out.write(f"ITEM: ATOMS {_COLUMNS}\n")
            for index, (mol, kind, r, length, q, c, k, extra) in enumerate(rows, start=1):
                out.write(
                    f"{index} {mol} 1 {r:.4f} {r:.4f} {length:.4f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                    f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {kind} {r:.4f} {k} " + " ".join(f"{v:.3f}" for v in extra) + "\n"
                )
        self.stages.append({"frame": len(self.stages), "stage": stage, "fibers": len(lines)})

    def _stable_ids(self, lines: list[np.ndarray]) -> list[int]:
        """An id per fiber: the id of the previous frame's fiber that at least half of it lies
        within 3 voxels of (the best match wins; one id per fiber), else a new one."""
        from scipy.spatial import cKDTree

        points = [_densify(line) for line in lines]
        ids = [0] * len(lines)
        if self.previous is not None and len(self.previous[0]):
            tree = cKDTree(self.previous[0])
            claims = []
            for fiber, p in enumerate(points):
                if not len(p):
                    continue
                d, j = tree.query(p)
                near = self.previous[1][j[d <= 3.0]]
                if len(near) >= 0.5 * len(p):
                    values, counts = np.unique(near, return_counts=True)
                    claims.append((int(counts.max()), fiber, int(values[np.argmax(counts)])))
            taken = set()
            for _, fiber, old in sorted(claims, reverse=True):
                if old not in taken:
                    ids[fiber] = old
                    taken.add(old)
        for fiber in range(len(lines)):
            if ids[fiber] == 0:
                ids[fiber] = self.next_id
                self.next_id += 1
            else:
                self.next_id = max(self.next_id, ids[fiber] + 1)
        kept = [p for p in points if len(p)]
        self.previous = (
            np.concatenate(kept) if kept else np.zeros((0, 3)),
            np.concatenate([np.full(len(p), i) for p, i in zip(points, ids) if len(p)]) if kept else np.zeros(0, int),
        )
        return ids

    def close(self) -> None:
        """Write the stage list and the OVITO viewing script."""
        (self.folder / "stages.txt").write_text(
            "frame\tstep\tfibers\n" + "".join(f"{s['frame']}\t{s['stage']}\t{s['fibers']}\n" for s in self.stages)
        )
        names = [s["stage"] for s in self.stages]
        (self.folder / "view_fits.py").write_text(_VIEW.format(names=json.dumps(names)))


def _densify(line: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Points along ``line`` about ``step`` voxels apart."""
    if len(line) < 2:
        return np.zeros((0, 3))
    out = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def _quaternion_from_z(direction: np.ndarray) -> tuple[float, float, float, float]:
    """(i, j, k, w) turning +z onto ``direction`` (a unit vector)."""
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(direction @ z, -1.0, 1.0))
    if dot < -1.0 + 1e-12:
        return (1.0, 0.0, 0.0, 0.0)  # half a turn about x
    axis = np.cross(z, direction)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w])
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


_VIEW = '''# Generated by tangle.ct. Run with: ovitos view_fits.py [--render] [--confidence]
# (or python with the ovito package). Saves fits.ovito, which the OVITO app opens with the fits
# colored by fiber; the step of each frame is in stages.txt. --render also writes one PNG per frame,
# labeled with its step, to frames/.
import sys
from pathlib import Path

import ovito
from ovito.io import import_file
from ovito.modifiers import ColorCodingModifier
from ovito.vis import ParticlesVis, TextLabelOverlay, Viewport

HERE = Path(__file__).resolve().parent
STAGES = {names}

pipeline = import_file(str(HERE / "fits.dump"), sort_particles=True)
# Color by fiber. Other columns to color by: fiber_type (fine/coarse), confidence and settled
# (0 unsure .. 1 sure; the redraw cuts below 0.5), surround, frozen (1 = a redraw froze it).
COLOR_BY = "confidence" if "--confidence" in sys.argv else "Molecule Identifier"
if COLOR_BY == "confidence":
    pipeline.modifiers.append(ColorCodingModifier(property="confidence", start_value=0.0, end_value=1.0))
else:
    pipeline.modifiers.append(ColorCodingModifier(property=COLOR_BY))
pipeline.add_to_scene()
pipeline.compute().particles.vis.shape = ParticlesVis.Shape.Spherocylinder
ovito.scene.save(str(HERE / "fits.ovito"))
print(f"{{pipeline.num_frames}} frames; saved {{HERE / 'fits.ovito'}}")

if "--render" in sys.argv:
    (HERE / "frames").mkdir(exist_ok=True)
    viewport = Viewport(type=Viewport.Type.Perspective, camera_dir=(-1, -1, -1))
    viewport.zoom_all()
    label = TextLabelOverlay(font_size=0.04, text_color=(0, 0, 0))
    viewport.overlays.append(label)
    for frame, name in enumerate(STAGES):
        label.text = f"{{frame}}: {{name}}"
        viewport.render_image(filename=str(HERE / "frames" / f"frame_{{frame:03d}}.png"), frame=frame, size=(900, 700))
    print(f"wrote {{len(STAGES)}} frames to {{HERE / 'frames'}}")
'''
