"""Tetrahedral meshes: each fiber's round or oval surface, meshed by gmsh."""

from __future__ import annotations

import collections
import itertools
import json
import math
import os
import pickle
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from ._hex import CORNERS, voxel_labels
from ._mesh import HEX_TETRAHEDRA, TET10_EDGES, TET_FACES, FemMesh, Tie, tet10_jacobians
from ._source import as_assembly, binder_name, material_table

# gmsh element types for 4- and 10-node tetrahedra.
GMSH_TET4, GMSH_TET10 = 4, 11
# gmsh numbers the last two mid-edge nodes of a 10-node tetrahedron (3-2 and
# 3-1) the other way round from Nastran and Abaqus (1-3 and 2-3, zero-based).
GMSH_TO_NASTRAN_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
# Swapping corners 1 and 2 turns a tetrahedron inside out, and back.
FLIP_TET4 = [0, 2, 1, 3]
FLIP_TET10 = [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]
# A 10-node tetrahedron curved to a fiber's surface whose Jacobian falls
# anywhere below this fraction of its straight-sided self's is straightened
# (gmsh's own threshold for curved elements).
DISTORTION_LIMIT = 0.1


def tet_mesh(
    assembly,
    element_size: float | None = None,
    *,
    order: int = 1,
    elements_around: int | None = None,
    clip_to_cell: bool = True,
    voxel_size: float | None = None,
    bond_radius_ratio: float | None = None,
    binder=None,
    workers: int | None = None,
    fiber_time_limit: float | None = None,
    progress: bool = True,
    verbose: bool = False,
) -> FemMesh:
    """Mesh the fibers with tetrahedra that follow each fiber's surface.

    Each fiber becomes a solid through a round or oval cross-section at
    every centerline vertex, straight from one vertex to the next as
    TANGLE's fibers are, with the oval's long axis where TANGLE puts it, and
    gmsh fills the solid with tetrahedra. Every fiber is its own body:
    fibers that touch or slightly overlap do not share nodes, so connect
    them in the solver (contact, or glued contact for bonded fibers). Fiber
    ends are flat. Where a cell wall slices a fiber lengthwise, gmsh may
    need finer elements in that fiber to mesh the thin edge of the cut, so
    such fibers get more elements.

    Binder, when asked for, comes from the same voxels as in
    :func:`hex_mesh`, and each binder voxel becomes six tetrahedra with
    fiber id 0 and the material ``"binder"``. The binder shares no nodes
    with the fibers. Instead the mesh's :class:`Tie` ties the binder's faces
    next to fibers to the fibers' surfaces, which the Abaqus file writes as
    a ``*TIE``.

    The fibers are meshed side by side in separate gmsh processes, one per
    CPU core by default, so a script that calls ``tet_mesh`` needs no
    ``if __name__ == "__main__":`` guard. With ``workers=1`` they are meshed
    one after another in this process, in its own gmsh session, so call it
    while gmsh is not initialized.

    Args:
        assembly: an ``Assembly`` or a recipe's ``RunResult``.
        element_size: largest element edge in meters. By default
            ``elements_around`` alone sets the size.
        order: 1 for 4-node tetrahedra (``CTETRA``/``C3D4``), 2 for 10-node
            tetrahedra (``CTETRA``/``C3D10``). The mid-edge nodes of a 10-node
            tetrahedron that lie on a fiber's surface are put on it, so the
            element's faces there curve with the fiber; an element that
            curving would fold over keeps straight edges.
        elements_around: element edges around each fiber's cross-section,
            which also sets the element size along that fiber. By default 16
            for 4-node tetrahedra, whose flat faces then keep a round fiber's
            volume within about 2%, and 8 for 10-node tetrahedra, whose
            curved faces keep it within about 0.5%. More follows the surface
            more closely, and fewer makes a lighter mesh.
        clip_to_cell: cut the fibers at the cell walls, with their periodic
            images on periodic axes, so the mesh fills the same box as
            :func:`hex_mesh`. A fiber that only touches a wall, such as one
            resting on the floor, is left whole, so it can stick out by up
            to 5% of its thickness. With ``False``, every fiber is meshed
            whole along its unwrapped centerline.
        voxel_size: the edge in meters of the binder's voxels, which must
            divide every cell length as in :func:`hex_mesh`. Binder needs it.
            Each binder voxel adds six tetrahedra, so binder from a CT scan
            is far lighter on voxels of a few scan voxels than on the scan's
            own.
        bond_radius_ratio: also mesh binder bridges at persistent junctions,
            with this radius as a fraction of the thinner fiber's radius, as
            :func:`hex_mesh` does.
        binder: also mesh these voxels as binder: a boolean ``(z, y, x)``
            array over the grid of ``voxel_size`` voxels across the cell, or
            over a grid a whole number of times finer such as a scan's own,
            as for :func:`hex_mesh`. Voxels a fiber fills stay fiber.
        workers: how many gmsh processes mesh fibers at once; by default one
            per CPU core. With 1 the fibers are meshed in this process.
        fiber_time_limit: seconds one fiber may take; a fiber that takes
            longer stops the mesh with an error that names it. By default
            there is no limit.
        progress: print how far the mesh has got once it has taken 10 s and
            every 30 s after that, naming the slowest fiber still being
            meshed once it has taken 30 s.
        verbose: let gmsh print its own progress.

    Returns:
        A :class:`FemMesh` of ``"tet4"`` or ``"tet10"`` elements.
    """
    try:
        import gmsh
    except ImportError as error:  # gmsh is an optional dependency
        raise ImportError("tangle.fem.tet_mesh needs gmsh: pip install gmsh") from error
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    if fiber_time_limit is not None and not (math.isfinite(fiber_time_limit) and fiber_time_limit > 0.0):
        raise ValueError("fiber_time_limit must be finite and positive")
    with_binder = binder is not None or bond_radius_ratio is not None
    if with_binder and voxel_size is None:
        raise ValueError("binder needs voxel_size: the edge of the voxels it is meshed from")
    if voxel_size is not None and not with_binder:
        raise ValueError("voxel_size only sets the binder's voxels; pass binder or bond_radius_ratio with it")
    if with_binder and not clip_to_cell:
        raise ValueError("binder fills the cell's voxels, so it needs clip_to_cell=True")
    assembly = as_assembly(assembly)
    centerlines = [np.asarray(line, dtype=float) for line in assembly.centerlines()]
    if not centerlines:
        raise ValueError("the assembly has no fibers")
    long_axes = [np.asarray(axes, dtype=float) for axes in assembly.long_axes()]
    semi_axes = assembly.section_semi_axes()
    fiber_ids = assembly.fiber_ids()
    materials, fiber_material = material_table(assembly)
    cell = assembly.cell
    origin = np.asarray(cell.origin, dtype=float)
    lengths = np.asarray(cell.lengths, dtype=float)
    periodic = list(cell.periodic)
    if element_size is not None and not (math.isfinite(element_size) and element_size > 0.0):
        raise ValueError("element_size must be finite and positive")
    if elements_around is None:
        elements_around = 16 if order == 1 else 8
    if elements_around < 3:
        raise ValueError("elements_around must be at least 3")
    # OpenCASCADE's tolerances suit lengths near 1, not micrometers written in
    # meters, so the geometry is built in units of the thinnest fiber's smaller
    # radius and scaled back at the end.
    unit = min(short for _, short in semi_axes)
    low, size = origin / unit, lengths / unit
    fiber_sizes = [_perimeter(long, short) / elements_around / unit for long, short in semi_axes]
    if element_size is not None:
        fiber_sizes = [min(fiber_size, element_size / unit) for fiber_size in fiber_sizes]

    jobs = []  # (fiber index, the arguments of _mesh_fiber after gmsh)
    for fiber, (points, axes, (long, short)) in enumerate(zip(centerlines, long_axes, semi_axes)):
        points, axes = _without_repeated_vertices(points / unit, axes)
        pieces = list(_pieces(points, long / unit, low, size, periodic, clip_to_cell))
        if pieces:
            box = (low, size) if clip_to_cell else None
            jobs.append((fiber, (fiber_ids[fiber], points, axes, long / unit, short / unit, pieces, box,
                                 fiber_sizes[fiber], order, elements_around, verbose)))
    if not jobs:
        raise ValueError("no fiber reaches into the cell")
    # Before the fibers, so a binder mask that does not fit fails at once.
    binder_mesh = _binder_mesh(assembly, voxel_size, bond_radius_ratio, binder, order) if with_binder else None
    processes = min(workers or _cpu_count(), len(jobs))
    report = _Progress(len(jobs), progress)
    if processes > 1 or fiber_time_limit is not None:
        meshes = _mesh_in_processes(jobs, processes, fiber_time_limit, verbose, report)
    else:
        meshes = _mesh_here(gmsh, jobs, verbose, report)
    report.done()

    node_blocks, element_blocks, block_fibers = [], [], []  # one block per fiber
    offset = 0
    for fiber, _ in jobs:
        if meshes[fiber] is None:  # nothing of it is left inside the cell
            continue
        nodes, elements = meshes[fiber]
        node_blocks.append(nodes)
        element_blocks.append(elements + offset)
        block_fibers.append(np.full(len(elements), fiber, dtype=np.int64))
        offset += len(nodes)
    if not element_blocks:
        raise ValueError("no fiber reaches into the cell")

    elements = np.concatenate(element_blocks)
    fibers = np.concatenate(block_fibers)
    # Keep only the nodes the tetrahedra use, numbered in order.
    used, elements = np.unique(elements, return_inverse=True)
    elements = elements.reshape(len(fibers), -1)
    nodes = np.concatenate(node_blocks)[used] * unit

    element_fibers = np.asarray(fiber_ids, dtype=np.int64)[fibers]
    element_materials = fiber_material[fibers]
    tie = None
    if binder_mesh is not None:
        binder_nodes, binder_elements, next_to_fibers = binder_mesh
        if len(next_to_fibers):
            # A binder node next to a fiber lies within about 1.2 voxels of the
            # fiber's round surface, and the flat facets of the fiber's mesh
            # lie up to a facet's sagitta inside that surface.
            sagitta = max((edge * unit) ** 2 * long / (8.0 * short * short)
                          for edge, (long, short) in zip(fiber_sizes, semi_axes))
            surfaces = _outer_faces(elements, [len(block) for block in element_blocks])
            tie = (next_to_fibers + [len(elements), 0], surfaces, 2.0 * voxel_size + 2.0 * sagitta)
        materials = (*materials, binder_name(materials))
        elements = np.concatenate([elements, binder_elements + len(nodes)])
        nodes = np.concatenate([nodes, binder_nodes])
        element_fibers = np.concatenate([element_fibers, np.zeros(len(binder_elements), dtype=np.int64)])
        element_materials = np.concatenate(
            [element_materials, np.full(len(binder_elements), len(materials) - 1, dtype=np.int64)]
        )
    sequence = np.lexsort((np.arange(len(elements)), element_fibers, element_materials))
    ties = ()
    if tie is not None:
        position = np.empty_like(sequence)
        position[sequence] = np.arange(len(sequence))
        secondary, main, tolerance = tie
        ties = (Tie("BINDER_TIE", _moved(secondary, position), _moved(main, position), tolerance),)
    return FemMesh(
        nodes=nodes,
        elements=elements[sequence],
        element_type="tet4" if order == 1 else "tet10",
        element_fibers=element_fibers[sequence],
        element_materials=element_materials[sequence],
        materials=materials,
        cell_origin=tuple(origin.tolist()),
        cell_lengths=tuple(lengths.tolist()),
        ties=ties,
    )


def _binder_mesh(assembly, voxel_size: float, bond_radius_ratio: float | None, binder, order: int):
    """The binder's tetrahedra, six to each binder voxel of :func:`hex_mesh`,
    on nodes of their own: the nodes in meters, the elements, and the
    ``(element, face)`` pairs of the faces next to a voxel that a fiber
    fills. None when no voxel is binder."""
    counts, phase, owner = voxel_labels(assembly, voxel_size, bond_radius_ratio, binder)
    nx, ny, nz = counts
    chosen = np.flatnonzero((phase != 0) & (owner == 0))
    if chosen.size == 0:
        return None
    voxels = np.stack([chosen % nx, (chosen // nx) % ny, chosen // (nx * ny)], axis=1)

    # Points are numbered on a grid of half voxels: corner a of the voxel grid
    # is point 2a, and the middle of the edge from a to b is point a + b. A
    # point's number is linear in its place, so the node of corner pair (p, q)
    # in the voxel whose low corner is v, point 2v + CORNERS[p] + CORNERS[q],
    # is numbered twice v's number plus that of CORNERS[p] + CORNERS[q].
    width = np.array([1, 2 * nx + 1, (2 * nx + 1) * (2 * ny + 1)], dtype=np.int64)
    pairs, tetrahedra = _voxel_tetrahedra(order)
    keys = 2 * (voxels @ width)[:, None] + ((CORNERS[pairs[:, 0]] + CORNERS[pairs[:, 1]]) @ width)[None, :]
    used, inverse = np.unique(keys, return_inverse=True)
    elements = inverse.reshape(keys.shape)[:, tetrahedra].reshape(-1, tetrahedra.shape[1])
    points = np.stack([used % width[1], (used // width[1]) % (2 * ny + 1), used // width[2]], axis=1)
    spacing = np.asarray(assembly.cell.lengths, dtype=float) / np.asarray(counts, dtype=float)
    nodes = np.asarray(assembly.cell.origin, dtype=float) + points * (spacing / 2.0)

    filled_by_fiber = (phase != 0) & (owner != 0)
    stride = (1, nx, nx * ny)
    next_to_fibers = []
    for (axis, high), faces in _voxel_sides().items():
        step = 1 if high else -1
        beside = voxels[:, axis] + step
        rows = np.flatnonzero((beside >= 0) & (beside < counts[axis]))
        rows = rows[filled_by_fiber[chosen[rows] + step * stride[axis]]]
        for tetrahedron, face in faces:
            next_to_fibers.append(np.stack([6 * rows + tetrahedron, np.full(len(rows), face)], axis=1))
    return nodes, elements, np.concatenate(next_to_fibers)


def _voxel_tetrahedra(order: int) -> tuple[np.ndarray, np.ndarray]:
    """One voxel's nodes as pairs of its corners, ``(c, c)`` for corner c
    and ``(a, b)`` for the middle of the edge from a to b, and its six
    tetrahedra (:data:`HEX_TETRAHEDRA`) as indices into those pairs."""
    pairs = [(corner, corner) for corner in range(8)]
    tetrahedra = []
    for corners in HEX_TETRAHEDRA.tolist():
        nodes = list(corners)
        if order == 2:
            for first, second in TET10_EDGES:
                pair = tuple(sorted((corners[first], corners[second])))
                if pair not in pairs:
                    pairs.append(pair)
                nodes.append(pairs.index(pair))
        tetrahedra.append(nodes)
    return np.array(pairs), np.array(tetrahedra)


def _voxel_sides() -> dict[tuple[int, int], list[tuple[int, int]]]:
    """For each side of a voxel, ``(axis, 0)`` for its low side and
    ``(axis, 1)`` for its high side, the ``(tetrahedron, face)`` pairs of
    its six tetrahedra (:data:`HEX_TETRAHEDRA`) that lie on it."""
    return {
        (axis, high): [
            (tetrahedron, face)
            for tetrahedron, corners in enumerate(HEX_TETRAHEDRA)
            for face, on_face in enumerate(TET_FACES)
            if (CORNERS[corners[on_face], axis] == high).all()
        ]
        for axis in range(3)
        for high in (0, 1)
    }


def _outer_faces(elements: np.ndarray, block_sizes: list[int]) -> np.ndarray:
    """``(element, face)`` of every face that belongs to one tetrahedron
    only: the surfaces of the fibers. The elements come in one block per
    fiber, and as fibers share no nodes, each block is done on its own."""
    found, start = [], 0
    for size in block_sizes:
        faces = np.sort(elements[start : start + size, :4][:, TET_FACES].reshape(-1, 3), axis=1)
        order = np.lexsort(faces.T)
        same = (faces[order[1:]] == faces[order[:-1]]).all(axis=1)
        alone = np.ones(len(order), dtype=bool)
        alone[1:] &= ~same
        alone[:-1] &= ~same
        which = order[alone]
        found.append(np.stack([start + which // 4, which % 4], axis=1))
        start += size
    return np.concatenate(found)


def _moved(faces: np.ndarray, position: np.ndarray) -> np.ndarray:
    """``(element, face)`` pairs after element ``e`` has moved to
    ``position[e]``, in element order."""
    faces = np.stack([position[faces[:, 0]], faces[:, 1]], axis=1)
    return faces[np.lexsort((faces[:, 1], faces[:, 0]))]


def _mesh_here(gmsh, jobs, verbose: bool, report: _Progress) -> dict:
    """Every job's mesh, one fiber after another in this process's own gmsh
    session."""
    if gmsh.isInitialized():
        raise RuntimeError("tet_mesh runs its own gmsh session; call gmsh.finalize() first, or pass workers > 1")
    _start_gmsh(gmsh, verbose)
    meshes = {}
    try:
        for fiber, job in jobs:
            meshes[fiber] = _mesh_fiber(gmsh, *job)
            report.finished(meshes[fiber])
            report.update()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
    return meshes


def _mesh_in_processes(jobs, processes: int, limit: float | None, verbose: bool, report: _Progress) -> dict:
    """Every job's mesh, from ``processes`` gmsh processes working at once.

    Each process takes the next fiber as soon as it is done with one. A
    fiber that gmsh cannot mesh, or that takes longer than ``limit``
    seconds, stops them all."""
    waiting = collections.deque(jobs)
    answers: queue.Queue = queue.Queue()
    pool: list[_Worker] = []
    meshes = {}
    complete = False
    try:
        for _ in range(processes):
            pool.append(_Worker(answers, verbose))
            pool[-1].send(*waiting.popleft())
        while len(meshes) < len(jobs):
            now = time.monotonic()
            busy = [worker for worker in pool if worker.fiber is not None]
            for worker in busy:
                if limit is not None and now - worker.since >= limit:
                    raise RuntimeError(
                        f"gmsh took longer than {limit:g} s on fiber {worker.fiber_id}; "
                        "raise fiber_time_limit, or mesh it with fewer elements_around"
                    )
            report.update([(worker.fiber_id, now - worker.since) for worker in busy])
            due = report.due
            if limit is not None and busy:
                due = min(due, min(worker.since for worker in busy) + limit)
            try:
                worker, answer = answers.get(timeout=None if due == math.inf else max(due - now, 0.0))
            except queue.Empty:
                continue
            if worker.fiber is None:  # an idle process stopped; it had nothing left to do
                continue
            if answer is None:
                raise RuntimeError(
                    f"the gmsh process meshing fiber {worker.fiber_id} stopped (exit code {worker.process.poll()}); "
                    "with workers=1, tet_mesh meshes in this process instead"
                )
            kind, value = answer
            if kind == "error":
                raise RuntimeError(value)
            meshes[worker.fiber] = value
            worker.fiber = None
            report.finished(value)
            if waiting:
                worker.send(*waiting.popleft())
        complete = True
    finally:
        for worker in pool:
            worker.close(kill=not complete)
    return meshes


# Run with ``python -c``: the process takes this one's module path, so it
# imports the same tangle and gmsh.
_WORKER_START = (
    "import json, sys; sys.path[:] = json.loads(sys.argv[1]); "
    "from tangle.fem._gmsh_worker import main; main(sys.argv[2] == 'verbose')"
)


class _Worker:
    """A gmsh process for :func:`_mesh_in_processes` (see ``_gmsh_worker``),
    and a thread that reads its answers into ``answers``."""

    def __init__(self, answers: queue.Queue, verbose: bool) -> None:
        command = [sys.executable, "-c", _WORKER_START, json.dumps(sys.path), "verbose" if verbose else "quiet"]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.fiber: int | None = None  # the index of the fiber being meshed
        self.fiber_id = 0  # its TANGLE id
        self.since = 0.0  # when it was sent
        self._reader = threading.Thread(target=self._read, args=(answers,), daemon=True)
        self._reader.start()

    def _read(self, answers: queue.Queue) -> None:
        while True:
            try:
                answer = pickle.load(self.process.stdout)
            except Exception:  # the stream ended: the process stopped
                answers.put((self, None))
                return
            answers.put((self, answer))

    def send(self, fiber: int, job: tuple) -> None:
        self.fiber, self.fiber_id, self.since = fiber, job[0], time.monotonic()
        try:
            pickle.dump(job, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.process.stdin.flush()
        except OSError:  # the process has stopped, which its reader reports
            pass

    def close(self, kill: bool) -> None:
        if kill:
            self.process.kill()
        try:
            self.process.stdin.close()  # the end of the jobs: the process finishes
        except OSError:
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._reader.join()
        self.process.stdout.close()


class _Progress:
    """Prints how far ``tet_mesh`` has got: first after 10 s, then every 30 s,
    and a last line when it is done if it printed any."""

    FIRST, EVERY = 10.0, 30.0

    def __init__(self, fibers: int, enabled: bool) -> None:
        self.fibers, self.meshed, self.elements = fibers, 0, 0
        self.started = time.monotonic()
        self.due = self.started + self.FIRST if enabled else math.inf
        self.printed = False

    def finished(self, mesh) -> None:
        self.meshed += 1
        if mesh is not None:
            self.elements += len(mesh[1])

    def update(self, running: list[tuple[int, float]] = ()) -> None:
        """Print a line if one is due, naming the slowest of the fibers being
        meshed (``running``: their TANGLE ids and seconds so far)."""
        now = time.monotonic()
        if now < self.due:
            return
        line = (f"tet_mesh: {self.meshed:,} of {self.fibers:,} fibers meshed, "
                f"{self.elements:,} tetrahedra, {_duration(now - self.started)}")
        slowest = max(running, key=lambda item: item[1], default=None)
        if slowest is not None and slowest[1] >= self.EVERY:
            line += f"; fiber {slowest[0]} has taken {_duration(slowest[1])} so far"
        print(line, flush=True)
        self.printed = True
        self.due = now + self.EVERY

    def done(self) -> None:
        if self.printed:
            print(f"tet_mesh: all {self.fibers:,} fibers meshed, {self.elements:,} tetrahedra, "
                  f"in {_duration(time.monotonic() - self.started)}", flush=True)


def _cpu_count() -> int:
    """The CPU cores this process may use (all of them, unless a scheduler
    such as Slurm limits it)."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds // 3600} h {round(seconds % 3600 / 60):02d} min"


def _start_gmsh(gmsh, verbose: bool) -> None:
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)


def _mesh_fiber(gmsh, fiber_id, points, axes, long, short, pieces, box, size, order, around, verbose):
    """Nodes and tetrahedra (in Nastran node order) of one fiber, in the
    scaled units, or None when nothing of it is left inside the cell.

    The fiber gets its own gmsh model. Where a wall cuts a fiber lengthwise
    at a grazing angle, the cut face meets the fiber's surface in a thin
    wedge that a uniform size can fail to mesh; the later tries refine with
    the surface curvature, which resolves the wedge at the cost of more
    elements in that fiber."""
    tries = (0, max(3, around // 2), around, 2 * around)
    for attempt, curvature in enumerate(tries):
        if attempt:  # after a failure gmsh meshes nothing more until it restarts
            gmsh.finalize()
            _start_gmsh(gmsh, verbose)
        gmsh.model.add(f"tangle_fem_fiber_{fiber_id}")
        try:
            volumes = _fiber_solid(gmsh.model.occ, fiber_id, points, axes, long, short, pieces, box)
            if not volumes:
                return None
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            try:
                gmsh.model.mesh.generate(3)
                if order == 2:
                    # Mid-edge nodes on the fiber's surface go onto it, not onto the chord. They only do when
                    # the mesh becomes second order after it is made: made second order at once, they stay
                    # on the chord.
                    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
                    gmsh.model.mesh.setOrder(2)
                node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
                if len(node_tags) == 0:
                    raise RuntimeError("gmsh made no nodes")
            except Exception as error:  # gmsh raises a plain Exception
                if attempt + 1 == len(tries):
                    raise RuntimeError(f"gmsh could not mesh fiber {fiber_id}: {error}") from error
                continue
            node_tags = np.asarray(node_tags, dtype=np.int64)
            node_index = np.full(node_tags.max() + 1, -1, dtype=np.int64)
            node_index[node_tags] = np.arange(len(node_tags))
            wanted = GMSH_TET4 if order == 1 else GMSH_TET10
            blocks = []
            for tag in volumes:
                types, _, element_nodes = gmsh.model.mesh.getElements(3, tag)
                for element_type, members in zip(types, element_nodes):
                    if element_type != wanted:
                        raise RuntimeError(f"gmsh made element type {element_type} instead of tetrahedra")
                    blocks.append(node_index[np.asarray(members, dtype=np.int64)].reshape(-1, 4 if order == 1 else 10))
            nodes = np.array(coordinates, dtype=float).reshape(-1, 3)
            elements = np.concatenate(blocks)
            if order == 2:
                elements = elements[:, GMSH_TO_NASTRAN_TET10]
            elements = _upright(nodes, elements)
            if order == 2:
                _straighten_distorted(nodes, elements)
            return nodes, elements
        finally:
            gmsh.model.remove()


def _upright(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """The tetrahedra, with any that are inside out turned back."""
    corners = nodes[elements[:, :4]]
    edges = corners[:, 1:] - corners[:, :1]
    inverted = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])) < 0.0
    elements[inverted] = elements[inverted][:, FLIP_TET4 if elements.shape[1] == 4 else FLIP_TET10]
    return elements


def _straighten_distorted(nodes: np.ndarray, elements: np.ndarray) -> int:
    """Straighten the edges of every 10-node tetrahedron that curving to
    the fiber's surface has distorted, moving their mid-edge nodes in
    ``nodes``, and return how many there were.

    Where the surface curves across a thin element, mid-edge nodes on the
    surface can fold it: somewhere inside, its Jacobian drops below
    DISTORTION_LIMIT of its straight-sided self's, or below zero, which
    solvers reject. Back in the middles of its edges, they make it its
    straight self again. That moves nodes its neighbors share, so they are
    checked again."""
    straightened = np.zeros(len(elements), dtype=bool)
    while True:
        corners = nodes[elements[:, :4]]
        edges = corners[:, 1:] - corners[:, :1]
        straight = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
        lowest, _ = tet10_jacobians(nodes, elements)
        distorted = (lowest < DISTORTION_LIMIT * straight) & ~straightened
        if not distorted.any():
            return int(straightened.sum())
        straightened |= distorted
        chosen = elements[distorted]
        nodes[chosen[:, 4:]] = nodes[chosen[:, TET10_EDGES]].mean(axis=2)


def _fiber_solid(occ, fiber_id, points, axes, long, short, pieces, box):
    """Volume tags of one fiber's solid pieces, cut to ``box`` (``(low,
    size)``) when it is given.

    A wall within 5% of the fiber's thickness of its surface neither cuts it
    nor brings in a periodic image: OpenCASCADE cannot cut a solid that
    touches a wall, as a fiber resting on the floor of the cell does."""
    sliver = 1.0e-6 * math.pi * long * short * short
    margin = 0.05 * short
    volumes = []
    for vertices, shift in pieces:
        piece = points[vertices] + shift
        tool, deep = None, np.zeros(len(piece), dtype=bool)
        if box is not None:
            low, high = box[0], box[0] + box[1]
            reach_low, reach_high = piece.min(axis=0) - long, piece.max(axis=0) + long
            if np.any(reach_high < low + margin) or np.any(reach_low > high - margin):
                continue  # this image only grazes the cell
            cut = (reach_low < low - margin) | (reach_high > high + margin)
            if cut.any():
                tool_low = np.where(cut, low, np.minimum(low, reach_low) - long)
                tool_high = np.where(cut, high, np.maximum(high, reach_high) + long)
                tool = occ.addBox(*tool_low, *(tool_high - tool_low))
            # Cross-sections at these vertices lie wholly inside the cell.
            deep = np.all((piece > low + long) & (piece < high - long), axis=1)
        try:
            solids = _loft(occ, piece, axes[vertices], long, short)
            if tool is not None:
                solids, _ = occ.intersect(solids, [(3, tool)], removeObject=True, removeTool=True)
        except Exception as error:  # gmsh raises a plain Exception
            raise RuntimeError(f"gmsh could not build fiber {fiber_id}: {error}") from error
        kept = 0
        for dim, tag in solids:
            if dim != 3:
                continue
            if occ.getMass(dim, tag) < sliver:
                occ.remove([(dim, tag)], recursive=True)
            else:
                volumes.append(tag)
                kept += 1
        if deep.any() and not kept:
            raise RuntimeError(f"gmsh lost fiber {fiber_id} while cutting it at the cell walls")
    occ.synchronize()
    return volumes


def _without_repeated_vertices(points: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], steps > 1.0e-9 * max(steps.max(initial=0.0), 1.0e-300)])
    return points[keep], axes[keep]


def _pieces(points, pad, origin, lengths, periodic, clip):
    """``(vertex slice, shift)`` for every stretch of the fiber to loft: the
    whole fiber, or each stretch of each periodic image that reaches the
    cell, with one extra vertex at both ends so the walls cut cleanly."""
    if not clip:
        yield slice(0, len(points)), np.zeros(3)
        return
    pad = 1.01 * pad
    low, high = points.min(axis=0) - pad, points.max(axis=0) + pad
    ranges = []
    for axis in range(3):
        if periodic[axis]:
            first = math.ceil((origin[axis] - high[axis]) / lengths[axis])
            last = math.floor((origin[axis] + lengths[axis] - low[axis]) / lengths[axis])
            ranges.append(range(first, last + 1))
        else:
            ranges.append(range(0, 1))
    for image in itertools.product(*ranges):
        shift = np.asarray(image, dtype=float) * lengths
        moved = points + shift
        segment_low = np.minimum(moved[:-1], moved[1:]) - pad
        segment_high = np.maximum(moved[:-1], moved[1:]) + pad
        reaches = np.all((segment_high >= origin) & (segment_low <= origin + lengths), axis=1)
        keep = reaches.copy()
        keep[1:] |= reaches[:-1]
        keep[:-1] |= reaches[1:]
        vertex = np.zeros(len(points), dtype=bool)
        vertex[:-1] |= keep
        vertex[1:] |= keep
        for start, stop in _runs(vertex):
            if stop - start >= 2:
                yield slice(start, stop), shift


def _perimeter(long: float, short: float) -> float:
    """An ellipse's perimeter (Ramanujan's approximation; exact for a circle)."""
    return math.pi * (3.0 * (long + short) - math.sqrt((3.0 * long + short) * (long + 3.0 * short)))


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return zip(edges[0::2], edges[1::2])


def _loft(occ, points: np.ndarray, long_axes: np.ndarray, long: float, short: float) -> list[tuple[int, int]]:
    """A solid through a cross-section at every vertex, an ellipse with its
    long semi-axis along TANGLE's long axis or a circle, and straight from
    one to the next, as TANGLE's fiber is.

    An inner section halves the angle between its two stretches, so they
    meet there as mitered pipes do. (One smooth surface through all the
    sections would follow the fiber no better, and over a long fiber gmsh
    can fail to mesh it in hours.)"""
    steps = np.diff(points, axis=0)
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    tangents = np.empty_like(points)
    tangents[0], tangents[-1] = steps[0], steps[-1]
    tangents[1:-1] = steps[:-1] + steps[1:]
    norms = np.linalg.norm(tangents, axis=1)
    folded = norms < 1.0e-9  # a stretch that turns straight back: square to the one before
    tangents[folded], norms[folded] = steps[np.flatnonzero(folded) - 1], 1.0
    tangents /= norms[:, None]
    oval = long > short * (1.0 + 1.0e-9)
    directions = _section_directions(tangents, long_axes if oval else None)
    wires = []
    for point, tangent, direction in zip(points, tangents, directions):
        if oval:
            curve = occ.addEllipse(*point, long, short, zAxis=tangent.tolist(), xAxis=direction.tolist())
        else:
            curve = occ.addCircle(*point, long, zAxis=tangent.tolist(), xAxis=direction.tolist())
        wires.append(occ.addWire([curve]))
    solid = occ.addThruSections(wires, makeSolid=True, makeRuled=True)
    occ.remove([(1, wire) for wire in wires], recursive=True)
    return [(dim, tag) for dim, tag in solid if dim == 3]


def _section_directions(tangents: np.ndarray, long_axes: np.ndarray | None) -> np.ndarray:
    """A unit direction across the fiber at every vertex, turning as little
    as possible between vertices: TANGLE's long axis for an oval fiber, or a
    transported one for a round fiber, whose seam then runs straight."""
    directions = np.empty_like(tangents)
    if long_axes is None:
        start = np.eye(3)[np.argmin(np.abs(tangents[0]))]
    else:
        start = long_axes[0]
    previous = start
    for index, tangent in enumerate(tangents):
        candidate = previous if long_axes is None else long_axes[index]
        candidate = candidate - np.dot(candidate, tangent) * tangent
        norm = np.linalg.norm(candidate)
        if norm < 1.0e-9:  # fall back to the previous direction
            candidate = previous - np.dot(previous, tangent) * tangent
            norm = np.linalg.norm(candidate)
        candidate = candidate / norm
        if index and np.dot(candidate, directions[index - 1]) < 0.0:
            candidate = -candidate
        directions[index] = candidate
        previous = candidate
    return directions
