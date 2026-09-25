"""Fit a large scan tile by tile, checkpoint every tile, and stitch the tiles.

A whole scan is too large to fit in one go: every stage of
:func:`fit_fibers` (tracing, the Hessians, rasterizing, the solver's image)
holds a few float copies of the volume, and one crashed or stopped run loses
everything. :func:`fit_tiled` splits the scan into a grid of **core** boxes
that partition it, and fits each core padded by ``overlap`` voxels on every
side, so a fiber near a core wall is fitted with scan on both sides of it.

**Ownership.** Every point of the scan belongs to exactly one core. From each
tile's fit only the stretches of centerline inside its own core are kept;
the padding is there only to make those stretches right.

**Stitching.** Where a kept stretch leaves its core, the fit goes on into the
neighbouring core (the padding). The neighbour tile has its own fit of the
same fiber there, and its kept stretch leaves toward this core in turn.
Within ``band`` voxels of the crossing the two fits cover the same stretch of
fiber from both sides; they are joined when they lie on each other there
(mean distance under ``0.75`` of the smaller radius, same fiber type) and
cut at the core wall otherwise. A fit that one tile ends just inside the
neighbour's core is joined the same way to the neighbour fit's end. Joins
are made best first, one per stretch end, and chains of joined stretches are
the stitched fibers. Pieces shorter than the type's minimum length that end
at a wall without a partner are dropped (they are the other tile's fiber,
seen from the edge).

**Checkpoints.** With ``checkpoint=<directory>``, every finished tile is
written to ``<directory>/tiles/`` in scan coordinates, next to a
``manifest.json`` of the inputs. Calling :func:`fit_tiled` again with the
same directory skips the tiles already there, so a stopped run restarts
where it stopped. :func:`load_tiles` stitches whatever tiles a checkpoint
holds, without the scan, to look at a run in progress.

**Grey levels.** A plain grey scan (no mask, ranges or profiles) is mapped
to void 0 and fiber 1 by levels read off the scan. Each tile reading its own
would type and size fibers differently from tile to tile (and an empty tile
has no fiber level), so the first tile fitted, the central one, sets the
levels for every other tile (``FitSettings.levels``, saved in the manifest).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ._fit import FiberSpec, FitResult, FitSettings, fit_fibers
from ._geometry import polyline_length
from ._image import Levels

SCHEMA = "tangle.ct.tiles/1"


@dataclass(frozen=True)
class TileGrid:
    """Core boxes that partition a ``(z, y, x)`` volume, each padded by ``overlap``."""

    shape: tuple[int, int, int]
    edges: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]  # per axis (z, y, x)
    overlap: int

    @classmethod
    def make(cls, shape: Sequence[int], tile: int | Sequence[int], overlap: int) -> "TileGrid":
        """Cores of at most ``tile`` voxels per axis, as equal as they can be."""
        sizes = (int(tile),) * 3 if np.isscalar(tile) else tuple(int(t) for t in tile)
        if len(sizes) != 3 or min(sizes) < 1:
            raise ValueError("tile must be a positive size or a (z, y, x) triple of them")
        edges = []
        for n, t in zip(shape, sizes):
            count = max(1, -(-int(n) // t))
            edges.append(tuple(int(round(k * int(n) / count)) for k in range(count + 1)))
        return cls(tuple(int(n) for n in shape), tuple(edges), int(overlap))

    @property
    def counts(self) -> tuple[int, int, int]:
        return tuple(len(e) - 1 for e in self.edges)

    def indices(self) -> list[tuple[int, int, int]]:
        return list(itertools.product(*[range(c) for c in self.counts]))

    def core(self, index: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """``[low, high)`` voxel indices, ``(z, y, x)``."""
        low = np.array([self.edges[a][index[a]] for a in range(3)])
        high = np.array([self.edges[a][index[a] + 1] for a in range(3)])
        return low, high

    def padded(self, index: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.core(index)
        return np.maximum(low - self.overlap, 0), np.minimum(high + self.overlap, self.shape)

    def owner(self, points: np.ndarray) -> np.ndarray:
        """The core ``(z, y, x)`` index of every ``(x, y, z)`` point (voxel units).

        Points outside the scan belong to the nearest core."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty((len(points), 3), dtype=int)
        for a in range(3):
            inner = np.asarray(self.edges[a][1:-1], dtype=np.float64)
            out[:, a] = np.searchsorted(inner, points[:, 2 - a], side="right")
        return out

    def central(self) -> tuple[int, int, int]:
        return tuple(c // 2 for c in self.counts)


def tile_key(index: Sequence[int]) -> str:
    return "z{}_y{}_x{}".format(*index)


@dataclass
class TileFit:
    """One tile's fit in scan coordinates (voxels, ``(x, y, z)``)."""

    index: tuple[int, int, int]
    centerlines: list[np.ndarray]
    radii: np.ndarray
    types: np.ndarray
    support: np.ndarray
    levels: Levels
    confidence: list[np.ndarray] | None = None
    seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "tile": list(self.index),
            "seconds": round(self.seconds, 3),
            "levels": asdict(self.levels),
            "fibers": [
                {
                    "centerline": np.round(line, 4).tolist(),
                    "radius": float(r),
                    "type": int(t),
                    "support": float(s),
                    **({"confidence": np.round(self.confidence[i], 3).tolist()} if self.confidence is not None else {}),
                }
                for i, (line, r, t, s) in enumerate(zip(self.centerlines, self.radii, self.types, self.support))
            ],
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileFit":
        fibers = data["fibers"]
        return cls(
            index=tuple(int(i) for i in data["tile"]),
            centerlines=[np.asarray(f["centerline"], dtype=np.float64).reshape(-1, 3) for f in fibers],
            radii=np.array([f["radius"] for f in fibers], dtype=np.float64),
            types=np.array([f.get("type", 0) for f in fibers], dtype=int),
            support=np.array([f.get("support", 0.0) for f in fibers], dtype=np.float64),
            levels=Levels(**data["levels"]),
            confidence=[np.asarray(f["confidence"], dtype=np.float64) for f in fibers] if all("confidence" in f for f in fibers) else None,
            seconds=float(data.get("seconds", 0.0)),
            history=data.get("history", []),
        )


# -- the tiled fit ---------------------------------------------------------------


def fit_tiled(
    volume: Any,
    voxel_size: float,
    spec: FiberSpec | Sequence[FiberSpec],
    settings: FitSettings | None = None,
    *,
    tile: int | Sequence[int] = 256,
    overlap: int | None = None,
    checkpoint: str | Path | None = None,
    restart: bool = False,
    tiles: Iterable[Sequence[int]] | None = None,
    exclude: Any | None = None,
    workers: int = 1,
    verbose: bool = False,
) -> FitResult:
    """:func:`fit_fibers` on a scan of any size, one tile at a time.

    ``volume`` (and ``exclude``) can be anything that slices like a
    ``(z, y, x)`` NumPy array, e.g. a ``numpy.memmap``, ``tifffile.memmap``
    or a zarr array (see :func:`open_scan`); only one padded tile is read
    into memory at a time.

    ``tile`` is the core size in voxels (one size, or ``(z, y, x)``); the
    grid splits each axis into equal cores no larger than that. ``overlap``
    is the padding on each side (voxels; default the larger of 16 and two
    of the largest diameters), and fits are stitched within half of it of
    each core wall. ``checkpoint`` is a directory that keeps every finished
    tile, so running again with it resumes (see the module notes);
    ``restart=True`` deletes that directory's tiles first. ``tiles`` fits
    only the listed ``(z, y, x)`` tile indices (the checkpoint then holds a
    part of the scan, e.g. for several processes or machines sharing one
    run); the result stitches every tile the checkpoint holds.

    ``workers`` > 1 fits that many tiles at once, each in its own process
    (most of a tile's fit is single-threaded Python, and the GPU solve is a
    small share of it, so tiles run side by side scale with the CPU cores).
    Each worker reads one tile and holds its own copies of it, so memory
    grows with ``workers``. Worker processes are spawned, so a script that
    calls this needs the ``if __name__ == "__main__":`` guard.

    Returns the stitched fit of the whole scan (``history`` holds one entry
    per tile). Settings and specs are those of :func:`fit_fibers`.
    """
    settings = settings or FitSettings()
    specs = [spec] if isinstance(spec, FiberSpec) else list(spec)
    if not specs:
        raise ValueError("give at least one FiberSpec")
    shape = tuple(int(n) for n in volume.shape)
    if len(shape) != 3:
        raise ValueError("volume must be a 3D (z, y, x) array")
    if exclude is not None and tuple(exclude.shape) != shape:
        raise ValueError("exclude must have the same shape as volume")
    largest = max(item.diameter for item in specs) / voxel_size
    overlap = int(np.ceil(max(16.0, 2.0 * largest))) if overlap is None else int(overlap)
    grid = TileGrid.make(shape, tile, overlap)
    mask_value = _mask_value(volume)
    plain_grey = (
        mask_value is None
        and settings.levels is None
        and all(item.intensity is None and item.profile is None for item in specs)
    )

    manifest = {
        "schema": SCHEMA,
        "shape_zyx": list(shape),
        "edges_zyx": [list(e) for e in grid.edges],
        "overlap": overlap,
        "voxel_size": voxel_size,
        "specs": [asdict(item) for item in specs],
        "settings": asdict(settings),
        "mask": mask_value is not None,
        "volume_sample": _fingerprint(volume),
        "exclude_sample": _fingerprint(exclude) if exclude is not None else None,
    }
    store = Checkpoint(checkpoint) if checkpoint is not None else None
    done: dict[tuple[int, int, int], TileFit] = {}
    levels = settings.levels
    if store is not None:
        if restart:
            store.clear()
        saved = store.open(manifest)
        if plain_grey and saved.get("levels"):
            levels = Levels(**saved["levels"])
        done = store.load()

    wanted = grid.indices() if tiles is None else [tuple(int(i) for i in index) for index in tiles]
    for index in wanted:
        if not all(0 <= index[a] < grid.counts[a] for a in range(3)):
            raise ValueError(f"tile {tuple(index)} is outside the {grid.counts} grid")
    todo = [index for index in wanted if index not in done]
    if plain_grey and levels is None:
        # The central tile (the one most likely to hold fibers) sets the levels.
        center = grid.central()
        todo.sort(key=lambda index: 0 if index == center else 1)
    started = time.perf_counter()
    finished = 0

    def finish(fit: TileFit) -> None:
        nonlocal levels, finished
        done[fit.index] = fit
        finished += 1
        if plain_grey and levels is None:
            levels = fit.levels
            if store is not None:
                store.set_levels(levels)
        if store is not None:
            store.save(fit)
        if verbose:
            elapsed = time.perf_counter() - started
            left = elapsed / finished * (len(todo) - finished)
            print(
                f"tile {tile_key(fit.index)} ({finished}/{len(todo)}): {len(fit.centerlines)} fibers in "
                f"{fit.seconds:.1f} s; about {left / 60:.1f} min left"
            )

    def job(index: tuple[int, int, int]) -> tuple:
        tile_settings = settings.replace(levels=levels) if plain_grey and levels is not None else settings
        return (index, *_read_tile(volume, exclude, grid, index, mask_value), voxel_size, specs, tile_settings, mask_value)

    queue = list(todo)
    if plain_grey and levels is None and queue:
        finish(_fit_crop(*job(queue.pop(0))))  # the levels tile, before any other
    if workers <= 1 or len(queue) <= 1:
        for index in queue:
            finish(_fit_crop(*job(index)))
    else:
        import multiprocessing
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        waiting = iter(queue)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            running = set()

            def submit() -> None:
                index = next(waiting, None)
                if index is not None:  # only ``workers`` tiles are read into memory at a time
                    running.add(pool.submit(_fit_crop, *job(index)))

            for _ in range(workers):
                submit()
            while running:
                complete, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in complete:
                    running.discard(future)
                    finish(future.result())
                    submit()
    if levels is None and done:
        levels = next(iter(done.values())).levels
    return stitch(grid, list(done.values()), specs, voxel_size, levels or Levels(0.0, 1.0, 0.5))


def _read_tile(volume, exclude, grid, index, mask_value) -> tuple:
    """The padded tile's scan (as the fit takes it), its exclude mask and its origin (x, y, z)."""
    low, high = grid.padded(index)
    window = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    crop = np.asarray(volume[window])
    if mask_value is not None:
        crop = crop.astype(bool) if crop.dtype == bool else crop == mask_value
    blocked = np.asarray(exclude[window], dtype=bool) if exclude is not None else None
    return crop, blocked, low[::-1].astype(np.float64)


def _fit_crop(index, crop, blocked, shift, voxel_size, specs, settings, mask_value) -> TileFit:
    """One tile's fit, in scan coordinates (runs in a worker process with ``workers`` > 1)."""
    started = time.perf_counter()
    if mask_value is not None and not crop.any():
        empty = np.zeros(0)
        return TileFit(tuple(index), [], empty, np.zeros(0, dtype=int), empty, Levels(0.0, 1.0, 0.5), [], 0.0)
    fit = fit_fibers(crop, voxel_size, specs[0] if len(specs) == 1 else specs, settings, exclude=blocked)
    types = fit.types if fit.types is not None else np.zeros(fit.fiber_count, dtype=int)
    return TileFit(
        index=tuple(index),
        centerlines=[line + shift for line in fit.centerlines],
        radii=np.asarray(fit.radii, dtype=np.float64),
        types=np.asarray(types, dtype=int),
        support=np.asarray(fit.support, dtype=np.float64),
        levels=fit.levels,
        confidence=fit.confidence if fit.confidence is not None or fit.centerlines else [],
        seconds=time.perf_counter() - started,
        history=_plain(fit.history),
    )


def _mask_value(volume: Any) -> Any | None:
    """The fiber value when ``volume`` is a binary mask, else None.

    Decided once for the scan (a tile of a grey scan can look binary, and a
    tile of a mask can be all void), from a strided sample, as
    :func:`fit_fibers` decides it for a whole volume."""
    if np.dtype(volume.dtype) == bool:
        return True
    values = np.unique(_sample(volume))
    if values.size > 2:
        return None
    return values[-1].item() if values.size == 2 else None


def _sample(volume: Any, target: int = 1 << 20) -> np.ndarray:
    shape = volume.shape
    step = max(1, int(np.ceil((np.prod(shape, dtype=np.float64) / target) ** (1.0 / 3.0))))
    return np.ascontiguousarray(np.asarray(volume[::step, ::step, ::step]))


def _fingerprint(volume: Any) -> str:
    """A hash of a strided sample, to catch resuming a checkpoint on another scan."""
    sample = _sample(volume)
    digest = hashlib.sha1(sample.tobytes())
    digest.update(str((sample.dtype.str, tuple(volume.shape))).encode())
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """``value`` with NumPy scalars and arrays turned into JSON types."""
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


# -- checkpoints -------------------------------------------------------------------


class Checkpoint:
    """A directory of finished tiles: ``manifest.json`` and ``tiles/<key>.json``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.tiles = self.directory / "tiles"
        self.manifest_path = self.directory / "manifest.json"

    def clear(self) -> None:
        """Delete the tiles and manifest this class wrote (nothing else)."""
        if self.tiles.is_dir():
            for path in self.tiles.glob("*.json"):
                path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def open(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Start a checkpoint, or check that a saved one was made from the same inputs."""
        manifest = _plain(manifest)
        if self.manifest_path.exists():
            saved = json.loads(self.manifest_path.read_text())
            different = sorted(k for k in manifest if k != "levels" and saved.get(k) != manifest[k])
            if different:
                raise ValueError(
                    f"{self.directory} holds tiles fitted from other inputs ({', '.join(different)} differ); "
                    "use another directory, or restart=True to delete its tiles"
                )
            return saved
        self.tiles.mkdir(parents=True, exist_ok=True)
        _write_json(self.manifest_path, manifest)
        return manifest

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def set_levels(self, levels: Levels) -> None:
        manifest = self.manifest()
        manifest["levels"] = asdict(levels)
        _write_json(self.manifest_path, manifest)

    def save(self, fit: TileFit) -> None:
        _write_json(self.tiles / f"{tile_key(fit.index)}.json", fit.to_dict())

    def load(self) -> dict[tuple[int, int, int], TileFit]:
        out = {}
        if self.tiles.is_dir():
            for path in sorted(self.tiles.glob("*.json")):
                fit = TileFit.from_dict(json.loads(path.read_text()))
                out[fit.index] = fit
        return out


def _write_json(path: Path, data: Any) -> None:
    """Write through a temporary file, so a stopped run never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data) + "\n")
    os.replace(temporary, path)


def load_tiles(checkpoint: str | Path) -> FitResult:
    """Stitch every tile a :func:`fit_tiled` checkpoint holds (a run can still be going)."""
    store = Checkpoint(checkpoint)
    if not store.manifest_path.exists():
        raise FileNotFoundError(f"{store.manifest_path} not found")
    manifest = store.manifest()
    specs = [_spec(item) for item in manifest["specs"]]
    grid = TileGrid(
        tuple(manifest["shape_zyx"]), tuple(tuple(e) for e in manifest["edges_zyx"]), int(manifest["overlap"])
    )
    done = store.load()
    levels = Levels(**manifest["levels"]) if manifest.get("levels") else None
    if levels is None:
        levels = next(iter(done.values())).levels if done else Levels(0.0, 1.0, 0.5)
    return stitch(grid, list(done.values()), specs, float(manifest["voxel_size"]), levels)


def _spec(values: dict[str, Any]) -> FiberSpec:
    values = dict(values)
    for key in ("intensity", "profile"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return FiberSpec(**values)


def open_scan(path: str | Path) -> Any:
    """A scan file as an array that reads from disk only what is sliced.

    ``.npy`` opens memory-mapped; a TIFF stack opens with
    ``tifffile.memmap`` when it is stored uncompressed and contiguous, else
    through ``tifffile``'s zarr store (still read tile by tile), else read
    whole."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    import tifffile

    try:
        return tifffile.memmap(path, mode="r")
    except ValueError:
        pass
    try:
        import zarr

        return zarr.open(tifffile.imread(path, aszarr=True), mode="r")
    except ImportError:
        return tifffile.imread(path)


# -- stitching ---------------------------------------------------------------------


@dataclass
class _End:
    stretch: int
    side: int  # 0: the stretch's first node, 1: its last
    tile: tuple[int, int, int]
    target: tuple[int, int, int] | None  # the core the fit goes on into, None where it ends
    node: np.ndarray  # the stretch's end node
    band: np.ndarray  # the fit near this end, in order along it: the stretch's last ``band`` voxels and the tail past the wall
    tail: np.ndarray  # the fit past the wall (empty where it ends)
    outward: np.ndarray | None  # unit direction the stretch leaves its end in (None for a one-node stretch)


def stitch(
    grid: TileGrid, fits: list[TileFit], specs: list[FiberSpec], voxel_size: float, levels: Levels
) -> FitResult:
    """Join the tiles' fits into one :class:`FitResult` (see the module notes)."""
    band = 0.5 * grid.overlap
    fits = sorted(fits, key=lambda fit: fit.index)  # the same result whatever order the tiles finished in
    stretches: list[dict[str, Any]] = []
    ends: list[_End] = []
    for fit in fits:
        for f, line in enumerate(fit.centerlines):
            if len(line) == 0:
                continue
            owners = grid.owner(line)
            own = np.all(owners == np.asarray(fit.index), axis=1)
            confidence = fit.confidence[f] if fit.confidence is not None else None
            for start, stop in _runs(own):
                sid = len(stretches)
                stretches.append(
                    {
                        "nodes": line[start:stop],
                        "confidence": confidence[start:stop] if confidence is not None else None,
                        "radius": float(fit.radii[f]),
                        "type": int(fit.types[f]),
                        "support": float(fit.support[f]),
                    }
                )
                ends.append(_end(sid, 0, fit.index, line, owners, start, -1, band))
                ends.append(_end(sid, 1, fit.index, line, owners, stop - 1, +1, band))

    links = _match(ends, stretches)
    overlap_joins = len(links) // 2
    _join_gaps(ends, stretches, links, grid, band)
    chains = _chains(len(stretches), links)

    min_length = [(item.min_length or 3.0 * item.diameter) / voxel_size for item in specs]
    by_key = {(e.stretch, e.side): e for e in ends}
    with_confidence = all(fit.confidence is not None or not fit.centerlines for fit in fits)
    lines, radii, types, support, confidence = [], [], [], [], []
    dropped = 0
    for chain in chains:
        parts = [stretches[sid]["nodes"] if enter == 0 else stretches[sid]["nodes"][::-1] for sid, enter in chain]
        line = np.concatenate(parts)
        first, last = chain[0], chain[-1]
        loose = by_key[first].target is not None or by_key[(last[0], 1 - last[1])].target is not None
        kind = stretches[first[0]]["type"]
        if len(line) < 2 or (loose and polyline_length(line) < min_length[kind]):
            dropped += 1
            continue
        weights = np.array([polyline_length(p) if len(p) > 1 else 0.0 for p in parts]) + 1e-6
        lines.append(line)
        radii.append(float(np.average([stretches[sid]["radius"] for sid, _ in chain], weights=weights)))
        types.append(kind)
        support.append(float(np.average([stretches[sid]["support"] for sid, _ in chain], weights=weights)))
        if with_confidence:
            confidence.append(
                np.concatenate([stretches[sid]["confidence"][:: 1 if enter == 0 else -1] for sid, enter in chain])
            )
    history = [
        {
            "stage": "tiles",
            "grid_zyx": list(grid.counts),
            "overlap": grid.overlap,
            "tiles": len(fits),
            "stretches": len(stretches),
            "joins": len(links) // 2,
            "gap_joins": len(links) // 2 - overlap_joins,
            "dropped_edge_pieces": dropped,
            "fibers": len(lines),
            "seconds": round(sum(fit.seconds for fit in fits), 2),
        }
    ] + [
        {"stage": f"tile {tile_key(fit.index)}", "fibers": len(fit.centerlines), "seconds": round(fit.seconds, 2)}
        for fit in fits
    ]
    multi = len(specs) > 1
    return FitResult(
        shape=grid.shape,
        voxel_size=voxel_size,
        spec=specs[0],
        centerlines=lines,
        radii=np.asarray(radii, dtype=np.float64),
        support=np.asarray(support, dtype=np.float64),
        levels=levels,
        history=history,
        specs=specs if multi else None,
        types=np.asarray(types, dtype=int) if multi else None,
        confidence=confidence if with_confidence else None,
    )


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of every run of True."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    change = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(change[::2], change[1::2])]


def _walk(line: np.ndarray, start: int, step: int, reach: float) -> np.ndarray:
    """Nodes from ``start`` in direction ``step`` while within ``reach`` of arc length."""
    out = [line[start]]
    travelled = 0.0
    k = start + step
    while 0 <= k < len(line):
        travelled += float(np.linalg.norm(line[k] - line[k - step]))
        if travelled > reach:
            break
        out.append(line[k])
        k += step
    return np.array(out)


def _end(sid, side, tile, line, owners, k, step, band) -> _End:
    """The end of the stretch at node ``k``; the fit goes on at ``k + step`` if there is such a node."""
    inward = _walk(line, k, -step, band)  # the stretch from its end node inward
    target = None
    tail = np.zeros((0, 3))
    nxt = k + step
    if 0 <= nxt < len(line):
        target = tuple(int(i) for i in owners[nxt])
        tail = _walk(line, nxt, step, band)
    ordered = np.concatenate([inward[::-1], tail])  # inward end ... end node, tail ...
    outward = None
    if len(inward) > 1:
        chord = line[k] - inward[min(len(inward) - 1, 4)]
        outward = chord / max(float(np.linalg.norm(chord)), 1e-12)
    return _End(sid, side, tuple(tile), target, line[k], ordered, tail, outward)


def _distance_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    if len(line) == 1:
        return np.linalg.norm(points - line[0], axis=1)
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.einsum("pij,ij->pi", points[:, None, :] - a[None], ab) / np.maximum((ab * ab).sum(axis=1), 1e-12)
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def _match(ends: list[_End], stretches: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[int, int]]:
    """Join stretch ends across core walls where two tiles' fits agree, best first."""
    by_tile: dict[tuple[int, int, int], list[_End]] = {}
    for end in ends:
        by_tile.setdefault(end.tile, []).append(end)
    candidates = []
    for a in ends:
        if a.target is None:
            continue
        ra = stretches[a.stretch]["radius"]
        for b in by_tile.get(a.target, []):
            if b.target is not None and b.target != a.tile:
                continue
            if stretches[b.stretch]["type"] != stretches[a.stretch]["type"]:
                continue
            if b.target is not None and (b.stretch, b.side) < (a.stretch, a.side):
                continue  # both are wall crossings: scored once
            reach = max(np.linalg.norm(a.tail[-1] - a.node), 1.0) + 2.0 * ra
            if np.linalg.norm(b.node - a.node) > reach:
                continue
            tolerance = max(1.0, 0.75 * min(ra, stretches[b.stretch]["radius"]))
            score = float(_distance_to_polyline(a.tail, b.band).mean())
            if b.target is not None:
                score = max(score, float(_distance_to_polyline(b.tail, a.band).mean()))
            if score < tolerance:
                candidates.append((score, (a.stretch, a.side), (b.stretch, b.side)))
    candidates.sort(key=lambda item: item[0])
    links: dict[tuple[int, int], tuple[int, int]] = {}
    for _, first, second in candidates:
        if first in links or second in links or first[0] == second[0]:
            continue
        links[first] = second
        links[second] = first
    return links


def _join_gaps(
    ends: list[_End], stretches: list[dict[str, Any]], links: dict[tuple[int, int], tuple[int, int]],
    grid: TileGrid, band: float, max_angle_degrees: float = 30.0,
) -> None:
    """Join, end to end, free ends that face each other across a core wall.

    The overlap match needs both tiles to have fitted the same stretch past
    the wall. Where one tile's fit stops short of it (a void trim, a split
    near the wall), the two pieces meet at the wall instead: each ends
    within ``band`` of it, in neighbouring cores, pointing at the other
    (both within ``max_angle_degrees`` of the line between them, which is
    at most ``band`` long), with the same type. Nearest pairs first.
    """
    cos = np.cos(np.radians(max_angle_degrees))
    free = [
        e for e in ends
        if (e.stretch, e.side) not in links and e.outward is not None and _wall_distance(grid, e.node, e.tile) <= band
    ]
    candidates = []
    for i, a in enumerate(free):
        for b in free[i + 1:]:
            if a.tile == b.tile or a.stretch == b.stretch:
                continue
            if stretches[a.stretch]["type"] != stretches[b.stretch]["type"]:
                continue
            if not np.all(np.abs(np.subtract(a.tile, b.tile)) <= 1):
                continue
            gap = b.node - a.node
            length = float(np.linalg.norm(gap))
            if length > band:
                continue
            if length > 1e-9:
                direction = gap / length
                if a.outward @ direction < cos or -(b.outward @ direction) < cos:
                    continue
            elif a.outward @ -b.outward < cos:
                continue
            candidates.append((length, (a.stretch, a.side), (b.stretch, b.side)))
    candidates.sort(key=lambda item: item[0])
    for _, first, second in candidates:
        if first in links or second in links:
            continue
        links[first] = second
        links[second] = first


def _wall_distance(grid: TileGrid, point: np.ndarray, tile: tuple[int, int, int]) -> float:
    """Distance from ``point`` (x, y, z) to the nearest wall of its core shared with another core."""
    low, high = grid.core(tile)
    best = np.inf
    for a in range(3):  # z, y, x
        coordinate = float(point[2 - a])
        if low[a] > 0:
            best = min(best, abs(coordinate - low[a]))
        if high[a] < grid.shape[a]:
            best = min(best, abs(high[a] - coordinate))
    return best


def _chains(count: int, links: dict[tuple[int, int], tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Joined stretches in order, each as ``(stretch, side it is entered from)``."""
    seen = [False] * count
    chains = []

    def walk(sid: int, enter: int) -> list[tuple[int, int]]:
        chain = []
        while not seen[sid]:
            seen[sid] = True
            chain.append((sid, enter))
            nxt = links.get((sid, 1 - enter))
            if nxt is None:
                break
            sid, enter = nxt
        return chain

    for sid in range(count):  # chains with a free end first, entered from it
        if seen[sid]:
            continue
        for side in (0, 1):
            if (sid, side) not in links:
                chains.append(walk(sid, side))
                break
    for sid in range(count):  # what is left are closed loops
        if not seen[sid]:
            chains.append(walk(sid, 0))
    return chains
