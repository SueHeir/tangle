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
fiber from both sides; they are joined when they lie on each other there (on
average within 1.25 of the smaller radius and nowhere a diameter apart: two
tiles' fits of one fiber drift apart a little in their padding, while a
neighbouring fiber lies a diameter away all along) and the joined fiber goes
on the way it was going; the partner can be in any of the 26 neighbouring
cores (a crossing at an edge or a corner), and the two tiles may have typed
the fiber differently. Otherwise the fiber is cut at the core wall. Ends
that meet at a wall without overlapping fits (one tile's fit stops short of
it, or the two end side by side) are joined when they point at each other
or sit side by side on one axis. Joins are made best first, one per stretch
end, never closing a loop, and chains of joined stretches are the stitched
fibers: a stretch's leading nodes behind the last one's end are dropped (no
step back at a join), a chain that turns over 60 degrees at a join is cut
there, and the chain takes the largest type covering a third of it (a tile
fits a large fiber as a small one far more often than the reverse). Pieces shorter
than the type's minimum length that end at a wall without a partner are
dropped (they are the other tile's fiber, seen from the edge).

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
from ._geometry import UnionFind, polyline_length, runs
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
    band: np.ndarray  # the fit near this end, in order along it: the stretch's last ``2 band`` voxels and the tail past the wall
    tail: np.ndarray  # the fit past the wall (empty where it ends)
    outward: np.ndarray | None  # unit direction the stretch leaves its end in (None when the fit has no other node
    # within ``band`` on the stretch's side: a one-node stretch at the fit's first or last node)


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
            for start, stop in runs(own):
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
    size = [item.diameter for item in specs]
    by_key = {(e.stretch, e.side): e for e in ends}
    with_confidence = all(fit.confidence is not None or not fit.centerlines for fit in fits)
    lines, radii, types, support, confidence = [], [], [], [], []
    dropped = turned = 0
    pieces = []  # (parts, kept, loose)
    for chain in chains:
        parts, kept = _parts(chain, stretches, with_confidence)
        first, last = chain[0], chain[-1]
        start_loose = by_key[first].target is not None
        end_loose = by_key[(last[0], 1 - last[1])].target is not None
        cuts = _turned_joins(parts)
        turned += len(cuts)
        bounds = [0, *cuts, len(parts)]
        for n, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            loose = (start_loose if n == 0 else True) or (end_loose if b == len(parts) else True)
            pieces.append((parts[a:b], kept[a:b], loose))
    for parts, kept, loose in pieces:
        line = np.concatenate([nodes for nodes, _ in parts])
        weights = np.array([polyline_length(nodes) if len(nodes) > 1 else 0.0 for nodes, _ in parts]) + 1e-6
        kinds = np.array([stretches[sid]["type"] for sid, _ in kept])
        kind = _chain_type(kinds, weights, size)
        if len(line) < 2 or (loose and polyline_length(line) < min_length[kind]):
            dropped += 1
            continue
        same = kinds == kind
        lines.append(line)
        radii.append(float(np.average([stretches[sid]["radius"] for sid, _ in kept], weights=weights * same + 1e-12)))
        types.append(kind)
        support.append(float(np.average([stretches[sid]["support"] for sid, _ in kept], weights=weights)))
        if with_confidence:
            confidence.append(np.concatenate([values for _, values in parts]))
    history = [
        {
            "stage": "tiles",
            "grid_zyx": list(grid.counts),
            "overlap": grid.overlap,
            "tiles": len(fits),
            "stretches": len(stretches),
            "joins": len(links) // 2,
            "gap_joins": len(links) // 2 - overlap_joins,
            "cross_type_joins": sum(
                stretches[a[0]]["type"] != stretches[b[0]]["type"] for a, b in links.items() if a < b
            ),
            "turned_joins_cut": turned,
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


def _chain_type(kinds: np.ndarray, weights: np.ndarray, size: list[float], share: float = 1.0 / 3.0) -> int:
    """The type of a chain whose stretches the tiles typed ``kinds`` (lengths ``weights``): the largest type
    covering at least ``share`` of it, else the most of it. A tile fits a large fiber as a small one far more
    often than the reverse (off a dim core, or as two lobes), and a large-typed stretch the stitch joined to a
    small one lies on it, within the small radius."""
    total = float(weights.sum())
    for k in sorted(set(kinds.tolist()), key=lambda k: -size[k]):
        if weights[kinds == k].sum() >= share * total:
            return int(k)
    return int(max(set(kinds.tolist()), key=lambda k: (weights[kinds == k].sum(), -k)))


def _turned_joins(parts, reach: float = 10.0, max_turn_degrees: float = 60.0) -> list[int]:
    """Indices of the parts that begin at a join where the fiber turns over ``max_turn_degrees`` between the
    ``reach`` voxels before and after it (the stitch cuts the chain there: the joins each looked right, but
    together they fold the fiber)."""
    cos = np.cos(np.radians(max_turn_degrees))
    cuts = []
    for i in range(1, len(parts)):
        before = np.concatenate([nodes for nodes, _ in parts[:i]])[::-1]
        after = np.concatenate([before[:1], *[nodes for nodes, _ in parts[i:]]])
        a, b = _arc_point(before, reach), _arc_point(after, reach)
        if a is None or b is None:
            continue
        u, v = before[0] - a, b - after[0]
        if float(u @ v) < cos * float(np.linalg.norm(u) * np.linalg.norm(v)):
            cuts.append(i)
    return cuts


def _arc_point(points: np.ndarray, reach: float) -> np.ndarray | None:
    """The first point at least ``reach`` of arc along ``points``, None when they are shorter than half of it."""
    if len(points) < 2:
        return None
    arc = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    k = int(np.searchsorted(arc, reach))
    if k >= len(arc):
        return points[-1] if arc[-1] >= 0.5 * reach else None
    return points[k + 1]


def _parts(chain, stretches, with_confidence):
    """The chain's stretches in order along it, ``[(nodes, confidence)]``, and the ``(stretch, enter)`` of each kept.

    Two tiles' fits of one fiber end a node or two apart at the wall, so the
    next stretch can begin a little behind where the last one ended: its
    leading nodes behind the last end (along the direction the last stretch
    left it in) are dropped, and a stretch that lies wholly behind it is
    left out, so the fiber never steps back at a join."""
    parts, kept = [], []
    for sid, enter in chain:
        step = 1 if enter == 0 else -1
        nodes = stretches[sid]["nodes"][::step]
        values = stretches[sid]["confidence"][::step] if with_confidence else None
        if parts and len(nodes):
            previous = np.concatenate([p for p, _ in parts])
            if len(previous) > 1:
                chord = previous[-1] - previous[max(len(previous) - 5, 0)]
                direction = chord / max(float(np.linalg.norm(chord)), 1e-12)
                ahead = (nodes - previous[-1]) @ direction > 0.0
                start = int(np.argmax(ahead)) if ahead.any() else len(nodes)
                nodes = nodes[start:]
                values = values[start:] if values is not None else None
        if len(nodes) or not parts:
            parts.append((nodes, values))
            kept.append((sid, enter))
    return parts, kept


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
    near = _walk(line, k, -step, band)  # the stretch from its end node inward: its direction ...
    inward = _walk(line, k, -step, 2.0 * band)  # ... and the band a tail is compared with, as long as the tail reaches
    target = None
    tail = np.zeros((0, 3))
    nxt = k + step
    if 0 <= nxt < len(line):
        target = tuple(int(i) for i in owners[nxt])
        tail = _walk(line, nxt, step, band)
    ordered = np.concatenate([inward[::-1], tail])  # inward end ... end node, tail ...
    outward = None
    if len(near) > 1:
        chord = line[k] - near[min(len(near) - 1, 4)]
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


def _neighbours(a: tuple[int, int, int], b: tuple[int, int, int] | None) -> bool:
    """Whether core ``b`` is ``a`` or one of its 26 neighbours."""
    return b is not None and bool(np.all(np.abs(np.subtract(a, b)) <= 1))


class _Chains(UnionFind):
    """Union-find over stretches, so no join closes a chain into a loop."""

    def __init__(self, count: int, links: dict[tuple[int, int], tuple[int, int]]) -> None:
        super().__init__()
        for first, second in links.items():
            self.union(first[0], second[0])


def _link(links, chains: _Chains, candidates) -> None:
    """Make the candidate joins ``(score, end, end)`` best first, one per end, never closing a loop."""
    candidates.sort(key=lambda item: item[0])
    for _, first, second in candidates:
        if first in links or second in links or first[0] == second[0]:
            continue
        if not chains.union(first[0], second[0]):
            continue
        links[first] = second
        links[second] = first


def _match(
    ends: list[_End], stretches: list[dict[str, Any]], max_turn_degrees: float = 60.0,
    tolerance_radii: float = 1.25, spread_radii: float = 2.0,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Join stretch ends across core walls where two tiles' fits agree, best first.

    ``a`` is a stretch end whose tile's fit goes on past the wall; ``b`` is an
    end in a neighbouring core (any of the 26, so a crossing at a core's edge
    or corner finds its partner) that ends there or goes on toward ``a``'s
    core or one of its neighbours. They agree when the fit past one wall lies
    on the other's band: on average within ``tolerance_radii`` of the smaller
    radius, and nowhere farther than ``spread_radii`` of it (one diameter:
    a steady offset between the two tiles' fits near the wall, not a
    neighbouring fiber, which lies a diameter away all along). Two wall
    crossings are scored from both sides and agree when either side does,
    the other side also staying within the spread everywhere: each tile's
    fit drifts in its padding, so one of the two tails may read far. The
    types may differ (one tile can type a fiber as another); the smaller
    radius sets the tolerance. Partners in a face-neighbouring core are
    joined first, then those in edge or corner cores, so a fit that crosses
    near an edge keeps the short stretch in the core between.

    A join must carry the fiber on the way it was going: from ``b``'s end the
    other stretch goes inward within ``max_turn_degrees`` of the direction
    ``a`` leaves its end in. A short stretch lying on ``a``'s tail agrees with
    it from either end, and entered from its far end the chain would turn
    back on itself at the wall.
    """
    from scipy.spatial import cKDTree

    cos = np.cos(np.radians(max_turn_degrees))
    face_candidates, corner_candidates = [], []
    if not ends:
        return {}
    tree = cKDTree(np.array([end.node for end in ends]))
    for a in ends:
        if a.target is None:
            continue
        ra = stretches[a.stretch]["radius"]
        reach = max(np.linalg.norm(a.tail[-1] - a.node), 1.0) + 2.0 * ra
        for k in tree.query_ball_point(a.node, reach):
            b = ends[k]
            if b.tile == a.tile or not _neighbours(a.tile, b.tile):
                continue
            if b.target is not None and not _neighbours(a.tile, b.target):
                continue
            if a.outward is not None and b.outward is not None and float(a.outward @ -b.outward) < cos:
                continue  # the chain would turn back at the join
            r = min(ra, stretches[b.stretch]["radius"])
            tolerance, spread = max(1.0, tolerance_radii * r), max(1.5, spread_radii * r)
            sides = [_distance_to_polyline(a.tail, b.band)]
            if b.target is not None and len(b.tail):
                sides.append(_distance_to_polyline(b.tail, a.band))
            if any(float(d.max()) > spread for d in sides):
                continue  # somewhere a diameter apart: a neighbouring fiber
            agree = [float(d.mean()) for d in sides if float(d.mean()) < tolerance]
            if agree:
                face = int(np.count_nonzero(np.subtract(a.tile, b.tile))) == 1
                (face_candidates if face else corner_candidates).append(
                    (min(agree), (a.stretch, a.side), (b.stretch, b.side))
                )
    links: dict[tuple[int, int], tuple[int, int]] = {}
    chains = _Chains(len(stretches), links)
    _link(links, chains, face_candidates)
    _link(links, chains, corner_candidates)  # after: a fit crossing near an edge goes through the core between
    return links


def _join_gaps(
    ends: list[_End], stretches: list[dict[str, Any]], links: dict[tuple[int, int], tuple[int, int]],
    grid: TileGrid, band: float, max_angle_degrees: float = 30.0, offset_radii: float = 1.0,
) -> None:
    """Join, end to end, free ends that meet at a core wall.

    The overlap match needs both tiles to have fitted the same stretch past
    the wall. Where one tile's fit stops short of it (a void trim, a split
    near the wall), or the two fits end side by side at it, the two pieces
    meet at the wall instead: each ends within ``band`` of it, in
    neighbouring cores, with the same type, and either

    - they point at each other: both within ``max_angle_degrees`` of the
      line between them, which is at most ``band`` long; or
    - they sit side by side or overlap a little: their directions are
      antiparallel within ``max_angle_degrees``, and the other end lies
      within ``offset_radii`` of the smaller radius of each end's axis and at
      most ``band`` along it either way.

    Nearest pairs first.
    """
    from scipy.spatial import cKDTree

    cos = np.cos(np.radians(max_angle_degrees))
    free = [
        e for e in ends
        if (e.stretch, e.side) not in links and e.outward is not None and _wall_distance(grid, e.node, e.tile) <= band
    ]
    candidates = []
    pairs = cKDTree(np.array([e.node for e in free])).query_pairs(2.0 * band) if free else set()
    for i, j in sorted(pairs):
        a, b = free[i], free[j]
        if a.tile == b.tile or a.stretch == b.stretch:
            continue
        if stretches[a.stretch]["type"] != stretches[b.stretch]["type"]:
            continue
        if not _neighbours(a.tile, b.tile):
            continue
        gap = b.node - a.node
        length = float(np.linalg.norm(gap))
        if length > 2.0 * band:
            continue
        if length > 1e-9:
            direction = gap / length
            if length <= band and a.outward @ direction >= cos and -(b.outward @ direction) >= cos:
                candidates.append((length, (a.stretch, a.side), (b.stretch, b.side)))
                continue
        if float(a.outward @ -b.outward) < cos:
            continue
        offset = offset_radii * min(stretches[a.stretch]["radius"], stretches[b.stretch]["radius"])
        along_a, along_b = float(gap @ a.outward), float(-gap @ b.outward)
        if abs(along_a) > band or abs(along_b) > band:
            continue
        if (np.linalg.norm(gap - along_a * a.outward) > offset
                or np.linalg.norm(-gap - along_b * b.outward) > offset):
            continue
        candidates.append((length, (a.stretch, a.side), (b.stretch, b.side)))
    _link(links, _Chains(len(stretches), links), candidates)


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
