"""Find the fibers in a CT scan with the trained CT map network, in one call that returns Tangle's own fit.

    >>> import tangle.ct as ct
    >>> settings = ct.NetworkSettings(network="best.pt", diameters=[12e-6, 30e-6], center_crop=256)
    >>> fit = ct.find_fibers("my_scan.tif", settings)     # also writes my_scan_fibers/: stacks, tables, summary
    >>> relaxed, run = fit.relax()

This runs the tutorial's ``find_fibers.py`` (``docs/ct_unet_tutorial.md``) from a copy of the network's code in
``tangle.ct._network``: the same scan reading, network, tracer and outputs, with the settings as one object in
meters. It needs PyTorch (``pip install torch``) and the trained network as a local ``.pt`` file.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ._fit import FitResult, load_fit


@dataclass(frozen=True)
class NetworkSettings:
    """What :func:`find_fibers` is told about the scan and the sample, and what it writes.

    Lengths are in meters, as everywhere in ``tangle.ct``; ``crop`` and
    ``center_crop`` count voxels of the scan. Only ``network`` is needed. Each
    field is the tutorial's ``find_fibers.py`` option of the same name, and
    ``docs/ct_unet_tutorial.md`` says what to change when a result looks wrong.

    ``network``
        The trained network, a ``.pt`` file on this computer.
    ``voxel_size``
        The voxel size. Default: what the scan file says (an error when it
        says nothing).
    ``diameters``
        The fiber diameters, one per fiber type. The network finds fibers
        more accurately when told them, and they decide each fiber's type.
        Default: the network measures every fiber and the types are guessed.
    ``types``
        How many fiber types there are, when their diameters aren't known.
    ``bonded``
        Whether the fibers are bonded (binder where they cross); ``None``:
        not known.
    ``crop``
        The box to analyse, in voxels: ``"x=100:612,z=200:456"`` or
        ``{"x": (100, 612), "z": (200, 456)}``, ends excluded; an axis left out
        is kept whole. Default: the whole scan.
    ``center_crop``
        Analyse a cube of this many voxels from the middle of the scan.
    ``bin``
        Average ``bin`` x ``bin`` x ``bin`` voxels into one first (for fibers
        wider than 28 voxels).
    ``invert``
        The fibers are darker than the space around them.
    ``levels``
        The grey of empty space and of bright fiber, in the scan's own grey
        values. Default: the region's median and 99.5th percentile; give them
        for a mostly solid scan.
    ``min_length``
        Traced fibers shorter than this are dropped. Default: 3 diameters of
        the thinnest type.
    ``stacks``
        Write ``overlay.tif`` and ``labels.tif``; ``None``: for regions under
        600 million voxels.
    ``binder``
        Also write ``binder.tif``: the binder (as the network sees it, unless
        ``bonded`` is False) outside every traced fiber.
    ``diameter_profile``
        Also measure the diameter all along every fiber (``diameters.csv``);
        runs the network a second time.
    ``device``
        Where the network runs: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``"auto"``
        (a GPU when there is one).
    ``restart``
        Throw away the progress saved in the output's ``work`` folder instead
        of carrying on from it.
    ``dataset``, ``raw_shape``, ``raw_dtype``, ``raw_endian``, ``raw_header_bytes``
        For files holding several arrays (HDF5, Zarr, ``.npz``, ``.vti``) and
        for raw binary data (``raw_shape`` is ``(x, y, z)`` voxels).
    """

    network: str | os.PathLike
    voxel_size: float | None = None
    diameters: Sequence[float] | None = None
    types: int | None = None
    bonded: bool | None = None
    crop: str | Mapping[str, tuple[int, int]] | None = None
    center_crop: int | None = None
    bin: int = 1
    invert: bool = False
    levels: tuple[float, float] | None = None
    min_length: float | None = None
    stacks: bool | None = None
    binder: bool = False
    diameter_profile: bool = False
    device: str = "auto"
    restart: bool = False
    dataset: str | None = None
    raw_shape: tuple[int, int, int] | None = None
    raw_dtype: str | None = None
    raw_endian: str = "little"
    raw_header_bytes: int = 0


def find_fibers(scan: str | os.PathLike, settings: NetworkSettings,
                out: str | os.PathLike | None = None) -> FitResult:
    """Find the fibers in a CT scan with the trained CT map network.

    ``scan`` is a file in any format the tutorial reads (a TIFF stack, a folder of slices, NumPy, MetaImage, NRRD,
    VTK, HDF5, Zarr, NIfTI, raw). Everything the tutorial's ``find_fibers.py`` writes goes to ``out`` (default:
    ``<scan name>_fibers`` in the current folder): ``summary.txt``, ``overlay.png``, ``overlay.tif`` and
    ``labels.tif`` (to check the fibers by eye in Fiji), ``fibers.csv``, ``centerlines.csv``, ``bonds.csv``,
    ``fibers.vtk``, ``fibers.dump``, ``fit.json`` and ``run.json``. Progress is printed as it goes, and saved in
    ``out/work``, so a stopped run carries on where it stopped and a rerun that changes only the tracing or the
    outputs doesn't run the network again.

    Returns the fibers as a :class:`FitResult`, as ``load_fit`` reads ``fit.json`` back: ready for
    ``to_assembly()``, ``relax()`` and the other exports. Three more attributes: ``folder``, where everything was
    written; ``binder``, with ``settings.binder``, the binder as a boolean ``(z, y, x)`` voxel mask as in
    ``binder.tif`` (else None), which ``tangle.fem.hex_mesh(..., binder=fit.binder)`` meshes; and ``bonds``, a list
    of the bonds the network found, which :func:`add_bpm_bonds` puts into a bonded-particle model. Each bond is a
    dict with ``fibers`` (the two fibers' indices in ``centerlines``), ``position`` (x, y, z in voxels, as the
    centerlines), ``strength`` (the height of the network's bond-point peak; NaN for a bond found where its binder
    joins two fibers with no peak) and ``binder_voxels`` (how many binder voxels join the pair there; 0 for a
    bond-point bond).
    """
    from ._network import find_fibers as pipeline  # the network needs PyTorch: imported only when used

    args = pipeline.arguments().parse_args(_command_line(scan, settings, out))
    folder = Path(pipeline.run(args))
    fit = load_fit(folder / "fit.json")
    fit.folder = folder
    fit.bonds = _read_bonds(folder / "bonds.csv", fit.voxel_size)
    fit.binder = _read_binder(folder / "binder.tif")
    return fit


def _command_line(scan, settings: NetworkSettings, out) -> list[str]:
    """``settings`` as the tutorial's ``find_fibers.py`` options (lengths in micrometers)."""

    def um(meters: float) -> str:
        return f"{meters * 1e6:.12g}um"

    s = settings
    argv = [str(Path(scan).expanduser()), "--weights", str(Path(s.network).expanduser())]
    if s.voxel_size is not None:
        argv += ["--voxel-size", um(s.voxel_size)]
    if s.diameters:
        argv += ["--diameters", ",".join(um(d) for d in s.diameters)]
    if s.types is not None:
        argv += ["--types", str(s.types)]
    argv += ["--bonded", {True: "yes", False: "no", None: "unknown"}[s.bonded]]
    if s.crop:
        crop = s.crop if isinstance(s.crop, str) else ",".join(f"{axis}={a}:{b}" for axis, (a, b) in s.crop.items())
        argv += ["--crop", crop]
    if s.center_crop:
        argv += ["--center-crop", str(s.center_crop)]
    if s.bin != 1:
        argv += ["--bin", str(s.bin)]
    if s.invert:
        argv.append("--invert")
    if s.levels is not None:
        argv.append(f"--levels={s.levels[0]:.12g},{s.levels[1]:.12g}")  # "=": a grey value may be negative
    if s.min_length is not None:
        argv += ["--min-length", um(s.min_length)]
    argv += ["--stacks", {True: "yes", False: "no", None: "auto"}[s.stacks]]
    if s.binder:
        argv.append("--binder")
    if s.diameter_profile:
        argv.append("--diameter-profile")
    argv += ["--device", s.device]
    if s.restart:
        argv.append("--restart")
    if s.dataset:
        argv += ["--dataset", s.dataset]
    if s.raw_shape:
        argv += ["--raw-shape", "x".join(str(n) for n in s.raw_shape)]
    if s.raw_dtype:
        argv += ["--raw-dtype", s.raw_dtype]
    if s.raw_endian != "little":
        argv += ["--raw-endian", s.raw_endian]
    if s.raw_header_bytes:
        argv += ["--raw-header-bytes", str(s.raw_header_bytes)]
    if out is not None:
        argv += ["--out", str(Path(out).expanduser())]
    return argv


def _read_bonds(path: Path, voxel_size: float) -> list[dict]:
    """``bonds.csv`` as dicts in the fit's terms: fiber indices from 0, positions in voxels."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scale = 1e-6 / voxel_size  # micrometers to voxels of the analysed region
    return [{"fibers": (int(row["fiber_a"]) - 1, int(row["fiber_b"]) - 1),
             "position": np.array([float(row[f"{axis}_um"]) * scale for axis in "xyz"]),
             "strength": float(row["strength"]) if row["strength"] else float("nan"),
             "binder_voxels": int(row.get("binder_voxels") or 0)} for row in rows]


def _read_binder(path: Path):
    """``binder.tif`` as a boolean (z, y, x) mask, or None without it."""
    if not path.exists():
        return None
    import tifffile

    return tifffile.imread(path) > 0


def add_bpm_bonds(path: str | os.PathLike, bonds, voxel_size: float, bond_type: int = 2) -> int:
    """Bond fibers to each other in a bonded-particle (DEM) file that ``RunResult.export_bpm`` wrote: where the
    network found binder or a bond point holding two fibers together, so the model holds them together too.

    ``bonds`` are as :func:`find_fibers` gives them (``fit.bonds``): the two fibers' indices and a position in
    voxels of ``voxel_size``. Each pair of fibers gets one bond of ``bond_type``, at its first bond in the list,
    between the two fibers' particles nearest that position (nearest the axis, for capsules). The new bonds go
    after the file's bonds along each fiber, and its bond and bond-type counts are updated. The fibers keep their
    order through ``coarse_grained``, ``relax`` and ``export_bpm``, so fiber ``k`` is the file's molecule
    ``k + 1``. Returns how many bonds were added.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    titles = {line.split()[0]: index for index, line in enumerate(lines)
              if line.split()[:1] in (["Atoms"], ["Capsules"], ["Bonds"])}
    if "Atoms" not in titles or "Bonds" not in titles:
        raise ValueError(f"{path} has no Atoms or no Bonds section; is it a file export_bpm wrote?")

    def section(name: str) -> tuple[list[list[str]], int]:
        """A section's data rows, and where its last line (row or comment) ends."""
        rows, end = [], titles[name] + 1
        for index in range(titles[name] + 1, len(lines)):
            line = lines[index]
            if line[:1].isalpha():  # the next section's title
                break
            if line.strip():
                end = index + 1
                if not line.lstrip().startswith("#"):
                    rows.append(line.split())
        return rows, end

    atoms = section("Atoms")[0]  # id molecule type diameter density x y z
    ids = np.array([int(row[0]) for row in atoms])
    molecules = np.array([int(row[1]) for row in atoms])
    centers = np.array([[float(value) for value in row[5:8]] for row in atoms])
    half, axes = np.zeros(len(atoms)), np.zeros((len(atoms), 3))  # spheres: a point each
    if "Capsules" in titles:  # atom-id half-length axis-x axis-y axis-z
        row_of = {atom: index for index, atom in enumerate(ids)}
        for row in section("Capsules")[0]:
            half[row_of[int(row[0])]] = float(row[1])
            axes[row_of[int(row[0])]] = [float(value) for value in row[2:5]]
    by_molecule = np.argsort(molecules, kind="stable")
    sorted_molecules = molecules[by_molecule]

    def nearest(molecule: int, point: np.ndarray) -> int | None:
        """The id of the molecule's particle nearest ``point``, or None if it has none."""
        start, stop = np.searchsorted(sorted_molecules, [molecule, molecule + 1])
        mine = by_molecule[start:stop]
        if not len(mine):
            return None
        along = np.clip(((point - centers[mine]) * axes[mine]).sum(1), -half[mine], half[mine])
        gap = np.linalg.norm(centers[mine] + along[:, None] * axes[mine] - point, axis=1)
        return int(ids[mine[np.argmin(gap)]])

    pairs, new = set(), []
    for bond in bonds:
        i, j = sorted(int(fiber) for fiber in bond["fibers"])
        if i == j or (i, j) in pairs:
            continue
        point = np.asarray(bond["position"], dtype=float) * voxel_size
        a, b = nearest(i + 1, point), nearest(j + 1, point)
        if a is None or b is None:  # a fiber the file doesn't have
            continue
        pairs.add((i, j))
        new.append((a, b))
    if not new:
        return 0
    rows, end = section("Bonds")
    first = max((int(row[0]) for row in rows), default=0) + 1
    lines[end:end] = [f"{first + k} {bond_type} {a} {b}" for k, (a, b) in enumerate(new)]
    for index in range(min(titles.values())):  # the header's counts
        words = lines[index].split()
        if words[1:] == ["bonds"]:
            lines[index] = f"{len(rows) + len(new)} bonds"
        elif words[1:] == ["bond", "types"]:
            lines[index] = f"{max(int(words[0]), bond_type)} bond types"
    path.write_text("\n".join(lines) + "\n")
    return len(new)
