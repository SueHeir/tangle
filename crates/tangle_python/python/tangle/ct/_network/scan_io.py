# Copied from examples/ct_unet/scan_io.py by its sync_package.py: change that file, then run it again.
"""Reading a CT scan from the common file formats, for find_fibers.py and check_scan.py.

A scan opens as a (z, y, x) array that reads only what is sliced, where the format allows it: z is the slice
number, y the row and x the column, as Fiji shows them (Fiji numbers slices from 1, these from 0). Formats:

* TIFF: one multi-page stack (ImageJ, OME or plain), or a folder of 2D slices (TIFF, PNG, JPEG, BMP), or a
  pattern such as ``"slices/*.png"``;
* NumPy ``.npy`` / ``.npz``; raw binary (any name, with ``--raw-shape`` and ``--raw-dtype``);
* MetaImage (``.mhd`` + data, ``.mha``), NRRD (``.nrrd``, ``.nhdr``), VTK image data (``.vti``);
* HDF5 (``.h5``, ``.hdf5``, ``.nxs``; needs h5py), Zarr folders (needs zarr), NIfTI (``.nii``, ``.nii.gz``;
  needs nibabel).

The voxel size is read from the file where it says (ImageJ and OME TIFF, MetaImage, NRRD with units, VTK
image data in meters, OME-Zarr, NIfTI, HDF5 ``element_size_um``, a SkyScan log next to the slices);
``--voxel-size`` always wins. RGB images are averaged to grey.
"""

from __future__ import annotations

import gzip
import importlib
import os
import re
import shutil
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TIFF = {".tif", ".tiff", ".btf", ".tf8", ".tf2"}
IMAGES = TIFF | {".png", ".jpg", ".jpeg", ".bmp"}
HDF5 = {".h5", ".hdf5", ".hdf", ".he5", ".nxs", ".nx5"}
FORMATS = ("a TIFF stack, a folder of slices (TIFF, PNG, JPEG, BMP), .npy, .npz, .mhd/.mha, .nrrd/.nhdr, .vti, "
           "HDF5, Zarr, NIfTI, or raw binary data with --raw-shape and --raw-dtype")


class ScanError(Exception):
    """A problem with the scan or the options, explained for the user (printed without a traceback)."""


def need(module: str, why: str, package: str | None = None):
    """Import an optional package, or explain how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError:
        name = package or module
        raise ScanError(f"{why} needs the Python package '{name}': python -m pip install {name}") from None


# -- lengths ---------------------------------------------------------------------------------------------------

_UNITS = {
    "m": 1e6, "meter": 1e6, "meters": 1e6, "metre": 1e6, "cm": 1e4, "mm": 1e3, "millimeter": 1e3,
    "millimeters": 1e3, "millimetre": 1e3, "um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometers": 1.0, "micrometre": 1.0, "nm": 1e-3, "nanometer": 1e-3, "nanometers": 1e-3,
}
_VOXEL_UNITS = {"vox", "voxel", "voxels", "px", "pixel", "pixels"}
_LENGTH = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([A-Za-zµμ]*)\s*$")


def unit_um(unit: str | None) -> float | None:
    """Micrometers per ``unit`` (None for an unknown unit or "pixel")."""
    if not unit:
        return None
    key = str(unit).strip().strip('"').lower().replace("\\u00b5", "µ")
    return _UNITS.get(key)


def parse_length(text: str, voxel_um: float | None = None, what: str = "length") -> float:
    """A length in micrometers from text such as ``1.3um``, ``1.3 µm``, ``650nm``, ``0.0013mm`` or ``1.3``
    (plain numbers are micrometers). ``8vox`` is 8 voxels, which needs ``voxel_um``."""
    match = _LENGTH.match(str(text))
    if not match:
        raise ScanError(f"Can't read '{text}' as a {what}: write it like 1.3um, 650nm or 0.0013mm "
                        "(a dot for decimals, no spaces inside the number)")
    value, unit = float(match.group(1)), match.group(2).lower()
    if unit in _VOXEL_UNITS:
        if voxel_um is None:
            raise ScanError(f"'{text}' is in voxels, which needs the voxel size")
        return value * voxel_um
    if not unit:
        return value
    scale = unit_um(unit)
    if scale is None:
        raise ScanError(f"Unknown unit in '{text}': use um, nm, mm or vox")
    return value * scale


def parse_lengths(text: str, voxel_um: float | None = None, what: str = "length") -> list[float]:
    """Comma-separated lengths, e.g. ``12um,30um``."""
    parts = [p for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ScanError(f"No {what} given")
    return [parse_length(p, voxel_um, what) for p in parts]


def parse_xyz(text: str, what: str = "--raw-shape") -> tuple[int, int, int]:
    """Three positive integers ``X x Y x Z`` (or comma-separated), width x height x slices."""
    parts = [p for p in re.split(r"[x×*,\s]+", str(text).strip().lower()) if p]
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        values = ()
    if len(values) != 3 or min(values) < 1:
        raise ScanError(f"{what} needs three whole numbers, width x height x slices, e.g. 1024x1024x800 (got '{text}')")
    return values


def human_bytes(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} TB"


# -- the scan --------------------------------------------------------------------------------------------------


@dataclass
class Scan:
    """A scan opened for reading. ``array`` is (z, y, x) and reads lazily where the format allows."""

    array: Any
    path: Path
    kind: str
    voxel_um: float | None = None
    voxel_from: str = ""
    notes: list[str] = field(default_factory=list)
    fingerprint: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.array.dtype)

    def describe(self) -> list[str]:
        """What was read, in plain words."""
        nz, ny, nx = self.shape
        size = human_bytes(nz * ny * nx * self.dtype.itemsize)
        lines = [f"{self.path.name}: {self.kind}, {nx} x {ny} x {nz} voxels (x, y, z), {self.dtype.name}, {size}"]
        if self.voxel_um:
            extent = " x ".join(_mm_or_um(n * self.voxel_um) for n in (nx, ny, nz))
            lines.append(f"Voxel size {self.voxel_um:.4g} um ({self.voxel_from}); the scan is {extent}")
        else:
            lines.append("Voxel size: the file doesn't say")
        return lines + self.notes


def _mm_or_um(um: float) -> str:
    return f"{um / 1000:.3g} mm" if um >= 1000 else f"{um:.4g} um"


def open_scan(path: str | Path, dataset: str | None = None, raw_shape: tuple[int, int, int] | None = None,
              raw_dtype: str | None = None, raw_endian: str = "little", raw_header: int = 0) -> Scan:
    """Open a scan file, folder of slices or pattern of slice files (see the module notes)."""
    text = os.path.expanduser(str(path))
    if any(c in Path(text).name for c in "*?[") and not Path(text).exists():
        import glob

        files = sorted((Path(f) for f in glob.glob(text)
                        if Path(f).suffix.lower() in IMAGES and not Path(f).name.startswith(".")), key=_natural)
        if not files:
            raise ScanError(f"No image files match {text}")
        scan = _open_slices(files, Path(text).parent)
        scan.fingerprint = _fingerprint(Path(text).parent, text, len(files), files[0].name)
        return scan
    path = Path(text)
    if not path.exists():
        raise ScanError(f"Can't find {path}")
    if raw_shape is not None or raw_dtype is not None:
        if raw_shape is None or raw_dtype is None:
            raise ScanError("Raw data needs both --raw-shape (width x height x slices) and --raw-dtype (e.g. uint16)")
        scan = _open_raw(path, raw_shape, raw_dtype, raw_endian, raw_header)
    elif path.is_dir():
        if path.suffix.lower() == ".zarr" or any((path / n).exists() for n in (".zarray", ".zgroup", "zarr.json")):
            scan = _open_zarr(path, dataset)
        else:
            scan = _open_slice_folder(path)
    else:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix in TIFF:
            scan = _open_tiff(path)
        elif suffix == ".npy":
            scan = Scan(_volume(np.load(path, mmap_mode="r"), None, path.name), path, "NumPy array")
        elif suffix == ".npz":
            scan = _open_npz(path, dataset)
        elif suffix in (".mhd", ".mha"):
            scan = _open_metaimage(path)
        elif suffix in (".nrrd", ".nhdr"):
            scan = _open_nrrd(path)
        elif suffix == ".vti":
            scan = _open_vti(path, dataset)
        elif suffix in HDF5:
            scan = _open_hdf5(path, dataset)
        elif name.endswith(".nii") or name.endswith(".nii.gz"):
            scan = _open_nifti(path)
        elif suffix in IMAGES:
            raise ScanError(f"{path.name} is one 2D image. Give the folder that holds all the slices, or a pattern "
                            f"such as \"{path.parent / ('*' + path.suffix)}\"")
        elif suffix in (".raw", ".vol", ".bin", ".dat", ".img"):
            raise ScanError(f"{path.name} looks like raw binary data, which doesn't say its size: give --raw-shape "
                            "(width x height x slices, e.g. 1024x1024x800) and --raw-dtype (e.g. uint16)")
        else:
            raise ScanError(f"Don't know how to read '{path.suffix}' files. This reads {FORMATS}. Most CT software "
                            "and Fiji can save a TIFF stack (Fiji: File > Save As > Tiff...)")
    scan.fingerprint = _fingerprint(path, dataset, raw_shape, raw_dtype, raw_endian, raw_header)
    return scan


def _fingerprint(path: Path, *options) -> dict:
    """Enough to tell whether a saved copy of the region still matches the scan."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p.name for p in path.iterdir())
        info = {"files": len(files), "first": files[0] if files else "", "last": files[-1] if files else ""}
        stat = path.stat()
    elif path.exists():
        stat = path.stat()
        info = {"bytes": stat.st_size}
    else:
        return {"path": str(path), "options": [str(o) for o in options]}
    return {"path": str(path.resolve()), "modified": int(stat.st_mtime), **info, "options": [str(o) for o in options]}


def scan_dims(scan: Scan) -> str:
    nz, ny, nx = scan.shape
    return f"{nx} x {ny} x {nz}"


# -- array views -------------------------------------------------------------------------------------------------


def _three(key) -> tuple:
    """A (z, y, x) index of slices or integers."""
    key = key if isinstance(key, tuple) else (key,)
    if len(key) > 3 or any(k is Ellipsis for k in key):
        raise IndexError("index a volume with up to three slices or integers (z, y, x)")
    return key + (slice(None),) * (3 - len(key))


class _View:
    """A (z, y, x) view of an array with extra length-1 axes and/or a trailing color axis (averaged to grey)."""

    def __init__(self, array, keep: list[int], rgb: bool):
        self.array, self.keep, self.rgb = array, keep, rgb
        self.shape = tuple(int(array.shape[k]) for k in keep)
        self.dtype = np.dtype(np.float32) if rgb else np.dtype(array.dtype)

    def __getitem__(self, key):
        index: list[Any] = [0] * len(self.array.shape)
        for k, part in zip(self.keep, _three(key)):
            index[k] = part
        if self.rgb:
            index[-1] = slice(0, 3)
        out = np.asarray(self.array[tuple(index)])
        return out.mean(axis=-1, dtype=np.float32) if self.rgb else out


def _volume(array, axes: str | None, name: str):
    """``array`` as a (z, y, x) grey volume: length-1 axes dropped, RGB averaged, anything else explained."""
    shape = tuple(int(n) for n in array.shape)
    rgb = False
    if axes and len(axes) == len(shape) and axes.upper().endswith("S") and shape[-1] in (3, 4):
        rgb = True
    elif not axes and len(shape) == 4 and shape[-1] in (3, 4) and min(shape[:3]) > 4:
        rgb = True
    core = list(range(len(shape) - 1 if rgb else len(shape)))
    keep = [k for k in core if shape[k] != 1]
    if len(keep) < 3:
        raise ScanError(f"{name} holds one 2D image ({' x '.join(map(str, shape))}), not a 3D scan. Give the "
                        "whole stack, or the folder that holds all the slices")
    if len(keep) > 3:
        raise ScanError(f"{name} holds more than one volume ({' x '.join(map(str, shape))}"
                        f"{', axes ' + axes if axes else ''}), e.g. several channels or time points. Save the one "
                        "you want as its own stack (Fiji: Image > Duplicate..., then File > Save As > Tiff...)")
    if not rgb and keep == list(range(len(shape))):
        return array
    return _View(array, keep, rgb)


class _SliceStack:
    """A folder of 2D slice files as a (z, y, x) volume, read slice by slice."""

    def __init__(self, files: list[Path], first: np.ndarray):
        self.files = files
        self.rgb = first.ndim == 3
        self.plane = first.shape[:2]
        self.shape = (len(files), *self.plane)
        self.dtype = np.dtype(np.float32) if self.rgb else first.dtype

    def read(self, k: int) -> np.ndarray:
        image = _read_image(self.files[k])
        if image.shape[:2] != self.plane:
            raise ScanError(f"{self.files[k].name} is {image.shape[1]} x {image.shape[0]} pixels, but "
                            f"{self.files[0].name} is {self.plane[1]} x {self.plane[0]}: all slices must be the same size")
        if image.ndim == 3:
            return image[..., :3].mean(axis=-1, dtype=np.float32)
        return image

    def __getitem__(self, key):
        zs, ys, xs = _three(key)
        if isinstance(zs, (int, np.integer)):
            return self.read(int(zs) % len(self.files))[ys, xs]
        planes = [self.read(k)[ys, xs] for k in range(*zs.indices(len(self.files)))]
        if not planes:
            return np.zeros((0,) + np.zeros(self.plane)[ys, xs].shape, self.dtype)
        return np.stack(planes)


class _XYZ:
    """An (x, y, z) array (NIfTI) seen as (z, y, x)."""

    def __init__(self, array, shape, dtype, extra: int):
        self.array, self.extra = array, extra
        self.shape = tuple(int(n) for n in shape[::-1])
        self.dtype = np.dtype(dtype)

    def __getitem__(self, key):
        zs, ys, xs = _three(key)
        return np.transpose(np.asarray(self.array[(xs, ys, zs) + (0,) * self.extra]))


def _natural(path) -> list:
    """Sort key that puts slice_2 before slice_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in TIFF:
        return need("tifffile", "Reading TIFF files").imread(path)
    pil = need("PIL.Image", f"Reading {path.suffix} images", "pillow")
    with pil.open(path) as image:
        if image.mode in ("P", "PA", "LA", "1"):
            image = image.convert("L")
        return np.asarray(image)


# -- formats -----------------------------------------------------------------------------------------------------


def _open_tiff(path: Path) -> Scan:
    tifffile = need("tifffile", "Reading TIFF files")
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        voxel, source, notes = _tiff_voxel(tif)
        kind = "TIFF stack (ImageJ)" if tif.is_imagej else "OME-TIFF stack" if tif.is_ome else "TIFF stack"
    try:
        array = tifffile.memmap(path, mode="r")
    except Exception:
        array = None
        try:
            zarr = importlib.import_module("zarr")
            array = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
            if not hasattr(array, "shape"):  # a pyramid opens as a group: level 0 is the full scan
                array = array["0"]
        except Exception:
            array = None
        if array is None:
            notes.append("The TIFF is compressed or not stored in one piece, so it is read whole into memory "
                         "(an uncompressed stack opens faster and lighter)")
            array = tifffile.imread(path)
    return Scan(_volume(array, axes, path.name), path, kind, voxel, source, notes)


def _tiff_voxel(tif) -> tuple[float | None, str, list[str]]:
    """Voxel size (um) from OME or ImageJ metadata; (None, "", notes) when neither says."""
    notes: list[str] = []
    sizes: dict[str, float] = {}
    source = ""
    if tif.is_ome and tif.ome_metadata:
        sizes, source = _ome_sizes(tif.ome_metadata), "OME metadata"
    if not sizes.get("X") and tif.is_imagej:
        meta = tif.imagej_metadata or {}
        scale = unit_um(meta.get("unit"))
        tag = tif.pages[0].tags.get("XResolution")
        if scale and tag is not None:
            value = tag.value
            num, den = (value if isinstance(value, tuple) else (value, 1))
            if num and den and num > 0:
                sizes = {"X": scale * den / num}
                if meta.get("spacing"):
                    sizes["Z"] = scale * float(meta["spacing"])
                source = "ImageJ metadata"
    x = sizes.get("X")
    if not x or x <= 0:
        return None, "", notes
    z = sizes.get("Z")
    if z and abs(z - x) > 0.01 * x:
        notes.append(f"Warning: the voxels are not cubes (x and y {x:.4g} um, z {z:.4g} um). The network expects "
                     f"cubic voxels: resample the scan first (Fiji: Image > Scale..., z scale {z / x:.4g})")
    return x, f"from the TIFF's {source}", notes


def _ome_sizes(xml: str) -> dict[str, float]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    except Exception:
        return {}
    for element in root.iter():
        if element.tag.endswith("Pixels"):
            sizes = {}
            for axis in "XYZ":
                value = element.get(f"PhysicalSize{axis}")
                scale = unit_um(element.get(f"PhysicalSize{axis}Unit", "µm"))
                if value and scale:
                    sizes[axis] = float(value) * scale
            return sizes
    return {}


def _open_slice_folder(folder: Path) -> Scan:
    files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGES]
    if not files:
        raise ScanError(f"{folder} holds no image slices (TIFF, PNG, JPEG or BMP files). This reads {FORMATS}")
    # Slices share a name pattern (scan_0001.tif, scan_0002.tif, ...); previews and other images don't.
    groups = Counter(re.sub(r"\d+", "#", p.name.lower()) for p in files)
    pattern, count = groups.most_common(1)[0]
    chosen = [p for p in files if re.sub(r"\d+", "#", p.name.lower()) == pattern]
    scan = _open_slices(sorted(chosen, key=_natural), folder)
    if count < len(files):
        kept = {p.name for p in chosen}
        others = sorted(p.name for p in files if p.name not in kept)
        listed = ", ".join(others[:3]) + (", ..." if len(others) > 3 else "")
        scan.notes.append(f"Using the {count} files named like {pattern.replace('#', '<n>')}; "
                          f"left out {len(others)} other image(s): {listed}")
    return scan


def _open_slices(files: list[Path], folder: Path) -> Scan:
    first = _read_image(files[0])
    if first.ndim == 3 and first.shape[-1] not in (3, 4):
        raise ScanError(f"{files[0].name} holds {first.shape[0]} images, not one slice: open it as a stack instead")
    if len(files) < 2:
        raise ScanError(f"Only one slice ({files[0].name}): a 3D scan needs a stack of slices")
    stack = _SliceStack(files, first)
    voxel, source, notes = None, "", []
    if files[0].suffix.lower() in TIFF:
        tifffile = need("tifffile", "Reading TIFF files")
        with tifffile.TiffFile(files[0]) as tif:
            voxel, source, notes = _tiff_voxel(tif)
        source = source.replace("the TIFF's", "the first slice's")
    if voxel is None:
        voxel, source = _scanner_log_voxel(folder)
    kind = f"folder of {len(files)} {files[0].suffix.lower().lstrip('.').upper()} slices"
    return Scan(stack, folder, kind, voxel, source, notes)


def _scanner_log_voxel(folder: Path) -> tuple[float | None, str]:
    """The voxel size from a SkyScan reconstruction log next to the slices, if there is one."""
    for log in sorted(folder.glob("*.log")):
        if log.name.startswith(".") or log.stat().st_size > 5_000_000:
            continue
        match = re.search(r"Image Pixel Size \(um\)\s*=\s*([0-9.]+)", log.read_text(errors="ignore"))
        if match and float(match.group(1)) > 0:
            return float(match.group(1)), f"'Image Pixel Size (um)' in {log.name}"
    return None, ""


def _open_npz(path: Path, dataset: str | None) -> Scan:
    data = np.load(path)
    names = list(data.files)
    if dataset:
        if dataset not in names:
            raise ScanError(f"{path.name} has no array '{dataset}'; it has: {', '.join(names)}")
        name = dataset
    else:
        volumes = [n for n in names if data[n].ndim >= 3]
        preferred = [n for n in ("volume", "scan", "image", "data") if n in volumes]
        if preferred:
            name = preferred[0]
        elif len(volumes) == 1:
            name = volumes[0]
        else:
            raise ScanError(f"{path.name} holds {len(volumes)} 3D arrays ({', '.join(volumes) or 'none'}): "
                            "say which with --dataset NAME")
    return Scan(_volume(data[name], None, f"{path.name}[{name}]"), path, f"NumPy archive, array '{name}'")


_DTYPES = {
    "uint8": "u1", "u1": "u1", "8bit": "u1", "8-bit": "u1", "ubyte": "u1", "byte": "u1",
    "int8": "i1", "i1": "i1",
    "uint16": "u2", "u2": "u2", "16bit": "u2", "16-bit": "u2", "ushort": "u2",
    "int16": "i2", "i2": "i2", "short": "i2",
    "uint32": "u4", "u4": "u4", "int32": "i4", "i4": "i4",
    "float32": "f4", "f4": "f4", "float": "f4", "32bit": "f4", "32-bit": "f4", "real": "f4",
    "float64": "f8", "f8": "f8", "double": "f8",
}


def _open_raw(path: Path, shape_xyz, dtype_text: str, endian: str, header: int) -> Scan:
    code = _DTYPES.get(str(dtype_text).strip().lower().replace(" ", ""))
    if code is None:
        raise ScanError(f"Unknown --raw-dtype '{dtype_text}': use uint8, uint16, int16, uint32, int32, float32 or float64")
    if endian not in ("little", "big"):
        raise ScanError("--raw-endian is little (most scanners and PCs) or big")
    dtype = np.dtype(("<" if endian == "little" else ">") + code)
    x, y, z = shape_xyz
    expected = header + x * y * z * dtype.itemsize
    size = path.stat().st_size
    notes = []
    if size < expected:
        fits = (size - header) // max(x * y * dtype.itemsize, 1)
        raise ScanError(f"{path.name} is {size:,} bytes, but {x} x {y} x {z} voxels of {dtype.name} take "
                        f"{expected - header:,} (plus {header:,} header bytes). Check --raw-shape and --raw-dtype; "
                        f"at {x} x {y} the file holds {fits} whole slices of {dtype.name}")
    if size > expected:
        notes.append(f"The file is {size - expected:,} bytes longer than {x} x {y} x {z} {dtype.name} voxels; the "
                     "extra bytes at the end are ignored (if the file starts with a header, give --raw-header-bytes)")
    array = np.memmap(path, dtype=dtype, mode="r", offset=header, shape=(z, y, x))
    return Scan(array, path, f"raw {dtype.name} data", None, "", notes)


_MET = {"MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_USHORT": "u2", "MET_SHORT": "i2", "MET_UINT": "u4", "MET_INT": "i4",
        "MET_ULONG_LONG": "u8", "MET_LONG_LONG": "i8", "MET_FLOAT": "f4", "MET_DOUBLE": "f8"}


def _header(path: Path, last_key: str, limit: int = 300) -> tuple[dict[str, str], int]:
    """``key = value`` header lines up to ``last_key``, and the byte offset just after them."""
    fields, offset = {}, 0
    with open(path, "rb") as f:
        for _ in range(limit):
            raw = f.readline()
            if not raw:
                break
            offset += len(raw)
            key, _, value = raw.decode("latin-1").partition("=")
            fields[key.strip()] = value.strip()
            if key.strip() == last_key:
                return fields, offset
    raise ScanError(f"{path.name} doesn't look like a MetaImage header (no {last_key} line)")


def _open_metaimage(path: Path) -> Scan:
    fields, offset = _header(path, "ElementDataFile")
    dims = [int(v) for v in fields.get("DimSize", "").split()]
    if len(dims) != 3 or int(fields.get("NDims", 3)) != 3:
        raise ScanError(f"{path.name} is not a 3D image (DimSize {fields.get('DimSize')})")
    if int(fields.get("ElementNumberOfChannels", 1)) != 1:
        raise ScanError(f"{path.name} has {fields['ElementNumberOfChannels']} values per voxel; save one channel")
    code = _MET.get(fields.get("ElementType", ""))
    if code is None:
        raise ScanError(f"{path.name}: element type {fields.get('ElementType')} isn't supported")
    msb = (fields.get("BinaryDataByteOrderMSB") or fields.get("ElementByteOrderMSB") or "False").lower() == "true"
    dtype = np.dtype((">" if msb else "<") + code)
    x, y, z = dims
    nbytes = x * y * z * dtype.itemsize
    name = fields["ElementDataFile"]
    if name.upper() == "LOCAL":
        source = path
    elif name.upper().startswith("LIST") or "%" in name:
        raise ScanError(f"{path.name} keeps its slices in several files, which isn't supported: save it as one "
                        "file (.mha) or as a TIFF stack")
    else:
        source, offset = path.parent / name, int(fields.get("HeaderSize", 0))
        if not source.exists():
            raise ScanError(f"{path.name} points to {name}, which isn't next to it")
        if offset < 0:
            offset = source.stat().st_size - nbytes
    notes = []
    if (fields.get("CompressedData") or "False").lower() == "true":
        with open(source, "rb") as f:
            f.seek(offset)
            data = zlib.decompress(f.read())
        array = np.frombuffer(data, dtype=dtype, count=x * y * z).reshape(z, y, x)
        notes.append("The data are compressed, so they are read whole into memory")
    else:
        array = np.memmap(source, dtype=dtype, mode="r", offset=offset, shape=(z, y, x))
    voxel, source_note = None, ""
    spacing = fields.get("ElementSpacing") or fields.get("ElementSize")
    if spacing:
        values = [float(v) for v in spacing.split()]
        if values and values[0] > 0:
            voxel = values[0] * 1000.0
            source_note = f"ElementSpacing {values[0]:g} in the header, read as millimeters as ITK writes it"
            if voxel > 100.0:
                notes.append(f"If that spacing is really in micrometers, pass --voxel-size {values[0]:g}um")
            if len(values) == 3 and abs(values[2] - values[0]) > 0.01 * values[0]:
                notes.append(f"Warning: the voxels are not cubes (spacing {spacing}); the network expects cubic voxels")
    return Scan(array, path, "MetaImage", voxel, source_note, notes)


_NRRD = {
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2", "uint16": "u2", "uint16_t": "u2",
    "short": "i2", "short int": "i2", "signed short": "i2", "signed short int": "i2", "int16": "i2", "int16_t": "i2",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "float": "f4", "double": "f8",
}


def _open_nrrd(path: Path) -> Scan:
    fields, offset = {}, 0
    with open(path, "rb") as f:
        magic = f.readline()
        offset += len(magic)
        if not magic.startswith(b"NRRD"):
            raise ScanError(f"{path.name} doesn't start like a NRRD file")
        for _ in range(500):
            raw = f.readline()
            offset += len(raw)
            line = raw.decode("latin-1").rstrip("\r\n")
            if not raw or line == "":
                break
            if line.startswith("#") or ":=" in line:
                continue
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()
    sizes = [int(v) for v in fields.get("sizes", "").split()]
    if len(sizes) != 3:
        raise ScanError(f"{path.name} is not a 3D grey image (sizes {fields.get('sizes')})")
    code = _NRRD.get(fields.get("type", "").lower())
    if code is None:
        raise ScanError(f"{path.name}: type '{fields.get('type')}' isn't supported")
    dtype = np.dtype((">" if fields.get("endian", "little") == "big" else "<") + code)
    x, y, z = sizes
    nbytes = x * y * z * dtype.itemsize
    datafile = fields.get("data file") or fields.get("datafile")
    source = path.parent / datafile if datafile else path
    if datafile:
        offset = 0
        if not source.exists():
            raise ScanError(f"{path.name} points to {datafile}, which isn't next to it")
    encoding = fields.get("encoding", "raw").lower()
    skip = int(fields.get("byte skip", 0))
    notes = []
    if encoding == "raw":
        offset = source.stat().st_size - nbytes if skip < 0 else offset + skip
        array = np.memmap(source, dtype=dtype, mode="r", offset=offset, shape=(z, y, x))
    elif encoding in ("gzip", "gz"):
        with open(source, "rb") as f:
            f.seek(offset)
            data = gzip.decompress(f.read())
        array = np.frombuffer(data, dtype=dtype, count=x * y * z).reshape(z, y, x)
        notes.append("The data are compressed, so they are read whole into memory")
    else:
        raise ScanError(f"{path.name}: '{encoding}' encoding isn't supported (raw and gzip are)")
    spacing = None
    if fields.get("spacings"):
        spacing = [float(v) for v in fields["spacings"].split()]
    elif fields.get("space directions"):
        vectors = re.findall(r"\(([^)]*)\)", fields["space directions"])
        spacing = [float(np.linalg.norm([float(c) for c in v.split(",")])) for v in vectors]
    voxel, note = None, ""
    units = re.findall(r'"([^"]*)"', fields.get("space units", ""))
    if spacing and spacing[0] > 0:
        scale = unit_um(units[0]) if units else None
        if scale:
            voxel, note = spacing[0] * scale, f"spacing {spacing[0]:g} {units[0]} in the header"
        else:
            notes.append(f"The header gives the spacing as {spacing[0]:g} without a unit: pass --voxel-size")
    return Scan(array, path, "NRRD", voxel, note, notes)


_VTK = {"Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2", "Int32": "i4", "UInt32": "u4",
        "Int64": "i8", "UInt64": "u8", "Float32": "f4", "Float64": "f8"}


def _attr(text: str, name: str) -> str | None:
    match = re.search(rf'\b{name}="([^"]*)"', text)
    return match.group(1) if match else None


def _open_vti(path: Path, dataset: str | None) -> Scan:
    """VTK image data with its arrays appended raw (as ParaView, PuMA and Tangle write it), zlib-compressed or not."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    marker = head.find(b"<AppendedData")
    if marker < 0:
        raise ScanError(f"{path.name}: only .vti files with appended data are supported (ParaView's default). Save the "
                        "volume as a TIFF stack instead (ParaView: File > Save Data > TIFF)")
    text = head[:marker].decode("latin-1")
    if _attr(head[marker:marker + 200].decode("latin-1"), "encoding") != "raw":
        raise ScanError(f"{path.name}: base64-encoded .vti isn't supported; save it with raw appended data or as TIFF")
    start = head.index(b"_", marker) + 1
    little = _attr(text, "byte_order") != "BigEndian"
    order = "<" if little else ">"
    header = np.dtype(order + ("u8" if _attr(text, "header_type") == "UInt64" else "u4"))
    compressed = _attr(text, "compressor") is not None
    arrays = []
    for match in re.finditer(r"<DataArray\b[^>]*>", text):
        tag = match.group(0)
        if (_attr(tag, "format") or "") == "appended":
            point = text.rfind("<PointData", 0, match.start()) > text.rfind("<CellData", 0, match.start())
            arrays.append((tag, point))
    if not arrays:
        raise ScanError(f"{path.name} holds no appended data arrays")
    if dataset:
        chosen = [a for a in arrays if _attr(a[0], "Name") == dataset]
        if not chosen:
            names = ", ".join(_attr(a[0], "Name") or "?" for a in arrays)
            raise ScanError(f"{path.name} has no array '{dataset}'; it has: {names}")
    else:
        chosen = [a for a in arrays if int(_attr(a[0], "NumberOfComponents") or 1) == 1]
        if not chosen:
            raise ScanError(f"{path.name} has no single-value (grey) array")
    tag, point = chosen[0]
    kind = _attr(tag, "type")
    if kind not in _VTK:
        raise ScanError(f"{path.name}: data type {kind} isn't supported")
    dtype = np.dtype(order + _VTK[kind])
    extent = [int(v) for v in (_attr(text, "WholeExtent") or "").split()]
    if len(extent) != 6:
        raise ScanError(f"{path.name} has no WholeExtent")
    x, y, z = (extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
    if point:
        x, y, z = x + 1, y + 1, z + 1
    count = x * y * z
    offset = start + int(_attr(tag, "offset") or 0)
    notes = []
    if not compressed:
        with open(path, "rb") as f:
            f.seek(offset)
            nbytes = int(np.frombuffer(f.read(header.itemsize), dtype=header)[0])
        if nbytes < count * dtype.itemsize:
            raise ScanError(f"{path.name}: the array holds {nbytes:,} bytes, not the {count * dtype.itemsize:,} its extent needs")
        array = np.memmap(path, dtype=dtype, mode="r", offset=offset + header.itemsize, shape=(z, y, x))
    else:
        with open(path, "rb") as f:
            f.seek(offset)
            blocks, _, _ = (int(v) for v in np.frombuffer(f.read(3 * header.itemsize), dtype=header))
            sizes = np.frombuffer(f.read(blocks * header.itemsize), dtype=header)
            data = b"".join(zlib.decompress(f.read(int(n))) for n in sizes)
        array = np.frombuffer(data, dtype=dtype, count=count).reshape(z, y, x)
        notes.append("The data are compressed, so they are read whole into memory")
    voxel, note = None, ""
    spacing = [float(v) for v in (_attr(text, "Spacing") or "").split()]
    if spacing and 0 < spacing[0] < 1e-3:
        voxel, note = spacing[0] * 1e6, f"Spacing {spacing[0]:g} in the file, read as meters (as Tangle and PuMA write it)"
    elif spacing and spacing[0] != 1.0:
        notes.append(f"The file gives the spacing as {spacing[0]:g} without a unit: pass --voxel-size")
    return Scan(array, path, f"VTK image data, array '{_attr(tag, 'Name')}'", voxel, note, notes)


def _open_hdf5(path: Path, dataset: str | None) -> Scan:
    h5py = need("h5py", "Reading HDF5 files")
    f = h5py.File(path, "r")
    found = []
    f.visititems(lambda name, obj: found.append((name, obj.shape)) if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 else None)
    if dataset:
        if dataset not in f:
            listed = ", ".join(f"/{n} {tuple(s)}" for n, s in found) or "none"
            raise ScanError(f"{path.name} has no dataset '{dataset}'; its 3D datasets: {listed}")
        name = dataset.strip("/")
    else:
        if not found:
            raise ScanError(f"{path.name} holds no 3D dataset")
        name = max(found, key=lambda item: int(np.prod(item[1])))[0]
    data = f[name]
    notes = []
    if not dataset and len(found) > 1:
        others = ", ".join(f"/{n}" for n, _ in found if n != name)
        notes.append(f"Using the biggest dataset, /{name}; the file also has {others} (pick one with --dataset)")
    voxel, note = None, ""
    for holder in (data, f):
        for key in ("element_size_um", "voxel_size_um", "pixel_size_um"):
            if key in holder.attrs:
                value = np.atleast_1d(np.asarray(holder.attrs[key], dtype=float))
                if value.size and value[-1] > 0:
                    voxel, note = float(value[-1]), f"the '{key}' attribute"
                    break
        if voxel:
            break
    return Scan(_volume(data, None, f"{path.name}:/{name}"), path, f"HDF5 dataset /{name}", voxel, note, notes)


def _open_zarr(path: Path, dataset: str | None) -> Scan:
    zarr = need("zarr", "Reading Zarr folders")
    node = zarr.open(str(path), mode="r")
    group = None
    if not hasattr(node, "shape"):
        group = node
        if dataset:
            node = group[dataset]
        elif "0" in group:
            node = group["0"]  # OME-Zarr: level 0 is the full resolution
        else:
            raise ScanError(f"{path.name} is a group of arrays ({', '.join(group.keys())}): say which with --dataset")
    voxel, note = None, ""
    try:
        attrs = dict(group.attrs) if group is not None else {}
        multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
        if multiscales:
            axes = multiscales[0].get("axes", [])
            scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
            unit = axes[-1].get("unit") if axes and isinstance(axes[-1], dict) else None
            factor = unit_um(unit)
            if factor and scale[-1] > 0:
                voxel, note = float(scale[-1]) * factor, "OME-Zarr metadata"
    except Exception:
        voxel = None
    return Scan(_volume(node, None, path.name), path, "Zarr array", voxel, note)


def _open_nifti(path: Path) -> Scan:
    nib = need("nibabel", "Reading NIfTI files")
    image = nib.load(str(path))
    shape = tuple(int(n) for n in image.shape)
    if len(shape) < 3 or any(n != 1 for n in shape[3:]):
        raise ScanError(f"{path.name} is not one 3D volume (shape {shape})")
    proxy = image.dataobj
    dtype = np.asarray(proxy[(slice(0, 1),) * 3 + (0,) * (len(shape) - 3)]).dtype
    voxel, note = None, ""
    zooms = image.header.get_zooms()[:3]
    scale = {"mm": 1e3, "micron": 1.0, "meter": 1e6}.get(image.header.get_xyzt_units()[0])
    if scale and zooms[0] > 0:
        voxel, note = float(zooms[0]) * scale, "the NIfTI header"
    return Scan(_XYZ(proxy, shape[:3], dtype, len(shape) - 3), path, "NIfTI", voxel, note)


# -- command-line options shared by the scripts ------------------------------------------------------------------


def add_input_arguments(parser) -> None:
    parser.add_argument("scan", help="the scan: a TIFF stack, a folder of slices, or another format (see --help)")
    parser.add_argument("--voxel-size", metavar="SIZE",
                        help="voxel size, e.g. 1.3um or 650nm (read from the file when it says; this wins)")
    group = parser.add_argument_group("other file formats")
    group.add_argument("--dataset", metavar="NAME", help="which array to read from an HDF5, Zarr, .npz or .vti file")
    group.add_argument("--raw-shape", metavar="XxYxZ", help="raw binary data: width x height x slices, e.g. 1024x1024x800")
    group.add_argument("--raw-dtype", metavar="TYPE", help="raw binary data: uint8, uint16, int16, float32, ...")
    group.add_argument("--raw-endian", default="little", choices=("little", "big"), help="raw binary data: byte order")
    group.add_argument("--raw-header-bytes", type=int, default=0, metavar="N", help="raw binary data: bytes to skip at the start")


def scan_from_args(args) -> Scan:
    """Open the scan the options describe; ``--voxel-size`` overrides what the file says."""
    raw_shape = parse_xyz(args.raw_shape) if args.raw_shape else None
    scan = open_scan(args.scan, dataset=args.dataset, raw_shape=raw_shape, raw_dtype=args.raw_dtype,
                     raw_endian=args.raw_endian, raw_header=args.raw_header_bytes)
    if args.voxel_size:
        given = parse_length(args.voxel_size, what="voxel size")
        if given <= 0:
            raise ScanError("--voxel-size must be above zero")
        if scan.voxel_um and abs(given - scan.voxel_um) > 0.01 * scan.voxel_um:
            scan.notes.append(f"Using --voxel-size {given:.4g} um; the file said {scan.voxel_um:.4g} um ({scan.voxel_from})")
        scan.voxel_um, scan.voxel_from = given, "from --voxel-size"
    return scan


def add_region_arguments(parser) -> None:
    group = parser.add_argument_group("which part of the scan, and how to read its grey levels")
    group.add_argument("--crop", metavar="RANGES", help="voxel ranges, e.g. x=100:612,y=0:512,z=200:456 "
                       "(an axis left out is kept whole; z counts slices from 0)")
    group.add_argument("--center-crop", type=int, metavar="N", help="an N-voxel cube from the middle of the scan "
                       "(quick tries)")
    group.add_argument("--bin", type=int, default=1, metavar="K", help="average K x K x K voxels into one (for fibers "
                       "over ~25 voxels across)")
    group.add_argument("--invert", action="store_true", help="the fibers are darker than the space around them")
    group.add_argument("--levels", metavar="VOID,BRIGHT", help="the grey value of empty space and of the brightest "
                       "fiber (default: the median and the 99.5th percentile)")


@dataclass
class Box:
    """Voxel ranges of the original scan, ends excluded."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def describe(self, shape) -> str:
        if self.shape == tuple(shape):
            return "the whole scan"
        return f"x {self.x0}-{self.x1}, y {self.y0}-{self.y1}, z {self.z0}-{self.z1}"


def parse_crop(text: str, shape: tuple[int, int, int]) -> Box:
    """``x=100:400,y=0:300,z=50:250`` (any axes, any order) or ``100:400,0:300,50:250`` (x, y, z)."""
    nz, ny, nx = shape
    ranges = {"x": (0, nx), "y": (0, ny), "z": (0, nz)}
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    named = all("=" in p for p in parts)
    if not parts or (not named and len(parts) != 3):
        raise ScanError(f"--crop '{text}': write it like x=100:612,y=0:512,z=200:456")
    for k, part in enumerate(parts):
        axis, _, span = part.partition("=") if named else ("xyz"[k], "", part)
        axis = axis.strip().lower()
        if axis not in ranges or ":" not in span:
            raise ScanError(f"--crop '{text}': write it like x=100:612,y=0:512,z=200:456")
        size = {"x": nx, "y": ny, "z": nz}[axis]
        low, _, high = span.partition(":")
        try:
            a = int(low) if low.strip() else 0
            b = int(high) if high.strip() else size
        except ValueError:
            raise ScanError(f"--crop '{text}': the ranges need whole voxel numbers") from None
        a, b = max(a, 0), min(b, size)
        if b - a < 8:
            raise ScanError(f"--crop {axis}={low}:{high} leaves {max(b - a, 0)} voxels of the {size} along {axis}")
        ranges[axis] = (a, b)
    return Box(*ranges["z"], *ranges["y"], *ranges["x"])


def center_box(size: int, shape: tuple[int, int, int]) -> Box:
    if size < 8:
        raise ScanError("--center-crop needs at least 8 voxels")
    spans = []
    for n in shape:
        m = min(size, n)
        spans += [(n - m) // 2, (n - m) // 2 + m]
    return Box(*spans)


def region_box(args, shape) -> Box:
    if args.crop and args.center_crop:
        raise ScanError("Give --crop or --center-crop, not both")
    if args.crop:
        box = parse_crop(args.crop, shape)
    elif args.center_crop:
        box = center_box(args.center_crop, shape)
    else:
        box = Box(0, shape[0], 0, shape[1], 0, shape[2])
    if args.bin < 1:
        raise ScanError("--bin is a whole number, 1 or more")
    if args.bin > 1:  # keep whole bins
        k = args.bin
        box = Box(box.z0, box.z1 - (box.z1 - box.z0) % k, box.y0, box.y1 - (box.y1 - box.y0) % k,
                  box.x0, box.x1 - (box.x1 - box.x0) % k)
        if min(box.shape) < 8 * k:
            raise ScanError(f"--bin {k} leaves too few voxels ({' x '.join(str(n // k) for n in box.shape[::-1])})")
    return box


def prepare_region(scan: Scan, box: Box, bin: int = 1, invert: bool = False, work: Path | None = None,
                   say=print) -> tuple[Any, bool]:
    """The region to analyse as a (z, y, x) array, and whether it was copied.

    A NumPy-readable scan (memory-mapped .npy, uncompressed TIFF, raw) with nothing to change is used in place
    when it is stored in this computer's byte order. Anything else (binning, inverting, slices, compressed or
    chunked files) is copied once to
    ``work/volume.npy`` slab by slab, so the network's tiles read it fast and a big scan never has to fit in
    memory. Integer scans stay in their type (an inverted one bit-flipped, so 65535 - v for uint16).
    """
    nz, ny, nx = (n // bin for n in box.shape)
    array = scan.array
    if (bin == 1 and not invert and isinstance(array, np.ndarray) and array.dtype.isnative and array.flags.aligned
            and (array.dtype.kind in "ui" or array.dtype in (np.float32, np.float64))):
        return array[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1], False
    if work is None:
        raise ScanError("this scan needs a work folder to copy the region into")
    dtype = np.dtype(scan.dtype.name)  # native byte order
    if dtype.kind not in "ui" or dtype.itemsize > 2:
        dtype = np.dtype(np.float32)
    work.mkdir(parents=True, exist_ok=True)
    target = work / "volume.npy"
    partial = work / "volume.partial.npy"
    check_space(work, nz * ny * nx * dtype.itemsize, "the copy of the region it analyses")
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=(nz, ny, nx))
    depth = max(bin, int(64e6 // max(box.shape[1] * box.shape[2], 1)) // bin * bin)
    last, started = 0.0, time.time()
    for z in range(box.z0, box.z1, depth):
        end = min(z + depth, box.z1)
        block = np.asarray(scan.array[z:end, box.y0:box.y1, box.x0:box.x1])
        if bin > 1:
            b = block.reshape(block.shape[0] // bin, bin, block.shape[1] // bin, bin, block.shape[2] // bin, bin)
            block = b.mean(axis=(1, 3, 5), dtype=np.float32)
        if dtype.kind in "ui":
            block = np.clip(np.rint(block), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype) \
                if block.dtype.kind == "f" else block.astype(dtype)
            if invert:
                block = np.invert(block)
        else:
            block = block.astype(np.float32)
            if invert:
                block = -block
        out[(z - box.z0) // bin:(end - box.z0) // bin] = block
        if time.time() - last > 10:
            last = time.time()
            say(f"  copying the region: {100 * (end - box.z0) / box.shape[0]:.0f}%")
    out.flush()
    del out
    os.replace(partial, target)
    say(f"  copied the region to {target} ({human_bytes(nz * ny * nx * dtype.itemsize)}, "
        f"{time.time() - started:.0f} s)")
    return np.load(target, mmap_mode="r"), True


def flip(dtype):
    """The inversion ``prepare_region`` applies to grey values of ``dtype`` (bit-flipped integers, so 65535 - v
    for uint16; negated floats), as a function of a value. Applied twice it gives the value back."""
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return lambda v: float(info.max) + float(info.min) - v
    return lambda v: -v


def check_space(folder: Path, needed: float, what: str,
                fix: str = "write the results elsewhere (--out), or analyse a smaller region (--crop, --bin 2)") -> None:
    """Stop with a clear message when ``folder``'s disk can't hold ``needed`` more bytes (a full disk under a
    memory-mapped file crashes Python instead of raising an error)."""
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return
    if needed > free - 200e6:
        raise ScanError(f"Not enough disk space in {folder} for {what}: it needs {human_bytes(needed)} and "
                        f"{human_bytes(free)} is free. Free some space, {fix}")


# -- grey levels -------------------------------------------------------------------------------------------------


def grey_sample(array, max_voxels: int = 8_000_000) -> np.ndarray:
    """An evenly spread sample of the volume's grey values (float32), read slice by slice."""
    nz, ny, nx = (int(n) for n in array.shape)
    step = max(1, int(np.ceil((nz * ny * nx / max_voxels) ** (1.0 / 3.0))))
    parts = [np.asarray(array[z, ::step, ::step], dtype=np.float32).ravel() for z in range(step // 2, nz, step)]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def otsu(values: np.ndarray, bins: int = 256) -> float:
    """The grey level that best splits the values into two classes (Otsu's method)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    low, high = np.percentile(values, [0.1, 99.9])
    if high <= low:
        return float(low)
    histogram, edges = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(histogram).astype(np.float64)
    total = weight[-1]
    mean = np.cumsum(histogram * centers)
    between = (mean[-1] * weight / total - mean) ** 2 / np.maximum(weight * (total - weight), 1e-12)
    return float(centers[np.argmax(between[:-1])])


@dataclass
class GreyCheck:
    """What the grey values say about the scan (from a sample)."""

    void: float  # the median: the network's "empty space" level
    bright: float  # the 99.5th percentile: its "bright fiber" level
    low: float  # the 0.5th percentile
    threshold: float  # Otsu's split between empty space and solid
    solid: float  # share of the sample above the threshold
    dark_void: float  # median of the values below the threshold
    fill: float | None  # one exact value filling part of the scan (outside the reconstructed field of view)
    fill_share: float
    warnings: list[str]


def check_grey(sample: np.ndarray, inverted: bool = False, original=None) -> GreyCheck:
    """Grey levels, and warnings when the scan breaks the network's assumptions (bright fibers, mostly void).

    A value that fills more than 2% of the scan exactly at its dark end (the outside of a reconstructed
    cylinder, often 0; the bright end once inverted) is left out of the levels, as are NaN voxels.
    ``original`` turns a grey value of ``sample`` back into the scan's own (for an inverted sample), so the
    warnings quote values the user can give back as options."""
    original = original or (lambda v: v)
    sample = np.asarray(sample, dtype=np.float32).ravel()
    fill, share, warnings = None, 0.0, []
    finite = np.isfinite(sample)
    if not finite.all():
        lost = 1.0 - float(finite.mean())
        sample = sample[finite]
        warnings.append(f"{100 * lost:.2g}% of the voxels are not numbers (NaN), probably outside the reconstructed "
                        "field of view; the network sees them as empty space. Crop to the inside of the sample "
                        "(--crop) so it doesn't trace that edge")
    if sample.size:
        values, counts = np.unique(sample, return_counts=True)
        top = int(np.argmax(counts))
        if counts[top] > 0.02 * sample.size and top == (len(values) - 1 if inverted else 0) and len(values) > 2:
            fill, share = float(values[top]), float(counts[top] / sample.size)
            sample = sample[sample != values[top]]
            warnings.append(f"{100 * share:.0f}% of the voxels are exactly {original(fill) + 0.0:g}, probably "
                            "outside the reconstructed field of view; they are left out of the grey levels. Crop to "
                            "the inside of the sample (--crop) so the network doesn't trace that edge")
    if sample.size < 100:
        raise ScanError("The scan holds (almost) a single grey value: nothing to find")
    low, void, bright = (float(v) for v in np.percentile(sample, [0.5, 50.0, 99.5]))
    threshold = otsu(sample)
    solid = float(np.mean(sample > threshold))
    below = sample[sample <= threshold]
    dark_void = float(np.median(below)) if below.size else void
    if bright - void < 0.6 * (void - low):
        warnings.append("The grey values reach further below the median than above it, as when the fibers are "
                        "darker than the space around them. "
                        + ("--invert was given: check that the fibers really are dark in the scan" if inverted
                           else "If they are, add --invert"))
    if solid > 0.45:
        warnings.append(f"About {100 * solid:.0f}% of the scan is {'darker' if inverted else 'brighter'} than the "
                        "solid/void split, so the median may be a fiber grey rather than empty space. If so, set the "
                        f"levels by hand: --levels={original(dark_void):.6g},{original(bright):.6g}")
    if bright <= void:
        warnings.append("The scan has almost no contrast (the 99.5th percentile equals the median)")
    return GreyCheck(void, bright, low, threshold, solid, dark_void, fill, share, warnings)
