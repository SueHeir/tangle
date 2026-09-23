"""Synthetic CT with known ground truth, rendered from any Tangle assembly."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_vti(path: str | Path) -> np.ndarray:
    """Read the first cell array of a Tangle-written ``.vti`` as ``(z, y, x)``."""
    raw = Path(path).read_bytes()
    head, _, data = raw.partition(b'<AppendedData encoding="raw">\n_')
    text = head.decode()
    extent = [int(v) for v in text.split('WholeExtent="')[1].split('"')[0].split()]
    nx, ny, nz = extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]
    kind = text.split('<DataArray type="')[1].split('"')[0]
    dtype = {"UInt8": "<u1", "UInt16": "<u2", "UInt32": "<u4", "UInt64": "<u8", "Float32": "<f4"}[kind]
    nbytes = int(np.frombuffer(data[:8], dtype="<u8")[0])
    array = np.frombuffer(data[8 : 8 + nbytes], dtype=dtype)
    return array[: nx * ny * nz].reshape(nz, ny, nx).copy()


@dataclass
class SyntheticScan:
    """A rendered scan and its ground truth (voxel units, ``(x, y, z)``)."""

    volume: np.ndarray
    labels: np.ndarray
    centerlines: list[np.ndarray]
    radii: np.ndarray
    voxel_size: float


def synthetic_ct(
    source,
    voxel_size: float,
    *,
    psf_sigma_voxels: float = 0.9,
    noise: float = 0.12,
    void_level: float = 0.05,
    drift: float = 0.05,
    seed: int = 0,
) -> SyntheticScan:
    """Render ``source`` (an ``Assembly`` or ``RunResult``) as a CT-like volume.

    Forward model: Tangle's smooth capsule-interface image gives partial-volume
    occupancy; a Gaussian point-spread blur, void/fiber attenuation contrast,
    Gaussian noise (``noise`` relative to the contrast) and a weak
    low-frequency drift are applied; the result is scaled to ``uint16``.
    Ground-truth labels come from the exporter's per-voxel fiber ids.
    """
    from scipy.ndimage import gaussian_filter

    with tempfile.TemporaryDirectory() as tmp:
        report = source.export_puma(Path(tmp) / "truth.puma", voxel_size, include_fiber_ids=True, include_interface=True)
        labels = read_vti(report.fiber_ids_path).astype(np.int32)
        interface = read_vti(report.interface_path)
    occupancy = np.clip(interface.astype(np.float32) / 255.0, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    attenuation = void_level + (1.0 - void_level) * gaussian_filter(occupancy, psf_sigma_voxels)
    attenuation += rng.normal(0.0, noise, size=attenuation.shape).astype(np.float32)
    if drift:
        z, y, x = np.indices(attenuation.shape, dtype=np.float32)
        ny, nx = attenuation.shape[1:]
        attenuation += drift * np.cos(np.pi * (x - nx / 2) / nx) * np.cos(np.pi * (y - ny / 2) / ny)
    low, high = np.percentile(attenuation, [0.5, 99.5])
    volume = np.clip((attenuation - low) / (high - low) * 65535, 0, 65535).astype(np.uint16)

    centerlines = [np.asarray(line) / voxel_size for line in source.centerlines()]
    # Export ids are one-based in source order for these single-material scans.
    radii = _radii(source, labels, centerlines, voxel_size)
    return SyntheticScan(volume, labels, centerlines, radii, voxel_size)


def _radii(source, labels, centerlines, voxel_size) -> np.ndarray:
    try:
        table = source.characterize().to_dict()
        values = [f.get("equivalent_diameter") for f in table.get("fiber_metrics", [])]
        if len(values) == len(centerlines) and all(v for v in values):
            return 0.5 * np.asarray(values, dtype=np.float64) / voxel_size
    except Exception:  # characterization is optional metadata here
        pass
    from ._geometry import polyline_length

    counts = np.bincount(labels.ravel(), minlength=len(centerlines) + 1)[1:]
    lengths = np.array([polyline_length(c) for c in centerlines])
    return np.sqrt(counts[: len(centerlines)] / (np.pi * np.maximum(lengths, 1e-9)))
