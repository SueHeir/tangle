"""A simulated CT acquisition: projections, propagation, photon noise, FBP.

``synthetic_ct(..., scanner=Scanner())`` renders a scan the way a
parallel-beam micro-CT makes one, instead of blurring the fiber occupancy
and adding noise to it:

1. The sample's attenuation (fiber occupancy times ``fiber_attenuation``,
   on ``void_attenuation``) is projected along x-ray paths at ``angles``
   angles over 180°, rotating about the z axis.
2. Each projection is the exit wave of a weak phase-and-absorption object:
   its phase is ``delta_beta`` times half its attenuation (the ratio of the
   refractive index decrement to the absorption index, for one material).
   The wave propagates ``propagation`` (the Fresnel distance, wavelength
   times sample-detector distance over the detector pixel squared) in free
   space before the detector records its intensity. That is what makes
   the phase-contrast fringe at every surface: bright just inside, dark
   just outside, strongest where two surfaces face each other. At 0 the
   projection is plain absorption.
3. The detector blurs by ``detector_blur`` pixels (source size and
   scintillator), counts Poisson photons out of ``photons`` per pixel
   unattenuated, and has a per-column gain error of ``ring_strength``
   (which reconstructs as rings; 0 by default).
4. The log of flat-field-corrected intensity is reconstructed slice by
   slice by filtered back-projection with a Shepp-Logan filter.

The noise texture, the blur and the edge fringes then come from the same
physics as a real scan's. Every setting is a physical one; none is fitted.
Voxels are the detector pixels (unit magnification).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scanner:
    """A parallel-beam CT scanner; see the module docstring."""

    photons: float = 5000.0
    angles: int | None = None  # default: the padded slice width
    fiber_attenuation: float = 0.02  # per voxel, for occupancy 1
    void_attenuation: float = 0.0005  # per voxel
    delta_beta: float = 0.0
    propagation: float = 0.0  # wavelength x distance / pixel², in pixels²
    detector_blur: float = 0.6  # pixels
    ring_strength: float = 0.0


def acquire(attenuation: np.ndarray, scanner: Scanner, rng: np.random.Generator) -> np.ndarray:
    """The reconstructed attenuation per voxel of ``attenuation`` (``(z, y, x)``, per voxel) scanned by ``scanner``."""
    from scipy.ndimage import gaussian_filter, rotate

    nz, ny, nx = attenuation.shape
    # Pad the slice to its diagonal so nothing leaves the field as it turns.
    width = int(np.ceil(np.hypot(ny, nx))) + 4
    pad_y, pad_x = (width - ny) // 2, (width - nx) // 2
    sample = np.zeros((nz, width, width), dtype=np.float32)
    sample[:, pad_y : pad_y + ny, pad_x : pad_x + nx] = attenuation
    count = scanner.angles or width
    angles = np.arange(count) * (180.0 / count)

    # Angles are spread over threads (the rotations release the GIL).
    workers = max(1, min(8, os.cpu_count() or 1, count))
    chunks = [range(w, count, workers) for w in range(workers)]

    # Projections (angle, z, u): line integrals along y of the turned sample.
    line = np.empty((count, nz, width), dtype=np.float32)

    def project(chunk):
        for k in chunk:
            turned = rotate(sample, angles[k], axes=(1, 2), reshape=False, order=1, mode="constant")
            line[k] = turned.sum(axis=1)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(project, chunks))

    intensity = np.exp(-line)
    if scanner.propagation > 0 and scanner.delta_beta > 0:
        intensity = _propagate(line, scanner.delta_beta, scanner.propagation)
    if scanner.detector_blur > 0:
        intensity = gaussian_filter(intensity, (0, scanner.detector_blur, scanner.detector_blur))
    gain = 1.0 + scanner.ring_strength * rng.standard_normal((1, nz, width)).astype(np.float32)
    counts = rng.poisson(np.clip(intensity * gain, 0.0, None) * scanner.photons).astype(np.float32)
    # Flat-field correction with an ideal flat; the gain error stays as rings.
    measured = -np.log(np.maximum(counts, 0.5) / scanner.photons)

    # Filtered back-projection, the inverse of the projector above.
    filtered = _filter(measured)

    def back_project(chunk):
        total = np.zeros_like(sample)
        for k in chunk:
            smear = np.broadcast_to(filtered[k][:, None, :], sample.shape)
            total += rotate(smear, -angles[k], axes=(1, 2), reshape=False, order=1, mode="constant")
        return total

    with ThreadPoolExecutor(workers) as pool:
        volume = sum(pool.map(back_project, chunks))
    volume *= np.pi / count
    return volume[:, pad_y : pad_y + ny, pad_x : pad_x + nx]


def _propagate(line: np.ndarray, delta_beta: float, distance: float) -> np.ndarray:
    """Intensity after the exit wave of each projection travels ``distance`` (pixels²)."""
    count, nz, width = line.shape
    # Pad against wrap-around, with the edge values (the field keeps going).
    pz, pu = nz // 2 + 8, width // 2 + 8
    padded = np.pad(line, ((0, 0), (pz, pz), (pu, pu)), mode="edge")
    fz = np.fft.fftfreq(padded.shape[1])[:, None]
    fu = np.fft.fftfreq(padded.shape[2])[None, :]
    kernel = np.exp(-1j * np.pi * distance * (fz**2 + fu**2))
    out = np.empty_like(line)
    for k in range(count):
        field = np.exp(-0.5 * padded[k] * (1.0 + 1j * delta_beta))
        wave = np.fft.ifft2(np.fft.fft2(field) * kernel)
        out[k] = (np.abs(wave) ** 2)[pz : pz + nz, pu : pu + width]
    return out


def _filter(sinogram: np.ndarray) -> np.ndarray:
    """Shepp-Logan ramp filter along the detector rows."""
    width = sinogram.shape[-1]
    size = int(2 ** np.ceil(np.log2(2 * width)))
    f = np.fft.rfftfreq(size)
    ramp = np.abs(f) * np.sinc(f)  # |f| (cycles/pixel) times the Shepp-Logan window
    spectrum = np.fft.rfft(sinogram, n=size, axis=-1) * ramp
    return np.fft.irfft(spectrum, n=size, axis=-1)[..., :width].astype(np.float32)
