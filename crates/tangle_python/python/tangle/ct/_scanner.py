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
   projection is plain absorption. Keep the projected phase to a few
   radians (``delta_beta`` times half a fiber's attenuation across it):
   far beyond that the fringes ring instead of edging each surface.
3. The detector blurs by ``detector_blur`` pixels (source size and
   scintillator), counts Poisson photons out of ``photons`` per pixel
   unattenuated, and has a per-column gain error of ``ring_strength``
   (which reconstructs as rings; 0 by default).
4. The log of flat-field-corrected intensity is reconstructed slice by
   slice by filtered back-projection with a Shepp-Logan filter.

The sample may move while it turns. ``fiber_motion`` (meters, root mean
square) moves each fiber on its own smooth random path, as loose fibers
settle or sway; ``drift`` (meters, root mean square) moves the whole
sample along one such path. The scan is split into ``motion_steps``
stretches of angles, each seeing the sample where it was then, so moving
fibers come out blurred and doubled while still ones stay sharp. The
truth is the fibers' rest position.

The noise texture, the blur and the edge fringes then come from the same
physics as a real scan's. Every setting is a physical one; none is fitted.
Voxels are the detector pixels (unit magnification).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scanner:
    """A parallel-beam CT scanner; see the module docstring."""

    photons: float = 5000.0
    angles: int | None = None  # default: the padded slice width
    fiber_attenuation: float = 0.02  # per voxel, for occupancy 1
    void_attenuation: float = 0.0005  # per voxel
    # One ratio for the whole sample, or one per fiber type (in the order of
    # ``synthetic_ct``'s ``profiles``): a denser, more absorbing material has
    # a lower ratio and shows less fringe.
    delta_beta: float | tuple[float, ...] = 0.0
    propagation: float = 0.0  # wavelength x distance / pixel², in pixels²
    detector_blur: float = 0.6  # pixels
    # The scanner's spatial resolution in meters: the full width at half
    # maximum of its blur (source, scintillator and detector together).
    # When set it replaces ``detector_blur``, so the blur stays the same
    # physical size whatever the voxel size.
    resolution: float | None = None
    ring_strength: float = 0.0
    fiber_motion: float = 0.0  # meters, RMS displacement of each fiber over the scan
    drift: float = 0.0  # meters, RMS displacement of the whole sample over the scan
    motion_steps: int = 8


def acquire(
    attenuation: np.ndarray,
    scanner: Scanner,
    rng: np.random.Generator,
    voxel_size: float | None = None,
    phase: np.ndarray | None = None,
    warp=None,
) -> np.ndarray:
    """The reconstructed attenuation per voxel of ``attenuation`` (``(z, y, x)``, per voxel) scanned by ``scanner``.

    ``voxel_size`` (meters) is needed only for ``scanner.resolution``.
    ``phase`` is the phase shift per voxel times two (``delta_beta`` times
    the attenuation, voxel by voxel, for a sample of several materials);
    by default ``scanner.delta_beta`` (a single ratio) times ``attenuation``.
    ``warp(step, volume)`` is the volume as the sample stood during
    stretch ``step`` of ``scanner.motion_steps`` (the motion); None for a
    still sample.
    """
    from scipy.ndimage import gaussian_filter

    from . import _native

    nz, ny, nx = attenuation.shape
    # The detector spans the slice's diagonal, so every ray through it is seen.
    width = int(np.ceil(np.hypot(ny, nx))) + 4
    count = scanner.angles or width
    angles = np.arange(count) * (np.pi / count)
    steps = max(int(scanner.motion_steps), 1) if warp is not None else 1
    blocks = np.array_split(np.arange(count), steps)
    line = np.empty((count, nz, width), dtype=np.float32)  # (angle, z, u)
    phase_line = np.empty_like(line) if phase is not None else None
    for step, block in enumerate(blocks):
        # Each stretch of angles sees the sample where it stood then.
        moved = attenuation if warp is None else warp(step, attenuation)
        line[block] = _native.project(moved, angles[block], width)
        if phase is not None:
            moved = phase if warp is None else warp(step, phase)
            phase_line[block] = _native.project(moved, angles[block], width)

    intensity = np.exp(-line)
    if phase_line is None and np.ndim(scanner.delta_beta) == 0 and scanner.delta_beta > 0:
        phase_line = float(scanner.delta_beta) * line
    if scanner.propagation > 0 and phase_line is not None:
        intensity = _propagate(line, phase_line, scanner.propagation)
    blur = scanner.detector_blur
    if scanner.resolution is not None:
        if voxel_size is None:
            raise ValueError("Scanner.resolution needs the voxel size")
        blur = scanner.resolution / (2.0 * np.sqrt(2.0 * np.log(2.0)) * voxel_size)
    if blur > 0:
        intensity = gaussian_filter(intensity, (0, blur, blur))
    gain = 1.0 + scanner.ring_strength * rng.standard_normal((1, nz, width)).astype(np.float32)
    counts = rng.poisson(np.clip(intensity * gain, 0.0, None) * scanner.photons).astype(np.float32)
    # Flat-field correction with an ideal flat; the gain error stays as rings.
    measured = -np.log(np.maximum(counts, 0.5) / scanner.photons)

    # Filtered back-projection, the inverse of the projector above.
    volume = _native.back_project(_filter(measured), angles, attenuation.shape)
    volume *= np.pi / count
    return volume


def _propagate(line: np.ndarray, phase_line: np.ndarray, distance: float) -> np.ndarray:
    """Intensity after the exit wave of each projection travels ``distance`` (pixels²).

    The wave's amplitude is ``exp(-line / 2)`` and its phase ``-phase_line / 2``.
    """
    count, nz, width = line.shape
    # Pad against wrap-around, with the edge values (the field keeps going).
    pz, pu = nz // 2 + 8, width // 2 + 8
    padded = np.pad(line, ((0, 0), (pz, pz), (pu, pu)), mode="edge")
    padded_phase = np.pad(phase_line, ((0, 0), (pz, pz), (pu, pu)), mode="edge")
    fz = np.fft.fftfreq(padded.shape[1])[:, None]
    fu = np.fft.fftfreq(padded.shape[2])[None, :]
    kernel = np.exp(-1j * np.pi * distance * (fz**2 + fu**2))
    out = np.empty_like(line)
    for k in range(count):
        field = np.exp(-0.5 * (padded[k] + 1j * padded_phase[k]))
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


def motion_warp(labels: np.ndarray, scanner: Scanner, voxel_size: float, rng: np.random.Generator):
    """``warp(step, volume)`` for ``acquire``: each fiber (``labels``, one-based) and the whole sample moving.

    Every voxel moves with the fiber it belongs to, or the nearest one; each
    fiber's path and the sample's drift are smooth random walks over the
    steps, centered on the rest position, with the requested RMS size.
    """
    from scipy.ndimage import distance_transform_edt, map_coordinates

    steps = max(int(scanner.motion_steps), 1)
    count = int(labels.max())

    def paths(n: int, rms: float) -> np.ndarray:
        """(steps, n, 3) smooth random displacements in voxels (z, y, x), RMS ``rms``."""
        walk = np.cumsum(rng.standard_normal((steps, n, 3)), axis=0)
        walk -= walk.mean(axis=0)
        size = np.sqrt((walk**2).sum(axis=2).mean())
        return walk * (rms / voxel_size / size) if size > 0 else walk

    fibers = paths(count + 1, scanner.fiber_motion) if scanner.fiber_motion > 0 and count else None
    drift = paths(1, scanner.drift)[:, 0] if scanner.drift > 0 else None
    owner = labels
    if fibers is not None:
        # The void moves with its nearest fiber, so partial-volume edges go along.
        _, index = distance_transform_edt(labels == 0, return_indices=True)
        owner = labels[tuple(index)]
    grid = np.indices(labels.shape, dtype=np.float32)

    def warp(step: int, volume: np.ndarray) -> np.ndarray:
        shift = np.zeros((3,) + labels.shape, dtype=np.float32)
        if fibers is not None:
            for axis in range(3):
                shift[axis] = fibers[step, :, axis].astype(np.float32)[owner]
        if drift is not None:
            shift += drift[step].astype(np.float32)[:, None, None, None]
        # A voxel shows what was a shift behind it.
        return map_coordinates(volume, grid - shift, order=1, mode="nearest").astype(np.float32)

    return warp

