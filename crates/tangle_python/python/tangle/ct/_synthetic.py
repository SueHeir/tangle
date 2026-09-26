"""Synthetic CT with known ground truth, rendered from any Tangle assembly."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
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
    # Cell period in voxels along each (x, y, z) axis; 0 where not periodic.
    period: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Fiber type of each true fiber (index into the ``profiles`` it was
    # rendered with), or None for a single-type scan.
    types: np.ndarray | None = None
    # Oval fibers: each true fiber's long and short semi-axes (voxels), and
    # the unit long-axis direction at each centerline node. None when every
    # fiber is round (``radii`` then says it all). ``radii`` is the
    # equivalent radius, sqrt(long * short), either way.
    semi_axes: np.ndarray | None = None
    long_axes: list[np.ndarray] | None = None

    def fiber_mask(self, level: float = 0.5, *, sigma: float = 0.7) -> np.ndarray:
        """A binary fiber mask, as a grey-level threshold of the scan gives.

        The scan is smoothed by ``sigma`` voxels and thresholded at ``level``
        of the way from the void to the fiber grey level (both read off the
        true labels). A ``level`` below 0.5 over-reaches: fibers come out
        thicker and more background is kept, as with a generous threshold.
        """
        from scipy.ndimage import gaussian_filter

        smooth = gaussian_filter(self.volume.astype(np.float32), sigma) if sigma > 0 else self.volume.astype(np.float32)
        void = float(np.median(smooth[self.labels == 0]))
        fiber = float(np.median(smooth[self.labels > 0]))
        return smooth >= void + level * (fiber - void)

    def crop(self, low: tuple[int, int, int], high: tuple[int, int, int]) -> "SyntheticScan":
        """The sub-volume ``[low, high)`` (voxel indices, ``(x, y, z)``).

        Cropping a scan of a larger cell gives fibers that run through the
        scan boundary, as in a real scan. Truth centerlines are clipped to the
        crop; a fiber that leaves and re-enters becomes several pieces, and
        labels are renumbered to match the remaining pieces. In a periodic
        cell, the periodic images of each centerline are clipped too.
        """
        import itertools
        from ._geometry import resample

        low_a, high_a = np.asarray(low, dtype=int), np.asarray(high, dtype=int)
        window = (slice(low_a[2], high_a[2]), slice(low_a[1], high_a[1]), slice(low_a[0], high_a[0]))
        volume = self.volume[window].copy()
        old_labels = self.labels[window]
        extent = (high_a - low_a).astype(np.float64)
        lines, radii, origin, types, axes = [], [], [], [], []
        shifts = [
            np.array(combo) * self.period
            # a fiber longer than the cell can reach past the next cell wall
            for combo in itertools.product(*[(-2, -1, 0, 1, 2) if p > 0 else (0,) for p in self.period])
        ]
        copies = [(index, shift) for index in range(len(self.centerlines)) for shift in shifts]
        for index, shift in copies:
            dense = resample(np.asarray(self.centerlines[index], dtype=np.float64), 0.5) + shift - low_a
            inside = np.all((dense >= 0) & (dense < extent), axis=1)
            start = None
            for k, flag in enumerate(list(inside) + [False]):
                if flag and start is None:
                    start = k
                elif not flag and start is not None:
                    if k - start >= 2:
                        lines.append(dense[start:k])
                        radii.append(self.radii[index])
                        origin.append(index + 1)
                        if self.types is not None:
                            types.append(self.types[index])
                        if self.long_axes is not None:
                            axes.append(_nearest_node_values(self.centerlines[index] + shift - low_a,
                                                             self.long_axes[index], dense[start:k]))
                    start = None
        # Voxels keep the id of the nearest surviving piece of their fiber.
        labels = np.zeros_like(old_labels)
        pieces_of: dict[int, list[int]] = {}
        for piece, fiber in enumerate(origin):
            pieces_of.setdefault(fiber, []).append(piece)
        for fiber, pieces in pieces_of.items():
            mask = old_labels == fiber
            if len(pieces) == 1:
                labels[mask] = pieces[0] + 1
                continue
            z, y, x = np.nonzero(mask)
            points = np.stack([x, y, z], axis=1) + 0.5
            from scipy.spatial import cKDTree

            nodes = np.concatenate([lines[p] for p in pieces])
            owner = np.concatenate([np.full(len(lines[p]), p) for p in pieces])
            _, nearest = cKDTree(nodes).query(points)
            labels[z, y, x] = owner[nearest] + 1
        return SyntheticScan(
            volume, labels, lines, np.asarray(radii, dtype=np.float64), self.voxel_size, np.zeros(3),
            np.asarray(types, dtype=int) if self.types is not None else None,
            self.semi_axes[np.asarray(origin, dtype=int) - 1].reshape(-1, 2) if self.semi_axes is not None else None,
            axes if self.long_axes is not None else None,
        )


def synthetic_ct(
    source,
    voxel_size: float,
    *,
    psf_sigma_voxels: float = 0.9,
    noise: float = 0.12,
    void_level: float = 0.05,
    drift: float = 0.05,
    seed: int = 0,
    profiles=None,
    noise_correlation: float = 0.0,
    phase_contrast: float = 0.0,
    phase_sigma_voxels: float | None = None,
    scanner=None,
    brightness_spread: float = 0.0,
) -> SyntheticScan:
    """Render ``source`` (an ``Assembly`` or ``RunResult``) as a CT-like volume.

    Forward model: Tangle's smooth capsule-interface image gives partial-volume
    occupancy; a Gaussian point-spread blur, void/fiber attenuation contrast,
    Gaussian noise (``noise`` relative to the contrast) and a weak
    low-frequency drift are applied; the result is scaled to ``uint16``.
    ``noise_correlation`` (voxels) blurs the noise field by that Gaussian
    sigma, keeping its standard deviation (CT reconstructions often have
    noise correlated over a voxel or two, so a light denoise removes
    little of it; 0.9 gives neighbouring voxels a correlation of about 0.75).
    Ground-truth labels come from the exporter's per-voxel fiber ids.

    ``phase_contrast`` adds the edge fringe of propagation-based phase
    contrast: the blurred image minus ``phase_contrast`` times its
    Laplacian of the occupancy at ``phase_sigma_voxels`` (default the blur),
    scaled by that variance so the setting does not depend on the width.
    Every surface gets a dark band just outside and a bright one just
    inside, deepest about ``phase_sigma_voxels`` from the surface and gone
    by about 2.5 times that; at 1, a lone straight edge's dark band dips
    about a quarter of the fiber contrast below the void. Where two
    surfaces come close the fringes add, so the gap between touching fibers
    reads darkest; in noise the shallow band breaks up into dark spots
    along the edges. 0 (the default) renders plain attenuation.

    ``scanner`` (a :class:`Scanner`) simulates the acquisition instead:
    projections, free-space propagation (phase contrast), detector blur,
    photon noise and filtered back-projection (see ``_scanner``). The
    blur, noise and fringes then come from the scanner, so
    ``psf_sigma_voxels``, ``noise``, ``noise_correlation`` and
    ``phase_contrast`` are ignored; ``drift`` still applies.

    ``profiles`` renders several fiber types with their own brightness:
    a sequence of ``(diameter, CrossSection)`` pairs. Each fiber gets the
    profile whose diameter is nearest its own, and every voxel's occupancy is
    scaled by that profile's brightness at the voxel's distance from the
    axis (a solid fiber of brightness 1 is the default). The result's
    ``types`` holds each fiber's profile index.

    ``brightness_spread`` scales each fiber's brightness by its own factor,
    log-normal with that sigma (median 1), drawn from ``seed``: real fibers
    of one type can differ a lot in grey, which a grey threshold then
    cannot follow. It needs ``profiles``; with ``scanner`` the phase follows
    the same factor.

    Oval fibers (a ``Material`` with a ``thickness``) render as ovals: the
    occupancy comes from the export, which draws each section, and a
    fiber's profile is picked by its long width (a Material's
    ``diameter``). A hollow profile's core still follows the equivalent
    radius. The result's ``semi_axes`` and ``long_axes`` hold the truth.
    """
    from scipy.ndimage import gaussian_filter

    with tempfile.TemporaryDirectory() as tmp:
        report = source.export_puma(Path(tmp) / "truth.puma", voxel_size, include_fiber_ids=True, include_interface=True)
        labels = read_vti(report.fiber_ids_path).astype(np.int32)
        interface = read_vti(report.interface_path)
    occupancy = np.clip(interface.astype(np.float32) / 255.0, 0.0, 1.0)
    centerlines = [np.asarray(line) / voxel_size for line in source.centerlines()]
    # Export ids are one-based in source order for these single-material scans.
    radii = _radii(source, labels, centerlines, voxel_size)
    semi_axes, long_axes = _oval_sections(source, voxel_size, len(centerlines))
    if semi_axes is not None:
        radii = np.sqrt(semi_axes[:, 0] * semi_axes[:, 1])
    reach = semi_axes[:, 0] if semi_axes is not None else radii
    period = np.zeros(3)
    cell = getattr(source, "cell", None)
    if cell is not None:
        period = np.array([n / voxel_size if p else 0.0 for n, p in zip(cell.lengths, cell.periodic)])
    types = None
    base = occupancy
    factor = None
    if brightness_spread > 0:
        # Its own stream, so the noise below is the same with or without it.
        factor = np.exp(np.random.default_rng([seed, 1]).normal(0.0, brightness_spread, len(centerlines)))
    if factor is not None and not profiles:
        raise ValueError("brightness_spread needs profiles")
    if profiles:
        occupancy, types = _apply_profiles(
            occupancy, centerlines, radii, voxel_size, profiles, period, labels, reach, factor
        )
    rng = np.random.default_rng(seed)
    if scanner is not None:
        from ._scanner import acquire

        sample = scanner.void_attenuation + (scanner.fiber_attenuation - scanner.void_attenuation) * occupancy
        phase = None
        if np.ndim(scanner.delta_beta) > 0:
            # One ratio per fiber type: the fibers' attenuation, type by type,
            # times its ratio (the void shifts no phase).
            ratios = [float(r) for r in scanner.delta_beta]
            if not profiles or len(ratios) != len(profiles):
                raise ValueError("give one Scanner.delta_beta per profile")
            scaled = [(d, replace(p, brightness=p.brightness * r)) for (d, p), r in zip(profiles, ratios)]
            weighted, _ = _apply_profiles(base, centerlines, radii, voxel_size, scaled, period, labels, reach, factor)
            phase = (scanner.fiber_attenuation * weighted).astype(np.float32)
        warp = None
        if scanner.fiber_motion > 0 or scanner.drift > 0:
            from ._scanner import motion_warp

            warp = motion_warp(labels, scanner, voxel_size, rng)
        attenuation = acquire(sample.astype(np.float32), scanner, rng, voxel_size, phase, warp)
        attenuation /= scanner.fiber_attenuation  # fiber 1, void about 0, as below
        if drift:
            z, y, x = np.indices(attenuation.shape, dtype=np.float32)
            ny, nx = attenuation.shape[1:]
            attenuation += drift * np.cos(np.pi * (x - nx / 2) / nx) * np.cos(np.pi * (y - ny / 2) / ny)
        low, high = np.percentile(attenuation, [0.5, 99.5])
        volume = np.clip((attenuation - low) / (high - low) * 65535, 0, 65535).astype(np.uint16)
        return SyntheticScan(volume, labels, centerlines, radii, voxel_size, period, types, semi_axes, long_axes)
    attenuation = void_level + (1.0 - void_level) * gaussian_filter(occupancy, psf_sigma_voxels)
    if phase_contrast:
        from scipy.ndimage import gaussian_laplace

        sigma = max(phase_sigma_voxels if phase_sigma_voxels is not None else psf_sigma_voxels, 0.5)
        attenuation -= phase_contrast * (1.0 - void_level) * sigma**2 * gaussian_laplace(occupancy, sigma)
    if noise_correlation > 0:
        field = gaussian_filter(rng.normal(0.0, 1.0, size=attenuation.shape).astype(np.float32), noise_correlation)
        attenuation += noise / max(float(field.std()), 1e-12) * field
    else:
        attenuation += rng.normal(0.0, noise, size=attenuation.shape).astype(np.float32)
    if drift:
        z, y, x = np.indices(attenuation.shape, dtype=np.float32)
        ny, nx = attenuation.shape[1:]
        attenuation += drift * np.cos(np.pi * (x - nx / 2) / nx) * np.cos(np.pi * (y - ny / 2) / ny)
    low, high = np.percentile(attenuation, [0.5, 99.5])
    volume = np.clip((attenuation - low) / (high - low) * 65535, 0, 65535).astype(np.uint16)

    return SyntheticScan(volume, labels, centerlines, radii, voxel_size, period, types, semi_axes, long_axes)


def _apply_profiles(occupancy, centerlines, radii, voxel_size, profiles, period, labels=None, reach=None, factor=None):
    """Scale occupancy by each fiber type's brightness profile.

    In a periodic cell a fiber's centerline runs past the cell walls while its
    voxels wrap around, so every periodic image of each centerline that
    reaches the volume is rendered as that fiber. A voxel the export labels
    belongs to that fiber; the rest (partial-volume edges) go to the nearest
    fiber within ``reach`` (its long semi-axis, default ``radii``) plus 2
    voxels. A fiber's profile is the one whose diameter is nearest twice
    its ``reach``. ``factor`` (one per fiber) scales each fiber's brightness.
    """
    import itertools

    from ._geometry import rasterize

    diameters = np.array([d for d, _ in profiles]) / voxel_size
    shapes = [profile for _, profile in profiles]
    reach = np.asarray(radii if reach is None else reach, dtype=np.float64)
    types = np.array([int(np.argmin(np.abs(diameters - 2.0 * r))) for r in reach], dtype=int)
    upper = np.array(occupancy.shape[::-1], dtype=np.float64)
    shifts = [
        np.array(combo) * period
        for combo in itertools.product(*[(-2, -1, 0, 1, 2) if p > 0 else (0,) for p in period])
    ]
    copies, copy_radii, copy_reach, source = [], [], [], []
    for index, line in enumerate(centerlines):
        line = np.asarray(line, dtype=np.float64)
        far = reach[index] + 2.0
        for shift in shifts:
            moved = line + shift
            if np.all(moved.min(axis=0) - far < upper) and np.all(moved.max(axis=0) + far > 0):
                copies.append(moved)
                copy_radii.append(radii[index])
                copy_reach.append(far)
                source.append(index)
    copy_radii = np.asarray(copy_radii, dtype=np.float64)
    owner, best, _ = rasterize(occupancy.shape, copies, copy_radii, reach=np.asarray(copy_reach), signed=True)
    level = np.ones(occupancy.shape, dtype=np.float32)
    owned = owner > 0
    fiber = np.asarray(source, dtype=int)[owner[owned] - 1]
    rho = best[owned] + radii[fiber]
    if labels is not None:
        # The export's own fiber ids (one-based, source order) where it has them.
        labelled = np.asarray(labels)[owned]
        fiber = np.where(labelled > 0, labelled - 1, fiber)
    brightness = np.array([p.brightness for p in shapes])[types[fiber]]
    if factor is not None:
        brightness = brightness * np.asarray(factor)[fiber]
    core = np.array([1.0 if p.solid else p.core for p in shapes])[types[fiber]]
    rim = np.array([np.inf if p.solid else p.rim / voxel_size for p in shapes])[types[fiber]]
    inner = np.maximum(radii[fiber] - rim, 0.0)
    values = (brightness * np.where(rho < inner, core, 1.0)).astype(np.float32)
    level[owned] = values
    return occupancy * level, types


def _oval_sections(source, voxel_size, count):
    """Semi-axes (voxels, long first) and per-node long axes when any fiber is oval, else ``(None, None)``."""
    assembly = getattr(source, "assembly", source)  # a RunResult's assembly is a property
    assembly = assembly() if callable(assembly) else assembly
    if not hasattr(assembly, "section_semi_axes"):
        return None, None
    semi_axes = np.asarray(assembly.section_semi_axes(), dtype=np.float64).reshape(-1, 2) / voxel_size
    if len(semi_axes) != count or np.allclose(semi_axes[:, 0], semi_axes[:, 1]):
        return None, None
    return semi_axes, [np.asarray(axes, dtype=np.float64) for axes in assembly.long_axes()]


def _nearest_node_values(nodes, values, points):
    """``values`` (one per node of ``nodes``) at the node nearest each of ``points``."""
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(np.asarray(nodes, dtype=np.float64)).query(np.asarray(points, dtype=np.float64))
    return np.asarray(values, dtype=np.float64)[nearest]


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
