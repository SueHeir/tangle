"""Fit Tangle fibers to a CT volume using what the fibers are known to look like."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import _moves, _refine
from ._ends import end_cost, end_statistics, evidence_scale
from ._geometry import polyline_length, rasterize, tangents
from ._image import HessianField, Levels, normalize
from ._profile import CrossSection
from ._trace import trace_fibers


@dataclass(frozen=True)
class FiberSpec:
    """What the fibers in the scan look like, in meters.

    ``diameter``
        Typical fiber diameter.
    ``diameter_tolerance``
        Fitted diameters stay within ``diameter * (1 ± tolerance)``.
    ``min_bend_radius``
        Tightest bend a fiber can take; limits how sharply a trace may turn
        and becomes the Tangle material's bend limit. Defaults to 5 diameters.
    ``min_length``
        Shorter fragments are discarded. Defaults to 3 diameters.
    ``max_length``
        Optional; longer fits are split where the image support is weakest.
    ``length``
        Optional typical (mean) fiber length. It turns on the fiber-length
        prior: fiber ends are assumed spread uniformly, about ``2 / length``
        per unit of fiber length, so each end inside the scan costs
        ``ln(length / diameter)`` nats. Splits, joins and the one-or-two-fiber
        decision then weigh the scan against that cost, and joins across
        longer gaps become possible. A rough value is enough.
    ``profile``
        Brightness across the fiber (:class:`CrossSection`): solid by default;
        set ``brightness`` for a type dimmer than the brightest one, and
        ``rim``/``core`` for fibers with a bright rim and a dimmer core.
    ``name``
        Material name used in the exported Tangle configuration; also names
        the type when several are fitted together.
    """

    diameter: float
    diameter_tolerance: float = 0.25
    min_bend_radius: float | None = None
    min_length: float | None = None
    max_length: float | None = None
    length: float | None = None
    profile: CrossSection = CrossSection()
    name: str = "ct fiber"

    def replace(self, **changes: Any) -> "FiberSpec":
        return replace(self, **changes)


@dataclass(frozen=True)
class FitSettings:
    """Numerical settings; the defaults are meant to work unchanged.

    Lengths here are in voxels or in fiber radii, as named.
    """

    denoise_sigma_voxels: float = 0.7
    rounds: int = 3
    iterations_per_round: int = 12
    data_rate: float = 0.6
    bend_rate: float = 0.3
    radius_prior_weight: float = 1.0
    node_spacing_radii: float = 1.0
    ownership_reach_radii: float = 1.6
    merge_gap_radii: float = 4.0
    prior_merge_gap_radii: float = 16.0
    prior_merge_angle_degrees: float = 45.0
    kink_threshold: float = 2.0
    min_support: float = 0.5
    separate_fibers: bool = True
    levels: Levels | None = None

    def replace(self, **changes: Any) -> "FitSettings":
        return replace(self, **changes)


@dataclass
class FitResult:
    """Fitted fibers in voxel coordinates, plus how to export them.

    ``centerlines`` and ``radii`` are in voxel units, ``(x, y, z)`` order,
    with voxel ``[k, j, i]`` centered at ``(i + .5, j + .5, k + .5)``.
    """

    shape: tuple[int, int, int]
    voxel_size: float
    spec: FiberSpec
    centerlines: list[np.ndarray]
    radii: np.ndarray
    support: np.ndarray
    levels: Levels
    history: list[dict[str, Any]] = field(default_factory=list)
    # With several fiber types: the specs, and each fiber's index into them.
    specs: list[FiberSpec] | None = None
    types: np.ndarray | None = None

    def spec_of(self, index: int) -> FiberSpec:
        """The spec (fiber type) of fiber ``index``."""
        if self.specs is None or self.types is None:
            return self.spec
        return self.specs[int(self.types[index])]

    # -- geometry in meters -------------------------------------------------
    @property
    def fiber_count(self) -> int:
        return len(self.centerlines)

    def centerlines_m(self) -> list[np.ndarray]:
        return [line * self.voxel_size for line in self.centerlines]

    def diameters_m(self) -> np.ndarray:
        return 2.0 * self.radii * self.voxel_size

    def cell_lengths_m(self) -> list[float]:
        return [n * self.voxel_size for n in self.shape[::-1]]

    # -- voxel outputs ------------------------------------------------------
    def label_volume(self) -> np.ndarray:
        """One-based fiber id for every voxel inside a fitted capsule (0 = void)."""
        labels, _, _ = rasterize(self.shape, self.centerlines, self.radii, signed=True)
        return labels

    # -- Tangle outputs -----------------------------------------------------
    def materials(self) -> list[Any]:
        """One Tangle material per distinct fitted diameter (rounded to 10 nm)."""
        import tangle

        cache: dict[tuple[str, int], Any] = {}
        result = []
        for index, diameter in enumerate(self.diameters_m()):
            spec = self.spec_of(index)
            bend = spec.min_bend_radius or 5.0 * spec.diameter
            key = (spec.name, int(round(diameter / 1e-8)))
            if key not in cache:
                cache[key] = tangle.Material(
                    f"{spec.name} {key[1] * 1e-2:.2f}um",
                    diameter=key[1] * 1e-8,
                    min_bend_radius=bend,
                )
            result.append(cache[key])
        return result

    def to_collection(self, name: str = "ct fit") -> Any:
        import tangle

        collection = tangle.FiberCollection(name)
        for line, material in zip(self.centerlines_m(), self.materials()):
            collection.add_fiber(line.tolist(), material, tags={"source": "ct fit"})
        return collection

    def to_assembly(self) -> Any:
        """A Tangle assembly whose cell is the scanned volume (not periodic)."""
        import tangle

        assembly = tangle.Assembly(tangle.Cell(self.cell_lengths_m()))
        assembly.insert(self.to_collection(), name="ct fit")
        return assembly

    def relax(self, settings: Any | None = None) -> tuple["FitResult", Any]:
        """Remove remaining overlaps with Tangle's own contact relaxation.

        Returns the relaxed fit and the ``RunResult`` (which reports
        ``max_penetration`` and ``max_curvature_ratio``). The default settings
        use Tangle's default backend (WGPU, the local GPU); pass
        ``RelaxationSettings(backend="cpu", ...)`` where no GPU is available.
        """
        import tangle

        recipe = tangle.Recipe(self.to_assembly())
        if settings is None:
            settings = tangle.RelaxationSettings(
                max_iterations=2000,
                max_step=0.25 * float(self.radii.mean()) * self.voxel_size,
                penetration_tolerance=0.02 * self.spec.diameter,
            )
        run = recipe.run(settings)
        relaxed = [np.asarray(line) / self.voxel_size for line in run.centerlines()]
        return replace(self, centerlines=relaxed, history=self.history + [{"stage": "tangle relax", "max_penetration_m": run.max_penetration}]), run

    # -- statistics for a generator config ------------------------------------
    def population_summary(self) -> dict[str, Any]:
        """Statistics of the fitted fibers, in meters, for building a population."""
        h = self.voxel_size
        lengths = np.array([polyline_length(line) for line in self.centerlines]) * h
        upper = np.array(self.shape[::-1], dtype=np.float64)
        margin = 1.5 * self.radii
        censored = np.array(
            [
                bool(np.any(line[[0, -1]] < margin[i]) or np.any(line[[0, -1]] > upper - margin[i]))
                for i, line in enumerate(self.centerlines)
            ]
        )
        tensor = np.zeros((3, 3))
        tilt = []
        waviness = []
        for line in self.centerlines:
            t = tangents(line)
            weights = np.linalg.norm(np.gradient(line, axis=0), axis=1)
            tensor += (weights[:, None, None] * t[:, :, None] * t[:, None, :]).sum(axis=0)
            chord = line[-1] - line[0]
            chord_length = np.linalg.norm(chord)
            if chord_length > 0:
                unit = chord / chord_length
                offsets = (line - line[0]) - ((line - line[0]) @ unit)[:, None] * unit
                waviness.append(float(np.sqrt(2.0) * np.sqrt((offsets**2).sum(axis=1).mean())) * h)
            tilt.append(t)
        total = np.trace(tensor)
        tensor = tensor / total if total > 0 else tensor
        values, vectors = np.linalg.eigh(tensor)
        diameters = self.diameters_m()
        volume = float(np.prod(self.shape))
        ends = end_statistics(self.centerlines, self.radii, self.shape, length=self.spec.length / h if self.spec.length else None)
        return {
            "fiber_count": self.fiber_count,
            "diameter_mean": float(diameters.mean()) if len(diameters) else None,
            "diameter_std": float(diameters.std()) if len(diameters) else None,
            "length_mean": float(lengths.mean()) if len(lengths) else None,
            "length_min": float(lengths.min()) if len(lengths) else None,
            "length_max": float(lengths.max()) if len(lengths) else None,
            "length_uncensored_mean": float(lengths[~censored].mean()) if (~censored).any() else None,
            "fibers_touching_boundary": int(censored.sum()),
            "curvature_amplitude_mean": float(np.mean(waviness)) if waviness else None,
            "orientation_tensor": tensor.tolist(),
            "orientation_eigenvalues": values.tolist(),
            "orientation_eigenvectors": vectors.T.tolist(),
            "volume_fraction": float((self.label_volume() > 0).sum()) / volume,
            "interior_ends": ends["interior_ends"],
            "implied_mean_length": ends["implied_length"] * h if ends["implied_length"] else None,
            "expected_interior_ends": ends.get("expected_interior_ends"),
            "expected_interior_ends_sd": ends.get("expected_interior_ends_sd"),
            "types": (
                {
                    item.name: {
                        "fiber_count": int((self.types == kind).sum()),
                        "diameter_mean": float(diameters[self.types == kind].mean()) if (self.types == kind).any() else None,
                    }
                    for kind, item in enumerate(self.specs)
                }
                if self.specs is not None and self.types is not None
                else None
            ),
            "_tilt": tilt,
        }

    def of_type(self, kind: int) -> "FitResult":
        """Only the fibers of type ``kind`` (an index into ``specs``)."""
        if self.specs is None or self.types is None:
            return self
        keep = np.flatnonzero(self.types == kind)
        return replace(
            self,
            spec=self.specs[kind],
            centerlines=[self.centerlines[i] for i in keep],
            radii=self.radii[keep],
            support=self.support[keep],
            specs=None,
            types=None,
        )

    def suggested_population(self, count: int | None = None, seed: int = 1) -> Any:
        """A ``tangle.FiberPopulation`` with the fitted statistics.

        Boundary-cut fibers make the fitted lengths a lower bound; the length
        range uses uncensored fibers when there are any. With several fiber
        types this returns one population per type.
        """
        import tangle

        if self.specs is not None and self.types is not None:
            return [self.of_type(kind).suggested_population(seed=seed + kind) for kind in range(len(self.specs))]

        summary = self.population_summary()
        values = np.array(summary["orientation_eigenvalues"])
        vectors = np.array(summary["orientation_eigenvectors"])
        if values[-1] > 0.8:
            axis = vectors[-1]
            angles = [np.arccos(np.clip(np.abs(t @ axis), 0, 1)) for t in np.concatenate(summary["_tilt"])]
            orientation = tangle.AlignedOrientation(axis.tolist(), max_angle=float(np.percentile(angles, 95)))
        elif values[0] < 0.1:
            normal = vectors[0]
            angles = [np.arcsin(np.clip(np.abs(t @ normal), 0, 1)) for t in np.concatenate(summary["_tilt"])]
            orientation = tangle.PlanarOrientation(normal=normal.tolist(), max_tilt=float(np.percentile(angles, 95)))
        else:
            orientation = tangle.IsotropicOrientation()
        lengths = np.array([polyline_length(line) for line in self.centerlines]) * self.voxel_size
        diameters = self.diameters_m()
        material = self.materials()[0].__class__(
            self.spec.name,
            diameter=float(np.mean(diameters)),
            min_bend_radius=self.spec.min_bend_radius or 5.0 * self.spec.diameter,
        )
        low_d, high_d = float(diameters.min()), float(diameters.max())
        return tangle.FiberPopulation(
            material=material,
            count=count or self.fiber_count,
            seed=seed,
            length=(float(lengths.min()), float(lengths.max())),
            diameter=(low_d, high_d) if high_d > low_d else None,
            curvature_amplitude=float(summary["curvature_amplitude_mean"] or 0.0),
            orientation=orientation,
        )

    # -- files ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        summary = self.population_summary()
        summary.pop("_tilt")
        return {
            "schema": "tangle.ct.fit/1",
            "units": "meters",
            "voxel_size": self.voxel_size,
            "shape_zyx": list(self.shape),
            "cell_lengths": self.cell_lengths_m(),
            "spec": asdict(self.spec),
            "levels": asdict(self.levels),
            "specs": [asdict(item) for item in self.specs] if self.specs else None,
            "fibers": [
                {
                    "id": i + 1,
                    "diameter": float(d),
                    "support": float(s),
                    "type": int(self.types[i]) if self.types is not None else 0,
                    "centerline": line.tolist(),
                }
                for i, (line, d, s) in enumerate(zip(self.centerlines_m(), self.diameters_m(), self.support))
            ],
            "population": summary,
            "history": self.history,
        }

    def write(self, directory: str | Path, volume: np.ndarray | None = None) -> dict[str, Path]:
        """Write the Tangle configuration, the label volume and the overlay.

        Files: ``fit.json`` (fibers + population statistics, reloadable with
        :func:`load_fit`), ``labels.tif`` (or ``.npy`` without ``tifffile``),
        and when ``volume`` is given ``overlay.tif`` (RGB stack) and
        ``overlay.png`` (three orthogonal slices, needs matplotlib).
        """
        from ._overlay import overlay_volume, save_overlay_figure

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"config": directory / "fit.json"}
        paths["config"].write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        labels = self.label_volume()
        paths["labels"] = _write_stack(directory / "labels", labels.astype(np.uint16), self.voxel_size)
        if volume is not None:
            paths["overlay_stack"] = _write_stack(directory / "overlay", overlay_volume(volume, labels), self.voxel_size, rgb=True)
            try:
                paths["overlay_png"] = save_overlay_figure(directory / "overlay.png", volume, labels, title=f"{self.fiber_count} fitted fibers")
            except ImportError:
                pass
        return paths


def _write_stack(stem: Path, array: np.ndarray, voxel_size: float, rgb: bool = False) -> Path:
    """ImageJ-format stack with the voxel spacing, so Fiji opens it as a volume."""
    try:
        import tifffile
    except ImportError:
        path = stem.with_suffix(".npy")
        np.save(path, array)
        return path
    path = stem.with_suffix(".tif")
    pixel = voxel_size * 1e6
    tifffile.imwrite(
        path,
        array,
        imagej=True,
        photometric="rgb" if rgb else "minisblack",
        resolution=(1.0 / pixel, 1.0 / pixel),
        metadata={"spacing": pixel, "unit": "um", "axes": "ZYXS" if rgb else "ZYX"},
    )
    return path


def load_fit(path: str | Path) -> FitResult:
    """Reload a ``fit.json`` written by :meth:`FitResult.write`."""
    data = json.loads(Path(path).read_text())
    h = data["voxel_size"]
    fibers = data["fibers"]

    def spec_from(values: dict[str, Any]) -> FiberSpec:
        values = dict(values)
        values["profile"] = CrossSection(**values.get("profile") or {})
        return FiberSpec(**values)

    specs = [spec_from(item) for item in data["specs"]] if data.get("specs") else None
    return FitResult(
        shape=tuple(data["shape_zyx"]),
        voxel_size=h,
        spec=spec_from(data["spec"]),
        specs=specs,
        types=np.array([f.get("type", 0) for f in fibers], dtype=int) if specs else None,
        centerlines=[np.asarray(f["centerline"]) / h for f in fibers],
        radii=np.array([0.5 * f["diameter"] / h for f in fibers]),
        support=np.array([f["support"] for f in fibers]),
        levels=Levels(**data["levels"]),
        history=data.get("history", []),
    )


def fit_fibers(
    volume: np.ndarray,
    voxel_size: float,
    spec: FiberSpec | Sequence[FiberSpec],
    settings: FitSettings | None = None,
    *,
    verbose: bool = False,
) -> FitResult:
    """Find the fibers in ``volume`` (a ``(z, y, x)`` array) that match ``spec``.

    Stages: normalize intensities; trace initial centerlines from
    distance-transform ridge seeds along the Hessian tube direction; then
    ``settings.rounds`` rounds of (continuous fit → remove duplicates and
    unsupported fibers → merge fragments → trace new fibers in what is still
    unexplained).

    ``spec`` may be a list of fiber types. Types are fitted one after another,
    largest diameter first, each on a detection image matched to its own
    brightness profile (``FiberSpec.profile``); the fibers already found claim
    their voxels, so a smaller type is not traced along a larger fiber's rim.
    """
    settings = settings or FitSettings()
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D (z, y, x) array")
    specs = [spec] if isinstance(spec, FiberSpec) else list(spec)
    if not specs:
        raise ValueError("give at least one FiberSpec")
    for item in specs:
        if 0.5 * item.diameter / voxel_size < 1.0:
            raise ValueError(
                f"fibers are only {item.diameter / voxel_size:.1f} voxels across; at least 2 are needed"
            )
    single = len(specs) == 1 and specs[0].profile.solid and specs[0].profile.brightness == 1.0

    if single or settings.levels is not None:
        image, levels = normalize(volume, denoise_sigma=settings.denoise_sigma_voxels, levels=settings.levels)
    else:
        image, levels = normalize(volume, denoise_sigma=settings.denoise_sigma_voxels, levels=_three_class_levels(volume, settings))

    history: list[dict[str, Any]] = []

    def log(stage: str, lines: list[np.ndarray], **extra: Any) -> None:
        entry = {"stage": stage, "fibers": len(lines), **extra}
        history.append(entry)
        if verbose:
            print(entry)

    order = sorted(range(len(specs)), key=lambda k: -specs[k].diameter)
    lines: list[np.ndarray] = []
    radii = np.zeros(0)
    types = np.zeros(0, dtype=int)
    for kind in order:
        item = specs[kind]
        if single:
            detect = image
        else:
            detect = _detection_image(image, item, voxel_size)
            log(f"type {item.name}", lines, diameter=item.diameter)
        found, found_radii, detect, levels = _fit_type(
            detect, None if single else image, item, settings, voxel_size, levels,
            frozen=lines, frozen_radii=radii, relevel=single, log=log,
        )
        if single:
            image = detect
        lines = lines + found
        radii = np.concatenate([radii, found_radii])
        types = np.concatenate([types, np.full(len(found), kind, dtype=int)])

    return FitResult(
        shape=tuple(int(n) for n in volume.shape),
        voxel_size=voxel_size,
        spec=specs[0],
        centerlines=lines,
        radii=np.asarray(radii, dtype=np.float64),
        support=_refine.support(image, lines),
        levels=levels,
        history=history,
        specs=None if len(specs) == 1 else specs,
        types=None if len(specs) == 1 else types,
    )


def _three_class_levels(volume: np.ndarray, settings: FitSettings) -> Levels:
    """Void and reference-fiber levels for a scan with several fiber types.

    A two-class threshold would split a dim fiber type from the bright one.
    Three classes (void, dim, bright) are found by exhaustive search over a
    histogram; void is the darkest class median and the reference fiber level
    (brightness 1) the brightest.
    """
    from scipy.ndimage import gaussian_filter

    image = np.asarray(volume, dtype=np.float32)
    if settings.denoise_sigma_voxels > 0:
        image = gaussian_filter(image, settings.denoise_sigma_voxels)
    sample = image[:: max(1, image.shape[0] // 64)].ravel()
    low, high = np.percentile(sample, [0.1, 99.9])
    histogram, edges = np.histogram(np.clip(sample, low, high), bins=128, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.concatenate([[0.0], np.cumsum(histogram)])
    moment = np.concatenate([[0.0], np.cumsum(histogram * centers)])
    total_w, total_m = weight[-1], moment[-1]
    best, split = -1.0, (1, 2)
    for a in range(1, len(centers) - 1):
        for b in range(a + 1, len(centers)):
            w0, w1, w2 = weight[a], weight[b] - weight[a], total_w - weight[b]
            if min(w0, w1, w2) <= 0:
                continue
            m0, m1, m2 = moment[a], moment[b] - moment[a], total_m - moment[b]
            between = m0 * m0 / w0 + m1 * m1 / w1 + m2 * m2 / w2
            if between > best:
                best, split = between, (a, b)
    t1, t2 = edges[split[0]], edges[split[1]]
    void = float(np.median(sample[sample < t1]))
    fiber = float(np.median(sample[sample >= t2]))
    return Levels(void=void, fiber=fiber, threshold=float(t1))


def _detection_image(image: np.ndarray, spec: FiberSpec, voxel_size: float) -> np.ndarray:
    """The scan as this fiber type sees it: about 1 on its axis, 0 in void.

    A rimmed type is smoothed at half its radius, which fills its dim core;
    the result is divided by the level such a fiber reaches on its axis.
    """
    from scipy.ndimage import gaussian_filter

    radius = 0.5 * spec.diameter / voxel_size
    sigma = 0.0 if spec.profile.solid else 0.5 * radius
    smoothed = gaussian_filter(image, sigma) if sigma > 0 else image
    return (smoothed / max(spec.profile.center_response(radius, voxel_size, sigma), 1e-3)).astype(np.float32)


def _fit_type(
    image: np.ndarray,
    mass_image: np.ndarray | None,
    spec: FiberSpec,
    settings: FitSettings,
    voxel_size: float,
    levels: Levels,
    *,
    frozen: list[np.ndarray],
    frozen_radii: np.ndarray,
    relevel: bool,
    log,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, Levels]:
    """Fit one fiber type; ``frozen`` fibers (earlier types) stay fixed but
    own their voxels and block tracing."""
    radius = 0.5 * spec.diameter / voxel_size
    bend = (spec.min_bend_radius or 5.0 * spec.diameter) / voxel_size
    min_length = (spec.min_length or 3.0 * spec.diameter) / voxel_size
    max_length = spec.max_length / voxel_size if spec.max_length else None
    length = spec.length / voxel_size if spec.length else None
    spacing = settings.node_spacing_radii * radius
    cost = end_cost(spec.length, spec.diameter)
    profile = None if spec.profile.solid and spec.profile.brightness == 1.0 else spec.profile
    frozen_count = len(frozen)

    hessian = HessianField(image, sigma=max(0.6 * radius, 1.0))
    foreground = image > 0.5
    frozen_claim = None
    if frozen_count:
        frozen_claim, _, _ = rasterize(image.shape, frozen, frozen_radii, reach=1.2 * frozen_radii)
        frozen_claim = np.where(frozen_claim > 0, -1, 0).astype(np.int32)

    def claim(lines: list[np.ndarray], radii: np.ndarray) -> np.ndarray | None:
        if not lines:
            return frozen_claim.copy() if frozen_claim is not None else None
        claimed, _, _ = rasterize(image.shape, lines, radii, reach=1.2 * radii)
        if frozen_claim is not None:
            claimed = np.where(claimed > 0, claimed, frozen_claim)
        return claimed

    lines = trace_fibers(
        image, hessian, radius=radius, min_bend_radius=bend, min_length=min_length, node_spacing=spacing,
        foreground=foreground, claimed=frozen_claim,
    )
    radii = np.full(len(lines), radius)
    log("trace", lines)
    if relevel and settings.levels is None and lines:
        # Otsu class medians put the fiber level below the fiber core (blurred
        # edge voxels are in the fiber class); re-level on the traced cores.
        image, levels = _relevel(image, levels, lines, radius)
        foreground = image > 0.5
        log("relevel", lines, void=levels.void, fiber=levels.fiber)

    for round_index in range(settings.rounds):
        for _ in range(settings.iterations_per_round):
            if not lines:
                break
            if frozen_count:
                # Earlier types own their voxels; only this type's fibers move.
                moved, mass = _refine.data_step(
                    image, frozen + lines, np.concatenate([frozen_radii, radii]),
                    reach_factor=settings.ownership_reach_radii, rate=settings.data_rate, mass_image=mass_image,
                )
                lines, mass = moved[frozen_count:], mass[frozen_count:]
            else:
                lines, mass = _refine.data_step(
                    image, lines, radii, reach_factor=settings.ownership_reach_radii, rate=settings.data_rate,
                    mass_image=mass_image,
                )
            lines = _refine.bend_step(lines, settings.bend_rate)
            radii = _refine.radius_step(
                lines, radii, mass, prior_radius=radius, tolerance=spec.diameter_tolerance,
                prior_weight=settings.radius_prior_weight, profile=profile, voxel_size=voxel_size,
            )
            occupied, _, _ = rasterize(image.shape, frozen + lines, np.concatenate([frozen_radii, radii]), signed=True)
            if frozen_count:
                occupied = np.where(occupied > frozen_count, occupied - frozen_count, np.where(occupied > 0, -1, 0))
            lines = _refine.end_step(image, lines, radii, step=spacing, occupied=occupied)
            if settings.separate_fibers:
                separated = _refine.separate_step(frozen + lines, np.concatenate([frozen_radii, radii]))
                lines = separated[frozen_count:]
            lines = _refine.respace(lines, spacing)
        scale = evidence_scale(image, lines, radii, radius) if cost > 0 else 1.0
        lines, radii, splits = _moves.split_kinks(
            lines, radii, min_bend_radius=bend, min_length=min_length, max_length=max_length,
            threshold=settings.kink_threshold, image=image, end_cost=cost, scale=scale,
        )
        lines, radii, duplicates = _moves.resolve_side_by_side(
            image, lines, radii, min_length=min_length, end_cost=cost, scale=scale
        )
        lines, radii = _moves.trim_duplicates(lines, radii, min_length=min_length)
        lines, radii = _moves.remove_unsupported(image, lines, radii, min_length=min_length, min_support=settings.min_support)
        lines, radii, merges = _moves.merge_fragments(
            image, lines, radii, max_gap=settings.merge_gap_radii * radius,
            min_bend_radius=bend, kink_threshold=settings.kink_threshold,
            end_cost=cost, scale=scale, max_prior_gap=settings.prior_merge_gap_radii * radius,
            max_prior_angle_degrees=settings.prior_merge_angle_degrees,
        )
        lines = _refine.respace(lines, spacing)
        explained = _explained_fraction(foreground, lines, radii)
        ends = end_statistics(lines, radii, image.shape, length=length)
        extra = {"interior_ends": ends["interior_ends"]}
        if ends["implied_length"]:
            extra["implied_length_m"] = ends["implied_length"] * voxel_size
        if length:
            extra["expected_interior_ends"] = round(ends["expected_interior_ends"], 1)
        log(
            f"round {round_index + 1}", lines, splits=splits, duplicates=duplicates, merges=merges,
            explained=explained, **extra,
        )
        if round_index + 1 < settings.rounds:
            born = trace_fibers(
                image, hessian, radius=radius, min_bend_radius=bend, min_length=min_length,
                node_spacing=spacing, claimed=claim(lines, radii), foreground=foreground, label_offset=len(lines),
            )
            lines = lines + born
            radii = np.concatenate([radii, np.full(len(born), radius)])
            log(f"births {round_index + 1}", lines, born=len(born))
    return lines, np.asarray(radii, dtype=np.float64), image, levels


def _relevel(image: np.ndarray, levels: Levels, lines: list[np.ndarray], radius: float) -> tuple[np.ndarray, Levels]:
    from ._geometry import sample_image

    core = float(np.median(sample_image(image, np.concatenate(lines))))
    near, _, _ = rasterize(image.shape, lines, np.full(len(lines), radius), reach=2.5 * radius)
    far = (near == 0) & (image < 0.5)
    void = float(np.median(image[far])) if far.any() else 0.0
    if core - void < 0.25:
        return image, levels
    scale = levels.fiber - levels.void
    new = Levels(
        void=levels.void + void * scale,
        fiber=levels.void + core * scale,
        threshold=levels.void + 0.5 * (void + core) * scale,
    )
    return (image - void) / (core - void), new


def _explained_fraction(foreground: np.ndarray, lines: list[np.ndarray], radii: np.ndarray) -> float:
    if not lines or not foreground.any():
        return 0.0
    labels, _, _ = rasterize(foreground.shape, lines, radii, signed=True)
    return float(((labels > 0) & foreground).sum()) / float(foreground.sum())
