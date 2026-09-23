"""Fit Tangle fibers to a CT volume using what the fibers are known to look like."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import _moves, _refine
from ._geometry import polyline_length, rasterize, tangents
from ._image import HessianField, Levels, normalize
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
    ``name``
        Material name used in the exported Tangle configuration.
    """

    diameter: float
    diameter_tolerance: float = 0.25
    min_bend_radius: float | None = None
    min_length: float | None = None
    max_length: float | None = None
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

        bend = self.spec.min_bend_radius or 5.0 * self.spec.diameter
        cache: dict[int, Any] = {}
        result = []
        for diameter in self.diameters_m():
            key = int(round(diameter / 1e-8))
            if key not in cache:
                cache[key] = tangle.Material(
                    f"{self.spec.name} {key * 1e-2:.2f}um",
                    diameter=key * 1e-8,
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
        ``max_penetration`` and ``max_curvature_ratio``).
        """
        import tangle

        recipe = tangle.Recipe(self.to_assembly())
        if settings is None:
            settings = tangle.RelaxationSettings(
                backend="cpu",
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
            "_tilt": tilt,
        }

    def suggested_population(self, count: int | None = None, seed: int = 1) -> Any:
        """A ``tangle.FiberPopulation`` with the fitted statistics.

        Boundary-cut fibers make the fitted lengths a lower bound; the length
        range uses uncensored fibers when there are any.
        """
        import tangle

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
            "fibers": [
                {
                    "id": i + 1,
                    "diameter": float(d),
                    "support": float(s),
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
        paths["labels"] = _write_stack(directory / "labels", labels.astype(np.uint16))
        if volume is not None:
            paths["overlay_stack"] = _write_stack(directory / "overlay", overlay_volume(volume, labels), rgb=True)
            try:
                paths["overlay_png"] = save_overlay_figure(directory / "overlay.png", volume, labels, title=f"{self.fiber_count} fitted fibers")
            except ImportError:
                pass
        return paths


def _write_stack(stem: Path, array: np.ndarray, rgb: bool = False) -> Path:
    try:
        import tifffile
    except ImportError:
        path = stem.with_suffix(".npy")
        np.save(path, array)
        return path
    path = stem.with_suffix(".tif")
    tifffile.imwrite(path, array, photometric="rgb" if rgb else "minisblack", metadata={"axes": "ZYXS" if rgb else "ZYX"})
    return path


def load_fit(path: str | Path) -> FitResult:
    """Reload a ``fit.json`` written by :meth:`FitResult.write`."""
    data = json.loads(Path(path).read_text())
    h = data["voxel_size"]
    fibers = data["fibers"]
    return FitResult(
        shape=tuple(data["shape_zyx"]),
        voxel_size=h,
        spec=FiberSpec(**data["spec"]),
        centerlines=[np.asarray(f["centerline"]) / h for f in fibers],
        radii=np.array([0.5 * f["diameter"] / h for f in fibers]),
        support=np.array([f["support"] for f in fibers]),
        levels=Levels(**data["levels"]),
        history=data.get("history", []),
    )


def fit_fibers(
    volume: np.ndarray,
    voxel_size: float,
    spec: FiberSpec,
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
    """
    settings = settings or FitSettings()
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D (z, y, x) array")
    radius = 0.5 * spec.diameter / voxel_size
    if radius < 1.0:
        raise ValueError(
            f"fibers are only {2 * radius:.1f} voxels across; at least 2 are needed"
        )
    bend = (spec.min_bend_radius or 5.0 * spec.diameter) / voxel_size
    min_length = (spec.min_length or 3.0 * spec.diameter) / voxel_size
    max_length = spec.max_length / voxel_size if spec.max_length else None
    spacing = settings.node_spacing_radii * radius

    image, levels = normalize(volume, denoise_sigma=settings.denoise_sigma_voxels, levels=settings.levels)
    hessian = HessianField(image, sigma=max(0.6 * radius, 1.0))
    foreground = image > 0.5

    def log(stage: str, lines: list[np.ndarray], **extra: Any) -> None:
        entry = {"stage": stage, "fibers": len(lines), **extra}
        history.append(entry)
        if verbose:
            print(entry)

    history: list[dict[str, Any]] = []
    lines = trace_fibers(
        image, hessian, radius=radius, min_bend_radius=bend, min_length=min_length, node_spacing=spacing,
        foreground=foreground,
    )
    radii = np.full(len(lines), radius)
    log("trace", lines)
    if settings.levels is None and lines:
        # Otsu class medians put the fiber level below the fiber core (blurred
        # edge voxels are in the fiber class); re-level on the traced cores.
        image, levels = _relevel(image, levels, lines, radius)
        foreground = image > 0.5
        log("relevel", lines, void=levels.void, fiber=levels.fiber)

    for round_index in range(settings.rounds):
        for _ in range(settings.iterations_per_round):
            if not lines:
                break
            lines, mass = _refine.data_step(
                image, lines, radii, reach_factor=settings.ownership_reach_radii, rate=settings.data_rate
            )
            lines = _refine.bend_step(lines, settings.bend_rate)
            radii = _refine.radius_step(
                lines, radii, mass, prior_radius=radius, tolerance=spec.diameter_tolerance,
                prior_weight=settings.radius_prior_weight,
            )
            occupied, _, _ = rasterize(image.shape, lines, radii, signed=True)
            lines = _refine.end_step(image, lines, radii, step=spacing, occupied=occupied)
            if settings.separate_fibers:
                lines = _refine.separate_step(lines, radii)
            lines = _refine.respace(lines, spacing)
        lines, radii, splits = _moves.split_kinks(
            lines, radii, min_bend_radius=bend, min_length=min_length, max_length=max_length,
            threshold=settings.kink_threshold, image=image,
        )
        lines, radii, duplicates = _moves.resolve_side_by_side(image, lines, radii, min_length=min_length)
        lines, radii = _moves.trim_duplicates(lines, radii, min_length=min_length)
        lines, radii = _moves.remove_unsupported(image, lines, radii, min_length=min_length, min_support=settings.min_support)
        lines, radii, merges = _moves.merge_fragments(
            image, lines, radii, max_gap=settings.merge_gap_radii * radius,
            min_bend_radius=bend, kink_threshold=settings.kink_threshold,
        )
        lines = _refine.respace(lines, spacing)
        explained = _explained_fraction(foreground, lines, radii)
        log(f"round {round_index + 1}", lines, splits=splits, duplicates=duplicates, merges=merges, explained=explained)
        if round_index + 1 < settings.rounds:
            claimed, _, _ = rasterize(image.shape, lines, radii, reach=1.2 * radii)
            born = trace_fibers(
                image, hessian, radius=radius, min_bend_radius=bend, min_length=min_length,
                node_spacing=spacing, claimed=claimed, foreground=foreground, label_offset=len(lines),
            )
            lines = lines + born
            radii = np.concatenate([radii, np.full(len(born), radius)])
            log(f"births {round_index + 1}", lines, born=len(born))

    return FitResult(
        shape=tuple(int(n) for n in volume.shape),
        voxel_size=voxel_size,
        spec=spec,
        centerlines=lines,
        radii=np.asarray(radii, dtype=np.float64),
        support=_refine.support(image, lines),
        levels=levels,
        history=history,
    )


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
