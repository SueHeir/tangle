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
    name: str = "ct fiber"

    def replace(self, **changes: Any) -> "FiberSpec":
        return replace(self, **changes)


@dataclass(frozen=True)
class FitSettings:
    """Numerical settings; the defaults are meant to work unchanged.

    Lengths here are in voxels or in fiber radii, as named. Fibers move only
    through Tangle's own relaxation with the scan as an extra force
    (``tangle.ImageRelaxer``); ``backend`` picks where it runs (``"wgpu"``,
    the GPU, by default; ``"cpu"`` where there is none).
    """

    denoise_sigma_voxels: float = 0.7
    fill_mask_holes: bool = True
    # How much thicker (voxels, in radius) the foreground makes every fiber
    # look, as a generous mask threshold does. None estimates it from the
    # fits; it is taken off before each fiber's type and radius are chosen.
    thickness_margin_voxels: float | None = None
    rounds: int = 3
    radius_prior_weight: float = 1.0
    node_spacing_radii: float = 1.0
    merge_gap_radii: float = 4.0
    prior_merge_gap_radii: float = 16.0
    prior_merge_angle_degrees: float = 45.0
    kink_threshold: float = 2.0
    min_support: float = 0.5
    levels: Levels | None = None
    backend: str | None = None
    solver_batches: int = 3
    solver_iterations: int = 300
    solver_settle_iterations: int = 100
    solver_image_rate: float = 0.3
    solver_reach_radii: float = 1.4

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
        values.pop("profile", None)  # fits written before types were chosen by size
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
    exclude: np.ndarray | None = None,
    verbose: bool = False,
) -> FitResult:
    """Find the fibers in ``volume`` (a ``(z, y, x)`` array) that match ``spec``.

    ``volume`` is a grey-level scan or a binary fiber mask (``bool``, or any
    array with two values; the larger one is fiber). A mask may include some
    background mistaken for fiber. ``exclude`` optionally marks voxels known
    not to be fiber; they are treated as void.

    Stages: trace initial centerlines from distance-transform ridge seeds
    along the Hessian tube direction; then ``settings.rounds`` rounds of
    (Tangle's relaxation with the scan as an extra force → split kinks,
    resolve side-by-side fits, remove duplicates and unsupported fibers,
    join fragments → trace new fibers in what is still unexplained), and a
    final relaxation. See ``docs/ct_fitting_internals.md``.

    ``spec`` may be a list of fiber types that differ in diameter. They are
    traced largest first, and every fiber's type is then decided by its
    size: the local thickness of the foreground along it picks the nearest
    diameter, after every solver batch, and sets its radius prior, bend
    limit and length prior.
    """
    from . import _device

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
    if exclude is not None and np.shape(exclude) != volume.shape:
        raise ValueError("exclude must have the same shape as volume")
    if not _device.available():
        raise RuntimeError("tangle.ct needs a Tangle build with tangle.ImageRelaxer")

    history: list[dict[str, Any]] = []

    def log(stage: str, lines: list[np.ndarray], **extra: Any) -> None:
        entry = {"stage": stage, "fibers": len(lines), **extra}
        history.append(entry)
        if verbose:
            print(entry)

    binary = _is_mask(volume)
    if binary:
        largest = max(0.5 * item.diameter for item in specs) / voxel_size
        image, levels = _mask_image(volume, exclude, settings, largest)
    else:
        image, levels = normalize(volume, denoise_sigma=settings.denoise_sigma_voxels, levels=settings.levels)
        if exclude is not None:
            image = np.where(np.asarray(exclude, dtype=bool), np.float32(0.0), image)
    log("input", [], mask=binary)

    fitter = _Fitter(image, specs, settings, voxel_size, log)
    lines = fitter.trace([], np.zeros(0))
    radii, types = fitter.classify(lines)
    log("trace", lines, types=fitter.counts(types), thickness_margin=round(fitter.margin, 2))
    if not binary and len(specs) == 1 and settings.levels is None and lines:
        # Otsu class medians put the fiber level below the fiber core (blurred
        # edge voxels are in the fiber class); re-level on the traced cores.
        image, levels = _relevel(image, levels, lines, float(fitter.radius[0]))
        fitter.set_image(image)
        radii, types = fitter.classify(lines)
        log("relevel", lines, void=levels.void, fiber=levels.fiber)

    for round_index in range(settings.rounds):
        for _ in range(settings.solver_batches):
            if not lines:
                break
            lines = fitter.solve(lines, radii, types)
            radii, types = fitter.classify(lines)
            lines = _refine.respace(lines, fitter.spacing)
        lines, radii, types, counts = fitter.topology(lines, radii, types)
        log(
            f"round {round_index + 1}", lines, **counts, thickness_margin=round(fitter.margin, 2),
            **fitter.end_summary(lines, radii, types),
        )
        if round_index + 1 < settings.rounds:
            born = fitter.trace(lines, radii)
            born_radii, born_types = fitter.classify(born)
            lines = lines + born
            radii = np.concatenate([radii, born_radii])
            types = np.concatenate([types, born_types])
            log(f"births {round_index + 1}", lines, born=len(born), types=fitter.counts(types))
    if lines:
        # The last round's splits and joins are not yet admissible fibers. The
        # fit is returned exactly as the solver leaves it, with the radii and
        # types (so bend limits) it was solved with.
        lines = fitter.solve(lines, radii, types)
        log("final solve", lines, types=fitter.counts(types))

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
        types=None if len(specs) == 1 else np.asarray(types, dtype=int),
    )


def _is_mask(volume: np.ndarray) -> bool:
    if volume.dtype == bool:
        return True
    if np.unique(volume.ravel()[::97]).size > 2:
        return False
    return np.unique(volume).size <= 2


def _mask_image(
    volume: np.ndarray, exclude: np.ndarray | None, settings: FitSettings, largest_radius: float
) -> tuple[np.ndarray, Levels]:
    """A binary fiber mask as a 0/1 image, with fiber cores filled (a
    threshold can miss a dim core) and a light blur, so the image force sees
    a smooth edge."""
    from scipy.ndimage import gaussian_filter

    if volume.dtype == bool:
        mask = volume.copy()
    else:
        values = np.unique(volume)
        mask = volume == values[-1] if values.size == 2 else np.zeros(volume.shape, dtype=bool)
    if settings.fill_mask_holes:
        mask |= _core_holes(mask, np.pi * (largest_radius + 1.0) ** 2)
    if exclude is not None:
        mask &= ~np.asarray(exclude, dtype=bool)
    image = mask.astype(np.float32)
    if settings.denoise_sigma_voxels > 0:
        image = gaussian_filter(image, settings.denoise_sigma_voxels)
    return image, Levels(void=0.0, fiber=1.0, threshold=0.5)


def _core_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Enclosed holes no larger than a fiber's cross-section, slice by slice
    along each axis. A hollow fiber is a closed ring in the slices across
    it (in 3D it is a tube open at both ends, so a 3D fill misses it); larger
    holes are void between fibers that happen to enclose it in a slice."""
    from scipy.ndimage import binary_fill_holes, label

    holes = np.zeros(mask.shape, dtype=bool)
    for axis in range(3):
        planes = np.moveaxis(mask, axis, 0)
        found = np.moveaxis(holes, axis, 0)  # a view: writes land in ``holes``
        for k, plane in enumerate(planes):
            enclosed = binary_fill_holes(plane) & ~plane
            if not enclosed.any():
                continue
            ids, count = label(enclosed)
            sizes = np.bincount(ids.ravel(), minlength=count + 1)
            small = sizes <= max_area
            small[0] = False
            found[k] |= small[ids]
    return holes


class _Fitter:
    """Per-type parameters and the steps of :func:`fit_fibers`."""

    def __init__(self, image: np.ndarray, specs: list[FiberSpec], settings: FitSettings, voxel_size: float, log) -> None:
        self.specs = specs
        self.settings = settings
        self.h = voxel_size
        self.log = log
        h = voxel_size
        self.radius = np.array([0.5 * item.diameter / h for item in specs])
        self.bend = np.array([(item.min_bend_radius or 5.0 * item.diameter) / h for item in specs])
        self.min_length = np.array([(item.min_length or 3.0 * item.diameter) / h for item in specs])
        self.max_length = [item.max_length / h if item.max_length else None for item in specs]
        self.length = [item.length / h if item.length else None for item in specs]
        self.cost = np.array([end_cost(item.length, item.diameter) for item in specs])
        self.tolerance = np.array([item.diameter_tolerance for item in specs])
        self.spacing = settings.node_spacing_radii * float(self.radius.min())
        self.order = [int(k) for k in np.argsort(-self.radius, kind="stable")]  # largest first
        self.hessians: dict[int, HessianField] = {}
        self.margin = settings.thickness_margin_voxels or 0.0
        self.set_image(image)

    def set_image(self, image: np.ndarray) -> None:
        from scipy.ndimage import distance_transform_edt, maximum_filter

        self.image = image
        self.foreground = image > 0.5
        # Local thickness: distance to the nearest void voxel center, taken
        # as the maximum over the 3x3x3 neighborhood so a centerline between
        # voxel centers reads the axis value rather than an interpolated,
        # lower one. On a fiber axis it is about the radius: the foreground
        # edge sits at the half-maximum, which is the fiber surface.
        self.depth = maximum_filter(distance_transform_edt(self.foreground), size=3).astype(np.float32)
        self.hessians = {}

    def hessian(self, kind: int) -> HessianField:
        if kind not in self.hessians:
            self.hessians[kind] = HessianField(self.image, sigma=max(0.6 * float(self.radius[kind]), 1.0))
        return self.hessians[kind]

    def counts(self, types: np.ndarray) -> dict[str, int] | None:
        if len(self.specs) == 1:
            return None
        return {item.name: int((np.asarray(types) == k).sum()) for k, item in enumerate(self.specs)}

    def claim(self, lines: list[np.ndarray], radii: np.ndarray) -> np.ndarray | None:
        if not lines:
            return None
        claimed, _, _ = rasterize(self.image.shape, lines, np.asarray(radii), reach=1.2 * np.asarray(radii))
        return claimed

    def trace(self, lines: list[np.ndarray], radii: np.ndarray) -> list[np.ndarray]:
        """New traces where the foreground is not yet claimed, largest type first."""
        found: list[np.ndarray] = []
        found_radii: list[float] = []
        smallest = float(self.radius.min())
        for kind in self.order:
            r = float(self.radius[kind])
            known = lines + found
            claimed = self.claim(known, np.concatenate([np.asarray(radii, dtype=np.float64), found_radii]))
            new = trace_fibers(
                self.image, self.hessian(kind), radius=r, min_bend_radius=float(self.bend[kind]),
                min_length=float(self.min_length[kind]), node_spacing=self.spacing, claimed=claimed,
                foreground=self.foreground, label_offset=len(known),
                # A larger type is only seeded where the foreground is thicker
                # than the smaller types could make it.
                seed_depth_radii=0.7 if r > smallest else 0.5,
            )
            found += new
            found_radii += [r] * len(new)
        return found

    def classify(self, lines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Each fiber's type (the diameter nearest its local thickness) and radius.

        The thickness is the foreground's depth along the centerline, less
        the margin by which the foreground over-reaches (see
        ``FitSettings.thickness_margin_voxels``); estimated, the margin is
        the median excess of the fibers over their nearest type.
        """
        from ._geometry import sample_image

        if not lines:
            return np.zeros(0), np.zeros(0, dtype=int)
        measured = np.empty(len(lines))
        for i, line in enumerate(lines):
            inner = line[1:-1] if len(line) > 2 else line
            measured[i] = max(float(np.median(sample_image(self.depth, inner))), 0.5)
        margin = self.settings.thickness_margin_voxels
        if margin is None:
            margin = 0.0
            for _ in range(3):
                types = self._nearest(measured - margin)
                margin = float(np.clip(np.median(measured - self.radius[types]), -0.5, float(self.radius.min())))
            self.margin = margin
        types = self._nearest(measured - margin)
        prior = self.radius[types]
        w = self.settings.radius_prior_weight
        blended = (np.maximum(measured - margin, 0.5) + w * prior) / (1.0 + w)
        tolerance = self.tolerance[types]
        radii = np.clip(blended, prior * (1 - tolerance), prior * (1 + tolerance))
        return radii, types

    def _nearest(self, radius: np.ndarray) -> np.ndarray:
        radius = np.maximum(radius, 0.25)
        return np.argmin(np.abs(np.log(radius[:, None]) - np.log(self.radius[None, :])), axis=1).astype(int)

    def solve(self, lines: list[np.ndarray], radii: np.ndarray, types: np.ndarray) -> list[np.ndarray]:
        from . import _device

        s = self.settings
        return _device.relax(
            self.image, lines, radii, self.bend[np.asarray(types, dtype=int)], voxel_size=self.h,
            spacing=self.spacing, rate=s.solver_image_rate, reach_radii=s.solver_reach_radii,
            iterations=s.solver_iterations, settle=s.solver_settle_iterations, backend=s.backend, log=self.log,
        )

    def topology(
        self, lines: list[np.ndarray], radii: np.ndarray, types: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        """Splits, side-by-side, duplicates, unsupported fibers and joins, one type at a time."""
        s = self.settings
        out_lines: list[np.ndarray] = []
        out_radii: list[np.ndarray] = []
        out_types: list[np.ndarray] = []
        counts = {"splits": 0, "duplicates": 0, "merges": 0}
        for kind in range(len(self.specs)):
            pick = np.flatnonzero(np.asarray(types) == kind)
            if pick.size == 0:
                continue
            group = [lines[i] for i in pick]
            group_radii = np.asarray(radii)[pick]
            r, bend, min_length = float(self.radius[kind]), float(self.bend[kind]), float(self.min_length[kind])
            cost = float(self.cost[kind])
            scale = evidence_scale(self.image, group, group_radii, r) if cost > 0 else 1.0
            group, group_radii, splits = _moves.split_kinks(
                group, group_radii, min_bend_radius=bend, min_length=min_length, max_length=self.max_length[kind],
                threshold=s.kink_threshold, image=self.image, end_cost=cost, scale=scale,
            )
            group, group_radii, duplicates = _moves.resolve_side_by_side(
                self.image, group, group_radii, min_length=min_length, end_cost=cost, scale=scale
            )
            group, group_radii = _moves.trim_duplicates(group, group_radii, min_length=min_length)
            group, group_radii = _moves.remove_unsupported(
                self.image, group, group_radii, min_length=min_length, min_support=s.min_support
            )
            group, group_radii, merges = _moves.merge_fragments(
                self.image, group, group_radii, max_gap=s.merge_gap_radii * r,
                min_bend_radius=bend, kink_threshold=s.kink_threshold,
                end_cost=cost, scale=scale, max_prior_gap=s.prior_merge_gap_radii * r,
                max_prior_angle_degrees=s.prior_merge_angle_degrees,
            )
            counts["splits"] += splits
            counts["duplicates"] += duplicates
            counts["merges"] += merges
            out_lines += group
            out_radii.append(np.asarray(group_radii, dtype=np.float64))
            out_types.append(np.full(len(group), kind, dtype=int))
        lines = _refine.respace(out_lines, self.spacing)
        radii = np.concatenate(out_radii) if out_radii else np.zeros(0)
        types = np.concatenate(out_types) if out_types else np.zeros(0, dtype=int)
        counts["explained"] = _explained_fraction(self.foreground, lines, radii)
        counts["types"] = self.counts(types)
        return lines, radii, types, counts

    def end_summary(self, lines: list[np.ndarray], radii: np.ndarray, types: np.ndarray) -> dict[str, Any]:
        ends = end_statistics(lines, radii, self.image.shape)
        extra: dict[str, Any] = {"interior_ends": ends["interior_ends"]}
        if ends["implied_length"]:
            extra["implied_length_m"] = ends["implied_length"] * self.h
        if all(self.length):
            expected = 0.0
            for kind, length in enumerate(self.length):
                group = [line for line, t in zip(lines, types) if t == kind]
                expected += 2.0 * sum(polyline_length(line) for line in group) / length
            extra["expected_interior_ends"] = round(expected, 1)
        return extra


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
