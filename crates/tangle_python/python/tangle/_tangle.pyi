from collections.abc import Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Any, Literal

Point = Sequence[float]
Centerline = Sequence[Point]
Matrix3 = Sequence[Sequence[float]]
Path = str | PathLike[str]

Axis = Literal["x", "y", "z", 0, 1, 2]
"""One Cartesian axis, by letter or index."""
AxisSet = str | Sequence[bool] | Axis
"""Several axes: letters such as ``"xy"``, one axis, or a bool triple."""
Direction = Axis | Sequence[float]
"""An axis, or an arbitrary nonzero 3-vector."""
Range = float | tuple[float, float]
"""A fixed value, or a ``(min, max)`` uniform range."""
ShapeDistribution = dict[str, Any]
"""``count``, ``mean``, ``standard_deviation`` and evenly spaced ``quantiles``."""

Backend = Literal["wgpu", "cpu"]
MotionModel = Literal["flexible", "rigid_translation"]
ContactAggregation = Literal["uniform_average", "penetration_weighted", "deepest_only"]
OvitoColoring = Literal["fiber", "curvature_ratio", "refinement_level"]
BpmExportMode = Literal[
    "spheres_exact",
    "spheres_dynamic",
    "spherocylinders_exact",
    "spherocylinders_constant",
]
CompactionKinematics = Literal["rigid_fiber_centers", "moving_walls", "affine_vertices"]
BudgetExhaustion = Literal["fail", "continue_if_hard_ok"]
CenterlineShape = Literal["straight", "curved"]
AdaptiveProfile = Literal["balanced", "fast", "strict"]
OverridePreset = Literal["contact_first", "curvature_cleanup", "contact_cleanup"]

class RecipeError(RuntimeError):
    """A recipe operation failed while ``Recipe.run()`` executed it."""

    operation_index: int
    operation: str
    iteration: int
    reason: str

# --- Results -----------------------------------------------------------------

class AnalysisReport:
    @property
    def schema_version(self) -> int: ...
    @property
    def fiber_count(self) -> int: ...
    @property
    def segment_count(self) -> int: ...
    @property
    def vertex_count(self) -> int: ...
    @property
    def junction_count(self) -> int: ...
    @property
    def total_centerline_length(self) -> float: ...
    @property
    def nominal_swept_volume_fraction(self) -> float: ...
    @property
    def length_weighted_orientation_tensor(self) -> list[list[float]]: ...
    @property
    def volume_weighted_orientation_tensor(self) -> list[list[float]]: ...
    @property
    def max_curvature(self) -> float: ...
    @property
    def max_curvature_ratio(self) -> float: ...
    @property
    def curvature_limit_violations(self) -> int: ...
    def to_json(self, pretty: bool = ...) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def write_json(self, path: Path) -> None: ...

class NeighborReport:
    @property
    def schema_version(self) -> int: ...
    @property
    def contact_gap(self) -> float: ...
    @property
    def neighbor_gap(self) -> float: ...
    @property
    def sample_spacing(self) -> float: ...
    @property
    def contact_count(self) -> int: ...
    @property
    def contacts_per_length(self) -> float: ...
    @property
    def random_baseline_contacts_per_length(self) -> float | None: ...
    @property
    def contact_ratio_to_random(self) -> float | None: ...
    @property
    def contact_count_dispersion(self) -> float | None: ...
    @property
    def in_axis_contact_fraction(self) -> float: ...
    @property
    def median_crossing_angle_degrees(self) -> float | None: ...
    @property
    def median_excess_persistence(self) -> float | None: ...
    @property
    def median_in_axis_contact_length(self) -> float | None: ...
    @property
    def mean_free_length(self) -> float | None: ...
    @property
    def mean_neighbors(self) -> float: ...
    @property
    def mean_in_axis_neighbors(self) -> float: ...
    @property
    def neighbor_count_histogram(self) -> list[int]: ...
    @property
    def in_axis_neighbor_count_histogram(self) -> list[int]: ...
    @property
    def turnover_lags(self) -> list[float]: ...
    @property
    def neighbor_turnover(self) -> list[float | None]: ...
    @property
    def in_axis_neighbor_turnover(self) -> list[float | None]: ...
    @property
    def neighbor_correlation_length(self) -> float | None: ...
    @property
    def in_axis_correlation_length(self) -> float | None: ...
    @property
    def free_lengths(self) -> list[float]: ...
    @property
    def crossing_angles_degrees(self) -> list[float]: ...
    @property
    def excess_persistence(self) -> list[float]: ...
    @property
    def contacts_per_fiber_length(self) -> list[float]: ...
    def to_json(self, pretty: bool = ...) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def write_json(self, path: Path) -> None: ...

class ShapeReport:
    @property
    def schema_version(self) -> int: ...
    @property
    def sample_spacing(self) -> float: ...
    @property
    def min_torsion_curvature(self) -> float: ...
    @property
    def orientation_axis(self) -> list[float]: ...
    @property
    def fiber_count(self) -> int: ...
    @property
    def sample_count(self) -> int: ...
    @property
    def total_length(self) -> float: ...
    @property
    def curvature(self) -> ShapeDistribution | None: ...
    @property
    def torsion(self) -> ShapeDistribution | None: ...
    @property
    def absolute_torsion(self) -> ShapeDistribution | None: ...
    @property
    def torsion_defined_fraction(self) -> float | None: ...
    @property
    def curl_index(self) -> ShapeDistribution | None: ...
    @property
    def fiber_length(self) -> ShapeDistribution | None: ...
    @property
    def axis_cosine(self) -> ShapeDistribution | None: ...
    @property
    def mean_squared_axis_cosine(self) -> float | None: ...
    @property
    def schladitz_beta(self) -> float | None: ...
    @property
    def schladitz_fit_distance(self) -> float | None: ...
    @property
    def tangent_correlation_lags(self) -> list[float]: ...
    @property
    def tangent_correlation(self) -> list[float | None]: ...
    @property
    def tangent_correlation_length(self) -> float | None: ...
    @property
    def persistence_length(self) -> float | None: ...
    @property
    def fiber_curl_indices(self) -> list[float | None]: ...
    @property
    def fiber_mean_curvatures(self) -> list[float | None]: ...
    def to_json(self, pretty: bool = ...) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def write_json(self, path: Path) -> None: ...

class PumaExportReport:
    @property
    def output_directory(self) -> Path: ...
    @property
    def voxel_counts(self) -> list[int]: ...
    @property
    def total_voxels(self) -> int: ...
    @property
    def occupied_voxels(self) -> int: ...
    @property
    def voxel_volume_fraction(self) -> float: ...
    @property
    def ambiguous_voxels(self) -> int: ...
    @property
    def domain_path(self) -> Path: ...
    @property
    def fiber_ids_path(self) -> Path | None: ...
    @property
    def interface_path(self) -> Path | None: ...
    @property
    def manifest_path(self) -> Path: ...
    @property
    def analysis_path(self) -> Path: ...

# --- Geometry ----------------------------------------------------------------

class Cell:
    def __init__(
        self,
        lengths: Point,
        periodic: AxisSet = ...,
        origin: Point = ...,
        *,
        stack_axis: Axis | None = ...,
    ) -> None: ...
    @property
    def lengths(self) -> list[float]: ...
    @property
    def periodic(self) -> list[bool]: ...
    @property
    def origin(self) -> list[float]: ...
    @property
    def stack_axis(self) -> int: ...

class Material:
    def __init__(
        self,
        name: str,
        diameter: float,
        min_bend_radius: float | None = ...,
        *,
        thickness: float | None = ...,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def diameter(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def min_bend_radius(self) -> float | None: ...
    @property
    def thickness(self) -> float | None: ...
    @property
    def is_oval(self) -> bool: ...

class Assembly:
    def __init__(self, cell: Cell) -> None: ...
    @property
    def fiber_count(self) -> int: ...
    @property
    def cell(self) -> Cell: ...
    def centerlines(self) -> list[list[list[float]]]: ...
    def long_axes(self) -> list[list[list[float]]]: ...
    def insert(
        self,
        collection: FiberCollection,
        *,
        name: str | None = ...,
        translation: Point = ...,
        rotation: Matrix3 | None = ...,
    ) -> FiberSelection: ...
    def characterize(self) -> AnalysisReport: ...
    def characterize_neighbors(
        self,
        contact_gap: float,
        *,
        neighbor_gap: float | None = ...,
        in_axis_angle_degrees: float = ...,
        sample_spacing: float | None = ...,
        max_lag: float | None = ...,
        lag_count: int = ...,
    ) -> NeighborReport: ...
    def characterize_shape(
        self,
        *,
        sample_spacing: float | None = ...,
        max_lag: float | None = ...,
        lag_count: int = ...,
        quantile_count: int = ...,
        orientation_axis: Point = ...,
        min_torsion_curvature: float | None = ...,
    ) -> ShapeReport: ...
    def export_puma(
        self,
        output_directory: Path,
        voxel_size: float,
        *,
        include_fiber_ids: bool = ...,
        include_interface: bool = ...,
        ambiguity_tolerance: float | None = ...,
    ) -> PumaExportReport: ...

class FiberCollection:
    name: str
    def __init__(self, name: str = ...) -> None: ...
    def add_fiber(
        self,
        centerline: Centerline,
        material: Material,
        *,
        rest_centerline: Centerline | None = ...,
        tags: Mapping[str, str] | None = ...,
        formation_layer: int | None = ...,
        long_axis: Point | Sequence[Point] | None = ...,
    ) -> int: ...
    @classmethod
    def from_centerlines(
        cls,
        centerlines: Sequence[Centerline],
        material: Material,
        *,
        name: str = ...,
        formation_layer: int | None = ...,
    ) -> FiberCollection: ...
    def centerlines(self) -> list[list[list[float]]]: ...
    def rest_centerlines(self) -> list[list[list[float]]]: ...
    def long_axes(self) -> list[list[list[float]]]: ...
    def extend(self, other: FiberCollection) -> None: ...
    def __add__(self, other: FiberCollection) -> FiberCollection: ...
    def layer_ids(self) -> list[int]: ...
    def select_layer(
        self,
        layer: int,
        *,
        name: str | None = ...,
        allow_empty: bool = ...,
    ) -> FiberCollection: ...
    def __len__(self) -> int: ...

# --- Generation --------------------------------------------------------------

class IsotropicOrientation:
    def __init__(self) -> None: ...

class PlanarOrientation:
    def __init__(self, *, normal: Direction | None = ..., max_tilt: float = ...) -> None: ...
    @property
    def normal(self) -> Direction | None: ...
    @property
    def max_tilt(self) -> float: ...

class LayeredBiaxialOrientation:
    def __init__(
        self,
        *,
        normal: Direction | None = ...,
        primary_fraction: float = ...,
        cross_fraction: float = ...,
        max_in_plane_deviation: float = ...,
        max_tilt: float = ...,
        seed: int = ...,
    ) -> None: ...
    @property
    def normal(self) -> Direction | None: ...
    @property
    def primary_fraction(self) -> float: ...
    @property
    def cross_fraction(self) -> float: ...
    @property
    def max_in_plane_deviation(self) -> float: ...
    @property
    def max_tilt(self) -> float: ...
    @property
    def seed(self) -> int: ...

class AlignedOrientation:
    def __init__(self, axis: Direction, *, max_angle: float = ...) -> None: ...
    @property
    def axis(self) -> Direction: ...
    @property
    def max_angle(self) -> float: ...

Orientation = (
    IsotropicOrientation | PlanarOrientation | LayeredBiaxialOrientation | AlignedOrientation
)

class UniformPosition:
    def __init__(self) -> None: ...

class LayeredPosition:
    def __init__(
        self,
        layer_count: int,
        *,
        axis: Axis | None = ...,
        jitter_fraction: float = ...,
    ) -> None: ...
    @property
    def layer_count(self) -> int: ...
    @property
    def axis(self) -> int | None: ...
    @property
    def jitter_fraction(self) -> float: ...

class DensityGradientPosition:
    def __init__(
        self,
        *,
        axis: Axis | None = ...,
        exponent: float = ...,
        toward_high: bool = ...,
    ) -> None: ...
    @property
    def axis(self) -> int | None: ...
    @property
    def exponent(self) -> float: ...
    @property
    def toward_high(self) -> bool: ...

Position = UniformPosition | LayeredPosition | DensityGradientPosition

class FiberPopulation:
    material: Material
    count: int
    segments_per_fiber: int
    seed: int
    length: Range
    diameter: Range | None
    curvature_amplitude: Range
    nominal_parent_length: float | None
    orientation: Orientation
    position: Position
    max_attempts_per_fiber: int
    def __init__(
        self,
        *,
        material: Material = ...,
        count: int = ...,
        segments_per_fiber: int = ...,
        seed: int = ...,
        length: Range = ...,
        diameter: Range | None = ...,
        curvature_amplitude: Range = ...,
        nominal_parent_length: float | None = ...,
        orientation: Orientation = ...,
        position: Position = ...,
        max_attempts_per_fiber: int = ...,
    ) -> None: ...
    def copy(self) -> FiberPopulation: ...
    def replace(self, **changes: Any) -> FiberPopulation: ...

def generate_point_crossing(
    cell: Cell,
    *,
    material: Material | None = ...,
    count: int = ...,
    length: float = ...,
    name: str = ...,
) -> FiberCollection: ...

def generate_multisegment_crossing(
    cell: Cell,
    *,
    material: Material | None = ...,
    count: int = ...,
    segments_per_fiber: int = ...,
    length: float = ...,
    placed_chord_fraction: float = ...,
    rest_shape: CenterlineShape = ...,
    rest_amplitude: float = ...,
    placed_shape: CenterlineShape = ...,
    placed_amplitude: float = ...,
    name: str = ...,
) -> FiberCollection: ...

def generate_fiber_pair_crossing(
    cell: Cell,
    *,
    material: Material | None = ...,
    segments_per_fiber: int = ...,
    length: float = ...,
    axis_separation: float = ...,
    crossing_angle_degrees: float = ...,
    name: str = ...,
) -> FiberCollection: ...

def generate_fiber_population(
    cell: Cell,
    population: FiberPopulation,
    *,
    name: str = ...,
) -> FiberCollection: ...

# --- Settings, policies, and overrides ---------------------------------------

class CheckpointSettings:
    case_id: str
    path: Path
    interval_iterations: int
    resume: bool
    resume_path: Path | None
    resume_case_id: str | None
    fresh_formation_on_resume: bool
    def __init__(
        self,
        case_id: str,
        path: Path,
        *,
        interval_iterations: int = ...,
        resume: bool = ...,
        resume_path: Path | None = ...,
        resume_case_id: str | None = ...,
        fresh_formation_on_resume: bool = ...,
    ) -> None: ...
    def copy(self) -> CheckpointSettings: ...
    def replace(self, **changes: Any) -> CheckpointSettings: ...

class AdaptiveSegmentationSettings:
    contact_length_over_diameter: float
    min_length_over_diameter: float
    max_refinement_levels: int
    refinement_interval: int
    refinement_persistence: int
    coarsening_persistence: int
    coarsening_error_over_diameter: float
    coarsening_curvature_ratio: float
    def __init__(
        self,
        *,
        contact_length_over_diameter: float = ...,
        min_length_over_diameter: float = ...,
        max_refinement_levels: int = ...,
        refinement_interval: int = ...,
        refinement_persistence: int = ...,
        coarsening_persistence: int = ...,
        coarsening_error_over_diameter: float = ...,
        coarsening_curvature_ratio: float = ...,
    ) -> None: ...
    @classmethod
    def profile(cls, name: AdaptiveProfile, **changes: Any) -> AdaptiveSegmentationSettings: ...
    def copy(self) -> AdaptiveSegmentationSettings: ...
    def replace(self, **changes: Any) -> AdaptiveSegmentationSettings: ...
    def to_dict(self) -> dict[str, Any]: ...

class VolumeFractionTarget:
    def __init__(self, value: float) -> None: ...
    @property
    def value(self) -> float: ...

class CellVolumeTarget:
    def __init__(self, value: float) -> None: ...
    @property
    def value(self) -> float: ...

class CellLengthsTarget:
    def __init__(self, lengths: Point) -> None: ...
    @property
    def lengths(self) -> list[float]: ...

class MeanPressureTarget:
    def __init__(self, value: float) -> None: ...
    @property
    def value(self) -> float: ...

class DirectionalPressureTarget:
    def __init__(self, pressures: Point) -> None: ...
    @property
    def pressures(self) -> list[float]: ...

class PenaltyEnergyTarget:
    def __init__(self, value: float) -> None: ...
    @property
    def value(self) -> float: ...

CompactionTarget = (
    VolumeFractionTarget
    | CellVolumeTarget
    | CellLengthsTarget
    | MeanPressureTarget
    | DirectionalPressureTarget
    | PenaltyEnergyTarget
)

class AxisWeightsPath:
    def __init__(self, weights: AxisSet | Point | None = ...) -> None: ...
    @property
    def weights(self) -> list[float] | None: ...

class EqualPressurePath:
    def __init__(self, axes: AxisSet, *, pressure_floor: float = ...) -> None: ...
    @property
    def axes(self) -> list[bool]: ...
    @property
    def pressure_floor(self) -> float: ...

class StressRatioPath:
    def __init__(self, ratio: Point, *, pressure_floor: float = ...) -> None: ...
    @property
    def ratio(self) -> list[float]: ...
    @property
    def pressure_floor(self) -> float: ...

class MinimumWorkPath:
    def __init__(self, axes: AxisSet) -> None: ...
    @property
    def axes(self) -> list[bool]: ...

CompactionPath = AxisWeightsPath | EqualPressurePath | StressRatioPath | MinimumWorkPath

class CompactionSettings:
    target: CompactionTarget
    path: CompactionPath
    kinematics: CompactionKinematics
    cell_anchor: list[float]
    balance_opposing_faces: bool
    face_pressure_floor: float
    face_balance_strength: float
    initial_log_strain: float
    min_log_strain: float
    max_log_strain: float
    growth_factor: float
    shrink_factor: float
    relax_iterations: int
    max_shortening_over_min_diameter: float
    max_penetration: float
    max_curvature_ratio: float
    max_pressure: float
    max_penalty_energy: float
    max_steps: int
    max_relax_windows: int
    contact_energy_stiffness: float
    stretch_energy_stiffness: float
    bending_energy_stiffness: float
    target_tolerance: float
    def __init__(
        self,
        target: CompactionTarget | None = ...,
        *,
        path: CompactionPath = ...,
        kinematics: CompactionKinematics = ...,
        cell_anchor: Point = ...,
        balance_opposing_faces: bool = ...,
        face_pressure_floor: float = ...,
        face_balance_strength: float = ...,
        initial_log_strain: float = ...,
        min_log_strain: float = ...,
        max_log_strain: float = ...,
        growth_factor: float = ...,
        shrink_factor: float = ...,
        relax_iterations: int = ...,
        max_shortening_over_min_diameter: float = ...,
        max_penetration: float = ...,
        max_curvature_ratio: float = ...,
        max_pressure: float = ...,
        max_penalty_energy: float = ...,
        max_steps: int = ...,
        max_relax_windows: int = ...,
        contact_energy_stiffness: float = ...,
        stretch_energy_stiffness: float = ...,
        bending_energy_stiffness: float = ...,
        target_tolerance: float = ...,
    ) -> None: ...
    @classmethod
    def volume_fraction(cls, target: float, **changes: Any) -> CompactionSettings: ...
    def copy(self) -> CompactionSettings: ...
    def replace(self, **changes: Any) -> CompactionSettings: ...

class RelaxationSettings:
    backend: Backend
    motion_model: MotionModel
    pin_fiber_ends: bool
    penetration_tolerance: float
    force_full_iterations: bool
    correction_fraction: float
    contact_aggregation: ContactAggregation
    stretch_stiffness: float
    bend_stiffness: float
    curvature_limit_stiffness: float
    curvature_limit_safety_margin: float
    curvature_ratio_tolerance: float
    constraint_iterations: int
    curvature_cleanup_sweeps: int
    twist_stiffness: float
    max_step: float
    max_iterations: int
    iterations_per_batch: int
    debug_snapshot_interval: int | None
    save_assembled_reference: bool
    cell_size_scale: float
    neighbor_skin_scale: float
    neighbor_capacity: int
    adaptive_segmentation: AdaptiveSegmentationSettings | None
    def __init__(
        self,
        *,
        backend: Backend = ...,
        motion_model: MotionModel = ...,
        pin_fiber_ends: bool = ...,
        penetration_tolerance: float = ...,
        force_full_iterations: bool = ...,
        correction_fraction: float = ...,
        contact_aggregation: ContactAggregation = ...,
        stretch_stiffness: float = ...,
        bend_stiffness: float = ...,
        curvature_limit_stiffness: float = ...,
        curvature_limit_safety_margin: float = ...,
        curvature_ratio_tolerance: float = ...,
        constraint_iterations: int = ...,
        curvature_cleanup_sweeps: int = ...,
        twist_stiffness: float = ...,
        max_step: float = ...,
        max_iterations: int = ...,
        iterations_per_batch: int = ...,
        debug_snapshot_interval: int | None = ...,
        save_assembled_reference: bool = ...,
        cell_size_scale: float = ...,
        neighbor_skin_scale: float = ...,
        neighbor_capacity: int = ...,
        adaptive_segmentation: AdaptiveSegmentationSettings | None = ...,
    ) -> None: ...
    def enable_adaptive_segmentation(self) -> None: ...
    def disable_adaptive_segmentation(self) -> None: ...
    def copy(self) -> RelaxationSettings: ...
    def replace(self, **changes: Any) -> RelaxationSettings: ...
    def to_dict(self) -> dict[str, Any]: ...

class RelaxationOverrides:
    motion_model: MotionModel | None
    correction_fraction: float | None
    contact_aggregation: ContactAggregation | None
    stretch_stiffness: float | None
    bend_stiffness: float | None
    curvature_limit_stiffness: float | None
    constraint_iterations: int | None
    curvature_cleanup_sweeps: int | None
    def __init__(
        self,
        *,
        motion_model: MotionModel | None = ...,
        correction_fraction: float | None = ...,
        contact_aggregation: ContactAggregation | None = ...,
        stretch_stiffness: float | None = ...,
        bend_stiffness: float | None = ...,
        curvature_limit_stiffness: float | None = ...,
        constraint_iterations: int | None = ...,
        curvature_cleanup_sweeps: int | None = ...,
    ) -> None: ...
    @classmethod
    def preset(cls, name: OverridePreset, **changes: Any) -> RelaxationOverrides: ...
    def copy(self) -> RelaxationOverrides: ...
    def replace(self, **changes: Any) -> RelaxationOverrides: ...

class SolvePolicy:
    name: str
    target_penetration: float
    target_curvature_ratio: float
    max_penetration: float
    max_curvature_ratio: float
    hard_penetration: bool
    hard_curvature: bool
    max_iterations: int
    max_extra_iterations: int
    on_budget_exhausted: BudgetExhaustion
    def __init__(
        self,
        name: str = ...,
        *,
        target_penetration: float = ...,
        target_curvature_ratio: float = ...,
        max_penetration: float | None = ...,
        max_curvature_ratio: float | None = ...,
        hard_penetration: bool = ...,
        hard_curvature: bool = ...,
        max_iterations: int = ...,
        max_extra_iterations: int | None = ...,
        on_budget_exhausted: BudgetExhaustion = ...,
    ) -> None: ...
    def copy(self) -> SolvePolicy: ...
    def replace(self, **changes: Any) -> SolvePolicy: ...

class JunctionPolicy:
    name: str
    law_name: str
    parameter_set: int
    max_surface_gap: float
    min_crossing_angle: float
    max_crossing_angle: float
    probability: float
    seed: int
    material_pairs: list[tuple[str, str]]
    max_per_fiber_pair: int
    min_anchor_separation: float
    candidate_capacity: int
    def __init__(
        self,
        name: str = ...,
        law_name: str = ...,
        *,
        parameter_set: int = ...,
        max_surface_gap: float = ...,
        min_crossing_angle: float = ...,
        max_crossing_angle: float = ...,
        probability: float = ...,
        seed: int = ...,
        material_pairs: Sequence[tuple[str, str]] = ...,
        max_per_fiber_pair: int = ...,
        min_anchor_separation: float = ...,
        candidate_capacity: int = ...,
    ) -> None: ...
    def copy(self) -> JunctionPolicy: ...
    def replace(self, **changes: Any) -> JunctionPolicy: ...

# --- Recipes -----------------------------------------------------------------

class FiberSelection:
    @property
    def name(self) -> str: ...
    @property
    def fiber_ids(self) -> list[int]: ...
    @property
    def formation_step(self) -> int: ...
    def __len__(self) -> int: ...

class CircularFootprint:
    def __init__(self, center: Sequence[float], *, diameter: float) -> None: ...
    @classmethod
    def random(cls, *, diameter: float, seed: int) -> CircularFootprint: ...
    @property
    def center(self) -> list[float] | None: ...
    @property
    def diameter(self) -> float: ...
    @property
    def seed(self) -> int | None: ...

class RandomFiberFraction:
    def __init__(self, fraction: float, *, seed: int = ...) -> None: ...
    @property
    def fraction(self) -> float: ...
    @property
    def seed(self) -> int: ...

NeedleFootprint = CircularFootprint | RandomFiberFraction

class HeldTargets:
    """Returned by operations that hold fibers on targets.

    Use it as a context manager to release the targets when the block ends,
    even if it raises, or ignore it and call the matching ``release_*`` method
    yourself. Releasing layer placement releases every held layer, including
    layers placed before the block.
    """

    def __enter__(self) -> Recipe: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc: BaseException | None = ...,
        traceback: TracebackType | None = ...,
    ) -> bool: ...

class Recipe:
    @property
    def stack_axis(self) -> int: ...
    def __init__(self, cell: Cell | Assembly, *, stack_axis: Axis | None = ...) -> None: ...
    def insert(
        self,
        collection: FiberCollection,
        *,
        name: str | None = ...,
        translation: Point = ...,
        rotation: Matrix3 | None = ...,
    ) -> FiberSelection: ...
    def relax_until_converged(self, *, max_iterations: int = ...) -> None: ...
    def relax_for(self, iterations: int) -> None: ...
    def settle_targets(self, *, tolerance: float, max_iterations: int) -> None: ...
    def solve(
        self,
        policy: SolvePolicy,
        overrides: RelaxationOverrides | None = ...,
    ) -> None: ...
    def set_min_bend_radius(
        self, material: Material | str, min_bend_radius: float
    ) -> None: ...
    def scale_layer_spacing(
        self,
        factor: float,
        *,
        stiffness: float = ...,
        max_translation: float = ...,
    ) -> HeldTargets: ...
    def place_layer_above(
        self,
        layer: int,
        *,
        gap: float,
        stiffness: float = ...,
        max_translation: float = ...,
    ) -> HeldTargets: ...
    def release_layer_placement(self) -> None: ...
    def needle_layer(
        self,
        layer: int,
        *,
        footprint: NeedleFootprint,
        depth: float,
        min_fiber_diameter: float | None = ...,
        stiffness: float = ...,
        max_translation: float = ...,
        max_translation_over_diameter: float = ...,
    ) -> HeldTargets: ...
    def release_needles(self) -> None: ...
    def fit_cell_to_active_fibers(
        self,
        *,
        axes: AxisSet | None = ...,
        padding: float = ...,
    ) -> None: ...
    def compact(
        self,
        settings: CompactionSettings,
        overrides: RelaxationOverrides | None = ...,
    ) -> None: ...
    def capture_junctions(self, policy: JunctionPolicy) -> None: ...
    def relax_and_capture(
        self, *, iterations: int, capture_every: int, policy: JunctionPolicy
    ) -> None: ...
    def operations(self) -> list[str]: ...
    def centerlines(self) -> list[list[list[float]]]: ...
    def run(
        self,
        settings: RelaxationSettings | None = ...,
        *,
        checkpoint: CheckpointSettings | None = ...,
        debug_ovito_path: Path | None = ...,
        debug_ovito_view_script_path: Path | None = ...,
        debug_ovito_session_path: Path | None = ...,
        debug_ovito_coloring: OvitoColoring = ...,
    ) -> RunResult: ...

class RunResult:
    @property
    def fiber_count(self) -> int: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def max_penetration(self) -> float: ...
    @property
    def max_curvature_ratio(self) -> float: ...
    @property
    def active_segments(self) -> int: ...
    @property
    def active_vertices(self) -> int: ...
    @property
    def segment_splits(self) -> int: ...
    @property
    def segment_merges(self) -> int: ...
    @property
    def refinement_passes(self) -> int: ...
    @property
    def coarsening_passes(self) -> int: ...
    @property
    def uploaded_bytes(self) -> int: ...
    @property
    def downloaded_bytes(self) -> int: ...
    @property
    def cell_count(self) -> int: ...
    @property
    def events(self) -> list[str]: ...
    @property
    def extra_iterations(self) -> int: ...
    @property
    def post_recipe_iterations(self) -> int: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def junction_captures(self) -> list[str]: ...
    @property
    def resumed(self) -> bool: ...
    @property
    def resumed_iteration(self) -> int | None: ...
    @property
    def checkpoint_saves(self) -> int: ...
    @property
    def last_checkpoint_iteration(self) -> int | None: ...
    @property
    def last_checkpoint_bytes(self) -> int | None: ...
    @property
    def debug_ovito_frames(self) -> int: ...
    @property
    def junction_count(self) -> int: ...
    @property
    def assembly(self) -> Assembly: ...
    def centerlines(self) -> list[list[list[float]]]: ...
    def characterize(self) -> AnalysisReport: ...
    def characterize_neighbors(
        self,
        contact_gap: float,
        *,
        neighbor_gap: float | None = ...,
        in_axis_angle_degrees: float = ...,
        sample_spacing: float | None = ...,
        max_lag: float | None = ...,
        lag_count: int = ...,
    ) -> NeighborReport: ...
    def characterize_shape(
        self,
        *,
        sample_spacing: float | None = ...,
        max_lag: float | None = ...,
        lag_count: int = ...,
        quantile_count: int = ...,
        orientation_axis: Point = ...,
        min_torsion_curvature: float | None = ...,
    ) -> ShapeReport: ...
    def write_ovito(
        self,
        path: Path,
        *,
        view_script_path: Path | None = ...,
        session_path: Path | None = ...,
        coloring: OvitoColoring = ...,
    ) -> None: ...
    def export_bpm(
        self,
        data_path: Path,
        *,
        mode: BpmExportMode = ...,
        sphere_spacing_over_radius: float = ...,
        density: float = ...,
        atom_type: int = ...,
        bond_type: int = ...,
    ) -> tuple[int, int]: ...
    def export_puma(
        self,
        output_directory: Path,
        voxel_size: float,
        *,
        include_fiber_ids: bool = ...,
        include_interface: bool = ...,
        ambiguity_tolerance: float | None = ...,
    ) -> PumaExportReport: ...

class ImageRelaxer:
    """Tangle relaxation with a CT image force, for fitting fibers to a scan.

    The world and the image are uploaded once. ``image`` is raw little-endian
    float32 bytes of a normalized volume (void ~0, fiber ~1) stored
    ``(z, y, x)``; voxel ``[k, j, i]`` is centered at
    ``origin + (i + 0.5, j + 0.5, k + 0.5) * voxel_size`` in meters.
    Adaptive segmentation and rigid motion are not supported, and fiber ends
    are never pinned.
    """

    def __init__(
        self,
        assembly: Assembly,
        settings: RelaxationSettings | None,
        image: bytes,
        shape_zyx: tuple[int, int, int],
        voxel_size: float,
        origin: tuple[float, float, float] = ...,
    ) -> None: ...
    def set_image_force(
        self,
        rate: float,
        reach_radii: float = ...,
        sigma_radii: float = ...,
        rings: int = ...,
        spokes: int = ...,
    ) -> None: ...
    def set_pinned(self, flags: list[list[bool]]) -> None: ...
    @property
    def pinned_count(self) -> int: ...
    @property
    def image_force_active(self) -> bool: ...
    @property
    def fiber_count(self) -> int: ...
    def run(self, iterations: int) -> dict[str, int | bool | float]: ...
    def centerlines(self) -> list[list[list[float]]]: ...
    def vertex_image_stats(self) -> list[list[tuple[float, float]]]: ...

# -- tangle.ct volume operations (used through tangle.ct._native) -------------

class CtHessian:
    """Gaussian-scale Hessian (times sigma squared) of a float32 (z, y, x) volume."""

    def __init__(self, image: Any, sigma: float) -> None: ...
    @property
    def sigma(self) -> float: ...
    def at(self, points: Any, out: Any) -> None: ...
    def directions(self, points: Any, axis: Any, tubularity: Any) -> None: ...

def ct_gaussian_filter(image: Any, out: Any, sigma: float, orders: tuple[int, int, int]) -> None: ...
def ct_sample(image: Any, points: Any, out: Any, fill: float) -> None: ...
def ct_foreground_depth(foreground: Any, edt: Any, peak: Any) -> None: ...
def ct_core_holes(mask: Any, out: Any, max_area: float) -> None: ...
def ct_rasterize(
    nodes: Any, counts: list[int], radii: list[float], reach: list[float], signed: bool,
    labels: Any, distance: Any, segment: Any,
) -> None: ...
def ct_paint(target: Any, line: Any, reach: float, value: int, only_empty: bool) -> None: ...
def ct_trace_one_way(
    image: Any, hessian: CtHessian, claimed: Any, radius: float, min_bend_radius: float, step: float,
    start: Any, direction: Any, max_steps: int, own_label: int,
) -> list[list[float]]: ...
def ct_trace_fibers(
    image: Any, hessian: CtHessian, claimed: Any, edt: Any, peak: Any, radius: float, min_bend_radius: float,
    step: float, min_length: float, node_spacing: float, label_offset: int, max_fibers: int | None,
    seed_depth_radii: float, bright_seed_strength: float | None = ..., peak_floor: float = ...,
    claim_radii: float = ...,
) -> list[list[list[float]]]: ...
def ct_owners(nodes: Any, counts: list[int], radii: list[float], voxels: Any, out: Any) -> None: ...
def ct_end_step(
    image: Any, nodes: Any, counts: list[int], radii: list[float], reach: list[float], step: float, max_moves: int,
) -> tuple[list[float], list[int]]: ...
def ct_cut_void(
    image: Any, nodes: Any, counts: list[int], radii: list[float], level: float, min_gap_radii: float,
    bridge_level: float, bridge_offset_radii: float, aligned_level: float, aligned_angle_degrees: float,
    directions: Any | None = None,
) -> tuple[tuple[list[float], list[int]], list[int], tuple[int, int, int]]: ...
def ct_resample(nodes: Any, counts: list[int], spacing: float) -> tuple[list[float], list[int]]: ...
def ct_support(image: Any, nodes: Any, counts: list[int]) -> list[float]: ...
def ct_curvature_ratio(nodes: Any, counts: list[int], min_bend_radius: float) -> list[float]: ...
def ct_render_occupancy(low: Sequence[int], high: Sequence[int], nodes: Any, counts: list[int], radii: list[float], edge: float, out: Any) -> None: ...
def ct_local_residual(image: Any, low: Sequence[int], high: Sequence[int], nodes: Any, counts: list[int], radii: list[float], edge: float, base: Any | None = None) -> float: ...
def ct_trim_duplicates(nodes: Any, counts: list[int], radii: list[float], min_length: float, closeness: float) -> tuple[tuple[list[float], list[int]], list[float]]: ...
def ct_remove_unsupported(image: Any, nodes: Any, counts: list[int], radii: list[float], min_length: float, min_support: float) -> tuple[tuple[list[float], list[int]], list[float]]: ...
def ct_merge_fragments(image: Any, nodes: Any, counts: list[int], radii: list[float], max_gap: float, max_angle_degrees: float, min_bridge_support: float, min_bend_radius: float | None, kink_threshold: float, end_cost: float, scale: float, max_prior_gap: float | None, max_prior_angle_degrees: float) -> tuple[tuple[list[float], list[int]], list[float], int]: ...
def ct_split_kinks(nodes: Any, counts: list[int], radii: list[float], min_bend_radius: float, min_length: float, max_length: float | None, threshold: float, min_angle_degrees: float, end_cost: float, scale: float, image: Any | None = None) -> tuple[tuple[list[float], list[int]], list[float], int]: ...
def ct_resolve_side_by_side(image: Any, nodes: Any, counts: list[int], radii: list[float], min_length: float, reach: float, end_cost: float, scale: float, max_angle_degrees: float) -> tuple[tuple[list[float], list[int]], list[float], int]: ...
def ct_render_grey(low: Sequence[int], high: Sequence[int], nodes: Any, counts: list[int], radii: list[float], profiles: list[list[float]], void: float, edge: float, out: Any) -> None: ...
def ct_node_confidence(image: Any, depth: Any, nodes: Any, counts: list[int], radii: list[float], spacing: float, margin: float, thickness_margin: float, ring: int, thickness_tolerance: float, previous_nodes: Any | None = None, previous_counts: list[int] | None = None) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[list[float]]]]: ...
def ct_overlap(low: Sequence[int], high: Sequence[int], nodes: Any, counts: list[int], radii: list[float], out: Any) -> None: ...
def ct_project(sample: Any, angles: list[float], out: Any) -> None: ...
def ct_back_project(projections: Any, angles: list[float], out: Any) -> None: ...
