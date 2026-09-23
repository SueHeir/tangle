from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, Literal

Point = Sequence[float]
Centerline = Sequence[Point]
Matrix3 = Sequence[Sequence[float]]
Path = str | PathLike[str]
BpmExportMode = Literal[
    "spheres-exact",
    "spheres-dynamic",
    "spherocylinders-exact",
    "spherocylinders-constant",
]

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
    def maximum_curvature(self) -> float: ...
    @property
    def maximum_curvature_ratio(self) -> float: ...
    @property
    def bend_limit_violations(self) -> int: ...
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

class Cell:
    def __init__(
        self,
        lengths: Point,
        periodic: Sequence[bool] = ...,
        origin: Point = ...,
    ) -> None: ...
    @property
    def lengths(self) -> list[float]: ...
    @property
    def periodic(self) -> list[bool]: ...
    @property
    def origin(self) -> list[float]: ...

class Material:
    def __init__(
        self,
        name: str,
        diameter: float,
        minimum_bend_radius: float | None = ...,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def diameter(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def minimum_bend_radius(self) -> float | None: ...

class Assembly:
    def __init__(self, cell: Cell) -> None: ...
    @property
    def fiber_count(self) -> int: ...
    @property
    def cell(self) -> Cell: ...
    def centerlines(self) -> list[list[list[float]]]: ...
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
        maximum_lag: float | None = ...,
        lag_count: int = ...,
    ) -> NeighborReport: ...
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
    def extend(self, other: FiberCollection) -> None: ...
    def layers(self) -> list[int]: ...
    def select_layer(
        self,
        layer: int,
        *,
        name: str | None = ...,
        allow_empty: bool = ...,
    ) -> FiberCollection: ...
    def __len__(self) -> int: ...

class FiberPopulationSettings:
    count: int
    segments_per_fiber: int
    seed: int
    length_minimum: float
    length_maximum: float
    nominal_parent_length: float | None
    radius_minimum: float
    radius_maximum: float
    curvature_amplitude_minimum: float
    curvature_amplitude_maximum: float
    orientation: str
    orientation_axis: list[float]
    maximum_angle: float
    maximum_tilt: float
    primary_fraction: float
    cross_fraction: float
    maximum_in_plane_deviation: float
    layer_orientation_seed: int
    position: str
    position_axis: int
    layers: int
    jitter_fraction: float
    density_exponent: float
    density_toward_high: bool
    minimum_bend_radius: float | None
    max_attempts_per_fiber: int
    material_name: str
    def __init__(self) -> None: ...
    def copy(self) -> FiberPopulationSettings: ...

def generate_point_crossing(
    cell: Cell,
    *,
    count: int = ...,
    length: float = ...,
    radius: float = ...,
    material_name: str = ...,
    name: str = ...,
) -> FiberCollection: ...

def generate_multisegment_crossing(
    cell: Cell,
    *,
    count: int = ...,
    segments_per_fiber: int = ...,
    length: float = ...,
    placed_chord_fraction: float = ...,
    radius: float = ...,
    rest_shape: str = ...,
    rest_amplitude: float = ...,
    placed_shape: str = ...,
    placed_amplitude: float = ...,
    minimum_bend_radius: float | None = ...,
    material_name: str = ...,
    name: str = ...,
) -> FiberCollection: ...

def generate_fiber_pair_crossing(
    cell: Cell,
    *,
    segments_per_fiber: int = ...,
    length: float = ...,
    radius: float = ...,
    axis_separation: float = ...,
    crossing_angle_degrees: float = ...,
    minimum_bend_radius: float | None = ...,
    material_name: str = ...,
    name: str = ...,
) -> FiberCollection: ...

def generate_fiber_population(
    cell: Cell,
    settings: FiberPopulationSettings,
    *,
    name: str = ...,
) -> FiberCollection: ...

class FiberSelection:
    @property
    def name(self) -> str: ...
    @property
    def fiber_ids(self) -> list[int]: ...
    @property
    def formation_step(self) -> int: ...
    def __len__(self) -> int: ...

class JunctionPolicy:
    name: str
    law_name: str
    parameter_set: int
    maximum_surface_gap: float
    minimum_crossing_angle: float
    maximum_crossing_angle: float
    probability: float
    seed: int
    material_pairs: list[tuple[str, str]]
    maximum_per_fiber_pair: int
    minimum_anchor_separation: float
    candidate_capacity: int
    def __init__(self, name: str = ..., law_name: str = ...) -> None: ...
    def copy(self) -> JunctionPolicy: ...

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
    ) -> None: ...
    def copy(self) -> CheckpointSettings: ...

class CellListSettings:
    cell_size_scale: float
    def __init__(self, cell_size_scale: float | None = ...) -> None: ...
    def copy(self) -> CellListSettings: ...
    def to_dict(self) -> dict[str, Any]: ...

class AdaptiveSegmentationSettings:
    contact_length_over_diameter: float
    minimum_length_over_diameter: float
    maximum_refinement_levels: int
    refinement_interval: int
    refinement_persistence: int
    coarsening_persistence: int
    coarsening_error_over_diameter: float
    coarsening_curvature_ratio: float
    def __init__(self) -> None: ...
    @classmethod
    def profile(cls, name: str) -> AdaptiveSegmentationSettings: ...
    def copy(self) -> AdaptiveSegmentationSettings: ...
    def to_dict(self) -> dict[str, Any]: ...

class CompactionSettings:
    target_type: str
    target_value: float
    target_values: list[float]
    path: str
    axis_weights: list[float]
    active_axes: list[bool]
    stress_ratio: list[float]
    pressure_floor: float
    kinematics: str
    cell_anchor: list[float]
    balance_opposing_faces: bool
    face_pressure_floor: float
    face_balance_strength: float
    initial_log_strain: float
    minimum_log_strain: float
    maximum_log_strain: float
    growth_factor: float
    shrink_factor: float
    relax_iterations: int
    maximum_shortening_over_minimum_diameter: float
    maximum_penetration: float
    maximum_bend_ratio: float
    maximum_pressure: float
    maximum_penalty_energy: float
    maximum_steps: int
    maximum_relax_windows: int
    contact_energy_stiffness: float
    stretch_energy_stiffness: float
    bending_energy_stiffness: float
    target_tolerance: float
    def __init__(
        self,
        target_volume_fraction: float = ...,
        axis_weights: Sequence[float] = ...,
    ) -> None: ...
    @classmethod
    def volume_fraction(
        cls, target: float, *, axis_weights: Sequence[float] = ...
    ) -> CompactionSettings: ...
    def copy(self) -> CompactionSettings: ...

class RelaxationSettings:
    backend: str
    motion_model: str
    pin_fiber_ends: bool
    penetration_tolerance: float
    force_full_iterations: bool
    correction_fraction: float
    contact_aggregation: str
    stretch_stiffness: float
    bend_stiffness: float
    curvature_limit_stiffness: float
    curvature_limit_safety_margin: float
    curvature_ratio_tolerance: float
    constraint_iterations: int
    curvature_cleanup_sweeps: int
    max_step: float
    max_iterations: int
    iterations_per_batch: int
    debug_snapshot_interval: int | None
    save_assembled_reference: bool
    cell_list: CellListSettings
    cell_size_scale: float
    adaptive_segmentation: AdaptiveSegmentationSettings | None
    def __init__(self) -> None: ...
    def enable_adaptive_segmentation(self) -> None: ...
    def disable_adaptive_segmentation(self) -> None: ...
    def copy(self) -> RelaxationSettings: ...
    def to_dict(self) -> dict[str, Any]: ...

class RelaxationOverrides:
    motion_model: str | None
    correction_fraction: float | None
    contact_aggregation: str | None
    stretch_stiffness: float | None
    bend_stiffness: float | None
    curvature_limit_stiffness: float | None
    constraint_iterations: int | None
    curvature_cleanup_sweeps: int | None
    def __init__(self) -> None: ...
    def copy(self) -> RelaxationOverrides: ...

class SolvePolicy:
    name: str
    solver_penetration: float
    solver_curvature_ratio: float
    acceptance_penetration: float
    penetration_enforcement: str
    acceptance_curvature_ratio: float
    curvature_enforcement: str
    maximum_iterations: int
    on_exhaustion: str
    def __init__(
        self,
        name: str = ...,
        *,
        solver_penetration: float = ...,
        solver_curvature_ratio: float = ...,
        acceptance_penetration: float = ...,
        penetration_enforcement: str = ...,
        acceptance_curvature_ratio: float = ...,
        curvature_enforcement: str = ...,
        maximum_iterations: int = ...,
        on_exhaustion: str = ...,
    ) -> None: ...
    def copy(self) -> SolvePolicy: ...

class Recipe:
    layer_axis: int
    def __init__(self, cell: Cell | Assembly, *, layer_axis: int = ...) -> None: ...
    def insert(
        self,
        collection: FiberCollection,
        *,
        name: str | None = ...,
        translation: Point = ...,
        rotation: Matrix3 | None = ...,
    ) -> FiberSelection: ...
    def relax(self, *, maximum_iterations: int = ...) -> None: ...
    def relax_for(self, iterations: int) -> None: ...
    def relax_with_policy(
        self,
        policy: SolvePolicy,
        overrides: RelaxationOverrides | None = ...,
    ) -> None: ...
    def set_material_bend_radius(
        self, material_name: str, minimum_bend_radius: float
    ) -> None: ...
    def relax_until_targets_reached(
        self, tolerance: float, maximum_iterations: int
    ) -> None: ...
    def move_layers(
        self,
        spacing_scale: float,
        *,
        stiffness: float = ...,
        max_translation: float = ...,
    ) -> None: ...
    def place_layer_above(
        self,
        layer: int,
        gap: float,
        *,
        stiffness: float = ...,
        max_translation: float = ...,
    ) -> None: ...
    def release_layer_targets(self) -> None: ...
    def needle_layer_circular(
        self,
        layer: int,
        center: Sequence[float],
        diameter: float,
        depth: float,
        *,
        minimum_fiber_diameter: float | None = ...,
        stiffness: float = ...,
        max_translation: float = ...,
        maximum_translation_over_fiber_diameter: float = ...,
    ) -> None: ...
    def needle_layer_random(
        self,
        layer: int,
        fraction: float,
        depth: float,
        *,
        seed: int = ...,
        minimum_fiber_diameter: float | None = ...,
        stiffness: float = ...,
        max_translation: float = ...,
        maximum_translation_over_fiber_diameter: float = ...,
    ) -> None: ...
    def release_needles(self) -> None: ...
    def fit_cell_to_active_fibers(
        self,
        *,
        axes: Sequence[bool] = ...,
        padding: float = ...,
    ) -> None: ...
    def compact(
        self,
        settings: CompactionSettings,
        overrides: RelaxationOverrides | None = ...,
    ) -> None: ...
    def capture_junctions(self, policy: JunctionPolicy) -> None: ...
    def relax_and_capture(
        self, iterations: int, every: int, policy: JunctionPolicy
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
        debug_ovito_coloring: str = ...,
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
    def centerlines(self) -> list[list[list[float]]]: ...
    def characterize(self) -> AnalysisReport: ...
    def characterize_neighbors(
        self,
        contact_gap: float,
        *,
        neighbor_gap: float | None = ...,
        in_axis_angle_degrees: float = ...,
        sample_spacing: float | None = ...,
        maximum_lag: float | None = ...,
        lag_count: int = ...,
    ) -> NeighborReport: ...
    def write_ovito(
        self,
        path: Path,
        *,
        view_script_path: Path | None = ...,
        session_path: Path | None = ...,
        coloring: str = ...,
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
