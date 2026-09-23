use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::FiberAssembly;

use crate::{
    AdaptiveSegmentationConfig, BatchStatus, CellListConfig, CompactionEnergyModel,
    CompactionKinematics, CompactionMetrics, ContactCapture, DeviceFiberWorld,
    DeviceWorldCheckpoint, FormationTargetError, ImageForceSettings, PackedAssembly, PackingError,
    RelaxationSnapshot,
};

/// CubeCL runtime selected for relaxation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxationBackend {
    /// CubeCL's WGPU runtime (Metal on supported Apple systems).
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CubeCL's native multithreaded CPU runtime.
    #[cfg(feature = "cpu")]
    Cpu,
    /// CubeCL's CUDA runtime.
    #[cfg(feature = "cuda")]
    Cuda,
    /// CubeCL's HIP runtime.
    #[cfg(feature = "hip")]
    Hip,
}

/// Kinematic model used when device contact corrections are applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FiberMotion {
    /// Move centerline vertices independently, followed by internal stretch
    /// and bending constraints.
    #[default]
    Flexible,
    /// Apply one translation to every vertex of a fiber, preserving its exact
    /// centerline shape and orientation.
    RigidTranslation,
}

/// How simultaneous capsule contacts on one segment are combined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ContactAggregation {
    /// Average all active contact projections for stable dense relaxation.
    #[default]
    UniformAverage = 0,
    /// Weight each projection by its penetration before averaging.
    PenetrationWeighted = 1,
    /// Apply only the deepest contact projection on each segment.
    DeepestOnly = 2,
}

#[allow(unreachable_code)]
impl Default for RelaxationBackend {
    fn default() -> Self {
        #[cfg(feature = "wgpu")]
        return Self::Wgpu;
        #[cfg(all(not(feature = "wgpu"), feature = "cpu"))]
        return Self::Cpu;
        #[cfg(all(not(feature = "wgpu"), not(feature = "cpu"), feature = "cuda"))]
        return Self::Cuda;
        #[cfg(all(
            not(feature = "wgpu"),
            not(feature = "cpu"),
            not(feature = "cuda"),
            feature = "hip"
        ))]
        return Self::Hip;
        panic!("tangle_relax requires at least one runtime feature");
    }
}

/// Parameters for flexible centerline relaxation on a CubeCL device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelaxationConfig {
    /// Runtime used to compile and execute kernels.
    pub backend: RelaxationBackend,
    /// Fiber kinematics used by the contact projection kernel.
    pub motion_model: FiberMotion,
    /// Optional device-resident midpoint subdivision policy.
    pub adaptive_segmentation: Option<AdaptiveSegmentationConfig>,
    /// Keep the first and last material point of every fiber fixed.
    pub pin_fiber_ends: bool,
    /// Maximum capsule penetration considered converged.
    pub penetration_tolerance: f32,
    /// Execute the complete requested device batch even if physical
    /// tolerances are reached. Formation controllers use this for prescribed
    /// relaxation intervals without altering physical tolerances.
    pub force_full_iterations: bool,
    /// Fraction of pair overlap projected per iteration.
    pub correction_fraction: f32,
    /// Rule used to combine simultaneous contacts on one segment.
    pub contact_aggregation: ContactAggregation,
    /// Fraction of intrinsic segment-length error projected per pass.
    pub stretch_stiffness: f32,
    /// Fraction of intrinsic two-edge chord error projected per pass.
    pub bend_stiffness: f32,
    /// Fraction of excess admissible curvature projected per pass.
    pub curvature_limit_stiffness: f32,
    /// Fractional margin used to project just inside the physical curvature
    /// limit, preventing other constraints from rebounding above one.
    pub curvature_limit_safety_margin: f32,
    /// Allowed dimensionless excess above a bend ratio of one.
    pub curvature_ratio_tolerance: f32,
    /// Number of rest-length/rest-shape passes per contact iteration.
    pub constraint_iterations: usize,
    /// Forward/backward in-place curvature sweeps performed after each contact
    /// correction before the next hard convergence audit.
    pub curvature_cleanup_sweeps: usize,
    /// Maximum contact displacement of one vertex in an iteration.
    pub max_step: f32,
    /// Maximum device-side correction iterations.
    pub max_iterations: usize,
    /// Maximum device iterations dispatched per GRASS-controlled batch.
    pub iterations_per_batch: usize,
    /// Device-side uniform-grid broad-phase parameters.
    pub cell_list: CellListConfig,
    /// Download geometry at interval checkpoints when debugging.
    pub debug_snapshot_interval: Option<usize>,
    /// Save the converged geometry as assembled reference geometry.
    pub save_assembled_reference: bool,
    /// Workflow stage requested after convergence.
    pub next_stage: TangleStage,
}

impl Default for RelaxationConfig {
    fn default() -> Self {
        Self {
            backend: RelaxationBackend::default(),
            motion_model: FiberMotion::default(),
            adaptive_segmentation: None,
            pin_fiber_ends: false,
            penetration_tolerance: 1.0e-4,
            force_full_iterations: false,
            correction_fraction: 0.8,
            contact_aggregation: ContactAggregation::default(),
            stretch_stiffness: 0.35,
            bend_stiffness: 0.03,
            curvature_limit_stiffness: 1.0,
            curvature_limit_safety_margin: 1.0e-3,
            curvature_ratio_tolerance: 1.0e-5,
            constraint_iterations: 2,
            curvature_cleanup_sweeps: 4,
            max_step: 0.006,
            max_iterations: 2_000,
            iterations_per_batch: 128,
            cell_list: CellListConfig::default(),
            debug_snapshot_interval: None,
            save_assembled_reference: true,
            next_stage: TangleStage::Export,
        }
    }
}

impl RelaxationConfig {
    /// Starts a configuration for independently moving centerline vertices.
    pub fn flexible() -> Self {
        Self::default()
    }

    /// Starts a configuration that translates each fiber without deforming it.
    pub fn rigid_translation() -> Self {
        Self {
            motion_model: FiberMotion::RigidTranslation,
            ..Self::default()
        }
    }

    /// Sets the contact penetration required for convergence.
    pub fn with_penetration_tolerance(mut self, tolerance: f32) -> Self {
        self.penetration_tolerance = tolerance;
        self
    }

    /// Sets the fraction of pair overlap projected per iteration.
    pub fn with_contact_correction(mut self, fraction: f32) -> Self {
        self.correction_fraction = fraction;
        self
    }

    /// Sets flexible-fiber stretch and rest-shape bending stiffnesses.
    pub fn with_flexible_stiffness(mut self, stretch: f32, bending: f32) -> Self {
        self.stretch_stiffness = stretch;
        self.bend_stiffness = bending;
        self
    }

    /// Sets admissible-curvature projection stiffness and bend-ratio tolerance.
    pub fn with_curvature_limit(mut self, stiffness: f32, ratio_tolerance: f32) -> Self {
        self.curvature_limit_stiffness = stiffness;
        self.curvature_ratio_tolerance = ratio_tolerance;
        self
    }

    /// Sets the largest contact displacement applied to one vertex per iteration.
    pub fn with_max_step(mut self, max_step: f32) -> Self {
        self.max_step = max_step;
        self
    }

    /// Enables device-resident midpoint subdivision near unresolved contacts.
    pub fn with_adaptive_segmentation(mut self, config: AdaptiveSegmentationConfig) -> Self {
        self.adaptive_segmentation = Some(config);
        self
    }

    /// Pins or releases the two material endpoints of every fiber.
    pub fn with_pinned_ends(mut self, pinned: bool) -> Self {
        self.pin_fiber_ends = pinned;
        self
    }

    /// Sets total and per-GRASS-batch iteration limits.
    pub fn with_iteration_limits(mut self, total: usize, per_batch: usize) -> Self {
        self.max_iterations = total;
        self.iterations_per_batch = per_batch;
        self
    }

    /// Sets the device-side cell-list broad-phase configuration.
    pub fn with_cell_list(mut self, cell_list: CellListConfig) -> Self {
        self.cell_list = cell_list;
        self
    }

    /// Requests host-readable geometry snapshots at `interval` checkpoints.
    ///
    /// Passing `None` keeps all intermediate geometry resident on the device.
    pub fn with_debug_snapshots(mut self, interval: Option<usize>) -> Self {
        self.debug_snapshot_interval = interval;
        self
    }

    /// Sets the workflow stage entered after successful convergence.
    pub fn with_next_stage(mut self, stage: TangleStage) -> Self {
        self.next_stage = stage;
        self
    }
}

/// Result of the most recent CubeCL relaxation.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RelaxationState {
    /// Number of corrections applied on-device.
    pub iterations: usize,
    /// Number of GRASS-controlled device batches completed.
    pub batches: usize,
    /// Final maximum penetration.
    pub max_penetration: f32,
    /// Largest vertex contact displacement in the last active iteration.
    pub max_displacement: f32,
    /// Whether penetration and curvature tolerances were reached.
    pub converged: bool,
    /// Whether a broad-phase cell exceeded its configured segment capacity.
    pub cell_list_overflow: bool,
    /// Largest current curvature divided by its admissible maximum.
    pub max_curvature_ratio: f32,
    /// Bytes uploaded for immutable topology and initial positions.
    pub uploaded_bytes: usize,
    /// Bytes downloaded for final positions and scalar status.
    pub downloaded_bytes: usize,
    /// Explicitly requested intermediate device geometry readbacks.
    pub snapshots: Vec<RelaxationSnapshot>,
    /// Most recent iteration captured, retained after trajectory consumers
    /// release the corresponding host-side geometry.
    #[serde(default)]
    pub last_snapshot_iteration: Option<usize>,
    /// Width of one broad-phase cell.
    pub cell_size: f32,
    /// Number of broad-phase cells.
    pub cell_count: usize,
    /// Number of device refinement passes that activated at least one midpoint.
    pub refinement_passes: usize,
    /// Total parent segments split on the selected device.
    pub segment_splits: usize,
    /// Number of sibling-pair merges performed on the selected device.
    pub segment_merges: usize,
    /// Number of device adaptation passes that removed at least one midpoint.
    pub coarsening_passes: usize,
    /// Current active segment count.
    pub active_segments: usize,
    /// Current active vertex count.
    pub active_vertices: usize,
}

/// Backend-specific persistent device world stored as a GRASS resource.
pub enum DeviceWorld {
    /// WGPU-resident fiber buffers.
    #[cfg(feature = "wgpu")]
    Wgpu(DeviceFiberWorld<cubecl::wgpu::WgpuRuntime>),
    /// CPU-resident fiber buffers executed by CubeCL's native CPU runtime.
    #[cfg(feature = "cpu")]
    Cpu(DeviceFiberWorld<cubecl::cpu::CpuRuntime>),
    /// CUDA-resident fiber buffers.
    #[cfg(feature = "cuda")]
    Cuda(DeviceFiberWorld<cubecl::cuda::CudaRuntime>),
    /// HIP-resident fiber buffers.
    #[cfg(feature = "hip")]
    Hip(DeviceFiberWorld<cubecl::hip::HipRuntime>),
}

impl DeviceWorld {
    fn upload(config: &RelaxationConfig, packed: PackedAssembly) -> Self {
        match config.backend {
            #[cfg(feature = "wgpu")]
            RelaxationBackend::Wgpu => Self::Wgpu(DeviceFiberWorld::upload(
                &cubecl::wgpu::WgpuDevice::default(),
                packed,
                config.cell_list,
                config.max_step,
            )),
            #[cfg(feature = "cpu")]
            RelaxationBackend::Cpu => Self::Cpu(DeviceFiberWorld::upload(
                &cubecl::cpu::CpuDevice::default(),
                packed,
                config.cell_list,
                config.max_step,
            )),
            #[cfg(feature = "cuda")]
            RelaxationBackend::Cuda => Self::Cuda(DeviceFiberWorld::upload(
                &cubecl::cuda::CudaDevice::default(),
                packed,
                config.cell_list,
                config.max_step,
            )),
            #[cfg(feature = "hip")]
            RelaxationBackend::Hip => Self::Hip(DeviceFiberWorld::upload(
                &cubecl::hip::AmdDevice::default(),
                packed,
                config.cell_list,
                config.max_step,
            )),
        }
    }

    /// Recreates the configured backend world from a restart checkpoint.
    pub fn restore(config: &RelaxationConfig, checkpoint: DeviceWorldCheckpoint) -> Self {
        match config.backend {
            #[cfg(feature = "wgpu")]
            RelaxationBackend::Wgpu => Self::Wgpu(DeviceFiberWorld::restore(
                &cubecl::wgpu::WgpuDevice::default(),
                checkpoint,
                config.cell_list,
                config.max_step,
            )),
            #[cfg(feature = "cpu")]
            RelaxationBackend::Cpu => Self::Cpu(DeviceFiberWorld::restore(
                &cubecl::cpu::CpuDevice::default(),
                checkpoint,
                config.cell_list,
                config.max_step,
            )),
            #[cfg(feature = "cuda")]
            RelaxationBackend::Cuda => Self::Cuda(DeviceFiberWorld::restore(
                &cubecl::cuda::CudaDevice::default(),
                checkpoint,
                config.cell_list,
                config.max_step,
            )),
            #[cfg(feature = "hip")]
            RelaxationBackend::Hip => Self::Hip(DeviceFiberWorld::restore(
                &cubecl::hip::AmdDevice::default(),
                checkpoint,
                config.cell_list,
                config.max_step,
            )),
        }
    }

    /// Downloads the persistent device state required for restart.
    pub fn checkpoint(&self) -> DeviceWorldCheckpoint {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.checkpoint(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.checkpoint(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.checkpoint(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.checkpoint(),
        }
    }

    /// Dispatches one bounded relaxation batch on the selected backend.
    pub fn run_batch(&mut self, config: &RelaxationConfig, iterations: usize) -> BatchStatus {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.run_batch(config, iterations),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.run_batch(config, iterations),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.run_batch(config, iterations),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.run_batch(config, iterations),
        }
    }

    /// Uploads a packed assembly to the configured backend outside a GRASS
    /// app, for callers that drive `run_batch` themselves.
    pub fn new(config: &RelaxationConfig, packed: PackedAssembly) -> Self {
        Self::upload(config, packed)
    }

    /// Uploads a normalized CT volume for the image force (initially off).
    /// See [`DeviceFiberWorld::set_image`].
    pub fn set_image(
        &mut self,
        image: &[f32],
        shape_zyx: [usize; 3],
        voxel_size: f32,
        origin: [f32; 3],
    ) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.set_image(image, shape_zyx, voxel_size, origin),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.set_image(image, shape_zyx, voxel_size, origin),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.set_image(image, shape_zyx, voxel_size, origin),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.set_image(image, shape_zyx, voxel_size, origin),
        }
    }

    /// Sets the image-force parameters; a rate of zero disables the force.
    pub fn set_image_force(&mut self, settings: ImageForceSettings) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.set_image_force(settings),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.set_image_force(settings),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.set_image_force(settings),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.set_image_force(settings),
        }
    }

    /// Whether an image is resident and its force is enabled.
    pub fn image_force_active(&self) -> bool {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.image_force_active(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.image_force_active(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.image_force_active(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.image_force_active(),
        }
    }

    /// Per-vertex `(owned mass in voxel², support)` at the current positions.
    pub fn image_vertex_stats(&self) -> Vec<f32> {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.image_vertex_stats(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.image_vertex_stats(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.image_vertex_stats(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.image_vertex_stats(),
        }
    }

    /// Downloads current interleaved xyz positions.
    pub fn download_positions(&self) -> Vec<f32> {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.download_positions(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.download_positions(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.download_positions(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.download_positions(),
        }
    }

    /// Downloads the active-vertex topology mask at an explicit workflow checkpoint.
    pub fn download_vertex_active(&self) -> Vec<u32> {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.download_vertex_active(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.download_vertex_active(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.download_vertex_active(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.download_vertex_active(),
        }
    }

    /// Downloads the active-segment topology mask at an explicit workflow checkpoint.
    pub fn download_segment_active(&self) -> Vec<u32> {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.download_segment_active(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.download_segment_active(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.download_segment_active(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.download_segment_active(),
        }
    }

    /// Activates reserved dyadic paths needed by selected formation vertices.
    pub fn activate_refinement_vertices(
        &mut self,
        vertices: &[u32],
        refinement_interval: usize,
    ) -> usize {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.activate_refinement_vertices(vertices, refinement_interval),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.activate_refinement_vertices(vertices, refinement_interval),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.activate_refinement_vertices(vertices, refinement_interval),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.activate_refinement_vertices(vertices, refinement_interval),
        }
    }

    /// Installs an already downloaded position buffer into a host assembly.
    pub fn unpack_positions(
        &self,
        positions: &[f32],
        assembly: &mut FiberAssembly,
    ) -> Result<(), PackingError> {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.unpack_positions(positions, assembly),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.unpack_positions(positions, assembly),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.unpack_positions(positions, assembly),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.unpack_positions(positions, assembly),
        }
    }

    /// Width of one device broad-phase cell.
    pub fn cell_size(&self) -> f32 {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.cell_size(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.cell_size(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.cell_size(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.cell_size(),
        }
    }

    /// Number of device broad-phase cells.
    pub fn cell_count(&self) -> usize {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.cell_count(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.cell_count(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.cell_count(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.cell_count(),
        }
    }

    /// Immutable packed topology backing the selected runtime world.
    pub fn packed(&self) -> &PackedAssembly {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.packed(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.packed(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.packed(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.packed(),
        }
    }

    /// Translates labeled fiber centers toward manufacturing-layer targets.
    pub fn apply_layer_targets(
        &mut self,
        axis: usize,
        targets: &[f32],
        stiffness: f32,
        max_translation: f32,
    ) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => {
                world.apply_layer_targets(axis, targets, stiffness, max_translation)
            }
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => {
                world.apply_layer_targets(axis, targets, stiffness, max_translation)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => {
                world.apply_layer_targets(axis, targets, stiffness, max_translation)
            }
            #[cfg(feature = "hip")]
            Self::Hip(world) => {
                world.apply_layer_targets(axis, targets, stiffness, max_translation)
            }
        }
    }

    /// Translates only one labeled layer, leaving the relaxed stack untethered.
    pub fn apply_single_layer_target(
        &mut self,
        axis: usize,
        targets: &[f32],
        layer: u32,
        stiffness: f32,
        max_translation: f32,
    ) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => {
                world.apply_single_layer_target(axis, targets, layer, stiffness, max_translation)
            }
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => {
                world.apply_single_layer_target(axis, targets, layer, stiffness, max_translation)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => {
                world.apply_single_layer_target(axis, targets, layer, stiffness, max_translation)
            }
            #[cfg(feature = "hip")]
            Self::Hip(world) => {
                world.apply_single_layer_target(axis, targets, layer, stiffness, max_translation)
            }
        }
    }

    /// Releases the persistent manufacturing-layer target.
    pub fn clear_layer_targets(&mut self) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.clear_layer_targets(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.clear_layer_targets(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.clear_layer_targets(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.clear_layer_targets(),
        }
    }

    /// Pulls selected vertices toward device-captured displaced coordinates.
    pub fn apply_vertex_displacement_targets(
        &mut self,
        axis: usize,
        vertex_indices: &[u32],
        displacement: f32,
        stiffness: f32,
        max_translation: f32,
    ) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.apply_vertex_displacement_targets(
                axis,
                vertex_indices,
                displacement,
                stiffness,
                max_translation,
            ),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.apply_vertex_displacement_targets(
                axis,
                vertex_indices,
                displacement,
                stiffness,
                max_translation,
            ),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.apply_vertex_displacement_targets(
                axis,
                vertex_indices,
                displacement,
                stiffness,
                max_translation,
            ),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.apply_vertex_displacement_targets(
                axis,
                vertex_indices,
                displacement,
                stiffness,
                max_translation,
            ),
        }
    }

    /// Releases persistent per-vertex formation targets.
    pub fn clear_vertex_targets(&mut self) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.clear_vertex_targets(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.clear_vertex_targets(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.clear_vertex_targets(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.clear_vertex_targets(),
        }
    }

    /// Returns the maximum residual for each active persistent formation target.
    pub fn formation_target_error(&self) -> FormationTargetError {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.formation_target_error(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.formation_target_error(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.formation_target_error(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.formation_target_error(),
        }
    }

    /// Activates all prepacked fibers assigned through a formation step.
    pub fn activate_formation_step(&mut self, maximum_step: u32) -> (usize, usize) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.activate_formation_step(maximum_step),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.activate_formation_step(maximum_step),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.activate_formation_step(maximum_step),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.activate_formation_step(maximum_step),
        }
    }

    /// Replaces the maximum admissible curvature of every resident fiber.
    pub fn set_fiber_maximum_curvatures(&mut self, maximum_curvatures: &[f32]) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.set_fiber_maximum_curvatures(maximum_curvatures),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.set_fiber_maximum_curvatures(maximum_curvatures),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.set_fiber_maximum_curvatures(maximum_curvatures),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.set_fiber_maximum_curvatures(maximum_curvatures),
        }
    }

    /// Captures current inter-fiber contacts without downloading geometry.
    pub fn capture_contacts(
        &mut self,
        maximum_surface_gap: f32,
        capacity: usize,
    ) -> ContactCapture {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.capture_contacts(maximum_surface_gap, capacity),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.capture_contacts(maximum_surface_gap, capacity),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.capture_contacts(maximum_surface_gap, capacity),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.capture_contacts(maximum_surface_gap, capacity),
        }
    }

    /// Changes the resident orthorhombic cell without downloading geometry.
    pub fn compact_cell(
        &mut self,
        new_lower: [f32; 3],
        new_upper: [f32; 3],
        kinematics: CompactionKinematics,
    ) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.compact_cell(new_lower, new_upper, kinematics),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.compact_cell(new_lower, new_upper, kinematics),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.compact_cell(new_lower, new_upper, kinematics),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.compact_cell(new_lower, new_upper, kinematics),
        }
    }

    /// Returns directional pressure and formation penalty-energy measures.
    pub fn compaction_metrics(
        &self,
        correction_fraction: f32,
        model: CompactionEnergyModel,
    ) -> CompactionMetrics {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.compaction_metrics(correction_fraction, model),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.compaction_metrics(correction_fraction, model),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.compaction_metrics(correction_fraction, model),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.compaction_metrics(correction_fraction, model),
        }
    }

    /// Lower/upper pressure on each axis face from the latest compaction
    /// metric reduction. Each row is `[lower, upper]`.
    pub fn wall_face_pressures(&self) -> [[f32; 2]; 3] {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.wall_face_pressures(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.wall_face_pressures(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.wall_face_pressures(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.wall_face_pressures(),
        }
    }

    /// Current resident orthorhombic cell bounds.
    pub fn cell_bounds(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(world) => world.cell_bounds(),
            #[cfg(feature = "cpu")]
            Self::Cpu(world) => world.cell_bounds(),
            #[cfg(feature = "cuda")]
            Self::Cuda(world) => world.cell_bounds(),
            #[cfg(feature = "hip")]
            Self::Hip(world) => world.cell_bounds(),
        }
    }
}

/// Persistent device-world resource installed by [`RelaxationPlugin`].
#[derive(Default)]
pub struct DeviceState {
    /// Device world after the generated assembly has been uploaded.
    pub world: Option<DeviceWorld>,
}

/// GRASS-level ownership control for multi-stage device formation protocols.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowControl {
    /// Keep the workflow in `Relax` after a converged batch so another plugin
    /// can change device constraints and request another batch.
    pub hold_relax_stage: bool,
    /// Optional controller-imposed cap for the next GPU batch.
    pub batch_iteration_limit: Option<usize>,
    /// Disable early convergence for a prescribed formation interval.
    pub force_full_batch: bool,
    /// Optional recipe-specific penetration target for the next GPU batch.
    pub penetration_tolerance: Option<f32>,
    /// Optional recipe-specific absolute maximum curvature ratio.
    pub maximum_curvature_ratio: Option<f32>,
}

/// Per-operation projection settings layered over [`RelaxationConfig`].
///
/// Formation recipes use these transient overrides to emphasize one family
/// of constraints without rebuilding or re-uploading the device world.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RelaxationOverrides {
    /// Contact kinematics used for this operation. Rigid translation is useful
    /// after curvature projection because it removes contacts without changing
    /// the corrected fiber shape.
    pub motion_model: Option<FiberMotion>,
    /// Fraction of pair overlap projected per iteration.
    pub correction_fraction: Option<f32>,
    /// Rule used to combine simultaneous contacts on one segment.
    pub contact_aggregation: Option<ContactAggregation>,
    /// Fraction of intrinsic segment-length error projected per pass.
    pub stretch_stiffness: Option<f32>,
    /// Fraction of intrinsic rest-shape bending error projected per pass.
    pub bend_stiffness: Option<f32>,
    /// Fraction of excess admissible curvature projected per pass.
    pub curvature_limit_stiffness: Option<f32>,
    /// Number of rest-length/rest-shape passes per contact iteration.
    pub constraint_iterations: Option<usize>,
    /// In-place hard-curvature sweeps performed after contact correction.
    pub curvature_cleanup_sweeps: Option<usize>,
}

/// GRASS plugin that owns contact detection and flexible relaxation on a
/// CubeCL device.
pub struct RelaxationPlugin {
    /// Solver and runtime configuration.
    pub config: RelaxationConfig,
}

impl Plugin for RelaxationPlugin {
    fn build(&self, app: &mut App) {
        assert!(self.config.penetration_tolerance >= 0.0);
        assert!(self.config.correction_fraction > 0.0 && self.config.correction_fraction <= 1.0);
        assert!((0.0..=1.0).contains(&self.config.stretch_stiffness));
        assert!((0.0..=1.0).contains(&self.config.bend_stiffness));
        assert!((0.0..=1.0).contains(&self.config.curvature_limit_stiffness));
        assert!((0.0..0.1).contains(&self.config.curvature_limit_safety_margin));
        assert!(self.config.curvature_ratio_tolerance >= 0.0);
        assert!(self.config.constraint_iterations > 0);
        assert!(self.config.curvature_cleanup_sweeps > 0);
        assert!(self.config.max_step > 0.0);
        assert!(self.config.max_iterations > 0);
        assert!(self.config.iterations_per_batch > 0);
        assert!(self.config.cell_list.cell_size_scale >= 1.0);
        if let Some(adaptive) = self.config.adaptive_segmentation {
            assert!(adaptive.contact_length_over_diameter > 0.0);
            assert!(adaptive.minimum_length_over_diameter > 0.0);
            assert!(adaptive.minimum_length_over_diameter <= adaptive.contact_length_over_diameter);
            assert!(adaptive.maximum_refinement_levels > 0);
            assert!(adaptive.maximum_refinement_levels < 31);
            assert!(adaptive.refinement_interval > 0);
            assert!(adaptive.refinement_persistence > 0);
            assert!(adaptive.coarsening_error_over_diameter >= 0.0);
            assert!(adaptive.coarsening_error_over_diameter.is_finite());
            assert!(adaptive.coarsening_curvature_ratio >= 0.0);
            assert!(adaptive.coarsening_curvature_ratio.is_finite());
            assert_eq!(self.config.motion_model, FiberMotion::Flexible);
        }
        assert!(self
            .config
            .debug_snapshot_interval
            .is_none_or(|value| value > 0));

        app.add_resource(self.config)
            .add_resource(RelaxationState::default())
            .add_resource(DeviceState::default())
            .add_resource(WorkflowControl::default())
            .add_resource(RelaxationOverrides::default())
            .add_update_system(
                upload_device_world.run_if(in_state(TangleStage::Relax)),
                TanglePhase::PrepareGeometry,
            )
            .add_update_system(
                run_device_batch.run_if(in_state(TangleStage::Relax)),
                TanglePhase::ApplyConstraints,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_CONTACTS.clone(), TANGLE_RELAXATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

fn upload_device_world(
    config: Res<RelaxationConfig>,
    assembly: Res<FiberAssembly>,
    mut device: ResMut<DeviceState>,
    mut state: ResMut<RelaxationState>,
) {
    if device.world.is_some() {
        return;
    }
    let packed = PackedAssembly::from_assembly_with_options(
        &assembly,
        config.adaptive_segmentation,
        config.pin_fiber_ends,
    )
    .unwrap_or_else(|error| panic!("CubeCL assembly packing failed: {error}"));
    state.uploaded_bytes = packed_upload_bytes(&packed);
    state.active_segments = packed
        .segment_active
        .iter()
        .filter(|active| **active != 0)
        .count();
    state.active_vertices = packed
        .vertex_active
        .iter()
        .filter(|active| **active != 0)
        .count();
    device.world = Some(DeviceWorld::upload(&config, packed));
    let world = device.world.as_ref().unwrap();
    state.cell_size = world.cell_size();
    state.cell_count = world.cell_count();
}

fn run_device_batch(
    config: Res<RelaxationConfig>,
    mut assembly: ResMut<FiberAssembly>,
    mut state: ResMut<RelaxationState>,
    mut device: ResMut<DeviceState>,
    workflow_control: Res<WorkflowControl>,
    overrides: Res<RelaxationOverrides>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    if state.converged || state.cell_list_overflow || state.iterations >= config.max_iterations {
        return;
    }
    let world = device
        .world
        .as_mut()
        .expect("CubeCL device world must be uploaded before relaxation");
    let remaining = config.max_iterations - state.iterations;
    let mut batch_iterations = remaining.min(config.iterations_per_batch);
    if let Some(limit) = workflow_control.batch_iteration_limit {
        batch_iterations = batch_iterations.min(limit);
    }
    if let Some(interval) = config
        .debug_snapshot_interval
        .filter(|interval| *interval < config.iterations_per_batch)
    {
        let until_snapshot = interval - state.iterations % interval;
        batch_iterations = batch_iterations.min(until_snapshot);
    }
    let mut batch_config = *config;
    if workflow_control.force_full_batch {
        batch_config.force_full_iterations = true;
    }
    if let Some(tolerance) = workflow_control.penetration_tolerance {
        batch_config.penetration_tolerance = tolerance;
    }
    if let Some(maximum_ratio) = workflow_control.maximum_curvature_ratio {
        batch_config.curvature_ratio_tolerance = (maximum_ratio - 1.0).max(0.0);
    }
    if let Some(value) = overrides.correction_fraction {
        batch_config.correction_fraction = value;
    }
    if let Some(value) = overrides.motion_model {
        batch_config.motion_model = value;
    }
    if let Some(value) = overrides.contact_aggregation {
        batch_config.contact_aggregation = value;
    }
    if let Some(value) = overrides.stretch_stiffness {
        batch_config.stretch_stiffness = value;
    }
    if let Some(value) = overrides.bend_stiffness {
        batch_config.bend_stiffness = value;
    }
    if let Some(value) = overrides.curvature_limit_stiffness {
        batch_config.curvature_limit_stiffness = value;
    }
    if let Some(value) = overrides.constraint_iterations {
        batch_config.constraint_iterations = value;
    }
    if let Some(value) = overrides.curvature_cleanup_sweeps {
        batch_config.curvature_cleanup_sweeps = value;
    }
    let status = world.run_batch(&batch_config, batch_iterations);
    state.batches += 1;
    let new_splits = status.segment_splits.saturating_sub(state.segment_splits);
    let new_merges = status.segment_merges.saturating_sub(state.segment_merges);
    state.refinement_passes = status.refinement_passes;
    state.segment_splits = status.segment_splits;
    state.segment_merges = status.segment_merges;
    state.coarsening_passes = status.coarsening_passes;
    state.active_segments = state
        .active_segments
        .saturating_add(new_splits)
        .saturating_sub(new_merges);
    state.active_vertices = state
        .active_vertices
        .saturating_add(new_splits)
        .saturating_sub(new_merges);
    state.iterations = status.total_iterations;
    state.max_penetration = status.max_penetration;
    state.max_displacement = status.max_displacement;
    state.max_curvature_ratio = status.max_curvature_ratio;
    state.converged = status.converged;
    state.cell_list_overflow = status.cell_list_overflow;
    state.downloaded_bytes += 11 * size_of::<u32>() + 3 * size_of::<f32>();

    let terminal = status.cell_list_overflow || status.total_iterations >= config.max_iterations;
    let terminal = terminal || (status.converged && !workflow_control.hold_relax_stage);
    let snapshot_due = config.debug_snapshot_interval.is_some_and(|interval| {
        if interval < config.iterations_per_batch {
            status.total_iterations % interval == 0
        } else {
            let previous = state.last_snapshot_iteration.unwrap_or(0);
            status.total_iterations / interval > previous / interval
        }
    }) && state.last_snapshot_iteration != Some(status.total_iterations);
    let downloaded_positions = (snapshot_due || terminal).then(|| world.download_positions());
    if snapshot_due {
        let positions = downloaded_positions.as_ref().unwrap();
        state.downloaded_bytes += positions.len() * size_of::<f32>();
        let snapshot_assembly = if config.adaptive_segmentation.is_some()
            || assembly
                .topology
                .fibers
                .iter()
                .any(|fiber| fiber.formation_step > 0)
        {
            let mut snapshot_assembly = assembly.clone();
            world
                .unpack_positions(positions, &mut snapshot_assembly)
                .expect("CubeCL adaptive snapshot topology was invalid");
            Some(snapshot_assembly)
        } else {
            None
        };
        state.snapshots.push(RelaxationSnapshot {
            iteration: status.total_iterations,
            positions: positions.clone(),
            assembly: snapshot_assembly,
        });
        state.last_snapshot_iteration = Some(status.total_iterations);
    }
    if !terminal {
        return;
    }
    let positions = downloaded_positions.as_ref().unwrap();
    if !snapshot_due {
        state.downloaded_bytes += positions.len() * size_of::<f32>();
    }
    world
        .unpack_positions(positions, &mut assembly)
        .expect("CubeCL result geometry had the wrong size");
    if config.save_assembled_reference && status.converged {
        assembly.capture_assembled_reference();
    }
    next.set(if status.converged {
        config.next_stage
    } else {
        TangleStage::Done
    });
}

fn packed_upload_bytes(packed: &PackedAssembly) -> usize {
    packed.positions.len() * size_of::<f32>()
        + packed.intrinsic_positions.len() * size_of::<f32>()
        + packed.segment_vertices.len() * size_of::<u32>()
        + packed.segment_fibers.len() * size_of::<u32>()
        + packed.segment_radii.len() * size_of::<f32>()
        + packed.segment_rest_lengths.len() * size_of::<f32>()
        + packed.segment_active.len() * size_of::<u32>()
        + packed.segment_children.len() * size_of::<u32>()
        + packed.segment_birth_epochs.len() * size_of::<u32>()
        + packed.segment_contact_epochs.len() * size_of::<u32>()
        + packed.segment_quiet_epochs.len() * size_of::<u32>()
        + packed.segment_refinement_levels.len() * size_of::<u32>()
        + packed.fiber_segment_spans.len() * size_of::<u32>()
        + packed.fiber_vertex_spans.len() * size_of::<u32>()
        + packed.vertex_fibers.len() * size_of::<u32>()
        + packed.fiber_formation_layers.len() * size_of::<u32>()
        + packed.fiber_formation_steps.len() * size_of::<u32>()
        + packed.vertex_segments.len() * size_of::<u32>()
        + packed.vertex_rest_chords.len() * size_of::<f32>()
        + packed.vertex_max_curvature.len() * size_of::<f32>()
        + packed.vertex_active.len() * size_of::<u32>()
        + packed.vertex_pinned.len() * size_of::<u32>()
        + packed.vertex_refinement_levels.len() * size_of::<u32>()
        + 6 * size_of::<f32>()
        + 3 * size_of::<u32>()
}
