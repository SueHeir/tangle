mod compaction_control;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, FiberBendLimit, Section};
use tangle_relax::{
    CompactionKinematics, CompactionMetrics, DeviceState, DeviceWorld, DeviceWorldCheckpoint,
    RelaxationConfig, RelaxationOverrides, RelaxationSnapshot, RelaxationState, WorkflowControl,
};

use crate::compaction::{
    axis_weights, diameter_limited_log_strain, nominal_fiber_volume, validate_compaction,
    CompactionConfig, CompactionPath, CompactionReport, CompactionStepReport, CompactionStopReason,
    CompactionTarget,
};
use crate::junctions::{
    form_junctions_from_capture, validate_junction_policy, JunctionCapturePolicy,
    JunctionCaptureReport,
};
use crate::layered::layer_targets;
use crate::{generate_biased_fiber_population, FiberPopulationSpec};
use compaction_control::{
    active_capsule_bounds, advance_compaction, record_debug_snapshot, update_host_cell,
};

/// One generated population and the recipe step that inserts it.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberInsertionPopulation {
    /// Biased population sampled during the Generate stage.
    pub spec: FiberPopulationSpec,
    /// Formation step at which this population becomes active on the GPU.
    pub formation_step: u32,
}

/// Generates all capacity required by a staged insertion recipe.
pub struct StagedFiberPopulationGeneratorPlugin {
    /// Populations generated in order and tagged for later insertion.
    pub populations: Vec<FiberInsertionPopulation>,
}

/// Generates one multilayer population whose deposition layer also controls
/// its staged GPU activation step.
pub struct LayerStagedFiberPopulationGeneratorPlugin {
    /// Layered population generated during the normal Generate stage.
    pub spec: FiberPopulationSpec,
    /// Formation step assigned to layer zero; later layers increment it.
    pub first_formation_step: u32,
}

/// Generates several material populations into shared deposition layers and
/// activates every population according to its common layer label.
pub struct MixedLayerStagedFiberPopulationGeneratorPlugin {
    /// Layered material populations generated into the same cell.
    pub specs: Vec<FiberPopulationSpec>,
    /// Formation step assigned to layer zero; later layers increment it.
    pub first_formation_step: u32,
    /// Optional minimum radius-expanded separation between adjacent generated
    /// layer envelopes along the layering axis.
    pub minimum_layer_clearance: Option<f64>,
}

#[derive(Clone)]
struct MixedLayerStagedSpecs {
    specs: Vec<FiberPopulationSpec>,
    first_formation_step: u32,
    minimum_layer_clearance: Option<f64>,
}

impl Plugin for MixedLayerStagedFiberPopulationGeneratorPlugin {
    fn build(&self, app: &mut App) {
        assert!(!self.specs.is_empty());
        let layout = self.specs[0].position;
        assert!(matches!(
            layout,
            crate::PositionDistribution::Layered { .. }
        ));
        assert!(self.specs.iter().all(|spec| spec.position == layout));
        if let Some(clearance) = self.minimum_layer_clearance {
            assert!(clearance >= 0.0 && clearance.is_finite());
        }
        app.add_resource(MixedLayerStagedSpecs {
            specs: self.specs.clone(),
            first_formation_step: self.first_formation_step,
            minimum_layer_clearance: self.minimum_layer_clearance,
        })
        .add_update_system(
            generate_mixed_layer_staged_populations.run_if(in_state(TangleStage::Generate)),
            TanglePhase::Generate,
        );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_GENERATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

fn generate_mixed_layer_staged_populations(
    populations: Res<MixedLayerStagedSpecs>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    for spec in &populations.specs {
        let first = assembly.topology.fibers.len();
        generate_biased_fiber_population(&mut assembly, spec)
            .unwrap_or_else(|error| panic!("mixed layer-staged generation failed: {error}"));
        for fiber in &mut assembly.topology.fibers[first..] {
            let layer = fiber
                .formation_layer
                .expect("mixed layer-staged generation requires layer labels");
            fiber.formation_step = populations.first_formation_step + layer;
        }
    }
    if let Some(minimum) = populations.minimum_layer_clearance {
        let axis = match populations.specs[0].position {
            crate::PositionDistribution::Layered { axis, .. } => axis,
            _ => unreachable!(),
        };
        let clearances = staged_layer_clearances(&assembly, axis);
        for (lower_layer, clearance) in clearances.iter().copied().enumerate() {
            assert!(
                clearance >= minimum,
                "generated layer {} intersects layer {} in its staging configuration: clearance {:.3e} m is below required {:.3e} m",
                lower_layer + 1,
                lower_layer,
                clearance,
                minimum
            );
        }
        let smallest = clearances.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "staged {} layers with minimum radius-expanded clearance {:.3} um",
            clearances.len() + 1,
            smallest * 1.0e6
        );
    }
    assembly.provenance.notes.push(format!(
        "{} material populations share layer-staged activation",
        populations.specs.len()
    ));
    next.set(TangleStage::Relax);
}

fn staged_layer_clearances(assembly: &FiberAssembly, axis: usize) -> Vec<f64> {
    let layer_count = assembly
        .topology
        .fibers
        .iter()
        .filter_map(|fiber| fiber.formation_layer)
        .max()
        .map_or(0, |maximum| maximum as usize + 1);
    let mut lower = vec![f64::INFINITY; layer_count];
    let mut upper = vec![f64::NEG_INFINITY; layer_count];
    for fiber in &assembly.topology.fibers {
        let Some(layer) = fiber.formation_layer.map(|layer| layer as usize) else {
            continue;
        };
        let radius = match assembly.sections.entries[fiber.section.0 as usize] {
            tangle_core::Section::Circular { radius } => radius,
            tangle_core::Section::Elliptical { semi_axes } => semi_axes[0].max(semi_axes[1]),
        };
        let start = fiber.vertices.start as usize;
        let end = fiber.vertices.checked_end().expect("valid fiber span") as usize;
        for point in &assembly.geometry.placed.positions[start..end] {
            lower[layer] = lower[layer].min(point[axis] - radius);
            upper[layer] = upper[layer].max(point[axis] + radius);
        }
    }
    assert!(
        lower
            .iter()
            .zip(&upper)
            .all(|(lower, upper)| lower.is_finite() && upper.is_finite()),
        "formation layers must be contiguous and populated"
    );
    (0..layer_count.saturating_sub(1))
        .map(|layer| lower[layer + 1] - upper[layer])
        .collect()
}

impl Plugin for LayerStagedFiberPopulationGeneratorPlugin {
    fn build(&self, app: &mut App) {
        assert!(matches!(
            self.spec.position,
            crate::PositionDistribution::Layered { .. }
        ));
        app.add_resource(self.spec.clone())
            .add_resource(self.first_formation_step)
            .add_update_system(
                generate_layer_staged_population.run_if(in_state(TangleStage::Generate)),
                TanglePhase::Generate,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_GENERATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

fn generate_layer_staged_population(
    spec: Res<FiberPopulationSpec>,
    first_formation_step: Res<u32>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    let first = assembly.topology.fibers.len();
    generate_biased_fiber_population(&mut assembly, &spec)
        .unwrap_or_else(|error| panic!("layer-staged fiber generation failed: {error}"));
    for fiber in &mut assembly.topology.fibers[first..] {
        let layer = fiber
            .formation_layer
            .expect("layer-staged generation requires a layer label");
        fiber.formation_step = *first_formation_step + layer;
    }
    assembly.provenance.notes.push(format!(
        "multilayer population of {} fibers activated by deposition layer",
        spec.count
    ));
    next.set(TangleStage::Relax);
}

/// Deterministic pseudo-random center for a circular needle footprint in
/// `layer`, uniform over `origin + [0, extent)` in the two coordinates
/// orthogonal to the layer axis.
pub fn random_footprint_center(
    seed: u64,
    layer: u32,
    origin: [f32; 2],
    extent: [f32; 2],
) -> [f32; 2] {
    let unit = |bits: u64| (((bits >> 40) as f64) * (1.0 / ((1_u64 << 24) as f64))) as f32;
    let first = splitmix64(seed ^ (2 * layer) as u64);
    let second = splitmix64(seed ^ (2 * layer + 1) as u64);
    [
        origin[0] + extent[0] * unit(first),
        origin[1] + extent[1] * unit(second),
    ]
}

/// Rule used to select at most one pulled vertex from each layer fiber.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NeedlingSelection {
    /// Independently select each fiber with a reproducible probability.
    RandomFiberFraction {
        /// Probability that a layer fiber supplies one internal vertex.
        fraction: f32,
        /// Reproducible selection seed.
        seed: u64,
    },
    /// Select one internal vertex from every layer fiber entering a circular
    /// footprint in the plane normal to the recipe's layer axis.
    CircularFootprint {
        /// Center in the two coordinates orthogonal to the layer axis.
        center: [f32; 2],
        /// Physical footprint diameter.
        diameter: f32,
    },
}

/// Deterministic needle-pull selection and device-target parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeedlingConfig {
    /// Deposition layer whose fibers supply needle vertices.
    pub layer: u32,
    /// Spatial or statistical rule used to choose fiber vertices.
    pub selection: NeedlingSelection,
    /// Optional inclusive lower fiber-diameter bound for needle engagement.
    pub minimum_fiber_diameter: Option<f32>,
    /// Positive pull distance toward the negative layer-normal direction.
    pub depth: f32,
    /// Fraction of remaining target error applied per solver iteration.
    pub stiffness: f32,
    /// Maximum needle translation applied in one solver iteration.
    pub max_translation: f32,
    /// Additional actuator cap relative to the smallest selected fiber
    /// diameter. Sub-diameter steps allow contacts to respond along the pull.
    pub maximum_translation_over_fiber_diameter: f32,
}

/// Host-visible summary of one needle command.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NeedlingReport {
    /// Needled deposition layer.
    pub layer: u32,
    /// Number of selected vertices, at most one per selected fiber.
    pub selected_vertices: usize,
    /// Pull distance toward the negative layer-normal direction.
    pub depth: f32,
    /// Selection rule used by this needle command.
    pub selection: NeedlingSelection,
    /// Inclusive lower fiber-diameter bound used for this command.
    pub minimum_fiber_diameter: Option<f32>,
}

impl Plugin for StagedFiberPopulationGeneratorPlugin {
    fn build(&self, app: &mut App) {
        assert!(!self.populations.is_empty());
        app.add_resource(self.populations.clone())
            .add_update_system(
                generate_staged_populations.run_if(in_state(TangleStage::Generate)),
                TanglePhase::Generate,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_GENERATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

fn generate_staged_populations(
    populations: Res<Vec<FiberInsertionPopulation>>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    for population in populations.iter() {
        let first = assembly.topology.fibers.len();
        generate_biased_fiber_population(&mut assembly, &population.spec)
            .unwrap_or_else(|error| panic!("staged fiber-population generation failed: {error}"));
        for fiber in &mut assembly.topology.fibers[first..] {
            fiber.formation_step = population.formation_step;
        }
        assembly.provenance.notes.push(format!(
            "population of {} fibers inserted at formation step {}",
            population.spec.count, population.formation_step
        ));
    }
    next.set(TangleStage::Relax);
}

/// Whether a recipe acceptance limit must pass or may be deferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LimitEnforcement {
    /// Failure to satisfy this limit rejects the recipe operation.
    Hard,
    /// Failure may be recorded as a warning and deferred to a later hard pass.
    Soft,
}

/// One scalar recipe acceptance limit and its enforcement level.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AcceptanceLimit {
    /// Largest accepted value.
    pub maximum: f32,
    /// Whether the limit is mandatory at this recipe stage.
    pub enforcement: LimitEnforcement,
}

impl AcceptanceLimit {
    /// Creates a mandatory acceptance limit.
    pub const fn hard(maximum: f32) -> Self {
        Self {
            maximum,
            enforcement: LimitEnforcement::Hard,
        }
    }

    /// Creates a limit that may be deferred with a recorded warning.
    pub const fn soft(maximum: f32) -> Self {
        Self {
            maximum,
            enforcement: LimitEnforcement::Soft,
        }
    }
}

/// Physical residual limits used to accept a recipe relaxation stage.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RelaxationAcceptance {
    /// Capsule penetration acceptance limit.
    pub penetration: AcceptanceLimit,
    /// Current curvature divided by admissible curvature.
    pub curvature_ratio: AcceptanceLimit,
}

/// Residual targets used by the GPU solver while a recipe policy is active.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RelaxationTargets {
    /// Capsule penetration target.
    pub penetration: f32,
    /// Target current-curvature to admissible-curvature ratio.
    pub curvature_ratio: f32,
}

/// What a policy does when its iteration budget is exhausted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SolveExhaustion {
    /// Reject the recipe operation whenever any acceptance limit remains unmet.
    Reject,
    /// Continue with a warning only when all hard limits already pass.
    ContinueIfHardLimitsSatisfied,
}

/// Named solver and acceptance policy for one recipe relaxation gate.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SolvePolicy {
    /// Human-readable name included in progress and warning reports.
    pub name: String,
    /// Strict residuals toward which the GPU kernels continue solving.
    pub solver_targets: RelaxationTargets,
    /// Hard and soft thresholds that determine recipe-stage acceptance.
    pub acceptance: RelaxationAcceptance,
    /// Maximum GPU iterations spent at this gate.
    pub maximum_iterations: usize,
    /// Action taken if the budget ends before every acceptance limit passes.
    pub on_exhaustion: SolveExhaustion,
}

/// Nonterminal soft-limit violation recorded by a formation recipe.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FormationWarning {
    /// Zero-based operation that continued with the warning.
    pub operation: usize,
    /// Device iteration at which the warning was recorded.
    pub iteration: usize,
    /// Human-readable warning details.
    pub reason: String,
}

/// One modular command in a device-resident manufacturing recipe.
#[derive(Clone, Debug, PartialEq)]
pub enum FormationOperation {
    /// Insert all prepacked fibers assigned through the supplied formation step.
    ActivateFibersThrough(u32),
    /// Assign an explicit admissible bend radius to every fiber carrying a
    /// named material label, updating both canonical and resident GPU data.
    SetMaterialBendRadius {
        /// Exact material-table name to update.
        material_name: String,
        /// Smallest admissible centerline bend radius.
        minimum_bend_radius: f64,
    },
    /// Relax for exactly this many GPU iterations before advancing the recipe.
    RelaxFor(usize),
    /// Require the current configuration to satisfy the relaxation tolerances
    /// before advancing, or stop the recipe after the supplied local budget.
    RelaxUntilConverged {
        /// Maximum GPU iterations spent on this convergence gate.
        maximum_iterations: usize,
    },
    /// Relax under explicit solver targets and hard/soft recipe limits.
    RelaxWithPolicy(SolvePolicy),
    /// Relax under an explicit policy while temporarily overriding projection
    /// weights for this operation only.
    RelaxWithOverrides {
        /// Solver targets, acceptance limits, and iteration budget.
        policy: SolvePolicy,
        /// Constraint-projection settings active for the operation.
        overrides: RelaxationOverrides,
    },
    /// Advance persistent layer or needle targets until their largest
    /// coordinate residual reaches the requested tolerance.
    RelaxUntilTargetsReached {
        /// Maximum allowed layer-center or targeted-vertex coordinate error.
        tolerance: f32,
        /// Maximum GPU iterations spent advancing the active targets.
        maximum_iterations: usize,
    },
    /// Move labeled planar layers toward a scaled spacing about the cell center
    /// and keep that target active through subsequent recipe operations.
    MoveLayers {
        /// New layer spacing divided by the initially generated spacing.
        spacing_scale: f32,
        /// Fraction of the layer-center error applied by this command.
        stiffness: f32,
        /// Maximum translation of one fiber along the layer-normal axis.
        max_translation: f32,
    },
    /// Place one active layer a fixed distance above the preceding layer while
    /// retaining the targets of the already deposited stack.
    PlaceLayerAbove {
        /// Layer to place; must be greater than zero.
        layer: u32,
        /// Target center-plane gap from `layer - 1`.
        gap: f32,
        /// Fraction of layer-center error applied per solver iteration.
        stiffness: f32,
        /// Maximum translation of one fiber per solver iteration.
        max_translation: f32,
    },
    /// Pull selected internal layer vertices toward device-resident
    /// displacement targets.
    NeedleLayer(NeedlingConfig),
    /// Release the currently active per-vertex needle targets.
    ReleaseNeedles,
    /// Stop reapplying the most recent manufacturing-layer target.
    ReleaseLayerTargets,
    /// Shrink selected nonperiodic cell axes to the active capsule bounds
    /// without translating or deforming fibers.
    FitCellToActiveFibers {
        /// Axes whose lower and upper faces should be fit to active fibers.
        axes: [bool; 3],
        /// Empty clearance retained outside the capsule surfaces.
        padding: f32,
    },
    /// Incrementally shrink the resident cell, relax, and measure until a
    /// volume, dimension, pressure, or penalty-energy target is reached.
    Compact(CompactionConfig),
    /// Incrementally compact while temporarily emphasizing a selected family
    /// of solver constraints. A common use is contact-first settling followed
    /// by explicit post-compaction mechanics recovery.
    CompactWithOverrides {
        /// Closed-loop compaction target, path, increment, and guards.
        config: CompactionConfig,
        /// Constraint-projection settings active throughout compaction.
        overrides: RelaxationOverrides,
    },
    /// Promote current GPU contacts using the supplied persistent-junction policy.
    CaptureJunctions(JunctionCapturePolicy),
    /// Relax for a fixed interval and capture junctions at an independent cadence.
    RelaxAndCapture {
        /// Total GPU relaxation iterations in this operation.
        iterations: usize,
        /// GPU iterations between junction-capture checkpoints.
        every: usize,
        /// Policy applied at every checkpoint.
        policy: JunctionCapturePolicy,
    },
}

/// Ordered manufacturing commands interpreted between GPU relaxation batches.
#[derive(Clone, Debug, PartialEq)]
pub struct FormationRecipeConfig {
    /// Cartesian normal of fibers carrying `formation_layer` labels.
    pub layer_axis: usize,
    /// Commands executed in order while the device world remains resident.
    pub operations: Vec<FormationOperation>,
}

/// Host-visible record of a completed recipe command.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FormationEvent {
    /// Zero-based command index in the recipe.
    pub operation: usize,
    /// Device relaxation iteration at which the command completed.
    pub iteration: usize,
    /// Human-readable command summary for reports and educational examples.
    pub description: String,
}

/// A formation operation that stopped without producing an admissible state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FormationFailure {
    /// Zero-based operation that failed.
    pub operation: usize,
    /// Device iteration at failure.
    pub iteration: usize,
    /// Human-readable reason the recipe stopped.
    pub reason: String,
}

/// Observable progress through a [`FormationRecipeConfig`].
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FormationRecipeState {
    /// Command that will be interpreted next.
    pub next_operation: usize,
    /// Whether all commands have completed and normal convergence was released.
    pub released: bool,
    /// Completed command history.
    pub events: Vec<FormationEvent>,
    /// Detailed reports from completed junction-capture commands.
    pub junction_captures: Vec<JunctionCaptureReport>,
    /// Per-increment reports from all compaction operations.
    pub compaction_steps: Vec<CompactionStepReport>,
    /// Final reports from completed or guarded compaction operations.
    pub compactions: Vec<CompactionReport>,
    /// Completed per-layer needle pulls.
    pub needling: Vec<NeedlingReport>,
    /// Terminal failure, when a hard convergence gate could not be satisfied.
    pub failure: Option<FormationFailure>,
    /// Nonterminal soft-limit violations accepted by recipe policy.
    pub warnings: Vec<FormationWarning>,
    /// Layer-spacing target currently held by the recipe, if any.
    pub active_layer_spacing_scale: Option<f32>,
    initial_layer_targets: Vec<f32>,
    current_layer_targets: Vec<f32>,
    active_formation_step: Option<u32>,
    active_compaction: Option<ActiveCompaction>,
    relaxation_started_at: Option<usize>,
    last_periodic_capture_iteration: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct ActiveCompaction {
    operation: usize,
    solid_volume: f64,
    next_log_strain: f32,
    pending_started_at: Option<usize>,
    relaxation_windows: usize,
    steps: usize,
    last_log_strain: f32,
    previous_lengths: [f64; 3],
    cumulative_work: [f32; 3],
    metrics: CompactionMetrics,
    baseline_pending: bool,
    accepted_device: Option<DeviceWorldCheckpoint>,
    accepted_relaxation: Option<AcceptedRelaxation>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct AcceptedRelaxation {
    max_penetration: f32,
    max_displacement: f32,
    max_curvature_ratio: f32,
    refinement_passes: usize,
    segment_splits: usize,
    segment_merges: usize,
    coarsening_passes: usize,
    active_segments: usize,
    active_vertices: usize,
    snapshots_len: usize,
}

impl AcceptedRelaxation {
    fn capture(state: &RelaxationState) -> Self {
        Self {
            max_penetration: state.max_penetration,
            max_displacement: state.max_displacement,
            max_curvature_ratio: state.max_curvature_ratio,
            refinement_passes: state.refinement_passes,
            segment_splits: state.segment_splits,
            segment_merges: state.segment_merges,
            coarsening_passes: state.coarsening_passes,
            active_segments: state.active_segments,
            active_vertices: state.active_vertices,
            snapshots_len: state.snapshots.len(),
        }
    }

    fn restore(self, state: &mut RelaxationState) {
        state.max_penetration = self.max_penetration;
        state.max_displacement = self.max_displacement;
        state.max_curvature_ratio = self.max_curvature_ratio;
        state.refinement_passes = self.refinement_passes;
        state.segment_splits = self.segment_splits;
        state.segment_merges = self.segment_merges;
        state.coarsening_passes = self.coarsening_passes;
        state.active_segments = self.active_segments;
        state.active_vertices = self.active_vertices;
        state.snapshots.truncate(self.snapshots_len);
        state.converged = true;
        state.cell_list_overflow = false;
    }
}

/// GRASS controller for an ordered, GPU-resident formation recipe.
pub struct FormationRecipePlugin {
    /// Recipe installed as a GRASS resource.
    pub config: FormationRecipeConfig,
}

impl Plugin for FormationRecipePlugin {
    fn build(&self, app: &mut App) {
        assert!(self.config.layer_axis < 3);
        assert!(!self.config.operations.is_empty());
        for operation in &self.config.operations {
            match operation {
                FormationOperation::RelaxFor(iterations) => assert!(*iterations > 0),
                FormationOperation::RelaxUntilConverged { maximum_iterations } => {
                    assert!(*maximum_iterations > 0)
                }
                FormationOperation::RelaxWithPolicy(policy) => validate_solve_policy(policy),
                FormationOperation::RelaxWithOverrides { policy, overrides } => {
                    validate_solve_policy(policy);
                    validate_relaxation_overrides(*overrides);
                }
                FormationOperation::SetMaterialBendRadius {
                    material_name,
                    minimum_bend_radius,
                } => {
                    assert!(!material_name.trim().is_empty());
                    assert!(*minimum_bend_radius > 0.0 && minimum_bend_radius.is_finite());
                }
                FormationOperation::RelaxUntilTargetsReached {
                    tolerance,
                    maximum_iterations,
                } => {
                    assert!(*tolerance >= 0.0 && tolerance.is_finite());
                    assert!(*maximum_iterations > 0);
                }
                FormationOperation::MoveLayers {
                    spacing_scale,
                    stiffness,
                    max_translation,
                } => {
                    assert!(*spacing_scale > 0.0);
                    assert!(*stiffness > 0.0 && *stiffness <= 1.0);
                    assert!(*max_translation > 0.0);
                }
                FormationOperation::PlaceLayerAbove {
                    layer,
                    gap,
                    stiffness,
                    max_translation,
                } => {
                    assert!(*layer > 0);
                    assert!(*gap > 0.0 && gap.is_finite());
                    assert!(*stiffness > 0.0 && *stiffness <= 1.0);
                    assert!(*max_translation > 0.0);
                }
                FormationOperation::NeedleLayer(needling) => {
                    match needling.selection {
                        NeedlingSelection::RandomFiberFraction { fraction, .. } => {
                            assert!(fraction > 0.0 && fraction <= 1.0);
                        }
                        NeedlingSelection::CircularFootprint { center, diameter } => {
                            assert!(center.into_iter().all(f32::is_finite));
                            assert!(diameter > 0.0 && diameter.is_finite());
                        }
                    }
                    if let Some(diameter) = needling.minimum_fiber_diameter {
                        assert!(diameter > 0.0 && diameter.is_finite());
                    }
                    assert!(needling.depth > 0.0 && needling.depth.is_finite());
                    assert!(needling.stiffness > 0.0 && needling.stiffness <= 1.0);
                    assert!(needling.max_translation > 0.0);
                    assert!(
                        needling.maximum_translation_over_fiber_diameter > 0.0
                            && needling.maximum_translation_over_fiber_diameter.is_finite()
                    );
                }
                FormationOperation::ActivateFibersThrough(_)
                | FormationOperation::ReleaseLayerTargets
                | FormationOperation::ReleaseNeedles => {}
                FormationOperation::FitCellToActiveFibers { axes, padding } => {
                    assert!(axes.iter().any(|selected| *selected));
                    assert!(*padding >= 0.0 && padding.is_finite());
                }
                FormationOperation::Compact(config)
                | FormationOperation::CompactWithOverrides { config, .. } => {
                    validate_compaction(config)
                }
                FormationOperation::CaptureJunctions(policy) => validate_junction_policy(policy),
                FormationOperation::RelaxAndCapture {
                    iterations,
                    every,
                    policy,
                } => {
                    assert!(*iterations > 0);
                    assert!(*every > 0 && *every <= *iterations);
                    validate_junction_policy(policy);
                }
            }
        }
        app.add_resource(self.config.clone())
            .add_resource(FormationRecipeState::default())
            .add_update_system(
                control_formation_recipe.run_if(in_state(TangleStage::Relax)),
                TanglePhase::BuildBroadPhase,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_FORMATION.clone(), TANGLE_JUNCTIONS.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![
            TANGLE_WORKFLOW.clone(),
            TANGLE_ASSEMBLY.clone(),
            TANGLE_RELAXATION.clone(),
        ]
    }
}

fn control_formation_recipe(
    config: Res<FormationRecipeConfig>,
    relaxation_config: Res<RelaxationConfig>,
    mut assembly: ResMut<FiberAssembly>,
    mut device: ResMut<DeviceState>,
    mut relaxation: ResMut<RelaxationState>,
    mut workflow: ResMut<WorkflowControl>,
    mut relaxation_overrides: ResMut<RelaxationOverrides>,
    mut state: ResMut<FormationRecipeState>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    if state.initial_layer_targets.is_empty()
        && config.operations.iter().any(|operation| {
            matches!(
                operation,
                FormationOperation::MoveLayers { .. } | FormationOperation::PlaceLayerAbove { .. }
            )
        })
    {
        state.initial_layer_targets = layer_targets(&assembly, config.layer_axis);
        state.current_layer_targets = state.initial_layer_targets.clone();
    }
    if state.released || relaxation.cell_list_overflow {
        *relaxation_overrides = RelaxationOverrides::default();
        if let Some(world) = device.world.as_mut() {
            world.clear_layer_targets();
            world.clear_vertex_targets();
        }
        release_workflow(&mut workflow);
        return;
    }

    loop {
        let Some(operation) = config.operations.get(state.next_operation) else {
            *relaxation_overrides = RelaxationOverrides::default();
            state.released = true;
            state.active_layer_spacing_scale = None;
            device
                .world
                .as_mut()
                .expect("formation recipe requires an uploaded CubeCL device world")
                .clear_layer_targets();
            device
                .world
                .as_mut()
                .expect("formation recipe requires an uploaded CubeCL device world")
                .clear_vertex_targets();
            release_workflow(&mut workflow);
            // The last prescribed interval may have stopped with deliberately
            // disabled convergence. Request one final unconstrained batch.
            relaxation.converged = false;
            return;
        };
        *relaxation_overrides = match operation {
            FormationOperation::RelaxWithOverrides { overrides, .. }
            | FormationOperation::CompactWithOverrides { overrides, .. } => *overrides,
            _ => RelaxationOverrides::default(),
        };
        match operation {
            FormationOperation::ActivateFibersThrough(step) => {
                let (segments, vertices) = device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world")
                    .activate_formation_step(*step);
                relaxation.active_segments = segments;
                relaxation.active_vertices = vertices;
                relaxation.converged = false;
                state.active_formation_step = Some(*step);
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    format!("insert fiber groups through step {step}"),
                );
                record_debug_snapshot(
                    &relaxation_config,
                    device.world.as_ref().unwrap(),
                    &assembly,
                    &mut relaxation,
                );
            }
            FormationOperation::SetMaterialBendRadius {
                material_name,
                minimum_bend_radius,
            } => {
                let material = assembly
                    .materials
                    .entries
                    .iter()
                    .position(|entry| entry.name == *material_name);
                let matching_fibers = material
                    .map(|material| {
                        assembly
                            .topology
                            .fibers
                            .iter()
                            .enumerate()
                            .filter_map(|(index, fiber)| {
                                (fiber.material.0 as usize == material).then_some(index)
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if matching_fibers.is_empty() {
                    let reason = if material.is_some() {
                        format!("material {material_name:?} has no fibers")
                    } else {
                        let known = assembly
                            .materials
                            .entries
                            .iter()
                            .map(|entry| format!("{:?}", entry.name))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("unknown material {material_name:?}; known materials: {known}")
                    };
                    state.failure = Some(FormationFailure {
                        operation: state.next_operation,
                        iteration: relaxation.iterations,
                        reason,
                    });
                    state.released = true;
                    if let Some(world) = device.world.as_mut() {
                        world.clear_layer_targets();
                        world.clear_vertex_targets();
                    }
                    release_workflow(&mut workflow);
                    next.set(TangleStage::Done);
                    return;
                }
                let limit = FiberBendLimit {
                    minimum_bend_radius: *minimum_bend_radius,
                };
                let changed = matching_fibers.len();
                for index in matching_fibers {
                    assembly.admissibility.bend_limits[index] = Some(limit);
                }
                let maximum_curvatures = assembly
                    .admissibility
                    .bend_limits
                    .iter()
                    .map(|limit| limit.map_or(0.0, |limit| limit.maximum_curvature() as f32))
                    .collect::<Vec<_>>();
                device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world")
                    .set_fiber_maximum_curvatures(&maximum_curvatures);
                relaxation.converged = false;
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    format!(
                        "set {changed} {material_name:?} fibers to minimum bend radius {minimum_bend_radius:.3e}"
                    ),
                );
                record_debug_snapshot(
                    &relaxation_config,
                    device.world.as_ref().unwrap(),
                    &assembly,
                    &mut relaxation,
                );
            }
            FormationOperation::MoveLayers {
                spacing_scale,
                stiffness,
                max_translation,
            } => {
                let axis = config.layer_axis;
                let center =
                    (assembly.cell.origin[axis] + 0.5 * assembly.cell.basis[axis][axis]) as f32;
                state.current_layer_targets = state
                    .initial_layer_targets
                    .iter()
                    .map(|target| center + (*target - center) * *spacing_scale)
                    .collect();
                apply_layer_targets(
                    &mut device,
                    axis,
                    &state.current_layer_targets,
                    *stiffness,
                    *max_translation,
                );
                state.active_layer_spacing_scale = Some(*spacing_scale);
                relaxation.converged = false;
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    format!("move layers to spacing scale {spacing_scale:.3}"),
                );
            }
            FormationOperation::PlaceLayerAbove {
                layer,
                gap,
                stiffness,
                max_translation,
            } => {
                let index = *layer as usize;
                assert!(
                    index < state.current_layer_targets.len(),
                    "layer {layer} is outside the generated layer range"
                );
                state.current_layer_targets[index] = state.current_layer_targets[index - 1] + *gap;
                device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world")
                    .apply_single_layer_target(
                        config.layer_axis,
                        &state.current_layer_targets,
                        *layer,
                        *stiffness,
                        *max_translation,
                    );
                state.active_layer_spacing_scale = None;
                relaxation.converged = false;
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    format!(
                        "place layer {layer} at gap {gap:.3e} above layer {}",
                        layer - 1
                    ),
                );
            }
            FormationOperation::NeedleLayer(needling) => {
                assert!(
                    state
                        .active_formation_step
                        .is_some_and(|step| step >= needling.layer),
                    "cannot needle inactive layer {}",
                    needling.layer
                );
                let world = device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world");
                let vertices = selected_needling_vertices(
                    &assembly,
                    world,
                    state.active_formation_step,
                    config.layer_axis,
                    *needling,
                );
                if let Some(adaptive) = relaxation_config.adaptive_segmentation {
                    let forced_splits =
                        world.activate_refinement_vertices(&vertices, adaptive.refinement_interval);
                    relaxation.active_segments += forced_splits;
                    relaxation.active_vertices += forced_splits;
                    relaxation.segment_splits += forced_splits;
                    relaxation.refinement_passes += usize::from(forced_splits > 0);
                }
                world.clear_vertex_targets();
                if !vertices.is_empty() {
                    let diameter_limited_translation = vertices
                        .iter()
                        .map(|vertex| world.packed().vertex_fibers[*vertex as usize] as usize)
                        .map(|fiber| {
                            match assembly.sections.entries
                                [assembly.topology.fibers[fiber].section.0 as usize]
                            {
                                Section::Circular { radius } => 2.0 * radius,
                                Section::Elliptical { semi_axes } => {
                                    2.0 * semi_axes[0].min(semi_axes[1])
                                }
                            }
                        })
                        .fold(f64::INFINITY, f64::min)
                        as f32
                        * needling.maximum_translation_over_fiber_diameter;
                    world.apply_vertex_displacement_targets(
                        config.layer_axis,
                        &vertices,
                        -needling.depth,
                        needling.stiffness,
                        needling.max_translation.min(diameter_limited_translation),
                    );
                }
                relaxation.converged = false;
                state.needling.push(NeedlingReport {
                    layer: needling.layer,
                    selected_vertices: vertices.len(),
                    depth: needling.depth,
                    selection: needling.selection,
                    minimum_fiber_diameter: needling.minimum_fiber_diameter,
                });
                let selection_description = match needling.selection {
                    NeedlingSelection::RandomFiberFraction { fraction, seed } => {
                        format!("random fraction {fraction:.3}, seed {seed}")
                    }
                    NeedlingSelection::CircularFootprint { center, diameter } => format!(
                        "circular footprint diameter {:.3e} centered at [{:.3e}, {:.3e}]",
                        diameter, center[0], center[1]
                    ),
                };
                let diameter_description = needling
                    .minimum_fiber_diameter
                    .map_or_else(String::new, |diameter| {
                        format!(", fiber diameter >= {diameter:.3e}")
                    });
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    format!(
                        "needle {} vertices in layer {} through depth {:.3e} ({selection_description}{diameter_description})",
                        vertices.len(),
                        needling.layer,
                        needling.depth
                    ),
                );
            }
            FormationOperation::ReleaseNeedles => {
                device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world")
                    .clear_vertex_targets();
                relaxation.converged = false;
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    "release needle targets".to_string(),
                );
            }
            FormationOperation::ReleaseLayerTargets => {
                state.active_layer_spacing_scale = None;
                device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world")
                    .clear_layer_targets();
                relaxation.converged = false;
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    "release manufacturing-layer targets".to_string(),
                );
            }
            FormationOperation::FitCellToActiveFibers { axes, padding } => {
                for axis in 0..3 {
                    assert!(
                        !axes[axis] || !assembly.cell.periodic[axis],
                        "cannot fit periodic cell axis {axis} to active fibers"
                    );
                }
                let world = device
                    .world
                    .as_mut()
                    .expect("formation recipe requires an uploaded CubeCL device world");
                let positions = world.download_positions();
                let segment_active = world.download_segment_active();
                relaxation.downloaded_bytes += positions.len() * std::mem::size_of::<f32>()
                    + segment_active.len() * std::mem::size_of::<u32>();
                let (old_lower, old_upper) = world.cell_bounds();
                let (fiber_lower, fiber_upper) =
                    active_capsule_bounds(world.packed(), &positions, &segment_active, *padding);
                let new_lower = std::array::from_fn(|axis| {
                    if axes[axis] {
                        fiber_lower[axis].max(old_lower[axis])
                    } else {
                        old_lower[axis]
                    }
                });
                let new_upper = std::array::from_fn(|axis| {
                    if axes[axis] {
                        fiber_upper[axis].min(old_upper[axis])
                    } else {
                        old_upper[axis]
                    }
                });
                world.compact_cell(new_lower, new_upper, CompactionKinematics::MovingWalls);
                update_host_cell(&mut assembly, new_lower, new_upper);
                relaxation.cell_count = world.cell_count();
                relaxation.converged = false;
                finish_operation(
                    &mut state,
                    relaxation.iterations,
                    format!(
                        "fit cell to active fibers with {:.3e} padding: [{:.3e}, {:.3e}, {:.3e}]",
                        padding,
                        new_upper[0] - new_lower[0],
                        new_upper[1] - new_lower[1],
                        new_upper[2] - new_lower[2]
                    ),
                );
                record_debug_snapshot(&relaxation_config, world, &assembly, &mut relaxation);
            }
            FormationOperation::Compact(compaction) => {
                clear_recipe_solver_targets(&mut workflow);
                match advance_compaction(
                    compaction,
                    &relaxation_config,
                    &mut assembly,
                    &mut device,
                    &mut relaxation,
                    &mut workflow,
                    &mut state,
                ) {
                    CompactionProgress::Pending => return,
                    CompactionProgress::Finished(description) => {
                        finish_operation(&mut state, relaxation.iterations, description);
                        record_debug_snapshot(
                            &relaxation_config,
                            device.world.as_ref().unwrap(),
                            &assembly,
                            &mut relaxation,
                        );
                        continue;
                    }
                }
            }
            FormationOperation::CompactWithOverrides {
                config: compaction, ..
            } => {
                // Contact-first compaction deliberately defers mechanics to a
                // later recipe operation. Keep the normal penetration target,
                // but use this compaction stage's bend guard as its temporary
                // convergence ceiling.
                workflow.penetration_tolerance = None;
                workflow.maximum_curvature_ratio = Some(compaction.guards.maximum_bend_ratio);
                match advance_compaction(
                    compaction,
                    &relaxation_config,
                    &mut assembly,
                    &mut device,
                    &mut relaxation,
                    &mut workflow,
                    &mut state,
                ) {
                    CompactionProgress::Pending => return,
                    CompactionProgress::Finished(description) => {
                        clear_recipe_solver_targets(&mut workflow);
                        finish_operation(&mut state, relaxation.iterations, description);
                        record_debug_snapshot(
                            &relaxation_config,
                            device.world.as_ref().unwrap(),
                            &assembly,
                            &mut relaxation,
                        );
                        continue;
                    }
                }
            }
            FormationOperation::RelaxFor(iterations) => {
                let started = *state
                    .relaxation_started_at
                    .get_or_insert(relaxation.iterations);
                let completed = relaxation.iterations.saturating_sub(started);
                if completed >= *iterations {
                    state.relaxation_started_at = None;
                    finish_operation(
                        &mut state,
                        relaxation.iterations,
                        format!("relax for {iterations} iterations"),
                    );
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    continue;
                }
                workflow.hold_relax_stage = true;
                workflow.force_full_batch = true;
                workflow.batch_iteration_limit = Some(*iterations - completed);
                relaxation.converged = false;
                return;
            }
            FormationOperation::RelaxUntilConverged { maximum_iterations } => {
                let started = *state
                    .relaxation_started_at
                    .get_or_insert(relaxation.iterations);
                let completed = relaxation.iterations.saturating_sub(started);
                if relaxation.converged {
                    state.relaxation_started_at = None;
                    finish_operation(
                        &mut state,
                        relaxation.iterations,
                        format!(
                            "relax to convergence in {completed} iterations (penetration {:.3e}, bend ratio {:.6})",
                            relaxation.max_penetration, relaxation.max_curvature_ratio
                        ),
                    );
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    continue;
                }
                if completed >= *maximum_iterations {
                    let reason = format!(
                        "failed to converge in {maximum_iterations} iterations: penetration {:.3e}, bend ratio {:.6}",
                        relaxation.max_penetration, relaxation.max_curvature_ratio
                    );
                    state.failure = Some(FormationFailure {
                        operation: state.next_operation,
                        iteration: relaxation.iterations,
                        reason,
                    });
                    state.released = true;
                    state.relaxation_started_at = None;
                    device
                        .world
                        .as_mut()
                        .expect("formation recipe requires an uploaded CubeCL device world")
                        .clear_layer_targets();
                    device
                        .world
                        .as_mut()
                        .expect("formation recipe requires an uploaded CubeCL device world")
                        .clear_vertex_targets();
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    release_workflow(&mut workflow);
                    next.set(TangleStage::Done);
                    return;
                }
                workflow.hold_relax_stage = true;
                workflow.force_full_batch = false;
                workflow.batch_iteration_limit = Some(*maximum_iterations - completed);
                relaxation.converged = false;
                return;
            }
            FormationOperation::RelaxWithPolicy(policy)
            | FormationOperation::RelaxWithOverrides { policy, .. } => {
                let started = *state
                    .relaxation_started_at
                    .get_or_insert(relaxation.iterations);
                let completed = relaxation.iterations.saturating_sub(started);
                let measured_current_geometry = completed > 0 || relaxation.converged;
                if measured_current_geometry && acceptance_satisfied(policy.acceptance, &relaxation)
                {
                    state.relaxation_started_at = None;
                    clear_recipe_solver_targets(&mut workflow);
                    finish_operation(
                        &mut state,
                        relaxation.iterations,
                        format!(
                            "{} accepted in {completed} iterations (penetration {:.3e}, bend ratio {:.6})",
                            policy.name,
                            relaxation.max_penetration,
                            relaxation.max_curvature_ratio
                        ),
                    );
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    continue;
                }
                if completed >= policy.maximum_iterations {
                    let unmet = unmet_acceptance_limits(policy.acceptance, &relaxation);
                    let hard_limits_satisfied =
                        hard_acceptance_satisfied(policy.acceptance, &relaxation);
                    if policy.on_exhaustion == SolveExhaustion::ContinueIfHardLimitsSatisfied
                        && hard_limits_satisfied
                    {
                        let reason = format!(
                            "{} exhausted {} iterations with deferred soft limits: {unmet}",
                            policy.name, policy.maximum_iterations
                        );
                        let operation = state.next_operation;
                        state.warnings.push(FormationWarning {
                            operation,
                            iteration: relaxation.iterations,
                            reason: reason.clone(),
                        });
                        state.relaxation_started_at = None;
                        clear_recipe_solver_targets(&mut workflow);
                        finish_operation(
                            &mut state,
                            relaxation.iterations,
                            format!("continue after soft-limit warning: {reason}"),
                        );
                        record_debug_snapshot(
                            &relaxation_config,
                            device.world.as_ref().unwrap(),
                            &assembly,
                            &mut relaxation,
                        );
                        continue;
                    }

                    let reason = format!(
                        "{} failed after {} iterations: {unmet}",
                        policy.name, policy.maximum_iterations
                    );
                    state.failure = Some(FormationFailure {
                        operation: state.next_operation,
                        iteration: relaxation.iterations,
                        reason,
                    });
                    state.released = true;
                    *relaxation_overrides = RelaxationOverrides::default();
                    state.relaxation_started_at = None;
                    device
                        .world
                        .as_mut()
                        .expect("formation recipe requires an uploaded CubeCL device world")
                        .clear_layer_targets();
                    device
                        .world
                        .as_mut()
                        .expect("formation recipe requires an uploaded CubeCL device world")
                        .clear_vertex_targets();
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    release_workflow(&mut workflow);
                    next.set(TangleStage::Done);
                    return;
                }
                workflow.hold_relax_stage = true;
                workflow.force_full_batch = false;
                workflow.batch_iteration_limit =
                    Some(policy.maximum_iterations.saturating_sub(completed).max(1));
                workflow.penetration_tolerance = Some(policy.solver_targets.penetration);
                workflow.maximum_curvature_ratio = Some(policy.solver_targets.curvature_ratio);
                relaxation.converged = false;
                return;
            }
            FormationOperation::RelaxUntilTargetsReached {
                tolerance,
                maximum_iterations,
            } => {
                let started = *state
                    .relaxation_started_at
                    .get_or_insert(relaxation.iterations);
                let completed = relaxation.iterations.saturating_sub(started);
                let residuals = device
                    .world
                    .as_ref()
                    .expect("formation recipe requires an uploaded CubeCL device world")
                    .formation_target_error();
                let Some(maximum_error) = residuals.maximum() else {
                    state.relaxation_started_at = None;
                    finish_operation(
                        &mut state,
                        relaxation.iterations,
                        "no active formation target to advance".to_string(),
                    );
                    continue;
                };
                if maximum_error <= *tolerance {
                    state.relaxation_started_at = None;
                    finish_operation(
                        &mut state,
                        relaxation.iterations,
                        format!(
                            "advance formation targets in {completed} iterations (residual {maximum_error:.3e})"
                        ),
                    );
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    continue;
                }
                if completed >= *maximum_iterations {
                    let reason = format!(
                        "formation targets remained at residual {maximum_error:.3e} after {maximum_iterations} iterations (tolerance {tolerance:.3e})"
                    );
                    state.failure = Some(FormationFailure {
                        operation: state.next_operation,
                        iteration: relaxation.iterations,
                        reason,
                    });
                    state.released = true;
                    state.relaxation_started_at = None;
                    device
                        .world
                        .as_mut()
                        .expect("formation recipe requires an uploaded CubeCL device world")
                        .clear_layer_targets();
                    device
                        .world
                        .as_mut()
                        .expect("formation recipe requires an uploaded CubeCL device world")
                        .clear_vertex_targets();
                    record_debug_snapshot(
                        &relaxation_config,
                        device.world.as_ref().unwrap(),
                        &assembly,
                        &mut relaxation,
                    );
                    release_workflow(&mut workflow);
                    next.set(TangleStage::Done);
                    return;
                }
                workflow.hold_relax_stage = true;
                workflow.force_full_batch = true;
                workflow.batch_iteration_limit = Some(*maximum_iterations - completed);
                relaxation.converged = false;
                return;
            }
            FormationOperation::CaptureJunctions(policy) => {
                let report = capture_junctions(&mut assembly, &mut device, policy);
                let description = format!(
                    "capture {} junctions with '{}' from {} candidates",
                    report.created, report.policy_name, report.candidates
                );
                state.junction_captures.push(report);
                finish_operation(&mut state, relaxation.iterations, description);
            }
            FormationOperation::RelaxAndCapture {
                iterations,
                every,
                policy,
            } => {
                let started = *state
                    .relaxation_started_at
                    .get_or_insert(relaxation.iterations);
                let completed = relaxation.iterations.saturating_sub(started);
                if completed > 0
                    && completed % *every == 0
                    && state.last_periodic_capture_iteration != Some(relaxation.iterations)
                {
                    let report = capture_junctions(&mut assembly, &mut device, policy);
                    let operation = state.next_operation;
                    state.events.push(FormationEvent {
                        operation,
                        iteration: relaxation.iterations,
                        description: format!(
                            "periodic capture created {} '{}' junctions from {} candidates",
                            report.created, report.policy_name, report.candidates
                        ),
                    });
                    state.junction_captures.push(report);
                    state.last_periodic_capture_iteration = Some(relaxation.iterations);
                }
                if completed >= *iterations {
                    state.relaxation_started_at = None;
                    state.last_periodic_capture_iteration = None;
                    finish_operation(
                        &mut state,
                        relaxation.iterations,
                        format!(
                            "relax for {iterations} iterations with junction capture every {every}"
                        ),
                    );
                    continue;
                }
                let next_capture = ((completed / *every) + 1) * *every;
                workflow.hold_relax_stage = true;
                workflow.force_full_batch = true;
                workflow.batch_iteration_limit = Some(next_capture.min(*iterations) - completed);
                relaxation.converged = false;
                return;
            }
        }
    }
}

enum CompactionProgress {
    Pending,
    Finished(String),
}

fn capture_junctions(
    assembly: &mut FiberAssembly,
    device: &mut DeviceState,
    policy: &JunctionCapturePolicy,
) -> JunctionCaptureReport {
    let world = device
        .world
        .as_mut()
        .expect("formation recipe requires an uploaded CubeCL device world");
    let capture = world.capture_contacts(policy.maximum_surface_gap, policy.candidate_capacity);
    form_junctions_from_capture(assembly, world.packed(), capture, policy)
}

fn apply_layer_targets(
    device: &mut DeviceState,
    axis: usize,
    targets: &[f32],
    stiffness: f32,
    max_translation: f32,
) {
    device
        .world
        .as_mut()
        .expect("formation recipe requires an uploaded CubeCL device world")
        .apply_layer_targets(axis, targets, stiffness, max_translation);
}

fn selected_needling_vertices(
    assembly: &FiberAssembly,
    world: &DeviceWorld,
    active_through: Option<u32>,
    layer_axis: usize,
    config: NeedlingConfig,
) -> Vec<u32> {
    let active = world.download_vertex_active();
    let segment_active = world.download_segment_active();
    let positions = world.download_positions();
    selected_needling_vertices_from_mask(
        assembly,
        world.packed(),
        &positions,
        &active,
        &segment_active,
        active_through,
        layer_axis,
        config,
    )
}

fn selected_needling_vertices_from_mask(
    assembly: &FiberAssembly,
    packed: &tangle_relax::PackedAssembly,
    positions: &[f32],
    active: &[u32],
    segment_active: &[u32],
    active_through: Option<u32>,
    layer_axis: usize,
    config: NeedlingConfig,
) -> Vec<u32> {
    let plane_axes: Vec<usize> = (0..3).filter(|axis| *axis != layer_axis).collect();
    assembly
        .topology
        .fibers
        .iter()
        .enumerate()
        .filter(|(_, fiber)| fiber.formation_layer == Some(config.layer))
        .filter(|(_, fiber)| active_through.is_none_or(|step| fiber.formation_step <= step))
        .filter(|(_, fiber)| {
            config.minimum_fiber_diameter.is_none_or(|minimum| {
                let diameter = match assembly.sections.entries[fiber.section.0 as usize] {
                    Section::Circular { radius } => 2.0 * radius,
                    Section::Elliptical { semi_axes } => 2.0 * semi_axes[0].max(semi_axes[1]),
                };
                diameter >= minimum as f64
            })
        })
        .filter_map(|(fiber_index, fiber)| {
            let start = packed.fiber_vertex_spans[2 * fiber_index] as usize;
            let count = packed.fiber_vertex_spans[2 * fiber_index + 1] as usize;
            let active_vertices: Vec<u32> = (start..start + count)
                .filter(|vertex| active[*vertex] != 0)
                .map(|vertex| vertex as u32)
                .collect();
            match config.selection {
                NeedlingSelection::RandomFiberFraction { fraction, seed } => {
                    if active_vertices.len() <= 2 {
                        return None;
                    }
                    let selection = splitmix64(seed ^ fiber.id.0 as u64);
                    let unit = ((selection >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
                    if unit >= fraction as f64 {
                        return None;
                    }
                    let internal = active_vertices.len() - 2;
                    let local = 1 + (splitmix64(selection) % internal as u64) as usize;
                    Some(active_vertices[local])
                }
                NeedlingSelection::CircularFootprint { center, diameter } => {
                    let segment_start = packed.fiber_segment_spans[2 * fiber_index] as usize;
                    let segment_count = packed.fiber_segment_spans[2 * fiber_index + 1] as usize;
                    (segment_start..segment_start + segment_count)
                        .filter(|segment| segment_active[*segment] != 0)
                        .filter_map(|segment| {
                            let first = packed.segment_vertices[2 * segment] as usize;
                            let second = packed.segment_vertices[2 * segment + 1] as usize;
                            let mut start_delta = [0.0_f32; 2];
                            let mut direction = [0.0_f32; 2];
                            for (plane_coordinate, axis) in plane_axes.iter().enumerate() {
                                start_delta[plane_coordinate] =
                                    positions[3 * first + axis] - center[plane_coordinate];
                                direction[plane_coordinate] =
                                    positions[3 * second + axis] - positions[3 * first + axis];
                                if packed.cell_periodic[*axis] != 0 {
                                    let width = packed.cell_upper[*axis] - packed.cell_lower[*axis];
                                    start_delta[plane_coordinate] -=
                                        (start_delta[plane_coordinate] / width).round() * width;
                                    direction[plane_coordinate] -=
                                        (direction[plane_coordinate] / width).round() * width;
                                }
                            }
                            let length_squared =
                                direction[0] * direction[0] + direction[1] * direction[1];
                            let coordinate = if length_squared > 1.0e-20 {
                                (-(start_delta[0] * direction[0] + start_delta[1] * direction[1])
                                    / length_squared)
                                    .clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            let dx = start_delta[0] + coordinate * direction[0];
                            let dy = start_delta[1] + coordinate * direction[1];
                            let radius_squared = dx * dx + dy * dy;
                            if radius_squared > 0.25 * diameter * diameter {
                                return None;
                            }
                            let first_internal = start + 1;
                            let last_internal = start + count - 2;
                            let material_vertex = (first as f32
                                + coordinate * (second - first) as f32)
                                .round() as usize;
                            Some((
                                material_vertex.clamp(first_internal, last_internal) as u32,
                                radius_squared,
                            ))
                        })
                        .min_by(|(_, first), (_, second)| first.total_cmp(second))
                        .map(|(vertex, _)| vertex)
                }
            }
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_solve_policy(policy: &SolvePolicy) {
    assert!(!policy.name.trim().is_empty());
    assert!(policy.maximum_iterations > 0);
    assert!(policy.solver_targets.penetration >= 0.0);
    assert!(policy.solver_targets.penetration.is_finite());
    assert!(policy.solver_targets.curvature_ratio >= 1.0);
    assert!(policy.solver_targets.curvature_ratio.is_finite());
    assert!(policy.acceptance.penetration.maximum >= 0.0);
    assert!(policy.acceptance.penetration.maximum.is_finite());
    assert!(policy.acceptance.curvature_ratio.maximum >= 1.0);
    assert!(policy.acceptance.curvature_ratio.maximum.is_finite());
}

fn validate_relaxation_overrides(overrides: RelaxationOverrides) {
    assert!(overrides
        .correction_fraction
        .is_none_or(|value| value > 0.0 && value <= 1.0));
    assert!(overrides
        .stretch_stiffness
        .is_none_or(|value| (0.0..=1.0).contains(&value)));
    assert!(overrides
        .bend_stiffness
        .is_none_or(|value| (0.0..=1.0).contains(&value)));
    assert!(overrides
        .curvature_limit_stiffness
        .is_none_or(|value| (0.0..=1.0).contains(&value)));
    assert!(overrides
        .constraint_iterations
        .is_none_or(|value| value > 0));
    assert!(overrides
        .curvature_cleanup_sweeps
        .is_none_or(|value| value > 0));
}

fn acceptance_satisfied(acceptance: RelaxationAcceptance, relaxation: &RelaxationState) -> bool {
    relaxation.max_penetration <= acceptance.penetration.maximum
        && relaxation.max_curvature_ratio <= acceptance.curvature_ratio.maximum
}

fn hard_acceptance_satisfied(
    acceptance: RelaxationAcceptance,
    relaxation: &RelaxationState,
) -> bool {
    (acceptance.penetration.enforcement != LimitEnforcement::Hard
        || relaxation.max_penetration <= acceptance.penetration.maximum)
        && (acceptance.curvature_ratio.enforcement != LimitEnforcement::Hard
            || relaxation.max_curvature_ratio <= acceptance.curvature_ratio.maximum)
}

fn unmet_acceptance_limits(
    acceptance: RelaxationAcceptance,
    relaxation: &RelaxationState,
) -> String {
    let mut unmet = Vec::new();
    if relaxation.max_penetration > acceptance.penetration.maximum {
        unmet.push(format!(
            "penetration {:.3e} > {:.3e} ({:?})",
            relaxation.max_penetration,
            acceptance.penetration.maximum,
            acceptance.penetration.enforcement
        ));
    }
    if relaxation.max_curvature_ratio > acceptance.curvature_ratio.maximum {
        unmet.push(format!(
            "bend ratio {:.6} > {:.6} ({:?})",
            relaxation.max_curvature_ratio,
            acceptance.curvature_ratio.maximum,
            acceptance.curvature_ratio.enforcement
        ));
    }
    if unmet.is_empty() {
        "no acceptance limit exceeded".to_string()
    } else {
        unmet.join(", ")
    }
}

fn clear_recipe_solver_targets(workflow: &mut WorkflowControl) {
    workflow.penetration_tolerance = None;
    workflow.maximum_curvature_ratio = None;
}

fn finish_operation(state: &mut FormationRecipeState, iteration: usize, description: String) {
    state.events.push(FormationEvent {
        operation: state.next_operation,
        iteration,
        description,
    });
    state.next_operation += 1;
}

fn release_workflow(workflow: &mut WorkflowControl) {
    workflow.hold_relax_stage = false;
    workflow.batch_iteration_limit = None;
    workflow.force_full_batch = false;
    clear_recipe_solver_targets(workflow);
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
