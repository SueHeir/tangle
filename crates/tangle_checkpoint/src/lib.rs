//! Versioned, atomic restart checkpoints for long TANGLE formation runs.
//!
//! Checkpoints preserve the solver-neutral assembly, host-visible relaxation
//! and recipe state, and the persistent subset of the CubeCL world. Scratch
//! broad-phase buffers are rebuilt after resume while positions, topology
//! masks, manufacturing targets, cell bounds, and counters continue exactly.

#![warn(missing_docs)]

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};
use tangle_app::prelude::*;
use tangle_core::FiberAssembly;
use tangle_generate::FormationRecipeState;
use tangle_relax::{
    DeviceState, DeviceWorld, DeviceWorldCheckpoint, RelaxationConfig, RelaxationState,
    WorkflowControl,
};

const CHECKPOINT_MAGIC: [u8; 8] = *b"TANGLE01";
const CHECKPOINT_SCHEMA: u32 = 4;
const CHECKPOINT_HEADER_BYTES: usize = CHECKPOINT_MAGIC.len() + core::mem::size_of::<u32>();

/// Periodic checkpoint and optional resume settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointConfig {
    /// Stable application identifier used to reject an incompatible file.
    pub case_id: String,
    /// File atomically replaced by each completed checkpoint.
    pub path: PathBuf,
    /// Device iterations between checkpoint readbacks.
    pub interval_iterations: usize,
    /// Load `path` during plugin installation and resume in the Relax stage.
    pub resume: bool,
    /// Optional checkpoint path used only as the resume source.
    ///
    /// When absent, [`Self::path`] is both the resume source and save target.
    pub resume_path: Option<PathBuf>,
    /// Optional case identifier expected in [`Self::resume_path`].
    ///
    /// When absent, [`Self::case_id`] is used for resume validation.
    pub resume_case_id: Option<String>,
    /// Restore geometry and solver state but begin the installed formation
    /// recipe from its first operation.
    pub fresh_formation_on_resume: bool,
}

impl CheckpointConfig {
    /// Creates an enabled checkpoint configuration.
    pub fn new(
        case_id: impl Into<String>,
        path: impl Into<PathBuf>,
        interval_iterations: usize,
    ) -> Self {
        Self {
            case_id: case_id.into(),
            path: path.into(),
            interval_iterations,
            resume: false,
            resume_path: None,
            resume_case_id: None,
            fresh_formation_on_resume: false,
        }
    }

    /// Requests restoration from the configured path before the schedule starts.
    pub fn with_resume(mut self, resume: bool) -> Self {
        self.resume = resume;
        self
    }

    /// Uses another checkpoint as the resume source while retaining this
    /// configuration's path and case identifier for newly saved checkpoints.
    pub fn with_resume_source(
        mut self,
        case_id: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        self.resume = true;
        self.resume_case_id = Some(case_id.into());
        self.resume_path = Some(path.into());
        self
    }

    /// Starts the currently installed formation recipe from the beginning
    /// after restoring the checkpoint's assembly and device geometry.
    pub fn with_fresh_formation_on_resume(mut self, enabled: bool) -> Self {
        self.fresh_formation_on_resume = enabled;
        self
    }
}

/// Host-visible checkpoint activity summary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointReport {
    /// Whether this app restored a checkpoint.
    pub resumed: bool,
    /// Iteration restored at app construction.
    pub resumed_iteration: Option<usize>,
    /// Number of checkpoints written by this process.
    pub saves: usize,
    /// Iteration in the most recently committed checkpoint.
    pub last_saved_iteration: Option<usize>,
    /// Most recently committed checkpoint size.
    pub last_saved_bytes: Option<u64>,
}

/// Errors encountered while loading or committing a checkpoint.
#[derive(Debug)]
pub enum CheckpointError {
    /// Filesystem operation failed.
    Io(std::io::Error),
    /// Binary encoding or decoding failed.
    Codec(Box<bincode::ErrorKind>),
    /// File uses an unsupported schema or does not carry TANGLE's magic bytes.
    IncompatibleSchema {
        /// Magic bytes read from the file.
        magic: [u8; 8],
        /// Schema version read from the file.
        schema: u32,
    },
    /// File belongs to a different application recipe.
    CaseMismatch {
        /// Identifier requested by this app.
        expected: String,
        /// Identifier stored in the file.
        found: String,
    },
    /// Checkpoint expected formation state but the current app lacks the plugin.
    MissingFormationPlugin,
    /// RelaxationPlugin has not yet uploaded or restored its device world.
    MissingDeviceWorld,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "checkpoint I/O failed: {error}"),
            Self::Codec(error) => write!(formatter, "checkpoint codec failed: {error}"),
            Self::IncompatibleSchema { magic, schema } => write!(
                formatter,
                "incompatible checkpoint magic {magic:?} or schema {schema}"
            ),
            Self::CaseMismatch { expected, found } => write!(
                formatter,
                "checkpoint case '{found}' does not match requested case '{expected}'"
            ),
            Self::MissingFormationPlugin => formatter.write_str(
                "checkpoint contains a formation recipe but no FormationRecipePlugin is installed",
            ),
            Self::MissingDeviceWorld => {
                formatter.write_str("checkpoint requested before a device world exists")
            }
        }
    }
}

impl Error for CheckpointError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Codec(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CheckpointError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<Box<bincode::ErrorKind>> for CheckpointError {
    fn from(error: Box<bincode::ErrorKind>) -> Self {
        Self::Codec(error)
    }
}

/// Complete validated contents of one TANGLE restart file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TangleCheckpoint {
    /// Stable application identifier stored with this restart.
    pub case_id: String,
    /// Solver-neutral assembly metadata and host geometry.
    pub assembly: FiberAssembly,
    /// Host-visible relaxation counters and residuals.
    pub relaxation: RelaxationState,
    /// Pending workflow controls.
    pub workflow: WorkflowControl,
    /// Optional manufacturing-recipe progress.
    pub formation: Option<FormationRecipeState>,
    /// Authoritative device geometry and adaptive topology.
    pub device: DeviceWorldCheckpoint,
}

#[derive(Clone, Debug, Default)]
struct CheckpointState {
    last_saved_iteration: Option<usize>,
}

/// GRASS plugin that saves and optionally restores one rolling checkpoint.
pub struct CheckpointPlugin {
    /// Save/resume policy.
    pub config: CheckpointConfig,
}

impl Plugin for CheckpointPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            self.config.interval_iterations > 0,
            "checkpoint interval must be positive"
        );
        assert!(
            !self.config.case_id.is_empty(),
            "checkpoint case_id is empty"
        );

        let mut report = CheckpointReport::default();
        let mut state = CheckpointState::default();
        if self.config.resume {
            let resume_path = self
                .config
                .resume_path
                .as_deref()
                .unwrap_or(&self.config.path);
            let resume_case_id = self
                .config
                .resume_case_id
                .as_deref()
                .unwrap_or(&self.config.case_id);
            let checkpoint = load_checkpoint(resume_path, resume_case_id)
                .unwrap_or_else(|error| panic!("could not resume TANGLE checkpoint: {error}"));
            let relaxation_config = *app
                .get_resource_ref::<RelaxationConfig>()
                .expect("CheckpointPlugin must be installed after RelaxationPlugin");
            let mut world = DeviceWorld::restore(&relaxation_config, checkpoint.device);
            let mut relaxation = checkpoint.relaxation;
            let iteration = relaxation.iterations;
            let terminal_failure = !self.config.fresh_formation_on_resume
                && checkpoint
                    .formation
                    .as_ref()
                    .is_some_and(|formation| formation.failure.is_some());
            let workflow = if self.config.fresh_formation_on_resume {
                world.clear_layer_targets();
                world.clear_vertex_targets();
                relaxation.converged = false;
                WorkflowControl::default()
            } else {
                checkpoint.workflow
            };
            app.add_resource(checkpoint.assembly)
                .add_resource(relaxation)
                .add_resource(DeviceState { world: Some(world) })
                .add_resource(workflow)
                .add_resource(CurrentState(if terminal_failure {
                    TangleStage::Done
                } else {
                    TangleStage::Relax
                }))
                .add_resource(NextState::<TangleStage>(None));
            if !self.config.fresh_formation_on_resume {
                if let Some(formation) = checkpoint.formation {
                    if app.get_resource_ref::<FormationRecipeState>().is_none() {
                        panic!("{}", CheckpointError::MissingFormationPlugin);
                    }
                    app.add_resource(formation);
                }
            }
            state.last_saved_iteration = Some(iteration);
            report.resumed = true;
            report.resumed_iteration = Some(iteration);
            report.last_saved_iteration = Some(iteration);
            println!(
                "resumed checkpoint {} at iteration {}{}",
                resume_path.display(),
                iteration,
                if self.config.fresh_formation_on_resume {
                    " with a fresh formation recipe"
                } else {
                    ""
                }
            );
        }

        app.add_resource(self.config.clone())
            .add_resource(state)
            .add_resource(report)
            .add_update_system(
                save_checkpoint.run_if(in_state(TangleStage::Relax)),
                TanglePhase::Observe,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        Vec::new()
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![
            TANGLE_WORKFLOW.clone(),
            TANGLE_ASSEMBLY.clone(),
            TANGLE_RELAXATION.clone(),
        ]
    }
}

fn save_checkpoint(
    config: Res<CheckpointConfig>,
    assembly: Res<FiberAssembly>,
    relaxation: Res<RelaxationState>,
    formation: Option<Res<FormationRecipeState>>,
    device: Res<DeviceState>,
    workflow: Res<WorkflowControl>,
    mut state: ResMut<CheckpointState>,
    mut report: ResMut<CheckpointReport>,
) {
    let iteration = relaxation.iterations;
    if iteration == 0 || state.last_saved_iteration == Some(iteration) {
        return;
    }
    let crossed_interval = state
        .last_saved_iteration
        .map_or(iteration >= config.interval_iterations, |previous| {
            iteration / config.interval_iterations > previous / config.interval_iterations
        });
    let terminal = relaxation.cell_list_overflow
        || (relaxation.converged && !workflow.hold_relax_stage)
        || formation
            .as_ref()
            .is_some_and(|recipe| recipe.failure.is_some());
    if !crossed_interval && !terminal {
        return;
    }

    let world = device
        .world
        .as_ref()
        .ok_or(CheckpointError::MissingDeviceWorld)
        .unwrap_or_else(|error| panic!("TANGLE checkpoint failed: {error}"));
    let mut relaxation = relaxation.clone();
    relaxation.snapshots.clear();
    let checkpoint = TangleCheckpoint {
        case_id: config.case_id.clone(),
        assembly: assembly.clone(),
        relaxation,
        workflow: *workflow,
        formation: formation.map(|state| state.clone()),
        device: world.checkpoint(),
    };
    let bytes = encode_checkpoint(&checkpoint)
        .unwrap_or_else(|error| panic!("TANGLE checkpoint encoding failed: {error}"));
    commit_atomically(&config.path, &bytes)
        .unwrap_or_else(|error| panic!("TANGLE checkpoint write failed: {error}"));
    state.last_saved_iteration = Some(iteration);
    report.saves += 1;
    report.last_saved_iteration = Some(iteration);
    report.last_saved_bytes = Some(bytes.len() as u64);
    println!(
        "  checkpoint: iteration {}, {:.1} MiB at {}",
        iteration,
        bytes.len() as f64 / (1024.0 * 1024.0),
        config.path.display()
    );
}

/// Reads and validates a restart without installing it into an application.
///
/// This is useful for offline diagnostics and conversion tools. Normal runs
/// should generally restore through [`CheckpointPlugin`].
pub fn load_checkpoint(path: &Path, case_id: &str) -> Result<TangleCheckpoint, CheckpointError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut magic = [0_u8; CHECKPOINT_MAGIC.len()];
    magic[..bytes.len().min(CHECKPOINT_MAGIC.len())]
        .copy_from_slice(&bytes[..bytes.len().min(CHECKPOINT_MAGIC.len())]);
    let schema = bytes
        .get(CHECKPOINT_MAGIC.len()..CHECKPOINT_HEADER_BYTES)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
        .unwrap_or(0);
    if magic != CHECKPOINT_MAGIC || schema != CHECKPOINT_SCHEMA {
        return Err(CheckpointError::IncompatibleSchema { magic, schema });
    }
    let checkpoint: TangleCheckpoint = bincode::deserialize(&bytes[CHECKPOINT_HEADER_BYTES..])?;
    if checkpoint.case_id != case_id {
        return Err(CheckpointError::CaseMismatch {
            expected: case_id.to_string(),
            found: checkpoint.case_id,
        });
    }
    Ok(checkpoint)
}

fn encode_checkpoint(checkpoint: &TangleCheckpoint) -> Result<Vec<u8>, CheckpointError> {
    let payload = bincode::serialize(checkpoint)?;
    let mut bytes = Vec::with_capacity(CHECKPOINT_HEADER_BYTES + payload.len());
    bytes.extend_from_slice(&CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&CHECKPOINT_SCHEMA.to_le_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn commit_atomically(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell, Section};
    use tangle_relax::PackedAssembly;

    fn checkpoint_fixture(case_id: &str) -> TangleCheckpoint {
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [true, true, false]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.01 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
            )
            .unwrap();
        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        TangleCheckpoint {
            case_id: case_id.to_string(),
            assembly,
            relaxation: RelaxationState {
                iterations: 42,
                ..RelaxationState::default()
            },
            workflow: WorkflowControl {
                hold_relax_stage: true,
                batch_iteration_limit: Some(7),
                force_full_batch: true,
                penetration_tolerance: None,
                maximum_curvature_ratio: None,
            },
            formation: None,
            device: DeviceWorldCheckpoint {
                wall_reactions: vec![0.0; 3 * packed.vertex_count()],
                packed,
                total_iterations: 42,
                refinement_count: [0; 6],
                layer_target: None,
                vertex_target: None,
            },
        }
    }

    #[test]
    fn atomic_file_round_trip_preserves_restart_state() {
        let checkpoint = checkpoint_fixture("round-trip");
        let path = std::env::temp_dir().join(format!(
            "tangle-checkpoint-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let bytes = encode_checkpoint(&checkpoint).unwrap();
        commit_atomically(&path, &bytes).unwrap();
        let restored = load_checkpoint(&path, "round-trip").unwrap();
        assert_eq!(restored.relaxation.iterations, 42);
        assert_eq!(restored.workflow, checkpoint.workflow);
        assert_eq!(restored.device, checkpoint.device);
        assert_eq!(restored.assembly, checkpoint.assembly);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_checkpoint_from_another_case() {
        let checkpoint = checkpoint_fixture("first");
        let path = std::env::temp_dir().join(format!(
            "tangle-checkpoint-wrong-case-{}.bin",
            std::process::id()
        ));
        let bytes = encode_checkpoint(&checkpoint).unwrap();
        commit_atomically(&path, &bytes).unwrap();
        let error = load_checkpoint(&path, "second").unwrap_err();
        assert!(matches!(error, CheckpointError::CaseMismatch { .. }));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_an_incompatible_schema_before_decoding_payload() {
        let checkpoint = checkpoint_fixture("schema");
        let path = std::env::temp_dir().join(format!(
            "tangle-checkpoint-schema-{}.bin",
            std::process::id()
        ));
        let mut bytes = encode_checkpoint(&checkpoint).unwrap();
        bytes[CHECKPOINT_MAGIC.len()..CHECKPOINT_HEADER_BYTES]
            .copy_from_slice(&(CHECKPOINT_SCHEMA + 1).to_le_bytes());
        commit_atomically(&path, &bytes).unwrap();
        let error = load_checkpoint(&path, "schema").unwrap_err();
        assert!(matches!(error, CheckpointError::IncompatibleSchema { .. }));
        fs::remove_file(path).unwrap();
    }
}
