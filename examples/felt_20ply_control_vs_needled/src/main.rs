//! Twenty-ply, two-material felt exported as bonded rigid capsules.
//!
//! Two specimens share one population and deposition sequence: an unneedled
//! `control` and a layer-by-layer `needled` continuation. Each specimen runs
//! through three stages, each with its own checkpoint and case id
//! `felt-20ply-{specimen}-{stage}`:
//!
//! ```text
//! --stage form      build the stack (and needle it, for --specimen needled)
//! --stage cleanup   restart from this specimen's form checkpoint
//! --stage polish    restart from this specimen's cleanup checkpoint
//! ```
//!
//! This is a hypothetical material-generation study, not a calibrated model.
//!
//! The 7 µm and 19 µm populations contribute equal expected solid volume,
//! rather than equal fiber count. Lengths and densities are SI units.

use std::{any::TypeId, f64::consts::PI, io::Write};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_checkpoint::{CheckpointConfig, CheckpointPlugin, CheckpointReport};
use tangle_core::{FiberAssembly, PeriodicCell};
use tangle_example_support::ExampleOutput;
use tangle_export::{BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin};
use tangle_generate::{
    random_footprint_center, AcceptanceLimit, AdaptiveCompactionIncrement, CompactionConfig,
    CompactionGuards, CompactionPath, CompactionTarget, FiberPopulationSpec, FormationOperation,
    FormationRecipeConfig, FormationRecipePlugin, FormationRecipeState,
    MixedLayerStagedFiberPopulationGeneratorPlugin, NeedlingConfig, NeedlingSelection,
    OrientationDistribution, PositionDistribution, RelaxationAcceptance, RelaxationTargets,
    ScalarDistribution, SolveExhaustion, SolvePolicy,
};
use tangle_relax::{
    AdaptiveSegmentationConfig, CompactionEnergyModel, CompactionKinematics, ContactAggregation,
    DeviceState, FiberMotion, RelaxationConfig, RelaxationOverrides, RelaxationPlugin,
    RelaxationState,
};

const FOOTPRINT: f64 = 1.0e-3;
const STAGING_THICKNESS: f64 = 12.0e-3;
const LAYER_COUNT: u32 = 20;
const LAYER_SPACING: f32 = 50.0e-6;
const SMALL_FIBERS: usize = 480;
const LARGE_FIBERS: usize = 65;
const SEGMENTS_PER_FIBER: usize = 32;
const TARGET_VOLUME_FRACTION: f64 = 0.13;
const DENSITY: f64 = 1_800.0;
const GPU_BATCH_ITERATIONS: usize = 10;
const CONTACT_TOLERANCE: f32 = 0.30e-6;
const DEM_CONTACT_TOLERANCE: f32 = 0.01e-6;
const FINAL_BEND_RATIO: f32 = 1.001;
const CHECKPOINT_INTERVAL: usize = 500;
const PROGRESS_INTERVAL: usize = 250;
const FIRST_NEEDLED_LAYER: u32 = 2;
const NEEDLE_DEPTH: f32 = 350.0e-6;
const NEEDLE_DIAMETER: f32 = 150.0e-6;
const NEEDLE_POSITION_SEED: u64 = 20_260_940;
const NEEDLE_TARGET_HOLD_ITERATIONS: usize = 150;

#[derive(Default)]
struct Progress {
    events: usize,
    next_iteration: usize,
}

struct ProgressPlugin;

impl Plugin for ProgressPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Progress::default()).add_update_system(
            print_progress.run_if(in_state(TangleStage::Relax)),
            TanglePhase::Observe,
        );
    }
}

/// Which felt is being built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Specimen {
    /// Unneedled twenty-ply stack.
    Control,
    /// The same stack, needled after each ply from the third onward.
    Needled,
}

impl Specimen {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Needled => "needled",
        }
    }
}

/// Manufacturing stage. Later stages restart from the previous stage's
/// accepted geometry with a new formation recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Form,
    Cleanup,
    Polish,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Form => "form",
            Self::Cleanup => "cleanup",
            Self::Polish => "polish",
        }
    }

    /// Stage whose checkpoint seeds this one.
    fn source(self) -> Option<Self> {
        match self {
            Self::Form => None,
            Self::Cleanup => Some(Self::Form),
            Self::Polish => Some(Self::Cleanup),
        }
    }
}

const USAGE: &str = "usage: felt_20ply_control_vs_needled [--specimen control|needled] \
[--stage form|cleanup|polish] [--resume | --retry] [--debug-ovito]";

struct Options {
    specimen: Specimen,
    stage: Stage,
    /// Continue this stage's own checkpoint where it stopped.
    resume: bool,
    /// Restart this stage's recipe from this stage's own last saved geometry.
    retry: bool,
    debug_ovito: bool,
}

fn parse_options() -> Options {
    let mut options = Options {
        specimen: Specimen::Control,
        stage: Stage::Form,
        resume: false,
        retry: false,
        debug_ovito: false,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let (flag, inline_value) = match argument.split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (argument, None),
        };
        let mut value = || {
            inline_value
                .clone()
                .or_else(|| arguments.next())
                .unwrap_or_else(|| panic!("{flag} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--specimen" => {
                options.specimen = match value().as_str() {
                    "control" => Specimen::Control,
                    "needled" => Specimen::Needled,
                    other => panic!("unknown specimen {other:?}\n{USAGE}"),
                }
            }
            "--stage" => {
                options.stage = match value().as_str() {
                    "form" => Stage::Form,
                    "cleanup" => Stage::Cleanup,
                    "polish" => Stage::Polish,
                    other => panic!("unknown stage {other:?}\n{USAGE}"),
                }
            }
            "--resume" => options.resume = true,
            "--retry" => options.retry = true,
            "--debug-ovito" => options.debug_ovito = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => panic!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    assert!(
        !(options.resume && options.retry),
        "--resume and --retry are mutually exclusive"
    );
    assert!(
        !(options.retry && options.stage == Stage::Form),
        "--retry applies to the cleanup and polish stages"
    );
    options
}

/// Checkpoint case id for one specimen and stage.
fn case_id(specimen: Specimen, stage: Stage) -> String {
    format!("felt-20ply-{}-{}", specimen.name(), stage.name())
}

/// Output file stem for one specimen and stage, e.g. `needled_polish`.
fn file_stem(specimen: Specimen, stage: Stage) -> String {
    format!("{}_{}", specimen.name(), stage.name())
}

fn main() {
    let Options {
        specimen,
        stage,
        resume,
        retry,
        debug_ovito,
    } = parse_options();
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), "");
    let stem = file_stem(specimen, stage);
    let checkpoint_path = output.directory.join(format!("{stem}.restart"));

    // Starting or retrying a stage creates a new diagnostic run. Only
    // --resume appends to an existing trajectory; --retry keeps the stage's
    // own checkpoint because it restarts from that saved geometry.
    if !resume {
        let mut stale = vec![
            output.directory.join(format!("{stem}.dump")),
            output.directory.join(format!("{stem}.ovito")),
            output.directory.join(format!("{stem}_view.py")),
            output.directory.join(format!("{stem}_capsules.data")),
            output.directory.join(format!("{stem}_dirt.toml")),
        ];
        if !retry {
            stale.push(checkpoint_path.clone());
        }
        for path in stale {
            if let Err(error) = std::fs::remove_file(&path) {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::NotFound,
                    "could not reset {stem} output {}: {error}",
                    path.display()
                );
            }
        }
    }
    // Needled formation and every continuation stage always produce a
    // diagnostic trajectory; these are the runs where seeing needle pulls and
    // residual removal is most valuable.
    let debug_ovito = debug_ovito || specimen == Specimen::Needled || stage != Stage::Form;

    let own_checkpoint = CheckpointConfig::new(
        case_id(specimen, stage),
        &checkpoint_path,
        CHECKPOINT_INTERVAL,
    );
    let checkpoint = match (stage.source(), resume, retry) {
        (_, true, _) => own_checkpoint.with_resume(true),
        (_, _, true) => own_checkpoint
            .with_resume(true)
            .with_fresh_formation_on_resume(true),
        (None, false, false) => own_checkpoint,
        // Continue from the previous stage's accepted geometry, but discard
        // its completed recipe state and run only this stage's operations.
        (Some(source), false, false) => own_checkpoint
            .with_resume_source(
                case_id(specimen, source),
                output
                    .directory
                    .join(format!("{}.restart", file_stem(specimen, source))),
            )
            .with_fresh_formation_on_resume(true),
    };

    print_population_design();
    println!(
        "{} felt, {} stage: {} checkpoint {}",
        specimen.name(),
        stage.name(),
        if resume {
            "resuming"
        } else if retry {
            "retrying a fresh recipe from"
        } else {
            "writing"
        },
        checkpoint_path.display()
    );
    if let (Some(source), false, false) = (stage.source(), resume, retry) {
        println!(
            "  starting from {}",
            output
                .directory
                .join(format!("{}.restart", file_stem(specimen, source)))
                .display()
        );
    }

    let dense_continuation = stage != Stage::Form;

    let mut app = App::new();
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [FOOTPRINT, FOOTPRINT, STAGING_THICKNESS],
            [true, true, false],
        )))
        .add_plugins(MixedLayerStagedFiberPopulationGeneratorPlugin {
            specs: vec![small_fibers(), large_fibers()],
            first_formation_step: 0,
            minimum_layer_clearance: Some(100.0e-6),
        })
        .add_plugins(RelaxationPlugin {
            config: RelaxationConfig {
                constraint_iterations: 8,
                curvature_limit_safety_margin: 0.06,
                // Dense felt contacts repeatedly reintroduce local bend-limit
                // violations. More fiber-local Gauss-Seidel sweeps are much
                // cheaper than carrying the same offenders through tens of
                // thousands of whole broad-phase/contact iterations.
                curvature_cleanup_sweeps: if dense_continuation { 16 } else { 4 },
                contact_aggregation: if dense_continuation {
                    ContactAggregation::DeepestOnly
                } else {
                    ContactAggregation::UniformAverage
                },
                ..RelaxationConfig::flexible()
                    .with_iteration_limits(500_000, GPU_BATCH_ITERATIONS)
                    .with_penetration_tolerance(if dense_continuation {
                        2.5e-6
                    } else {
                        CONTACT_TOLERANCE
                    })
                    .with_contact_correction(if dense_continuation { 0.8 } else { 0.35 })
                    // Dense cleanup uses several fiber-local sweeps, so damp
                    // each curvature projection to avoid trading the same
                    // overlap and bend violation back and forth.
                    .with_curvature_limit(
                        if dense_continuation { 0.75 } else { 1.0 },
                        if dense_continuation { 1.7 } else { 1.0e-5 },
                    )
                    .with_max_step(2.0e-6)
                    .with_adaptive_segmentation(AdaptiveSegmentationConfig {
                        contact_length_over_diameter: 4.0,
                        minimum_length_over_diameter: 2.0,
                        maximum_refinement_levels: 6,
                        refinement_interval: 32,
                        refinement_persistence: 3,
                        coarsening_persistence: 8,
                        coarsening_error_over_diameter: 0.1,
                        coarsening_curvature_ratio: 0.25,
                    })
                    .with_next_stage(TangleStage::Export)
                    .with_debug_snapshots(debug_ovito.then_some(usize::MAX))
            },
        })
        .add_plugins(FormationRecipePlugin {
            config: FormationRecipeConfig {
                layer_axis: 2,
                operations: match (specimen, stage) {
                    (Specimen::Control, Stage::Form) => felt_recipe(),
                    (Specimen::Needled, Stage::Form) => needled_felt_recipe(),
                    (_, Stage::Cleanup) => cleanup_recipe(),
                    (_, Stage::Polish) => dem_polish_recipe(),
                },
            },
        })
        .add_plugins(CheckpointPlugin { config: checkpoint })
        .add_plugins(ProgressPlugin);

    if debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: tangle_export::OvitoTrajectoryConfig::fiber_segments(
                output.directory.join(format!("{stem}.dump")),
                usize::MAX,
            )
            .with_coloring(OvitoColoring::CurvatureRatio)
            .with_viewing_files(
                output.directory.join(format!("{stem}_view.py")),
                output.directory.join(format!("{stem}.ovito")),
            )
            .with_initial_frame(dense_continuation),
        });
    }
    let dem_path = output.directory.join(format!("{stem}_capsules.data"));
    app.add_plugins(BpmExportPlugin {
        config: tangle_export::BpmExportConfig::new(dem_path).with_density(DENSITY),
    });
    app.start();
    report(&app);
}

mod recipe;
mod reporting;

use recipe::{
    cleanup_recipe, dem_polish_recipe, felt_recipe, large_fibers, needled_felt_recipe,
    print_population_design, small_fibers,
};
use reporting::{print_progress, report};
