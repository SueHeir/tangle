//! Twenty-ply, two-material fake felt exported as bonded rigid capsules.
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
use tangle_export::{
    DemCapsuleBpmExportPlugin, DemCapsuleBpmExportReport, OvitoColoring, OvitoTrajectoryPlugin,
};
use tangle_generate::{
    AcceptanceLimit, AdaptiveCompactionIncrement, CompactionConfig, CompactionGuards,
    CompactionPath, CompactionTarget, FiberPopulationSpec, FormationOperation,
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

fn main() {
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), "");
    let checkpoint_path = output.directory.join("relaxation.restart");
    let cleanup_checkpoint_path = output.directory.join("cleanup.restart");
    let needled_checkpoint_path = output.directory.join("needled.restart");
    let arguments = std::env::args().collect::<Vec<_>>();
    let cleanup_from_v7 = arguments
        .iter()
        .any(|argument| argument == "--cleanup-from-checkpoint");
    let resume_cleanup = arguments
        .iter()
        .any(|argument| argument == "--resume-cleanup");
    let retry_cleanup = arguments
        .iter()
        .any(|argument| argument == "--retry-cleanup");
    let dem_polish = arguments.iter().any(|argument| argument == "--dem-polish");
    let needled_fresh = arguments.iter().any(|argument| argument == "--needled");
    let resume_needled = arguments
        .iter()
        .any(|argument| argument == "--resume-needled");
    let needled_cleanup = arguments
        .iter()
        .any(|argument| argument == "--needled-cleanup");
    let resume_needled_cleanup = arguments
        .iter()
        .any(|argument| argument == "--resume-needled-cleanup");
    let needled_dem_polish = arguments
        .iter()
        .any(|argument| argument == "--needled-dem-polish");
    let needled = needled_fresh
        || resume_needled
        || needled_cleanup
        || resume_needled_cleanup
        || needled_dem_polish;
    let cleanup = cleanup_from_v7 || resume_cleanup || retry_cleanup || dem_polish;
    let resume = arguments.iter().any(|argument| argument == "--resume");
    assert!(
        usize::from(cleanup_from_v7)
            + usize::from(resume_cleanup)
            + usize::from(retry_cleanup)
            + usize::from(dem_polish)
            + usize::from(needled_fresh)
            + usize::from(resume_needled)
            + usize::from(needled_cleanup)
            + usize::from(resume_needled_cleanup)
            + usize::from(needled_dem_polish)
            + usize::from(resume)
            <= 1,
        "resume and continuation modes are mutually exclusive"
    );
    // Starting or retrying the cleanup recipe creates a new diagnostic run.
    // Only --resume-cleanup is allowed to append to an existing trajectory.
    if cleanup_from_v7 || retry_cleanup {
        for path in [
            cleanup_checkpoint_path.clone(),
            output.directory.join("cleanup.dump"),
            output.directory.join("cleanup.ovito"),
            output.directory.join("cleanup_view.py"),
            output.directory.join("cleanup_capsules.data"),
            output.directory.join("cleanup_dirt.toml"),
        ] {
            if let Err(error) = std::fs::remove_file(&path) {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::NotFound,
                    "could not reset cleanup output {}: {error}",
                    path.display()
                );
            }
        }
    }
    if needled_fresh {
        for path in [
            needled_checkpoint_path.clone(),
            output.directory.join("needled.dump"),
            output.directory.join("needled.ovito"),
            output.directory.join("needled_view.py"),
            output.directory.join("needled_capsules.data"),
            output.directory.join("needled_dirt.toml"),
        ] {
            if let Err(error) = std::fs::remove_file(&path) {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::NotFound,
                    "could not reset needled output {}: {error}",
                    path.display()
                );
            }
        }
    }
    // Cleanup continuations always produce a diagnostic trajectory; this is
    // the run where seeing residual removal is most valuable.
    let debug_ovito =
        cleanup || needled || arguments.iter().any(|argument| argument == "--debug-ovito");

    let checkpoint = if needled_dem_polish {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-needled",
            &needled_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
        .with_fresh_formation_on_resume(true)
    } else if needled_cleanup {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-needled",
            &needled_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
        .with_fresh_formation_on_resume(true)
    } else if resume_needled_cleanup {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-needled",
            &needled_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
    } else if needled_fresh {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-needled",
            &needled_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
    } else if resume_needled {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-needled",
            &needled_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
    } else if cleanup_from_v7 {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-cleanup",
            &cleanup_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume_source("fake-felted-20-layer-v7", &checkpoint_path)
        .with_fresh_formation_on_resume(true)
    } else if resume_cleanup {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-cleanup",
            &cleanup_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
    } else if retry_cleanup {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-cleanup",
            &cleanup_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
        .with_fresh_formation_on_resume(true)
    } else if dem_polish {
        // Continue from the accepted cleanup geometry, but discard its completed
        // recipe state and run only the DEM handoff polish below. Reuse the
        // cleanup checkpoint/output stem so this does not create another family
        // of nearly identical result files.
        CheckpointConfig::new(
            "fake-felted-20-layer-v7-cleanup",
            &cleanup_checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(true)
        .with_fresh_formation_on_resume(true)
    } else {
        CheckpointConfig::new(
            "fake-felted-20-layer-v7",
            &checkpoint_path,
            CHECKPOINT_INTERVAL,
        )
        .with_resume(resume)
    };

    print_population_design();
    if cleanup_from_v7 {
        println!(
            "cleanup continuation: {} -> {}",
            checkpoint_path.display(),
            cleanup_checkpoint_path.display()
        );
    } else if resume_cleanup {
        println!(
            "resuming cleanup continuation from {}",
            cleanup_checkpoint_path.display()
        );
    } else if retry_cleanup {
        println!(
            "retrying a fresh cleanup recipe from rejected state {}",
            cleanup_checkpoint_path.display()
        );
    } else if dem_polish {
        println!(
            "polishing accepted cleanup geometry for DEM contact startup from {}",
            cleanup_checkpoint_path.display()
        );
    } else if needled_dem_polish {
        println!(
            "polishing accepted needled geometry for DEM contact startup from {}",
            needled_checkpoint_path.display()
        );
    } else if needled_cleanup {
        println!(
            "starting staged cleanup of the layer-by-layer needled stack from {}",
            needled_checkpoint_path.display()
        );
    } else if resume_needled_cleanup {
        println!(
            "resuming staged needled cleanup from {}",
            needled_checkpoint_path.display()
        );
    } else if needled_fresh {
        println!(
            "starting layer-by-layer needled formation with checkpoint {}",
            needled_checkpoint_path.display()
        );
    } else if resume_needled {
        println!(
            "resuming needling continuation from {}",
            needled_checkpoint_path.display()
        );
    }

    let dense_continuation =
        cleanup || needled_cleanup || resume_needled_cleanup || needled_dem_polish;

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
                operations: if needled_dem_polish {
                    dem_polish_recipe()
                } else if needled_cleanup || resume_needled_cleanup {
                    cleanup_recipe()
                } else if needled {
                    needled_felt_recipe()
                } else if dem_polish {
                    dem_polish_recipe()
                } else if cleanup {
                    cleanup_recipe()
                } else {
                    felt_recipe()
                },
            },
        })
        .add_plugins(CheckpointPlugin { config: checkpoint })
        .add_plugins(ProgressPlugin);

    if debug_ovito {
        let stem = if needled {
            "needled"
        } else if cleanup {
            "cleanup"
        } else {
            "relaxation"
        };
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
    let stem = if needled {
        "needled"
    } else if cleanup {
        "cleanup"
    } else {
        "relaxation"
    };
    let dem_path = output.directory.join(format!("{stem}_capsules.data"));
    let dirt_path = output.directory.join(format!("{stem}_dirt.toml"));
    app.add_plugins(DemCapsuleBpmExportPlugin {
        config: tangle_export::DemCapsuleBpmExportConfig::new(dem_path)
            .with_density(DENSITY)
            .with_maximum_length_over_diameter(4.0)
            .with_dirt_config(dirt_path),
    });
    app.start();
    report(&app);
}

/// The control recipe with one additional manufacturing block: after each new
/// ply has been lowered, released, and relaxed, needle that ply before the next
/// one is inserted. All population and non-needling operations match
/// [`felt_recipe`].
mod recipe;
mod reporting;

use recipe::{
    cleanup_recipe, dem_polish_recipe, felt_recipe, large_fibers, needled_felt_recipe,
    print_population_design, small_fibers,
};
use reporting::{print_progress, report};
