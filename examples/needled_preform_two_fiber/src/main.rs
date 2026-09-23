//! Physical-scale two-material needled preform experiment.
//!
//! Lengths are SI meters: a 2.5 mm square periodic footprint contains 7 µm
//! relatively bend-resistant fibers and 19 µm fibers that bend more readily.

use std::{f64::consts::PI, io::Write};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_checkpoint::{CheckpointConfig, CheckpointPlugin, CheckpointReport};
use tangle_core::{FiberAssembly, PeriodicCell};
use tangle_example_support::ExampleOutput;
use tangle_export::{
    write_ovito_assembly_frame, write_ovito_view_script, BpmExportPlugin, BpmExportReport,
    OvitoColoring, OvitoTrajectoryConfig, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{
    random_footprint_center, AcceptanceLimit, AdaptiveCompactionIncrement, CompactionConfig,
    CompactionGuards, CompactionPath, CompactionTarget, FiberPopulationSpec, FormationOperation,
    FormationRecipeConfig, FormationRecipePlugin, FormationRecipeState,
    MixedLayerStagedFiberPopulationGeneratorPlugin, NeedlingConfig, NeedlingSelection,
    OrientationDistribution, PositionDistribution, RelaxationAcceptance, RelaxationTargets,
    ScalarDistribution, SolveExhaustion, SolvePolicy,
};
use tangle_relax::{
    AdaptiveSegmentationConfig, CompactionEnergyModel, CompactionKinematics, RelaxationConfig,
    RelaxationPlugin, RelaxationState,
};

const FOOTPRINT: f64 = 0.002_5;
// The temporary staging height keeps six plies disjoint and leaves the bottom
// ply above the 350 um needle-pull destination.
const INITIAL_THICKNESS: f64 = 0.004_800;
const LAYER_COUNT: u32 = 6;
// One sixteenth of the old 10 mm-square population preserves fibers per unit
// area while making the detailed formation trajectory practical to inspect.
const FIBERS_PER_TYPE: usize = 188;
// Preserve a coarse intrinsic shape; contact-driven midpoint subdivision adds
// local resolution on the GPU only where the solver needs it.
const SEGMENTS_PER_FIBER: usize = 4;
const LAYER_GAP: f32 = 50.0e-6;
const LAYER_APPROACH_GAPS: [f32; 2] = [65.0e-6, LAYER_GAP];
const FIRST_NEEDLED_LAYER: u32 = 2;
const NEEDLE_DEPTH: f32 = 7.0 * LAYER_GAP;
const NEEDLE_DIAMETER: f32 = 150.0e-6;
const NEEDLE_POSITION_SEED: u64 = 20_260_940;
const LAYER_ORIENTATION_SEED: u64 = 20_260_941;
const GPU_BATCH_ITERATIONS: usize = 10;
const DEFAULT_SNAPSHOT_INTERVAL: usize = 100;
const FORMATION_STEPS_ONLY: usize = usize::MAX;
const PROGRESS_INTERVAL: usize = 250;
const CHECKPOINT_INTERVAL: usize = 500;
const FORMATION_RELAXATION_BUDGET: usize = 15_000;
const FINAL_RELAXATION_BUDGET: usize = 100_000;
const CONTACT_ACCEPTANCE: f32 = 0.30e-6;
const SOLVER_BEND_TARGET: f32 = 1.000_01;
const FORMATION_BEND_ACCEPTANCE: f32 = 1.05;
const FINAL_BEND_ACCEPTANCE: f32 = 1.001;
const TARGET_ADVANCE_BUDGET: usize = 3_000;
const TARGET_POSITION_TOLERANCE: f32 = 0.5e-6;
const LAYER_TARGET_HOLD_ITERATIONS: usize = 100;
const NEEDLE_TARGET_HOLD_ITERATIONS: usize = 150;

#[derive(Default)]
struct ConsoleProgress {
    started: bool,
    events_printed: usize,
    compaction_steps_printed: usize,
    next_iteration: usize,
}

struct ConsoleProgressPlugin;

impl Plugin for ConsoleProgressPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(ConsoleProgress::default())
            .add_update_system(
                print_progress.run_if(in_state(TangleStage::Relax)),
                TanglePhase::Observe,
            );
    }
}

fn main() {
    let ovito_interval = requested_ovito_interval();
    let debug_ovito = ovito_interval.is_some();
    let export_dem = std::env::args().any(|argument| argument == "--export-dem");
    let final_ovito = std::env::args().any(|argument| argument == "--final-ovito");
    let checkpoint = std::env::args().any(|argument| argument == "--checkpoint");
    let resume = std::env::args().any(|argument| argument == "--resume");
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), "");
    let checkpoint_path = output.directory.join("needled_preform_two_fiber.restart");

    let mut app = App::new();
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [FOOTPRINT, FOOTPRINT, INITIAL_THICKNESS],
            [true, true, false],
        )))
        .add_plugins(MixedLayerStagedFiberPopulationGeneratorPlugin {
            specs: vec![thin_stiff_fibers(), thick_bendy_fibers()],
            first_formation_step: 0,
            minimum_layer_clearance: Some(50.0e-6),
        })
        .add_plugins(RelaxationPlugin {
            config: RelaxationConfig {
                constraint_iterations: 8,
                // Leave numerical room for contact and natural-shape
                // corrections to rebound while accepting only ratios <= 1.0.
                curvature_limit_safety_margin: 0.06,
                curvature_cleanup_sweeps: 4,
                ..RelaxationConfig::flexible()
                    .with_iteration_limits(500_000, GPU_BATCH_ITERATIONS)
                    .with_penetration_tolerance(0.30e-6)
                    .with_contact_correction(0.35)
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
                    .with_next_stage(if export_dem {
                        TangleStage::Export
                    } else {
                        TangleStage::Done
                    })
                    .with_debug_snapshots(ovito_interval)
            },
        })
        .add_plugins(FormationRecipePlugin {
            config: FormationRecipeConfig {
                layer_axis: 2,
                operations: needled_recipe(),
            },
        })
        .add_plugins(ConsoleProgressPlugin);

    if checkpoint || resume {
        app.add_plugins(CheckpointPlugin {
            config: CheckpointConfig::new(
                "needled_preform_two_fiber-v19-staged-solve-policy",
                &checkpoint_path,
                CHECKPOINT_INTERVAL,
            )
            .with_resume(resume),
        });
    }

    if debug_ovito {
        let trajectory = output
            .ovito_segments(ovito_interval.unwrap(), OvitoColoring::CurvatureRatio)
            .with_initial_frame(false);
        app.add_plugins(OvitoTrajectoryPlugin { config: trajectory });
    }
    // A full bead-chain export contains millions of particles at these aspect
    // ratios, so it is explicit rather than part of the normal teaching run.
    if export_dem {
        app.add_plugins(BpmExportPlugin {
            config: output.sphere_bpm(1_800.0),
        });
    }
    app.start();

    report(&app);
    if final_ovito && !debug_ovito {
        write_final_ovito(&app, &output);
    }
}

fn requested_ovito_interval() -> Option<usize> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == "--debug-ovito-every-step")
    {
        println!("OVITO cadence: formation steps and accepted compaction steps only");
        return Some(FORMATION_STEPS_ONLY);
    }
    if arguments
        .iter()
        .any(|argument| argument == "--debug-ovito-every-iteration")
    {
        eprintln!(
            "warning: OVITO output every solver iteration forces a full geometry readback and can be extremely large"
        );
        return Some(1);
    }
    if let Some(value) = arguments
        .iter()
        .find_map(|argument| argument.strip_prefix("--debug-ovito-interval="))
    {
        let interval = value
            .parse::<usize>()
            .expect("--debug-ovito-interval must be a positive integer");
        assert!(interval > 0, "--debug-ovito-interval must be positive");
        println!("OVITO cadence: formation steps plus every {interval} solver iterations");
        return Some(interval);
    }
    arguments
        .iter()
        .any(|argument| argument == "--debug-ovito")
        .then_some(DEFAULT_SNAPSHOT_INTERVAL)
}

fn needled_recipe() -> Vec<FormationOperation> {
    let mut operations = vec![
        FormationOperation::ActivateFibersThrough(0),
        relax_until_formation_ready(),
    ];
    for layer in 1..LAYER_COUNT {
        operations.push(FormationOperation::ActivateFibersThrough(layer));
        // Resolve crossings inside the isolated staging ply before it is
        // lowered into contact with the previously deposited stack.
        operations.push(relax_until_formation_ready());
        for gap in LAYER_APPROACH_GAPS {
            operations.extend([
                place_layer(layer, gap),
                relax_until_targets_reached(),
                // The layer-center target acts as a temporary platen. Give
                // contacts time to respond, then remove the kinematic clamp
                // before requiring hard contact clearance and the interim
                // formation bend limit in a free configuration.
                FormationOperation::RelaxFor(LAYER_TARGET_HOLD_ITERATIONS),
                FormationOperation::ReleaseLayerTargets,
                relax_until_formation_ready(),
            ]);
        }
        // Begin needling only after three batting plies (0, 1, and 2) have
        // been deposited. The extra depth passes below the bottom ply into
        // the deliberately empty staging space.
        if layer >= FIRST_NEEDLED_LAYER {
            operations.extend([
                FormationOperation::NeedleLayer(NeedlingConfig {
                    layer,
                    selection: NeedlingSelection::CircularFootprint {
                        center: random_footprint_center(
                            NEEDLE_POSITION_SEED,
                            layer,
                            [0.0, 0.0],
                            [FOOTPRINT as f32, FOOTPRINT as f32],
                        ),
                        diameter: NEEDLE_DIAMETER,
                    },
                    minimum_fiber_diameter: Some(0.99 * 19.0e-6),
                    depth: NEEDLE_DEPTH,
                    stiffness: 0.5,
                    // A 1 um step remains well below the 7 um obstacle
                    // diameter while halving the nominal pull duration.
                    max_translation: 1.0e-6,
                    maximum_translation_over_fiber_diameter: 0.25,
                }),
                // End the pull from measured target error rather than a
                // conservative fixed-duration travel estimate. A shorter,
                // explicit dwell follows before the needle is released.
                relax_until_targets_reached(),
                FormationOperation::RelaxFor(NEEDLE_TARGET_HOLD_ITERATIONS),
                FormationOperation::ReleaseNeedles,
                relax_until_formation_ready(),
            ]);
        }
    }
    operations.extend([
        FormationOperation::ReleaseNeedles,
        FormationOperation::ReleaseLayerTargets,
        relax_until_final_admissible(),
        FormationOperation::Compact(final_compaction()),
        relax_until_final_admissible(),
    ]);
    operations
}

fn relax_until_formation_ready() -> FormationOperation {
    FormationOperation::RelaxWithPolicy(SolvePolicy {
        name: "formation relaxation".to_string(),
        solver_targets: strict_solver_targets(),
        acceptance: RelaxationAcceptance {
            penetration: AcceptanceLimit::hard(CONTACT_ACCEPTANCE),
            curvature_ratio: AcceptanceLimit::soft(FORMATION_BEND_ACCEPTANCE),
        },
        maximum_iterations: FORMATION_RELAXATION_BUDGET,
        extra_iterations: SolvePolicy::default_extra_iterations(FORMATION_RELAXATION_BUDGET),
        on_exhaustion: SolveExhaustion::ContinueIfHardLimitsSatisfied,
    })
}

fn relax_until_final_admissible() -> FormationOperation {
    FormationOperation::RelaxWithPolicy(SolvePolicy {
        name: "final hard relaxation".to_string(),
        solver_targets: strict_solver_targets(),
        acceptance: RelaxationAcceptance {
            penetration: AcceptanceLimit::hard(CONTACT_ACCEPTANCE),
            curvature_ratio: AcceptanceLimit::hard(FINAL_BEND_ACCEPTANCE),
        },
        maximum_iterations: FINAL_RELAXATION_BUDGET,
        extra_iterations: SolvePolicy::default_extra_iterations(FINAL_RELAXATION_BUDGET),
        on_exhaustion: SolveExhaustion::Reject,
    })
}

fn strict_solver_targets() -> RelaxationTargets {
    RelaxationTargets {
        penetration: CONTACT_ACCEPTANCE,
        curvature_ratio: SOLVER_BEND_TARGET,
    }
}

fn relax_until_targets_reached() -> FormationOperation {
    FormationOperation::RelaxUntilTargetsReached {
        tolerance: TARGET_POSITION_TOLERANCE,
        maximum_iterations: TARGET_ADVANCE_BUDGET,
    }
}

fn place_layer(layer: u32, gap: f32) -> FormationOperation {
    FormationOperation::PlaceLayerAbove {
        layer,
        gap,
        stiffness: 1.0,
        // Move quasi-statically so contact resolution can keep pace with the
        // descending batting layer.
        max_translation: 2.0e-6,
    }
}

fn common_population(material_name: &str, seed: u64) -> FiberPopulationSpec {
    FiberPopulationSpec {
        count: FIBERS_PER_TYPE,
        segments_per_fiber: SEGMENTS_PER_FIBER,
        seed,
        nominal_parent_length: Some(0.050_8),
        orientation: OrientationDistribution::LayeredBiaxial {
            normal: [0.0, 0.0, 1.0],
            primary_fraction: 0.40,
            cross_fraction: 0.40,
            maximum_in_plane_deviation: PI / 18.0,
            maximum_tilt: PI / 1_800.0,
            layer_seed: LAYER_ORIENTATION_SEED,
        },
        position: PositionDistribution::Layered {
            axis: 2,
            layers: LAYER_COUNT as usize,
            // Give each deposited ply finite thickness. A zero-thickness
            // plane creates hundreds of exact centerline crossings that no
            // physical deposition process would begin with.
            jitter_fraction: 0.25,
        },
        max_attempts_per_fiber: 2_048,
        material_name: material_name.to_string(),
        ..FiberPopulationSpec::default()
    }
}

fn thin_stiff_fibers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        length: ScalarDistribution::Uniform {
            minimum: 0.003,
            maximum: 0.004,
        },
        radius: ScalarDistribution::Constant(3.5e-6),
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 2.0e-6,
        },
        // Roughly 70 fiber diameters: deliberately bend-resistant.
        minimum_bend_radius: Some(0.000_500),
        ..common_population("fine_7um", 20_260_938)
    }
}

fn thick_bendy_fibers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        length: ScalarDistribution::Uniform {
            minimum: 0.003,
            maximum: 0.004,
        },
        radius: ScalarDistribution::Constant(9.5e-6),
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 8.0e-6,
        },
        // About three diameters: permits tight local turns during needling.
        minimum_bend_radius: Some(60.0e-6),
        ..common_population("coarse_19um", 20_260_939)
    }
}

fn final_compaction() -> CompactionConfig {
    CompactionConfig {
        target: CompactionTarget::NominalVolumeFraction(0.13),
        path: CompactionPath::AxisWeights([0.0, 0.0, 1.0]),
        kinematics: CompactionKinematics::MovingWalls,
        cell_anchor: [0.0, 0.0, 0.0],
        balance_opposing_faces: false,
        face_pressure_floor: 1.0e-12,
        face_balance_strength: 0.5,
        increment: AdaptiveCompactionIncrement {
            initial_log_strain: 0.01,
            minimum_log_strain: 0.001,
            maximum_log_strain: 0.02,
            growth_factor: 1.2,
            shrink_factor: 0.5,
            relax_iterations: 1_000,
            maximum_shortening_over_minimum_diameter: 0.5,
        },
        guards: CompactionGuards {
            maximum_penetration: 0.301e-6,
            maximum_bend_ratio: 1.001,
            maximum_steps: 256,
            maximum_relax_windows: 8,
            ..CompactionGuards::default()
        },
        energy_model: CompactionEnergyModel::default(),
        target_tolerance: 1.0e-3,
    }
}

fn print_progress(
    config: Res<FormationRecipeConfig>,
    recipe: Res<FormationRecipeState>,
    relaxation: Res<RelaxationState>,
    checkpoint: Option<Res<CheckpointReport>>,
    mut progress: ResMut<ConsoleProgress>,
) {
    if !progress.started {
        if !checkpoint.as_ref().is_some_and(|report| report.resumed) {
            println!(
                "starting {} formation operations; progress every {} relaxation iterations",
                config.operations.len(),
                PROGRESS_INTERVAL
            );
        } else {
            println!(
                "continuing operation {}/{} from iteration {}",
                recipe
                    .next_operation
                    .saturating_add(1)
                    .min(config.operations.len()),
                config.operations.len(),
                relaxation.iterations
            );
            progress.events_printed = recipe.events.len();
            progress.compaction_steps_printed = recipe.compaction_steps.len();
        }
        progress.started = true;
        progress.next_iteration =
            (relaxation.iterations / PROGRESS_INTERVAL + 1) * PROGRESS_INTERVAL;
    }

    for event in &recipe.events[progress.events_printed..] {
        println!(
            "  operation {:>2}/{} complete at iteration {:>6}: {}",
            event.operation + 1,
            config.operations.len(),
            event.iteration,
            event.description
        );
    }
    progress.events_printed = recipe.events.len();

    for step in &recipe.compaction_steps[progress.compaction_steps_printed..] {
        println!(
            "  compaction {:>3}: Vf {:.3}, thickness {:>7.1} um, penetration {:.3} um, energy {:.3e}",
            step.step,
            step.nominal_volume_fraction,
            1.0e6 * step.cell_lengths[2],
            1.0e6 * step.max_penetration,
            step.metrics.total_energy
        );
    }
    progress.compaction_steps_printed = recipe.compaction_steps.len();

    if relaxation.iterations >= progress.next_iteration {
        println!(
            "  relax {:>6}: operation {:>2}/{}, penetration {:.3} um, bend ratio {:.6}, segments {}, splits {}, merges {}",
            relaxation.iterations,
            recipe
                .next_operation
                .saturating_add(1)
                .min(config.operations.len()),
            config.operations.len(),
            1.0e6 * relaxation.max_penetration,
            relaxation.max_curvature_ratio,
            relaxation.active_segments,
            relaxation.segment_splits,
            relaxation.segment_merges
        );
        progress.next_iteration =
            (relaxation.iterations / PROGRESS_INTERVAL + 1) * PROGRESS_INTERVAL;
    }

    std::io::stdout()
        .flush()
        .expect("failed to flush progress output");
}

fn write_final_ovito(app: &App, output: &ExampleOutput) {
    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .expect("RelaxationPlugin should install RelaxationState");
    if !relaxation.converged {
        for stale in [
            output.directory.join("final_assembly.dump"),
            output.directory.join("final_assembly_view.py"),
            output.directory.join("final_assembly.ovito"),
        ] {
            if let Err(error) = std::fs::remove_file(&stale) {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::NotFound,
                    "could not remove stale rejected output {}: {error}",
                    stale.display()
                );
            }
        }
        eprintln!(
            "refusing final OVITO export: state is not relaxed (penetration {:.3} um, bend ratio {:.3})",
            1.0e6 * relaxation.max_penetration,
            relaxation.max_curvature_ratio
        );
        return;
    }
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .expect("TangleAssemblyPlugin should install FiberAssembly");
    let dump_path = output.directory.join("final_assembly.dump");
    let script_path = output.directory.join("final_assembly_view.py");
    let session_path = output.directory.join("final_assembly.ovito");
    let config = OvitoTrajectoryConfig::fiber_segments(&dump_path, 1)
        .with_coloring(OvitoColoring::Fiber)
        .with_viewing_files(&script_path, &session_path);
    write_ovito_assembly_frame(&assembly, &config, 0, false)
        .unwrap_or_else(|error| panic!("final OVITO output failed: {error}"));
    write_ovito_view_script(&config, &script_path)
        .unwrap_or_else(|error| panic!("final OVITO viewing recipe failed: {error}"));
    println!("  final OVITO frame: {}", dump_path.display());
}

fn report(app: &App) {
    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .expect("RelaxationPlugin should install RelaxationState");
    let recipe = app
        .get_resource_ref::<FormationRecipeState>()
        .expect("FormationRecipePlugin should install FormationRecipeState");
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .expect("TangleAssemblyPlugin should install FiberAssembly");

    println!("two-fiber needled preform physical-scale recipe:");
    println!(
        "  footprint: {:.1} mm x {:.1} mm; x/y periodic",
        1.0e3 * assembly.cell.basis[0][0],
        1.0e3 * assembly.cell.basis[1][1]
    );
    for needling in &recipe.needling {
        match needling.selection {
            NeedlingSelection::CircularFootprint { center, diameter } => println!(
                "  layer {}: {:.1} um needle at ({:.1}, {:.1}) um pulled {} vertices by {:.1} um",
                needling.layer,
                1.0e6 * diameter,
                1.0e6 * center[0],
                1.0e6 * center[1],
                needling.selected_vertices,
                1.0e6 * needling.depth
            ),
            NeedlingSelection::RandomFiberFraction { .. } => println!(
                "  layer {}: pulled {} vertices by {:.1} um",
                needling.layer,
                needling.selected_vertices,
                1.0e6 * needling.depth
            ),
        }
    }
    if let Some(failure) = &recipe.failure {
        println!(
            "  REJECTED at operation {} and iteration {}: {}",
            failure.operation + 1,
            failure.iteration,
            failure.reason
        );
    }
    for warning in &recipe.warnings {
        println!(
            "  WARNING at operation {} and iteration {}: {}",
            warning.operation + 1,
            warning.iteration,
            warning.reason
        );
    }
    if let Some(compaction) = recipe.compactions.last() {
        println!(
            "  compaction: {:?}; Vf {:.3}; final thickness {:.1} um",
            compaction.reason,
            compaction.nominal_volume_fraction,
            1.0e6 * compaction.cell_lengths[2]
        );
    } else if let Some(step) = recipe.compaction_steps.last() {
        println!(
            "  compaction incomplete at step {}; Vf {:.3}; thickness {:.1} um",
            step.step,
            step.nominal_volume_fraction,
            1.0e6 * step.cell_lengths[2]
        );
    }
    println!(
        "  final: {} fibers, {} active segments after {} splits/{} merges in {}/{} adaptation passes, {} iterations, penetration {:.3} um, bend ratio {:.3}",
        assembly.topology.fibers.len(),
        relaxation.active_segments,
        relaxation.segment_splits,
        relaxation.segment_merges,
        relaxation.refinement_passes,
        relaxation.coarsening_passes,
        relaxation.iterations,
        1.0e6 * relaxation.max_penetration,
        relaxation.max_curvature_ratio
    );
    if let Some(export) = app.get_resource_ref::<BpmExportReport>() {
        println!(
            "  DEM-BPM: {} particles and {} bonds at {}",
            export.particles,
            export.bonds,
            export.data_path.display()
        );
        for mapping in &export.atom_types {
            println!(
                "    atom type {}: '{}' ({} particles)",
                mapping.atom_type, mapping.material_name, mapping.particles
            );
        }
    }
    if let Some(trajectory) = app.get_resource_ref::<OvitoTrajectoryReport>() {
        println!(
            "  OVITO trajectory: {} ({} frames)",
            trajectory.dump_path.display(),
            trajectory.frames
        );
    }
    if let Some(checkpoint) = app.get_resource_ref::<CheckpointReport>() {
        if let Some(iteration) = checkpoint.last_saved_iteration {
            println!(
                "  checkpoint: iteration {} ({} saves this process)",
                iteration, checkpoint.saves
            );
        }
    }
}
