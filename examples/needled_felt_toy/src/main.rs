//! Educational layer-by-layer needled-felt formation recipe.
//!
//! All layers are prepacked once, then activated, placed, needled, and relaxed
//! through small commands while the fiber geometry remains on the GPU.

use std::f64::consts::PI;

use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, PeriodicCell};
use tangle_example_support::{debug_ovito_requested, ExampleOutput};
use tangle_export::{
    BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{
    AdaptiveCompactionIncrement, CompactionConfig, CompactionGuards, CompactionPath,
    CompactionTarget, FiberPopulationSpec, FormationOperation, FormationRecipeConfig,
    FormationRecipePlugin, FormationRecipeState, LayerStagedFiberPopulationGeneratorPlugin,
    NeedlingConfig, NeedlingSelection, OrientationDistribution, PositionDistribution,
    ScalarDistribution,
};
use tangle_relax::{
    CompactionEnergyModel, CompactionKinematics, RelaxationConfig, RelaxationPlugin,
    RelaxationState,
};

const LAYER_COUNT: u32 = 6;
const FIBER_COUNT: usize = 180;
const SEGMENTS_PER_FIBER: usize = 10;
const LAYER_GAP: f32 = 0.040;
const NEEDLE_DEPTH: f32 = 2.0 * LAYER_GAP;
const GPU_BATCH_ITERATIONS: usize = 20;
const SNAPSHOT_INTERVAL: usize = 200;

fn main() {
    let debug_ovito = debug_ovito_requested();
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), "");

    let mut app = App::new();
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0, 1.0, 0.32],
            [false; 3],
        )))
        // One generated population carries six deposition-layer labels. The
        // labels become activation steps, so future layers are physically
        // absent even though their capacity is already resident on the GPU.
        .add_plugins(LayerStagedFiberPopulationGeneratorPlugin {
            spec: planar_layers(),
            first_formation_step: 0,
        })
        .add_plugins(RelaxationPlugin {
            config: RelaxationConfig {
                // Dense needled states benefit from additional rod-constraint
                // passes between contact projections.
                constraint_iterations: 4,
                ..RelaxationConfig::flexible()
                    // Solver batches stay small while OVITO readbacks are
                    // deliberately much sparser.
                    .with_iteration_limits(24_000, GPU_BATCH_ITERATIONS)
                    .with_penetration_tolerance(5.0e-4)
                    .with_max_step(0.005)
                    .with_debug_snapshots(debug_ovito.then_some(SNAPSHOT_INTERVAL))
            },
        })
        .add_plugins(FormationRecipePlugin {
            config: FormationRecipeConfig {
                layer_axis: 2,
                operations: needled_recipe(),
            },
        });

    if debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: output
                .ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::CurvatureRatio)
                .with_initial_frame(false),
        });
    }
    app.add_plugins(BpmExportPlugin {
        config: output.sphere_bpm(1_800.0),
    });
    app.start();

    report(&app);
}

fn needled_recipe() -> Vec<FormationOperation> {
    let mut operations = vec![
        // Begin with the first two layers present, then lower the upper one
        // onto the base layer before the repeated deposition cycle.
        FormationOperation::ActivateFibersThrough(1),
        FormationOperation::RelaxFor(80),
        place_layer(1),
        FormationOperation::RelaxFor(120),
    ];

    for layer in 2..LAYER_COUNT {
        operations.extend([
            FormationOperation::ActivateFibersThrough(layer),
            place_layer(layer),
            FormationOperation::RelaxFor(100),
            // Once deposited, release every layer-center tether. The needle
            // then loads a mechanically free stack, so contact transmits its
            // motion into fibers in the layers below.
            FormationOperation::ReleaseLayerTargets,
            FormationOperation::NeedleLayer(NeedlingConfig {
                layer,
                selection: NeedlingSelection::RandomFiberFraction {
                    fraction: 0.30,
                    seed: 20_260_930 + layer as u64,
                },
                minimum_fiber_diameter: None,
                depth: NEEDLE_DEPTH,
                stiffness: 1.0,
                max_translation: 0.012,
                maximum_translation_over_fiber_diameter: 0.25,
            }),
            // Only the selected vertex targets are prescribed here; all other
            // vertices and fibers respond through contact and rod constraints.
            FormationOperation::RelaxFor(160),
            FormationOperation::ReleaseNeedles,
            FormationOperation::RelaxFor(40),
        ]);
    }

    operations.extend([
        FormationOperation::ReleaseNeedles,
        FormationOperation::ReleaseLayerTargets,
        FormationOperation::Compact(final_compaction()),
    ]);
    operations
}

fn place_layer(layer: u32) -> FormationOperation {
    FormationOperation::PlaceLayerAbove {
        layer,
        gap: LAYER_GAP,
        stiffness: 1.0,
        max_translation: 0.08,
    }
}

fn planar_layers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        count: FIBER_COUNT,
        segments_per_fiber: SEGMENTS_PER_FIBER,
        seed: 20_260_929,
        length: ScalarDistribution::Uniform {
            minimum: 0.36,
            maximum: 0.48,
        },
        radius: ScalarDistribution::Uniform {
            minimum: 0.016,
            maximum: 0.021,
        },
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 0.006,
        },
        orientation: OrientationDistribution::Planar {
            normal: [0.0, 0.0, 1.0],
            maximum_tilt: 6.0 * PI / 180.0,
        },
        position: PositionDistribution::Layered {
            axis: 2,
            layers: LAYER_COUNT as usize,
            jitter_fraction: 0.12,
        },
        minimum_bend_radius: Some(0.050),
        max_attempts_per_fiber: 1_024,
        material_name: "needled felt fiber".to_string(),
        ..FiberPopulationSpec::default()
    }
}

fn final_compaction() -> CompactionConfig {
    CompactionConfig {
        target: CompactionTarget::NominalVolumeFraction(0.40),
        path: CompactionPath::AxisWeights([0.0, 0.0, 1.0]),
        kinematics: CompactionKinematics::MovingWalls,
        // Keep the deposition surface fixed and lower the top platen.
        cell_anchor: [0.0, 0.0, 0.0],
        balance_opposing_faces: false,
        face_pressure_floor: 1.0e-12,
        face_balance_strength: 0.5,
        increment: AdaptiveCompactionIncrement {
            initial_log_strain: 0.02,
            minimum_log_strain: 0.005,
            maximum_log_strain: 0.04,
            growth_factor: 1.2,
            shrink_factor: 0.5,
            relax_iterations: 60,
            maximum_shortening_over_minimum_diameter: 0.5,
        },
        guards: CompactionGuards {
            // This is a catastrophic-step guard, while the tighter solver
            // tolerance above controls every accepted relaxed state.
            maximum_penetration: 1.0e-3,
            maximum_bend_ratio: 1.001,
            maximum_steps: 128,
            maximum_relax_windows: 20,
            ..CompactionGuards::default()
        },
        energy_model: CompactionEnergyModel::default(),
        target_tolerance: 1.0e-3,
    }
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
    let export = app
        .get_resource_ref::<BpmExportReport>()
        .expect("BpmExportPlugin should install BpmExportReport");

    println!("needled felt toy formation recipe:");
    for event in &recipe.events {
        println!("  {:>4}: {}", event.iteration, event.description);
    }
    for needling in &recipe.needling {
        println!(
            "  layer {}: pulled {} vertices through {:.3} cell units",
            needling.layer, needling.selected_vertices, needling.depth
        );
    }
    for compaction in &recipe.compactions {
        println!(
            "  compaction: {:?}; Vf {:.3}; cell [{:.3}, {:.3}, {:.3}]; pressure [{:.3e}, {:.3e}, {:.3e}]",
            compaction.reason,
            compaction.nominal_volume_fraction,
            compaction.cell_lengths[0],
            compaction.cell_lengths[1],
            compaction.cell_lengths[2],
            compaction.metrics.pressure[0],
            compaction.metrics.pressure[1],
            compaction.metrics.pressure[2],
        );
    }
    if recipe.compactions.is_empty() {
        if let Some(step) = recipe.compaction_steps.last() {
            println!(
                "  compaction incomplete after step {}; Vf {:.3}; cell [{:.3}, {:.3}, {:.3}]",
                step.step,
                step.nominal_volume_fraction,
                step.cell_lengths[0],
                step.cell_lengths[1],
                step.cell_lengths[2],
            );
        }
    }
    println!(
        "  final: {} fibers, {} iterations, penetration {:.3e}, bend ratio {:.3}",
        assembly.topology.fibers.len(),
        relaxation.iterations,
        relaxation.max_penetration,
        relaxation.max_curvature_ratio
    );
    println!(
        "  DEM-BPM: {} particles and {} bonds at {}",
        export.particles,
        export.bonds,
        export.data_path.display()
    );
    if let Some(trajectory) = app.get_resource_ref::<OvitoTrajectoryReport>() {
        println!(
            "  OVITO trajectory: {} ({} frames)",
            trajectory.dump_path.display(),
            trajectory.frames
        );
    }
}
