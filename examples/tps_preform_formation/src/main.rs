//! Educational, uncalibrated TPS-preform manufacturing recipe.
//!
//! The important idea is the schedule, not the chosen statistics: generate
//! capacity once, then insert, relax, move, insert, and relax while the fiber
//! world remains resident on the GPU.

use std::f64::consts::PI;

use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, JunctionParameterId, PeriodicCell, Section};
use tangle_example_support::{debug_ovito_requested, ExampleOutput};
use tangle_export::{
    BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{
    AdaptiveCompactionIncrement, CompactionConfig, CompactionGuards, CompactionPath,
    CompactionTarget, FiberInsertionPopulation, FiberPopulationSpec, FormationOperation,
    FormationRecipeConfig, FormationRecipePlugin, FormationRecipeState, JunctionCapturePolicy,
    JunctionMaterialPair, OrientationDistribution, PositionDistribution, ScalarDistribution,
    StagedFiberPopulationGeneratorPlugin,
};
use tangle_relax::{
    CompactionEnergyModel, CompactionKinematics, RelaxationConfig, RelaxationPlugin,
    RelaxationState,
};

const SNAPSHOT_INTERVAL: usize = 20;
// Six initially well-separated layer centers are about 1/6 cell apart. The
// mean fiber diameter is 0.015 cell units, so 0.09 brings neighboring layer
// centers to approximately one fiber diameter: 0.09 * (1/6) = 0.015.
const TOUCHING_LAYER_SPACING_SCALE: f32 = 0.09;

fn main() {
    let debug_ovito = debug_ovito_requested();
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), "");

    let mut app = App::new();

    // 1. Generate two populations into reserved GPU capacity. Only formation
    //    step zero is initially active; step one is our later "insertion".
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0; 3], [false; 3],
        )))
        .add_plugins(StagedFiberPopulationGeneratorPlugin {
            populations: vec![
                FiberInsertionPopulation {
                    spec: in_plane_layers(),
                    formation_step: 0,
                },
                FiberInsertionPopulation {
                    spec: through_thickness_fibers(),
                    formation_step: 1,
                },
            ],
        });

    // 2. The relaxation plugin owns one persistent CubeCL world. Recipe
    //    commands only change small control values or launch resident kernels.
    app.add_plugins(RelaxationPlugin {
        config: RelaxationConfig::flexible()
            // Keep the solver cadence fixed whether or not the optional
            // 20-iteration OVITO readback is enabled. Debugging then observes
            // exactly the same formation trajectory.
            .with_iteration_limits(1_500, SNAPSHOT_INTERVAL)
            .with_debug_snapshots(debug_ovito.then_some(SNAPSHOT_INTERVAL)),
    });

    // 3. A manufacturing recipe is ordinary data. Reorder, repeat, remove, or
    //    replace these commands to model another hypothetical process.
    app.add_plugins(FormationRecipePlugin {
        config: FormationRecipeConfig {
            layer_axis: 2,
            operations: vec![
                FormationOperation::ActivateFibersThrough(0),
                FormationOperation::RelaxFor(80),
                FormationOperation::MoveLayers {
                    spacing_scale: 0.60,
                    stiffness: 1.0,
                    max_translation: 0.08,
                },
                FormationOperation::RelaxFor(60),
                FormationOperation::MoveLayers {
                    spacing_scale: 0.30,
                    stiffness: 1.0,
                    max_translation: 0.08,
                },
                FormationOperation::RelaxFor(60),
                FormationOperation::MoveLayers {
                    spacing_scale: 0.16,
                    stiffness: 1.0,
                    max_translation: 0.08,
                },
                FormationOperation::RelaxFor(80),
                FormationOperation::MoveLayers {
                    spacing_scale: TOUCHING_LAYER_SPACING_SCALE,
                    stiffness: 1.0,
                    max_translation: 0.08,
                },
                FormationOperation::RelaxFor(120),
                FormationOperation::CaptureJunctions(planar_bonding()),
                FormationOperation::ActivateFibersThrough(1),
                FormationOperation::RelaxAndCapture {
                    iterations: 160,
                    every: 40,
                    policy: through_thickness_bonding(),
                },
                FormationOperation::ReleaseLayerTargets,
                FormationOperation::Compact(final_thickness_compaction()),
            ],
        },
    });

    // 4. Debug output is optional. The prepacked host frame is omitted so the
    //    first OVITO frame contains only fibers actually inserted at step zero.
    if debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: output
                .ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::CurvatureRatio)
                .with_initial_frame(false),
        });
    }

    // 5. Export sees the final active assembly, independent of how it formed.
    app.add_plugins(BpmExportPlugin {
        config: output.sphere_bpm(1_800.0),
    });
    app.start();

    report(&app);
}

fn planar_bonding() -> JunctionCapturePolicy {
    JunctionCapturePolicy {
        name: "planar consolidation".to_string(),
        law_name: "felt contact bond".to_string(),
        parameters: JunctionParameterId(0),
        maximum_surface_gap: 2.0e-4,
        minimum_crossing_angle: 12.0_f32.to_radians(),
        probability: 0.35,
        seed: 20_260_921,
        material_pairs: vec![JunctionMaterialPair::new(
            "in-plane felt fiber",
            "in-plane felt fiber",
        )],
        ..JunctionCapturePolicy::touching("unused", "unused")
    }
}

fn through_thickness_bonding() -> JunctionCapturePolicy {
    JunctionCapturePolicy {
        name: "through-thickness tying".to_string(),
        law_name: "through-thickness tie".to_string(),
        parameters: JunctionParameterId(1),
        maximum_surface_gap: 2.0e-4,
        minimum_crossing_angle: 20.0_f32.to_radians(),
        probability: 0.8,
        seed: 20_260_922,
        material_pairs: vec![JunctionMaterialPair::new(
            "in-plane felt fiber",
            "through-thickness fiber",
        )],
        ..JunctionCapturePolicy::touching("unused", "unused")
    }
}

fn final_thickness_compaction() -> CompactionConfig {
    CompactionConfig {
        target: CompactionTarget::NominalVolumeFraction(0.01),
        path: CompactionPath::AxisWeights([0.0, 0.0, 1.0]),
        kinematics: CompactionKinematics::MovingWalls,
        cell_anchor: [0.5; 3],
        balance_opposing_faces: false,
        face_pressure_floor: 1.0e-12,
        face_balance_strength: 0.5,
        increment: AdaptiveCompactionIncrement {
            initial_log_strain: 0.03,
            minimum_log_strain: 0.002,
            maximum_log_strain: 0.06,
            growth_factor: 1.25,
            shrink_factor: 0.5,
            relax_iterations: 40,
            maximum_shortening_over_minimum_diameter: 0.5,
        },
        guards: CompactionGuards {
            maximum_penetration: 1.5e-4,
            maximum_bend_ratio: 1.001,
            maximum_steps: 32,
            ..CompactionGuards::default()
        },
        energy_model: CompactionEnergyModel {
            contact_stiffness: 1.0,
            stretch_stiffness: 1.0,
            bending_stiffness: 1.0,
        },
        target_tolerance: 1.0e-3,
    }
}

fn in_plane_layers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        count: 96,
        segments_per_fiber: 6,
        seed: 20_260_919,
        length: ScalarDistribution::Uniform {
            minimum: 0.28,
            maximum: 0.42,
        },
        radius: ScalarDistribution::Uniform {
            minimum: 0.006,
            maximum: 0.009,
        },
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 0.008,
        },
        orientation: OrientationDistribution::Planar {
            normal: [0.0, 0.0, 1.0],
            maximum_tilt: 8.0 * PI / 180.0,
        },
        position: PositionDistribution::Layered {
            axis: 2,
            layers: 6,
            jitter_fraction: 0.18,
        },
        minimum_bend_radius: Some(0.06),
        max_attempts_per_fiber: 512,
        material_name: "in-plane felt fiber".to_string(),
        ..FiberPopulationSpec::default()
    }
}

fn through_thickness_fibers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        count: 24,
        segments_per_fiber: 6,
        seed: 20_260_920,
        length: ScalarDistribution::Uniform {
            minimum: 0.50,
            maximum: 0.72,
        },
        radius: ScalarDistribution::Uniform {
            minimum: 0.005,
            maximum: 0.007,
        },
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 0.006,
        },
        orientation: OrientationDistribution::Aligned {
            axis: [0.0, 0.0, 1.0],
            maximum_angle: 10.0 * PI / 180.0,
        },
        position: PositionDistribution::Uniform,
        minimum_bend_radius: Some(0.06),
        max_attempts_per_fiber: 512,
        material_name: "through-thickness fiber".to_string(),
        ..FiberPopulationSpec::default()
    }
}

fn report(app: &App) {
    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .expect("RelaxationPlugin should install RelaxationState");
    assert!(
        relaxation.converged,
        "TPS preform formation did not converge: {relaxation:?}"
    );
    let recipe = app
        .get_resource_ref::<FormationRecipeState>()
        .expect("FormationRecipePlugin should install FormationRecipeState");
    let export = app
        .get_resource_ref::<BpmExportReport>()
        .expect("BpmExportPlugin should install BpmExportReport");
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .expect("TangleAssemblyPlugin should install FiberAssembly");

    println!("TPS preform formation recipe:");
    for event in &recipe.events {
        println!("  {:>4}: {}", event.iteration, event.description);
    }
    println!(
        "  converged at iteration {}; penetration {:.3e}; bend ratio {:.3}",
        relaxation.iterations, relaxation.max_penetration, relaxation.max_curvature_ratio
    );
    println!(
        "  persistent junctions: {} across {} law types",
        assembly.junctions.junctions.len(),
        assembly.junction_laws.entries.len()
    );
    for compaction in &recipe.compactions {
        println!(
            "  compaction: {:?}; Vf {:.4}; cell [{:.3}, {:.3}, {:.3}]; pressure [{:.3e}, {:.3e}, {:.3e}]; work {:.3e}",
            compaction.reason,
            compaction.nominal_volume_fraction,
            compaction.cell_lengths[0],
            compaction.cell_lengths[1],
            compaction.cell_lengths[2],
            compaction.metrics.pressure[0],
            compaction.metrics.pressure[1],
            compaction.metrics.pressure[2],
            compaction.cumulative_work.iter().sum::<f32>()
        );
    }
    let (mean_spacing, mean_diameter) = planar_layer_spacing(&assembly);
    println!(
        "  planar layer spacing: {mean_spacing:.4} ({:.2} mean fiber diameters)",
        mean_spacing / mean_diameter
    );
    println!(
        "  exported {} particles and {} bonds ({} inter-fiber junction bonds) to {}",
        export.particles,
        export.bonds,
        export.junction_bonds,
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

fn planar_layer_spacing(assembly: &FiberAssembly) -> (f64, f64) {
    let layer_count = assembly
        .topology
        .fibers
        .iter()
        .filter_map(|fiber| fiber.formation_layer)
        .max()
        .map_or(0, |layer| layer as usize + 1);
    let mut layer_sums = vec![0.0; layer_count];
    let mut layer_fibers = vec![0_usize; layer_count];
    let mut diameter_sum = 0.0;
    let mut diameter_count = 0_usize;

    for fiber in assembly
        .topology
        .fibers
        .iter()
        .filter(|fiber| fiber.formation_layer.is_some())
    {
        let start = fiber.vertices.start as usize;
        let end = start + fiber.vertices.len as usize;
        let center = assembly.geometry.placed.positions[start..end]
            .iter()
            .map(|position| position[2])
            .sum::<f64>()
            / fiber.vertices.len as f64;
        let layer = fiber.formation_layer.unwrap() as usize;
        layer_sums[layer] += center;
        layer_fibers[layer] += 1;
        diameter_sum += match assembly.sections.entries[fiber.section.0 as usize] {
            Section::Circular { radius } => 2.0 * radius,
            Section::Elliptical { semi_axes } => semi_axes[0] + semi_axes[1],
        };
        diameter_count += 1;
    }

    let layer_centers: Vec<_> = layer_sums
        .into_iter()
        .zip(layer_fibers)
        .filter(|(_, count)| *count > 0)
        .map(|(sum, count)| sum / count as f64)
        .collect();
    let mean_spacing = layer_centers
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .sum::<f64>()
        / (layer_centers.len() - 1) as f64;
    (mean_spacing, diameter_sum / diameter_count as f64)
}
