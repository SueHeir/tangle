use grass_app::prelude::*;
use multisegment_flexible_relaxation::{shape_cases, ShapeCase, MINIMUM_BEND_RADIUS};
use tangle_app::prelude::*;
use tangle_characterize::{AssemblyCharacterizationPlugin, AssemblyCharacterizationReport};
use tangle_core::PeriodicCell;
use tangle_example_support::{debug_ovito_requested, ExampleOutput};
use tangle_export::{
    BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{MultiSegmentCrossingConfig, MultiSegmentCrossingGeneratorPlugin};
use tangle_relax::{RelaxationConfig, RelaxationPlugin, RelaxationState};

const SNAPSHOT_INTERVAL: usize = 10;

fn main() {
    let debug_ovito = debug_ovito_requested();
    for case in shape_cases() {
        run_case(case, debug_ovito);
    }
}

fn run_case(case: ShapeCase, debug_ovito: bool) {
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), case.slug);
    let relaxation = RelaxationConfig::flexible()
        .with_contact_correction(1.0)
        .with_flexible_stiffness(0.35, 0.03)
        .with_curvature_limit(1.0, 1.0e-4)
        .with_max_step(0.008)
        .with_iteration_limits(5_000, 128)
        .with_debug_snapshots(debug_ovito.then_some(SNAPSHOT_INTERVAL));

    let mut app = App::new();

    // 1. Generate fibers with distinct unloaded and initially placed shapes.
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0; 3], [false; 3],
        )))
        .add_plugins(MultiSegmentCrossingGeneratorPlugin {
            config: MultiSegmentCrossingConfig {
                count: 8,
                segments_per_fiber: 8,
                length: 0.7,
                placed_chord_fraction: case.placed_chord_fraction,
                radius: 0.018,
                intrinsic_shape: case.intrinsic_shape,
                placed_shape: case.placed_shape,
                minimum_bend_radius: Some(MINIMUM_BEND_RADIUS),
                material_name: "fiber".to_string(),
            },
        });

    // 2. Relax contact, stretch, natural bending, and admissible curvature.
    app.add_plugins(RelaxationPlugin { config: relaxation })
        .add_plugins(AssemblyCharacterizationPlugin);

    // 3. Optionally color segment capsules by bend-limit utilization in OVITO.
    if debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: output.ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::CurvatureRatio),
        });
    }

    // 4. Export the converged geometry as a bonded-particle model.
    app.add_plugins(BpmExportPlugin {
        config: output.sphere_bpm(1_800.0),
    });
    app.start();

    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .expect("RelaxationPlugin should install RelaxationState");
    assert!(
        relaxation.converged,
        "{} did not converge: {relaxation:?}",
        case.slug
    );
    let characterization = app
        .get_resource_ref::<AssemblyCharacterizationReport>()
        .expect("AssemblyCharacterizationPlugin should install its report");
    let metrics = characterization
        .final_state
        .as_ref()
        .expect("characterization should capture the relaxed assembly");
    let export = app
        .get_resource_ref::<BpmExportReport>()
        .expect("BpmExportPlugin should install BpmExportReport");

    println!("\n{}: {}", case.slug, case.description);
    println!(
        "  converged in {} iterations; penetration {:.3e}",
        relaxation.iterations, relaxation.max_penetration
    );
    println!(
        "  maximum curvature {:.3e}; bend-limit utilization {:.3}",
        metrics.maximum_curvature, metrics.maximum_curvature_ratio
    );
    println!(
        "  exported {} particles and {} bonds to {}",
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
    };
}
