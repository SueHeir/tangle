use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_core::PeriodicCell;
use tangle_example_support::{debug_ovito_requested, ExampleOutput};
use tangle_export::{DemBpmExportPlugin, OvitoColoring, OvitoTrajectoryPlugin};
use tangle_generate::{PointCrossingConfig, PointCrossingGeneratorPlugin};
use tangle_relax::{RelaxationConfig, RelaxationPlugin, RelaxationState};

const FIBER_COUNT: usize = 8;
const SNAPSHOT_INTERVAL: usize = 1;

fn main() {
    let debug_ovito = debug_ovito_requested();
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), "");

    let relaxation = RelaxationConfig::rigid_translation()
        .with_penetration_tolerance(1.0e-6)
        .with_contact_correction(0.75)
        .with_max_step(0.01)
        .with_debug_snapshots(debug_ovito.then_some(SNAPSHOT_INTERVAL));

    let mut app = App::new();

    // 1. Define the domain and the deterministic fiber generator.
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0; 3], [false; 3],
        )))
        .add_plugins(PointCrossingGeneratorPlugin {
            config: PointCrossingConfig {
                count: FIBER_COUNT,
                length: 0.7,
                radius: 0.025,
                material_name: "fiber".to_string(),
            },
        });

    // 2. Separate the fibers on the GPU while preserving their rigid shapes.
    app.add_plugins(RelaxationPlugin { config: relaxation });

    // 3. Optionally record the relaxation as oriented OVITO spherocylinders.
    if debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: output.ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::Fiber),
        });
    }

    // 4. Discretize each relaxed centerline into a bonded-particle chain.
    app.add_plugins(DemBpmExportPlugin {
        config: output.dem_bpm(1_800.0),
    });
    app.start();

    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .expect("RelaxationPlugin should install RelaxationState");
    assert!(
        relaxation.converged,
        "fiber relaxation failed: {relaxation:?}"
    );

    println!(
        "relaxed {FIBER_COUNT} rigid fibers in {} iterations",
        relaxation.iterations
    );
    println!("DEM-BPM data: {}", output.dem_data.display());
    if debug_ovito {
        println!("OVITO trajectory: {}", output.ovito_dump.display());
    }
}
