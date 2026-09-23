use biased_fiber_box::{population_cases, population_spec, PopulationCase};
use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_characterize::{
    AssemblyCharacterizationPlugin, AssemblyCharacterizationReport, AssemblyMetrics,
};
use tangle_core::PeriodicCell;
use tangle_example_support::{debug_ovito_requested, ExampleOutput};
use tangle_export::{
    BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{
    BiasedFiberPopulationGeneratorPlugin, LayeredFormationPlugin, LayeredFormationState,
};
use tangle_relax::{RelaxationConfig, RelaxationPlugin, RelaxationState};

const FIBER_COUNT: usize = 160;
const SEGMENTS_PER_FIBER: usize = 8;
const SNAPSHOT_INTERVAL: usize = 10;

fn main() {
    let debug_ovito = debug_ovito_requested();
    for case in population_cases() {
        run_case(case, debug_ovito);
    }
}

fn run_case(case: PopulationCase, debug_ovito: bool) {
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), case.slug);
    let relaxation =
        RelaxationConfig::flexible().with_debug_snapshots(debug_ovito.then_some(SNAPSHOT_INTERVAL));

    let mut app = App::new();

    // 1. Sample a deterministic population with this case's manufacturing bias.
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0; 3], [false; 3],
        )))
        .add_plugins(BiasedFiberPopulationGeneratorPlugin {
            spec: population_spec(case, FIBER_COUNT, SEGMENTS_PER_FIBER),
        });

    // 2. Relax contacts and fiber shape in a persistent GPU world.
    app.add_plugins(RelaxationPlugin { config: relaxation });

    // 3. Apply a manufacturing schedule when the material case defines one.
    if let Some(formation) = case.formation {
        app.add_plugins(LayeredFormationPlugin { config: formation });
    }
    app.add_plugins(AssemblyCharacterizationPlugin);

    // 4. Optionally record sparse geometry snapshots for OVITO.
    if debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: output.ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::CurvatureRatio),
        });
    }

    // 5. Export the relaxed fibers as bonded-particle chains.
    app.add_plugins(BpmExportPlugin {
        config: output.sphere_bpm(1_800.0),
    });
    app.start();

    report_case(&app, case);
}

fn report_case(app: &App, case: PopulationCase) {
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
    let initial = characterization
        .initial
        .as_ref()
        .expect("generation metrics should be captured");
    let final_state = characterization
        .final_state
        .as_ref()
        .expect("relaxed metrics should be captured");
    let export = app
        .get_resource_ref::<BpmExportReport>()
        .expect("BpmExportPlugin should install BpmExportReport");

    println!("\n{}: {}", case.slug, case.description);
    println!(
        "  relaxed {} fibers in {} iterations; penetration {:.3e}",
        final_state.fibers, relaxation.iterations, relaxation.max_penetration
    );
    print_metrics("initial", initial);
    print_metrics("final", final_state);
    if let Some(formation) = app.get_resource_ref::<LayeredFormationState>() {
        println!(
            "  compacted {} source layers to {:.0}% of their initial spacing",
            formation.initial_targets.len(),
            100.0 * formation.current_spacing_scale
        );
    }
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

fn print_metrics(label: &str, metrics: &AssemblyMetrics) {
    let tensor = metrics.orientation_tensor;
    println!(
        "  {label}: volume fraction {:.4}; A11/A22/A33 = {:.3}/{:.3}/{:.3}; curvature use {:.3}",
        metrics.solid_volume_fraction,
        tensor[0][0],
        tensor[1][1],
        tensor[2][2],
        metrics.maximum_curvature_ratio
    );
}
