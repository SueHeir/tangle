use std::path::PathBuf;

use biased_fiber_box::{population_cases, population_spec, PopulationCase};
use clap::Parser;
use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_characterize::{
    AssemblyCharacterizationPlugin, AssemblyCharacterizationReport, AssemblyMetrics,
};
use tangle_core::PeriodicCell;
use tangle_example_support::ExampleOutput;
use tangle_export::{
    BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{BiasedFiberPopulationGeneratorPlugin, LayeredFormationPlugin};
use tangle_relax::{RelaxationConfig, RelaxationPlugin, RelaxationState};

const SNAPSHOT_INTERVAL: usize = 10;

/// Configurable high-density companion to the biased-population tutorial.
#[derive(Clone, Debug, Parser)]
struct RunOptions {
    /// Record sparse GPU geometry readbacks for OVITO.
    #[arg(long)]
    debug_ovito: bool,
    /// Run only isotropic_3d, planar_layered, or aligned_x.
    #[arg(long)]
    case: Option<String>,
    /// Number of fibers in each selected population.
    #[arg(long, default_value_t = 800)]
    fiber_count: usize,
    /// Piecewise-linear segments per fiber.
    #[arg(long, default_value_t = 8)]
    segments_per_fiber: usize,
    /// Maximum relaxation iterations.
    #[arg(long, default_value_t = 6_000)]
    max_iterations: usize,
    /// Maximum GPU iterations in one GRASS-controlled batch.
    #[arg(long, default_value_t = 128)]
    batch_iterations: usize,
    /// Skip the staged compaction used by the planar layered case.
    #[arg(long)]
    no_layered_formation: bool,
    /// Relaxation iterations before the first layer-spacing reduction.
    #[arg(long, default_value_t = 128)]
    initial_layer_iterations: usize,
    /// Number of layer-spacing reductions.
    #[arg(long, default_value_t = 4)]
    compaction_steps: usize,
    /// Relaxation iterations at each compacted spacing.
    #[arg(long, default_value_t = 128)]
    layer_iterations_per_step: usize,
    /// Final divided by initial layer spacing.
    #[arg(long, default_value_t = 0.8)]
    final_layer_spacing_scale: f32,
}

fn main() {
    let options = RunOptions::parse();
    let selected: Vec<_> = population_cases()
        .into_iter()
        .filter(|case| options.case.as_deref().is_none_or(|slug| slug == case.slug))
        .collect();
    assert!(
        selected.len() > 0,
        "unknown population case: {:?}",
        options.case
    );

    for case in selected {
        run_case(case, &options);
    }
}

fn run_case(case: PopulationCase, options: &RunOptions) {
    let output = ExampleOutput::for_case(
        env!("CARGO_MANIFEST_DIR"),
        PathBuf::from("stress").join(case.slug),
    );
    let relaxation = RelaxationConfig::flexible()
        .with_iteration_limits(options.max_iterations, options.batch_iterations)
        .with_debug_snapshots(options.debug_ovito.then_some(SNAPSHOT_INTERVAL));

    let mut app = App::new();
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0; 3], [false; 3],
        )))
        .add_plugins(BiasedFiberPopulationGeneratorPlugin {
            spec: population_spec(case, options.fiber_count, options.segments_per_fiber),
        })
        .add_plugins(RelaxationPlugin { config: relaxation });

    if let Some(mut formation) = case.formation {
        if !options.no_layered_formation {
            formation.initial_relax_iterations = options.initial_layer_iterations;
            formation.compaction_steps = options.compaction_steps;
            formation.iterations_per_step = options.layer_iterations_per_step;
            formation.final_spacing_scale = options.final_layer_spacing_scale;
            app.add_plugins(LayeredFormationPlugin { config: formation });
        }
    }
    app.add_plugins(AssemblyCharacterizationPlugin);

    if options.debug_ovito {
        app.add_plugins(OvitoTrajectoryPlugin {
            config: output.ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::CurvatureRatio),
        });
    }
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
    let characterization = app
        .get_resource_ref::<AssemblyCharacterizationReport>()
        .expect("AssemblyCharacterizationPlugin should install its report");
    let initial = characterization
        .initial
        .as_ref()
        .expect("generation metrics should be captured");

    println!("\n{}: {}", case.slug, case.description);
    println!(
        "  GPU transfer: {} bytes uploaded, {} downloaded; {} cells",
        relaxation.uploaded_bytes, relaxation.downloaded_bytes, relaxation.cell_count
    );
    if !relaxation.converged {
        println!(
            "  stopped after {} iterations: penetration {:.3e}, maximum bend ratio {:.6}",
            relaxation.iterations, relaxation.max_penetration, relaxation.max_curvature_ratio
        );
        print_metrics("initial", initial);
        return;
    }

    let final_state = characterization
        .final_state
        .as_ref()
        .expect("relaxed metrics should be captured");
    let export = app
        .get_resource_ref::<BpmExportReport>()
        .expect("BpmExportPlugin should install BpmExportReport");
    println!(
        "  converged in {} iterations; penetration {:.3e}",
        relaxation.iterations, relaxation.max_penetration
    );
    print_metrics("initial", initial);
    print_metrics("final", final_state);
    println!("  DEM-BPM data: {}", export.data_path.display());
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
