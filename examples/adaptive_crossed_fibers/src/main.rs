use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_core::PeriodicCell;
use tangle_example_support::ExampleOutput;
use tangle_export::{
    BpmExportPlugin, BpmExportReport, OvitoColoring, OvitoTrajectoryPlugin, OvitoTrajectoryReport,
};
use tangle_generate::{FiberPairCrossingConfig, FiberPairCrossingGeneratorPlugin};
use tangle_relax::{
    AdaptiveSegmentationConfig, RelaxationConfig, RelaxationPlugin, RelaxationState,
};

const LENGTH: f64 = 0.8;
const RADIUS: f64 = 0.025;
const UNIFORM_SEGMENTS: usize = 8;
const REFINEMENT_INTERVAL: usize = 3;
const SNAPSHOT_INTERVAL: usize = 4;

#[derive(Clone, Copy)]
struct Case {
    slug: &'static str,
    description: &'static str,
    crossing_angle_degrees: f64,
    axis_separation_over_diameter: f64,
    initial_segments: usize,
    adaptive: bool,
}

const CASES: [Case; 3] = [
    Case {
        slug: "orthogonal_stop_early",
        description:
            "90-degree point contact; adaptive refinement should stop when one split is enough",
        crossing_angle_degrees: 90.0,
        axis_separation_over_diameter: 0.8,
        initial_segments: 1,
        adaptive: true,
    },
    Case {
        slug: "shallow_uniform_reference",
        description: "12-degree deep distributed contact with eight uniform segments per fiber",
        crossing_angle_degrees: 12.0,
        axis_separation_over_diameter: 0.1,
        initial_segments: UNIFORM_SEGMENTS,
        adaptive: false,
    },
    Case {
        slug: "shallow_adaptive",
        description: "12-degree deep distributed contact starting with one segment per fiber",
        crossing_angle_degrees: 12.0,
        axis_separation_over_diameter: 0.1,
        initial_segments: 1,
        adaptive: true,
    },
];

fn main() {
    let results = CASES.map(run_case);

    println!("\ncomparison:");
    println!(
        "  case                         active segments  splits  merges  refine  coarsen  iterations"
    );
    for (case, result) in CASES.iter().zip(results) {
        println!(
            "  {:<28} {:>15} {:>7} {:>7} {:>7} {:>8} {:>11}",
            case.slug,
            result.active_segments,
            result.segment_splits,
            result.segment_merges,
            result.refinement_passes,
            result.coarsening_passes,
            result.iterations
        );
    }
}

fn run_case(case: Case) -> CaseResult {
    let output = ExampleOutput::for_case(env!("CARGO_MANIFEST_DIR"), case.slug);
    let mut relaxation = RelaxationConfig::flexible()
        .with_contact_correction(1.0)
        .with_flexible_stiffness(0.5, 0.15)
        .with_curvature_limit(1.0, 1.0e-4)
        .with_max_step(0.004)
        .with_pinned_ends(true)
        // GRASS returns every four iterations for a debug frame, while the
        // resident device world checks refinement every three iterations.
        .with_iteration_limits(1_000, 17)
        .with_debug_snapshots(Some(SNAPSHOT_INTERVAL));
    relaxation.constraint_iterations = 4;
    if case.adaptive {
        relaxation = relaxation.with_adaptive_segmentation(AdaptiveSegmentationConfig {
            // Refine a contacting segment until it is at most two diameters long.
            contact_length_over_diameter: 2.0,
            // Never reserve or create a segment shorter than one diameter.
            minimum_length_over_diameter: 1.0,
            maximum_refinement_levels: 6,
            refinement_interval: REFINEMENT_INTERVAL,
            refinement_persistence: 2,
            coarsening_persistence: 8,
            coarsening_error_over_diameter: 0.1,
            coarsening_curvature_ratio: 0.25,
        });
    }

    let mut app = App::new();

    // 1. Generate the same overlapping fiber pair at the requested angle.
    app.add_plugins(TangleWorkflowPlugin::default())
        .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
            [1.0; 3], [false; 3],
        )))
        .add_plugins(FiberPairCrossingGeneratorPlugin {
            config: FiberPairCrossingConfig {
                segments_per_fiber: case.initial_segments,
                length: LENGTH,
                radius: RADIUS,
                axis_separation: case.axis_separation_over_diameter * (2.0 * RADIUS),
                crossing_angle_degrees: case.crossing_angle_degrees,
                minimum_bend_radius: Some(0.1),
                material_name: "fiber".to_string(),
            },
        });

    // 2. Pin the four ends, relax contact, and optionally split contacting segments.
    app.add_plugins(RelaxationPlugin { config: relaxation });

    // 3. Write spherocylinder frames colored by dyadic refinement level.
    app.add_plugins(OvitoTrajectoryPlugin {
        config: output.ovito_segments(SNAPSHOT_INTERVAL, OvitoColoring::RefinementLevel),
    });

    // 4. Export the final centerlines as an ordinary DEM-BPM model.
    app.add_plugins(BpmExportPlugin {
        config: output.sphere_bpm(1_800.0),
    });

    app.start();

    let state = app
        .get_resource_ref::<RelaxationState>()
        .expect("RelaxationPlugin should install RelaxationState");
    assert!(state.converged, "{} did not converge: {state:?}", case.slug);
    let trajectory = app
        .get_resource_ref::<OvitoTrajectoryReport>()
        .expect("OvitoTrajectoryPlugin should install OvitoTrajectoryReport");
    let export = app
        .get_resource_ref::<BpmExportReport>()
        .expect("BpmExportPlugin should install BpmExportReport");

    println!("\n{}: {}", case.slug, case.description);
    println!(
        "  {} active vertices, {} active segments, {} GPU splits/{} merges in {}/{} productive passes",
        state.active_vertices,
        state.active_segments,
        state.segment_splits,
        state.segment_merges,
        state.refinement_passes,
        state.coarsening_passes
    );
    println!(
        "  converged in {} iterations; penetration {:.3e}, maximum bend ratio {:.6}",
        state.iterations, state.max_penetration, state.max_curvature_ratio
    );
    println!(
        "  OVITO: {} ({} frames)",
        trajectory.dump_path.display(),
        trajectory.frames
    );
    println!(
        "  DEM-BPM: {} particles, {} bonds at {}",
        export.particles,
        export.bonds,
        export.data_path.display()
    );

    CaseResult {
        iterations: state.iterations,
        active_segments: state.active_segments,
        segment_splits: state.segment_splits,
        segment_merges: state.segment_merges,
        refinement_passes: state.refinement_passes,
        coarsening_passes: state.coarsening_passes,
    }
}

#[derive(Clone, Copy)]
struct CaseResult {
    iterations: usize,
    active_segments: usize,
    segment_splits: usize,
    segment_merges: usize,
    refinement_passes: usize,
    coarsening_passes: usize,
}
