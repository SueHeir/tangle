use biased_fiber_box_dem_bpm::{population_cases, population_spec, PopulationCase};
use grass_app::prelude::*;
use tangle_app::prelude::*;
use tangle_characterize::{characterize_assembly, AssemblyMetrics};
use tangle_core::{FiberAssembly, PeriodicCell};
use tangle_generate::{
    generate_biased_fiber_population, BiasedFiberPopulationGeneratorPlugin, LayeredFormationPlugin,
};
use tangle_relax::{RelaxationConfig, RelaxationPlugin, RelaxationState};

const FIBER_COUNT: usize = 160;
const SEGMENTS_PER_FIBER: usize = 8;

#[test]
fn generated_and_relaxed_populations_retain_their_material_bias() {
    for case in population_cases() {
        let initial = generated_metrics(case);

        let mut app = App::new();
        app.add_plugins(TangleWorkflowPlugin::default())
            .add_plugins(TangleAssemblyPlugin::new(unit_cell()))
            .add_plugins(BiasedFiberPopulationGeneratorPlugin {
                spec: population_spec(case, FIBER_COUNT, SEGMENTS_PER_FIBER),
            })
            .add_plugins(RelaxationPlugin {
                // Layer compaction consumes 640 deterministic formation
                // iterations before the final unconstrained solve. Keep this
                // regression's convergence budget independent of that recipe
                // overhead instead of relying on the short tutorial default.
                config: RelaxationConfig::flexible()
                    .with_iteration_limits(6_000, 128)
                    .with_next_stage(TangleStage::Done),
            });
        if let Some(formation) = case.formation {
            app.add_plugins(LayeredFormationPlugin { config: formation });
        }
        app.start();

        let relaxation = app
            .get_resource_ref::<RelaxationState>()
            .expect("relaxation state should exist");
        assert!(relaxation.converged, "{} did not converge", case.slug);
        assert!(relaxation.max_penetration <= 1.0e-4);
        assert!(relaxation.max_curvature_ratio <= 1.0 + 1.0e-5);
        let final_state = characterize_assembly(
            &app.get_resource_ref::<FiberAssembly>()
                .expect("assembly should exist"),
        );

        assert_physical_metrics(&initial);
        assert_physical_metrics(&final_state);
        // Manufacturing compaction intentionally produces more out-of-plane
        // bending than contact-only relaxation, while retaining the material's
        // strongly planar character.
        let maximum_tensor_change = if case.formation.is_some() { 0.1 } else { 0.05 };
        assert!(
            tensor_distance(initial.orientation_tensor, final_state.orientation_tensor)
                < maximum_tensor_change,
            "{} changed orientation bias during relaxation",
            case.slug
        );
        assert_expected_bias(case.slug, &final_state);
    }
}

fn generated_metrics(case: PopulationCase) -> AssemblyMetrics {
    let mut assembly = FiberAssembly::new(unit_cell());
    generate_biased_fiber_population(
        &mut assembly,
        &population_spec(case, FIBER_COUNT, SEGMENTS_PER_FIBER),
    )
    .expect("population generation should succeed");
    characterize_assembly(&assembly)
}

fn unit_cell() -> PeriodicCell {
    PeriodicCell::orthorhombic([1.0; 3], [false; 3])
}

fn assert_physical_metrics(metrics: &AssemblyMetrics) {
    let trace = metrics.orientation_tensor[0][0]
        + metrics.orientation_tensor[1][1]
        + metrics.orientation_tensor[2][2];
    assert!((trace - 1.0).abs() < 1.0e-10);
    assert!(metrics.solid_volume_fraction > 0.0);
    assert!(metrics.maximum_curvature_ratio <= 1.0 + 1.0e-6);
}

fn assert_expected_bias(case: &str, metrics: &AssemblyMetrics) {
    match case {
        "isotropic_3d" => assert!(metrics
            .orientation_tensor
            .iter()
            .enumerate()
            .all(|(axis, row)| (0.15..0.55).contains(&row[axis]))),
        "planar_layered" => assert!(metrics.orientation_tensor[2][2] < 0.1),
        "aligned_x" => assert!(metrics.orientation_tensor[0][0] > 0.9),
        _ => panic!("unrecognized population case: {case}"),
    }
}

fn tensor_distance(first: [[f64; 3]; 3], second: [[f64; 3]; 3]) -> f64 {
    first
        .iter()
        .flatten()
        .zip(second.iter().flatten())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}
