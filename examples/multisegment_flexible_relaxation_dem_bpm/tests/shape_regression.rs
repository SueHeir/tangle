use grass_app::prelude::*;
use multisegment_flexible_relaxation_dem_bpm::{shape_cases, ShapeCase, MINIMUM_BEND_RADIUS};
use tangle_app::prelude::*;
use tangle_characterize::characterize_assembly;
use tangle_core::{FiberAssembly, PeriodicCell};
use tangle_generate::{MultiSegmentCrossingConfig, MultiSegmentCrossingGeneratorPlugin};
use tangle_relax::{RelaxationConfig, RelaxationPlugin, RelaxationState};

const PENETRATION_TOLERANCE: f32 = 1.0e-4;
const CURVATURE_RATIO_TOLERANCE: f32 = 1.0e-4;

#[test]
fn every_shape_case_converges_to_an_admissible_assembled_reference() {
    for case in shape_cases() {
        let mut app = App::new();
        app.add_plugins(TangleWorkflowPlugin::default())
            .add_plugins(TangleAssemblyPlugin::new(PeriodicCell::orthorhombic(
                [1.0; 3], [false; 3],
            )))
            .add_plugins(MultiSegmentCrossingGeneratorPlugin {
                config: generator_config(case),
            })
            .add_plugins(RelaxationPlugin {
                config: RelaxationConfig::flexible()
                    .with_contact_correction(1.0)
                    .with_flexible_stiffness(0.35, 0.03)
                    .with_curvature_limit(1.0, CURVATURE_RATIO_TOLERANCE)
                    .with_max_step(0.008)
                    .with_iteration_limits(5_000, 128)
                    .with_next_stage(TangleStage::Done),
            });
        app.start();

        let relaxation = app
            .get_resource_ref::<RelaxationState>()
            .expect("relaxation state should exist");
        assert!(relaxation.converged, "{} did not converge", case.slug);
        assert!(relaxation.max_penetration <= PENETRATION_TOLERANCE);
        assert!(relaxation.max_curvature_ratio <= 1.0 + CURVATURE_RATIO_TOLERANCE);

        let assembly = app
            .get_resource_ref::<FiberAssembly>()
            .expect("assembly should exist");
        assert!(assembly.geometry.assembled_reference.is_some());
        assert!(characterize_assembly(&assembly).maximum_curvature_ratio <= 1.0 + 1.0e-5);
    }
}

fn generator_config(case: ShapeCase) -> MultiSegmentCrossingConfig {
    MultiSegmentCrossingConfig {
        count: 8,
        segments_per_fiber: 8,
        length: 0.7,
        placed_chord_fraction: case.placed_chord_fraction,
        radius: 0.018,
        intrinsic_shape: case.intrinsic_shape,
        placed_shape: case.placed_shape,
        minimum_bend_radius: Some(MINIMUM_BEND_RADIUS),
        material_name: "fiber".to_string(),
    }
}
