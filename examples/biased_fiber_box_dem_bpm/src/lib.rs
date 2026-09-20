//! Shared material cases for the biased-population tutorial and stress runner.

use std::f64::consts::PI;

use tangle_generate::{
    FiberPopulationSpec, LayeredFormationConfig, OrientationDistribution, PositionDistribution,
    ScalarDistribution,
};

/// One deterministic material-generation case.
#[derive(Clone, Copy, Debug)]
pub struct PopulationCase {
    /// Stable output and command-line name.
    pub slug: &'static str,
    /// Short human-readable explanation.
    pub description: &'static str,
    /// Fiber direction distribution.
    pub orientation: OrientationDistribution,
    /// Fiber-center distribution.
    pub position: PositionDistribution,
    /// Optional post-generation manufacturing schedule.
    pub formation: Option<LayeredFormationConfig>,
}

/// Returns the three material-generation cases demonstrated by the example.
pub fn population_cases() -> [PopulationCase; 3] {
    [
        PopulationCase {
            slug: "isotropic_3d",
            description: "uniform three-dimensional orientation and position",
            orientation: OrientationDistribution::Isotropic3d,
            position: PositionDistribution::Uniform,
            formation: None,
        },
        PopulationCase {
            slug: "planar_layered",
            description: "primarily in-plane fibers deposited in four layers",
            orientation: OrientationDistribution::Planar {
                normal: [0.0, 0.0, 1.0],
                maximum_tilt: 10.0 * PI / 180.0,
            },
            position: PositionDistribution::Layered {
                axis: 2,
                layers: 4,
                jitter_fraction: 0.25,
            },
            formation: Some(LayeredFormationConfig {
                axis: 2,
                ..LayeredFormationConfig::default()
            }),
        },
        PopulationCase {
            slug: "aligned_x",
            description: "fibers aligned within 15 degrees of the x axis",
            orientation: OrientationDistribution::Aligned {
                axis: [1.0, 0.0, 0.0],
                maximum_angle: 15.0 * PI / 180.0,
            },
            position: PositionDistribution::Uniform,
            formation: None,
        },
    ]
}

/// Builds the common seeded fiber specification for one material case.
pub fn population_spec(
    case: PopulationCase,
    count: usize,
    segments_per_fiber: usize,
) -> FiberPopulationSpec {
    FiberPopulationSpec {
        count,
        segments_per_fiber,
        seed: 20_260_918,
        length: ScalarDistribution::Uniform {
            minimum: 0.4,
            maximum: 0.58,
        },
        radius: ScalarDistribution::Uniform {
            minimum: 0.012,
            maximum: 0.017,
        },
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 0.014,
        },
        orientation: case.orientation,
        position: case.position,
        minimum_bend_radius: Some(0.08),
        max_attempts_per_fiber: 256,
        material_name: "fiber".to_string(),
        ..FiberPopulationSpec::default()
    }
}
