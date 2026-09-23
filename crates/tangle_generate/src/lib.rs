//! Deterministic fiber-generation algorithms and GRASS plugins.

#![warn(missing_docs)]

mod compaction;
mod junctions;
mod layered;
mod multisegment;
mod orthogonal;
mod population;
mod recipe;

pub use junctions::{JunctionCapturePolicy, JunctionCaptureReport, JunctionMaterialPair};
pub use layered::{LayeredFormationConfig, LayeredFormationPlugin, LayeredFormationState};
pub use multisegment::{
    generate_multisegment_crossing, CenterlineShape, MultiSegmentCrossingConfig,
    MultiSegmentCrossingGeneratorPlugin, MultiSegmentGenerationError,
};
pub use orthogonal::{
    generate_fiber_pair_crossing, FiberPairCrossingConfig, FiberPairCrossingError,
    FiberPairCrossingGeneratorPlugin,
};
pub use population::{
    generate_biased_fiber_population, BiasedFiberPopulationGeneratorPlugin, FiberPopulationSpec,
    OrientationDistribution, PopulationBoundaryPolicy, PopulationGenerationError,
    PositionDistribution, ScalarDistribution,
};
pub use recipe::{
    AcceptanceLimit, FiberInsertionPopulation, FormationEvent, FormationFailure,
    FormationOperation, FormationRecipeConfig, FormationRecipePlugin, FormationRecipeState,
    FormationWarning, LayerStagedFiberPopulationGeneratorPlugin, LimitEnforcement,
    MixedLayerStagedFiberPopulationGeneratorPlugin, NeedlingConfig, NeedlingReport,
    NeedlingSelection, RelaxationAcceptance, RelaxationTargets, SolveExhaustion, SolvePolicy,
    StagedFiberPopulationGeneratorPlugin, random_footprint_center,
};

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, FiberId, Section, Vec3};

/// Configuration for straight fibers whose centerlines share the cell center.
#[derive(Clone, Debug, PartialEq)]
pub struct PointCrossingConfig {
    /// Number of fibers.
    pub count: usize,
    /// Centerline length of every fiber.
    pub length: f64,
    /// Circular fiber radius.
    pub radius: f64,
    /// Solver-neutral material name.
    pub material_name: String,
}

impl Default for PointCrossingConfig {
    fn default() -> Self {
        Self {
            count: 8,
            length: 0.7,
            radius: 0.025,
            material_name: "fiber".to_string(),
        }
    }
}

/// Invalid point-crossing generator configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum GenerationError {
    /// At least two fibers are required for the crossing example.
    TooFewFibers(usize),
    /// Fiber length must be positive and finite.
    InvalidLength(f64),
    /// Fiber radius must be positive and finite.
    InvalidRadius(f64),
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewFibers(count) => {
                write!(
                    f,
                    "point-crossing generation needs at least 2 fibers, got {count}"
                )
            }
            Self::InvalidLength(length) => {
                write!(f, "fiber length must be positive and finite, got {length}")
            }
            Self::InvalidRadius(radius) => {
                write!(f, "fiber radius must be positive and finite, got {radius}")
            }
        }
    }
}

impl Error for GenerationError {}

/// Appends a deterministic Fibonacci-sphere set of centered straight fibers.
pub fn generate_point_crossing(
    assembly: &mut FiberAssembly,
    config: &PointCrossingConfig,
) -> Result<(), GenerationError> {
    if config.count < 2 {
        return Err(GenerationError::TooFewFibers(config.count));
    }
    if !config.length.is_finite() || config.length <= 0.0 {
        return Err(GenerationError::InvalidLength(config.length));
    }
    if !config.radius.is_finite() || config.radius <= 0.0 {
        return Err(GenerationError::InvalidRadius(config.radius));
    }

    let material = assembly.materials.add(config.material_name.clone());
    let section = assembly.sections.add(Section::Circular {
        radius: config.radius,
    });
    let center = cell_center(assembly);
    let half = 0.5 * config.length;
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());

    for index in 0..config.count {
        let fraction = (index as f64 + 0.5) / config.count as f64;
        let z = 1.0 - 2.0 * fraction;
        let radial = (1.0 - z * z).sqrt();
        let azimuth = index as f64 * golden_angle;
        let direction = [radial * azimuth.cos(), radial * azimuth.sin(), z];
        let intrinsic = [scale(direction, -half), scale(direction, half)];
        let placed = [add(center, intrinsic[0]), add(center, intrinsic[1])];
        assembly
            .add_fiber(
                FiberId(index as u32 + 1),
                material,
                section,
                &intrinsic,
                &placed,
            )
            .expect("validated point-crossing fiber must be constructible");
    }

    assembly.provenance.source = "tangle_generate::point_crossing".to_string();
    assembly.provenance.version = env!("CARGO_PKG_VERSION").to_string();
    assembly.provenance.notes.push(format!(
        "{} straight fibers initially share the cell center",
        config.count
    ));
    Ok(())
}

/// GRASS plugin that generates a point-crossing assembly once.
pub struct PointCrossingGeneratorPlugin {
    /// Generator configuration.
    pub config: PointCrossingConfig,
}

impl Plugin for PointCrossingGeneratorPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(self.config.clone()).add_update_system(
            generate_system.run_if(in_state(TangleStage::Generate)),
            TanglePhase::Generate,
        );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_GENERATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

fn generate_system(
    config: Res<PointCrossingConfig>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    generate_point_crossing(&mut assembly, &config)
        .unwrap_or_else(|error| panic!("point-crossing generation failed: {error}"));
    next.set(TangleStage::Relax);
}

fn cell_center(assembly: &FiberAssembly) -> Vec3 {
    let mut center = assembly.cell.origin;
    for edge in assembly.cell.basis {
        center = add(center, scale(edge, 0.5));
    }
    center
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(value: Vec3, factor: f64) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::PeriodicCell;

    #[test]
    fn every_generated_fiber_passes_through_cell_center() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let config = PointCrossingConfig {
            count: 6,
            ..PointCrossingConfig::default()
        };
        generate_point_crossing(&mut assembly, &config).unwrap();
        assembly.validate().unwrap();

        for fiber in &assembly.topology.fibers {
            let start = fiber.vertices.start as usize;
            let a = assembly.geometry.placed.positions[start];
            let b = assembly.geometry.placed.positions[start + 1];
            let midpoint = scale(add(a, b), 0.5);
            for coordinate in midpoint {
                assert!((coordinate - 0.5).abs() < 1.0e-12);
            }
        }
    }
}
pub use compaction::{
    nominal_fiber_volume, AdaptiveCompactionIncrement, CompactionConfig, CompactionGuards,
    CompactionPath, CompactionReport, CompactionStepReport, CompactionStopReason, CompactionTarget,
};
