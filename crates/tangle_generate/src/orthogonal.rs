use std::error::Error;
use std::fmt;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, FiberBendLimit, FiberId, Section, Vec3};

/// Two straight fibers crossing at a prescribed in-plane angle and axial offset.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberPairCrossingConfig {
    /// Number of initial centerline segments per fiber.
    pub segments_per_fiber: usize,
    /// End-to-end centerline length.
    pub length: f64,
    /// Circular cross-section radius.
    pub radius: f64,
    /// Distance between the two centerline axes at the crossing.
    pub axis_separation: f64,
    /// Smaller in-plane angle between the fiber directions, in degrees.
    pub crossing_angle_degrees: f64,
    /// Optional smallest admissible bend radius.
    pub minimum_bend_radius: Option<f64>,
    /// Solver-neutral material name.
    pub material_name: String,
}

impl Default for FiberPairCrossingConfig {
    fn default() -> Self {
        Self {
            segments_per_fiber: 1,
            length: 0.8,
            radius: 0.025,
            axis_separation: 0.04,
            crossing_angle_degrees: 90.0,
            minimum_bend_radius: None,
            material_name: "fiber".to_string(),
        }
    }
}

/// Invalid fiber-pair crossing generator input.
#[derive(Clone, Debug, PartialEq)]
pub enum FiberPairCrossingError {
    /// At least one segment is required.
    TooFewSegments,
    /// Length must be positive and finite.
    InvalidLength(f64),
    /// Radius must be positive and finite.
    InvalidRadius(f64),
    /// Axis separation must be finite and non-negative.
    InvalidSeparation(f64),
    /// Crossing angle must be finite and strictly between zero and 180 degrees.
    InvalidCrossingAngle(f64),
    /// Minimum bend radius must be positive and finite.
    InvalidMinimumBendRadius(f64),
}

impl fmt::Display for FiberPairCrossingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSegments => formatter.write_str("at least one segment is required"),
            Self::InvalidLength(value) => write!(formatter, "invalid fiber length {value}"),
            Self::InvalidRadius(value) => write!(formatter, "invalid fiber radius {value}"),
            Self::InvalidSeparation(value) => write!(formatter, "invalid axis separation {value}"),
            Self::InvalidCrossingAngle(value) => {
                write!(formatter, "invalid crossing angle {value} degrees")
            }
            Self::InvalidMinimumBendRadius(value) => {
                write!(formatter, "invalid minimum bend radius {value}")
            }
        }
    }
}

impl Error for FiberPairCrossingError {}

/// Appends the deterministic angled pair to an assembly.
pub fn generate_fiber_pair_crossing(
    assembly: &mut FiberAssembly,
    config: &FiberPairCrossingConfig,
) -> Result<(), FiberPairCrossingError> {
    if config.segments_per_fiber == 0 {
        return Err(FiberPairCrossingError::TooFewSegments);
    }
    if !config.length.is_finite() || config.length <= 0.0 {
        return Err(FiberPairCrossingError::InvalidLength(config.length));
    }
    if !config.radius.is_finite() || config.radius <= 0.0 {
        return Err(FiberPairCrossingError::InvalidRadius(config.radius));
    }
    if !config.axis_separation.is_finite() || config.axis_separation < 0.0 {
        return Err(FiberPairCrossingError::InvalidSeparation(
            config.axis_separation,
        ));
    }
    if !config.crossing_angle_degrees.is_finite()
        || config.crossing_angle_degrees <= 0.0
        || config.crossing_angle_degrees >= 180.0
    {
        return Err(FiberPairCrossingError::InvalidCrossingAngle(
            config.crossing_angle_degrees,
        ));
    }
    if config
        .minimum_bend_radius
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(FiberPairCrossingError::InvalidMinimumBendRadius(
            config.minimum_bend_radius.unwrap(),
        ));
    }

    let material = assembly.materials.add(config.material_name.clone());
    let section = assembly.sections.add(Section::Circular {
        radius: config.radius,
    });
    let center = cell_center(assembly);
    let angle = config.crossing_angle_degrees.to_radians();
    let directions = [[1.0, 0.0, 0.0], [angle.cos(), angle.sin(), 0.0]];
    for (fiber_index, direction) in directions.into_iter().enumerate() {
        let z_offset = if fiber_index == 0 { -0.5 } else { 0.5 } * config.axis_separation;
        let mut intrinsic = Vec::with_capacity(config.segments_per_fiber + 1);
        let mut placed = Vec::with_capacity(config.segments_per_fiber + 1);
        for point in 0..=config.segments_per_fiber {
            let parameter = (point as f64 / config.segments_per_fiber as f64 - 0.5) * config.length;
            let local = scale(direction, parameter);
            intrinsic.push(local);
            placed.push([
                center[0] + local[0],
                center[1] + local[1],
                center[2] + local[2] + z_offset,
            ]);
        }
        let id = FiberId(fiber_index as u32 + 1);
        assembly
            .add_fiber(id, material, section, &intrinsic, &placed)
            .expect("validated orthogonal fiber must be constructible");
        if let Some(minimum_bend_radius) = config.minimum_bend_radius {
            assembly
                .set_fiber_bend_limit(
                    id,
                    Some(FiberBendLimit {
                        minimum_bend_radius,
                    }),
                )
                .expect("new orthogonal fiber must accept a bend limit");
        }
    }
    assembly.provenance.source = "tangle_generate::fiber_pair_crossing".to_string();
    assembly.provenance.version = env!("CARGO_PKG_VERSION").to_string();
    assembly.provenance.notes.push(format!(
        "two fibers crossing at {:.3} degrees, {} initial segments each, axis separation {:.6e}",
        config.crossing_angle_degrees, config.segments_per_fiber, config.axis_separation
    ));
    Ok(())
}

/// GRASS plugin generating one angled fiber pair.
pub struct FiberPairCrossingGeneratorPlugin {
    /// Generator configuration.
    pub config: FiberPairCrossingConfig,
}

impl Plugin for FiberPairCrossingGeneratorPlugin {
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
    config: Res<FiberPairCrossingConfig>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    generate_fiber_pair_crossing(&mut assembly, &config)
        .unwrap_or_else(|error| panic!("fiber-pair crossing generation failed: {error}"));
    next.set(TangleStage::Relax);
}

fn cell_center(assembly: &FiberAssembly) -> Vec3 {
    let mut center = assembly.cell.origin;
    for edge in assembly.cell.basis {
        for axis in 0..3 {
            center[axis] += 0.5 * edge[axis];
        }
    }
    center
}

fn scale(vector: Vec3, factor: f64) -> Vec3 {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::PeriodicCell;

    #[test]
    fn generates_an_angled_overlapping_pair() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let config = FiberPairCrossingConfig {
            crossing_angle_degrees: 12.0,
            ..FiberPairCrossingConfig::default()
        };
        generate_fiber_pair_crossing(&mut assembly, &config).unwrap();
        assembly.validate().unwrap();
        assert_eq!(assembly.topology.fibers.len(), 2);
        assert_eq!(assembly.geometry.placed.positions.len(), 4);
        assert!(config.axis_separation < 2.0 * config.radius);
    }
}
