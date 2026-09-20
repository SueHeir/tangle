use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{
    maximum_polyline_curvature, FiberAssembly, FiberBendLimit, FiberId, Section, Vec3,
};

/// Centerline shape used independently for intrinsic and placed geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CenterlineShape {
    /// A line with zero transverse displacement.
    #[default]
    Straight,
    /// A deterministic, smooth three-dimensional curve.
    Curved {
        /// Maximum displacement of the fundamental transverse mode.
        amplitude: f64,
    },
}

impl CenterlineShape {
    fn amplitude(self) -> f64 {
        match self {
            Self::Straight => 0.0,
            Self::Curved { amplitude } => amplitude,
        }
    }
}

/// Configuration for deterministic multi-segment fibers concentrated around
/// the center of a cell.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiSegmentCrossingConfig {
    /// Number of fibers.
    pub count: usize,
    /// Number of centerline segments per fiber.
    pub segments_per_fiber: usize,
    /// Intrinsic end-to-end parameter length before curvature is added.
    pub length: f64,
    /// Placed end-to-end distance divided by `length`.
    ///
    /// Values below one shorten the initial placement relative to the
    /// intrinsic axial parameterization and therefore introduce slack.
    pub placed_chord_fraction: f64,
    /// Circular fiber radius.
    pub radius: f64,
    /// Stress-free centerline shape the fibers mechanically prefer.
    pub intrinsic_shape: CenterlineShape,
    /// Centerline shape at initial placement in the network.
    pub placed_shape: CenterlineShape,
    /// Optional smallest admissible bend radius for every generated fiber.
    pub minimum_bend_radius: Option<f64>,
    /// Solver-neutral material name.
    pub material_name: String,
}

impl Default for MultiSegmentCrossingConfig {
    fn default() -> Self {
        Self {
            count: 8,
            segments_per_fiber: 8,
            length: 0.7,
            placed_chord_fraction: 1.0,
            radius: 0.018,
            intrinsic_shape: CenterlineShape::Straight,
            placed_shape: CenterlineShape::Straight,
            minimum_bend_radius: None,
            material_name: "fiber".to_string(),
        }
    }
}

/// Invalid multi-segment generator configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiSegmentGenerationError {
    /// At least two fibers are required.
    TooFewFibers(usize),
    /// At least two segments are required per fiber.
    TooFewSegments(usize),
    /// Fiber length must be positive and finite.
    InvalidLength(f64),
    /// Placed chord fraction must be in `(0, 1]`.
    InvalidChordFraction(f64),
    /// Fiber radius must be positive and finite.
    InvalidRadius(f64),
    /// A curvature amplitude was negative or non-finite.
    InvalidAmplitude(f64),
    /// Minimum bend radius was non-positive or non-finite.
    InvalidMinimumBendRadius(f64),
    /// The requested natural shape violates the requested bend limit.
    IntrinsicBendLimitExceeded {
        /// Largest curvature found in the intrinsic centerline.
        curvature: f64,
        /// Maximum curvature allowed by the minimum bend radius.
        maximum: f64,
    },
}

impl fmt::Display for MultiSegmentGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewFibers(count) => write!(f, "at least 2 fibers are required, got {count}"),
            Self::TooFewSegments(count) => {
                write!(f, "at least 2 segments per fiber are required, got {count}")
            }
            Self::InvalidLength(value) => {
                write!(f, "fiber length must be positive and finite, got {value}")
            }
            Self::InvalidChordFraction(value) => {
                write!(f, "placed_chord_fraction must be in (0, 1], got {value}")
            }
            Self::InvalidRadius(value) => {
                write!(f, "fiber radius must be positive and finite, got {value}")
            }
            Self::InvalidAmplitude(value) => {
                write!(
                    f,
                    "curvature amplitudes must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidMinimumBendRadius(value) => {
                write!(f, "minimum bend radius must be positive and finite, got {value}")
            }
            Self::IntrinsicBendLimitExceeded { curvature, maximum } => write!(
                f,
                "intrinsic curvature {curvature:.6e} exceeds maximum admissible curvature {maximum:.6e}"
            ),
        }
    }
}

impl Error for MultiSegmentGenerationError {}

/// Appends deterministic multi-segment fibers with independently selected
/// intrinsic and initially placed shapes.
///
/// Every placed centerline passes through the cell center, producing a
/// deliberately difficult but reproducible relaxation problem.
pub fn generate_multisegment_crossing(
    assembly: &mut FiberAssembly,
    config: &MultiSegmentCrossingConfig,
) -> Result<(), MultiSegmentGenerationError> {
    validate_config(config)?;
    let material = assembly.materials.add(config.material_name.clone());
    let section = assembly.sections.add(Section::Circular {
        radius: config.radius,
    });
    let center = cell_center(assembly);
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());

    for fiber_index in 0..config.count {
        let fraction = (fiber_index as f64 + 0.5) / config.count as f64;
        let z = 1.0 - 2.0 * fraction;
        let radial = (1.0 - z * z).sqrt();
        let azimuth = fiber_index as f64 * golden_angle;
        let direction = [radial * azimuth.cos(), radial * azimuth.sin(), z];
        let transverse_a = perpendicular(direction);
        let transverse_b = normalize(cross(direction, transverse_a));
        let handedness = if fiber_index % 4 < 2 { 1.0 } else { -1.0 };
        let intrinsic_amplitude = config.intrinsic_shape.amplitude();
        let placed_amplitude = config.placed_shape.amplitude();

        let mut intrinsic = Vec::with_capacity(config.segments_per_fiber + 1);
        let mut placed = Vec::with_capacity(config.segments_per_fiber + 1);
        for point_index in 0..=config.segments_per_fiber {
            let normalized = point_index as f64 / config.segments_per_fiber as f64;
            let centered = normalized - 0.5;
            let parameter = centered * config.length;
            let fundamental = (2.0 * PI * centered).sin();
            let harmonic = (4.0 * PI * centered).sin();

            intrinsic.push([
                parameter,
                intrinsic_amplitude * fundamental,
                0.5 * intrinsic_amplitude * handedness * harmonic,
            ]);

            let mut position = center;
            position = add(
                position,
                scale(direction, parameter * config.placed_chord_fraction),
            );
            position = add(
                position,
                scale(transverse_a, placed_amplitude * fundamental),
            );
            position = add(
                position,
                scale(transverse_b, 0.5 * placed_amplitude * handedness * harmonic),
            );
            placed.push(position);
        }

        if let Some(minimum_bend_radius) = config.minimum_bend_radius {
            let curvature = maximum_polyline_curvature(&intrinsic);
            let maximum = minimum_bend_radius.recip();
            if curvature > maximum * (1.0 + 1.0e-10) {
                return Err(MultiSegmentGenerationError::IntrinsicBendLimitExceeded {
                    curvature,
                    maximum,
                });
            }
        }

        let fiber_id = FiberId(fiber_index as u32 + 1);
        assembly
            .add_fiber(fiber_id, material, section, &intrinsic, &placed)
            .expect("validated multi-segment fiber must be constructible");
        if let Some(minimum_bend_radius) = config.minimum_bend_radius {
            assembly
                .set_fiber_bend_limit(
                    fiber_id,
                    Some(FiberBendLimit {
                        minimum_bend_radius,
                    }),
                )
                .expect("newly generated fiber must accept a bend limit");
        }
    }

    assembly.provenance.source = "tangle_generate::multisegment_crossing".to_string();
    assembly.provenance.version = env!("CARGO_PKG_VERSION").to_string();
    assembly.provenance.notes.push(format!(
        "{} fibers with {} segments each initially share the cell center",
        config.count, config.segments_per_fiber
    ));
    assembly.provenance.notes.push(format!(
        "intrinsic shape: {}; initially placed shape: {}",
        shape_name(config.intrinsic_shape),
        shape_name(config.placed_shape)
    ));
    if let Some(radius) = config.minimum_bend_radius {
        assembly
            .provenance
            .notes
            .push(format!("minimum admissible bend radius: {radius:.6e}"));
    }
    Ok(())
}

/// GRASS plugin that generates a multi-segment crossing assembly once.
pub struct MultiSegmentCrossingGeneratorPlugin {
    /// Generator configuration.
    pub config: MultiSegmentCrossingConfig,
}

impl Plugin for MultiSegmentCrossingGeneratorPlugin {
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
    config: Res<MultiSegmentCrossingConfig>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    generate_multisegment_crossing(&mut assembly, &config)
        .unwrap_or_else(|error| panic!("multi-segment generation failed: {error}"));
    next.set(TangleStage::Relax);
}

fn validate_config(config: &MultiSegmentCrossingConfig) -> Result<(), MultiSegmentGenerationError> {
    if config.count < 2 {
        return Err(MultiSegmentGenerationError::TooFewFibers(config.count));
    }
    if config.segments_per_fiber < 2 {
        return Err(MultiSegmentGenerationError::TooFewSegments(
            config.segments_per_fiber,
        ));
    }
    if !config.length.is_finite() || config.length <= 0.0 {
        return Err(MultiSegmentGenerationError::InvalidLength(config.length));
    }
    if !config.placed_chord_fraction.is_finite()
        || config.placed_chord_fraction <= 0.0
        || config.placed_chord_fraction > 1.0
    {
        return Err(MultiSegmentGenerationError::InvalidChordFraction(
            config.placed_chord_fraction,
        ));
    }
    if !config.radius.is_finite() || config.radius <= 0.0 {
        return Err(MultiSegmentGenerationError::InvalidRadius(config.radius));
    }
    for amplitude in [
        config.intrinsic_shape.amplitude(),
        config.placed_shape.amplitude(),
    ] {
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(MultiSegmentGenerationError::InvalidAmplitude(amplitude));
        }
    }
    if let Some(radius) = config.minimum_bend_radius {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(MultiSegmentGenerationError::InvalidMinimumBendRadius(
                radius,
            ));
        }
    }
    Ok(())
}

fn shape_name(shape: CenterlineShape) -> &'static str {
    match shape {
        CenterlineShape::Straight => "straight",
        CenterlineShape::Curved { .. } => "curved",
    }
}

fn cell_center(assembly: &FiberAssembly) -> Vec3 {
    let mut center = assembly.cell.origin;
    for edge in assembly.cell.basis {
        center = add(center, scale(edge, 0.5));
    }
    center
}

fn perpendicular(direction: Vec3) -> Vec3 {
    let axis =
        if direction[0].abs() <= direction[1].abs() && direction[0].abs() <= direction[2].abs() {
            [1.0, 0.0, 0.0]
        } else if direction[1].abs() <= direction[2].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
    normalize(cross(direction, axis))
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(value: Vec3, factor: f64) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(value: Vec3) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn normalize(value: Vec3) -> Vec3 {
    scale(value, norm(value).recip())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::PeriodicCell;

    #[test]
    fn creates_dense_multisegment_spans() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let config = MultiSegmentCrossingConfig {
            count: 4,
            segments_per_fiber: 6,
            ..MultiSegmentCrossingConfig::default()
        };
        generate_multisegment_crossing(&mut assembly, &config).unwrap();
        assembly.validate().unwrap();

        assert_eq!(assembly.topology.fibers.len(), 4);
        assert!(assembly
            .topology
            .fibers
            .iter()
            .all(|fiber| fiber.vertices.len == 7));
    }

    #[test]
    fn straight_intrinsic_and_placed_shapes_are_both_lines() {
        let assembly = generate_case(CenterlineShape::Straight, CenterlineShape::Straight);
        assert!(is_intrinsic_straight(&assembly));
        assert!(is_placed_straight(&assembly));
    }

    #[test]
    fn curved_placement_can_have_a_straight_intrinsic_shape() {
        let assembly = generate_case(
            CenterlineShape::Straight,
            CenterlineShape::Curved { amplitude: 0.04 },
        );
        assert!(is_intrinsic_straight(&assembly));
        assert!(!is_placed_straight(&assembly));
    }

    #[test]
    fn curved_intrinsic_and_placed_shapes_can_match() {
        let assembly = generate_case(
            CenterlineShape::Curved { amplitude: 0.04 },
            CenterlineShape::Curved { amplitude: 0.04 },
        );
        assert!(!is_intrinsic_straight(&assembly));
        assert!(!is_placed_straight(&assembly));

        let intrinsic = &assembly.geometry.intrinsic.positions[0..9];
        let placed = &assembly.geometry.placed.positions[0..9];
        for first in 0..intrinsic.len() {
            for second in first + 1..intrinsic.len() {
                let intrinsic_distance = point_distance(intrinsic[first], intrinsic[second]);
                let placed_distance = point_distance(placed[first], placed[second]);
                assert!((intrinsic_distance - placed_distance).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn over_bent_placement_is_allowed_when_intrinsic_shape_is_admissible() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let minimum_bend_radius = 0.12;
        generate_multisegment_crossing(
            &mut assembly,
            &MultiSegmentCrossingConfig {
                count: 2,
                placed_chord_fraction: 0.7,
                intrinsic_shape: CenterlineShape::Straight,
                placed_shape: CenterlineShape::Curved { amplitude: 0.12 },
                minimum_bend_radius: Some(minimum_bend_radius),
                ..MultiSegmentCrossingConfig::default()
            },
        )
        .unwrap();
        assembly.validate().unwrap();

        let first = &assembly.topology.fibers[0];
        let start = first.vertices.start as usize;
        let end = first.vertices.checked_end().unwrap() as usize;
        let range = start..end;
        let current = maximum_polyline_curvature(&assembly.geometry.placed.positions[range]);
        assert!(current > minimum_bend_radius.recip());
        assert_eq!(
            assembly.admissibility.bend_limits[0]
                .unwrap()
                .minimum_bend_radius,
            minimum_bend_radius
        );
    }

    #[test]
    fn every_center_vertex_starts_at_cell_center() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let config = MultiSegmentCrossingConfig::default();
        generate_multisegment_crossing(&mut assembly, &config).unwrap();

        for fiber in &assembly.topology.fibers {
            let center_index = fiber.vertices.start as usize + config.segments_per_fiber / 2;
            assert_eq!(
                assembly.geometry.placed.positions[center_index],
                [0.5, 0.5, 0.5]
            );
        }
    }

    fn generate_case(
        intrinsic_shape: CenterlineShape,
        placed_shape: CenterlineShape,
    ) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        generate_multisegment_crossing(
            &mut assembly,
            &MultiSegmentCrossingConfig {
                count: 2,
                intrinsic_shape,
                placed_shape,
                ..MultiSegmentCrossingConfig::default()
            },
        )
        .unwrap();
        assembly
    }

    fn is_intrinsic_straight(assembly: &FiberAssembly) -> bool {
        assembly.geometry.intrinsic.positions[0..9]
            .iter()
            .all(|point| point[1].abs() < 1.0e-12 && point[2].abs() < 1.0e-12)
    }

    fn is_placed_straight(assembly: &FiberAssembly) -> bool {
        let points = &assembly.geometry.placed.positions[0..9];
        let chord = [
            points[8][0] - points[0][0],
            points[8][1] - points[0][1],
            points[8][2] - points[0][2],
        ];
        points[1..8].iter().all(|point| {
            let offset = [
                point[0] - points[0][0],
                point[1] - points[0][1],
                point[2] - points[0][2],
            ];
            norm(cross(chord, offset)) < 1.0e-12
        })
    }

    fn point_distance(first: [f64; 3], second: [f64; 3]) -> f64 {
        norm([
            second[0] - first[0],
            second[1] - first[1],
            second[2] - first[2],
        ])
    }
}
