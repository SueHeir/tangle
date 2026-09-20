use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{
    maximum_polyline_curvature, FiberAssembly, FiberBendLimit, FiberId, Section, Vec3,
};

/// Deterministic scalar sampling rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarDistribution {
    /// Always return one value.
    Constant(f64),
    /// Sample uniformly from the inclusive numeric range.
    Uniform {
        /// Lower bound.
        minimum: f64,
        /// Upper bound.
        maximum: f64,
    },
}

impl ScalarDistribution {
    fn sample(self, random: &mut SplitMix64) -> f64 {
        match self {
            Self::Constant(value) => value,
            Self::Uniform { minimum, maximum } => minimum + (maximum - minimum) * random.unit(),
        }
    }

    fn bounds(self) -> (f64, f64) {
        match self {
            Self::Constant(value) => (value, value),
            Self::Uniform { minimum, maximum } => (minimum, maximum),
        }
    }
}

/// Statistical orientation rule for undirected fibers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrientationDistribution {
    /// Uniform directions over the unit sphere.
    Isotropic3d,
    /// Directions concentrated around a plane.
    Planar {
        /// Unit-normal direction is computed internally.
        normal: Vec3,
        /// Maximum absolute tilt out of the plane, in radians.
        maximum_tilt: f64,
    },
    /// Layer-aware planar mixture around a seeded reference direction, its
    /// in-plane perpendicular, and a uniformly random residual population.
    LayeredBiaxial {
        /// Unit-normal direction is computed internally.
        normal: Vec3,
        /// Fraction sampled near the layer's seeded reference direction.
        primary_fraction: f64,
        /// Fraction sampled near the in-plane perpendicular direction.
        cross_fraction: f64,
        /// Maximum angular deviation around either biased in-plane direction.
        maximum_in_plane_deviation: f64,
        /// Maximum absolute tilt out of the plane, in radians.
        maximum_tilt: f64,
        /// Seed used to assign one reference direction to each layer.
        layer_seed: u64,
    },
    /// Directions contained in a cone around a preferred axis.
    Aligned {
        /// Preferred direction; normalization is computed internally.
        axis: Vec3,
        /// Cone half-angle in radians.
        maximum_angle: f64,
    },
}

/// Statistical center-position rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionDistribution {
    /// Uniform centers over each fiber's feasible contained region.
    Uniform,
    /// Centers concentrated on evenly spaced layers.
    Layered {
        /// Cartesian layer-normal axis: 0, 1, or 2.
        axis: usize,
        /// Number of layers.
        layers: usize,
        /// Fraction of layer spacing used as uniform jitter in `[0, 1]`.
        jitter_fraction: f64,
    },
    /// Monotonic density gradient along one Cartesian axis.
    DensityGradient {
        /// Cartesian gradient axis: 0, 1, or 2.
        axis: usize,
        /// Positive power controlling concentration.
        exponent: f64,
        /// Concentrate fibers toward the high side when true.
        toward_high: bool,
    },
}

/// Boundary treatment used by biased-box generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PopulationBoundaryPolicy {
    /// Sample only center positions for which the full swept fiber fits.
    #[default]
    FullyContained,
}

/// Reproducible specification of a biased fiber population.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberPopulationSpec {
    /// Number of fibers to generate.
    pub count: usize,
    /// Number of piecewise-linear segments per fiber.
    pub segments_per_fiber: usize,
    /// Deterministic random seed.
    pub seed: u64,
    /// Fiber centerline-length parameter distribution.
    pub length: ScalarDistribution,
    /// Optional physical parent-fiber length represented by the generated
    /// local centerline window.
    pub nominal_parent_length: Option<f64>,
    /// Circular-radius distribution.
    pub radius: ScalarDistribution,
    /// Natural waviness-amplitude distribution.
    pub intrinsic_curvature_amplitude: ScalarDistribution,
    /// Fiber orientation distribution.
    pub orientation: OrientationDistribution,
    /// Fiber center-position distribution.
    pub position: PositionDistribution,
    /// Optional minimum admissible bend radius.
    pub minimum_bend_radius: Option<f64>,
    /// Cell-boundary treatment.
    pub boundary: PopulationBoundaryPolicy,
    /// Maximum attempts allowed to find each admissible fiber.
    pub max_attempts_per_fiber: usize,
    /// Solver-neutral material name.
    pub material_name: String,
}

impl Default for FiberPopulationSpec {
    fn default() -> Self {
        Self {
            count: 32,
            segments_per_fiber: 8,
            seed: 1,
            length: ScalarDistribution::Uniform {
                minimum: 0.4,
                maximum: 0.6,
            },
            nominal_parent_length: None,
            radius: ScalarDistribution::Uniform {
                minimum: 0.01,
                maximum: 0.015,
            },
            intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
                minimum: 0.0,
                maximum: 0.02,
            },
            orientation: OrientationDistribution::Isotropic3d,
            position: PositionDistribution::Uniform,
            minimum_bend_radius: Some(0.08),
            boundary: PopulationBoundaryPolicy::FullyContained,
            max_attempts_per_fiber: 128,
            material_name: "fiber".to_string(),
        }
    }
}

/// Invalid population specification or an exhausted placement search.
#[derive(Clone, Debug, PartialEq)]
pub enum PopulationGenerationError {
    /// At least one fiber is required.
    EmptyPopulation,
    /// At least two segments are required.
    TooFewSegments(usize),
    /// A named scalar distribution was invalid.
    InvalidDistribution(&'static str),
    /// An orientation vector was zero or non-finite.
    InvalidDirection,
    /// An angular spread was invalid.
    InvalidAngle(f64),
    /// Biaxial mixture fractions were negative or summed above one.
    InvalidOrientationWeights([f64; 2]),
    /// A position distribution was invalid.
    InvalidPositionDistribution,
    /// Minimum bend radius was invalid.
    InvalidMinimumBendRadius(f64),
    /// The cell is not a positive orthorhombic box.
    NonOrthorhombicCell,
    /// Placement attempts were exhausted for a fiber.
    PlacementExhausted(usize),
}

impl fmt::Display for PopulationGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPopulation => f.write_str("fiber population count must be positive"),
            Self::TooFewSegments(count) => {
                write!(
                    f,
                    "at least two segments per fiber are required, got {count}"
                )
            }
            Self::InvalidDistribution(name) => write!(f, "invalid {name} distribution"),
            Self::InvalidDirection => {
                f.write_str("orientation direction must be finite and nonzero")
            }
            Self::InvalidAngle(value) => write!(f, "orientation angle is invalid: {value}"),
            Self::InvalidOrientationWeights(weights) => write!(
                f,
                "orientation fractions must be nonnegative and sum to at most one, got [{}, {}]",
                weights[0], weights[1]
            ),
            Self::InvalidPositionDistribution => f.write_str("invalid position distribution"),
            Self::InvalidMinimumBendRadius(value) => {
                write!(
                    f,
                    "minimum bend radius must be positive and finite, got {value}"
                )
            }
            Self::NonOrthorhombicCell => {
                f.write_str("biased-box generation requires a positive orthorhombic cell")
            }
            Self::PlacementExhausted(index) => write!(
                f,
                "could not place fiber {index} within max_attempts_per_fiber"
            ),
        }
    }
}

impl Error for PopulationGenerationError {}

/// Generates a deterministic, statistically biased fiber population.
///
/// Swept fibers remain contained on bounded axes. On periodic axes their
/// centers span the complete cell and centerlines may cross a periodic face.
pub fn generate_biased_fiber_population(
    assembly: &mut FiberAssembly,
    spec: &FiberPopulationSpec,
) -> Result<(), PopulationGenerationError> {
    validate_spec(spec)?;
    let (box_low, box_high) = orthorhombic_bounds(assembly)?;
    let material = assembly.materials.add(spec.material_name.clone());
    let mut random = SplitMix64::new(spec.seed);
    let first_id = assembly
        .topology
        .fibers
        .iter()
        .map(|fiber| fiber.id.0)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    for fiber_index in 0..spec.count {
        let assigned_formation_layer = sample_formation_layer(spec.position, &mut random);
        let mut generated = None;
        for _ in 0..spec.max_attempts_per_fiber {
            let length = spec.length.sample(&mut random);
            let radius = spec.radius.sample(&mut random);
            let amplitude = spec.intrinsic_curvature_amplitude.sample(&mut random);
            let handedness = if random.unit() < 0.5 { -1.0 } else { 1.0 };
            let intrinsic =
                local_centerline(spec.segments_per_fiber, length, amplitude, handedness);
            if spec.minimum_bend_radius.is_some_and(|minimum| {
                maximum_polyline_curvature(&intrinsic) > minimum.recip() * (1.0 + 1.0e-10)
            }) {
                continue;
            }
            let direction =
                sample_orientation(spec.orientation, assigned_formation_layer, &mut random);
            let transverse_a = perpendicular(direction);
            let transverse_b = normalize(cross(direction, transverse_a));
            let offsets: Vec<Vec3> = intrinsic
                .iter()
                .map(|point| {
                    add(
                        add(scale(direction, point[0]), scale(transverse_a, point[1])),
                        scale(transverse_b, point[2]),
                    )
                })
                .collect();
            let Some((center, formation_layer)) = sample_contained_center(
                &offsets,
                radius,
                box_low,
                box_high,
                assembly.cell.periodic,
                spec.position,
                assigned_formation_layer,
                &mut random,
            ) else {
                continue;
            };
            let placed: Vec<Vec3> = offsets.iter().map(|offset| add(center, *offset)).collect();
            generated = Some((intrinsic, placed, radius, formation_layer));
            break;
        }

        let Some((intrinsic, placed, radius, formation_layer)) = generated else {
            return Err(PopulationGenerationError::PlacementExhausted(fiber_index));
        };
        let section = assembly.sections.add(Section::Circular { radius });
        let id = FiberId(first_id + fiber_index as u32);
        assembly
            .add_fiber(id, material, section, &intrinsic, &placed)
            .expect("validated population fiber must be constructible");
        assembly
            .set_fiber_formation_layer(id, formation_layer)
            .expect("new population fiber must accept its formation layer");
        if let Some(minimum_bend_radius) = spec.minimum_bend_radius {
            assembly
                .set_fiber_bend_limit(
                    id,
                    Some(FiberBendLimit {
                        minimum_bend_radius,
                    }),
                )
                .expect("new population fiber must accept its bend limit");
        }
    }

    assembly.provenance.source = "tangle_generate::biased_fiber_population".to_string();
    assembly.provenance.version = env!("CARGO_PKG_VERSION").to_string();
    assembly.provenance.seed = Some(spec.seed);
    assembly.provenance.notes.push(format!(
        "generated {} fibers with {} segments each using bounded/periodic cell rules",
        spec.count, spec.segments_per_fiber
    ));
    if let Some(parent_length) = spec.nominal_parent_length {
        assembly.provenance.notes.push(format!(
            "local centerline windows represent nominal parent fibers of length {parent_length:.6e}"
        ));
    }
    assembly
        .provenance
        .notes
        .push(format!("orientation distribution: {:?}", spec.orientation));
    assembly
        .provenance
        .notes
        .push(format!("position distribution: {:?}", spec.position));
    Ok(())
}

/// GRASS plugin for seeded, statistically biased box generation.
pub struct BiasedFiberPopulationGeneratorPlugin {
    /// Population specification.
    pub spec: FiberPopulationSpec,
}

impl Plugin for BiasedFiberPopulationGeneratorPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(self.spec.clone()).add_update_system(
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
    spec: Res<FiberPopulationSpec>,
    mut assembly: ResMut<FiberAssembly>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    generate_biased_fiber_population(&mut assembly, &spec)
        .unwrap_or_else(|error| panic!("biased fiber-population generation failed: {error}"));
    next.set(TangleStage::Relax);
}

fn validate_spec(spec: &FiberPopulationSpec) -> Result<(), PopulationGenerationError> {
    if spec.count == 0 {
        return Err(PopulationGenerationError::EmptyPopulation);
    }
    if spec.segments_per_fiber < 2 {
        return Err(PopulationGenerationError::TooFewSegments(
            spec.segments_per_fiber,
        ));
    }
    validate_scalar(spec.length, "length", true)?;
    if spec
        .nominal_parent_length
        .is_some_and(|length| !length.is_finite() || length <= 0.0)
    {
        return Err(PopulationGenerationError::InvalidDistribution(
            "nominal_parent_length",
        ));
    }
    validate_scalar(spec.radius, "radius", true)?;
    validate_scalar(
        spec.intrinsic_curvature_amplitude,
        "intrinsic curvature amplitude",
        false,
    )?;
    match spec.orientation {
        OrientationDistribution::Isotropic3d => {}
        OrientationDistribution::Planar {
            normal,
            maximum_tilt,
        } => {
            validate_direction(normal)?;
            if !maximum_tilt.is_finite() || !(0.0..=0.5 * PI).contains(&maximum_tilt) {
                return Err(PopulationGenerationError::InvalidAngle(maximum_tilt));
            }
        }
        OrientationDistribution::LayeredBiaxial {
            normal,
            primary_fraction,
            cross_fraction,
            maximum_in_plane_deviation,
            maximum_tilt,
            ..
        } => {
            validate_direction(normal)?;
            let weights = [primary_fraction, cross_fraction];
            if weights
                .into_iter()
                .any(|weight| !weight.is_finite() || weight < 0.0)
                || primary_fraction + cross_fraction > 1.0
            {
                return Err(PopulationGenerationError::InvalidOrientationWeights(
                    weights,
                ));
            }
            if !maximum_in_plane_deviation.is_finite()
                || !(0.0..=0.5 * PI).contains(&maximum_in_plane_deviation)
            {
                return Err(PopulationGenerationError::InvalidAngle(
                    maximum_in_plane_deviation,
                ));
            }
            if !maximum_tilt.is_finite() || !(0.0..=0.5 * PI).contains(&maximum_tilt) {
                return Err(PopulationGenerationError::InvalidAngle(maximum_tilt));
            }
            if !matches!(spec.position, PositionDistribution::Layered { .. }) {
                return Err(PopulationGenerationError::InvalidPositionDistribution);
            }
        }
        OrientationDistribution::Aligned {
            axis,
            maximum_angle,
        } => {
            validate_direction(axis)?;
            if !maximum_angle.is_finite() || !(0.0..=PI).contains(&maximum_angle) {
                return Err(PopulationGenerationError::InvalidAngle(maximum_angle));
            }
        }
    }
    match spec.position {
        PositionDistribution::Uniform => {}
        PositionDistribution::Layered {
            axis,
            layers,
            jitter_fraction,
        } if axis < 3
            && layers > 0
            && jitter_fraction.is_finite()
            && (0.0..=1.0).contains(&jitter_fraction) => {}
        PositionDistribution::DensityGradient { axis, exponent, .. }
            if axis < 3 && exponent.is_finite() && exponent > 0.0 => {}
        _ => return Err(PopulationGenerationError::InvalidPositionDistribution),
    }
    if let Some(radius) = spec.minimum_bend_radius {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(PopulationGenerationError::InvalidMinimumBendRadius(radius));
        }
    }
    if spec.max_attempts_per_fiber == 0 {
        return Err(PopulationGenerationError::PlacementExhausted(0));
    }
    Ok(())
}

fn validate_scalar(
    distribution: ScalarDistribution,
    name: &'static str,
    strictly_positive: bool,
) -> Result<(), PopulationGenerationError> {
    let (minimum, maximum) = distribution.bounds();
    let lower_valid = if strictly_positive {
        minimum > 0.0
    } else {
        minimum >= 0.0
    };
    if !minimum.is_finite() || !maximum.is_finite() || !lower_valid || maximum < minimum {
        return Err(PopulationGenerationError::InvalidDistribution(name));
    }
    Ok(())
}

fn validate_direction(direction: Vec3) -> Result<(), PopulationGenerationError> {
    if direction.iter().all(|value| value.is_finite()) && norm(direction) > f64::EPSILON {
        Ok(())
    } else {
        Err(PopulationGenerationError::InvalidDirection)
    }
}

fn orthorhombic_bounds(
    assembly: &FiberAssembly,
) -> Result<(Vec3, Vec3), PopulationGenerationError> {
    let basis = assembly.cell.basis;
    if basis[0][1].abs() > 1.0e-14
        || basis[0][2].abs() > 1.0e-14
        || basis[1][0].abs() > 1.0e-14
        || basis[1][2].abs() > 1.0e-14
        || basis[2][0].abs() > 1.0e-14
        || basis[2][1].abs() > 1.0e-14
        || ![basis[0][0], basis[1][1], basis[2][2]]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(PopulationGenerationError::NonOrthorhombicCell);
    }
    Ok((
        assembly.cell.origin,
        [
            assembly.cell.origin[0] + basis[0][0],
            assembly.cell.origin[1] + basis[1][1],
            assembly.cell.origin[2] + basis[2][2],
        ],
    ))
}

fn local_centerline(segments: usize, length: f64, amplitude: f64, handedness: f64) -> Vec<Vec3> {
    (0..=segments)
        .map(|index| {
            let centered = index as f64 / segments as f64 - 0.5;
            [
                centered * length,
                amplitude * (2.0 * PI * centered).sin(),
                0.5 * amplitude * handedness * (4.0 * PI * centered).sin(),
            ]
        })
        .collect()
}

fn sample_orientation(
    distribution: OrientationDistribution,
    formation_layer: Option<u32>,
    random: &mut SplitMix64,
) -> Vec3 {
    match distribution {
        OrientationDistribution::Isotropic3d => {
            let z = 2.0 * random.unit() - 1.0;
            let azimuth = 2.0 * PI * random.unit();
            let radial = (1.0 - z * z).sqrt();
            [radial * azimuth.cos(), radial * azimuth.sin(), z]
        }
        OrientationDistribution::Planar {
            normal,
            maximum_tilt,
        } => {
            let normal = normalize(normal);
            let in_plane_a = perpendicular(normal);
            let in_plane_b = normalize(cross(normal, in_plane_a));
            let azimuth = 2.0 * PI * random.unit();
            let tilt = maximum_tilt * (2.0 * random.unit() - 1.0);
            add(
                scale(
                    add(
                        scale(in_plane_a, azimuth.cos()),
                        scale(in_plane_b, azimuth.sin()),
                    ),
                    tilt.cos(),
                ),
                scale(normal, tilt.sin()),
            )
        }
        OrientationDistribution::LayeredBiaxial {
            normal,
            primary_fraction,
            cross_fraction,
            maximum_in_plane_deviation,
            maximum_tilt,
            layer_seed,
        } => {
            let layer = formation_layer.expect("layered biaxial orientation requires a layer");
            let normal = normalize(normal);
            let in_plane_a = perpendicular(normal);
            let in_plane_b = normalize(cross(normal, in_plane_a));
            let reference = layer_reference_angle(layer_seed, layer);
            let selection = random.unit();
            let azimuth = if selection < primary_fraction {
                reference + maximum_in_plane_deviation * (2.0 * random.unit() - 1.0)
            } else if selection < primary_fraction + cross_fraction {
                reference + 0.5 * PI + maximum_in_plane_deviation * (2.0 * random.unit() - 1.0)
            } else {
                2.0 * PI * random.unit()
            };
            let tilt = maximum_tilt * (2.0 * random.unit() - 1.0);
            add(
                scale(
                    add(
                        scale(in_plane_a, azimuth.cos()),
                        scale(in_plane_b, azimuth.sin()),
                    ),
                    tilt.cos(),
                ),
                scale(normal, tilt.sin()),
            )
        }
        OrientationDistribution::Aligned {
            axis,
            maximum_angle,
        } => {
            let axis = normalize(axis);
            let transverse_a = perpendicular(axis);
            let transverse_b = normalize(cross(axis, transverse_a));
            let minimum_cosine = maximum_angle.cos();
            let cosine = minimum_cosine + (1.0 - minimum_cosine) * random.unit();
            let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
            let azimuth = 2.0 * PI * random.unit();
            add(
                scale(axis, cosine),
                scale(
                    add(
                        scale(transverse_a, azimuth.cos()),
                        scale(transverse_b, azimuth.sin()),
                    ),
                    sine,
                ),
            )
        }
    }
}

fn sample_contained_center(
    offsets: &[Vec3],
    radius: f64,
    box_low: Vec3,
    box_high: Vec3,
    periodic: [bool; 3],
    distribution: PositionDistribution,
    formation_layer: Option<u32>,
    random: &mut SplitMix64,
) -> Option<(Vec3, Option<u32>)> {
    let mut lower = [0.0; 3];
    let mut upper = [0.0; 3];
    for axis in 0..3 {
        if periodic[axis] {
            lower[axis] = box_low[axis];
            upper[axis] = box_high[axis];
            continue;
        }
        let minimum_offset = offsets
            .iter()
            .map(|point| point[axis])
            .fold(f64::INFINITY, f64::min);
        let maximum_offset = offsets
            .iter()
            .map(|point| point[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        lower[axis] = box_low[axis] + radius - minimum_offset;
        upper[axis] = box_high[axis] - radius - maximum_offset;
        if lower[axis] > upper[axis] {
            return None;
        }
    }

    let mut center = [0.0; 3];
    for axis in 0..3 {
        center[axis] = sample_range(lower[axis], upper[axis], random.unit());
    }
    let formation_layer = match distribution {
        PositionDistribution::Uniform => None,
        PositionDistribution::Layered {
            axis,
            layers,
            jitter_fraction,
        } => {
            let layer = formation_layer
                .expect("layered position distribution requires an assigned layer")
                as f64;
            let spacing = (upper[axis] - lower[axis]) / layers as f64;
            let jitter = jitter_fraction * (random.unit() - 0.5) * spacing;
            center[axis] =
                (lower[axis] + (layer + 0.5) * spacing + jitter).clamp(lower[axis], upper[axis]);
            Some(layer as u32)
        }
        PositionDistribution::DensityGradient {
            axis,
            exponent,
            toward_high,
        } => {
            let unit = random.unit();
            let biased = if toward_high {
                1.0 - (1.0 - unit).powf(exponent)
            } else {
                unit.powf(exponent)
            };
            center[axis] = sample_range(lower[axis], upper[axis], biased);
            None
        }
    };
    Some((center, formation_layer))
}

fn sample_formation_layer(
    distribution: PositionDistribution,
    random: &mut SplitMix64,
) -> Option<u32> {
    match distribution {
        PositionDistribution::Layered { layers, .. } => Some(
            (random.unit() * layers as f64)
                .floor()
                .min(layers as f64 - 1.0) as u32,
        ),
        PositionDistribution::Uniform | PositionDistribution::DensityGradient { .. } => None,
    }
}

fn layer_reference_angle(seed: u64, layer: u32) -> f64 {
    PI * splitmix64_unit(seed ^ (layer as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn splitmix64_unit(mut value: u64) -> f64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn sample_range(minimum: f64, maximum: f64, unit: f64) -> f64 {
    minimum + (maximum - minimum) * unit
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

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^= value >> 31;
        ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::PeriodicCell;

    #[test]
    fn same_seed_reproduces_identical_assembly() {
        let spec = FiberPopulationSpec {
            count: 12,
            seed: 42,
            ..FiberPopulationSpec::default()
        };
        let mut first = unit_assembly();
        let mut second = unit_assembly();
        generate_biased_fiber_population(&mut first, &spec).unwrap();
        generate_biased_fiber_population(&mut second, &spec).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn generated_swept_fibers_are_fully_contained() {
        let spec = FiberPopulationSpec {
            count: 24,
            position: PositionDistribution::Layered {
                axis: 2,
                layers: 4,
                jitter_fraction: 0.2,
            },
            ..FiberPopulationSpec::default()
        };
        let mut assembly = unit_assembly();
        generate_biased_fiber_population(&mut assembly, &spec).unwrap();
        assembly.validate().unwrap();

        for fiber in &assembly.topology.fibers {
            assert!(fiber.formation_layer.is_some_and(|layer| layer < 4));
            let radius = match assembly.sections.entries[fiber.section.0 as usize] {
                Section::Circular { radius } => radius,
                Section::Elliptical { .. } => unreachable!(),
            };
            let start = fiber.vertices.start as usize;
            let end = fiber.vertices.checked_end().unwrap() as usize;
            for point in &assembly.geometry.placed.positions[start..end] {
                assert!(point.iter().all(|coordinate| {
                    *coordinate >= radius - 1.0e-12 && *coordinate <= 1.0 - radius + 1.0e-12
                }));
            }
        }
    }

    #[test]
    fn periodic_axes_have_no_containment_edge_depletion() {
        let spec = FiberPopulationSpec {
            count: 64,
            segments_per_fiber: 4,
            length: ScalarDistribution::Constant(0.9),
            radius: ScalarDistribution::Constant(0.01),
            intrinsic_curvature_amplitude: ScalarDistribution::Constant(0.0),
            orientation: OrientationDistribution::Aligned {
                axis: [1.0, 0.0, 0.0],
                maximum_angle: 0.0,
            },
            seed: 9,
            ..FiberPopulationSpec::default()
        };
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [true, false, false]));
        generate_biased_fiber_population(&mut assembly, &spec).unwrap();

        assert!(assembly
            .geometry
            .placed
            .positions
            .iter()
            .any(|point| point[0] < 0.0 || point[0] > 1.0));
        assert!(assembly.geometry.placed.positions.iter().all(|point| {
            point[1] >= 0.01 && point[1] <= 0.99 && point[2] >= 0.01 && point[2] <= 0.99
        }));
    }

    #[test]
    fn planar_and_aligned_samples_respect_their_angular_support() {
        let maximum_tilt = 0.2;
        let maximum_angle = 0.25;
        let mut random = SplitMix64::new(17);
        for _ in 0..1_000 {
            let planar = sample_orientation(
                OrientationDistribution::Planar {
                    normal: [0.0, 0.0, 1.0],
                    maximum_tilt,
                },
                None,
                &mut random,
            );
            assert!(planar[2].abs() <= maximum_tilt.sin() + 1.0e-12);

            let aligned = sample_orientation(
                OrientationDistribution::Aligned {
                    axis: [1.0, 0.0, 0.0],
                    maximum_angle,
                },
                None,
                &mut random,
            );
            assert!(aligned[0] >= maximum_angle.cos() - 1.0e-12);
        }
    }

    #[test]
    fn layered_biaxial_samples_follow_the_requested_mixture() {
        let seed = 83;
        let layer = 3;
        let reference = layer_reference_angle(seed, layer);
        let mut random = SplitMix64::new(19);
        let mut primary = 0;
        let mut cross = 0;
        for _ in 0..10_000 {
            let direction = sample_orientation(
                OrientationDistribution::LayeredBiaxial {
                    normal: [0.0, 0.0, 1.0],
                    primary_fraction: 0.4,
                    cross_fraction: 0.4,
                    maximum_in_plane_deviation: 0.0,
                    maximum_tilt: 0.0,
                    layer_seed: seed,
                },
                Some(layer),
                &mut random,
            );
            let azimuth = (-direction[0]).atan2(direction[1]);
            if undirected_angle_difference(azimuth, reference) < 1.0e-10 {
                primary += 1;
            } else if undirected_angle_difference(azimuth, reference + 0.5 * PI) < 1.0e-10 {
                cross += 1;
            }
        }
        assert!((3_700..=4_300).contains(&primary), "primary={primary}");
        assert!((3_700..=4_300).contains(&cross), "cross={cross}");
    }

    fn undirected_angle_difference(first: f64, second: f64) -> f64 {
        ((first - second + 0.5 * PI).rem_euclid(PI) - 0.5 * PI).abs()
    }

    fn unit_assembly() -> FiberAssembly {
        FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]))
    }
}
