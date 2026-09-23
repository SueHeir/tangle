//! Fiber shape statistics: curvature, torsion, tangent correlation, curl and
//! the orientation distribution.
//!
//! Volume fraction and orientation tensors say how much fiber there is and
//! which way it points on average. These metrics describe how individual
//! fibers bend and wander, so a generator's waviness can be calibrated against
//! centerlines tracked from a CT scan. Like the neighbor analysis they use only
//! placed centerlines, so the same call applies to both.
//!
//! Every fiber is resampled at a uniform arc-length `sample_spacing`, and the
//! chords between successive samples are its tangents. Curvature and torsion
//! are measured at that scale, not at the scale of the stored polyline, so two
//! structures are only comparable when they are analyzed with the same
//! spacing:
//!
//! * **curvature** at a sample is the turning angle between the chords on
//!   either side divided by the spacing (exact for a circular arc);
//! * **torsion** is the signed rotation of the binormal about the tangent per
//!   unit length, positive for a right-handed helix. It is undefined where a
//!   fiber is nearly straight, so it is only measured where the curvature on
//!   both sides is at least `minimum_torsion_curvature`;
//! * **tangent correlation** `C(s)` is the mean of `t(u) · t(u + s)` over
//!   every fiber and position;
//! * **curl index** is contour length over end-to-end distance minus one
//!   (zero for a straight fiber);
//! * the **Schladitz β** is the maximum-likelihood fit of the one-parameter
//!   orientation density `β / (4π (1 + (β² − 1) cos² θ)^{3/2})` about an axis:
//!   `β = 1` is isotropic, `β → 0` aligns fibers with the axis and `β → ∞`
//!   lays them in the plane normal to it (the usual model for nonwovens, with
//!   the axis through the thickness).

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use tangle_core::{FiberAssembly, FiberId, Vec3};

use crate::distribution::Distribution;
use crate::neighbors::{add, collect_fibers, decay_length, dot, norm, scale, sub, FiberPath};

/// Schema version of [`ShapeMetrics`].
pub const SHAPE_SCHEMA_VERSION: u32 = 1;

/// Range of `ln β` searched by the Schladitz fit.
const LOG_BETA_LIMIT: f64 = 12.0;

/// Correlation below which the persistence-length fit stops using lags.
const PERSISTENCE_FIT_FLOOR: f64 = 0.05;

/// Settings for [`analyze_shape`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShapeAnalysisConfig {
    /// Arc-length spacing at which fibers are resampled. `None` uses the
    /// larger of the smallest fiber diameter and the median polyline segment
    /// length: CT trackers resolve direction changes over about one diameter,
    /// and a spacing below the segment length would only see the polyline's
    /// corners.
    pub sample_spacing: Option<f64>,
    /// Largest arc-length lag of the tangent-correlation curve. `None` uses the
    /// smaller of half the median fiber length and 200 sample spacings.
    pub maximum_lag: Option<f64>,
    /// Number of logarithmically spaced lags in the tangent-correlation curve.
    pub lag_count: usize,
    /// Number of evenly spaced quantiles stored for each distribution (at
    /// least two).
    pub quantile_count: usize,
    /// Reference axis for the orientation distribution and the Schladitz fit.
    /// It need not be normalized. The default is the z axis, through the
    /// thickness of a layered structure.
    pub orientation_axis: Vec3,
    /// Smallest curvature at which torsion is measured. `None` uses a turning
    /// angle of 0.02 radians per sample spacing.
    pub minimum_torsion_curvature: Option<f64>,
}

impl Default for ShapeAnalysisConfig {
    fn default() -> Self {
        Self {
            sample_spacing: None,
            maximum_lag: None,
            lag_count: 24,
            quantile_count: 101,
            orientation_axis: [0.0, 0.0, 1.0],
            minimum_torsion_curvature: None,
        }
    }
}

/// Shape summary for one fiber.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberShapeMetrics {
    /// Stable source fiber identifier.
    pub fiber_id: FiberId,
    /// Placed centerline length.
    pub length: f64,
    /// Distance between the first and last centerline vertices.
    pub end_to_end_distance: f64,
    /// Contour length over end-to-end distance, minus one. `None` for a closed
    /// loop.
    pub curl_index: Option<f64>,
    /// Mean sampled curvature. `None` when the fiber is shorter than two
    /// sample spacings.
    pub mean_curvature: Option<f64>,
    /// Root-mean-square sampled curvature.
    pub rms_curvature: Option<f64>,
    /// Largest sampled curvature.
    pub maximum_curvature: Option<f64>,
    /// Mean absolute torsion where torsion is defined.
    pub mean_absolute_torsion: Option<f64>,
}

/// Fiber shape statistics of one assembly state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShapeMetrics {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Arc-length sample spacing actually used.
    pub sample_spacing: f64,
    /// Curvature threshold for torsion actually used.
    pub minimum_torsion_curvature: f64,
    /// Unit orientation axis used.
    pub orientation_axis: Vec3,
    /// Number of fibers analyzed.
    pub fiber_count: usize,
    /// Number of resampled points.
    pub samples: usize,
    /// Total placed centerline length.
    pub total_length: f64,
    /// Sampled curvature, one value per interior sample.
    pub curvature: Option<Distribution>,
    /// Signed torsion where it is defined.
    pub torsion: Option<Distribution>,
    /// Absolute torsion where it is defined.
    pub absolute_torsion: Option<Distribution>,
    /// Fraction of torsion sites whose curvature met the threshold.
    pub torsion_defined_fraction: Option<f64>,
    /// Curl index, one value per fiber.
    pub curl_index: Option<Distribution>,
    /// Placed fiber length, one value per fiber.
    pub fiber_length: Option<Distribution>,
    /// `|cos θ|` between each sampled chord and the orientation axis.
    pub axis_cosine: Option<Distribution>,
    /// Mean of `cos² θ` over sampled chords; one third for isotropic fibers.
    /// It equals the orientation tensor's axis-axis component.
    pub mean_squared_axis_cosine: Option<f64>,
    /// Maximum-likelihood Schladitz β, clamped to `[e⁻¹², e¹²]`.
    pub schladitz_beta: Option<f64>,
    /// Wasserstein distance between the observed `|cos θ|` distribution and
    /// the fitted model's. Near zero when the one-parameter model describes
    /// the orientations; large when they are, for example, bimodal.
    pub schladitz_fit_distance: Option<f64>,
    /// Arc-length lags of the tangent-correlation curve.
    pub tangent_correlation_lags: Vec<f64>,
    /// Mean `t(u) · t(u + lag)` at each lag.
    pub tangent_correlation: Vec<Option<f64>>,
    /// Lag at which the tangent correlation first falls to `1/e`,
    /// interpolated from one at zero lag. `None` when it stays above `1/e`.
    pub tangent_correlation_length: Option<f64>,
    /// `L` of the least-squares fit `C(s) = exp(−s / L)` over the lags before
    /// the correlation first drops below 0.05. This is the persistence length
    /// of a three-dimensional worm-like chain; a chain confined to a plane
    /// decays as `exp(−s / 2L_p)`, so its planar persistence length is half
    /// this value. `None` when the correlation does not decay.
    pub persistence_length: Option<f64>,
    /// Per-fiber summaries in topology order.
    pub fibers: Vec<FiberShapeMetrics>,
}

/// Invalid shape-analysis settings.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeAnalysisError {
    /// A length setting was not positive and finite.
    InvalidLength {
        /// Setting name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// The lag count was zero.
    InvalidLagCount,
    /// Fewer than two quantiles were requested.
    InvalidQuantileCount(usize),
    /// The orientation axis had zero or non-finite length.
    InvalidOrientationAxis(Vec3),
}

impl fmt::Display for ShapeAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { name, value } => {
                write!(formatter, "{name} must be positive and finite, got {value}")
            }
            Self::InvalidLagCount => write!(formatter, "lag_count must be at least 1"),
            Self::InvalidQuantileCount(count) => {
                write!(formatter, "quantile_count must be at least 2, got {count}")
            }
            Self::InvalidOrientationAxis(axis) => write!(
                formatter,
                "orientation_axis must be a nonzero finite vector, got {axis:?}"
            ),
        }
    }
}

impl Error for ShapeAnalysisError {}

/// Measures curvature, torsion, tangent correlation, curl and orientation.
pub fn analyze_shape(
    assembly: &FiberAssembly,
    config: &ShapeAnalysisConfig,
) -> Result<ShapeMetrics, ShapeAnalysisError> {
    let fibers = collect_fibers(assembly);
    let settings = resolve_settings(config, &fibers)?;
    let quantiles = config.quantile_count;

    let mut curvatures = Vec::new();
    let mut torsions = Vec::new();
    let mut torsion_sites = 0usize;
    let mut axis_cosines = Vec::new();
    let mut samples = 0usize;
    let mut per_fiber = Vec::with_capacity(fibers.len());
    let mut chains = Vec::with_capacity(fibers.len());

    for fiber in &fibers {
        let points = resample(fiber, settings.spacing);
        samples += points.len();
        let tangents = chord_tangents(&points);
        axis_cosines.extend(
            tangents
                .iter()
                .flatten()
                .map(|t| dot(*t, settings.axis).abs().min(1.0)),
        );

        // Curvature at interior sample j + 1, between chords j and j + 1.
        let fiber_curvatures: Vec<Option<f64>> = tangents
            .windows(2)
            .map(|pair| match (pair[0], pair[1]) {
                (Some(a), Some(b)) => Some(angle_between(a, b) / settings.spacing),
                _ => None,
            })
            .collect();
        let mut fiber_torsions = Vec::new();
        for j in 0..fiber_curvatures.len().saturating_sub(1) {
            torsion_sites += 1;
            let defined = [fiber_curvatures[j], fiber_curvatures[j + 1]]
                .iter()
                .all(|k| k.is_some_and(|k| k >= settings.minimum_torsion_curvature));
            if !defined {
                continue;
            }
            let (Some(t0), Some(t1), Some(t2)) = (tangents[j], tangents[j + 1], tangents[j + 2])
            else {
                continue;
            };
            let b0 = normalized(cross(t0, t1));
            let b1 = normalized(cross(t1, t2));
            if let (Some(b0), Some(b1)) = (b0, b1) {
                let angle = dot(cross(b0, b1), t1).atan2(dot(b0, b1));
                fiber_torsions.push(angle / settings.spacing);
            }
        }

        let sampled: Vec<f64> = fiber_curvatures.iter().flatten().copied().collect();
        let end_to_end = norm(sub(
            *fiber.points.last().expect("collected fibers have vertices"),
            fiber.points[0],
        ));
        per_fiber.push(FiberShapeMetrics {
            fiber_id: fiber.id,
            length: fiber.length,
            end_to_end_distance: end_to_end,
            curl_index: (end_to_end > 0.0).then(|| fiber.length / end_to_end - 1.0),
            mean_curvature: mean(&sampled),
            rms_curvature: mean(&sampled.iter().map(|k| k * k).collect::<Vec<_>>()).map(f64::sqrt),
            maximum_curvature: sampled.iter().copied().reduce(f64::max),
            mean_absolute_torsion: mean(
                &fiber_torsions.iter().map(|t| t.abs()).collect::<Vec<_>>(),
            ),
        });
        curvatures.extend(sampled);
        torsions.extend(fiber_torsions);
        chains.push(tangents);
    }

    let lag_steps = lag_steps(&fibers, &settings);
    let tangent_correlation = tangent_correlation(&chains, &lag_steps);
    let tangent_correlation_lags: Vec<f64> = lag_steps
        .iter()
        .map(|step| *step as f64 * settings.spacing)
        .collect();

    let axis_cosine = Distribution::from_values(axis_cosines.iter().copied(), quantiles);
    let schladitz_beta = fit_schladitz_beta(&axis_cosines);
    let schladitz_fit_distance = schladitz_beta
        .zip(axis_cosine.as_ref())
        .map(|(beta, observed)| schladitz_distance(beta, observed));

    Ok(ShapeMetrics {
        schema_version: SHAPE_SCHEMA_VERSION,
        sample_spacing: settings.spacing,
        minimum_torsion_curvature: settings.minimum_torsion_curvature,
        orientation_axis: settings.axis,
        fiber_count: fibers.len(),
        samples,
        total_length: fibers.iter().map(|f| f.length).sum(),
        curvature: Distribution::from_values(curvatures, quantiles),
        absolute_torsion: Distribution::from_values(torsions.iter().map(|t| t.abs()), quantiles),
        torsion_defined_fraction: (torsion_sites > 0)
            .then(|| torsions.len() as f64 / torsion_sites as f64),
        torsion: Distribution::from_values(torsions, quantiles),
        curl_index: Distribution::from_values(
            per_fiber.iter().filter_map(|f| f.curl_index),
            quantiles,
        ),
        fiber_length: Distribution::from_values(per_fiber.iter().map(|f| f.length), quantiles),
        mean_squared_axis_cosine: mean(&axis_cosines.iter().map(|c| c * c).collect::<Vec<_>>()),
        axis_cosine,
        schladitz_beta,
        schladitz_fit_distance,
        tangent_correlation_length: decay_length(&tangent_correlation_lags, &tangent_correlation),
        persistence_length: persistence_length(&tangent_correlation_lags, &tangent_correlation),
        tangent_correlation_lags,
        tangent_correlation,
        fibers: per_fiber,
    })
}

// ---------------------------------------------------------------------------
// Settings

struct Settings {
    spacing: f64,
    maximum_lag: Option<f64>,
    lag_count: usize,
    axis: Vec3,
    minimum_torsion_curvature: f64,
}

fn resolve_settings(
    config: &ShapeAnalysisConfig,
    fibers: &[FiberPath],
) -> Result<Settings, ShapeAnalysisError> {
    let positive = |name, value: f64| {
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(ShapeAnalysisError::InvalidLength { name, value })
        }
    };
    if config.lag_count == 0 {
        return Err(ShapeAnalysisError::InvalidLagCount);
    }
    if config.quantile_count < 2 {
        return Err(ShapeAnalysisError::InvalidQuantileCount(
            config.quantile_count,
        ));
    }
    let axis = normalized(config.orientation_axis)
        .filter(|axis| axis.iter().all(|c| c.is_finite()))
        .ok_or(ShapeAnalysisError::InvalidOrientationAxis(
            config.orientation_axis,
        ))?;
    let spacing = positive(
        "sample_spacing",
        config
            .sample_spacing
            .unwrap_or_else(|| default_spacing(fibers)),
    )?;
    let maximum_lag = config
        .maximum_lag
        .map(|lag| positive("maximum_lag", lag))
        .transpose()?;
    let minimum_torsion_curvature = match config.minimum_torsion_curvature {
        Some(value) if value.is_finite() && value >= 0.0 => value,
        Some(value) => {
            return Err(ShapeAnalysisError::InvalidLength {
                name: "minimum_torsion_curvature",
                value,
            })
        }
        None => 0.02 / spacing,
    };
    Ok(Settings {
        spacing,
        maximum_lag,
        lag_count: config.lag_count,
        axis,
        minimum_torsion_curvature,
    })
}

fn default_spacing(fibers: &[FiberPath]) -> f64 {
    let diameter = fibers
        .iter()
        .map(|f| 2.0 * f.radius)
        .filter(|d| *d > 0.0)
        .fold(f64::INFINITY, f64::min);
    let mut segments: Vec<f64> = fibers
        .iter()
        .flat_map(|f| f.arc.windows(2).map(|pair| pair[1] - pair[0]))
        .collect();
    segments.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_segment = segments.get(segments.len() / 2).copied().unwrap_or(0.0);
    let spacing = if diameter.is_finite() {
        diameter.max(median_segment)
    } else {
        median_segment
    };
    if spacing > 0.0 {
        spacing
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Sampling

/// Points at arc lengths `0, h, 2h, …` up to the fiber length.
fn resample(fiber: &FiberPath, spacing: f64) -> Vec<Vec3> {
    let count = (fiber.length / spacing).floor() as usize + 1;
    let mut points = Vec::with_capacity(count);
    let mut segment = 0;
    for step in 0..count {
        let arc = step as f64 * spacing;
        while segment + 2 < fiber.points.len() && fiber.arc[segment + 1] <= arc {
            segment += 1;
        }
        let a = fiber.points[segment];
        let b = fiber.points[segment + 1];
        let segment_length = fiber.arc[segment + 1] - fiber.arc[segment];
        let t = if segment_length > 0.0 {
            ((arc - fiber.arc[segment]) / segment_length).clamp(0.0, 1.0)
        } else {
            0.0
        };
        points.push(add(a, scale(sub(b, a), t)));
    }
    points
}

/// Unit chord directions; `None` where two samples coincide.
fn chord_tangents(points: &[Vec3]) -> Vec<Option<Vec3>> {
    points
        .windows(2)
        .map(|pair| normalized(sub(pair[1], pair[0])))
        .collect()
}

fn lag_steps(fibers: &[FiberPath], settings: &Settings) -> Vec<usize> {
    let mut lengths: Vec<f64> = fibers.iter().map(|f| f.length).collect();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_length = lengths.get(lengths.len() / 2).copied().unwrap_or(0.0);
    let maximum_lag = settings
        .maximum_lag
        .unwrap_or_else(|| (0.5 * median_length).min(200.0 * settings.spacing));
    let maximum_step = ((maximum_lag / settings.spacing).round() as usize).max(1);
    let mut steps: Vec<usize> = (0..settings.lag_count)
        .map(|index| {
            let fraction = if settings.lag_count > 1 {
                index as f64 / (settings.lag_count - 1) as f64
            } else {
                1.0
            };
            (maximum_step as f64).powf(fraction).round() as usize
        })
        .collect();
    steps.dedup();
    steps
}

fn tangent_correlation(chains: &[Vec<Option<Vec3>>], lag_steps: &[usize]) -> Vec<Option<f64>> {
    lag_steps
        .iter()
        .map(|&lag| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for chain in chains {
                for index in 0..chain.len().saturating_sub(lag) {
                    if let (Some(a), Some(b)) = (chain[index], chain[index + lag]) {
                        sum += dot(a, b);
                        count += 1;
                    }
                }
            }
            (count > 0).then(|| sum / count as f64)
        })
        .collect()
}

fn persistence_length(lags: &[f64], curve: &[Option<f64>]) -> Option<f64> {
    // Least squares of ln C = −s / L through the origin.
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (lag, value) in lags.iter().zip(curve) {
        let Some(value) = *value else { continue };
        if value < PERSISTENCE_FIT_FLOOR {
            break;
        }
        numerator += lag * lag;
        // Rounding can leave a perfectly straight chain just below one.
        if value < 1.0 - 1e-12 {
            denominator += lag * value.ln();
        }
    }
    (denominator < 0.0 && numerator > 0.0)
        .then(|| -numerator / denominator)
        .filter(|length| length.is_finite())
}

// ---------------------------------------------------------------------------
// Schladitz orientation model

/// Maximum-likelihood β for `|cos θ|` samples, searched over
/// `ln β ∈ [−12, 12]` on a coarse grid and refined by golden-section search.
fn fit_schladitz_beta(cosines: &[f64]) -> Option<f64> {
    if cosines.is_empty() {
        return None;
    }
    let squares: Vec<f64> = cosines.iter().map(|c| c * c).collect();
    let n = squares.len() as f64;
    let log_likelihood = |log_beta: f64| {
        let excess = (2.0 * log_beta).exp() - 1.0;
        n * log_beta - 1.5 * squares.iter().map(|c2| (excess * c2).ln_1p()).sum::<f64>()
    };
    let step = 0.5;
    let grid_points = (2.0 * LOG_BETA_LIMIT / step).round() as usize;
    let best = (0..=grid_points)
        .map(|k| -LOG_BETA_LIMIT + k as f64 * step)
        .map(|x| (x, log_likelihood(x)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))?
        .0;
    let (mut low, mut high) = (
        (best - step).max(-LOG_BETA_LIMIT),
        (best + step).min(LOG_BETA_LIMIT),
    );
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..60 {
        let first = high - ratio * (high - low);
        let second = low + ratio * (high - low);
        if log_likelihood(first) < log_likelihood(second) {
            low = first;
        } else {
            high = second;
        }
    }
    Some((0.5 * (low + high)).exp())
}

/// `|cos θ|` quantile of the Schladitz model: its CDF is
/// `β u / √(1 + (β² − 1) u²)`, which inverts in closed form.
fn schladitz_axis_cosine_quantile(beta: f64, p: f64) -> f64 {
    let beta2 = beta * beta;
    let p = p.clamp(0.0, 1.0);
    (p / (beta2 - (beta2 - 1.0) * p * p).sqrt()).min(1.0)
}

fn schladitz_distance(beta: f64, observed: &Distribution) -> f64 {
    let points = observed.quantiles.len().max(2);
    let step = 1.0 / (points - 1) as f64;
    let difference = |k: usize| {
        let p = k as f64 * step;
        (observed.quantile(p) - schladitz_axis_cosine_quantile(beta, p)).abs()
    };
    (0..points - 1)
        .map(|k| 0.5 * (difference(k) + difference(k + 1)) * step)
        .sum()
}

// ---------------------------------------------------------------------------
// Small helpers

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(value: Vec3) -> Option<Vec3> {
    let length = norm(value);
    (length > 0.0 && length.is_finite()).then(|| scale(value, length.recip()))
}

/// Angle between two unit vectors, accurate for small and large angles.
fn angle_between(a: Vec3, b: Vec3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use tangle_core::{PeriodicCell, Section};

    fn assembly_with(radius: f64, centerlines: &[Vec<Vec3>]) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([10.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius });
        for (index, centerline) in centerlines.iter().enumerate() {
            assembly
                .add_fiber(
                    FiberId(index as u32 + 1),
                    material,
                    section,
                    centerline,
                    centerline,
                )
                .unwrap();
        }
        assembly
    }

    fn with_spacing(spacing: f64) -> ShapeAnalysisConfig {
        ShapeAnalysisConfig {
            sample_spacing: Some(spacing),
            ..ShapeAnalysisConfig::default()
        }
    }

    /// Helix `(a cos t, a sin t, c t)` with `segments` vertices per turn.
    fn helix(a: f64, c: f64, turns: f64, segments_per_turn: usize) -> Vec<Vec3> {
        let segments = (turns * segments_per_turn as f64) as usize;
        (0..=segments)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / segments_per_turn as f64;
                [a * t.cos(), a * t.sin(), c * t]
            })
            .collect()
    }

    fn relative_error(value: f64, expected: f64) -> f64 {
        ((value - expected) / expected).abs()
    }

    /// Deterministic uniform and normal variates for synthetic fibers.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            (-2.0 * self.uniform().ln()).sqrt() * (2.0 * PI * self.uniform()).cos()
        }
    }

    #[test]
    fn a_planar_arc_has_its_curvature_and_no_torsion() {
        let radius = 2.0;
        let arc: Vec<Vec3> = (0..=4000)
            .map(|i| {
                let t = PI * i as f64 / 4000.0;
                [radius * t.cos(), radius * t.sin(), 0.0]
            })
            .collect();
        let metrics = analyze_shape(&assembly_with(0.01, &[arc]), &with_spacing(0.05)).unwrap();
        let curvature = metrics.curvature.unwrap();
        assert!(relative_error(curvature.quantile(0.0), 0.5) < 1e-3);
        assert!(relative_error(curvature.quantile(1.0), 0.5) < 1e-3);
        let torsion = metrics.absolute_torsion.unwrap();
        assert!(torsion.quantile(1.0) < 1e-6);
        assert_eq!(metrics.torsion_defined_fraction, Some(1.0));
        // Half circle: contour πR over diameter 2R.
        assert!(relative_error(metrics.fibers[0].curl_index.unwrap(), PI / 2.0 - 1.0) < 1e-4);
    }

    #[test]
    fn helices_have_their_curvature_and_signed_torsion() {
        let (a, c) = (1.0, 0.5);
        let denominator = a * a + c * c;
        let expected_curvature = a / denominator;
        let expected_torsion = c / denominator;
        // Fine vertices keep resampling error well below the torsion signal.
        let right = helix(a, c, 4.0, 20_000);
        let left: Vec<Vec3> = right.iter().map(|p| [p[0], -p[1], p[2]]).collect();
        let metrics =
            analyze_shape(&assembly_with(0.01, &[right, left]), &with_spacing(0.05)).unwrap();
        assert!(relative_error(metrics.curvature.unwrap().median(), expected_curvature) < 5e-3);
        let torsion = metrics.torsion.unwrap();
        // The right-handed fiber gives +τ, its mirror image −τ.
        assert!(relative_error(torsion.quantile(0.75), expected_torsion) < 5e-3);
        assert!(relative_error(-torsion.quantile(0.25), expected_torsion) < 5e-3);
        assert!(
            relative_error(metrics.absolute_torsion.unwrap().median(), expected_torsion) < 5e-3
        );
        let per_fiber = &metrics.fibers[0];
        assert!(relative_error(per_fiber.mean_curvature.unwrap(), expected_curvature) < 5e-3);
    }

    #[test]
    fn straight_fibers_are_uncurled_and_perfectly_correlated() {
        let fibers = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.0], [0.0, 4.0, 0.0]],
        ];
        let metrics = analyze_shape(&assembly_with(0.05, &fibers), &with_spacing(0.1)).unwrap();
        assert!(metrics.curvature.unwrap().quantile(1.0) < 1e-12);
        assert!(metrics.torsion.is_none());
        assert_eq!(metrics.torsion_defined_fraction, Some(0.0));
        assert!(metrics.curl_index.unwrap().quantile(1.0).abs() < 1e-12);
        assert!(metrics
            .tangent_correlation
            .iter()
            .all(|c| (c.unwrap() - 1.0).abs() < 1e-12));
        assert_eq!(metrics.tangent_correlation_length, None);
        assert_eq!(metrics.persistence_length, None);
        // Both fibers lie in the xy plane, normal to the default axis.
        assert!(metrics.mean_squared_axis_cosine.unwrap() < 1e-12);
        assert!(metrics.schladitz_beta.unwrap() > 1e4);
    }

    #[test]
    fn a_right_angle_bend_has_the_expected_curl_index() {
        let fiber = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let metrics = analyze_shape(&assembly_with(0.05, &[fiber]), &with_spacing(0.1)).unwrap();
        let curl = metrics.fibers[0].curl_index.unwrap();
        assert!((curl - (2.0 / 2.0_f64.sqrt() - 1.0)).abs() < 1e-12);
        // The corner falls on a sample, so one turning angle is π/2.
        let maximum = metrics.fibers[0].maximum_curvature.unwrap();
        assert!((maximum - 0.5 * PI / 0.1).abs() < 1e-6);
    }

    #[test]
    fn worm_like_chains_recover_their_persistence_length() {
        // Each step perturbs the tangent isotropically, so C(k) = C(1)^k and
        // the persistence length is −h / ln C(1).
        let mut rng = Rng(7);
        let (step, sigma) = (0.1, 0.2);
        let mut fibers = Vec::new();
        let mut consecutive = (0.0, 0usize);
        for _ in 0..300 {
            let mut tangent: Vec3 = normalized([rng.normal(), rng.normal(), rng.normal()]).unwrap();
            let mut point = [0.0; 3];
            let mut fiber = vec![point];
            for _ in 0..400 {
                point = add(point, scale(tangent, step));
                fiber.push(point);
                let next = normalized(add(
                    tangent,
                    [
                        sigma * rng.normal(),
                        sigma * rng.normal(),
                        sigma * rng.normal(),
                    ],
                ))
                .unwrap();
                consecutive.0 += dot(tangent, next);
                consecutive.1 += 1;
                tangent = next;
            }
            fibers.push(fiber);
        }
        let expected = -step / (consecutive.0 / consecutive.1 as f64).ln();
        let config = ShapeAnalysisConfig {
            sample_spacing: Some(step),
            maximum_lag: Some(1.5 * expected),
            ..ShapeAnalysisConfig::default()
        };
        let metrics = analyze_shape(&assembly_with(0.01, &fibers), &config).unwrap();
        let persistence = metrics.persistence_length.unwrap();
        assert!(
            relative_error(persistence, expected) < 0.08,
            "{persistence} vs {expected}"
        );
        let decay = metrics.tangent_correlation_length.unwrap();
        assert!(
            relative_error(decay, expected) < 0.1,
            "{decay} vs {expected}"
        );
        // A random walk has no preferred handedness.
        let torsion = metrics.torsion.unwrap();
        assert!(torsion.mean.abs() < 0.05 * torsion.standard_deviation);
    }

    #[test]
    fn schladitz_fit_recovers_beta_from_model_samples() {
        for beta in [0.3, 1.0, 4.0] {
            let mut rng = Rng(11);
            let fibers: Vec<Vec<Vec3>> = (0..20_000)
                .map(|_| {
                    let cosine = schladitz_axis_cosine_quantile(beta, rng.uniform())
                        * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                    let sine = (1.0 - cosine * cosine).sqrt();
                    let phi = 2.0 * PI * rng.uniform();
                    let direction = [sine * phi.cos(), sine * phi.sin(), cosine];
                    vec![[0.0; 3], direction]
                })
                .collect();
            let metrics = analyze_shape(&assembly_with(0.01, &fibers), &with_spacing(0.5)).unwrap();
            let fitted = metrics.schladitz_beta.unwrap();
            assert!(
                relative_error(fitted, beta) < 0.06,
                "β {beta}: fitted {fitted}"
            );
            assert!(metrics.schladitz_fit_distance.unwrap() < 0.02);
        }
    }

    #[test]
    fn a_bimodal_orientation_is_a_poor_schladitz_fit() {
        // Half the fibers along the axis, half normal to it.
        let fibers: Vec<Vec<Vec3>> = (0..200)
            .map(|i| {
                if i % 2 == 0 {
                    vec![[0.0; 3], [0.0, 0.0, 1.0]]
                } else {
                    vec![[0.0; 3], [1.0, 0.0, 0.0]]
                }
            })
            .collect();
        let metrics = analyze_shape(&assembly_with(0.01, &fibers), &with_spacing(0.25)).unwrap();
        assert!(metrics.schladitz_fit_distance.unwrap() > 0.2);
        assert!((metrics.mean_squared_axis_cosine.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn default_spacing_is_the_larger_of_diameter_and_median_segment() {
        let fine = vec![helix(1.0, 0.2, 1.0, 400)];
        let metrics =
            analyze_shape(&assembly_with(0.05, &fine), &ShapeAnalysisConfig::default()).unwrap();
        assert!((metrics.sample_spacing - 0.1).abs() < 1e-12);
        let coarse = vec![vec![[0.0; 3], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]];
        let metrics = analyze_shape(
            &assembly_with(0.05, &coarse),
            &ShapeAnalysisConfig::default(),
        )
        .unwrap();
        assert!((metrics.sample_spacing - 0.5).abs() < 1e-12);
        assert!((metrics.minimum_torsion_curvature - 0.04).abs() < 1e-12);
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let assembly = assembly_with(0.05, &[vec![[0.0; 3], [1.0, 0.0, 0.0]]]);
        let check = |config: ShapeAnalysisConfig| analyze_shape(&assembly, &config).unwrap_err();
        assert!(matches!(
            check(with_spacing(0.0)),
            ShapeAnalysisError::InvalidLength {
                name: "sample_spacing",
                ..
            }
        ));
        assert_eq!(
            check(ShapeAnalysisConfig {
                lag_count: 0,
                ..ShapeAnalysisConfig::default()
            }),
            ShapeAnalysisError::InvalidLagCount
        );
        assert_eq!(
            check(ShapeAnalysisConfig {
                quantile_count: 1,
                ..ShapeAnalysisConfig::default()
            }),
            ShapeAnalysisError::InvalidQuantileCount(1)
        );
        assert!(matches!(
            check(ShapeAnalysisConfig {
                orientation_axis: [0.0; 3],
                ..ShapeAnalysisConfig::default()
            }),
            ShapeAnalysisError::InvalidOrientationAxis(_)
        ));
        assert!(matches!(
            check(ShapeAnalysisConfig {
                minimum_torsion_curvature: Some(-1.0),
                ..ShapeAnalysisConfig::default()
            }),
            ShapeAnalysisError::InvalidLength {
                name: "minimum_torsion_curvature",
                ..
            }
        ));
    }

    #[test]
    fn empty_assemblies_and_short_fibers_are_handled() {
        let empty = assembly_with(0.05, &[]);
        let metrics = analyze_shape(&empty, &ShapeAnalysisConfig::default()).unwrap();
        assert_eq!(metrics.fiber_count, 0);
        assert!(metrics.curvature.is_none());
        assert!(metrics.schladitz_beta.is_none());
        assert!(metrics.tangent_correlation.iter().all(Option::is_none));

        // Shorter than one spacing: a length and a curl index, no curvature.
        let short = assembly_with(0.05, &[vec![[0.0; 3], [0.05, 0.0, 0.0]]]);
        let metrics = analyze_shape(&short, &with_spacing(0.1)).unwrap();
        assert_eq!(metrics.fiber_count, 1);
        assert!(metrics.curvature.is_none());
        assert_eq!(metrics.fibers[0].curl_index, Some(0.0));
        assert!(metrics.fibers[0].mean_curvature.is_none());
    }
}
