//! Scores a generated structure against a reference (usually a CT scan).
//!
//! A single number cannot say whether a generated structure "looks like" a
//! scan, and neither can raw differences: a curvature distribution that
//! differs by 10% may be well inside the scan's own variation or far outside
//! it. The scorecard therefore measures every metric in several subvolumes of
//! the same physical size and normalizes each difference by the reference's
//! own subvolume-to-subvolume spread:
//!
//! ```text
//! score = median over (candidate, reference) subvolume pairs of distance
//!         ───────────────────────────────────────────────────────────────
//!         median over (reference, reference) subvolume pairs of distance
//! ```
//!
//! Distances are absolute differences for scalar metrics and first
//! Wasserstein distances for distributions. A score near one means the
//! candidate differs from the scan about as much as the scan differs from
//! itself at that scale; well above one is a real difference. Every metric is
//! computed with the same resolved settings (sample spacing, gaps, lags) on
//! both sides, taken from the whole reference region when not given.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use tangle_core::{FiberAssembly, FiberId, PeriodicCell, Vec3};

use crate::distribution::Distribution;
use crate::neighbors::{add, norm, scale, sub};
use crate::{
    analyze_neighbors, analyze_shape, characterize_assembly, NeighborAnalysisConfig,
    NeighborAnalysisError, ShapeAnalysisConfig, ShapeAnalysisError,
};

/// Schema version of [`Scorecard`] and [`StructureProfile`].
pub const SCORECARD_SCHEMA_VERSION: u32 = 1;

/// The metrics of one structure (or subvolume) that the scorecard compares.
///
/// Scalars and distributions are keyed by metric name. A value is `None` when
/// the metric is undefined for that structure, for example a persistence
/// length when tangents never decorrelate.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StructureProfile {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Number of fibers (or fiber pieces, after cropping).
    pub fibers: usize,
    /// Scalar metrics.
    pub scalars: BTreeMap<String, Option<f64>>,
    /// Distribution metrics.
    pub distributions: BTreeMap<String, Option<Distribution>>,
}

/// Measures the scorecard metrics of one assembly.
///
/// Scalars: `volume_fraction`, `length_density`, `mean_squared_axis_cosine`,
/// `log_schladitz_beta`, `persistence_length`, `tangent_correlation_length`,
/// `contacts_per_length`, `contact_ratio_to_random`,
/// `in_axis_contact_fraction`, `mean_neighbors`, `neighbor_correlation_length`.
///
/// Distributions: `curvature`, `absolute_torsion`, `curl_index`,
/// `axis_cosine`, `fiber_length`, `crossing_angle` (radians), `free_length`,
/// `excess_persistence`.
pub fn profile_structure(
    assembly: &FiberAssembly,
    shape: &ShapeAnalysisConfig,
    neighbors: &NeighborAnalysisConfig,
) -> Result<StructureProfile, ScorecardError> {
    let basic = characterize_assembly(assembly);
    let shape_metrics = analyze_shape(assembly, shape)?;
    let neighbor_metrics = analyze_neighbors(assembly, neighbors)?;
    let quantiles = shape.quantile_count;
    let volume = basic.cell.volume;
    let has_length = neighbor_metrics.total_length > 0.0;

    let scalars: BTreeMap<String, Option<f64>> = [
        (
            "volume_fraction",
            (volume > 0.0).then_some(basic.nominal_swept_volume_fraction),
        ),
        (
            "length_density",
            (volume > 0.0).then(|| basic.total_centerline_length / volume),
        ),
        (
            "mean_squared_axis_cosine",
            shape_metrics.mean_squared_axis_cosine,
        ),
        (
            "log_schladitz_beta",
            shape_metrics.schladitz_beta.map(f64::ln),
        ),
        ("persistence_length", shape_metrics.persistence_length),
        (
            "tangent_correlation_length",
            shape_metrics.tangent_correlation_length,
        ),
        (
            "contacts_per_length",
            has_length.then_some(neighbor_metrics.contacts_per_length),
        ),
        (
            "contact_ratio_to_random",
            neighbor_metrics.contact_ratio_to_random,
        ),
        (
            "in_axis_contact_fraction",
            (neighbor_metrics.contacts > 0).then_some(neighbor_metrics.in_axis_contact_fraction),
        ),
        (
            "mean_neighbors",
            (neighbor_metrics.samples > 0).then_some(neighbor_metrics.mean_neighbors),
        ),
        (
            "neighbor_correlation_length",
            neighbor_metrics.neighbor_correlation_length,
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();

    let events = &neighbor_metrics.events;
    let distributions: BTreeMap<String, Option<Distribution>> = [
        ("curvature", shape_metrics.curvature),
        ("absolute_torsion", shape_metrics.absolute_torsion),
        ("curl_index", shape_metrics.curl_index),
        ("axis_cosine", shape_metrics.axis_cosine),
        ("fiber_length", shape_metrics.fiber_length),
        (
            "crossing_angle",
            Distribution::from_values(events.iter().map(|e| e.crossing_angle), quantiles),
        ),
        (
            "free_length",
            Distribution::from_values(neighbor_metrics.free_lengths.iter().copied(), quantiles),
        ),
        (
            "excess_persistence",
            Distribution::from_values(
                events
                    .iter()
                    .filter(|e| !e.in_axis)
                    .map(|e| e.excess_persistence()),
                quantiles,
            ),
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();

    Ok(StructureProfile {
        schema_version: SCORECARD_SCHEMA_VERSION,
        fibers: shape_metrics.fiber_count,
        scalars,
        distributions,
    })
}

/// Settings for [`score_structure`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScorecardConfig {
    /// Number of reference subvolumes along each axis. The subvolume size is
    /// the reference region divided by these counts, and the candidate is
    /// tiled with subvolumes of that same size. At least two subvolumes in
    /// total are needed for a reference spread.
    pub subdivisions: [usize; 3],
    /// Region of the reference to analyze, as `(lower, upper)` corners.
    /// `None` uses the reference cell.
    pub reference_region: Option<(Vec3, Vec3)>,
    /// Region of the candidate to tile with subvolumes. `None` uses the
    /// candidate cell. Restrict it when fibers do not fill the cell, for
    /// example to the thickness actually occupied by a generated stack.
    pub candidate_region: Option<(Vec3, Vec3)>,
    /// Largest number of candidate subvolumes analyzed; beyond it an evenly
    /// strided subset is used.
    pub maximum_candidate_subvolumes: usize,
    /// Fiber pieces shorter than this after cropping are discarded, as a CT
    /// tracker discards fibers clipping a corner. `None` uses the largest
    /// fiber diameter in the reference.
    pub minimum_piece_length: Option<f64>,
    /// Shape-analysis settings. Unset lengths are resolved from the whole
    /// reference region and then used for every subvolume.
    pub shape: ShapeAnalysisConfig,
    /// Neighbor-analysis settings. Unset lengths are resolved from the whole
    /// reference region and then used for every subvolume.
    pub neighbors: NeighborAnalysisConfig,
}

impl ScorecardConfig {
    /// Creates a configuration with the supplied contact tolerance, a 2×2×2
    /// reference subdivision and defaults for everything else.
    pub fn new(contact_gap: f64) -> Self {
        Self {
            subdivisions: [2, 2, 2],
            reference_region: None,
            candidate_region: None,
            maximum_candidate_subvolumes: 64,
            minimum_piece_length: None,
            shape: ShapeAnalysisConfig::default(),
            neighbors: NeighborAnalysisConfig::new(contact_gap),
        }
    }
}

/// One metric's comparison.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoreRow {
    /// Metric name, as in [`StructureProfile`].
    pub metric: String,
    /// Whether the metric is a distribution (compared by Wasserstein
    /// distance) rather than a scalar.
    pub distribution: bool,
    /// The metric over the whole candidate region (the median, for a
    /// distribution).
    pub candidate: Option<f64>,
    /// The metric over the whole reference region (the median, for a
    /// distribution).
    pub reference: Option<f64>,
    /// Median distance over candidate–reference subvolume pairs.
    pub distance: Option<f64>,
    /// Median distance over reference–reference subvolume pairs.
    pub reference_spread: Option<f64>,
    /// `distance / reference_spread`. `None` when either is undefined or the
    /// spread is zero.
    pub score: Option<f64>,
}

/// Metric-by-metric comparison of a candidate structure with a reference.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scorecard {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Edge lengths of every subvolume.
    pub subvolume_size: Vec3,
    /// Number of reference subvolumes analyzed.
    pub reference_subvolumes: usize,
    /// Number of candidate subvolumes analyzed.
    pub candidate_subvolumes: usize,
    /// Fiber pieces shorter than this were discarded after cropping.
    pub minimum_piece_length: f64,
    /// Shape settings actually used, with lengths resolved.
    pub shape: ShapeAnalysisConfig,
    /// Neighbor settings actually used, with lengths resolved.
    pub neighbors: NeighborAnalysisConfig,
    /// One row per metric, scalars first, in name order.
    pub rows: Vec<ScoreRow>,
    /// Metrics of the whole candidate region.
    pub candidate: StructureProfile,
    /// Metrics of the whole reference region.
    pub reference: StructureProfile,
}

/// Why a scorecard could not be computed.
#[derive(Clone, Debug, PartialEq)]
pub enum ScorecardError {
    /// The shape settings were invalid.
    Shape(ShapeAnalysisError),
    /// The neighbor settings were invalid.
    Neighbors(NeighborAnalysisError),
    /// A subdivision count was zero, or the total was below two.
    InvalidSubdivisions([usize; 3]),
    /// A region had a non-positive or non-finite extent.
    InvalidRegion {
        /// Lower corner.
        lower: Vec3,
        /// Upper corner.
        upper: Vec3,
    },
    /// A cell used as a region, or cropped with periodic images, was not
    /// orthorhombic.
    NonOrthorhombicCell,
    /// The candidate region is smaller than one subvolume along an axis.
    CandidateRegionTooSmall {
        /// Axis index.
        axis: usize,
        /// Candidate region length along that axis.
        length: f64,
        /// Subvolume length along that axis.
        subvolume: f64,
    },
    /// `maximum_candidate_subvolumes` was zero.
    InvalidCandidateSubvolumeLimit,
}

impl fmt::Display for ScorecardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(error) => write!(formatter, "{error}"),
            Self::Neighbors(error) => write!(formatter, "{error}"),
            Self::InvalidSubdivisions(counts) => write!(
                formatter,
                "subdivisions must be positive and give at least two subvolumes, got {counts:?}"
            ),
            Self::InvalidRegion { lower, upper } => write!(
                formatter,
                "region upper corner {upper:?} must exceed lower corner {lower:?} on every axis"
            ),
            Self::NonOrthorhombicCell => write!(
                formatter,
                "scorecards and cropping support only orthorhombic cells"
            ),
            Self::CandidateRegionTooSmall {
                axis,
                length,
                subvolume,
            } => write!(
                formatter,
                "candidate region length {length} on axis {axis} is smaller than the subvolume \
                 length {subvolume}"
            ),
            Self::InvalidCandidateSubvolumeLimit => {
                write!(formatter, "maximum_candidate_subvolumes must be at least 1")
            }
        }
    }
}

impl Error for ScorecardError {}

impl From<ShapeAnalysisError> for ScorecardError {
    fn from(error: ShapeAnalysisError) -> Self {
        Self::Shape(error)
    }
}

impl From<NeighborAnalysisError> for ScorecardError {
    fn from(error: NeighborAnalysisError) -> Self {
        Self::Neighbors(error)
    }
}

/// Scores `candidate` against `reference` metric by metric.
pub fn score_structure(
    candidate: &FiberAssembly,
    reference: &FiberAssembly,
    config: &ScorecardConfig,
) -> Result<Scorecard, ScorecardError> {
    let counts = config.subdivisions;
    if counts.contains(&0) || counts.iter().product::<usize>() < 2 {
        return Err(ScorecardError::InvalidSubdivisions(counts));
    }
    if config.maximum_candidate_subvolumes == 0 {
        return Err(ScorecardError::InvalidCandidateSubvolumeLimit);
    }
    let (reference_lower, reference_upper) = region(reference, config.reference_region)?;
    let (candidate_lower, candidate_upper) = region(candidate, config.candidate_region)?;
    let size: Vec3 = std::array::from_fn(|axis| {
        (reference_upper[axis] - reference_lower[axis]) / counts[axis] as f64
    });
    let mut candidate_counts = [0usize; 3];
    for axis in 0..3 {
        let length = candidate_upper[axis] - candidate_lower[axis];
        // Tolerate rounding when the candidate region is a whole multiple.
        candidate_counts[axis] = (length / size[axis] * (1.0 + 1e-9)).floor() as usize;
        if candidate_counts[axis] == 0 {
            return Err(ScorecardError::CandidateRegionTooSmall {
                axis,
                length,
                subvolume: size[axis],
            });
        }
    }

    let minimum_piece_length = match config.minimum_piece_length {
        Some(length) if length.is_finite() && length >= 0.0 => length,
        Some(length) => {
            return Err(ScorecardError::Shape(ShapeAnalysisError::InvalidLength {
                name: "minimum_piece_length",
                value: length,
            }))
        }
        None => largest_diameter(reference),
    };

    // Resolve every data-dependent default once, from the whole reference
    // region, so all subvolumes on both sides are measured identically.
    let reference_whole = crop_assembly(
        reference,
        reference_lower,
        reference_upper,
        minimum_piece_length,
    )?;
    let first_shape = analyze_shape(&reference_whole, &config.shape)?;
    let first_neighbors = analyze_neighbors(&reference_whole, &config.neighbors)?;
    let shape = ShapeAnalysisConfig {
        sample_spacing: Some(first_shape.sample_spacing),
        maximum_lag: Some(config.shape.maximum_lag.unwrap_or_else(|| {
            first_shape
                .tangent_correlation_lags
                .last()
                .copied()
                .unwrap_or(first_shape.sample_spacing)
        })),
        minimum_torsion_curvature: Some(first_shape.minimum_torsion_curvature),
        ..config.shape.clone()
    };
    let neighbors = NeighborAnalysisConfig {
        neighbor_gap: Some(first_neighbors.neighbor_gap),
        sample_spacing: Some(first_neighbors.sample_spacing),
        maximum_lag: Some(config.neighbors.maximum_lag.unwrap_or_else(|| {
            first_neighbors
                .turnover_lags
                .last()
                .copied()
                .unwrap_or(first_neighbors.sample_spacing)
        })),
        ..config.neighbors.clone()
    };

    let profile = |assembly: &FiberAssembly| profile_structure(assembly, &shape, &neighbors);
    let reference_profile = profile(&reference_whole)?;
    let candidate_whole = crop_assembly(
        candidate,
        candidate_lower,
        candidate_upper,
        minimum_piece_length,
    )?;
    let candidate_profile = profile(&candidate_whole)?;

    let reference_boxes = tiles(reference_lower, size, counts, usize::MAX);
    let candidate_boxes = tiles(
        candidate_lower,
        size,
        candidate_counts,
        config.maximum_candidate_subvolumes,
    );
    let profile_boxes = |assembly: &FiberAssembly, boxes: &[(Vec3, Vec3)]| {
        boxes
            .iter()
            .map(|(lower, upper)| {
                profile(&crop_assembly(
                    assembly,
                    *lower,
                    *upper,
                    minimum_piece_length,
                )?)
            })
            .collect::<Result<Vec<_>, ScorecardError>>()
    };
    let reference_parts = profile_boxes(reference, &reference_boxes)?;
    let candidate_parts = profile_boxes(candidate, &candidate_boxes)?;

    let mut rows = Vec::new();
    for name in reference_profile.scalars.keys() {
        let value = |p: &StructureProfile| p.scalars.get(name).copied().flatten();
        let distance = |a: &StructureProfile, b: &StructureProfile| {
            value(a).zip(value(b)).map(|(a, b)| (a - b).abs())
        };
        rows.push(score_row(
            name,
            false,
            value(&candidate_profile),
            value(&reference_profile),
            &candidate_parts,
            &reference_parts,
            distance,
        ));
    }
    for name in reference_profile.distributions.keys() {
        let value = |p: &StructureProfile| p.distributions.get(name).cloned().flatten();
        let distance = |a: &StructureProfile, b: &StructureProfile| {
            value(a)
                .zip(value(b))
                .map(|(a, b)| a.wasserstein_distance(&b))
        };
        rows.push(score_row(
            name,
            true,
            value(&candidate_profile).map(|d| d.median()),
            value(&reference_profile).map(|d| d.median()),
            &candidate_parts,
            &reference_parts,
            distance,
        ));
    }

    Ok(Scorecard {
        schema_version: SCORECARD_SCHEMA_VERSION,
        subvolume_size: size,
        reference_subvolumes: reference_parts.len(),
        candidate_subvolumes: candidate_parts.len(),
        minimum_piece_length,
        shape,
        neighbors,
        rows,
        candidate: candidate_profile,
        reference: reference_profile,
    })
}

fn score_row(
    name: &str,
    distribution: bool,
    candidate: Option<f64>,
    reference: Option<f64>,
    candidate_parts: &[StructureProfile],
    reference_parts: &[StructureProfile],
    distance: impl Fn(&StructureProfile, &StructureProfile) -> Option<f64>,
) -> ScoreRow {
    let cross = median(
        candidate_parts
            .iter()
            .flat_map(|c| reference_parts.iter().filter_map(|r| distance(c, r)))
            .collect(),
    );
    let within = median(
        reference_parts
            .iter()
            .enumerate()
            .flat_map(|(i, a)| {
                reference_parts[i + 1..]
                    .iter()
                    .filter_map(|b| distance(a, b))
                    .collect::<Vec<_>>()
            })
            .collect(),
    );
    ScoreRow {
        metric: name.to_owned(),
        distribution,
        candidate,
        reference,
        distance: cross,
        reference_spread: within,
        score: cross
            .zip(within)
            .filter(|(_, within)| *within > 0.0)
            .map(|(cross, within)| cross / within),
    }
}

// ---------------------------------------------------------------------------
// Cropping

/// Clips every fiber to the box `[lower, upper]` and returns the pieces as a
/// new, non-periodic assembly whose cell is that box.
///
/// A fiber that leaves and re-enters the box becomes several pieces, and on
/// periodic axes every periodic image that reaches the box is included.
/// Pieces shorter than `minimum_piece_length` are discarded. Materials and
/// sections keep their identifiers; fibers are renumbered from one.
pub fn crop_assembly(
    assembly: &FiberAssembly,
    lower: Vec3,
    upper: Vec3,
    minimum_piece_length: f64,
) -> Result<FiberAssembly, ScorecardError> {
    check_box(lower, upper)?;
    let cell = &assembly.cell;
    if cell.periodic.iter().any(|p| *p) && !is_orthorhombic(cell) {
        return Err(ScorecardError::NonOrthorhombicCell);
    }
    let mut cropped = FiberAssembly::new(PeriodicCell {
        origin: lower,
        basis: [
            [upper[0] - lower[0], 0.0, 0.0],
            [0.0, upper[1] - lower[1], 0.0],
            [0.0, 0.0, upper[2] - lower[2]],
        ],
        periodic: [false; 3],
    });
    for material in &assembly.materials.entries {
        cropped.materials.add(material.name.clone());
    }
    for section in &assembly.sections.entries {
        cropped.sections.add(*section);
    }

    let mut next_id = 1u32;
    for fiber in &assembly.topology.fibers {
        let start = fiber.vertices.start as usize;
        let Some(end) = fiber.vertices.checked_end().map(|end| end as usize) else {
            continue;
        };
        let Some(points) = assembly.geometry.placed.positions.get(start..end) else {
            continue;
        };
        if points.len() < 2 {
            continue;
        }
        let mut emit = |piece: &mut Vec<Vec3>| {
            let length: f64 = piece.windows(2).map(|p| norm(sub(p[1], p[0]))).sum();
            if piece.len() >= 2 && length > 0.0 && length >= minimum_piece_length {
                // Identifiers are fresh and material/section ids were copied,
                // so adding a piece cannot fail.
                let points: &[Vec3] = piece;
                let _ = cropped.add_fiber(
                    FiberId(next_id),
                    fiber.material,
                    fiber.section,
                    points,
                    points,
                );
                next_id += 1;
            }
            piece.clear();
        };
        for shift in image_shifts(cell, points, lower, upper) {
            let mut piece: Vec<Vec3> = Vec::new();
            for segment in points.windows(2) {
                let a = add(segment[0], shift);
                let b = add(segment[1], shift);
                let Some((t0, t1)) = clip_segment(a, b, lower, upper) else {
                    emit(&mut piece);
                    continue;
                };
                if t0 > 0.0 {
                    emit(&mut piece);
                }
                let direction = sub(b, a);
                if piece.is_empty() {
                    piece.push(add(a, scale(direction, t0)));
                }
                let exit = add(a, scale(direction, t1));
                if piece.last() != Some(&exit) {
                    piece.push(exit);
                }
                if t1 < 1.0 {
                    emit(&mut piece);
                }
            }
            emit(&mut piece);
        }
    }
    Ok(cropped)
}

/// Parametric range of segment `a → b` inside the box (Liang–Barsky).
fn clip_segment(a: Vec3, b: Vec3, lower: Vec3, upper: Vec3) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for axis in 0..3 {
        let d = b[axis] - a[axis];
        for (p, q) in [(-d, a[axis] - lower[axis]), (d, upper[axis] - a[axis])] {
            if p == 0.0 {
                if q < 0.0 {
                    return None;
                }
            } else {
                let r = q / p;
                if p < 0.0 {
                    if r > t1 {
                        return None;
                    }
                    t0 = t0.max(r);
                } else {
                    if r < t0 {
                        return None;
                    }
                    t1 = t1.min(r);
                }
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// Periodic translations of a fiber that can reach the box.
fn image_shifts(cell: &PeriodicCell, points: &[Vec3], lower: Vec3, upper: Vec3) -> Vec<Vec3> {
    let mut ranges = [(0i64, 0i64); 3];
    for axis in 0..3 {
        let length = cell.basis[axis][axis];
        if !cell.periodic[axis] || length <= 0.0 {
            continue;
        }
        let low = points.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min);
        let high = points
            .iter()
            .map(|p| p[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        ranges[axis] = (
            ((lower[axis] - high) / length).ceil() as i64,
            ((upper[axis] - low) / length).floor() as i64,
        );
    }
    let mut shifts = Vec::new();
    for i in ranges[0].0..=ranges[0].1 {
        for j in ranges[1].0..=ranges[1].1 {
            for k in ranges[2].0..=ranges[2].1 {
                shifts.push([
                    i as f64 * cell.basis[0][0],
                    j as f64 * cell.basis[1][1],
                    k as f64 * cell.basis[2][2],
                ]);
            }
        }
    }
    shifts
}

// ---------------------------------------------------------------------------
// Regions and helpers

fn region(
    assembly: &FiberAssembly,
    requested: Option<(Vec3, Vec3)>,
) -> Result<(Vec3, Vec3), ScorecardError> {
    let (lower, upper) = match requested {
        Some(corners) => corners,
        None => {
            let cell = &assembly.cell;
            if !is_orthorhombic(cell) {
                return Err(ScorecardError::NonOrthorhombicCell);
            }
            let origin = cell.origin;
            let upper = std::array::from_fn(|axis| origin[axis] + cell.basis[axis][axis]);
            (origin, upper)
        }
    };
    check_box(lower, upper)?;
    Ok((lower, upper))
}

fn check_box(lower: Vec3, upper: Vec3) -> Result<(), ScorecardError> {
    let valid = (0..3).all(|axis| {
        lower[axis].is_finite() && upper[axis].is_finite() && upper[axis] > lower[axis]
    });
    if valid {
        Ok(())
    } else {
        Err(ScorecardError::InvalidRegion { lower, upper })
    }
}

fn is_orthorhombic(cell: &PeriodicCell) -> bool {
    (0..3).all(|row| (0..3).all(|column| row == column || cell.basis[row][column] == 0.0))
        && (0..3).all(|axis| cell.basis[axis][axis] > 0.0)
}

/// Subvolume boxes on a regular grid, at most `limit` of them, evenly strided.
fn tiles(origin: Vec3, size: Vec3, counts: [usize; 3], limit: usize) -> Vec<(Vec3, Vec3)> {
    let mut boxes = Vec::new();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                let index = [i, j, k];
                let lower: Vec3 =
                    std::array::from_fn(|axis| origin[axis] + index[axis] as f64 * size[axis]);
                let upper: Vec3 = std::array::from_fn(|axis| lower[axis] + size[axis]);
                boxes.push((lower, upper));
            }
        }
    }
    if boxes.len() <= limit {
        return boxes;
    }
    let total = boxes.len();
    (0..limit).map(|n| boxes[n * total / limit]).collect()
}

fn largest_diameter(assembly: &FiberAssembly) -> f64 {
    assembly
        .sections
        .entries
        .iter()
        .map(|section| match section {
            tangle_core::Section::Circular { radius } => 2.0 * radius,
            tangle_core::Section::Elliptical { semi_axes } => 2.0 * semi_axes[0].max(semi_axes[1]),
        })
        .fold(0.0, f64::max)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use tangle_core::Section;

    fn assembly_with(cell: PeriodicCell, radius: f64, centerlines: &[Vec<Vec3>]) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(cell);
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

    fn centerlines(assembly: &FiberAssembly) -> Vec<Vec<Vec3>> {
        assembly
            .topology
            .fibers
            .iter()
            .map(|fiber| {
                let start = fiber.vertices.start as usize;
                let end = fiber.vertices.checked_end().unwrap() as usize;
                assembly.geometry.placed.positions[start..end].to_vec()
            })
            .collect()
    }

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
    }

    /// Wavy fibers in the xy plane of a periodic slab, with random position,
    /// direction and phase. `amplitude` sets the waviness.
    fn wavy_slab(seed: u64, side: f64, fibers: usize, amplitude: f64) -> FiberAssembly {
        let mut rng = Rng(seed);
        let length = 0.4 * side;
        let lines: Vec<Vec<Vec3>> = (0..fibers)
            .map(|_| {
                let start = [
                    side * rng.uniform(),
                    side * rng.uniform(),
                    0.1 + 0.8 * rng.uniform(),
                ];
                let angle = 2.0 * PI * rng.uniform();
                let phase = 2.0 * PI * rng.uniform();
                let (u, v) = ([angle.cos(), angle.sin()], [-angle.sin(), angle.cos()]);
                (0..=80)
                    .map(|i| {
                        let s = length * i as f64 / 80.0;
                        let w = amplitude * (2.0 * PI * s / (0.1 * side) + phase).sin();
                        [
                            start[0] + s * u[0] + w * v[0],
                            start[1] + s * u[1] + w * v[1],
                            start[2],
                        ]
                    })
                    .collect()
            })
            .collect();
        assembly_with(
            PeriodicCell::orthorhombic([side, side, 1.0], [true, true, false]),
            0.01,
            &lines,
        )
    }

    fn config() -> ScorecardConfig {
        ScorecardConfig {
            subdivisions: [2, 2, 1],
            ..ScorecardConfig::new(0.005)
        }
    }

    #[test]
    fn cropping_splits_fibers_at_the_box_and_keeps_re_entries() {
        let cell = PeriodicCell::orthorhombic([4.0, 4.0, 4.0], [false; 3]);
        // Leaves the unit box at x = 1 and comes back in.
        let fiber = vec![
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [1.5, 0.8, 0.5],
            [0.5, 0.8, 0.5],
        ];
        let assembly = assembly_with(cell, 0.01, &[fiber]);
        let cropped = crop_assembly(&assembly, [0.0; 3], [1.0; 3], 0.0).unwrap();
        let lines = centerlines(&cropped);
        assert_eq!(
            lines,
            vec![
                vec![[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]],
                vec![[1.0, 0.8, 0.5], [0.5, 0.8, 0.5]],
            ]
        );
        assert_eq!(cropped.cell.periodic, [false; 3]);
        assert_eq!(cropped.cell.origin, [0.0; 3]);

        // Pieces shorter than the minimum are dropped.
        let cropped = crop_assembly(&assembly, [0.0; 3], [1.0; 3], 0.6).unwrap();
        assert_eq!(cropped.topology.fibers.len(), 0);
    }

    #[test]
    fn cropping_includes_periodic_images() {
        let cell = PeriodicCell::orthorhombic([2.0, 2.0, 2.0], [true, false, false]);
        // Unwrapped past the cell's right face; its image re-enters at x = 0.
        let fiber = vec![[1.5, 0.5, 0.5], [2.5, 0.5, 0.5]];
        let assembly = assembly_with(cell, 0.01, &[fiber]);
        let cropped = crop_assembly(&assembly, [0.0; 3], [1.0; 3], 0.0).unwrap();
        assert_eq!(
            centerlines(&cropped),
            vec![vec![[0.0, 0.5, 0.5], [0.5, 0.5, 0.5]]]
        );
    }

    #[test]
    fn a_structure_scores_about_one_against_an_independent_copy_of_itself() {
        let reference = wavy_slab(1, 4.0, 120, 0.02);
        let candidate = wavy_slab(2, 4.0, 120, 0.02);
        let card = score_structure(&candidate, &reference, &config()).unwrap();
        assert_eq!(card.reference_subvolumes, 4);
        assert_eq!(card.candidate_subvolumes, 4);
        assert_eq!(card.subvolume_size, [2.0, 2.0, 1.0]);
        let curvature = card.rows.iter().find(|r| r.metric == "curvature").unwrap();
        let score = curvature.score.unwrap();
        assert!(score < 2.0, "curvature score {score}");
    }

    #[test]
    fn a_wavier_structure_scores_high_on_curvature() {
        let reference = wavy_slab(1, 4.0, 120, 0.02);
        let candidate = wavy_slab(2, 4.0, 120, 0.08);
        let card = score_structure(&candidate, &reference, &config()).unwrap();
        let curvature = card.rows.iter().find(|r| r.metric == "curvature").unwrap();
        assert!(curvature.score.unwrap() > 5.0, "{curvature:?}");
        assert!(curvature.candidate.unwrap() > curvature.reference.unwrap());
        // Same planar orientation on both sides.
        let beta = card
            .rows
            .iter()
            .find(|r| r.metric == "log_schladitz_beta")
            .unwrap();
        assert!(beta.reference.unwrap() > 3.0);
    }

    #[test]
    fn settings_resolved_from_the_reference_are_reported_and_shared() {
        let reference = wavy_slab(1, 4.0, 60, 0.02);
        let candidate = wavy_slab(2, 8.0, 240, 0.02);
        let card = score_structure(&candidate, &reference, &config()).unwrap();
        assert!(card.shape.sample_spacing.is_some());
        assert!(card.neighbors.neighbor_gap.is_some());
        assert!((card.minimum_piece_length - 0.02).abs() < 1e-12);
        // An 8×8 candidate holds 4×4 subvolumes of 2×2.
        assert_eq!(card.candidate_subvolumes, 16);
        let limited = ScorecardConfig {
            maximum_candidate_subvolumes: 5,
            ..config()
        };
        let card = score_structure(&candidate, &reference, &limited).unwrap();
        assert_eq!(card.candidate_subvolumes, 5);
    }

    #[test]
    fn invalid_scorecard_settings_are_rejected() {
        let reference = wavy_slab(1, 4.0, 20, 0.02);
        let small = wavy_slab(2, 1.0, 5, 0.02);
        let check = |candidate: &FiberAssembly, config: ScorecardConfig| {
            score_structure(candidate, &reference, &config).unwrap_err()
        };
        assert_eq!(
            check(
                &reference,
                ScorecardConfig {
                    subdivisions: [1, 1, 1],
                    ..config()
                }
            ),
            ScorecardError::InvalidSubdivisions([1, 1, 1])
        );
        assert!(matches!(
            check(&small, config()),
            ScorecardError::CandidateRegionTooSmall { axis: 0, .. }
        ));
        assert!(matches!(
            check(
                &reference,
                ScorecardConfig {
                    reference_region: Some(([0.0; 3], [1.0, 1.0, 0.0])),
                    ..config()
                }
            ),
            ScorecardError::InvalidRegion { .. }
        ));
        assert_eq!(
            check(
                &reference,
                ScorecardConfig {
                    maximum_candidate_subvolumes: 0,
                    ..config()
                }
            ),
            ScorecardError::InvalidCandidateSubvolumeLimit
        );
    }
}
