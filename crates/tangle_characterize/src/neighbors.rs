//! Fiber-to-fiber contact and neighbor statistics.
//!
//! These metrics describe how fibers touch and travel together, which volume
//! fraction and orientation tensors cannot see. They use only placed
//! centerlines and cross-section radii, so the same analysis applies to a
//! generated assembly and to centerlines tracked from a CT scan.
//!
//! Each fiber is sampled at a uniform arc-length spacing. At every sample the
//! analysis finds the closest approach to every other fiber's centerline and
//! converts it to a surface gap (axis distance minus both radii):
//!
//! * a fiber within `contact_gap` is a **contact**;
//! * a fiber within `neighbor_gap` is a **neighbor**;
//! * a contact or neighbor is **in-axis** when the acute angle between the two
//!   tangents is below `in_axis_angle` (fibers running side by side), and
//!   **out-of-axis** otherwise (fibers crossing).
//!
//! A **contact event** is one maximal run of samples along a fiber over which
//! the same other fiber stays a contact. Two fibers that touch twice give two
//! events, and each contact is counted once from each fiber.

use std::cmp::Ordering;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use tangle_core::{FiberAssembly, FiberId, Section, Vec3};

/// Schema version of [`NeighborMetrics`].
pub const NEIGHBOR_SCHEMA_VERSION: u32 = 1;

/// Largest number of cells the candidate grid may allocate.
const MAXIMUM_GRID_CELLS: usize = 1 << 22;

/// Settings for [`analyze_neighbors`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NeighborAnalysisConfig {
    /// Largest surface gap counted as a contact, in assembly length units.
    ///
    /// CT segmentation cannot resolve gaps below about one voxel, so use a
    /// tolerance of that order when comparing with tracked CT centerlines.
    pub contact_gap: f64,
    /// Largest surface gap counted as a neighbor. `None` uses twice the
    /// largest fiber radius, or `contact_gap` if that is larger.
    pub neighbor_gap: Option<f64>,
    /// Crossing angle, in radians, below which a contact or neighbor is
    /// in-axis.
    pub in_axis_angle: f64,
    /// Arc-length spacing between samples. `None` uses a quarter of the
    /// smallest fiber radius.
    pub sample_spacing: Option<f64>,
    /// Largest arc-length lag of the neighbor-turnover curve. `None` uses the
    /// smaller of half the median fiber length and 200 sample spacings.
    pub maximum_lag: Option<f64>,
    /// Number of logarithmically spaced lags in the turnover curve.
    pub lag_count: usize,
    /// Number of random sample pairs used to estimate the random-placement
    /// contact baseline.
    pub baseline_pairs: usize,
    /// Seed for the random-placement baseline estimate.
    pub seed: u64,
}

impl NeighborAnalysisConfig {
    /// Creates a configuration with the supplied contact tolerance and
    /// defaults for every other setting.
    pub fn new(contact_gap: f64) -> Self {
        Self {
            contact_gap,
            neighbor_gap: None,
            in_axis_angle: 20.0_f64.to_radians(),
            sample_spacing: None,
            maximum_lag: None,
            lag_count: 24,
            baseline_pairs: 200_000,
            seed: 0,
        }
    }
}

/// One maximal run of arc length over which another fiber stays a contact.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContactEvent {
    /// Fiber along which the event is measured.
    pub fiber_id: FiberId,
    /// The contacting fiber.
    pub other_fiber_id: FiberId,
    /// Arc length along `fiber_id` where the event starts.
    pub start: f64,
    /// Arc length along `fiber_id` covered by the event.
    pub length: f64,
    /// Mean acute angle between the two tangents, in radians.
    pub crossing_angle: f64,
    /// Smallest centerline-to-centerline distance during the event.
    pub closest_axis_distance: f64,
    /// Length the event would cover if both fibers were straight lines with
    /// the same crossing angle and closest approach, capped at the fiber
    /// length.
    pub straight_crossing_length: f64,
    /// Whether the crossing angle is below the in-axis threshold.
    pub in_axis: bool,
}

impl ContactEvent {
    /// Observed length divided by the straight-crossing length.
    ///
    /// Random crossings give about one; fibers that wrap around or travel
    /// along each other give more. It is only meaningful for out-of-axis
    /// events, because a straight in-axis crossing never ends.
    pub fn excess_persistence(&self) -> f64 {
        self.length / self.straight_crossing_length
    }
}

/// Contact and neighbor summary for one fiber.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberNeighborMetrics {
    /// Stable source fiber identifier.
    pub fiber_id: FiberId,
    /// Placed centerline length.
    pub length: f64,
    /// Number of contact events along this fiber.
    pub contacts: usize,
    /// Number of in-axis contact events along this fiber.
    pub in_axis_contacts: usize,
    /// Mean number of neighbors per sample.
    pub mean_neighbors: f64,
}

/// Contact and neighbor statistics of one assembly state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NeighborMetrics {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Settings used, with defaults resolved.
    pub contact_gap: f64,
    /// Neighbor surface-gap tolerance actually used.
    pub neighbor_gap: f64,
    /// In-axis angle threshold, in radians.
    pub in_axis_angle: f64,
    /// Arc-length sample spacing actually used.
    pub sample_spacing: f64,
    /// Number of centerline samples.
    pub samples: usize,
    /// Total placed centerline length.
    pub total_length: f64,
    /// Number of contact events, counted once from each participating fiber.
    pub contacts: usize,
    /// Contact events per unit centerline length.
    pub contacts_per_length: f64,
    /// Fraction of contact events that are in-axis.
    pub in_axis_contact_fraction: f64,
    /// Contacts per unit length expected if the same fibers, with the same
    /// length density and orientation distribution, were placed independently
    /// with overlap allowed: `2 λ_L E[(r_i + r_j + gap) sin γ]`, from the
    /// Onsager excluded volume of two cylinders. It ignores fiber ends and uses
    /// the cell volume, so it is only meaningful when fibers fill the cell.
    pub random_baseline_contacts_per_length: Option<f64>,
    /// `contacts_per_length` divided by the random baseline. Values above one
    /// mean more contact than random placement (bonding, needling,
    /// compaction); values below one mean fewer (repulsion, alignment).
    pub contact_ratio_to_random: Option<f64>,
    /// Variance over mean of contacts per fiber. Random placement of
    /// equal-length fibers gives about one; larger values indicate clustering.
    pub contact_count_dispersion: Option<f64>,
    /// Median crossing angle of all contact events, in radians.
    pub median_crossing_angle: Option<f64>,
    /// Median excess persistence of out-of-axis contact events.
    pub median_excess_persistence: Option<f64>,
    /// Median length of in-axis contact events.
    pub median_in_axis_contact_length: Option<f64>,
    /// Mean free arc length between successive contact events on a fiber.
    pub mean_free_length: Option<f64>,
    /// Mean number of neighbors per sample.
    pub mean_neighbors: f64,
    /// Mean number of in-axis neighbors per sample.
    pub mean_in_axis_neighbors: f64,
    /// Samples with each neighbor count; entry `n` counts samples with `n`
    /// neighbors.
    pub neighbor_count_histogram: Vec<usize>,
    /// Samples with each in-axis neighbor count.
    pub in_axis_neighbor_count_histogram: Vec<usize>,
    /// Arc-length lags of the turnover curves.
    pub turnover_lags: Vec<f64>,
    /// Mean Jaccard similarity between a fiber's neighbor sets at `s` and
    /// `s + lag`, over samples where either set is nonempty.
    pub neighbor_turnover: Vec<Option<f64>>,
    /// The same curve restricted to in-axis neighbors.
    pub in_axis_neighbor_turnover: Vec<Option<f64>>,
    /// Lag at which `neighbor_turnover` first falls to `1/e`, interpolated
    /// from an implicit value of one at zero lag. `None` when neighbors do not
    /// turn over within the largest lag.
    pub neighbor_correlation_length: Option<f64>,
    /// The same decay length for in-axis neighbors.
    pub in_axis_correlation_length: Option<f64>,
    /// Free arc lengths between successive contact events on each fiber.
    pub free_lengths: Vec<f64>,
    /// Every contact event.
    pub events: Vec<ContactEvent>,
    /// Per-fiber summaries in topology order.
    pub fibers: Vec<FiberNeighborMetrics>,
}

/// Invalid neighbor-analysis settings or an unsupported cell.
#[derive(Clone, Debug, PartialEq)]
pub enum NeighborAnalysisError {
    /// A length setting was negative, zero where it must be positive, or not
    /// finite.
    InvalidLength {
        /// Setting name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// The neighbor gap was smaller than the contact gap.
    NeighborGapBelowContactGap,
    /// The in-axis angle was outside `[0, π/2]`.
    InvalidAngle(f64),
    /// The lag count was zero.
    InvalidLagCount,
    /// A periodic cell had a non-diagonal basis.
    NonOrthorhombicPeriodicCell,
    /// A periodic axis was shorter than twice the neighbor search distance,
    /// so the nearest periodic image is ambiguous.
    PeriodicCellTooSmall {
        /// Cell axis.
        axis: usize,
        /// Cell length along that axis.
        length: f64,
        /// Neighbor search distance.
        search_distance: f64,
    },
}

impl fmt::Display for NeighborAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { name, value } => {
                write!(formatter, "{name} must be positive and finite, got {value}")
            }
            Self::NeighborGapBelowContactGap => {
                write!(formatter, "neighbor_gap must be at least contact_gap")
            }
            Self::InvalidAngle(angle) => {
                write!(
                    formatter,
                    "in_axis_angle must be in [0, π/2] radians, got {angle}"
                )
            }
            Self::InvalidLagCount => write!(formatter, "lag_count must be at least 1"),
            Self::NonOrthorhombicPeriodicCell => write!(
                formatter,
                "neighbor analysis supports periodic axes only for orthorhombic cells"
            ),
            Self::PeriodicCellTooSmall {
                axis,
                length,
                search_distance,
            } => write!(
                formatter,
                "periodic axis {axis} has length {length}, which is below twice the neighbor \
                 search distance {search_distance}"
            ),
        }
    }
}

impl Error for NeighborAnalysisError {}

/// Measures contacts, neighbor persistence, and neighbor turnover.
pub fn analyze_neighbors(
    assembly: &FiberAssembly,
    config: &NeighborAnalysisConfig,
) -> Result<NeighborMetrics, NeighborAnalysisError> {
    let fibers = collect_fibers(assembly);
    let maximum_radius = fibers.iter().map(|f| f.radius).fold(0.0, f64::max);
    let minimum_radius = fibers
        .iter()
        .map(|f| f.radius)
        .filter(|r| *r > 0.0)
        .fold(f64::INFINITY, f64::min);
    let settings = resolve_settings(config, maximum_radius, minimum_radius)?;
    let geometry = CellGeometry::new(assembly, &settings)?;

    let samples = sample_fibers(&fibers, settings.spacing);
    let grid = SegmentGrid::build(&fibers, &geometry, settings.search);
    let neighbors = find_neighbors(&fibers, &samples, &grid, &geometry, &settings);

    let events = contact_events(&fibers, &samples, &neighbors, &settings);
    let total_length: f64 = fibers.iter().map(|f| f.length).sum();

    let mut per_fiber: Vec<FiberNeighborMetrics> = fibers
        .iter()
        .map(|fiber| FiberNeighborMetrics {
            fiber_id: fiber.id,
            length: fiber.length,
            contacts: 0,
            in_axis_contacts: 0,
            mean_neighbors: 0.0,
        })
        .collect();
    for (event, fiber) in events.iter().map(|(event, fiber)| (event, *fiber)) {
        per_fiber[fiber].contacts += 1;
        if event.in_axis {
            per_fiber[fiber].in_axis_contacts += 1;
        }
    }

    let mut neighbor_histogram = Vec::new();
    let mut in_axis_histogram = Vec::new();
    let mut neighbor_sum = 0usize;
    let mut in_axis_sum = 0usize;
    let mut fiber_sample_counts = vec![0usize; fibers.len()];
    let mut fiber_neighbor_sums = vec![0usize; fibers.len()];
    for (sample, list) in samples.iter().zip(&neighbors) {
        let count = list.len();
        let in_axis = list.iter().filter(|n| n.in_axis).count();
        bump(&mut neighbor_histogram, count);
        bump(&mut in_axis_histogram, in_axis);
        neighbor_sum += count;
        in_axis_sum += in_axis;
        fiber_sample_counts[sample.fiber] += 1;
        fiber_neighbor_sums[sample.fiber] += count;
    }
    for (index, metrics) in per_fiber.iter_mut().enumerate() {
        if fiber_sample_counts[index] > 0 {
            metrics.mean_neighbors =
                fiber_neighbor_sums[index] as f64 / fiber_sample_counts[index] as f64;
        }
    }

    let lag_steps = lag_steps(&fibers, &settings);
    let neighbor_turnover = turnover_curve(&samples, &neighbors, &lag_steps, false);
    let in_axis_neighbor_turnover = turnover_curve(&samples, &neighbors, &lag_steps, true);
    let turnover_lags: Vec<f64> = lag_steps
        .iter()
        .map(|step| *step as f64 * settings.spacing)
        .collect();

    let contact_count = events.len();
    let contacts_per_length = if total_length > 0.0 {
        contact_count as f64 / total_length
    } else {
        0.0
    };
    let random_baseline = random_baseline(&fibers, &samples, &settings, geometry.volume);
    let events: Vec<ContactEvent> = events.into_iter().map(|(event, _)| event).collect();
    let in_axis_events = events.iter().filter(|e| e.in_axis).count();
    let free_lengths = free_lengths(&events);

    Ok(NeighborMetrics {
        schema_version: NEIGHBOR_SCHEMA_VERSION,
        contact_gap: settings.contact_gap,
        neighbor_gap: settings.neighbor_gap,
        in_axis_angle: settings.in_axis_angle,
        sample_spacing: settings.spacing,
        samples: samples.len(),
        total_length,
        contacts: contact_count,
        contacts_per_length,
        in_axis_contact_fraction: if contact_count > 0 {
            in_axis_events as f64 / contact_count as f64
        } else {
            0.0
        },
        random_baseline_contacts_per_length: random_baseline,
        contact_ratio_to_random: random_baseline
            .filter(|baseline| *baseline > 0.0)
            .map(|baseline| contacts_per_length / baseline),
        contact_count_dispersion: dispersion(per_fiber.iter().map(|f| f.contacts as f64)),
        median_crossing_angle: median(events.iter().map(|e| e.crossing_angle)),
        median_excess_persistence: median(
            events
                .iter()
                .filter(|e| !e.in_axis)
                .map(ContactEvent::excess_persistence),
        ),
        median_in_axis_contact_length: median(
            events.iter().filter(|e| e.in_axis).map(|e| e.length),
        ),
        mean_free_length: mean(free_lengths.iter().copied()),
        mean_neighbors: ratio(neighbor_sum, samples.len()),
        mean_in_axis_neighbors: ratio(in_axis_sum, samples.len()),
        neighbor_count_histogram: neighbor_histogram,
        in_axis_neighbor_count_histogram: in_axis_histogram,
        neighbor_correlation_length: decay_length(&turnover_lags, &neighbor_turnover),
        in_axis_correlation_length: decay_length(&turnover_lags, &in_axis_neighbor_turnover),
        turnover_lags,
        neighbor_turnover,
        in_axis_neighbor_turnover,
        free_lengths,
        events,
        fibers: per_fiber,
    })
}

// ---------------------------------------------------------------------------
// Inputs and settings

pub(crate) struct FiberPath {
    pub(crate) id: FiberId,
    pub(crate) radius: f64,
    pub(crate) points: Vec<Vec3>,
    /// Cumulative arc length at each vertex.
    pub(crate) arc: Vec<f64>,
    pub(crate) length: f64,
}

struct Settings {
    contact_gap: f64,
    neighbor_gap: f64,
    in_axis_angle: f64,
    spacing: f64,
    search: f64,
    maximum_lag: Option<f64>,
    lag_count: usize,
    baseline_pairs: usize,
    seed: u64,
}

pub(crate) fn collect_fibers(assembly: &FiberAssembly) -> Vec<FiberPath> {
    assembly
        .topology
        .fibers
        .iter()
        .filter_map(|fiber| {
            let start = fiber.vertices.start as usize;
            let end = fiber.vertices.checked_end()? as usize;
            let mut points: Vec<Vec3> = Vec::with_capacity(end.saturating_sub(start));
            // Repeated vertices (common in tracked CT centerlines) would give
            // zero-length segments with undefined tangents.
            for point in assembly.geometry.placed.positions.get(start..end)? {
                if points.last() != Some(point) {
                    points.push(*point);
                }
            }
            let radius = match assembly.sections.entries.get(fiber.section.0 as usize)? {
                Section::Circular { radius } => *radius,
                // Equal-area circle for elliptical sections.
                Section::Elliptical { semi_axes } => (semi_axes[0] * semi_axes[1]).sqrt(),
            };
            let mut arc = Vec::with_capacity(points.len());
            let mut length = 0.0;
            arc.push(0.0);
            for pair in points.windows(2) {
                length += norm(sub(pair[1], pair[0]));
                arc.push(length);
            }
            (points.len() >= 2 && length > 0.0).then_some(FiberPath {
                id: fiber.id,
                radius,
                points,
                arc,
                length,
            })
        })
        .collect()
}

fn resolve_settings(
    config: &NeighborAnalysisConfig,
    maximum_radius: f64,
    minimum_radius: f64,
) -> Result<Settings, NeighborAnalysisError> {
    let non_negative = |name, value: f64| {
        if value.is_finite() && value >= 0.0 {
            Ok(value)
        } else {
            Err(NeighborAnalysisError::InvalidLength { name, value })
        }
    };
    let positive = |name, value: f64| {
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(NeighborAnalysisError::InvalidLength { name, value })
        }
    };
    let contact_gap = non_negative("contact_gap", config.contact_gap)?;
    let neighbor_gap = non_negative(
        "neighbor_gap",
        config
            .neighbor_gap
            .unwrap_or((2.0 * maximum_radius).max(contact_gap)),
    )?;
    if neighbor_gap < contact_gap {
        return Err(NeighborAnalysisError::NeighborGapBelowContactGap);
    }
    if !(0.0..=0.5 * PI).contains(&config.in_axis_angle) {
        return Err(NeighborAnalysisError::InvalidAngle(config.in_axis_angle));
    }
    if config.lag_count == 0 {
        return Err(NeighborAnalysisError::InvalidLagCount);
    }
    let default_spacing = if minimum_radius.is_finite() {
        0.25 * minimum_radius
    } else {
        1.0
    };
    let spacing = positive(
        "sample_spacing",
        config.sample_spacing.unwrap_or(default_spacing),
    )?;
    let maximum_lag = config
        .maximum_lag
        .map(|lag| positive("maximum_lag", lag))
        .transpose()?;
    Ok(Settings {
        contact_gap,
        neighbor_gap,
        in_axis_angle: config.in_axis_angle,
        spacing,
        search: 2.0 * maximum_radius + neighbor_gap,
        maximum_lag,
        lag_count: config.lag_count,
        baseline_pairs: config.baseline_pairs,
        seed: config.seed,
    })
}

// ---------------------------------------------------------------------------
// Cell handling

struct CellGeometry {
    origin: Vec3,
    lengths: Vec3,
    periodic: [bool; 3],
    volume: f64,
}

impl CellGeometry {
    fn new(assembly: &FiberAssembly, settings: &Settings) -> Result<Self, NeighborAnalysisError> {
        let cell = &assembly.cell;
        let lengths = [
            norm(cell.basis[0]),
            norm(cell.basis[1]),
            norm(cell.basis[2]),
        ];
        if cell.periodic.iter().any(|p| *p) {
            let scale = lengths.iter().copied().fold(0.0, f64::max).max(1.0e-300);
            for (row, basis) in cell.basis.iter().enumerate() {
                for (column, value) in basis.iter().enumerate() {
                    if row != column && value.abs() > 1.0e-12 * scale {
                        return Err(NeighborAnalysisError::NonOrthorhombicPeriodicCell);
                    }
                }
            }
            for axis in 0..3 {
                if cell.periodic[axis] && lengths[axis] < 2.0 * settings.search {
                    return Err(NeighborAnalysisError::PeriodicCellTooSmall {
                        axis,
                        length: lengths[axis],
                        search_distance: settings.search,
                    });
                }
            }
        }
        Ok(Self {
            origin: cell.origin,
            lengths,
            periodic: cell.periodic,
            volume: cell.signed_volume().abs(),
        })
    }

    fn minimum_image(&self, mut delta: Vec3) -> Vec3 {
        for axis in 0..3 {
            if self.periodic[axis] {
                let length = self.lengths[axis];
                delta[axis] -= length * (delta[axis] / length).round();
            }
        }
        delta
    }
}

/// Uniform grid of segment references. Every segment is registered in every
/// cell its search-expanded bounding box overlaps, so a query only needs the
/// cell containing the query point.
struct SegmentGrid {
    low: Vec3,
    width: Vec3,
    counts: [usize; 3],
    periodic: [bool; 3],
    offsets: Vec<usize>,
    entries: Vec<(u32, u32)>,
}

impl SegmentGrid {
    fn build(fibers: &[FiberPath], geometry: &CellGeometry, search: f64) -> Self {
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        for point in fibers.iter().flat_map(|f| f.points.iter()) {
            for axis in 0..3 {
                low[axis] = low[axis].min(point[axis]);
                high[axis] = high[axis].max(point[axis]);
            }
        }
        let mut cell_size = search.max(f64::MIN_POSITIVE);
        let mut grid_low = [0.0; 3];
        let mut span = [1.0; 3];
        let mut counts = [1usize; 3];
        loop {
            for axis in 0..3 {
                if geometry.periodic[axis] {
                    grid_low[axis] = geometry.origin[axis];
                    span[axis] = geometry.lengths[axis];
                } else if low[axis].is_finite() {
                    grid_low[axis] = low[axis] - search;
                    span[axis] = (high[axis] - low[axis] + 2.0 * search).max(cell_size);
                }
                counts[axis] =
                    ((span[axis] / cell_size).floor() as usize).clamp(1, MAXIMUM_GRID_CELLS);
            }
            if counts
                .iter()
                .try_fold(1usize, |total, count| total.checked_mul(*count))
                .is_some_and(|total| total <= MAXIMUM_GRID_CELLS)
            {
                break;
            }
            cell_size *= 2.0;
        }
        let width = [
            span[0] / counts[0] as f64,
            span[1] / counts[1] as f64,
            span[2] / counts[2] as f64,
        ];
        let mut grid = Self {
            low: grid_low,
            width,
            counts,
            periodic: geometry.periodic,
            offsets: Vec::new(),
            entries: Vec::new(),
        };

        // Two passes: count, then fill a compressed row layout.
        let total_cells = counts.iter().product::<usize>();
        let mut cell_counts = vec![0usize; total_cells + 1];
        grid.visit_segments(fibers, search, |cell, _, _| cell_counts[cell + 1] += 1);
        for index in 1..cell_counts.len() {
            cell_counts[index] += cell_counts[index - 1];
        }
        let mut cursor = cell_counts.clone();
        let mut entries = vec![(0u32, 0u32); *cell_counts.last().unwrap_or(&0)];
        grid.visit_segments(fibers, search, |cell, fiber, segment| {
            entries[cursor[cell]] = (fiber, segment);
            cursor[cell] += 1;
        });
        grid.offsets = cell_counts;
        grid.entries = entries;
        grid
    }

    fn visit_segments(
        &self,
        fibers: &[FiberPath],
        search: f64,
        mut visit: impl FnMut(usize, u32, u32),
    ) {
        // Long segments are registered piece by piece so a diagonal segment
        // covers a tube of cells rather than its whole bounding box.
        let piece_length = self.width.iter().copied().fold(f64::INFINITY, f64::min);
        let mut indices: [Vec<usize>; 3] = Default::default();
        let mut cells = Vec::new();
        for (fiber_index, fiber) in fibers.iter().enumerate() {
            for (segment_index, pair) in fiber.points.windows(2).enumerate() {
                let direction = sub(pair[1], pair[0]);
                let pieces = ((norm(direction) / piece_length).ceil() as usize).max(1);
                cells.clear();
                for piece in 0..pieces {
                    let start = add(pair[0], scale(direction, piece as f64 / pieces as f64));
                    let end = add(
                        pair[0],
                        scale(direction, (piece + 1) as f64 / pieces as f64),
                    );
                    for axis in 0..3 {
                        let lower = start[axis].min(end[axis]) - search;
                        let upper = start[axis].max(end[axis]) + search;
                        self.axis_range(axis, lower, upper, &mut indices[axis]);
                    }
                    for &x in &indices[0] {
                        for &y in &indices[1] {
                            for &z in &indices[2] {
                                cells.push(self.flatten([x, y, z]));
                            }
                        }
                    }
                }
                cells.sort_unstable();
                cells.dedup();
                for &cell in &cells {
                    visit(cell, fiber_index as u32, segment_index as u32);
                }
            }
        }
    }

    fn axis_range(&self, axis: usize, lower: f64, upper: f64, out: &mut Vec<usize>) {
        out.clear();
        let count = self.counts[axis];
        let first = ((lower - self.low[axis]) / self.width[axis]).floor() as i64;
        let last = ((upper - self.low[axis]) / self.width[axis]).floor() as i64;
        if self.periodic[axis] {
            if last - first + 1 >= count as i64 {
                out.extend(0..count);
            } else {
                out.extend((first..=last).map(|i| i.rem_euclid(count as i64) as usize));
            }
        } else {
            let first = first.clamp(0, count as i64 - 1) as usize;
            let last = last.clamp(0, count as i64 - 1) as usize;
            out.extend(first..=last);
        }
    }

    fn cell_of(&self, point: Vec3) -> usize {
        let mut index = [0usize; 3];
        for axis in 0..3 {
            let count = self.counts[axis] as i64;
            let raw = ((point[axis] - self.low[axis]) / self.width[axis]).floor() as i64;
            index[axis] = if self.periodic[axis] {
                raw.rem_euclid(count)
            } else {
                raw.clamp(0, count - 1)
            } as usize;
        }
        self.flatten(index)
    }

    fn flatten(&self, index: [usize; 3]) -> usize {
        (index[2] * self.counts[1] + index[1]) * self.counts[0] + index[0]
    }

    fn cell_entries(&self, cell: usize) -> &[(u32, u32)] {
        &self.entries[self.offsets[cell]..self.offsets[cell + 1]]
    }
}

// ---------------------------------------------------------------------------
// Sampling and neighbor search

struct Sample {
    fiber: usize,
    /// Arc length from the fiber start.
    arc: f64,
    position: Vec3,
    tangent: Vec3,
}

#[derive(Clone, Copy)]
struct Neighbor {
    fiber: usize,
    axis_distance: f64,
    gap: f64,
    angle: f64,
    in_axis: bool,
}

fn sample_fibers(fibers: &[FiberPath], spacing: f64) -> Vec<Sample> {
    let mut samples = Vec::new();
    for (fiber_index, fiber) in fibers.iter().enumerate() {
        let count = (fiber.length / spacing).floor() as usize + 1;
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
            let direction = sub(b, a);
            samples.push(Sample {
                fiber: fiber_index,
                arc,
                position: add(a, scale(direction, t)),
                tangent: scale(direction, norm(direction).recip()),
            });
        }
    }
    samples
}

fn find_neighbors(
    fibers: &[FiberPath],
    samples: &[Sample],
    grid: &SegmentGrid,
    geometry: &CellGeometry,
    settings: &Settings,
) -> Vec<Vec<Neighbor>> {
    let cos_in_axis = settings.in_axis_angle.cos();
    samples
        .iter()
        .map(|sample| {
            let mut best: Vec<(usize, f64, Vec3)> = Vec::new();
            for &(fiber, segment) in grid.cell_entries(grid.cell_of(sample.position)) {
                let fiber = fiber as usize;
                if fiber == sample.fiber {
                    continue;
                }
                let path = &fibers[fiber];
                let a = path.points[segment as usize];
                let b = path.points[segment as usize + 1];
                let relative = geometry.minimum_image(sub(sample.position, a));
                let direction = sub(b, a);
                let length_squared = dot(direction, direction);
                if length_squared <= 0.0 {
                    continue;
                }
                let t = (dot(relative, direction) / length_squared).clamp(0.0, 1.0);
                let distance = norm(sub(relative, scale(direction, t)));
                let reach = fibers[sample.fiber].radius + path.radius + settings.neighbor_gap;
                if distance > reach {
                    continue;
                }
                match best.iter_mut().find(|entry| entry.0 == fiber) {
                    Some(entry) if distance < entry.1 => {
                        entry.1 = distance;
                        entry.2 = direction;
                    }
                    Some(_) => {}
                    None => best.push((fiber, distance, direction)),
                }
            }
            let mut list: Vec<Neighbor> = best
                .into_iter()
                .map(|(fiber, axis_distance, direction)| {
                    let cosine = (dot(sample.tangent, direction).abs() / norm(direction)).min(1.0);
                    Neighbor {
                        fiber,
                        axis_distance,
                        gap: axis_distance - fibers[sample.fiber].radius - fibers[fiber].radius,
                        angle: cosine.acos(),
                        in_axis: cosine > cos_in_axis,
                    }
                })
                .collect();
            list.sort_by_key(|neighbor| neighbor.fiber);
            list
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Contact events

fn contact_events(
    fibers: &[FiberPath],
    samples: &[Sample],
    neighbors: &[Vec<Neighbor>],
    settings: &Settings,
) -> Vec<(ContactEvent, usize)> {
    // (fiber, other, sample index, axis distance, angle), grouped into runs.
    let mut contacts: Vec<(usize, usize, usize, f64, f64)> = Vec::new();
    for (index, (sample, list)) in samples.iter().zip(neighbors).enumerate() {
        for neighbor in list.iter().filter(|n| n.gap <= settings.contact_gap) {
            contacts.push((
                sample.fiber,
                neighbor.fiber,
                index,
                neighbor.axis_distance,
                neighbor.angle,
            ));
        }
    }
    contacts.sort_by_key(|contact| (contact.0, contact.1, contact.2));

    let mut events = Vec::new();
    let mut run_start = 0;
    while run_start < contacts.len() {
        let (fiber, other, first_sample, _, _) = contacts[run_start];
        let mut run_end = run_start + 1;
        while run_end < contacts.len()
            && contacts[run_end].0 == fiber
            && contacts[run_end].1 == other
            && contacts[run_end].2 == contacts[run_end - 1].2 + 1
        {
            run_end += 1;
        }
        let run = &contacts[run_start..run_end];
        let count = run.len() as f64;
        let angle = run.iter().map(|c| c.4).sum::<f64>() / count;
        let closest = run.iter().map(|c| c.3).fold(f64::INFINITY, f64::min);
        let length = count * settings.spacing;
        let fiber_length = fibers[fiber].length;
        let reach = fibers[fiber].radius + fibers[other].radius + settings.contact_gap;
        let half_width = (reach * reach - closest * closest).max(0.0).sqrt();
        let sine = angle.sin();
        let straight = if sine > 1.0e-9 {
            2.0 * half_width / sine
        } else {
            fiber_length
        };
        events.push((
            ContactEvent {
                fiber_id: fibers[fiber].id,
                other_fiber_id: fibers[other].id,
                start: samples[first_sample].arc,
                length,
                crossing_angle: angle,
                closest_axis_distance: closest,
                straight_crossing_length: straight
                    .clamp(settings.spacing, fiber_length.max(settings.spacing)),
                in_axis: angle < settings.in_axis_angle,
            },
            fiber,
        ));
        run_start = run_end;
    }
    events
}

fn free_lengths(events: &[ContactEvent]) -> Vec<f64> {
    let mut ordered: Vec<&ContactEvent> = events.iter().collect();
    ordered.sort_by(|a, b| {
        a.fiber_id
            .0
            .cmp(&b.fiber_id.0)
            .then(a.start.partial_cmp(&b.start).unwrap_or(Ordering::Equal))
    });
    // Contacts can overlap, so a gap only opens after the furthest end any
    // earlier contact on the same fiber has reached.
    let mut free = Vec::new();
    let mut reached: Option<(FiberId, f64)> = None;
    for event in ordered {
        let end = event.start + event.length;
        match reached {
            Some((fiber, furthest)) if fiber == event.fiber_id => {
                if event.start > furthest {
                    free.push(event.start - furthest);
                }
                reached = Some((fiber, furthest.max(end)));
            }
            _ => reached = Some((event.fiber_id, end)),
        }
    }
    free
}

// ---------------------------------------------------------------------------
// Neighbor turnover

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

fn turnover_curve(
    samples: &[Sample],
    neighbors: &[Vec<Neighbor>],
    lag_steps: &[usize],
    in_axis_only: bool,
) -> Vec<Option<f64>> {
    let keep = |n: &&Neighbor| !in_axis_only || n.in_axis;
    lag_steps
        .iter()
        .map(|&lag| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for index in 0..samples.len().saturating_sub(lag) {
                if samples[index].fiber != samples[index + lag].fiber {
                    continue;
                }
                let first = neighbors[index].iter().filter(keep).map(|n| n.fiber);
                let second = neighbors[index + lag].iter().filter(keep).map(|n| n.fiber);
                let (intersection, union) = sorted_overlap(first, second);
                if union > 0 {
                    sum += intersection as f64 / union as f64;
                    count += 1;
                }
            }
            (count > 0).then(|| sum / count as f64)
        })
        .collect()
}

fn sorted_overlap(
    first: impl Iterator<Item = usize>,
    second: impl Iterator<Item = usize>,
) -> (usize, usize) {
    let mut first = first.peekable();
    let mut second = second.peekable();
    let (mut intersection, mut union) = (0, 0);
    loop {
        match (first.peek(), second.peek()) {
            (Some(a), Some(b)) => {
                match a.cmp(b) {
                    Ordering::Less => {
                        first.next();
                    }
                    Ordering::Greater => {
                        second.next();
                    }
                    Ordering::Equal => {
                        intersection += 1;
                        first.next();
                        second.next();
                    }
                }
                union += 1;
            }
            (Some(_), None) => {
                first.next();
                union += 1;
            }
            (None, Some(_)) => {
                second.next();
                union += 1;
            }
            (None, None) => return (intersection, union),
        }
    }
}

pub(crate) fn decay_length(lags: &[f64], curve: &[Option<f64>]) -> Option<f64> {
    let target = (-1.0_f64).exp();
    let (mut previous_lag, mut previous_value) = (0.0, 1.0);
    for (lag, value) in lags.iter().zip(curve) {
        let Some(value) = *value else { continue };
        if value <= target {
            if previous_value - value <= f64::EPSILON {
                return Some(*lag);
            }
            let fraction = (previous_value - target) / (previous_value - value);
            return Some(previous_lag + fraction * (lag - previous_lag));
        }
        previous_lag = *lag;
        previous_value = value;
    }
    None
}

// ---------------------------------------------------------------------------
// Random-placement baseline

fn random_baseline(
    fibers: &[FiberPath],
    samples: &[Sample],
    settings: &Settings,
    volume: f64,
) -> Option<f64> {
    if fibers.len() < 2 || samples.len() < 2 || volume <= 0.0 || settings.baseline_pairs == 0 {
        return None;
    }
    // Samples are uniform in arc length, so uniformly chosen samples give a
    // length-weighted orientation distribution.
    let total_length: f64 = fibers.iter().map(|f| f.length).sum();
    let mut random = SplitMix64(settings.seed);
    let mut sum = 0.0;
    let mut pairs = 0usize;
    for _ in 0..settings.baseline_pairs {
        let first = &samples[random.index(samples.len())];
        let second = &samples[random.index(samples.len())];
        if first.fiber == second.fiber {
            continue;
        }
        let cosine = dot(first.tangent, second.tangent).abs().min(1.0);
        let sine = (1.0 - cosine * cosine).sqrt();
        let reach = fibers[first.fiber].radius + fibers[second.fiber].radius + settings.contact_gap;
        sum += reach * sine;
        pairs += 1;
    }
    (pairs > 0).then(|| 2.0 * (total_length / volume) * sum / pairs as f64)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next() % length as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Small statistics and vector helpers

fn bump(histogram: &mut Vec<usize>, index: usize) {
    if histogram.len() <= index {
        histogram.resize(index + 1, 0);
    }
    histogram[index] += 1;
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, count) = values.fold((0.0, 0usize), |(s, c), v| (s + v, c + 1));
    (count > 0).then(|| sum / count as f64)
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.filter(|v| v.is_finite()).collect();
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

fn dispersion(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    let mean = mean(values.iter().copied())?;
    if mean <= 0.0 {
        return None;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    Some(variance / mean)
}

pub(crate) fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(value: Vec3, factor: f64) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

pub(crate) fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(value: Vec3) -> f64 {
    dot(value, value).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::PeriodicCell;

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

    fn line(start: Vec3, end: Vec3, segments: usize) -> Vec<Vec3> {
        (0..=segments)
            .map(|i| {
                let t = i as f64 / segments as f64;
                add(start, scale(sub(end, start), t))
            })
            .collect()
    }

    #[test]
    fn touching_orthogonal_fibers_form_one_out_of_axis_contact_each() {
        let radius = 0.05;
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([2.0; 3], [false; 3]),
            radius,
            &[
                line([0.5, 1.0, 1.0], [1.5, 1.0, 1.0], 4),
                line(
                    [1.0, 0.5, 1.0 + 2.0 * radius],
                    [1.0, 1.5, 1.0 + 2.0 * radius],
                    4,
                ),
            ],
        );
        let mut config = NeighborAnalysisConfig::new(0.01);
        config.sample_spacing = Some(0.002);
        let metrics = analyze_neighbors(&assembly, &config).unwrap();
        assert_eq!(metrics.contacts, 2);
        assert_eq!(metrics.in_axis_contact_fraction, 0.0);
        let event = &metrics.events[0];
        assert!((event.crossing_angle - 0.5 * PI).abs() < 1.0e-9);
        // Straight crossing: 2 * sqrt((2r + gap)^2 - (2r)^2).
        let expected = 2.0 * ((2.0 * radius + 0.01_f64).powi(2) - (2.0 * radius).powi(2)).sqrt();
        assert!((event.straight_crossing_length - expected).abs() < 1.0e-9);
        assert!((event.excess_persistence() - 1.0).abs() < 0.1);
        assert_eq!(metrics.fibers[0].contacts, 1);
    }

    #[test]
    fn parallel_touching_fibers_form_an_in_axis_contact_that_never_turns_over() {
        let radius = 0.05;
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([2.0; 3], [false; 3]),
            radius,
            &[
                line([0.5, 1.0, 1.0], [1.5, 1.0, 1.0], 3),
                line(
                    [0.5, 1.0 + 2.0 * radius, 1.0],
                    [1.5, 1.0 + 2.0 * radius, 1.0],
                    5,
                ),
            ],
        );
        let metrics = analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(1.0e-3)).unwrap();
        assert_eq!(metrics.contacts, 2);
        assert_eq!(metrics.in_axis_contact_fraction, 1.0);
        let length = metrics.median_in_axis_contact_length.unwrap();
        assert!((length - 1.0).abs() < 2.0 * metrics.sample_spacing);
        assert!(metrics
            .neighbor_turnover
            .iter()
            .all(|value| (value.unwrap() - 1.0).abs() < 1.0e-12));
        assert_eq!(metrics.neighbor_correlation_length, None);
        assert_eq!(metrics.mean_free_length, None);
    }

    #[test]
    fn periodic_images_are_neighbors() {
        let radius = 0.05;
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([2.0; 3], [true, true, true]),
            radius,
            &[
                line([0.02, 0.5, 1.0], [0.02, 1.5, 1.0], 2),
                line(
                    [2.02 - 2.0 * radius, 0.5, 1.0],
                    [2.02 - 2.0 * radius, 1.5, 1.0],
                    2,
                ),
            ],
        );
        let metrics = analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(0.01)).unwrap();
        assert_eq!(metrics.contacts, 2);
        assert_eq!(metrics.in_axis_contact_fraction, 1.0);
    }

    #[test]
    fn random_straight_fibers_match_the_random_baseline() {
        let mut random = SplitMix64(7);
        let mut unit = || (random.next() >> 11) as f64 / (1u64 << 53) as f64;
        let (length, radius) = (0.5, 0.002);
        let centerlines: Vec<Vec<Vec3>> = (0..400)
            .map(|_| {
                let center = [unit(), unit(), unit()];
                let z = 2.0 * unit() - 1.0;
                let azimuth = 2.0 * PI * unit();
                let r = (1.0 - z * z).sqrt();
                let direction = [r * azimuth.cos(), r * azimuth.sin(), z];
                line(
                    sub(center, scale(direction, 0.5 * length)),
                    add(center, scale(direction, 0.5 * length)),
                    20,
                )
            })
            .collect();
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([1.0; 3], [true; 3]),
            radius,
            &centerlines,
        );
        let mut config = NeighborAnalysisConfig::new(0.001);
        config.sample_spacing = Some(0.001);
        let metrics = analyze_neighbors(&assembly, &config).unwrap();
        let ratio = metrics.contact_ratio_to_random.unwrap();
        assert!((0.85..1.25).contains(&ratio), "ratio {ratio}");
        let dispersion = metrics.contact_count_dispersion.unwrap();
        assert!((0.6..1.5).contains(&dispersion), "dispersion {dispersion}");
        let persistence = metrics.median_excess_persistence.unwrap();
        assert!(
            (0.9..1.1).contains(&persistence),
            "persistence {persistence}"
        );
        // Every value must serialize; JSON has no NaN or infinity.
        let json = serde_json::to_value(&metrics).unwrap();
        assert_eq!(json["schema_version"], NEIGHBOR_SCHEMA_VERSION);
    }

    #[test]
    fn invalid_settings_and_cells_are_rejected() {
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([1.0; 3], [false; 3]),
            0.05,
            &[line([0.2, 0.5, 0.5], [0.8, 0.5, 0.5], 1)],
        );
        let mut config = NeighborAnalysisConfig::new(-1.0);
        assert!(matches!(
            analyze_neighbors(&assembly, &config),
            Err(NeighborAnalysisError::InvalidLength { .. })
        ));
        config.contact_gap = 0.1;
        config.neighbor_gap = Some(0.05);
        assert_eq!(
            analyze_neighbors(&assembly, &config),
            Err(NeighborAnalysisError::NeighborGapBelowContactGap)
        );

        let mut sheared = PeriodicCell::orthorhombic([1.0; 3], [true; 3]);
        sheared.basis[1][0] = 0.2;
        let assembly = assembly_with(sheared, 0.05, &[line([0.2, 0.5, 0.5], [0.8, 0.5, 0.5], 1)]);
        assert_eq!(
            analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(0.01)),
            Err(NeighborAnalysisError::NonOrthorhombicPeriodicCell)
        );
    }

    #[test]
    fn empty_assemblies_and_repeated_vertices_are_handled() {
        let empty = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [true; 3]));
        let metrics = analyze_neighbors(&empty, &NeighborAnalysisConfig::new(0.0)).unwrap();
        assert_eq!(metrics.samples, 0);
        assert_eq!(metrics.contacts, 0);

        let radius = 0.05;
        let mut repeated = line([0.5, 1.0, 1.0], [1.5, 1.0, 1.0], 2);
        repeated.push(repeated[2]);
        repeated.insert(1, repeated[0]);
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([2.0; 3], [false; 3]),
            radius,
            &[
                repeated,
                line(
                    [1.0, 0.5, 1.0 + 2.0 * radius],
                    [1.0, 1.5, 1.0 + 2.0 * radius],
                    4,
                ),
            ],
        );
        let metrics = analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(0.01)).unwrap();
        assert_eq!(metrics.contacts, 2);
        serde_json::to_string(&metrics).unwrap();
    }

    #[test]
    fn overlapping_contacts_do_not_open_a_free_length() {
        let event = |start: f64, length: f64| ContactEvent {
            fiber_id: FiberId(1),
            other_fiber_id: FiberId(2),
            start,
            length,
            crossing_angle: 0.5 * PI,
            closest_axis_distance: 0.0,
            straight_crossing_length: 1.0,
            in_axis: false,
        };
        let free = free_lengths(&[
            event(0.0, 10.0),
            event(2.0, 1.0),
            event(5.0, 1.0),
            event(12.0, 1.0),
        ]);
        assert_eq!(free, vec![2.0]);
    }

    #[test]
    fn default_neighbor_gap_is_never_below_the_contact_gap() {
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([1.0; 3], [false; 3]),
            0.01,
            &[line([0.2, 0.5, 0.5], [0.8, 0.5, 0.5], 1)],
        );
        let metrics = analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(0.05)).unwrap();
        assert_eq!(metrics.neighbor_gap, 0.05);
    }

    #[test]
    fn long_diagonal_segments_register_a_tube_of_cells() {
        let radius = 0.001;
        let fibers: Vec<Vec<Vec3>> = (0..20)
            .map(|i| {
                let offset = 0.04 * i as f64;
                vec![[0.1 + offset, 0.0, 0.0], [0.1 + offset, 1.0, 1.0]]
            })
            .collect();
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([1.0; 3], [false; 3]),
            radius,
            &fibers,
        );
        let paths = collect_fibers(&assembly);
        let settings = resolve_settings(&NeighborAnalysisConfig::new(0.0), radius, radius).unwrap();
        let geometry = CellGeometry::new(&assembly, &settings).unwrap();
        let grid = SegmentGrid::build(&paths, &geometry, settings.search);
        let total_cells: usize = grid.counts.iter().product();
        // A whole bounding box per segment would be about total_cells / 20
        // each; a tube along the diagonal is far smaller.
        assert!(
            grid.entries.len() < total_cells / 10,
            "{} entries",
            grid.entries.len()
        );
    }

    #[test]
    fn a_single_fiber_has_no_contacts_or_baseline() {
        let assembly = assembly_with(
            PeriodicCell::orthorhombic([1.0; 3], [false; 3]),
            0.05,
            &[line([0.2, 0.5, 0.5], [0.8, 0.5, 0.5], 1)],
        );
        let metrics = analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(0.01)).unwrap();
        assert_eq!(metrics.contacts, 0);
        assert_eq!(metrics.random_baseline_contacts_per_length, None);
        assert_eq!(metrics.contact_count_dispersion, None);
        assert_eq!(metrics.mean_neighbors, 0.0);
    }
}
