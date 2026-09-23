//! Cross-section statistics: where fibers pierce evenly spaced planes.
//!
//! A CT scan is read slice by slice, and the classic slice measurements are
//! the positions of the fiber sections in each plane: how far each section is
//! from its nearest neighbor, and the pair correlation function `g(r)`, the
//! density of sections at distance `r` from a typical section relative to a
//! random (Poisson) arrangement. `g(r) < 1` at short range means sections
//! avoid each other (excluded volume); a peak near one diameter means they
//! pack against each other; `g → 1` at long range means no order.
//!
//! Sections are the centerline crossings of planes normal to one cell axis,
//! so a fiber lying in a plane contributes no section there and an oblique
//! fiber contributes the center of an elongated one. Periodic in-plane axes
//! use minimum-image distances. Along a non-periodic in-plane axis, sections
//! near the edge have unseen neighbors, so two standard edge corrections are
//! applied: a nearest-neighbor distance counts only when it is no larger
//! than the section's distance to the edge (Hanisch), and `g(r)` is averaged
//! only over sections at least `maximum_radius` from every edge
//! (minus-sampling).

use std::fmt;

use tangle_core::{FiberAssembly, PeriodicCell};

use crate::distribution::Distribution;
use crate::neighbors::collect_fibers;

/// Schema version of [`SliceMetrics`].
pub const SLICE_SCHEMA_VERSION: u32 = 1;

/// Settings for [`analyze_slices`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SliceAnalysisConfig {
    /// Cell axis normal to the slices (0, 1 or 2).
    pub axis: usize,
    /// Number of evenly spaced slices, at the centers of equal slabs.
    pub slice_count: usize,
    /// Largest distance at which `g(r)` is measured. `None` uses eight mean
    /// section spacings, capped at half a periodic in-plane cell length and a
    /// quarter of a non-periodic one.
    pub maximum_radius: Option<f64>,
    /// Number of equal-width `g(r)` bins; zero skips `g(r)`.
    pub bin_count: usize,
    /// Number of evenly spaced quantiles stored for each distribution.
    pub quantile_count: usize,
}

impl Default for SliceAnalysisConfig {
    fn default() -> Self {
        Self {
            axis: 2,
            slice_count: 16,
            maximum_radius: None,
            bin_count: 40,
            quantile_count: 101,
        }
    }
}

/// Cross-section statistics of one assembly state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SliceMetrics {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Cell axis normal to the slices.
    pub axis: usize,
    /// Coordinate of each slice along `axis`.
    pub slice_positions: Vec<f64>,
    /// Number of fiber sections in each slice.
    pub section_counts: Vec<usize>,
    /// In-plane area of one slice.
    pub slice_area: f64,
    /// Mean number of sections per unit slice area.
    pub sections_per_area: Option<f64>,
    /// Center-to-center distance from each section to its nearest neighbor
    /// in the same slice, edge-corrected.
    pub nearest_neighbor_distance: Option<Distribution>,
    /// Mean nearest-neighbor distance over its expectation for a Poisson
    /// arrangement of the same density, `0.5 / √λ` (Clark and Evans 1954):
    /// below one is clustered, above one is more regular than random.
    pub clark_evans_ratio: Option<f64>,
    /// Largest `g(r)` distance actually used.
    pub maximum_radius: f64,
    /// Center of each `g(r)` bin.
    pub pair_correlation_radii: Vec<f64>,
    /// `g(r)` in each bin; `None` when no section was far enough from the
    /// edges.
    pub pair_correlation: Vec<Option<f64>>,
    /// Sections over which `g(r)` was averaged.
    pub pair_correlation_sections: usize,
}

/// Invalid slice settings or an unsupported cell.
#[derive(Clone, Debug, PartialEq)]
pub enum SliceAnalysisError {
    /// The slice axis was not 0, 1 or 2.
    InvalidAxis(usize),
    /// The slice count was zero.
    InvalidSliceCount,
    /// The quantile count was below two.
    InvalidQuantileCount(usize),
    /// A length setting was not positive and finite.
    InvalidLength {
        /// Setting name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// `maximum_radius` exceeded what the cell supports.
    RadiusTooLarge {
        /// Requested radius.
        maximum_radius: f64,
        /// Largest supported radius.
        limit: f64,
    },
    /// The cell basis was not diagonal with positive lengths.
    NonOrthorhombicCell,
}

impl fmt::Display for SliceAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAxis(axis) => write!(formatter, "slice axis must be 0, 1 or 2, got {axis}"),
            Self::InvalidSliceCount => write!(formatter, "slice_count must be positive"),
            Self::InvalidQuantileCount(count) => {
                write!(formatter, "quantile_count must be at least 2, got {count}")
            }
            Self::InvalidLength { name, value } => {
                write!(formatter, "{name} must be positive and finite, got {value}")
            }
            Self::RadiusTooLarge {
                maximum_radius,
                limit,
            } => write!(
                formatter,
                "maximum_radius {maximum_radius} exceeds {limit}, half a periodic or a quarter of a non-periodic in-plane cell length"
            ),
            Self::NonOrthorhombicCell => {
                write!(formatter, "slice analysis supports only orthorhombic cells")
            }
        }
    }
}

impl std::error::Error for SliceAnalysisError {}

/// Measures fiber cross-sections in `config.slice_count` planes normal to
/// `config.axis`.
pub fn analyze_slices(
    assembly: &FiberAssembly,
    config: &SliceAnalysisConfig,
) -> Result<SliceMetrics, SliceAnalysisError> {
    let axis = config.axis;
    if axis > 2 {
        return Err(SliceAnalysisError::InvalidAxis(axis));
    }
    if config.slice_count == 0 {
        return Err(SliceAnalysisError::InvalidSliceCount);
    }
    if config.quantile_count < 2 {
        return Err(SliceAnalysisError::InvalidQuantileCount(
            config.quantile_count,
        ));
    }
    let cell = &assembly.cell;
    let lengths = orthorhombic_lengths(cell).ok_or(SliceAnalysisError::NonOrthorhombicCell)?;
    let plane = [(axis + 1) % 3, (axis + 2) % 3];
    let plane_origin = plane.map(|a| cell.origin[a]);
    let plane_lengths = plane.map(|a| lengths[a]);
    let plane_periodic = plane.map(|a| cell.periodic[a]);
    let slice_area = plane_lengths[0] * plane_lengths[1];

    let step = lengths[axis] / config.slice_count as f64;
    let slice_positions: Vec<f64> = (0..config.slice_count)
        .map(|k| cell.origin[axis] + (k as f64 + 0.5) * step)
        .collect();

    let fibers = collect_fibers(assembly);
    let mut slices: Vec<Vec<[f64; 2]>> = vec![Vec::new(); config.slice_count];
    for fiber in &fibers {
        for pair in fiber.points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            for (slice, &position) in slice_positions.iter().enumerate() {
                for level in plane_levels(
                    a[axis],
                    b[axis],
                    position,
                    lengths[axis],
                    cell.periodic[axis],
                ) {
                    let t = (level - a[axis]) / (b[axis] - a[axis]);
                    let mut point = [0.0; 2];
                    let mut inside = true;
                    for k in 0..2 {
                        let value = a[plane[k]] + t * (b[plane[k]] - a[plane[k]]);
                        let offset = value - plane_origin[k];
                        if plane_periodic[k] {
                            point[k] = offset.rem_euclid(plane_lengths[k]);
                            if point[k] >= plane_lengths[k] {
                                point[k] = 0.0;
                            }
                        } else if (0.0..plane_lengths[k]).contains(&offset) {
                            point[k] = offset;
                        } else {
                            inside = false;
                        }
                    }
                    if inside {
                        slices[slice].push(point);
                    }
                }
            }
        }
    }

    let section_counts: Vec<usize> = slices.iter().map(Vec::len).collect();
    let total_sections: usize = section_counts.iter().sum();
    let sections_per_area = (slice_area > 0.0)
        .then(|| total_sections as f64 / (config.slice_count as f64 * slice_area));

    let radius_limit = (0..2)
        .map(|k| {
            if plane_periodic[k] {
                0.5 * plane_lengths[k]
            } else {
                0.25 * plane_lengths[k]
            }
        })
        .fold(f64::INFINITY, f64::min);
    let maximum_radius = match config.maximum_radius {
        Some(radius) if !(radius.is_finite() && radius > 0.0) => {
            return Err(SliceAnalysisError::InvalidLength {
                name: "maximum_radius",
                value: radius,
            })
        }
        Some(radius) if config.bin_count > 0 && radius > radius_limit => {
            return Err(SliceAnalysisError::RadiusTooLarge {
                maximum_radius: radius,
                limit: radius_limit,
            })
        }
        Some(radius) => radius,
        None => match sections_per_area {
            Some(density) if density > 0.0 => (8.0 / density.sqrt()).min(radius_limit),
            _ => radius_limit,
        },
    };

    let bin_count = config.bin_count;
    let bin_width = maximum_radius / bin_count.max(1) as f64;
    let mut pair_counts = vec![0.0_f64; bin_count];
    let mut expected_per_area = 0.0;
    let mut pair_correlation_sections = 0;
    let mut nearest = Vec::new();
    let mut expected_nearest = 0.0;
    for points in &slices {
        if points.len() < 2 {
            continue;
        }
        let grid = PlanarGrid::new(points, plane_lengths, plane_periodic);
        let intensity = (points.len() - 1) as f64 / slice_area;
        for index in 0..points.len() {
            let edge = grid.edge_distance(points[index]);
            if let Some(distance) = grid.nearest(index) {
                if distance <= edge {
                    nearest.push(distance);
                    expected_nearest += 0.5 / intensity.sqrt();
                }
            }
            if bin_count > 0 && edge >= maximum_radius {
                pair_correlation_sections += 1;
                expected_per_area += intensity;
                grid.within(index, maximum_radius, |distance| {
                    let bin = ((distance / bin_width) as usize).min(bin_count - 1);
                    pair_counts[bin] += 1.0;
                });
            }
        }
    }

    let pair_correlation_radii: Vec<f64> = (0..bin_count)
        .map(|k| (k as f64 + 0.5) * bin_width)
        .collect();
    let pair_correlation = (0..bin_count)
        .map(|k| {
            let inner = k as f64 * bin_width;
            let outer = inner + bin_width;
            let annulus = std::f64::consts::PI * (outer * outer - inner * inner);
            (expected_per_area > 0.0).then(|| pair_counts[k] / (expected_per_area * annulus))
        })
        .collect();
    let clark_evans_ratio =
        (expected_nearest > 0.0).then(|| nearest.iter().sum::<f64>() / expected_nearest);

    Ok(SliceMetrics {
        schema_version: SLICE_SCHEMA_VERSION,
        axis,
        slice_positions,
        section_counts,
        slice_area,
        sections_per_area,
        nearest_neighbor_distance: Distribution::from_values(nearest, config.quantile_count),
        clark_evans_ratio,
        maximum_radius,
        pair_correlation_radii,
        pair_correlation,
        pair_correlation_sections,
    })
}

fn orthorhombic_lengths(cell: &PeriodicCell) -> Option<[f64; 3]> {
    let diagonal =
        (0..3).all(|row| (0..3).all(|column| row == column || cell.basis[row][column] == 0.0));
    let lengths = [cell.basis[0][0], cell.basis[1][1], cell.basis[2][2]];
    (diagonal && lengths.iter().all(|l| l.is_finite() && *l > 0.0)).then_some(lengths)
}

/// Coordinates at which the segment from `a` to `b` crosses the plane at
/// `position` or, along a periodic axis, one of its images. The interval is
/// half-open so a crossing exactly at a vertex is counted once.
fn plane_levels(a: f64, b: f64, position: f64, length: f64, periodic: bool) -> Vec<f64> {
    let (low, high) = if a < b { (a, b) } else { (b, a) };
    if low == high {
        return Vec::new();
    }
    let contains = |level: f64| low <= level && level < high;
    if !periodic {
        return if contains(position) {
            vec![position]
        } else {
            Vec::new()
        };
    }
    let first = ((low - position) / length).ceil() as i64;
    let last = ((high - position) / length).floor() as i64;
    (first..=last)
        .map(|k| position + k as f64 * length)
        .filter(|level| contains(*level))
        .collect()
}

/// Points of one slice binned on a uniform grid for neighbor queries.
struct PlanarGrid<'a> {
    points: &'a [[f64; 2]],
    lengths: [f64; 2],
    periodic: [bool; 2],
    bins: [usize; 2],
    width: [f64; 2],
    offsets: Vec<usize>,
    entries: Vec<u32>,
}

impl<'a> PlanarGrid<'a> {
    fn new(points: &'a [[f64; 2]], lengths: [f64; 2], periodic: [bool; 2]) -> Self {
        // About two points per bin.
        let target = (2.0 * lengths[0] * lengths[1] / points.len() as f64).sqrt();
        let bins = [0, 1].map(|k| ((lengths[k] / target).floor() as usize).clamp(1, 1024));
        let width = [0, 1].map(|k| lengths[k] / bins[k] as f64);
        let mut grid = Self {
            points,
            lengths,
            periodic,
            bins,
            width,
            offsets: vec![0; bins[0] * bins[1] + 1],
            entries: vec![0; points.len()],
        };
        for point in points {
            grid.offsets[grid.bin_of(*point) + 1] += 1;
        }
        for k in 1..grid.offsets.len() {
            grid.offsets[k] += grid.offsets[k - 1];
        }
        let mut cursor = grid.offsets.clone();
        for (index, point) in points.iter().enumerate() {
            let bin = grid.bin_of(*point);
            grid.entries[cursor[bin]] = index as u32;
            cursor[bin] += 1;
        }
        grid
    }

    fn cell_of(&self, point: [f64; 2]) -> [usize; 2] {
        [0, 1].map(|k| ((point[k] / self.width[k]) as usize).min(self.bins[k] - 1))
    }

    fn bin_of(&self, point: [f64; 2]) -> usize {
        let cell = self.cell_of(point);
        cell[0] * self.bins[1] + cell[1]
    }

    fn distance(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let mut squared = 0.0;
        for k in 0..2 {
            let mut delta = b[k] - a[k];
            if self.periodic[k] {
                delta -= self.lengths[k] * (delta / self.lengths[k]).round();
            }
            squared += delta * delta;
        }
        squared.sqrt()
    }

    /// Distance to the nearest non-periodic edge, infinite when none.
    fn edge_distance(&self, point: [f64; 2]) -> f64 {
        (0..2)
            .filter(|k| !self.periodic[*k])
            .map(|k| point[k].min(self.lengths[k] - point[k]))
            .fold(f64::INFINITY, f64::min)
    }

    /// Distinct bin indices along one axis at signed offsets `-ring..=ring`
    /// from `center`, wrapped on a periodic axis and clipped otherwise.
    fn axis_bins(&self, k: usize, center: usize, ring: usize) -> Vec<(usize, usize)> {
        let n = self.bins[k] as i64;
        let mut seen = vec![false; self.bins[k]];
        let mut out = Vec::new();
        // Visit offsets from small to large so each bin keeps its smallest
        // offset magnitude.
        for magnitude in 0..=ring {
            for offset in [-(magnitude as i64), magnitude as i64] {
                let raw = center as i64 + offset;
                let wrapped = if self.periodic[k] {
                    raw.rem_euclid(n)
                } else {
                    raw
                };
                if !(0..n).contains(&wrapped) {
                    continue;
                }
                let bin = wrapped as usize;
                if !seen[bin] {
                    seen[bin] = true;
                    out.push((bin, magnitude));
                }
            }
        }
        out
    }

    /// Calls `visit` with each bin whose offset from `center` has Chebyshev
    /// magnitude exactly `ring`, once per bin.
    fn visit_ring(&self, center: [usize; 2], ring: usize, mut visit: impl FnMut(usize)) {
        let rows = self.axis_bins(0, center[0], ring);
        let columns = self.axis_bins(1, center[1], ring);
        for &(row, row_ring) in &rows {
            for &(column, column_ring) in &columns {
                if row_ring.max(column_ring) == ring {
                    visit(row * self.bins[1] + column);
                }
            }
        }
    }

    /// Largest ring that can still reach a new bin.
    fn ring_limit(&self) -> usize {
        (0..2)
            .map(|k| {
                if self.periodic[k] {
                    self.bins[k] / 2
                } else {
                    self.bins[k] - 1
                }
            })
            .max()
            .unwrap_or(0)
    }

    fn entries(&self, bin: usize) -> &[u32] {
        &self.entries[self.offsets[bin]..self.offsets[bin + 1]]
    }

    fn nearest(&self, index: usize) -> Option<f64> {
        let point = self.points[index];
        let center = self.cell_of(point);
        let minimum_width = self.width[0].min(self.width[1]);
        let mut best = f64::INFINITY;
        for ring in 0..=self.ring_limit() {
            // Points in rings not yet searched are at least this far away.
            if best <= ring.saturating_sub(1) as f64 * minimum_width {
                break;
            }
            self.visit_ring(center, ring, |bin| {
                for &other in self.entries(bin) {
                    if other as usize != index {
                        best = best.min(self.distance(point, self.points[other as usize]));
                    }
                }
            });
        }
        best.is_finite().then_some(best)
    }

    /// Calls `visit` with the distance to every other point closer than
    /// `radius`.
    fn within(&self, index: usize, radius: f64, mut visit: impl FnMut(f64)) {
        let point = self.points[index];
        let center = self.cell_of(point);
        let minimum_width = self.width[0].min(self.width[1]);
        let rings = ((radius / minimum_width).ceil() as usize + 1).min(self.ring_limit());
        for ring in 0..=rings {
            self.visit_ring(center, ring, |bin| {
                for &other in self.entries(bin) {
                    if other as usize != index {
                        let distance = self.distance(point, self.points[other as usize]);
                        if distance < radius {
                            visit(distance);
                        }
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, Section, Vec3};

    fn assembly_from_centerlines(
        cell: PeriodicCell,
        centerlines: &[Vec<Vec3>],
        radius: f64,
    ) -> FiberAssembly {
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

    /// Straight fibers along z through the given in-plane points.
    fn columns(points: &[[f64; 2]], cell: PeriodicCell, radius: f64) -> FiberAssembly {
        let height = cell.basis[2][2];
        let lines: Vec<Vec<Vec3>> = points
            .iter()
            .map(|p| vec![[p[0], p[1], 0.0], [p[0], p[1], height]])
            .collect();
        assembly_from_centerlines(cell, &lines, radius)
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn square_lattice_has_unit_spacing_and_regular_clark_evans_ratio() {
        let points: Vec<[f64; 2]> = (0..10)
            .flat_map(|i| (0..10).map(move |j| [i as f64 + 0.5, j as f64 + 0.5]))
            .collect();
        let cell = PeriodicCell::orthorhombic([10.0, 10.0, 4.0], [true, true, false]);
        let metrics = analyze_slices(
            &columns(&points, cell, 0.1),
            &SliceAnalysisConfig::default(),
        )
        .unwrap();
        assert!(metrics.section_counts.iter().all(|count| *count == 100));
        assert!((metrics.sections_per_area.unwrap() - 1.0).abs() < 1e-12);
        let nearest = metrics.nearest_neighbor_distance.unwrap();
        assert_eq!(nearest.count, 1600);
        assert!((nearest.quantiles[0] - 1.0).abs() < 1e-9);
        assert!((nearest.quantiles[100] - 1.0).abs() < 1e-9);
        // 1 / (0.5 / sqrt(99 / 100)).
        let expected = 1.0 / (0.5 / (99.0_f64 / 100.0).sqrt());
        assert!((metrics.clark_evans_ratio.unwrap() - expected).abs() < 1e-9);
        // No pairs closer than the lattice spacing.
        for (radius, value) in metrics
            .pair_correlation_radii
            .iter()
            .zip(&metrics.pair_correlation)
        {
            if *radius < 0.9 {
                assert_eq!(*value, Some(0.0));
            }
        }
    }

    #[test]
    fn random_sections_have_flat_pair_correlation() {
        let mut state = 7;
        let points: Vec<[f64; 2]> = (0..4000)
            .map(|_| [20.0 * lcg(&mut state), 20.0 * lcg(&mut state)])
            .collect();
        let cell = PeriodicCell::orthorhombic([20.0, 20.0, 1.0], [true, true, false]);
        let config = SliceAnalysisConfig {
            slice_count: 1,
            maximum_radius: Some(2.0),
            bin_count: 8,
            ..SliceAnalysisConfig::default()
        };
        let metrics = analyze_slices(&columns(&points, cell, 0.01), &config).unwrap();
        for value in &metrics.pair_correlation {
            assert!((value.unwrap() - 1.0).abs() < 0.1, "{value:?}");
        }
        assert!((metrics.clark_evans_ratio.unwrap() - 1.0).abs() < 0.05);
        assert_eq!(metrics.pair_correlation_sections, 4000);
    }

    #[test]
    fn nearest_neighbor_search_matches_brute_force() {
        let mut state = 11;
        let points: Vec<[f64; 2]> = (0..300)
            .map(|_| [7.0 * lcg(&mut state), 3.0 * lcg(&mut state)])
            .collect();
        for periodic in [[true, true], [false, true], [false, false]] {
            let grid = PlanarGrid::new(&points, [7.0, 3.0], periodic);
            for index in 0..points.len() {
                let brute = (0..points.len())
                    .filter(|other| *other != index)
                    .map(|other| grid.distance(points[index], points[other]))
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(grid.nearest(index), Some(brute));
                let mut count = 0;
                grid.within(index, 1.2, |_| count += 1);
                let brute_count = (0..points.len())
                    .filter(|other| {
                        *other != index && grid.distance(points[index], points[*other]) < 1.2
                    })
                    .count();
                assert_eq!(count, brute_count);
            }
        }
    }

    #[test]
    fn periodic_fibers_crossing_the_boundary_are_counted_once_per_slice() {
        // A fiber tilted in x wraps across the periodic x face and runs along
        // z for two cell heights.
        let cell = PeriodicCell::orthorhombic([2.0, 2.0, 1.0], [true, true, true]);
        let lines = vec![vec![[1.5, 1.0, -0.5], [2.5, 1.0, 1.5]]];
        let assembly = assembly_from_centerlines(cell, &lines, 0.05);
        let config = SliceAnalysisConfig {
            slice_count: 4,
            bin_count: 0,
            ..SliceAnalysisConfig::default()
        };
        let metrics = analyze_slices(&assembly, &config).unwrap();
        // Each slice position is crossed at two z images.
        assert_eq!(metrics.section_counts, vec![2, 2, 2, 2]);
        assert!(metrics.pair_correlation.is_empty());
    }

    #[test]
    fn fibers_in_the_slice_plane_give_no_sections() {
        let cell = PeriodicCell::orthorhombic([2.0, 2.0, 1.0], [true, true, false]);
        let lines = vec![vec![[0.0, 1.0, 0.25], [2.0, 1.0, 0.25]]];
        let assembly = assembly_from_centerlines(cell, &lines, 0.05);
        let config = SliceAnalysisConfig {
            slice_count: 2,
            ..SliceAnalysisConfig::default()
        };
        let metrics = analyze_slices(&assembly, &config).unwrap();
        assert_eq!(metrics.section_counts, vec![0, 0]);
        assert!(metrics.nearest_neighbor_distance.is_none());
        assert!(metrics.clark_evans_ratio.is_none());
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let cell = PeriodicCell::orthorhombic([2.0, 2.0, 1.0], [false, false, false]);
        let assembly = assembly_from_centerlines(cell, &[], 0.05);
        let config = |edit: fn(&mut SliceAnalysisConfig)| {
            let mut config = SliceAnalysisConfig::default();
            edit(&mut config);
            analyze_slices(&assembly, &config)
        };
        assert_eq!(
            config(|c| c.axis = 3).unwrap_err(),
            SliceAnalysisError::InvalidAxis(3)
        );
        assert_eq!(
            config(|c| c.slice_count = 0).unwrap_err(),
            SliceAnalysisError::InvalidSliceCount
        );
        assert!(matches!(
            config(|c| c.maximum_radius = Some(-1.0)).unwrap_err(),
            SliceAnalysisError::InvalidLength { .. }
        ));
        assert!(matches!(
            config(|c| c.maximum_radius = Some(1.0)).unwrap_err(),
            SliceAnalysisError::RadiusTooLarge { .. }
        ));
    }
}
