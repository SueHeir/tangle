//! Solid and pore statistics along test lines, exact for capsule fibers.
//!
//! Image-based tools such as PuMA measure the pore space on voxels. The same
//! statistics can be computed without voxelizing by casting straight test
//! lines through the fiber capsules (each segment swept by its radius) and
//! intersecting them exactly: every line is cut into solid and void
//! intervals, overlapping fibers are counted once, and nothing depends on a
//! voxel size.
//!
//! * **Solid fraction:** covered length over line length, averaged over all
//!   lines. Unlike the nominal volume fraction it does not double-count
//!   overlaps.
//! * **Chord lengths:** lengths of the solid and void intervals along each
//!   axis. The mean void chord is the mean intercept length; the full
//!   distributions separate a few large pores from many small ones and show
//!   anisotropy between in-plane and through-thickness directions. Chords cut
//!   by a non-periodic cell face are dropped, as their true length is unknown.
//! * **Two-point correlation `S₂(r)`:** probability that two points a distance
//!   `r` apart along an axis are both solid. It starts at the solid fraction,
//!   falls over about a fiber diameter, and levels at the solid fraction
//!   squared when solid positions decorrelate.
//! * **Solid-fraction profile:** solid fraction as a function of position
//!   along `profile_axis`, for example through the thickness of a stack.
//!
//! Lines lie on a regular `line_count × line_count` grid in each of the
//! three axis directions.
//!
//! Oval (elliptical) sections are treated as round with the equal-area
//! radius, as in the other analyses.

use std::fmt;

use tangle_core::{FiberAssembly, PeriodicCell, Vec3};

use crate::distribution::Distribution;
use crate::neighbors::collect_fibers;

/// Schema version of [`PhaseMetrics`].
pub const PHASE_SCHEMA_VERSION: u32 = 1;

/// Settings for [`analyze_phases`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PhaseAnalysisConfig {
    /// Lines per in-plane axis, so each direction casts `line_count²` lines.
    pub line_count: usize,
    /// Largest `S₂` lag. `None` uses half the smallest cell length.
    pub maximum_lag: Option<f64>,
    /// Number of `S₂` lags after zero.
    pub lag_count: usize,
    /// Axis along which the solid-fraction profile is reported.
    pub profile_axis: usize,
    /// Number of evenly spaced quantiles stored for each distribution.
    pub quantile_count: usize,
}

impl Default for PhaseAnalysisConfig {
    fn default() -> Self {
        Self {
            line_count: 64,
            maximum_lag: None,
            lag_count: 32,
            profile_axis: 2,
            quantile_count: 101,
        }
    }
}

/// Solid and pore statistics of one assembly state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PhaseMetrics {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Lines per in-plane axis actually used.
    pub line_count: usize,
    /// Solid fraction averaged over the lines of all three directions.
    pub solid_fraction: f64,
    /// Solid fraction measured along x, y and z lines.
    pub solid_fraction_by_axis: Vec<f64>,
    /// Solid chord lengths along x, y and z.
    pub solid_chord_length: Vec<Option<Distribution>>,
    /// Void chord (intercept) lengths along x, y and z.
    pub void_chord_length: Vec<Option<Distribution>>,
    /// Lags at which `S₂` is reported, starting at zero.
    pub correlation_lags: Vec<f64>,
    /// `S₂` along x, y and z at each lag.
    pub two_point_correlation: Vec<Vec<f64>>,
    /// Axis of the solid-fraction profile.
    pub profile_axis: usize,
    /// Coordinates along `profile_axis` of the profile samples.
    pub profile_positions: Vec<f64>,
    /// Solid fraction at each profile position.
    pub solid_fraction_profile: Vec<f64>,
}

/// Invalid phase settings or an unsupported cell.
#[derive(Clone, Debug, PartialEq)]
pub enum PhaseAnalysisError {
    /// The line count was zero.
    InvalidLineCount,
    /// The profile axis was not 0, 1 or 2.
    InvalidAxis(usize),
    /// The quantile count was below two.
    InvalidQuantileCount(usize),
    /// `maximum_lag` was not positive and finite, or not below the cell
    /// length along a non-periodic axis.
    InvalidMaximumLag(f64),
    /// The cell basis was not diagonal with positive lengths.
    NonOrthorhombicCell,
}

impl fmt::Display for PhaseAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLineCount => write!(formatter, "line_count must be positive"),
            Self::InvalidAxis(axis) => {
                write!(formatter, "profile axis must be 0, 1 or 2, got {axis}")
            }
            Self::InvalidQuantileCount(count) => {
                write!(formatter, "quantile_count must be at least 2, got {count}")
            }
            Self::InvalidMaximumLag(lag) => write!(
                formatter,
                "maximum_lag must be positive, finite and at most half the smallest cell length, got {lag}"
            ),
            Self::NonOrthorhombicCell => {
                write!(formatter, "phase analysis supports only orthorhombic cells")
            }
        }
    }
}

impl std::error::Error for PhaseAnalysisError {}

/// Casts `config.line_count²` lines along each axis through the fiber
/// capsules and measures solid and void statistics.
pub fn analyze_phases(
    assembly: &FiberAssembly,
    config: &PhaseAnalysisConfig,
) -> Result<PhaseMetrics, PhaseAnalysisError> {
    if config.line_count == 0 {
        return Err(PhaseAnalysisError::InvalidLineCount);
    }
    if config.profile_axis > 2 {
        return Err(PhaseAnalysisError::InvalidAxis(config.profile_axis));
    }
    if config.quantile_count < 2 {
        return Err(PhaseAnalysisError::InvalidQuantileCount(
            config.quantile_count,
        ));
    }
    let cell = &assembly.cell;
    let lengths = orthorhombic_lengths(cell).ok_or(PhaseAnalysisError::NonOrthorhombicCell)?;
    let smallest = lengths.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_lag = match config.maximum_lag {
        None => 0.5 * smallest,
        Some(lag) if lag.is_finite() && lag > 0.0 && lag <= 0.5 * smallest => lag,
        Some(lag) => return Err(PhaseAnalysisError::InvalidMaximumLag(lag)),
    };
    let correlation_lags: Vec<f64> = (0..=config.lag_count)
        .map(|k| maximum_lag * k as f64 / config.lag_count.max(1) as f64)
        .collect();

    let segments = capsules(assembly);
    let n = config.line_count;
    let mut solid_fraction_by_axis = Vec::with_capacity(3);
    let mut solid_chord_length = Vec::with_capacity(3);
    let mut void_chord_length = Vec::with_capacity(3);
    let mut two_point_correlation = Vec::with_capacity(3);
    let mut profile_sum = vec![0.0; n];
    let mut profile_lines = vec![0usize; n];
    for axis in 0..3 {
        let lines = cast_lines(&segments, cell, lengths, axis, n);
        let length = lengths[axis];
        let periodic = cell.periodic[axis];
        let mut covered = 0.0;
        let mut solid_chords = Vec::new();
        let mut void_chords = Vec::new();
        let mut correlation = vec![0.0; correlation_lags.len()];
        let plane = [(axis + 1) % 3, (axis + 2) % 3];
        for (line_index, intervals) in lines.iter().enumerate() {
            let merged = merge(intervals.clone());
            let solid: f64 = merged.iter().map(|(a, b)| b - a).sum();
            covered += solid;
            chords(
                &merged,
                length,
                periodic,
                &mut solid_chords,
                &mut void_chords,
            );
            for (value, &lag) in correlation.iter_mut().zip(&correlation_lags) {
                *value += two_point(&merged, length, periodic, lag);
            }
            // Line (i, j) sits at index i along plane[0] and j along plane[1].
            let grid = [line_index / n, line_index % n];
            for k in 0..2 {
                if plane[k] == config.profile_axis {
                    profile_sum[grid[k]] += solid / length;
                    profile_lines[grid[k]] += 1;
                }
            }
        }
        let line_total = (n * n) as f64;
        solid_fraction_by_axis.push(covered / (line_total * length));
        solid_chord_length.push(Distribution::from_values(
            solid_chords,
            config.quantile_count,
        ));
        void_chord_length.push(Distribution::from_values(
            void_chords,
            config.quantile_count,
        ));
        two_point_correlation.push(correlation.iter().map(|v| v / line_total).collect());
    }

    let profile_axis = config.profile_axis;
    let profile_positions: Vec<f64> = (0..n)
        .map(|i| cell.origin[profile_axis] + (i as f64 + 0.5) * lengths[profile_axis] / n as f64)
        .collect();
    let solid_fraction_profile: Vec<f64> = profile_sum
        .iter()
        .zip(&profile_lines)
        .map(|(sum, count)| sum / (*count).max(1) as f64)
        .collect();

    Ok(PhaseMetrics {
        schema_version: PHASE_SCHEMA_VERSION,
        line_count: n,
        solid_fraction: solid_fraction_by_axis.iter().sum::<f64>() / 3.0,
        solid_fraction_by_axis,
        solid_chord_length,
        void_chord_length,
        correlation_lags,
        two_point_correlation,
        profile_axis,
        profile_positions,
        solid_fraction_profile,
    })
}

struct Capsule {
    start: Vec3,
    end: Vec3,
    radius: f64,
}

fn capsules(assembly: &FiberAssembly) -> Vec<Capsule> {
    collect_fibers(assembly)
        .into_iter()
        .flat_map(|fiber| {
            let radius = fiber.radius;
            fiber
                .points
                .windows(2)
                .map(|pair| Capsule {
                    start: pair[0],
                    end: pair[1],
                    radius,
                })
                .collect::<Vec<_>>()
        })
        .filter(|capsule| capsule.radius > 0.0)
        .collect()
}

fn orthorhombic_lengths(cell: &PeriodicCell) -> Option<[f64; 3]> {
    let diagonal =
        (0..3).all(|row| (0..3).all(|column| row == column || cell.basis[row][column] == 0.0));
    let lengths = [cell.basis[0][0], cell.basis[1][1], cell.basis[2][2]];
    (diagonal && lengths.iter().all(|l| l.is_finite() && *l > 0.0)).then_some(lengths)
}

/// Solid intervals, in line coordinates `[0, length)`, of every line along
/// `axis`. Line `i * n + j` passes through the `i`-th position along the
/// first plane axis and the `j`-th along the second.
fn cast_lines(
    capsules: &[Capsule],
    cell: &PeriodicCell,
    lengths: [f64; 3],
    axis: usize,
    n: usize,
) -> Vec<Vec<(f64, f64)>> {
    let plane = [(axis + 1) % 3, (axis + 2) % 3];
    let spacing = plane.map(|a| lengths[a] / n as f64);
    let length = lengths[axis];
    let mut lines = vec![Vec::new(); n * n];
    for capsule in capsules {
        let bounds = plane.map(|a| {
            let low = capsule.start[a].min(capsule.end[a]) - capsule.radius - cell.origin[a];
            let high = capsule.start[a].max(capsule.end[a]) + capsule.radius - cell.origin[a];
            (low, high)
        });
        let images =
            [0, 1].map(|k| image_shifts(bounds[k], lengths[plane[k]], cell.periodic[plane[k]]));
        for &shift_u in &images[0] {
            let (low, high) = (bounds[0].0 + shift_u, bounds[0].1 + shift_u);
            for i in line_range(low, high, spacing[0], n) {
                let u = (i as f64 + 0.5) * spacing[0] + cell.origin[plane[0]] - shift_u;
                for &shift_v in &images[1] {
                    let (low, high) = (bounds[1].0 + shift_v, bounds[1].1 + shift_v);
                    for j in line_range(low, high, spacing[1], n) {
                        let v = (j as f64 + 0.5) * spacing[1] + cell.origin[plane[1]] - shift_v;
                        if let Some((enter, exit)) =
                            line_capsule_interval(capsule, axis, plane, [u, v])
                        {
                            let enter = enter - cell.origin[axis];
                            let exit = exit - cell.origin[axis];
                            place_interval(
                                &mut lines[i * n + j],
                                enter,
                                exit,
                                length,
                                cell.periodic[axis],
                            );
                        }
                    }
                }
            }
        }
    }
    lines
}

/// Lattice shifts that bring `[low, high]` onto `[0, length)`; only zero
/// along a non-periodic axis.
fn image_shifts((low, high): (f64, f64), length: f64, periodic: bool) -> Vec<f64> {
    if !periodic {
        return vec![0.0];
    }
    let first = (-high / length).ceil() as i64;
    let last = ((length - low) / length).floor() as i64;
    (first..=last).map(|k| k as f64 * length).collect()
}

/// Indices of lines at `(i + 0.5) * spacing` inside `[low, high]`.
fn line_range(low: f64, high: f64, spacing: f64, n: usize) -> std::ops::Range<usize> {
    let first = ((low / spacing - 0.5).ceil()).max(0.0);
    let last = ((high / spacing - 0.5).floor() + 1.0).min(n as f64);
    if last <= first {
        0..0
    } else {
        first as usize..last as usize
    }
}

/// Range of the coordinate along `axis` over which the line through
/// `position` (coordinates along `plane`) lies inside the capsule.
fn line_capsule_interval(
    capsule: &Capsule,
    axis: usize,
    plane: [usize; 2],
    position: [f64; 2],
) -> Option<(f64, f64)> {
    let r2 = capsule.radius * capsule.radius;
    // Squared in-plane distance g(s) = A s² + 2 B s + C from the line to the
    // segment point at parameter s.
    let offset = [0, 1].map(|k| capsule.start[plane[k]] - position[k]);
    let direction = [0, 1].map(|k| capsule.end[plane[k]] - capsule.start[plane[k]]);
    let a = direction[0] * direction[0] + direction[1] * direction[1];
    let b = offset[0] * direction[0] + offset[1] * direction[1];
    let c = offset[0] * offset[0] + offset[1] * offset[1];
    let along = |s: f64| capsule.start[axis] + s * (capsule.end[axis] - capsule.start[axis]);
    let half_width = |s: f64| (r2 - (a * s * s + 2.0 * b * s + c)).max(0.0).sqrt();

    let (s_low, s_high) = if a <= 1e-300 {
        if c > r2 {
            return None;
        }
        (0.0, 1.0)
    } else {
        let discriminant = b * b - a * (c - r2);
        if discriminant < 0.0 {
            return None;
        }
        let root = discriminant.sqrt();
        let low = ((-b - root) / a).max(0.0);
        let high = ((-b + root) / a).min(1.0);
        if low > high {
            return None;
        }
        (low, high)
    };
    // along(s) ± half_width(s) is concave (+) or convex (−) in s.
    let exit = golden_maximum(s_low, s_high, |s| along(s) + half_width(s));
    let enter = -golden_maximum(s_low, s_high, |s| -(along(s) - half_width(s)));
    (exit > enter).then_some((enter, exit))
}

/// Maximum of a concave function on `[low, high]`, including the ends.
fn golden_maximum(mut low: f64, mut high: f64, f: impl Fn(f64) -> f64) -> f64 {
    let ends = f(low).max(f(high));
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut x1 = high - ratio * (high - low);
    let mut x2 = low + ratio * (high - low);
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..80 {
        if high - low <= 1e-15 {
            break;
        }
        if f1 < f2 {
            low = x1;
            x1 = x2;
            f1 = f2;
            x2 = low + ratio * (high - low);
            f2 = f(x2);
        } else {
            high = x2;
            x2 = x1;
            f2 = f1;
            x1 = high - ratio * (high - low);
            f1 = f(x1);
        }
    }
    ends.max(f1).max(f2).max(f(0.5 * (low + high)))
}

/// Adds `[enter, exit]` to a line of `length`, wrapped along a periodic
/// axis and clipped otherwise.
fn place_interval(line: &mut Vec<(f64, f64)>, enter: f64, exit: f64, length: f64, periodic: bool) {
    if periodic {
        if exit - enter >= length {
            line.push((0.0, length));
            return;
        }
        let shift = (enter / length).floor() * length;
        let (enter, exit) = (enter - shift, exit - shift);
        if exit > length {
            line.push((enter, length));
            line.push((0.0, exit - length));
        } else {
            line.push((enter, exit));
        }
    } else {
        let (enter, exit) = (enter.max(0.0), exit.min(length));
        if exit > enter {
            line.push((enter, exit));
        }
    }
}

/// Sorted union of intervals.
fn merge(mut intervals: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(intervals.len());
    for (start, end) in intervals {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Appends the complete solid and void chords of one merged line.
fn chords(
    merged: &[(f64, f64)],
    length: f64,
    periodic: bool,
    solid: &mut Vec<f64>,
    void: &mut Vec<f64>,
) {
    if merged.is_empty() {
        return;
    }
    if periodic {
        let covered: f64 = merged.iter().map(|(a, b)| b - a).sum();
        if covered >= length {
            return;
        }
        // Join the intervals that meet across the periodic boundary.
        let wraps = merged.len() > 1 && merged[0].0 <= 0.0 && merged[merged.len() - 1].1 >= length;
        let body = if wraps {
            &merged[1..merged.len() - 1]
        } else {
            merged
        };
        if wraps {
            let first = merged[0];
            let last = merged[merged.len() - 1];
            solid.push((first.1 - first.0) + (last.1 - last.0));
        }
        solid.extend(body.iter().map(|(a, b)| b - a));
        void.extend(merged.windows(2).map(|pair| pair[1].0 - pair[0].1));
        if !wraps {
            void.push(length - merged[merged.len() - 1].1 + merged[0].0);
        }
    } else {
        solid.extend(
            merged
                .iter()
                .filter(|(a, b)| *a > 0.0 && *b < length)
                .map(|(a, b)| b - a),
        );
        void.extend(merged.windows(2).map(|pair| pair[1].0 - pair[0].1));
    }
}

/// Fraction of line positions `x` (with `x + lag` still on the line when it
/// is not periodic) at which both `x` and `x + lag` are solid.
fn two_point(merged: &[(f64, f64)], length: f64, periodic: bool, lag: f64) -> f64 {
    let shifted: Vec<(f64, f64)> = if periodic {
        merged
            .iter()
            .map(|(a, b)| (a - lag, b - lag))
            .chain(
                merged
                    .iter()
                    .map(|(a, b)| (a - lag + length, b - lag + length)),
            )
            .collect()
    } else {
        merged.iter().map(|(a, b)| (a - lag, b - lag)).collect()
    };
    let mut overlap = 0.0;
    let (mut i, mut j) = (0, 0);
    while i < merged.len() && j < shifted.len() {
        let (a, b) = merged[i];
        let (c, d) = shifted[j];
        overlap += (b.min(d) - a.max(c)).max(0.0);
        if b < d {
            i += 1;
        } else {
            j += 1;
        }
    }
    if periodic {
        overlap / length
    } else {
        overlap / (length - lag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use tangle_core::{FiberId, Section};

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

    #[test]
    fn a_periodic_rod_has_the_exact_solid_fraction_and_chords() {
        // A rod along z through a periodic 1×1×1 cell, radius 0.2.
        let cell = PeriodicCell::orthorhombic([1.0, 1.0, 1.0], [true; 3]);
        let radius = 0.2;
        let assembly =
            assembly_from_centerlines(cell, &[vec![[0.5, 0.5, -0.5], [0.5, 0.5, 1.5]]], radius);
        let config = PhaseAnalysisConfig {
            line_count: 200,
            ..PhaseAnalysisConfig::default()
        };
        let metrics = analyze_phases(&assembly, &config).unwrap();
        let area = PI * radius * radius;
        // Lines along z are either fully solid or fully void.
        assert!((metrics.solid_fraction_by_axis[2] - area).abs() < 0.01);
        assert!(metrics.solid_chord_length[2].is_none());
        // Lines along x through the rod: chord 2√(r² − d²), void the rest.
        assert!((metrics.solid_fraction_by_axis[0] - area).abs() < 1e-3);
        let solid = metrics.solid_chord_length[0].as_ref().unwrap();
        assert!((solid.quantiles[100] - 2.0 * radius).abs() < 1e-3);
        let void = metrics.void_chord_length[0].as_ref().unwrap();
        assert!((void.quantiles[0] - (1.0 - 2.0 * radius)).abs() < 1e-3);
        // The mean chord over lines that hit is area / (2r) → πr/2.
        assert!((solid.mean - PI * radius / 2.0).abs() < 2e-3);
        // S₂(0) is the solid fraction; the profile along z is flat.
        assert!(
            (metrics.two_point_correlation[0][0] - metrics.solid_fraction_by_axis[0]).abs() < 1e-12
        );
        for value in &metrics.solid_fraction_profile {
            assert!((value - area).abs() < 2e-3);
        }
    }

    #[test]
    fn line_capsule_interval_matches_spheres_and_cylinders() {
        let capsule = Capsule {
            start: [0.0, 0.0, 0.0],
            end: [1.0, 0.0, 0.0],
            radius: 0.5,
        };
        // A line along x at distance 0.3 crosses the whole capsule.
        let (enter, exit) = line_capsule_interval(&capsule, 0, [1, 2], [0.3, 0.0]).unwrap();
        let cap = (0.25_f64 - 0.09).sqrt();
        assert!((enter + cap).abs() < 1e-9);
        assert!((exit - 1.0 - cap).abs() < 1e-9);
        // A line along z through the axis crosses the cylinder.
        let (enter, exit) = line_capsule_interval(&capsule, 2, [0, 1], [0.5, 0.0]).unwrap();
        assert!((enter + 0.5).abs() < 1e-9 && (exit - 0.5).abs() < 1e-9);
        // A line along z past the end only crosses the spherical cap.
        let (enter, exit) = line_capsule_interval(&capsule, 2, [0, 1], [1.3, 0.0]).unwrap();
        assert!((enter + cap).abs() < 1e-9 && (exit - cap).abs() < 1e-9);
        // A tilted segment: the chord through a point on its axis.
        let tilted = Capsule {
            start: [0.0, 0.0, 0.0],
            end: [1.0, 0.0, 1.0],
            radius: 0.1,
        };
        let (enter, exit) = line_capsule_interval(&tilted, 2, [0, 1], [0.5, 0.0]).unwrap();
        // Along z the cylinder at 45° is 2r / sin 45° long.
        assert!((exit - enter - 0.2 * 2.0_f64.sqrt()).abs() < 1e-9);
        assert!(line_capsule_interval(&tilted, 2, [0, 1], [0.5, 0.2]).is_none());
    }

    #[test]
    fn two_point_correlation_of_a_periodic_stripe() {
        // One solid stripe [0.2, 0.5) on a periodic unit line.
        let merged = [(0.2, 0.5)];
        assert!((two_point(&merged, 1.0, true, 0.0) - 0.3).abs() < 1e-12);
        assert!((two_point(&merged, 1.0, true, 0.1) - 0.2).abs() < 1e-12);
        assert!((two_point(&merged, 1.0, true, 0.3)).abs() < 1e-12);
        assert!((two_point(&merged, 1.0, true, 0.8) - 0.1).abs() < 1e-12);
        // Not periodic: positions x in [0, 0.9) with x + 0.1 on the line.
        assert!((two_point(&merged, 1.0, false, 0.1) - 0.2 / 0.9).abs() < 1e-12);
    }

    #[test]
    fn chords_drop_truncated_ends_and_join_across_a_periodic_boundary() {
        let merged = [(0.0, 0.1), (0.3, 0.4), (0.9, 1.0)];
        let (mut solid, mut void) = (Vec::new(), Vec::new());
        chords(&merged, 1.0, false, &mut solid, &mut void);
        assert_eq!(solid.len(), 1);
        assert!((solid[0] - 0.1).abs() < 1e-12);
        assert_eq!(void.len(), 2);
        let (mut solid, mut void) = (Vec::new(), Vec::new());
        chords(&merged, 1.0, true, &mut solid, &mut void);
        solid.sort_by(f64::total_cmp);
        void.sort_by(f64::total_cmp);
        assert_eq!(solid.len(), 2);
        assert!((solid[0] - 0.1).abs() < 1e-12 && (solid[1] - 0.2).abs() < 1e-12);
        assert_eq!(void.len(), 2);
        assert!((void[0] - 0.2).abs() < 1e-12 && (void[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn overlapping_fibers_are_counted_once_and_profiles_follow_a_layer() {
        // Two identical rods along x in the lower half of a z-slab.
        let cell = PeriodicCell::orthorhombic([1.0, 1.0, 1.0], [true, true, false]);
        let rod = vec![[-0.5, 0.5, 0.25], [1.5, 0.5, 0.25]];
        let assembly = assembly_from_centerlines(cell, &[rod.clone(), rod], 0.1);
        let config = PhaseAnalysisConfig {
            line_count: 100,
            ..PhaseAnalysisConfig::default()
        };
        let metrics = analyze_phases(&assembly, &config).unwrap();
        let area = PI * 0.01;
        assert!((metrics.solid_fraction_by_axis[1] - area).abs() < 2e-3);
        assert!((metrics.solid_fraction - area).abs() < 5e-3);
        let (upper, lower): (Vec<_>, Vec<_>) = metrics
            .profile_positions
            .iter()
            .zip(&metrics.solid_fraction_profile)
            .partition(|(z, _)| **z > 0.5);
        assert!(upper.iter().all(|(_, value)| **value == 0.0));
        assert!(lower.iter().any(|(_, value)| **value > 0.1));
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let cell = PeriodicCell::orthorhombic([1.0, 2.0, 3.0], [false; 3]);
        let assembly = assembly_from_centerlines(cell, &[], 0.1);
        let run = |edit: fn(&mut PhaseAnalysisConfig)| {
            let mut config = PhaseAnalysisConfig::default();
            edit(&mut config);
            analyze_phases(&assembly, &config)
        };
        assert_eq!(
            run(|c| c.line_count = 0).unwrap_err(),
            PhaseAnalysisError::InvalidLineCount
        );
        assert_eq!(
            run(|c| c.profile_axis = 5).unwrap_err(),
            PhaseAnalysisError::InvalidAxis(5)
        );
        assert_eq!(
            run(|c| c.maximum_lag = Some(0.6)).unwrap_err(),
            PhaseAnalysisError::InvalidMaximumLag(0.6)
        );
        let empty = run(|_| {}).unwrap();
        assert_eq!(empty.solid_fraction, 0.0);
        assert!(empty.void_chord_length.iter().all(Option::is_none));
    }
}
