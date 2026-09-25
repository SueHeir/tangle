//! How sure the fit is of each fitted node (`_confidence.node_confidence`).
//!
//! A node is trusted when, where it sits: the scan is fiber across the core
//! of its capsule (**image**); just outside its capsule the scan is void or
//! inside another fit (**surround**); its core is not inside another fit
//! (**ownership**); the scan's thickness there matches its radius
//! (**thickness**); and it barely moved in the last solve (**stability**,
//! when the previous positions are given). Each is a score in [0, 1] and
//! the confidence is their product, read on rings of points around the
//! centerline every `spacing` along it.

use std::collections::HashMap;

use crate::line::{cross, norm, resample, scale, sub, tangents, Point};
use crate::raster::segment_distance;
use crate::sample::trilinear;
use crate::Shape;

/// The five component names, in the order their product is taken.
pub const COMPONENTS: [&str; 5] = ["image", "surround", "ownership", "thickness", "stability"];

/// Every fit's segments, to ask whether points lie inside another fit.
pub struct Segments {
    a: Vec<Point>,
    b: Vec<Point>,
    fiber: Vec<usize>,
    radius: Vec<f64>,
    cell: f64,
    reach: f64,
    buckets: HashMap<[i64; 3], Vec<usize>>,
}

impl Segments {
    /// `extra` is the largest margin [`Segments::covered`] will be asked for.
    pub fn new(lines: &[Vec<Point>], radii: &[f64], extra: f64) -> Self {
        let (mut a, mut b, mut fiber, mut radius) = (vec![], vec![], vec![], vec![]);
        for (f, line) in lines.iter().enumerate() {
            match line.len() {
                0 => {}
                1 => {
                    a.push(line[0]);
                    b.push(line[0]);
                    fiber.push(f);
                    radius.push(radii[f]);
                }
                _ => {
                    for pair in line.windows(2) {
                        a.push(pair[0]);
                        b.push(pair[1]);
                        fiber.push(f);
                        radius.push(radii[f]);
                    }
                }
            }
        }
        let half = a
            .iter()
            .zip(&b)
            .map(|(&p, &q)| 0.5 * norm(sub(q, p)))
            .fold(0.0, f64::max);
        let reach = radius.iter().cloned().fold(0.0, f64::max) + extra + half;
        let cell = reach.max(1.0);
        let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for s in 0..a.len() {
            let middle = scale(crate::line::add(a[s], b[s]), 0.5);
            buckets.entry(key(middle, cell)).or_default().push(s);
        }
        Self {
            a,
            b,
            fiber,
            radius,
            cell,
            reach,
            buckets,
        }
    }

    /// Whether `p` is within radius + `extra` of a fit other than `skip`.
    pub fn covered(&self, p: Point, skip: usize, extra: f64) -> bool {
        let center = key(p, self.cell);
        let span = (self.reach / self.cell).ceil() as i64;
        for dx in -span..=span {
            for dy in -span..=span {
                for dz in -span..=span {
                    let Some(bucket) =
                        self.buckets
                            .get(&[center[0] + dx, center[1] + dy, center[2] + dz])
                    else {
                        continue;
                    };
                    for &s in bucket {
                        if self.fiber[s] != skip
                            && segment_distance(p, self.a[s], self.b[s]) <= self.radius[s] + extra
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

fn key(p: Point, cell: f64) -> [i64; 3] {
    [
        (p[0] / cell).floor() as i64,
        (p[1] / cell).floor() as i64,
        (p[2] / cell).floor() as i64,
    ]
}

/// Two unit normals per point (`_confidence._normals`).
fn normals(points: &[Point]) -> Vec<(Point, Point)> {
    let t = if points.len() > 1 {
        tangents(points)
    } else {
        vec![[1.0, 0.0, 0.0]; points.len()]
    };
    t.iter()
        .map(|&t| {
            let helper = if t[2].abs() < 0.9 {
                [0.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let u = cross(t, helper);
            let u = scale(u, 1.0 / norm(u).max(1e-12));
            (u, cross(t, u))
        })
        .collect()
}

/// Where a profile falls below half its peak (peak searched within
/// `peak_reach`), walking outward (`_image.half_radius`).
pub fn half_radius(profile: &[f64], distances: &[f64], peak_reach: f64) -> f64 {
    let limit = distances
        .iter()
        .filter(|&&d| d <= peak_reach)
        .count()
        .max(1);
    let mut top = 0;
    for k in 1..limit.min(profile.len()) {
        if profile[k] > profile[top] {
            top = k;
        }
    }
    let half = 0.5 * profile[top];
    if !(half > 0.0) {
        return 0.0;
    }
    let Some(k) = (top + 1..profile.len()).find(|&k| profile[k] < half) else {
        return distances[distances.len() - 1];
    };
    let (a, b) = (profile[k - 1], profile[k]);
    let step = if distances.len() > 1 {
        distances[1] - distances[0]
    } else {
        0.0
    };
    distances[k - 1] + step * (a - half) / (a - b).max(1e-12)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Median over each value and its two neighbors (edges repeated).
fn smooth(value: &[f64]) -> Vec<f64> {
    if value.len() < 3 {
        return value.to_vec();
    }
    let n = value.len();
    (0..n)
        .map(|i| {
            let mut three = [
                value[i.saturating_sub(1)],
                value[i],
                value[(i + 1).min(n - 1)],
            ];
            median(&mut three)
        })
        .collect()
}

/// Each node's value: the lowest sample within half a segment of it
/// (`_confidence._to_nodes`).
fn to_nodes(line: &[Point], samples: usize, value: &[f64]) -> Vec<f64> {
    if line.len() < 2 || samples < 2 {
        let low = value.iter().cloned().fold(f64::INFINITY, f64::min);
        return vec![if value.is_empty() { 0.0 } else { low }; line.len()];
    }
    let mut arc = vec![0.0];
    for pair in line.windows(2) {
        arc.push(arc[arc.len() - 1] + norm(sub(pair[1], pair[0])));
    }
    let total = arc[arc.len() - 1];
    let sample_arc: Vec<f64> = (0..samples)
        .map(|k| {
            if k + 1 == samples {
                total
            } else {
                k as f64 * (total / (samples - 1) as f64)
            }
        })
        .collect();
    (0..line.len())
        .map(|i| {
            let low = if i == 0 {
                f64::NEG_INFINITY
            } else {
                0.5 * (arc[i - 1] + arc[i])
            };
            let high = if i + 1 == line.len() {
                f64::INFINITY
            } else {
                0.5 * (arc[i] + arc[i + 1])
            };
            let picked = (0..samples)
                .filter(|&k| sample_arc[k] >= low && sample_arc[k] <= high)
                .map(|k| value[k])
                .fold(f64::INFINITY, f64::min);
            if picked.is_finite() {
                picked
            } else {
                interp(arc[i], &sample_arc, value)
            }
        })
        .collect()
}

fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// Distance from `p` to polyline `line`.
fn distance_to_polyline(p: Point, line: &[Point]) -> f64 {
    line.windows(2)
        .map(|pair| segment_distance(p, pair[0], pair[1]))
        .fold(f64::INFINITY, f64::min)
}

/// Settings of [`node_confidence`].
#[derive(Clone, Copy, Debug)]
pub struct ConfidenceSettings {
    pub spacing: f64,
    /// How far the foreground over-reaches the fibers.
    pub margin: f64,
    /// How far the fibers' cross-section radius over-reaches.
    pub thickness_margin: f64,
    pub ring: usize,
    pub thickness_tolerance: f64,
}

/// Per fiber: per-node confidence, the same without stability, and per
/// sample every component (for the summary).
pub struct Confidence {
    pub per_node: Vec<Vec<f64>>,
    pub settled: Vec<Vec<f64>>,
    /// Per fiber, the smoothed per-sample confidence.
    pub per_sample: Vec<Vec<f64>>,
    /// Per component (in [`COMPONENTS`] order), per fiber, per sample.
    pub parts: [Vec<Vec<f64>>; 5],
}

/// Confidence of every node of `lines` (`_confidence.node_confidence`).
/// `image` is the normalized scan (void ~0, fiber ~1) and `depth` the
/// foreground's local thickness, both `(z, y, x)` of `shape`.
pub fn node_confidence(
    image: &[f32],
    depth: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    previous: Option<&[Vec<Point>]>,
    s: ConfidenceSettings,
) -> Confidence {
    let segments = Segments::new(lines, radii, s.margin + 0.5);
    let ring = s.ring;
    let angles: Vec<f64> = (0..ring)
        .map(|k| 2.0 * std::f64::consts::PI * k as f64 / ring as f64)
        .collect();
    let per_fiber: Vec<([Vec<f64>; 5], usize)> = std::thread::scope(|scope| {
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(lines.len().max(1));
        let chunk = lines.len().div_ceil(workers.max(1)).max(1);
        let handles: Vec<_> = (0..lines.len())
            .step_by(chunk)
            .map(|start| {
                let segments = &segments;
                let angles = &angles;
                scope.spawn(move || {
                    (start..(start + chunk).min(lines.len()))
                        .map(|f| {
                            fiber_parts(
                                image, depth, shape, lines, radii, previous, &s, segments, angles,
                                f,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("confidence worker"))
            .collect()
    });
    let mut parts: [Vec<Vec<f64>>; 5] = Default::default();
    let mut per_node = Vec::with_capacity(lines.len());
    let mut settled = Vec::with_capacity(lines.len());
    let mut per_sample = Vec::with_capacity(lines.len());
    for (f, (fiber, samples)) in per_fiber.into_iter().enumerate() {
        let product = |with_stability: bool| -> Vec<f64> {
            (0..samples)
                .map(|k| {
                    let mut value = fiber[0][k];
                    for part in &fiber[1..if with_stability { 5 } else { 4 }] {
                        value *= part[k];
                    }
                    value
                })
                .collect()
        };
        let full = smooth(&product(true));
        let calm = smooth(&product(false));
        per_node.push(to_nodes(&lines[f], samples, &full));
        settled.push(to_nodes(&lines[f], samples, &calm));
        per_sample.push(full);
        for (c, part) in fiber.into_iter().enumerate() {
            parts[c].push(part);
        }
    }
    Confidence {
        per_node,
        settled,
        per_sample,
        parts,
    }
}

#[allow(clippy::too_many_arguments)]
fn fiber_parts(
    image: &[f32],
    depth: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    previous: Option<&[Vec<Point>]>,
    s: &ConfidenceSettings,
    segments: &Segments,
    angles: &[f64],
    f: usize,
) -> ([Vec<f64>; 5], usize) {
    let line = &lines[f];
    let points = if line.len() > 1 {
        resample(line, s.spacing)
    } else {
        line.clone()
    };
    let n = points.len();
    let r = radii[f];
    let frames = normals(&points);
    let sample = |p: Point| trilinear(image, shape, p, 0.0);
    let offset = |i: usize, k: usize, distance: f64| -> Point {
        let (u, v) = frames[i];
        let d = crate::line::add(scale(u, angles[k].cos()), scale(v, angles[k].sin()));
        crate::line::add(points[i], scale(d, distance))
    };
    let ring = angles.len();
    let step = 0.5;
    let reach = 1.6 * r + 0.5 * step;
    let distances: Vec<f64> = (0..((reach / step).ceil() as usize))
        .map(|k| k as f64 * step)
        .collect();
    // Rays for the thickness: per sample, per ring direction, per distance.
    let rays: Vec<Vec<Vec<f64>>> = (0..n)
        .map(|i| {
            (0..ring)
                .map(|k| distances.iter().map(|&d| sample(offset(i, k, d))).collect())
                .collect()
        })
        .collect();
    let window = 2usize;
    let mut out: [Vec<f64>; 5] = Default::default();
    for i in 0..n {
        let mut core = vec![points[i]];
        core.extend((0..ring).map(|k| offset(i, k, 0.5 * r)));
        out[0].push(
            core.iter()
                .map(|&p| ((sample(p) - 0.5) / 0.4).clamp(0.0, 1.0))
                .sum::<f64>()
                / core.len() as f64,
        );
        let shared = core
            .iter()
            .filter(|&&p| segments.covered(p, f, 0.0))
            .count();
        out[2].push(1.0 - shared as f64 / core.len() as f64);
        let unexplained = (0..ring)
            .filter(|&k| {
                let p = offset(i, k, r + s.margin + 1.0);
                sample(p) > 0.5 && !segments.covered(p, f, s.margin + 0.5)
            })
            .count();
        out[1].push(1.0 - unexplained as f64 / ring as f64);
        let profile: Vec<f64> = (0..distances.len())
            .map(|d| {
                let mut pooled: Vec<f64> = (0..=2 * window)
                    .flat_map(|w| {
                        let row = (i + w).saturating_sub(window).min(n - 1);
                        rays[row].iter().map(move |ray| ray[d])
                    })
                    .collect();
                median(&mut pooled)
            })
            .collect();
        let width = half_radius(&profile, &distances, 0.8 * r) - s.thickness_margin;
        let deep = trilinear(depth, shape, points[i], 0.0) - s.margin;
        let ratio = (width / r - 1.0).abs().min((deep / r - 1.0).abs());
        out[3].push((-0.5 * (ratio / s.thickness_tolerance).powi(2)).exp());
        let stable = match previous {
            Some(previous) if previous[f].len() > 1 => {
                let moved = distance_to_polyline(points[i], &previous[f]);
                (-0.5 * (moved / (0.5 * r)).powi(2)).exp()
            }
            _ => 1.0,
        };
        out[4].push(stable);
    }
    (out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_radius_finds_the_falling_edge() {
        let distances = [0.0, 0.5, 1.0, 1.5, 2.0];
        let profile = [1.0, 1.0, 0.8, 0.2, 0.0];
        // Half of 1.0 is crossed between 1.0 (0.8) and 1.5 (0.2): 1.0 + 0.5 × 0.3 / 0.6.
        assert!((half_radius(&profile, &distances, 1.0) - 1.25).abs() < 1e-12);
        assert_eq!(half_radius(&[0.0; 5], &distances, 1.0), 0.0);
        assert_eq!(half_radius(&[1.0; 5], &distances, 1.0), 2.0);
    }

    #[test]
    fn a_lone_fiber_on_its_scan_is_sure() {
        let shape = [24, 24, 40];
        let r = 3.0;
        let mut image = vec![0.0f32; 24 * 24 * 40];
        let mut depth = vec![0.0f32; 24 * 24 * 40];
        for z in 0..24 {
            for y in 0..24 {
                let d = ((y as f64 + 0.5 - 12.0).powi(2) + (z as f64 + 0.5 - 12.0).powi(2)).sqrt();
                for x in 0..40 {
                    let index = (z * 24 + y) * 40 + x;
                    image[index] = (0.5 - (d - r) / 2.4).clamp(0.0, 1.0) as f32;
                    depth[index] = (r - d).max(0.0) as f32;
                }
            }
        }
        let line: Vec<Point> = (0..=20).map(|k| [10.0 + k as f64, 12.0, 12.0]).collect();
        let settings = ConfidenceSettings {
            spacing: 3.0,
            margin: 0.0,
            thickness_margin: 0.0,
            ring: 8,
            thickness_tolerance: 0.3,
        };
        let c = node_confidence(&image, &depth, shape, &[line], &[r], None, settings);
        let middle = c.per_node[0][10];
        assert!(middle > 0.5, "confidence {middle}");
        assert!(c.parts[2][0].iter().all(|&v| v == 1.0)); // nothing else to share with
    }
}
