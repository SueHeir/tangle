//! Initial fiber centerlines: seed on the distance-transform ridge, then trace.
//!
//! Each trace steps along the local tube axis (Hessian eigenvector blended
//! with momentum), re-centers in the cross-sectional plane on the image
//! intensity, limits the turn per step by the admissible bend radius, and
//! stops when the core intensity drops below the fiber/void midpoint. Voxels
//! claimed by earlier traces are down-weighted, so a trace keeps its
//! direction through a crossing instead of turning onto the other fiber.
//!
//! This follows `tangle/ct/_trace.py` step for step (same sums in the same
//! order), so traces match it up to floating-point rounding.

use crate::hessian::HessianField;
use crate::raster::paint;
use crate::sample::{trilinear, voxel_of};
use crate::{strides, Shape};

type Point = [f64; 3];

#[inline]
fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale(a: Point, s: f64) -> Point {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Two unit vectors perpendicular to `direction` and to each other.
fn perpendicular_basis(direction: Point) -> (Point, Point) {
    let helper = if direction[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = cross(direction, helper);
    let e1 = scale(e1, 1.0 / dot(e1, e1).sqrt());
    (e1, cross(direction, e1))
}

/// Offsets `(u, v)` on a `spacing` grid within `radius` of the origin, in
/// the order `np.meshgrid` and a boolean mask give them (v outer, u inner).
pub fn disk_offsets(radius: f64, spacing: f64) -> Vec<[f64; 2]> {
    // np.arange(-radius, radius + 1e-9, spacing)
    let count = ((2.0 * radius + 1e-9) / spacing).ceil().max(0.0) as usize;
    let ticks: Vec<f64> = (0..count).map(|i| -radius + i as f64 * spacing).collect();
    let mut out = Vec::new();
    for &v in &ticks {
        for &u in &ticks {
            if u * u + v * v <= radius * radius {
                out.push([u, v]);
            }
        }
    }
    out
}

/// Arc length of a polyline.
pub fn polyline_length(points: &[Point]) -> f64 {
    points.windows(2).map(|p| norm(sub(p[1], p[0]))).sum()
}

/// A polyline resampled to (nearly) uniform arc-length `spacing`, as
/// `_geometry.resample` (`np.linspace` targets, `np.interp` per axis).
pub fn resample(points: &[Point], spacing: f64) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut kept = vec![points[0]];
    for pair in points.windows(2) {
        if norm(sub(pair[1], pair[0])) > 1e-9 {
            kept.push(pair[1]);
        }
    }
    if kept.len() < 2 {
        return kept;
    }
    let mut arc = vec![0.0];
    for pair in kept.windows(2) {
        let last = *arc.last().unwrap();
        arc.push(last + norm(sub(pair[1], pair[0])));
    }
    let total = *arc.last().unwrap();
    let count = ((total / spacing).round_ties_even() as usize).max(1) + 1;
    let step = total / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    let mut j = 0;
    for t in 0..count {
        let target = if t + 1 == count { total } else { t as f64 * step };
        if target >= total {
            out.push(*kept.last().unwrap());
            continue;
        }
        while j + 2 < arc.len() && arc[j + 1] <= target {
            j += 1;
        }
        let mut point = [0.0; 3];
        for (axis, value) in point.iter_mut().enumerate() {
            let slope = (kept[j + 1][axis] - kept[j][axis]) / (arc[j + 1] - arc[j]);
            *value = slope * (target - arc[j]) + kept[j][axis];
        }
        out.push(point);
    }
    out
}

/// The tracer's settings for one fiber type.
#[derive(Clone, Copy, Debug)]
pub struct TraceSettings {
    pub radius: f64,
    pub min_bend_radius: f64,
    pub step: f64,
}

/// Traces fibers through a normalized image (void 0, fiber 1).
pub struct Tracer<'a> {
    image: &'a [f32],
    shape: Shape,
    hessian: &'a HessianField,
    radius: f64,
    pub step: f64,
    max_turn: f64,
    window: Vec<[f64; 2]>,
    core: Vec<[f64; 2]>,
    upper: Point,
}

impl<'a> Tracer<'a> {
    pub fn new(
        image: &'a [f32],
        shape: Shape,
        hessian: &'a HessianField,
        settings: TraceSettings,
    ) -> Self {
        let TraceSettings {
            radius,
            min_bend_radius,
            step,
        } = settings;
        Self {
            image,
            shape,
            hessian,
            radius,
            step,
            max_turn: (step / min_bend_radius.max(1e-9)).min(std::f64::consts::FRAC_PI_4),
            window: disk_offsets(1.4 * radius, 0.5),
            core: disk_offsets((0.5 * radius).max(0.75), 0.5),
            upper: [shape[2] as f64, shape[1] as f64, shape[0] as f64],
        }
    }

    fn inside(&self, point: Point) -> bool {
        (0..3).all(|a| point[a] >= 0.0 && point[a] <= self.upper[a])
    }

    fn plane(&self, center: Point, direction: Point, offsets: &[[f64; 2]]) -> Vec<Point> {
        let (e1, e2) = perpendicular_basis(direction);
        offsets
            .iter()
            .map(|[u, v]| add(add(center, scale(e1, *u)), scale(e2, *v)))
            .collect()
    }

    fn sample(&self, point: Point) -> f64 {
        trilinear(self.image, self.shape, point, 0.0)
    }

    /// Mean image value over the core disk across `direction` at `center`.
    pub fn core_intensity(&self, center: Point, direction: Point) -> f64 {
        let points = self.plane(center, direction, &self.core);
        points.iter().map(|&p| self.sample(p)).sum::<f64>() / points.len() as f64
    }

    /// `center` moved (at most 0.4 radii) to the intensity-weighted centroid
    /// of the cross-section, with voxels of other fibers down-weighted.
    pub fn recenter(&self, center: Point, direction: Point, claimed: &[i32], own_label: i32) -> Point {
        let points = self.plane(center, direction, &self.window);
        let s = strides(self.shape);
        let spread = 2.0 * (0.8 * self.radius) * (0.8 * self.radius);
        let mut total = 0.0;
        let mut shift = [0.0; 3];
        let mut weights = Vec::with_capacity(points.len());
        for &p in &points {
            let mut weight = self.sample(p).max(0.0);
            let owner = voxel_of(self.shape, p).map_or(0, |[k, j, i]| claimed[k * s[0] + j * s[1] + i]);
            if owner > 0 && owner != own_label {
                weight *= 0.15;
            }
            let offset = sub(p, center);
            weight *= (-dot(offset, offset) / spread).exp();
            weights.push(weight);
            total += weight;
        }
        if total <= 1e-9 {
            return center;
        }
        for (&p, &weight) in points.iter().zip(&weights) {
            shift = add(shift, scale(sub(p, center), weight));
        }
        shift = scale(shift, 1.0 / total);
        let limit = 0.4 * self.radius;
        let length = norm(shift);
        if length > limit {
            shift = scale(shift, limit / length);
        }
        add(center, shift)
    }

    fn turn(&self, current: Point, proposed: Point) -> Point {
        let cosine = dot(current, proposed).clamp(-1.0, 1.0);
        let angle = cosine.acos();
        if angle <= self.max_turn || angle < 1e-9 {
            return proposed;
        }
        let axis = sub(proposed, scale(current, cosine));
        let axis = scale(axis, 1.0 / norm(axis));
        add(
            scale(current, self.max_turn.cos()),
            scale(axis, self.max_turn.sin()),
        )
    }

    /// Steps from `start` along `direction` until the core leaves the fiber.
    /// Voxels of `claimed` labelled `own_label` are not down-weighted (the
    /// fiber being extended).
    pub fn trace_one_way(
        &self,
        start: Point,
        direction: Point,
        max_steps: usize,
        claimed: &[i32],
        own_label: i32,
    ) -> Vec<Point> {
        let mut points = Vec::new();
        let mut position = start;
        let mut direction = direction;
        let mut misses = 0;
        for _ in 0..max_steps {
            let candidate = add(position, scale(direction, self.step));
            if !self.inside(candidate) {
                break;
            }
            let candidate = self.recenter(candidate, direction, claimed, own_label);
            let (mut axis, tubularity) = self.hessian.direction(candidate);
            if dot(axis, direction) < 0.0 {
                axis = scale(axis, -1.0);
            }
            let moved = sub(candidate, position);
            let moved_length = norm(moved);
            let moved = if moved_length > 1e-9 {
                scale(moved, 1.0 / moved_length)
            } else {
                direction
            };
            // Trust the Hessian only where the image looks like a single tube.
            let hessian_weight = if tubularity > 0.05 { 0.35 } else { 0.0 };
            let proposed = add(
                add(scale(direction, 0.5), scale(moved, 0.15)),
                scale(axis, hessian_weight),
            );
            let proposed = scale(proposed, 1.0 / norm(proposed));
            direction = self.turn(direction, proposed);
            if self.core_intensity(candidate, direction) < 0.5 {
                misses += 1;
                if misses >= 2 {
                    break;
                }
            } else {
                misses = 0;
            }
            position = candidate;
            points.push(candidate);
        }
        // Drop trailing unsupported points.
        while let Some(&last) = points.last() {
            if self.core_intensity(last, direction) >= 0.5 {
                break;
            }
            points.pop();
        }
        points
    }

    /// A trace both ways from `seed` along the Hessian axis there.
    pub fn trace(&self, seed: Point, max_steps: usize, claimed: &[i32]) -> Vec<Point> {
        let (axis, _) = self.hessian.direction(seed);
        let direction = scale(axis, 1.0 / norm(axis));
        let center = self.recenter(seed, direction, claimed, 0);
        let forward = self.trace_one_way(center, direction, max_steps, claimed, 0);
        let backward = self.trace_one_way(center, scale(direction, -1.0), max_steps, claimed, 0);
        let mut line: Vec<Point> = backward.into_iter().rev().collect();
        line.push(center);
        line.extend(forward);
        line
    }
}

/// `line` with its end runs over already claimed voxels trimmed; empty when
/// more than `max_covered` of what is left is claimed too. (A new trace can
/// start in a gap between fits and then follow an existing fiber.)
pub fn drop_claimed(line: &[Point], claimed: &[i32], shape: Shape, max_covered: f64) -> Vec<Point> {
    if line.is_empty() {
        return Vec::new();
    }
    let s = strides(shape);
    let covered: Vec<bool> = line
        .iter()
        .map(|p| {
            let mut index = [0usize; 3];
            for axis in 0..3 {
                let c = p[2 - axis].floor().clamp(0.0, (shape[axis] - 1) as f64);
                index[axis] = c as usize;
            }
            claimed[index[0] * s[0] + index[1] * s[1] + index[2]] > 0
        })
        .collect();
    let (mut start, mut stop) = (0, line.len());
    while start < stop && covered[start] {
        start += 1;
    }
    while stop > start && covered[stop - 1] {
        stop -= 1;
    }
    let kept = &covered[start..stop];
    if kept.is_empty() {
        return Vec::new();
    }
    let fraction = kept.iter().filter(|&&c| c).count() as f64 / kept.len() as f64;
    if fraction > max_covered {
        return Vec::new();
    }
    line[start..stop].to_vec()
}

/// Voxel-center points of distance-transform ridge voxels at least
/// `min_depth_radii` radii deep (and not `exclude`d), deepest first; ties
/// keep raster order.
pub fn ridge_seeds(
    edt: &[f32],
    peak: &[f32],
    shape: Shape,
    radius: f64,
    exclude: Option<&[i32]>,
    min_depth_radii: f64,
) -> Vec<Point> {
    // NumPy compares the float32 depths with the threshold cast to float32.
    let depth = (min_depth_radii * radius).max(1.0) as f32;
    let s = strides(shape);
    let mut seeds: Vec<(usize, f32)> = (0..edt.len())
        .filter(|&v| {
            edt[v] >= depth && edt[v] >= peak[v] && exclude.is_none_or(|e| e[v] == 0)
        })
        .map(|v| (v, edt[v]))
        .collect();
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1)); // stable
    seeds
        .into_iter()
        .map(|(v, _)| {
            let (k, j, i) = (v / s[0], (v / s[1]) % shape[1], v % shape[2]);
            [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5]
        })
        .collect()
}

/// What `trace_fibers` needs besides the image and Hessian.
#[derive(Clone, Copy, Debug)]
pub struct FiberSearch {
    pub trace: TraceSettings,
    pub min_length: f64,
    pub node_spacing: f64,
    pub label_offset: i32,
    pub max_fibers: Option<usize>,
    pub seed_depth_radii: f64,
}

/// Traces fibers from ridge seeds not yet explained by `claimed`, which is
/// painted as fibers are found (label `label_offset + n` for the n-th, and
/// -1 around rejected traces). Seeds must be at least `seed_depth_radii`
/// radii from the void (by `edt`, with `peak` its 3×3×3 maximum).
pub fn trace_fibers(
    image: &[f32],
    shape: Shape,
    hessian: &HessianField,
    claimed: &mut [i32],
    edt: &[f32],
    peak: &[f32],
    search: FiberSearch,
) -> Vec<Vec<Point>> {
    let tracer = Tracer::new(image, shape, hessian, search.trace);
    let radius = search.trace.radius;
    let max_steps = (4.0 * (shape[0] + shape[1] + shape[2]) as f64 / tracer.step) as usize;
    let s = strides(shape);
    let seeds = ridge_seeds(
        edt,
        peak,
        shape,
        radius,
        Some(&*claimed),
        search.seed_depth_radii,
    );
    let mut fibers: Vec<Vec<Point>> = Vec::new();
    for seed in seeds {
        let [k, j, i] = voxel_of(shape, seed).expect("seeds are voxel centers");
        if claimed[k * s[0] + j * s[1] + i] != 0 {
            continue;
        }
        let line = drop_claimed(&tracer.trace(seed, max_steps, claimed), claimed, shape, 0.3);
        let label = search.label_offset + fibers.len() as i32 + 1;
        if line.len() >= 2 && polyline_length(&line) >= search.min_length {
            let line = resample(&line, search.node_spacing);
            paint(claimed, shape, &line, 1.1 * radius, label, false);
            fibers.push(line);
            if search.max_fibers.is_some_and(|m| fibers.len() >= m) {
                break;
            }
        } else {
            // Mark the rejected trace (or the seed) so nearby seeds on the
            // same blob are not traced again.
            let rejected = if line.is_empty() { vec![seed] } else { line };
            paint(claimed, shape, &rejected, 0.75 * radius, -1, true);
        }
    }
    fibers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_offsets_match_numpy() {
        // _disk_offsets(1.0): ticks -1, -0.5, 0, 0.5, 1 → 13 points within radius 1
        let offsets = disk_offsets(1.0, 0.5);
        assert_eq!(offsets.len(), 13);
        assert_eq!(offsets[0], [0.0, -1.0]);
        assert_eq!(offsets[12], [0.0, 1.0]);
    }

    #[test]
    fn resample_spaces_nodes_evenly() {
        let line = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let out = resample(&line, 2.5);
        assert_eq!(out.len(), 5);
        assert_eq!(out[1], [2.5, 0.0, 0.0]);
        assert_eq!(out[4], [10.0, 0.0, 0.0]);
        // round half to even, as numpy: 5 / 2 = 2.5 → 2 segments
        assert_eq!(resample(&[[0.0; 3], [5.0, 0.0, 0.0]], 2.0).len(), 3);
    }

    #[test]
    fn traces_a_straight_bright_rod_end_to_end() {
        let shape = [24, 24, 64];
        let mut rod = vec![0i32; shape[0] * shape[1] * shape[2]];
        paint(&mut rod, shape, &[[6.0, 12.0, 12.0], [58.0, 12.0, 12.0]], 3.0, 1, false);
        let image: Vec<f32> = rod.iter().map(|&v| v as f32).collect();
        let hessian = HessianField::new(&image, shape, 1.8);
        let foreground: Vec<bool> = image.iter().map(|&v| v > 0.5).collect();
        let edt = crate::edt::distance_transform(&foreground, shape);
        let peak = crate::filter::maximum_filter3(&edt, shape);
        let mut claimed = vec![0i32; rod.len()];
        let search = FiberSearch {
            trace: TraceSettings {
                radius: 3.0,
                min_bend_radius: 30.0,
                step: 1.5,
            },
            min_length: 10.0,
            node_spacing: 3.0,
            label_offset: 0,
            max_fibers: None,
            seed_depth_radii: 0.5,
        };
        let fibers = trace_fibers(&image, shape, &hessian, &mut claimed, &edt, &peak, search);
        assert_eq!(fibers.len(), 1, "{fibers:?}");
        let xs: Vec<f64> = fibers[0].iter().map(|p| p[0]).collect();
        let (low, high) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
        assert!(low < 12.0 && high > 52.0, "{low} {high}");
        assert!(fibers[0].iter().all(|p| (p[1] - 12.0).abs() < 1.0 && (p[2] - 12.0).abs() < 1.0));
    }
}
