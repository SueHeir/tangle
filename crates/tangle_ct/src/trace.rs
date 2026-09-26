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
use crate::line::{add, cross, dot, norm, polyline_length, resample, scale, sub, Point};
use crate::raster::paint;
use crate::sample::{trilinear, voxel_of};
use crate::{strides, Shape};

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
    pub fn recenter(
        &self,
        center: Point,
        direction: Point,
        claimed: &[i32],
        own_label: i32,
    ) -> Point {
        let points = self.plane(center, direction, &self.window);
        let s = strides(self.shape);
        let spread = 2.0 * (0.8 * self.radius) * (0.8 * self.radius);
        let mut total = 0.0;
        let mut shift = [0.0; 3];
        let mut weights = Vec::with_capacity(points.len());
        for &p in &points {
            let mut weight = self.sample(p).max(0.0);
            let owner =
                voxel_of(self.shape, p).map_or(0, |[k, j, i]| claimed[k * s[0] + j * s[1] + i]);
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
        .filter(|&v| edt[v] >= depth && edt[v] >= peak[v] && exclude.is_none_or(|e| e[v] == 0))
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

/// Voxel-center points where the image is fiber (at least 0.5), the tube
/// strength (`HessianField::tube_strength`) is at least `min_strength` and
/// no lower than at any of the 26 neighbours, and `exclude` is 0; strongest
/// first, ties in raster order.
pub fn bright_seeds(
    image: &[f32],
    strength: &[f32],
    shape: Shape,
    exclude: &[i32],
    min_strength: f32,
) -> Vec<Point> {
    let s = strides(shape);
    let (nz, ny, nx) = (shape[0] as isize, shape[1] as isize, shape[2] as isize);
    let mut seeds: Vec<(usize, f32)> = Vec::new();
    for v in 0..image.len() {
        let value = strength[v];
        if value < min_strength || image[v] < 0.5 || exclude[v] != 0 {
            continue;
        }
        let (k, j, i) = (
            (v / s[0]) as isize,
            ((v / s[1]) % shape[1]) as isize,
            (v % shape[2]) as isize,
        );
        let mut peak = true;
        'around: for dk in -1..=1isize {
            for dj in -1..=1isize {
                for di in -1..=1isize {
                    let (kk, jj, ii) = (k + dk, j + dj, i + di);
                    if (dk, dj, di) == (0, 0, 0)
                        || kk < 0
                        || jj < 0
                        || ii < 0
                        || kk >= nz
                        || jj >= ny
                        || ii >= nx
                    {
                        continue;
                    }
                    if strength[kk as usize * s[0] + jj as usize * s[1] + ii as usize] > value {
                        peak = false;
                        break 'around;
                    }
                }
            }
        }
        if peak {
            seeds.push((v, value));
        }
    }
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
    /// After the depth seeds, also seed on `bright_seeds` of the Hessian's
    /// tube strength at least this high (None: depth seeds only). Packed
    /// fibers share one deep foreground ridge, so only the bundle's middle
    /// fiber gets a depth seed; each of them is a tube-strength peak.
    pub bright_seed_strength: Option<f32>,
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
    let follow = |seed: Point, claimed: &mut [i32], fibers: &mut Vec<Vec<Point>>| -> bool {
        let [k, j, i] = voxel_of(shape, seed).expect("seeds are voxel centers");
        if claimed[k * s[0] + j * s[1] + i] != 0 {
            return false;
        }
        let line = drop_claimed(&tracer.trace(seed, max_steps, claimed), claimed, shape, 0.3);
        let label = search.label_offset + fibers.len() as i32 + 1;
        if line.len() >= 2 && polyline_length(&line) >= search.min_length {
            let line = resample(&line, search.node_spacing);
            paint(claimed, shape, &line, 1.1 * radius, label, false);
            fibers.push(line);
            search.max_fibers.is_some_and(|m| fibers.len() >= m)
        } else {
            // Mark the rejected trace (or the seed) so nearby seeds on the
            // same blob are not traced again.
            let rejected = if line.is_empty() { vec![seed] } else { line };
            paint(claimed, shape, &rejected, 0.75 * radius, -1, true);
            false
        }
    };
    for seed in seeds {
        if follow(seed, claimed, &mut fibers) {
            return fibers;
        }
    }
    if let Some(min_strength) = search.bright_seed_strength {
        let strength = hessian.tube_strength();
        for seed in bright_seeds(image, &strength, shape, claimed, min_strength) {
            if follow(seed, claimed, &mut fibers) {
                return fibers;
            }
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
    fn traces_a_straight_bright_rod_end_to_end() {
        let shape = [24, 24, 64];
        let mut rod = vec![0i32; shape[0] * shape[1] * shape[2]];
        paint(
            &mut rod,
            shape,
            &[[6.0, 12.0, 12.0], [58.0, 12.0, 12.0]],
            3.0,
            1,
            false,
        );
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
            bright_seed_strength: None,
        };
        let fibers = trace_fibers(&image, shape, &hessian, &mut claimed, &edt, &peak, search);
        assert_eq!(fibers.len(), 1, "{fibers:?}");
        let xs: Vec<f64> = fibers[0].iter().map(|p| p[0]).collect();
        let (low, high) = (
            xs.iter().cloned().fold(f64::MAX, f64::min),
            xs.iter().cloned().fold(f64::MIN, f64::max),
        );
        assert!(low < 12.0 && high > 52.0, "{low} {high}");
        assert!(fibers[0]
            .iter()
            .all(|p| (p[1] - 12.0).abs() < 1.0 && (p[2] - 12.0).abs() < 1.0));
    }

    #[test]
    fn bright_seeds_find_every_fiber_of_a_packed_bundle() {
        // Seven rods of radius 3 along x, hexagonally packed, at grey 1 over
        // a 0.7 fill across the bundle: one foreground blob, one deep ridge.
        let shape = [40, 40, 64];
        let (r, pitch, center) = (3.0, 6.2, 20.0);
        let mut axes = vec![[center, center]];
        for k in 0..6 {
            let a = k as f64 * std::f64::consts::PI / 3.0;
            axes.push([center + pitch * a.cos(), center + pitch * a.sin()]);
        }
        let mut image = vec![0.0f32; shape[0] * shape[1] * shape[2]];
        let s = strides(shape);
        for k in 0..shape[0] {
            for j in 0..shape[1] {
                for i in 6..58 {
                    let (y, z) = (j as f64 + 0.5, k as f64 + 0.5);
                    let hull = ((y - center).powi(2) + (z - center).powi(2)).sqrt() < pitch + r;
                    let rod = axes
                        .iter()
                        .any(|[ay, az]| ((y - ay).powi(2) + (z - az).powi(2)).sqrt() < r);
                    image[k * s[0] + j * s[1] + i] = if rod {
                        1.0
                    } else if hull {
                        0.7
                    } else {
                        0.0
                    };
                }
            }
        }
        let hessian = HessianField::new(&image, shape, 1.8);
        let foreground: Vec<bool> = image.iter().map(|&v| v > 0.5).collect();
        let edt = crate::edt::distance_transform(&foreground, shape);
        let peak = crate::filter::maximum_filter3(&edt, shape);
        let run = |bright: Option<f32>| {
            let mut claimed = vec![0i32; image.len()];
            let search = FiberSearch {
                trace: TraceSettings {
                    radius: r,
                    min_bend_radius: 30.0,
                    step: 1.5,
                },
                min_length: 20.0,
                node_spacing: 3.0,
                label_offset: 0,
                max_fibers: None,
                seed_depth_radii: 0.5,
                bright_seed_strength: bright,
            };
            trace_fibers(&image, shape, &hessian, &mut claimed, &edt, &peak, search)
        };
        let found = |fibers: &[Vec<Point>]| {
            axes.iter()
                .filter(|[ay, az]| {
                    fibers.iter().any(|line| {
                        let middle = line[line.len() / 2];
                        (middle[1] - ay).abs() < 1.5 && (middle[2] - az).abs() < 1.5
                    })
                })
                .count()
        };
        assert!(found(&run(None)) < 7);
        let fibers = run(Some(0.02));
        assert_eq!(found(&fibers), 7, "{} fibers", fibers.len());
    }
}
