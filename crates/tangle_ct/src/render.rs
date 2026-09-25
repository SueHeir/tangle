//! Soft capsule renderings of fibers over a voxel box, and their residual
//! against the scan, as `_moves.render_occupancy` and `_ends.local_residual`.
//!
//! A box is `[low, high)` in voxel indices `(x, y, z)`; its values are in
//! `(z, y, x)` order, like the scan.

use crate::line::Point;
use crate::raster::segment_distance;
use crate::{strides, Shape};

/// Voxel box corners `(x, y, z)`: `[low, high)`.
pub type Corner = [i64; 3];

/// Number of voxels in box `[low, high)` (0 when it is empty).
pub fn box_size(low: Corner, high: Corner) -> usize {
    (0..3).map(|a| (high[a] - low[a]).max(0) as usize).product()
}

/// The box around `points` grown by `margin`, clipped to the scan
/// (`_ends.local_box`).
pub fn local_box(points: &[Point], margin: f64, shape: Shape) -> (Corner, Corner) {
    let mut low = [0i64; 3];
    let mut high = [0i64; 3];
    for a in 0..3 {
        let min = points.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min);
        let max = points
            .iter()
            .map(|p| p[a])
            .fold(f64::NEG_INFINITY, f64::max);
        low[a] = ((min - margin).floor() as i64).max(0);
        high[a] = ((max + margin).ceil() as i64).min(shape[2 - a] as i64);
    }
    (low, high)
}

/// Indices of the lines (of at least 2 nodes, not in `skip`) whose bounding
/// boxes grown by two radii reach into `[low, high)` (`_ends.near_box`).
pub fn near_box(
    lines: &[Vec<Point>],
    radii: &[f64],
    low: Corner,
    high: Corner,
    skip: &[usize],
) -> Vec<usize> {
    (0..lines.len())
        .filter(|&k| !skip.contains(&k) && lines[k].len() >= 2)
        .filter(|&k| {
            (0..3).all(|a| {
                let min = lines[k].iter().map(|p| p[a]).fold(f64::INFINITY, f64::min);
                let max = lines[k]
                    .iter()
                    .map(|p| p[a])
                    .fold(f64::NEG_INFINITY, f64::max);
                min - 2.0 * radii[k] < high[a] as f64 && max + 2.0 * radii[k] > low[a] as f64
            })
        })
        .collect()
}

/// Soft union occupancy of the capsules of `lines` over box `[low, high)`:
/// per voxel the largest `clip(0.5 - (d - r) / (2 edge), 0, 1)` over the
/// segments, `d` the distance to the segment and `r` its line's radius
/// (`_moves.render_occupancy`).
pub fn render_occupancy(
    low: Corner,
    high: Corner,
    lines: &[&[Point]],
    radii: &[f64],
    edge: f64,
) -> Vec<f64> {
    let mut occupancy = vec![0.0; box_size(low, high)];
    if occupancy.is_empty() {
        return occupancy;
    }
    let size = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    for (line, &radius) in lines.iter().zip(radii) {
        let pad = radius + 2.0 * edge;
        let limit = radius + edge;
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut lo = [0i64; 3];
            let mut hi = [0i64; 3];
            for k in 0..3 {
                lo[k] = ((a[k].min(b[k]) - pad).floor() as i64).max(low[k]);
                hi[k] = ((a[k].max(b[k]) + pad).ceil() as i64).min(high[k]);
            }
            if (0..3).any(|k| hi[k] <= lo[k]) {
                continue;
            }
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        let p = [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5];
                        let distance = segment_distance(p, a, b);
                        if distance > limit {
                            continue;
                        }
                        let value = (0.5 - (distance - radius) / (2.0 * edge)).clamp(0.0, 1.0);
                        let index = (((z - low[2]) * size[1] + (y - low[1])) * size[0]
                            + (x - low[0])) as usize;
                        if value > occupancy[index] {
                            occupancy[index] = value;
                        }
                    }
                }
            }
        }
    }
    occupancy
}

/// The scan's values over box `[low, high)`, in `(z, y, x)` order.
pub fn crop(image: &[f32], shape: Shape, low: Corner, high: Corner) -> Vec<f32> {
    let s = strides(shape);
    let mut out = Vec::with_capacity(box_size(low, high));
    for z in low[2]..high[2] {
        for y in low[1]..high[1] {
            let row = z as usize * s[0] + y as usize * s[1];
            out.extend_from_slice(&image[row + low[0] as usize..row + high[0] as usize]);
        }
    }
    out
}

/// Sum of squared differences between the scan over the box and `rendered`.
pub fn squared_residual(observed: &[f32], rendered: &[f64]) -> f64 {
    observed
        .iter()
        .zip(rendered)
        .map(|(&o, &r)| {
            let d = o as f64 - r;
            d * d
        })
        .sum()
}

/// Squared residual of rendering `lines` (added to `base`, when given, by
/// max-union) against the scan over box `[low, high)` (`_ends.local_residual`).
#[allow(clippy::too_many_arguments)]
pub fn local_residual(
    image: &[f32],
    shape: Shape,
    low: Corner,
    high: Corner,
    lines: &[&[Point]],
    radii: &[f64],
    base: Option<&[f64]>,
    edge: f64,
) -> f64 {
    if (0..3).any(|a| high[a] <= low[a]) {
        return 0.0;
    }
    let mut rendered = render_occupancy(low, high, lines, radii, edge);
    if let Some(base) = base {
        for (r, &b) in rendered.iter_mut().zip(base) {
            *r = r.max(b);
        }
    }
    squared_residual(&crop(image, shape, low, high), &rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_capsule_is_full_inside_and_fades_over_the_edge() {
        let line = [[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]];
        let occupancy = render_occupancy([0, 0, 0], [20, 10, 10], &[&line], &[2.0], 1.2);
        let at = |x: usize, y: usize, z: usize| occupancy[(z * 10 + y) * 20 + x];
        assert_eq!(at(9, 4, 4), 1.0); // center (9.5, 4.5, 4.5): 0.71 from the axis
        assert_eq!(at(9, 8, 4), 0.0); // 3.54 from the axis, past r + edge
        let v = at(9, 6, 4); // (9.5, 6.5, 4.5): 1.58 from the axis
        assert!((v - (0.5 - (1.5811388300841898 - 2.0) / 2.4)).abs() < 1e-12);
    }

    #[test]
    fn the_residual_of_a_perfect_rendering_is_zero() {
        let shape = [10, 10, 20];
        let line = [[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]];
        let rendered = render_occupancy([0, 0, 0], [20, 10, 10], &[&line], &[2.0], 1.2);
        let image: Vec<f32> = rendered.iter().map(|&v| v as f32).collect();
        let residual = local_residual(
            &image,
            shape,
            [0, 0, 0],
            [20, 10, 10],
            &[&line],
            &[2.0],
            None,
            1.2,
        );
        assert!(residual < 1e-10);
    }
}
