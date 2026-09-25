//! A parallel-beam CT scanner's projector and back-projector
//! (`_scanner.acquire`).
//!
//! The sample turns about the z axis. At angle `θ` the detector axis is
//! `e = (cos θ, sin θ)` in the (x, y) plane and rays run along
//! `d = (−sin θ, cos θ)`, through the slice center; detector pixel `k` of a
//! row `width` wide sits at `u = k + 0.5 − width / 2` along `e`. Projection
//! is Joseph's method: step one voxel along whichever of x and y the ray
//! runs closer to, interpolate linearly across the other, and scale by the
//! step's length. Back-projection is pixel-driven with linear interpolation
//! along the detector row. Volumes are `(z, y, x)` and projections
//! `(angle, z, u)`, both C-ordered `f32`.

use crate::filter::parallel_rows;
use crate::{voxel_count, Shape};

/// Line integrals of `sample` (per voxel) at `angles` (radians) into rows
/// `width` pixels wide: an `(angles, nz, width)` array.
pub fn project(sample: &[f32], shape: Shape, angles: &[f64], width: usize) -> Vec<f32> {
    assert_eq!(sample.len(), voxel_count(shape));
    let [nz, ny, nx] = shape;
    let (cx, cy) = (nx as f64 / 2.0, ny as f64 / 2.0);
    let half = width as f64 / 2.0;
    let mut out = vec![0f32; angles.len() * nz * width];
    if out.is_empty() {
        return out;
    }
    parallel_rows(&mut out, width, |row, values| {
        let (a, z) = (row / nz, row % nz);
        let (sin, cos) = angles[a].sin_cos();
        let slice = &sample[z * ny * nx..(z + 1) * ny * nx];
        let (dx, dy) = (-sin, cos);
        for (k, value) in values.iter_mut().enumerate() {
            let u = k as f64 + 0.5 - half;
            // A point on the ray: the detector position through the center.
            let (px, py) = (cx + u * cos, cy + u * sin);
            let mut total = 0.0f64;
            if dy.abs() >= dx.abs() {
                // Step along y: one sample per row, interpolated along x.
                for j in 0..ny {
                    let t = (j as f64 + 0.5 - py) / dy;
                    let x = px + t * dx - 0.5;
                    total += along(&slice[j * nx..(j + 1) * nx], x);
                }
                total /= dy.abs();
            } else {
                for i in 0..nx {
                    let t = (i as f64 + 0.5 - px) / dx;
                    let y = py + t * dy - 0.5;
                    let f = y.floor();
                    let w = (y - f) as f32;
                    let j = f as i64;
                    let mut v = 0f32;
                    if j >= 0 && (j as usize) < ny {
                        v += (1.0 - w) * slice[j as usize * nx + i];
                    }
                    if j + 1 >= 0 && ((j + 1) as usize) < ny {
                        v += w * slice[(j + 1) as usize * nx + i];
                    }
                    total += v as f64;
                }
                total /= dx.abs();
            }
            *value = total as f32;
        }
    });
    out
}

/// Linear interpolation of `row` at index-space `x`, zero outside.
#[inline]
fn along(row: &[f32], x: f64) -> f64 {
    let f = x.floor();
    let w = x - f;
    let i = f as i64;
    let n = row.len() as i64;
    let mut v = 0.0;
    if i >= 0 && i < n {
        v += (1.0 - w) * row[i as usize] as f64;
    }
    if i + 1 >= 0 && i + 1 < n {
        v += w * row[(i + 1) as usize] as f64;
    }
    v
}

/// The sum over `angles` of each `(angles, nz, width)` row of `projections`
/// smeared back along its rays, over a `shape` volume: the back-projection
/// (unscaled; filtered back-projection multiplies by `π / angles`).
pub fn back_project(projections: &[f32], angles: &[f64], width: usize, shape: Shape) -> Vec<f32> {
    let [nz, ny, nx] = shape;
    assert_eq!(projections.len(), angles.len() * nz * width);
    let (cx, cy) = (nx as f64 / 2.0, ny as f64 / 2.0);
    let half = width as f64 / 2.0;
    let trig: Vec<(f64, f64)> = angles.iter().map(|a| a.sin_cos()).collect();
    let mut out = vec![0f32; voxel_count(shape)];
    if out.is_empty() {
        return out;
    }
    parallel_rows(&mut out, nx, |row, values| {
        let (z, j) = (row / ny, row % ny);
        let y = j as f64 + 0.5 - cy;
        for (i, value) in values.iter_mut().enumerate() {
            let x = i as f64 + 0.5 - cx;
            let mut total = 0.0f64;
            for (a, &(sin, cos)) in trig.iter().enumerate() {
                let u = x * cos + y * sin + half - 0.5;
                let start = (a * nz + z) * width;
                total += along(&projections[start..start + width], u);
            }
            *value = total as f32;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_centered_disk_projects_to_its_chord_at_every_angle() {
        let n = 32;
        let r = 8.0;
        let mut sample = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as f64 + 0.5 - 16.0, j as f64 + 0.5 - 16.0);
                if x * x + y * y <= r * r {
                    sample[j * n + i] = 1.0;
                }
            }
        }
        let angles = [0.0, 0.4, 1.1, 2.0];
        let width = 46;
        let p = project(&sample, [1, n, n], &angles, width);
        for row in p.chunks(width) {
            // Through the center the chord is the diameter; the total is the area.
            let middle = 0.5 * (row[width / 2 - 1] + row[width / 2]) as f64;
            assert!((middle - 2.0 * r).abs() < 1.0, "{middle}");
            let area: f64 = row.iter().map(|&v| v as f64).sum();
            assert!((area - std::f64::consts::PI * r * r).abs() < 6.0, "{area}");
        }
    }

    #[test]
    fn a_uniform_projection_back_projects_evenly() {
        let (angles, width, shape) = ([0.0, 0.7, 1.9], 20, [1, 10, 10]);
        let rows = vec![1f32; angles.len() * width];
        let volume = back_project(&rows, &angles, width, shape);
        assert!(volume.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    }
}
