//! Fibers drawn as the grey the scan should show, from a measured grey
//! profile per fiber (`_grey.render_grey`), and their misfit to the scan.
//!
//! Each voxel takes the profile of the fiber whose capsule surface is
//! nearest, at its distance from that fiber's axis; outside the surface the
//! grey fades from the profile's last value to the void grey over
//! `2 × edge` voxels. Boxes are `[low, high)` in `(x, y, z)`, values in
//! `(z, y, x)` order.

use crate::line::Point;
use crate::raster::segment_distance;
use crate::render::{box_size, Corner};
use crate::{strides, Shape};

/// `np.interp(x, linspace(0, 1, len(profile)), profile)`.
pub fn profile_at(profile: &[f64], x: f64) -> f64 {
    let n = profile.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || x <= 0.0 {
        return profile[0];
    }
    if x >= 1.0 {
        return profile[n - 1];
    }
    let step = 1.0 / (n - 1) as f64;
    let j = ((x / step).floor() as usize).min(n - 2);
    let x0 = j as f64 * step;
    let x1 = if j + 2 == n {
        1.0
    } else {
        (j + 1) as f64 * step
    };
    let slope = (profile[j + 1] - profile[j]) / (x1 - x0);
    slope * (x - x0) + profile[j]
}

/// The grey `lines` should show over box `[low, high)` (`_grey.render_grey`):
/// `profiles[f]` is line `f`'s profile, `void` the background grey.
pub fn render_grey(
    low: Corner,
    high: Corner,
    lines: &[&[Point]],
    radii: &[f64],
    profiles: &[&[f64]],
    void: f64,
    edge: f64,
) -> Vec<f64> {
    let size = box_size(low, high);
    let mut value = vec![void; size];
    if size == 0 || lines.is_empty() {
        return value;
    }
    let dims = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    // Nearest surface per voxel; segments in order with a strict `<`, so ties
    // go to the lower segment (as `_geometry.nearest_segments`).
    let mut surface = vec![f64::INFINITY; size];
    let mut owner = vec![usize::MAX; size];
    for (f, (line, &radius)) in lines.iter().zip(radii).enumerate() {
        let reach = radius + 2.0 * edge;
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut lo = [0i64; 3];
            let mut hi = [0i64; 3];
            for k in 0..3 {
                lo[k] = ((a[k].min(b[k]) - reach).floor() as i64).max(low[k]);
                hi[k] = ((a[k].max(b[k]) + reach).ceil() as i64).min(high[k]);
            }
            if (0..3).any(|k| hi[k] <= lo[k]) {
                continue;
            }
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        let p = [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5];
                        let d = segment_distance(p, a, b);
                        if d > reach {
                            continue;
                        }
                        let index = (((z - low[2]) * dims[1] + (y - low[1])) * dims[0]
                            + (x - low[0])) as usize;
                        let s = d - radius;
                        if s < surface[index] {
                            surface[index] = s;
                            owner[index] = f;
                        }
                    }
                }
            }
        }
    }
    for index in 0..size {
        let f = owner[index];
        if f == usize::MAX || surface[index] >= 2.0 * edge {
            continue;
        }
        let radius = radii[f];
        let fraction = ((surface[index] + radius) / radius.max(1e-6)).min(1.0);
        let inside = profile_at(profiles[f], fraction);
        let weight = (1.0 - surface[index] / (2.0 * edge)).clamp(0.0, 1.0);
        value[index] = void + weight * (inside - void);
    }
    value
}

/// Per voxel of box `[low, high)`, `(scan − drawn)²`.
pub fn misfit(grey: &[f32], shape: Shape, low: Corner, high: Corner, drawn: &[f64]) -> Vec<f64> {
    let s = strides(shape);
    let mut out = Vec::with_capacity(drawn.len());
    let mut index = 0;
    for z in low[2]..high[2] {
        for y in low[1]..high[1] {
            let row = z as usize * s[0] + y as usize * s[1];
            for x in low[0]..high[0] {
                let d = grey[row + x as usize] as f64 - drawn[index];
                out.push(d * d);
                index += 1;
            }
        }
    }
    out
}

/// Per voxel of box `[low, high)`, how many fibers beyond the first have
/// it inside their capsule (distance to the axis at most the radius): the
/// voxels where fits overlap.
pub fn overlap(low: Corner, high: Corner, lines: &[&[Point]], radii: &[f64]) -> Vec<u16> {
    let size = box_size(low, high);
    let mut count = vec![0u16; size];
    if size == 0 {
        return count;
    }
    let dims = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    // The last line that counted each voxel, so a line counts it once.
    let mut last = vec![usize::MAX; size];
    for (f, (line, &radius)) in lines.iter().zip(radii).enumerate() {
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut lo = [0i64; 3];
            let mut hi = [0i64; 3];
            for k in 0..3 {
                lo[k] = ((a[k].min(b[k]) - radius).floor() as i64).max(low[k]);
                hi[k] = ((a[k].max(b[k]) + radius).ceil() as i64).min(high[k]);
            }
            if (0..3).any(|k| hi[k] <= lo[k]) {
                continue;
            }
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        let index = (((z - low[2]) * dims[1] + (y - low[1])) * dims[0]
                            + (x - low[0])) as usize;
                        if last[index] == f {
                            continue;
                        }
                        let p = [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5];
                        if segment_distance(p, a, b) <= radius {
                            if last[index] != usize::MAX {
                                count[index] = count[index].saturating_add(1);
                            }
                            last[index] = f;
                        }
                    }
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_interpolates_like_numpy() {
        let p = [1.0, 0.5, 0.0];
        assert_eq!(profile_at(&p, 0.0), 1.0);
        assert!((profile_at(&p, 0.25) - 0.75).abs() < 1e-12);
        assert!((profile_at(&p, 0.75) - 0.25).abs() < 1e-12);
        assert_eq!(profile_at(&p, 1.0), 0.0);
    }

    #[test]
    fn a_fiber_shows_its_profile_and_fades_to_void() {
        let line = [[2.0, 5.0, 5.0], [18.0, 5.0, 5.0]];
        let profile = [1.0, 1.0, 0.6];
        let drawn = render_grey(
            [0, 0, 0],
            [20, 10, 10],
            &[&line],
            &[3.0],
            &[&profile],
            0.1,
            1.2,
        );
        let at = |x: usize, y: usize, z: usize| drawn[(z * 10 + y) * 20 + x];
        // (9.5, 4.5, 4.5): 0.71 from the axis, fraction 0.24 of the radius.
        assert!((at(9, 4, 4) - 1.0).abs() < 1e-12);
        // (9.5, 9.5, 4.5): 4.53 from the axis, past the surface by 1.53 of 2.4.
        let surface = (4.5f64 * 4.5 + 0.5 * 0.5).sqrt() - 3.0;
        let want = 0.1 + (1.0 - surface / 2.4) * (0.6 - 0.1);
        assert!((at(9, 9, 4) - want).abs() < 1e-12);
        assert_eq!(at(9, 0, 9), 0.1);
    }

    #[test]
    fn crossing_fibers_overlap_where_both_capsules_reach() {
        let a = [[0.0, 5.5, 5.5], [11.0, 5.5, 5.5]];
        let b = [[5.5, 0.0, 5.5], [5.5, 11.0, 5.5]];
        let count = overlap(
            [0, 0, 0],
            [11, 11, 11],
            &[&a, &b, &a[..1]],
            &[2.0, 2.0, 2.0],
        );
        let at = |x: usize, y: usize, z: usize| count[(z * 11 + y) * 11 + x];
        assert_eq!(at(5, 5, 5), 1);
        assert_eq!(at(1, 5, 5), 0); // inside a only
        assert_eq!(count.iter().filter(|&&c| c > 1).count(), 0);
    }
}
