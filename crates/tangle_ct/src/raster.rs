//! Drawing fitted fibers (capsules around centerline polylines) into voxels.

use crate::{strides, voxel_count, Shape};

/// Distance from `p` to segment `ab` (to `a` when the segment is a point).
#[inline]
pub fn segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let denominator = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if denominator <= 1e-12 {
        0.0
    } else {
        ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / denominator).clamp(0.0, 1.0)
    };
    let d = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Voxel index box `[low, high)` (`x, y, z`) around segment `ab` grown by `pad`, clipped to the volume.
fn segment_box(shape: Shape, a: [f64; 3], b: [f64; 3], pad: f64) -> Option<([usize; 3], [usize; 3])> {
    let mut low = [0usize; 3];
    let mut high = [0usize; 3];
    for axis in 0..3 {
        let upper = shape[2 - axis] as f64;
        let lo = (a[axis].min(b[axis]) - pad).floor().max(0.0);
        let hi = (a[axis].max(b[axis]) + pad).ceil().min(upper);
        if !(hi > lo) {
            return None;
        }
        low[axis] = lo as usize;
        high[axis] = hi as usize;
    }
    Some((low, high))
}

/// Nearest-fiber ownership of every voxel within `reach[f]` of fiber `f`'s
/// centerline, as `tangle.ct._geometry.rasterize`.
pub struct Raster {
    /// One-based fiber of each voxel, 0 where none reaches.
    pub labels: Vec<i32>,
    /// The owning distance (minus the owner's radius when signed); +∞ where none.
    pub distance: Vec<f32>,
    /// Global segment index of the owner (segment `s` of fiber `f` is
    /// `offsets[f] + s`, offsets from node counts minus one); −1 where none.
    pub segment: Vec<i32>,
}

/// `lines` are `(x, y, z)` polylines. A voxel is owned by the segment whose
/// distance to the voxel center (minus the fiber's radius when `signed`, so
/// the nearest capsule surface wins) is smallest among segments within their
/// fiber's `reach`; ties go to the lower segment.
pub fn rasterize(shape: Shape, lines: &[Vec<[f64; 3]>], radii: &[f64], reach: &[f64], signed: bool) -> Raster {
    let n = voxel_count(shape);
    let s = strides(shape);
    let mut best = vec![f64::INFINITY; n];
    let mut owner = vec![-1i32; n];
    let mut labels = vec![0i32; n];
    let mut segment = 0i32;
    for (f, line) in lines.iter().enumerate() {
        let limit = reach[f];
        let key = if signed { radii[f] } else { 0.0 };
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if let Some((low, high)) = segment_box(shape, a, b, limit + 0.5) {
                for k in low[2]..high[2] {
                    for j in low[1]..high[1] {
                        for i in low[0]..high[0] {
                            let p = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                            let distance = segment_distance(p, a, b);
                            if distance > limit {
                                continue;
                            }
                            let value = distance - key;
                            let index = k * s[0] + j * s[1] + i;
                            if value < best[index] {
                                best[index] = value;
                                owner[index] = segment;
                                labels[index] = f as i32 + 1;
                            }
                        }
                    }
                }
            }
            segment += 1;
        }
    }
    Raster { labels, distance: best.iter().map(|&v| v as f32).collect(), segment: owner }
}

/// Sets the voxels of `target` within `reach` of `line` to `value` (only
/// where `target` is 0 when `only_empty`), as `tangle.ct._geometry.paint`.
pub fn paint(target: &mut [i32], shape: Shape, line: &[[f64; 3]], reach: f64, value: i32, only_empty: bool) {
    let s = strides(shape);
    let doubled;
    let line = if line.len() == 1 {
        doubled = [line[0], line[0]];
        &doubled[..]
    } else {
        line
    };
    for pair in line.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if let Some((low, high)) = segment_box(shape, a, b, reach + 0.5) {
            for k in low[2]..high[2] {
                for j in low[1]..high[1] {
                    for i in low[0]..high[0] {
                        let p = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                        let index = k * s[0] + j * s[1] + i;
                        if (only_empty && target[index] != 0) || segment_distance(p, a, b) > reach {
                            continue;
                        }
                        target[index] = value;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearer_surface_owns_the_voxel() {
        let shape = [1, 1, 10];
        let lines = vec![vec![[1.5, 0.5, 0.5], [1.5, 0.5, 0.5]], vec![[7.5, 0.5, 0.5], [7.5, 0.5, 0.5]]];
        let raster = rasterize(shape, &lines, &[1.0, 4.0], &[6.0, 6.0], true);
        // voxel 3 (center 3.5): 2 − 1 = 1 from fiber 1, 4 − 4 = 0 from fiber 2
        assert_eq!(raster.labels[3], 2);
        assert_eq!(raster.labels[1], 1);
        assert_eq!(raster.segment[3], 1);
        let plain = rasterize(shape, &lines, &[1.0, 4.0], &[6.0, 6.0], false);
        assert_eq!(plain.labels[3], 1);
    }

    #[test]
    fn paint_fills_a_capsule() {
        let shape = [5, 5, 5];
        let mut target = vec![0i32; 125];
        paint(&mut target, shape, &[[2.5, 2.5, 2.5]], 1.0, 7, false);
        assert_eq!(target.iter().filter(|&&v| v == 7).count(), 7);
    }
}
