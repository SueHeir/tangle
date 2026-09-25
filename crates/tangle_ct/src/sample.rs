//! Reading a volume at arbitrary points.

use crate::{strides, Shape};

/// Trilinear sample of a `(z, y, x)` volume at point `(x, y, z)` (voxel
/// units, voxel centers at `+0.5`), as `map_coordinates(order=1,
/// mode="constant", cval=fill)`: a point beyond the outermost voxel centers
/// reads `fill`.
#[inline]
pub fn trilinear(values: &[f32], shape: Shape, point: [f64; 3], fill: f64) -> f64 {
    let s = strides(shape);
    let mut base = [0usize; 3];
    let mut frac = [0.0f64; 3];
    for axis in 0..3 {
        let c = point[2 - axis] - 0.5;
        let n = shape[axis];
        if !(c >= 0.0 && c <= (n - 1) as f64) {
            return fill; // also NaN
        }
        let mut i = c.floor() as usize;
        if i + 1 >= n {
            i = n.saturating_sub(2);
        }
        base[axis] = i;
        frac[axis] = if n == 1 { 0.0 } else { c - i as f64 };
    }
    let mut sum = 0.0;
    for dz in 0..2 {
        let wz = if dz == 0 { 1.0 - frac[0] } else { frac[0] };
        if wz == 0.0 {
            continue;
        }
        for dy in 0..2 {
            let wy = if dy == 0 { 1.0 - frac[1] } else { frac[1] };
            if wy == 0.0 {
                continue;
            }
            for dx in 0..2 {
                let wx = if dx == 0 { 1.0 - frac[2] } else { frac[2] };
                if wx == 0.0 {
                    continue;
                }
                let index = (base[0] + dz) * s[0] + (base[1] + dy) * s[1] + base[2] + dx;
                sum += wz * wy * wx * values[index] as f64;
            }
        }
    }
    sum
}

/// [`trilinear`] at every point.
pub fn trilinear_many(values: &[f32], shape: Shape, points: &[[f64; 3]], fill: f64) -> Vec<f64> {
    points.iter().map(|&p| trilinear(values, shape, p, fill)).collect()
}

/// The voxel `[k, j, i]` containing point `(x, y, z)`, or `None` outside the volume.
#[inline]
pub fn voxel_of(shape: Shape, point: [f64; 3]) -> Option<[usize; 3]> {
    let mut out = [0usize; 3];
    for axis in 0..3 {
        let c = point[2 - axis].floor();
        if !(c >= 0.0 && c < shape[axis] as f64) {
            return None;
        }
        out[axis] = c as usize;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trilinear_reads_centers_and_blends_between() {
        let shape = [2, 2, 2];
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        assert_eq!(trilinear(&values, shape, [0.5, 0.5, 0.5], -1.0), 0.0);
        assert_eq!(trilinear(&values, shape, [1.5, 1.5, 1.5], -1.0), 7.0);
        assert!((trilinear(&values, shape, [1.0, 1.0, 1.0], -1.0) - 3.5).abs() < 1e-12);
        assert_eq!(trilinear(&values, shape, [0.4, 1.0, 1.0], -1.0), -1.0);
    }
}
