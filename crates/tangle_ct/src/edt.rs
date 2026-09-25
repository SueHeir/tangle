//! Exact Euclidean distance transform (Felzenszwalb and Huttenlocher's
//! lower envelope of parabolas, one axis at a time).

use crate::filter::line_starts;
use crate::{strides, voxel_count, Shape};

/// For every voxel of `foreground`, the distance from its center to the
/// nearest background voxel center (0 on background), as
/// `scipy.ndimage.distance_transform_edt`. Without any background every
/// voxel reads the volume's diagonal, a bound on any real distance.
pub fn distance_transform(foreground: &[bool], shape: Shape) -> Vec<f32> {
    assert_eq!(foreground.len(), voxel_count(shape));
    let far = f64::INFINITY;
    let mut squared: Vec<f64> = foreground
        .iter()
        .map(|&f| if f { far } else { 0.0 })
        .collect();
    if !foreground.iter().any(|&f| !f) {
        let diagonal =
            ((shape[0] * shape[0] + shape[1] * shape[1] + shape[2] * shape[2]) as f64).sqrt();
        return vec![diagonal as f32; squared.len()];
    }
    let s = strides(shape);
    let longest = *shape.iter().max().unwrap();
    let mut f = vec![0.0; longest];
    let mut d = vec![0.0; longest];
    let mut v = vec![0usize; longest];
    let mut z = vec![0.0; longest + 1];
    for axis in [2, 1, 0] {
        let n = shape[axis];
        let stride = s[axis];
        for start in line_starts(shape, axis) {
            for i in 0..n {
                f[i] = squared[start + i * stride];
            }
            lower_envelope(&f[..n], &mut d[..n], &mut v, &mut z);
            for i in 0..n {
                squared[start + i * stride] = d[i];
            }
        }
    }
    squared.iter().map(|&q| q.sqrt() as f32).collect()
}

/// `d[q] = min_p (q − p)² + f[p]` over one line.
fn lower_envelope(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    let mut first = None;
    for (q, value) in f.iter().enumerate() {
        if value.is_finite() {
            first = Some(q);
            break;
        }
    }
    let Some(first) = first else {
        d.iter_mut().for_each(|x| *x = f64::INFINITY);
        return;
    };
    v[0] = first;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in first + 1..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            let p = v[k];
            let s =
                ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
            if s <= z[k] {
                if k == 0 {
                    // Cannot happen: z[0] is −∞.
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let dq = q as f64 - p as f64;
        d[q] = dq * dq + f[p];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_to_a_single_background_voxel() {
        let shape = [3, 4, 5];
        let mut foreground = vec![true; 60];
        foreground[0] = false; // voxel [0, 0, 0]
        let distance = distance_transform(&foreground, shape);
        let at = |k: usize, j: usize, i: usize| distance[(k * 4 + j) * 5 + i];
        assert_eq!(at(0, 0, 0), 0.0);
        assert!((at(2, 3, 4) - (4.0f32 + 9.0 + 16.0).sqrt()).abs() < 1e-6);
        assert!((at(0, 1, 1) - 2.0f32.sqrt()).abs() < 1e-6);
    }
}
