//! Separable Gaussian (and Gaussian-derivative) filters and a 3-voxel maximum
//! filter, as `scipy.ndimage.gaussian_filter` / `maximum_filter(size=3)` compute
//! them: `truncate = 4`, mirror (`"reflect"`) boundaries, axes filtered in
//! `z, y, x` order, each pass accumulated in `f64` and stored as `f32`.

use crate::{strides, voxel_count, Shape};

/// SciPy's `_gaussian_kernel1d(sigma, order, radius)`: the sampled Gaussian,
/// normalized to sum 1, times the polynomial that makes it the `order`-th
/// derivative. Index `radius + x` holds the value at offset `x`.
pub fn gaussian_kernel(sigma: f64, order: usize, radius: usize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let r = radius as i64;
    let mut phi: Vec<f64> = (-r..=r)
        .map(|x| (-0.5 / sigma2 * (x * x) as f64).exp())
        .collect();
    let total: f64 = phi.iter().sum();
    phi.iter_mut().for_each(|v| *v /= total);
    if order == 0 {
        return phi;
    }
    // q holds polynomial coefficients; each step differentiates q(x)·phi(x):
    // q ← q' + q·(−x/σ²).
    let mut q = vec![0.0; order + 1];
    q[0] = 1.0;
    for _ in 0..order {
        let mut next = vec![0.0; order + 1];
        for i in 0..=order {
            let derivative = if i < order {
                (i + 1) as f64 * q[i + 1]
            } else {
                0.0
            };
            let product = if i > 0 { -q[i - 1] / sigma2 } else { 0.0 };
            next[i] = derivative + product;
        }
        q = next;
    }
    (-r..=r)
        .zip(phi)
        .map(|(x, p)| {
            let x = x as f64;
            let mut value = 0.0;
            let mut power = 1.0;
            for coefficient in &q {
                value += coefficient * power;
                power *= x;
            }
            value * p
        })
        .collect()
}

/// Index `m` of a line of length `n` under mirror boundaries (`d c b a | a b c d | d c b a`).
#[inline]
fn reflect(m: i64, n: i64) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n;
    let mut m = m.rem_euclid(period);
    if m >= n {
        m = period - 1 - m;
    }
    m as usize
}

/// The flat index of the first voxel of every line along `axis`.
pub(crate) fn line_starts(shape: Shape, axis: usize) -> Vec<usize> {
    let s = strides(shape);
    let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
    let mut starts = Vec::with_capacity(shape[others[0]] * shape[others[1]]);
    for a in 0..shape[others[0]] {
        for b in 0..shape[others[1]] {
            starts.push(a * s[others[0]] + b * s[others[1]]);
        }
    }
    starts
}

/// Correlates every line of `input` along `axis` with `kernel` (SciPy's
/// `gaussian_filter1d`: `correlate1d` with the kernel reversed), mirror
/// boundaries, accumulated in `f64` as SciPy does.
///
/// Along `x` each line is mirror-padded once and swept tap by tap. Along `y`
/// and `z` the lines are strided, so each output row (all voxels sharing the
/// other two indices, contiguous in memory) is summed from whole input rows
/// instead, which reads memory in order. Rows are split across threads.
pub(crate) fn filter_axis(input: &[f32], shape: Shape, axis: usize, kernel: &[f64]) -> Vec<f32> {
    let n = shape[axis];
    let radius = kernel.len() / 2;
    // correlate1d with weights = kernel[::-1]: out[i] = Σ_k kernel[r − k] · in[i + k − r].
    let weights: Vec<f64> = kernel.iter().rev().copied().collect();
    let mut out = vec![0.0_f32; input.len()];
    if out.is_empty() {
        return out;
    }
    if axis == 2 {
        parallel_rows(&mut out, n, |row, target| {
            let line = &input[row * n..(row + 1) * n];
            let padded: Vec<f64> = (0..n + 2 * radius)
                .map(|m| line[reflect(m as i64 - radius as i64, n as i64)] as f64)
                .collect();
            for (i, value) in target.iter_mut().enumerate() {
                let window = &padded[i..i + weights.len()];
                let mut sum = 0.0;
                for (a, w) in window.iter().zip(&weights) {
                    sum += w * a;
                }
                *value = sum as f32;
            }
        });
        return out;
    }
    // Rows of `inner` contiguous voxels; row `o * n + i` is index `i` along `axis`.
    let inner: usize = shape[axis + 1..].iter().product();
    parallel_rows(&mut out, inner, |row, target| {
        let (o, i) = (row / n, row % n);
        let mut sum = vec![0.0_f64; inner];
        for (k, weight) in weights.iter().enumerate() {
            let m = reflect(i as i64 + k as i64 - radius as i64, n as i64);
            let source = &input[(o * n + m) * inner..(o * n + m + 1) * inner];
            for (s, v) in sum.iter_mut().zip(source) {
                *s += weight * *v as f64;
            }
        }
        for (t, s) in target.iter_mut().zip(&sum) {
            *t = *s as f32;
        }
    });
    out
}

/// Runs `work(row, slice)` on every `row_len`-long row of `out`, spreading
/// contiguous runs of rows over the available cores.
pub(crate) fn parallel_rows<F>(out: &mut [f32], row_len: usize, work: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let rows = out.len() / row_len;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows)
        .max(1);
    // Small volumes are not worth the threads.
    if threads == 1 || out.len() < 1 << 16 {
        for (row, slice) in out.chunks_mut(row_len).enumerate() {
            work(row, slice);
        }
        return;
    }
    let per_thread = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (block, slab) in out.chunks_mut(per_thread * row_len).enumerate() {
            let work = &work;
            scope.spawn(move || {
                for (offset, slice) in slab.chunks_mut(row_len).enumerate() {
                    work(block * per_thread + offset, slice);
                }
            });
        }
    });
}

/// The kernel radius SciPy uses for `sigma` with `truncate = 4`.
pub fn kernel_radius(sigma: f64) -> usize {
    (4.0 * sigma + 0.5) as usize
}

/// `gaussian_filter(image, sigma, order=orders)` for a `(z, y, x)` `f32`
/// volume; `orders` are the derivative orders along `z, y, x`.
pub fn gaussian_filter(image: &[f32], shape: Shape, sigma: f64, orders: [usize; 3]) -> Vec<f32> {
    assert_eq!(image.len(), voxel_count(shape));
    let mut current = image.to_vec();
    if sigma <= 1e-15 {
        return current;
    }
    let radius = kernel_radius(sigma);
    for axis in 0..3 {
        let kernel = gaussian_kernel(sigma, orders[axis], radius);
        current = filter_axis(&current, shape, axis, &kernel);
    }
    current
}

/// `maximum_filter(values, size=3)`: the maximum over each voxel's 3×3×3
/// neighborhood (the boundary voxel stands in for the one beyond it).
pub fn maximum_filter3(values: &[f32], shape: Shape) -> Vec<f32> {
    let mut current = values.to_vec();
    for axis in 0..3 {
        let n = shape[axis];
        let stride = strides(shape)[axis];
        let mut out = current.clone();
        for index in 0..current.len() {
            let i = (index / stride) % n;
            let mut best = current[index];
            if i > 0 {
                best = best.max(current[index - stride]);
            }
            if i + 1 < n {
                best = best.max(current[index + stride]);
            }
            out[index] = best;
        }
        current = out;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_match_scipy() {
        // scipy.ndimage._filters._gaussian_kernel1d(1.5, order, 6)
        let k0 = gaussian_kernel(1.5, 0, 6);
        assert!((k0.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((k0[6] - 0.26596426).abs() < 1e-7);
        let k1 = gaussian_kernel(1.5, 1, 6);
        assert!((k1[7] - -0.09465223).abs() < 1e-7, "{}", k1[7]); // −x/σ² · phi at x = 1
        let k2 = gaussian_kernel(1.5, 2, 6);
        assert!((k2[6] - -0.11820634).abs() < 1e-7, "{}", k2[6]); // −1/σ² · phi at x = 0
    }

    #[test]
    fn smoothing_keeps_a_constant_and_its_derivative_is_zero() {
        let shape = [5, 6, 7];
        let image = vec![2.0_f32; voxel_count(shape)];
        let smooth = gaussian_filter(&image, shape, 1.2, [0, 0, 0]);
        assert!(smooth.iter().all(|v| (v - 2.0).abs() < 1e-5));
        let slope = gaussian_filter(&image, shape, 1.2, [0, 0, 1]);
        assert!(slope.iter().all(|v| v.abs() < 1e-5));
    }

    /// The direct line-by-line correlation the fast passes replace.
    fn reference_axis(input: &[f32], shape: Shape, axis: usize, kernel: &[f64]) -> Vec<f32> {
        let n = shape[axis];
        let stride = strides(shape)[axis];
        let radius = (kernel.len() / 2) as i64;
        let weights: Vec<f64> = kernel.iter().rev().copied().collect();
        let mut out = vec![0.0_f32; input.len()];
        for start in line_starts(shape, axis) {
            for i in 0..n {
                let mut sum = 0.0;
                for (k, weight) in weights.iter().enumerate() {
                    let m = reflect(i as i64 + k as i64 - radius, n as i64);
                    sum += weight * input[start + m * stride] as f64;
                }
                out[start + i * stride] = sum as f32;
            }
        }
        out
    }

    #[test]
    fn every_axis_pass_matches_the_direct_correlation() {
        // The second volume is large enough to take the threaded path; the
        // short z axis mirrors past both ends.
        let small = [3, 50, 47];
        let big = [4, 130, 129];
        for (seed, shape) in [(7919, small), (104729, big)] {
            let image: Vec<f32> = (0..voxel_count(shape))
                .map(|i| ((i * seed) % 101) as f32 / 101.0)
                .collect();
            for axis in 0..3 {
                for order in 0..3 {
                    let kernel = gaussian_kernel(1.7, order, kernel_radius(1.7));
                    let fast = filter_axis(&image, shape, axis, &kernel);
                    let direct = reference_axis(&image, shape, axis, &kernel);
                    assert_eq!(fast, direct, "axis {axis} order {order}");
                }
            }
        }
    }

    #[test]
    fn maximum_filter_spreads_one_voxel() {
        let shape = [3, 3, 3];
        let mut values = vec![0.0_f32; 27];
        values[0] = 1.0;
        let out = maximum_filter3(&values, shape);
        let lit = out.iter().filter(|v| **v == 1.0).count();
        assert_eq!(lit, 8);
    }
}
