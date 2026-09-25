//! The Gaussian-scale Hessian of a scan and the local tube direction.

use crate::filter::{filter_axis, gaussian_kernel, kernel_radius};
use crate::sample::trilinear;
use crate::Shape;

/// Components in `(x, y, z)` index pairs, with the derivative orders along
/// `z, y, x` that give each one.
const COMPONENTS: [((usize, usize), [usize; 3]); 6] = [
    ((0, 0), [0, 0, 2]),
    ((1, 1), [0, 2, 0]),
    ((2, 2), [2, 0, 0]),
    ((0, 1), [0, 1, 1]),
    ((0, 2), [1, 0, 1]),
    ((1, 2), [1, 1, 0]),
];

/// The six Hessian components of a normalized image at Gaussian scale
/// `sigma`, times `sigma²`, read at arbitrary points by trilinear
/// interpolation.
pub struct HessianField {
    pub sigma: f64,
    shape: Shape,
    components: Vec<Vec<f32>>,
}

impl HessianField {
    /// Each component is `gaussian_filter(image, sigma, order)` times `sigma²`,
    /// with the passes along `z` and then `y` shared between the components
    /// that need the same ones (15 axis passes rather than 18).
    pub fn new(image: &[f32], shape: Shape, sigma: f64) -> Self {
        let scale = (sigma * sigma) as f32;
        let radius = kernel_radius(sigma);
        let kernels: Vec<Vec<f64>> = (0..3)
            .map(|order| gaussian_kernel(sigma, order, radius))
            .collect();
        let along_z: Vec<Vec<f32>> = kernels
            .iter()
            .map(|kernel| filter_axis(image, shape, 0, kernel))
            .collect();
        let mut along_zy: Vec<([usize; 2], Vec<f32>)> = Vec::new();
        let components = COMPONENTS
            .iter()
            .map(|(_, [oz, oy, ox])| {
                let key = [*oz, *oy];
                let position = match along_zy.iter().position(|(k, _)| *k == key) {
                    Some(position) => position,
                    None => {
                        let values = filter_axis(&along_z[*oz], shape, 1, &kernels[*oy]);
                        along_zy.push((key, values));
                        along_zy.len() - 1
                    }
                };
                let mut values = filter_axis(&along_zy[position].1, shape, 2, &kernels[*ox]);
                values.iter_mut().for_each(|v| *v *= scale);
                values
            })
            .collect();
        Self {
            sigma,
            shape,
            components,
        }
    }

    /// The symmetric Hessian at `point` (`(x, y, z)` rows and columns).
    pub fn at(&self, point: [f64; 3]) -> [[f64; 3]; 3] {
        let mut h = [[0.0; 3]; 3];
        for (((i, j), _), values) in COMPONENTS.iter().zip(&self.components) {
            let v = trilinear(values, self.shape, point, 0.0);
            h[*i][*j] = v;
            h[*j][*i] = v;
        }
        h
    }

    /// The tube axis (eigenvector of the smallest-magnitude eigenvalue) and
    /// the tubularity `-(λ2 + λ3) / 2` of the other two, positive on a
    /// bright tube and near zero off it.
    pub fn direction(&self, point: [f64; 3]) -> ([f64; 3], f64) {
        let (values, vectors) = symmetric_eigen(self.at(point));
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| values[a].abs().total_cmp(&values[b].abs()));
        let axis = [
            vectors[0][order[0]],
            vectors[1][order[0]],
            vectors[2][order[0]],
        ];
        (axis, -0.5 * (values[order[1]] + values[order[2]]))
    }
}

/// Eigenvalues and eigenvectors (as columns) of a symmetric 3×3 matrix, by
/// cyclic Jacobi rotations.
pub fn symmetric_eigen(matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = matrix;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let scale: f64 = a.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return ([0.0; 3], v);
    }
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off <= 1e-15 * scale {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() <= 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A ← Jᵀ A J for the rotation J in the (p, q) plane.
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_decomposes_a_symmetric_matrix() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.2], [0.5, -0.2, -1.0]];
        let (values, vectors) = symmetric_eigen(m);
        for k in 0..3 {
            let x = [vectors[0][k], vectors[1][k], vectors[2][k]];
            for i in 0..3 {
                let mx: f64 = (0..3).map(|j| m[i][j] * x[j]).sum();
                assert!((mx - values[k] * x[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn a_bright_rod_along_x_points_along_x() {
        let shape = [15, 15, 15];
        let mut image = vec![0.0_f32; 15 * 15 * 15];
        for k in 0..15 {
            for j in 0..15 {
                for i in 0..15 {
                    let (dy, dz) = (j as f64 - 7.0, k as f64 - 7.0);
                    image[(k * 15 + j) * 15 + i] = (-(dy * dy + dz * dz) / 4.0).exp() as f32;
                }
            }
        }
        let field = HessianField::new(&image, shape, 1.5);
        let (axis, tubularity) = field.direction([7.5, 7.5, 7.5]);
        assert!(axis[0].abs() > 0.99, "{axis:?}");
        assert!(tubularity > 0.0);
    }
}
