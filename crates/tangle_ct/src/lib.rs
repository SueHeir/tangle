//! Volume operations behind `tangle.ct`, the fitter of Tangle fibers to CT scans.
//!
//! Volumes are flat, C-ordered `(z, y, x)` arrays with a [`Shape`] of
//! `[nz, ny, nx]`. Points are `(x, y, z)` in voxel units, with voxel
//! `[k, j, i]` centered at `(i + 0.5, j + 0.5, k + 0.5)`, as in the Python
//! package. The operations follow the SciPy calls they replace
//! (`gaussian_filter`, `map_coordinates(order=1)`, `distance_transform_edt`,
//! `label`, `maximum_filter`) closely enough that fits do not change beyond
//! floating-point rounding.

pub mod edt;
pub mod filter;
pub mod hessian;
pub mod label;
pub mod line;
pub mod raster;
pub mod refine;
pub mod sample;
pub mod trace;

/// `[nz, ny, nx]`.
pub type Shape = [usize; 3];

/// Number of voxels.
#[inline]
pub fn voxel_count(shape: Shape) -> usize {
    shape[0] * shape[1] * shape[2]
}

/// C-order strides of `shape`, in elements.
#[inline]
pub fn strides(shape: Shape) -> [usize; 3] {
    [shape[1] * shape[2], shape[2], 1]
}
