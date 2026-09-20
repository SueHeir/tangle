//! Device-resident broad-phase construction and capsule contact detection.
//!
//! Contact is a first-class TANGLE concept. CubeCL supplies the portable GPU
//! implementation, while relaxation owns how the resulting corrections move
//! fibers.

#![warn(missing_docs)]

/// CubeCL contact kernels used by the device-resident relaxation world.
#[doc(hidden)]
pub mod device;

/// Device-side uniform-grid broad-phase parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellListConfig {
    /// Cell width divided by the conservative capsule interaction span.
    /// Values above one trade more candidate checks for additional margin.
    pub cell_size_scale: f32,
}

impl Default for CellListConfig {
    fn default() -> Self {
        Self {
            cell_size_scale: 1.0,
        }
    }
}
