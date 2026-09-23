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
    /// Neighbor-list skin as a multiple of the largest fiber radius.
    ///
    /// Each segment keeps the segments within this distance of touching and
    /// the lists are rebuilt once any vertex has moved half the skin. Larger
    /// skins rebuild less often but check more pairs per iteration; zero
    /// rebuilds whenever anything moves.
    pub neighbor_skin_scale: f32,
    /// Neighbor slots stored per segment. A segment with more neighbors than
    /// this falls back to scanning its cells, so the value only affects speed
    /// and memory, never which contacts are found.
    pub neighbor_capacity: u32,
}

impl Default for CellListConfig {
    fn default() -> Self {
        Self {
            cell_size_scale: 1.0,
            neighbor_skin_scale: 2.0,
            neighbor_capacity: 48,
        }
    }
}
