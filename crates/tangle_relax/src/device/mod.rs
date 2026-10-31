mod image_force;
mod junction;
mod kernels;
mod packed;
mod plugin;
mod world;

pub use packed::{
    oval_lane_count, AdaptiveSegmentationConfig, PackedAssembly, PackingError, MAXIMUM_OVAL_LANES,
    OVAL_LANE_SPACING,
};
pub use plugin::{
    ContactAggregation, DeviceState, DeviceWorld, FiberMotion, RelaxationBackend, RelaxationConfig,
    RelaxationOverrides, RelaxationPlugin, RelaxationState, WorkflowControl,
};
pub use world::{
    BatchStatus, ContactCapture, DeviceFiberWorld, DeviceWorldCheckpoint, FormationTargetError,
    ImageForceSettings, LayerTargetCheckpoint, SegmentContactCandidate, VertexTargetCheckpoint,
};

/// The most cubes a launch may have along one axis: WebGPU's
/// `maxComputeWorkgroupsPerDimension`.
const MAX_CUBES_PER_AXIS: usize = 65_535;

/// A launch of `cubes` cubes as a grid every backend accepts: one row while it
/// fits, and more rows of at most [`MAX_CUBES_PER_AXIS`] for a large model. The
/// kernels index by `ABSOLUTE_POS` and `CUBE_POS`, which run on from row to row,
/// and skip the units past their length, so the extra cubes of the last row do
/// nothing.
pub(crate) fn cube_grid(cubes: usize) -> cubecl::prelude::CubeCount {
    let rows = cubes.div_ceil(MAX_CUBES_PER_AXIS).max(1);
    // An empty launch stays empty: zero cubes in one row.
    let row = cubes.div_ceil(rows);
    let rows = u32::try_from(rows).expect("too many cubes for one launch");
    cubecl::prelude::CubeCount::Static(row as u32, rows, 1)
}

/// One explicitly requested host-visible debug snapshot.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RelaxationSnapshot {
    /// Device-side correction iteration represented by this frame.
    pub iteration: usize,
    /// Interleaved xyz coordinates downloaded at that iteration.
    pub positions: Vec<f32>,
    /// Active topology reconstructed for adaptive layouts.
    pub assembly: Option<tangle_core::FiberAssembly>,
}

#[cfg(test)]
mod cube_grid_tests {
    use super::{cube_grid, MAX_CUBES_PER_AXIS};
    use cubecl::prelude::CubeCount;

    #[test]
    fn large_launches_spread_over_rows_of_at_most_65535_cubes() {
        for cubes in [0, 1, 64, 65_535, 65_536, 180_261, 10_000_000] {
            let CubeCount::Static(x, y, z) = cube_grid(cubes) else {
                panic!("cube_grid gives a static count");
            };
            assert!(x as usize <= MAX_CUBES_PER_AXIS && y as usize <= MAX_CUBES_PER_AXIS);
            assert_eq!(z, 1);
            assert!(x as usize * y as usize >= cubes);
            assert!(x as usize * y as usize - cubes < y as usize);
        }
        assert!(matches!(cube_grid(65_535), CubeCount::Static(65_535, 1, 1)));
        assert!(matches!(cube_grid(180_261), CubeCount::Static(60_087, 3, 1)));
    }
}
