//! Device-resident contact and centerline relaxation for TANGLE.
//!
//! Relaxation is implemented with CubeCL kernels and persistent device buffers.
//! The public API is organized around mechanics concepts rather than a
//! particular accelerator or CPU backend.

#![warn(missing_docs)]

mod compaction;
mod device;

pub use compaction::{CompactionEnergyModel, CompactionKinematics, CompactionMetrics};

pub use device::{
    oval_lane_count, AdaptiveSegmentationConfig, BatchStatus, ContactAggregation, ContactCapture,
    DeviceFiberWorld, DeviceState, DeviceWorld, DeviceWorldCheckpoint, FiberMotion,
    FormationTargetError, ImageForceSettings, LayerTargetCheckpoint, PackedAssembly, PackingError,
    RelaxationBackend, RelaxationConfig, RelaxationOverrides, RelaxationPlugin, RelaxationSnapshot,
    RelaxationState, SegmentContactCandidate, VertexTargetCheckpoint, WorkflowControl,
    MAXIMUM_OVAL_LANES, OVAL_LANE_SPACING,
};
pub use tangle_contact::CellListConfig;
