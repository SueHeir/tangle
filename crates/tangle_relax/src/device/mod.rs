mod kernels;
mod packed;
mod plugin;
mod world;

pub use packed::{AdaptiveSegmentationConfig, PackedAssembly, PackingError};
pub use plugin::{
    ContactAggregation, DeviceState, DeviceWorld, FiberMotion, RelaxationBackend, RelaxationConfig,
    RelaxationOverrides, RelaxationPlugin, RelaxationState, WorkflowControl,
};
pub use world::{
    BatchStatus, ContactCapture, DeviceFiberWorld, DeviceWorldCheckpoint, FormationTargetError,
    LayerTargetCheckpoint, SegmentContactCandidate, VertexTargetCheckpoint,
};

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
