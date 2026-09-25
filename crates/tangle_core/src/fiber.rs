use crate::{FiberId, MaterialId, SectionId, Span, Vec3};

/// Metadata and topology for one fiber.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fiber {
    /// Stable identity, independent of dense storage order.
    pub id: FiberId,
    /// Contiguous range in each geometry state's position array.
    pub vertices: Span,
    /// Material descriptor used by the fiber.
    pub material: MaterialId,
    /// Cross-section used by the fiber.
    pub section: SectionId,
    /// Optional manufacturing/deposition layer assigned by a generator.
    ///
    /// This is provenance rather than a permanent kinematic constraint;
    /// formation plugins may use it and later release all layer constraints.
    pub formation_layer: Option<u32>,
    /// Manufacturing recipe step at which this fiber becomes present.
    ///
    /// Step zero fibers exist at the beginning of formation. Higher steps are
    /// packed up front and activated by a device-resident formation recipe.
    pub formation_step: u32,
}

/// Ordered fibers whose vertices occupy flat contiguous arrays.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FiberTopology {
    /// Fibers in dense working order.
    pub fibers: Vec<Fiber>,
}

/// One centerline configuration for every fiber in an assembly.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GeometryState {
    /// Flat vertex positions addressed by [`Fiber::vertices`].
    pub positions: Vec<Vec3>,
    /// Unit long-axis direction of the cross-section at each vertex, for
    /// non-circular sections. Either empty (no fiber needs one) or one entry
    /// per position; see [`crate::default_directors`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub directors: Vec<Vec3>,
}

/// Intrinsic, placed, and optional manufactured reference geometry.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberGeometry {
    /// Stress-free isolated-fiber centerlines. Each fiber may use an arbitrary
    /// rigid coordinate frame; lengths and relative directions carry meaning.
    pub intrinsic: GeometryState,
    /// Current centerlines in the assembly cell's world frame.
    pub placed: GeometryState,
    /// Optional manufactured geometry chosen as a downstream reference state.
    pub assembled_reference: Option<GeometryState>,
}
