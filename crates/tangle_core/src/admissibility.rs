/// Geometric bend limit attached to a fiber independently of mechanics.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberBendLimit {
    /// Smallest admissible local bend radius.
    pub minimum_bend_radius: f64,
}

impl FiberBendLimit {
    /// Maximum admissible curvature implied by the minimum bend radius.
    pub fn maximum_curvature(self) -> f64 {
        self.minimum_bend_radius.recip()
    }
}

/// Per-fiber geometric admissibility data in dense topology order.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberAdmissibility {
    /// Optional bend limit for every fiber in [`crate::FiberTopology::fibers`].
    pub bend_limits: Vec<Option<FiberBendLimit>>,
}
