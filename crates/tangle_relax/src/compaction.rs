/// How fiber geometry follows a changing orthorhombic cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompactionKinematics {
    /// Translate each active fiber so its center keeps the same fractional cell
    /// coordinate while preserving its centerline exactly.
    #[default]
    RigidFiberCenters,
    /// Move only the cell faces. Vertices outside the new faces are projected
    /// onto the walls and subsequent relaxation redistributes the fibers.
    MovingWalls,
    /// Apply the cell deformation affinely to every active vertex.
    AffineVertices,
}

/// Stiffness scales used to turn constraint violations into comparable energy
/// and pressure measures during formation.
///
/// These values define a penalty model for formation control. They are not a
/// replacement for a calibrated post-generation constitutive model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionEnergyModel {
    /// Penalty stiffness associated with fiber-fiber overlap.
    pub contact_stiffness: f32,
    /// Penalty stiffness associated with segment stretch.
    pub stretch_stiffness: f32,
    /// Penalty stiffness associated with admissible-curvature excess.
    pub bending_stiffness: f32,
}

impl Default for CompactionEnergyModel {
    fn default() -> Self {
        Self {
            contact_stiffness: 1.0,
            stretch_stiffness: 1.0,
            bending_stiffness: 1.0,
        }
    }
}

/// Small host-visible summary reduced from the resident GPU world.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompactionMetrics {
    /// Directional penalty pressure inferred from contact corrections.
    pub pressure: [f32; 3],
    /// Fiber-fiber overlap penalty energy.
    pub contact_energy: f32,
    /// Segment stretch penalty energy.
    pub stretch_energy: f32,
    /// Excess-curvature penalty energy.
    pub bending_energy: f32,
    /// Sum of contact, stretch, and bending penalty energies.
    pub total_energy: f32,
}
