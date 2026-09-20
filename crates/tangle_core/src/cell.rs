use crate::math::{cross, dot};
use crate::Vec3;

/// A periodic or bounded three-dimensional material cell.
///
/// The entries of `basis` are the cell edge vectors. An orthorhombic cell has
/// a diagonal basis, while a sheared or triclinic cell may use off-diagonal
/// entries. Fiber coordinates are stored unwrapped.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PeriodicCell {
    /// World-space origin of the cell.
    pub origin: Vec3,
    /// Cell edge vectors, one per entry.
    pub basis: [Vec3; 3],
    /// Whether opposite faces are periodic along each cell direction.
    pub periodic: [bool; 3],
}

impl PeriodicCell {
    /// Creates an orthorhombic cell with the supplied edge lengths.
    pub fn orthorhombic(lengths: Vec3, periodic: [bool; 3]) -> Self {
        Self {
            origin: [0.0; 3],
            basis: [
                [lengths[0], 0.0, 0.0],
                [0.0, lengths[1], 0.0],
                [0.0, 0.0, lengths[2]],
            ],
            periodic,
        }
    }

    /// Returns the signed cell volume.
    pub fn signed_volume(&self) -> f64 {
        dot(self.basis[0], cross(self.basis[1], self.basis[2]))
    }
}
