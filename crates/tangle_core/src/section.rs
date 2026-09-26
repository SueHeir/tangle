use crate::SectionId;

/// Fiber cross-section geometry.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Section {
    /// A circular cross-section.
    Circular {
        /// Circle radius.
        radius: f64,
    },
    /// An elliptical cross-section.
    Elliptical {
        /// The two semi-axis lengths. The first lies along the fiber's
        /// per-vertex director (see [`crate::GeometryState::directors`]),
        /// the second perpendicular to both director and tangent.
        semi_axes: [f64; 2],
    },
}

impl Section {
    /// Whether the section is round, so its orientation does not matter.
    pub fn is_circular(&self) -> bool {
        match self {
            Self::Circular { .. } => true,
            Self::Elliptical { semi_axes } => semi_axes[0] == semi_axes[1],
        }
    }

    /// Cross-sectional area.
    pub fn area(&self) -> f64 {
        match self {
            Self::Circular { radius } => std::f64::consts::PI * radius * radius,
            Self::Elliptical { semi_axes } => std::f64::consts::PI * semi_axes[0] * semi_axes[1],
        }
    }

    /// Largest distance from the centerline to the section boundary.
    pub fn bounding_radius(&self) -> f64 {
        match self {
            Self::Circular { radius } => *radius,
            Self::Elliptical { semi_axes } => semi_axes[0].max(semi_axes[1]),
        }
    }
}

/// Cross-sections referenced by fibers.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SectionTable {
    /// Sections in identifier order.
    pub entries: Vec<Section>,
}

impl SectionTable {
    /// Adds a section and returns its dense identifier.
    pub fn add(&mut self, section: Section) -> SectionId {
        let id = SectionId(self.entries.len() as u32);
        self.entries.push(section);
        id
    }
}
