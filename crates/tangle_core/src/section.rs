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
        /// The two semi-axis lengths.
        semi_axes: [f64; 2],
    },
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
