macro_rules! id_type {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Clone,
            Copy,
            Debug,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
        )]
        pub struct $name(pub u32);
    };
}

id_type!(FiberId, "A stable identifier for one physical fiber.");
id_type!(MaterialId, "An index into a material table.");
id_type!(SectionId, "An index into a section table.");
id_type!(
    JunctionId,
    "A stable identifier for one persistent junction."
);
id_type!(JunctionLawId, "An index into a junction-law table.");
id_type!(
    JunctionParameterId,
    "A plugin-owned junction parameter-set identifier."
);

/// A compact range into a flat table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Span {
    /// First element in the range.
    pub start: u32,
    /// Number of elements in the range.
    pub len: u32,
}

impl Span {
    /// Returns the exclusive end of the range, if it does not overflow.
    pub fn checked_end(self) -> Option<u32> {
        self.start.checked_add(self.len)
    }

    pub(crate) fn as_usize_range(self) -> Option<std::ops::Range<usize>> {
        Some(self.start as usize..self.checked_end()? as usize)
    }
}
