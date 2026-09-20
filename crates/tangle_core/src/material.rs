use crate::MaterialId;

/// A solver-neutral material label.
///
/// Constitutive properties belong to mechanics or transport plugins. The core
/// table only preserves identity and a human-readable name.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterialDescriptor {
    /// Human-readable material name.
    pub name: String,
}

/// Material descriptors referenced by fibers.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterialTable {
    /// Materials in identifier order.
    pub entries: Vec<MaterialDescriptor>,
}

impl MaterialTable {
    /// Returns the identifier of the named material, inserting it when needed.
    ///
    /// Material names are interned: repeated insertion of the same symbolic
    /// name returns the original identifier instead of creating an ambiguous
    /// duplicate entry.
    pub fn add(&mut self, name: impl Into<String>) -> MaterialId {
        let name = name.into();
        if let Some(index) = self.entries.iter().position(|entry| entry.name == name) {
            return MaterialId(index as u32);
        }
        let id = MaterialId(self.entries.len() as u32);
        self.entries.push(MaterialDescriptor { name });
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_names_are_interned() {
        let mut materials = MaterialTable::default();
        let first = materials.add("fiber");
        let repeated = materials.add("fiber");

        assert_eq!(first, repeated);
        assert_eq!(materials.entries.len(), 1);
    }
}
