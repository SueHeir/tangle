use crate::{
    BuildError, Fiber, FiberAdmissibility, FiberAnchor, FiberBendLimit, FiberGeometry, FiberId,
    FiberTopology, Junction, JunctionId, JunctionLawId, JunctionLawTable, JunctionParameterId,
    JunctionTable, MaterialId, MaterialTable, PeriodicCell, Provenance, SectionId, SectionTable,
    Span, Vec3,
};

/// Canonical, solver-neutral description of a fibrous material assembly.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberAssembly {
    /// Material cell and periodicity.
    pub cell: PeriodicCell,
    /// Fiber identity and flat-array ranges.
    pub topology: FiberTopology,
    /// Intrinsic and placed centerline states.
    pub geometry: FiberGeometry,
    /// Solver-neutral geometric limits associated with each fiber.
    pub admissibility: FiberAdmissibility,
    /// Solver-neutral material labels.
    pub materials: MaterialTable,
    /// Fiber cross-sections.
    pub sections: SectionTable,
    /// Persistent material junctions.
    pub junctions: JunctionTable,
    /// Junction-law names; behavior is supplied by plugins.
    pub junction_laws: JunctionLawTable,
    /// Generation or import provenance.
    pub provenance: Provenance,
}

impl FiberAssembly {
    /// Creates an empty assembly in the supplied cell.
    pub fn new(cell: PeriodicCell) -> Self {
        Self {
            cell,
            topology: FiberTopology::default(),
            geometry: FiberGeometry::default(),
            admissibility: FiberAdmissibility::default(),
            materials: MaterialTable::default(),
            sections: SectionTable::default(),
            junctions: JunctionTable::default(),
            junction_laws: JunctionLawTable::default(),
            provenance: Provenance::default(),
        }
    }

    /// Captures the current placed centerlines as the stress-free assembled
    /// reference used by downstream mechanics and export plugins.
    pub fn capture_assembled_reference(&mut self) {
        self.geometry.assembled_reference = Some(self.geometry.placed.clone());
    }

    /// Assigns or removes the bend limit for a fiber identified by stable ID.
    pub fn set_fiber_bend_limit(
        &mut self,
        id: FiberId,
        limit: Option<FiberBendLimit>,
    ) -> Result<(), BuildError> {
        let index = self
            .topology
            .fibers
            .iter()
            .position(|fiber| fiber.id == id)
            .ok_or(BuildError::UnknownFiber(id))?;
        self.admissibility.bend_limits[index] = limit;
        Ok(())
    }

    /// Adds a fiber with matching intrinsic and placed discretizations.
    pub fn add_fiber(
        &mut self,
        id: FiberId,
        material: MaterialId,
        section: SectionId,
        intrinsic: &[Vec3],
        placed: &[Vec3],
    ) -> Result<(), BuildError> {
        if intrinsic.len() != placed.len() {
            return Err(BuildError::GeometryLengthMismatch {
                intrinsic: intrinsic.len(),
                placed: placed.len(),
            });
        }
        if intrinsic.len() < 2 {
            return Err(BuildError::TooFewVertices(intrinsic.len()));
        }
        if self.topology.fibers.iter().any(|fiber| fiber.id == id) {
            return Err(BuildError::DuplicateFiberId(id));
        }
        if material.0 as usize >= self.materials.entries.len() {
            return Err(BuildError::UnknownMaterial(material));
        }
        if section.0 as usize >= self.sections.entries.len() {
            return Err(BuildError::UnknownSection(section));
        }

        let start = u32::try_from(self.geometry.intrinsic.positions.len())
            .map_err(|_| BuildError::IndexOverflow)?;
        let len = u32::try_from(intrinsic.len()).map_err(|_| BuildError::IndexOverflow)?;
        self.geometry
            .intrinsic
            .positions
            .extend_from_slice(intrinsic);
        self.geometry.placed.positions.extend_from_slice(placed);
        if let Some(reference) = &mut self.geometry.assembled_reference {
            reference.positions.extend_from_slice(placed);
        }
        self.topology.fibers.push(Fiber {
            id,
            vertices: Span { start, len },
            material,
            section,
            formation_layer: None,
            formation_step: 0,
        });
        self.admissibility.bend_limits.push(None);
        Ok(())
    }

    /// Assigns the manufacturing recipe step at which a fiber is inserted.
    pub fn set_fiber_formation_step(&mut self, id: FiberId, step: u32) -> Result<(), BuildError> {
        let fiber = self
            .topology
            .fibers
            .iter_mut()
            .find(|fiber| fiber.id == id)
            .ok_or(BuildError::UnknownFiber(id))?;
        fiber.formation_step = step;
        Ok(())
    }

    /// Assigns or removes a manufacturing/deposition layer for a fiber.
    pub fn set_fiber_formation_layer(
        &mut self,
        id: FiberId,
        layer: Option<u32>,
    ) -> Result<(), BuildError> {
        let fiber = self
            .topology
            .fibers
            .iter_mut()
            .find(|fiber| fiber.id == id)
            .ok_or(BuildError::UnknownFiber(id))?;
        fiber.formation_layer = layer;
        Ok(())
    }

    /// Adds a persistent junction containing at least two anchors.
    pub fn add_junction(
        &mut self,
        id: JunctionId,
        law: JunctionLawId,
        parameters: JunctionParameterId,
        anchors: &[FiberAnchor],
    ) -> Result<(), BuildError> {
        if anchors.len() < 2 {
            return Err(BuildError::TooFewAnchors(anchors.len()));
        }
        if self.junctions.junctions.iter().any(|joint| joint.id == id) {
            return Err(BuildError::DuplicateJunctionId(id));
        }
        if law.0 as usize >= self.junction_laws.entries.len() {
            return Err(BuildError::UnknownJunctionLaw(law));
        }
        for anchor in anchors {
            if !self
                .topology
                .fibers
                .iter()
                .any(|fiber| fiber.id == anchor.fiber)
            {
                return Err(BuildError::UnknownFiber(anchor.fiber));
            }
        }

        let start =
            u32::try_from(self.junctions.anchors.len()).map_err(|_| BuildError::IndexOverflow)?;
        let len = u32::try_from(anchors.len()).map_err(|_| BuildError::IndexOverflow)?;
        self.junctions.anchors.extend_from_slice(anchors);
        self.junctions.junctions.push(Junction {
            id,
            anchors: Span { start, len },
            law,
            parameters,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FiberAnchor, JunctionParameterId, Section};

    #[test]
    fn junction_may_connect_more_than_two_fibers() {
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([1.0, 1.0, 1.0], [false; 3]));
        let material = assembly.materials.add("test fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.01 });
        for id in [7, 8, 9] {
            assembly
                .add_fiber(
                    FiberId(id),
                    material,
                    section,
                    &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    &[[0.5, 0.0, 0.5], [0.5, 1.0, 0.5]],
                )
                .unwrap();
        }
        let law = assembly.junction_laws.add("welded");
        let anchors = [7, 8, 9].map(|id| FiberAnchor {
            fiber: FiberId(id),
            rest_arc_length: 0.5,
            section_offset: None,
        });
        assembly
            .add_junction(JunctionId(3), law, JunctionParameterId(0), &anchors)
            .unwrap();
        assembly.validate().unwrap();
        assert_eq!(assembly.junctions.junctions[0].anchors.len, 3);
    }

    #[test]
    fn captures_an_independent_assembled_reference() {
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([1.0, 1.0, 1.0], [false; 3]));
        let material = assembly.materials.add("test fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.01 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
            )
            .unwrap();

        assembly.capture_assembled_reference();
        assembly.geometry.placed.positions[0][1] = 0.75;

        assert_eq!(
            assembly.geometry.assembled_reference.unwrap().positions[0],
            [0.0, 0.5, 0.0]
        );
    }
}
