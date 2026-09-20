use crate::math::distance;
use crate::{
    AnchorError, FiberAssembly, FiberId, JunctionId, JunctionLawId, JunctionParameterId, Span, Vec3,
};

/// A material location on a fiber.
///
/// The rest arc length is measured along the intrinsic centerline, so the
/// anchor remains attached to the same material point when the polyline is
/// remeshed.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberAnchor {
    /// Stable fiber identity.
    pub fiber: FiberId,
    /// Arc length from the beginning of the intrinsic fiber.
    pub rest_arc_length: f64,
    /// Optional coordinates in the local cross-section. None anchors to the
    /// centerline and does not require a material frame.
    pub section_offset: Option<[f64; 2]>,
}

/// A material anchor resolved against the current polyline discretization.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedAnchor {
    /// Dense fiber index in the fiber topology.
    pub fiber_index: u32,
    /// Segment index local to the fiber.
    pub local_segment: u32,
    /// Interpolation coordinate from the segment's first vertex to its second.
    pub coordinate: f64,
}

/// A persistent junction connecting two or more fiber material locations.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Junction {
    /// Stable junction identity.
    pub id: JunctionId,
    /// Range in the junction table's flat anchor array.
    pub anchors: Span,
    /// Law family that interprets this junction.
    pub law: JunctionLawId,
    /// Parameter set interpreted by the law plugin.
    pub parameters: JunctionParameterId,
}

/// Persistent junction topology and its flat anchor storage.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JunctionTable {
    /// Junctions in dense working order.
    pub junctions: Vec<Junction>,
    /// All anchors, grouped contiguously by junction.
    pub anchors: Vec<FiberAnchor>,
}

/// A symbolic junction-law family.
///
/// Mechanics plugins use the ID to locate typed parameter and state tables.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JunctionLawDescriptor {
    /// Stable human-readable law name, such as welded or frictional-slip.
    pub name: String,
}

/// Junction-law descriptors referenced by persistent junctions.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JunctionLawTable {
    /// Laws in identifier order.
    pub entries: Vec<JunctionLawDescriptor>,
}

impl JunctionLawTable {
    /// Returns the identifier of the named law, inserting it when needed.
    ///
    /// Law names are interned so name-based recipe lookup is unambiguous.
    pub fn add(&mut self, name: impl Into<String>) -> JunctionLawId {
        let name = name.into();
        if let Some(index) = self.entries.iter().position(|entry| entry.name == name) {
            return JunctionLawId(index as u32);
        }
        let id = JunctionLawId(self.entries.len() as u32);
        self.entries.push(JunctionLawDescriptor { name });
        id
    }
}

impl FiberAssembly {
    /// Resolves a material-coordinate anchor onto the intrinsic polyline.
    pub fn resolve_anchor(&self, anchor: FiberAnchor) -> Result<ResolvedAnchor, AnchorError> {
        let (fiber_index, fiber) = self
            .topology
            .fibers
            .iter()
            .enumerate()
            .find(|(_, fiber)| fiber.id == anchor.fiber)
            .ok_or(AnchorError::UnknownFiber(anchor.fiber))?;
        let range = fiber
            .vertices
            .as_usize_range()
            .ok_or(AnchorError::InvalidFiberSpan(anchor.fiber))?;
        let points = self
            .geometry
            .intrinsic
            .positions
            .get(range)
            .ok_or(AnchorError::InvalidFiberSpan(anchor.fiber))?;
        resolve_on_polyline(fiber_index as u32, points, anchor.rest_arc_length)
    }
}

fn resolve_on_polyline(
    fiber_index: u32,
    points: &[Vec3],
    rest_arc_length: f64,
) -> Result<ResolvedAnchor, AnchorError> {
    if !rest_arc_length.is_finite() || rest_arc_length < 0.0 {
        return Err(AnchorError::InvalidArcLength(rest_arc_length));
    }

    let mut traversed = 0.0;
    let mut final_nonzero = None;
    for (segment, pair) in points.windows(2).enumerate() {
        let length = distance(pair[0], pair[1]);
        if length <= f64::EPSILON {
            continue;
        }
        final_nonzero = Some((segment, length, traversed));
        let end = traversed + length;
        if rest_arc_length <= end {
            return Ok(ResolvedAnchor {
                fiber_index,
                local_segment: segment as u32,
                coordinate: ((rest_arc_length - traversed) / length).clamp(0.0, 1.0),
            });
        }
        traversed = end;
    }

    let Some((segment, length, before)) = final_nonzero else {
        return Err(AnchorError::DegenerateFiber);
    };
    let total = before + length;
    let tolerance = f64::EPSILON.sqrt() * total.max(1.0);
    if rest_arc_length <= total + tolerance {
        Ok(ResolvedAnchor {
            fiber_index,
            local_segment: segment as u32,
            coordinate: 1.0,
        })
    } else {
        Err(AnchorError::BeyondFiber {
            requested: rest_arc_length,
            length: total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn junction_law_names_are_interned() {
        let mut laws = JunctionLawTable::default();
        let first = laws.add("sprayed");
        let repeated = laws.add("sprayed");

        assert_eq!(first, repeated);
        assert_eq!(laws.entries.len(), 1);
    }
    use crate::{FiberId, PeriodicCell, Section};

    fn basic_assembly() -> FiberAssembly {
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([1.0, 1.0, 1.0], [false; 3]));
        let material = assembly.materials.add("test fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.01 });
        assembly
            .add_fiber(
                FiberId(7),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.0, 0.5, 0.5], [0.4, 0.5, 0.5], [1.0, 0.5, 0.5]],
            )
            .unwrap();
        assembly
    }

    #[test]
    fn resolves_material_coordinate_across_nonuniform_segments() {
        let assembly = basic_assembly();
        let resolved = assembly
            .resolve_anchor(FiberAnchor {
                fiber: FiberId(7),
                rest_arc_length: 0.7,
                section_offset: None,
            })
            .unwrap();
        assert_eq!(resolved.local_segment, 1);
        assert!((resolved.coordinate - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_anchor_beyond_intrinsic_fiber() {
        let assembly = basic_assembly();
        let error = assembly
            .resolve_anchor(FiberAnchor {
                fiber: FiberId(7),
                rest_arc_length: 1.1,
                section_offset: None,
            })
            .unwrap_err();
        assert!(matches!(error, AnchorError::BeyondFiber { .. }));
    }
}
