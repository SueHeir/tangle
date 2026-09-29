//! Binder bridges drawn at persistent junctions.
//!
//! In bonded fibrous materials such as carbon-bonded carbon fiber (FiberForm,
//! CBCF) the binder collects where fibers cross and joins them there. A
//! junction's binder is modeled as a bridge: a round capsule from the anchor
//! point on one fiber's centerline to the anchor point on the other, so the
//! part outside both fibers is a neck around their contact. Its radius is a
//! fraction of the thinner fiber's radius (the short semi-axis for an oval
//! fiber), because published bond sizes relative to fiber diameter are not
//! available and the fraction is left to the recipe.

use std::collections::HashMap;

use tangle_core::{FiberAnchor, FiberAssembly, FiberId, Section, Vec3};

use crate::ExportError;

/// Default binder-bridge radius as a fraction of the thinner fiber's radius.
pub const DEFAULT_BOND_RADIUS_RATIO: f64 = 1.0;

/// One binder bridge between two anchored fiber centerline points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JunctionBridge {
    /// Dense index of the junction in the assembly's junction table.
    pub junction_index: usize,
    /// Stable junction identifier.
    pub junction_id: u32,
    /// Junction-law identifier.
    pub law: u32,
    /// Placed centerline point of the first anchor.
    pub start: Vec3,
    /// Placed centerline point of the other anchor, taken at the periodic
    /// image nearest to `start`.
    pub end: Vec3,
    /// Bridge radius.
    pub radius: f64,
}

impl JunctionBridge {
    /// Distance between the two anchor points.
    pub fn length(&self) -> f64 {
        let delta = [
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
            self.end[2] - self.start[2],
        ];
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
    }
}

/// Resolves every persistent junction into binder bridges on the placed
/// geometry: one bridge from a junction's first anchor to each other anchor.
///
/// Periodic axes use the minimum image, so a junction captured across a cell
/// wall gives a short bridge rather than one spanning the cell.
pub fn junction_bridges(
    assembly: &FiberAssembly,
    radius_ratio: f64,
) -> Result<Vec<JunctionBridge>, ExportError> {
    if !radius_ratio.is_finite() || radius_ratio <= 0.0 {
        return Err(ExportError::InvalidBondRadiusRatio(radius_ratio));
    }
    if assembly.junctions.junctions.is_empty() {
        return Ok(Vec::new());
    }
    let (box_low, box_high) = crate::orthorhombic_bounds(assembly)?;
    let lengths = [
        box_high[0] - box_low[0],
        box_high[1] - box_low[1],
        box_high[2] - box_low[2],
    ];
    let fiber_index: HashMap<FiberId, usize> = assembly
        .topology
        .fibers
        .iter()
        .enumerate()
        .map(|(index, fiber)| (fiber.id, index))
        .collect();

    let mut bridges = Vec::new();
    for (junction_index, junction) in assembly.junctions.junctions.iter().enumerate() {
        let anchor_start = junction.anchors.start as usize;
        let anchors = junction
            .anchors
            .checked_end()
            .and_then(|end| assembly.junctions.anchors.get(anchor_start..end as usize))
            .ok_or(ExportError::InvalidJunction(junction.id.0))?;
        let Some((first, others)) = anchors.split_first() else {
            continue;
        };
        let (start, first_radius) = anchor_point(assembly, &fiber_index, *first)?;
        for anchor in others {
            let (point, radius) = anchor_point(assembly, &fiber_index, *anchor)?;
            let mut end = point;
            for axis in 0..3 {
                if assembly.cell.periodic[axis] {
                    let delta = end[axis] - start[axis];
                    end[axis] -= (delta / lengths[axis]).round() * lengths[axis];
                }
            }
            bridges.push(JunctionBridge {
                junction_index,
                junction_id: junction.id.0,
                law: junction.law.0,
                start,
                end,
                radius: radius_ratio * first_radius.min(radius),
            });
        }
    }
    Ok(bridges)
}

/// Placed centerline point of an anchor and the fiber's smallest section
/// half-width.
fn anchor_point(
    assembly: &FiberAssembly,
    fiber_index: &HashMap<FiberId, usize>,
    anchor: FiberAnchor,
) -> Result<(Vec3, f64), ExportError> {
    let invalid = || ExportError::InvalidFiberSpan(anchor.fiber.0);
    let index = *fiber_index.get(&anchor.fiber).ok_or_else(invalid)?;
    let fiber = &assembly.topology.fibers[index];
    let range =
        fiber.vertices.start as usize..fiber.vertices.checked_end().ok_or_else(invalid)? as usize;
    let intrinsic = assembly
        .geometry
        .intrinsic
        .positions
        .get(range.clone())
        .ok_or_else(invalid)?;
    let placed = assembly
        .geometry
        .placed
        .positions
        .get(range)
        .ok_or_else(invalid)?;
    let radius = match assembly.sections.entries.get(fiber.section.0 as usize) {
        Some(Section::Circular { radius }) => *radius,
        Some(Section::Elliptical { semi_axes }) => semi_axes[0].min(semi_axes[1]),
        None => return Err(ExportError::MissingSection(fiber.id.0)),
    };

    // Walk the intrinsic (rest) centerline to the anchor's material point,
    // then evaluate the same segment coordinate on the placed centerline.
    let requested = anchor.rest_arc_length;
    if !requested.is_finite() || requested < 0.0 {
        return Err(invalid());
    }
    let mut traversed = 0.0;
    let mut last = None;
    for (segment, pair) in intrinsic.windows(2).enumerate() {
        let length = crate::distance(pair[0], pair[1]);
        if length <= f64::EPSILON {
            continue;
        }
        last = Some(segment);
        if requested <= traversed + length {
            let coordinate = ((requested - traversed) / length).clamp(0.0, 1.0);
            return Ok((
                lerp(placed[segment], placed[segment + 1], coordinate),
                radius,
            ));
        }
        traversed += length;
    }
    let segment = last.ok_or_else(invalid)?;
    Ok((placed[segment + 1], radius))
}

fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tangle_core::{FiberAnchor, JunctionId, JunctionParameterId, PeriodicCell};

    /// Two round fibers of radius 0.1 crossing at right angles and touching,
    /// optionally joined by one junction at their crossing.
    pub(crate) fn touching_cross(periodic: bool, junction: bool) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic(
            [1.0; 3],
            [periodic, false, false],
        ));
        let material = assembly.materials.add("carbon");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        // Periodic: the fibers touch across the x wall (centers 0.05 and
        // 0.85, 0.2 apart through the wall); otherwise they touch in z.
        let (first, second): ([Vec3; 2], [Vec3; 2]) = if periodic {
            (
                [[0.05, 0.2, 0.5], [0.05, 0.8, 0.5]],
                [[0.85, 0.5, 0.2], [0.85, 0.5, 0.8]],
            )
        } else {
            (
                [[0.2, 0.5, 0.4], [0.8, 0.5, 0.4]],
                [[0.5, 0.2, 0.6], [0.5, 0.8, 0.6]],
            )
        };
        for (id, points) in [(1, first), (2, second)] {
            assembly
                .add_fiber(FiberId(id), material, section, &points, &points)
                .unwrap();
        }
        if junction {
            let law = assembly.junction_laws.add("bond");
            let anchor = |fiber| FiberAnchor {
                fiber: FiberId(fiber),
                rest_arc_length: 0.3,
                section_offset: None,
            };
            assembly
                .add_junction(
                    JunctionId(9),
                    law,
                    JunctionParameterId(0),
                    &[anchor(1), anchor(2)],
                )
                .unwrap();
        }
        assembly
    }

    #[test]
    fn bridge_spans_the_crossing_between_centerlines() {
        let bridges = junction_bridges(&touching_cross(false, true), 0.5).unwrap();
        assert_eq!(bridges.len(), 1);
        let bridge = bridges[0];
        assert_eq!(bridge.junction_id, 9);
        assert!((bridge.radius - 0.05).abs() < 1.0e-12);
        assert!(crate::distance(bridge.start, [0.5, 0.5, 0.4]) < 1.0e-12);
        assert!(crate::distance(bridge.end, [0.5, 0.5, 0.6]) < 1.0e-12);
    }

    #[test]
    fn bridge_across_a_periodic_wall_takes_the_nearest_image() {
        let bridges = junction_bridges(&touching_cross(true, true), 0.5).unwrap();
        assert_eq!(bridges.len(), 1);
        assert!((bridges[0].length() - 0.2).abs() < 1.0e-12);
        assert!(crate::distance(bridges[0].end, [-0.15, 0.5, 0.5]) < 1.0e-12);
    }

    #[test]
    fn rejects_a_nonpositive_radius_ratio() {
        let assembly = touching_cross(false, true);
        assert!(junction_bridges(&assembly, 0.0).is_err());
        assert!(junction_bridges(&assembly, f64::NAN).is_err());
    }
}
