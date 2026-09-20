use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::math::distance;
use crate::{maximum_polyline_curvature, FiberAssembly, Section};

/// One validation diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationIssue {
    /// Stable machine-readable diagnostic code.
    pub code: &'static str,
    /// Human-readable detail.
    pub message: String,
}

impl ValidationIssue {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// All validation issues found in one pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationReport {
    /// Validation issues in deterministic discovery order.
    pub issues: Vec<ValidationIssue>,
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "fiber assembly has {} validation issue(s):",
            self.issues.len()
        )?;
        for issue in &self.issues {
            writeln!(f, "- {}: {}", issue.code, issue.message)?;
        }
        Ok(())
    }
}

impl Error for ValidationReport {}

impl FiberAssembly {
    /// Checks storage invariants and geometric validity.
    pub fn validate(&self) -> Result<(), ValidationReport> {
        let mut issues = Vec::new();
        self.validate_structure(&mut issues);
        self.validate_geometry(&mut issues);
        if issues.is_empty() {
            Ok(())
        } else {
            Err(ValidationReport { issues })
        }
    }

    fn validate_structure(&self, issues: &mut Vec<ValidationIssue>) {
        let intrinsic_len = self.geometry.intrinsic.positions.len();
        if self.geometry.placed.positions.len() != intrinsic_len {
            issues.push(ValidationIssue::new(
                "geometry.length",
                "intrinsic and placed position arrays have different lengths",
            ));
        }
        if self
            .geometry
            .assembled_reference
            .as_ref()
            .is_some_and(|state| state.positions.len() != intrinsic_len)
        {
            issues.push(ValidationIssue::new(
                "geometry.reference-length",
                "assembled reference and intrinsic position arrays have different lengths",
            ));
        }

        let mut next_vertex = 0_u32;
        let mut fiber_ids = HashSet::new();
        for fiber in &self.topology.fibers {
            if !fiber_ids.insert(fiber.id) {
                issues.push(ValidationIssue::new(
                    "fiber.duplicate-id",
                    format!("fiber ID {:?} appears more than once", fiber.id),
                ));
            }
            if fiber.vertices.start != next_vertex {
                issues.push(ValidationIssue::new(
                    "fiber.non-dense-span",
                    format!(
                        "fiber {:?} does not begin at the next dense vertex",
                        fiber.id
                    ),
                ));
            }
            if fiber.vertices.len < 2 {
                issues.push(ValidationIssue::new(
                    "fiber.too-short",
                    format!("fiber {:?} has fewer than two vertices", fiber.id),
                ));
            }
            match fiber.vertices.checked_end() {
                Some(end) => next_vertex = end,
                None => issues.push(ValidationIssue::new(
                    "fiber.span-overflow",
                    format!("fiber {:?} vertex span overflows u32", fiber.id),
                )),
            }
            if fiber.material.0 as usize >= self.materials.entries.len() {
                issues.push(ValidationIssue::new(
                    "fiber.unknown-material",
                    format!("fiber {:?} references an unknown material", fiber.id),
                ));
            }
            if fiber.section.0 as usize >= self.sections.entries.len() {
                issues.push(ValidationIssue::new(
                    "fiber.unknown-section",
                    format!("fiber {:?} references an unknown section", fiber.id),
                ));
            }
        }
        if next_vertex as usize != intrinsic_len {
            issues.push(ValidationIssue::new(
                "fiber.unclaimed-vertices",
                "fiber spans do not densely cover the geometry arrays",
            ));
        }
        if self.admissibility.bend_limits.len() != self.topology.fibers.len() {
            issues.push(ValidationIssue::new(
                "admissibility.length",
                "bend-limit table length differs from the fiber topology length",
            ));
        }

        let mut next_anchor = 0_u32;
        let mut junction_ids = HashSet::new();
        for junction in &self.junctions.junctions {
            if !junction_ids.insert(junction.id) {
                issues.push(ValidationIssue::new(
                    "junction.duplicate-id",
                    format!("junction ID {:?} appears more than once", junction.id),
                ));
            }
            if junction.anchors.start != next_anchor || junction.anchors.len < 2 {
                issues.push(ValidationIssue::new(
                    "junction.invalid-span",
                    format!("junction {:?} has an invalid anchor span", junction.id),
                ));
            }
            match junction.anchors.checked_end() {
                Some(end) => next_anchor = end,
                None => issues.push(ValidationIssue::new(
                    "junction.span-overflow",
                    format!("junction {:?} anchor span overflows u32", junction.id),
                )),
            }
            if junction.law.0 as usize >= self.junction_laws.entries.len() {
                issues.push(ValidationIssue::new(
                    "junction.unknown-law",
                    format!("junction {:?} references an unknown law", junction.id),
                ));
            }
        }
        if next_anchor as usize != self.junctions.anchors.len() {
            issues.push(ValidationIssue::new(
                "junction.unclaimed-anchors",
                "junction spans do not densely cover the anchor table",
            ));
        }
    }

    fn validate_geometry(&self, issues: &mut Vec<ValidationIssue>) {
        let cell_volume = self.cell.signed_volume();
        if !self.cell.origin.iter().all(|value| value.is_finite())
            || !self
                .cell
                .basis
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            || !cell_volume.is_finite()
            || cell_volume.abs() <= f64::EPSILON
        {
            issues.push(ValidationIssue::new(
                "cell.degenerate",
                "cell coordinates must be finite and span a nonzero volume",
            ));
        }

        for (index, section) in self.sections.entries.iter().enumerate() {
            let valid = match section {
                Section::Circular { radius } => radius.is_finite() && *radius > 0.0,
                Section::Elliptical { semi_axes } => semi_axes
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
            };
            if !valid {
                issues.push(ValidationIssue::new(
                    "section.invalid-size",
                    format!("section {index} has a non-positive or non-finite size"),
                ));
            }
        }

        for (fiber_index, fiber) in self.topology.fibers.iter().enumerate() {
            let Some(range) = fiber.vertices.as_usize_range() else {
                continue;
            };
            let Some(points) = self.geometry.intrinsic.positions.get(range.clone()) else {
                continue;
            };
            if points.iter().flatten().any(|value| !value.is_finite()) {
                issues.push(ValidationIssue::new(
                    "fiber.non-finite-intrinsic",
                    format!("fiber {:?} has a non-finite intrinsic position", fiber.id),
                ));
            }
            if points
                .windows(2)
                .any(|pair| distance(pair[0], pair[1]) <= f64::EPSILON)
            {
                issues.push(ValidationIssue::new(
                    "fiber.zero-length-segment",
                    format!("fiber {:?} has a zero-length intrinsic segment", fiber.id),
                ));
            }
            if let Some(limit) = self
                .admissibility
                .bend_limits
                .get(fiber_index)
                .copied()
                .flatten()
            {
                if !limit.minimum_bend_radius.is_finite() || limit.minimum_bend_radius <= 0.0 {
                    issues.push(ValidationIssue::new(
                        "fiber.invalid-bend-limit",
                        format!("fiber {:?} has an invalid minimum bend radius", fiber.id),
                    ));
                } else {
                    let intrinsic_maximum = maximum_polyline_curvature(points);
                    if intrinsic_maximum > limit.maximum_curvature() * (1.0 + 1.0e-10) {
                        issues.push(ValidationIssue::new(
                            "fiber.intrinsic-bend-limit",
                            format!(
                                "fiber {:?} intrinsic curvature {intrinsic_maximum:.6e} exceeds its maximum {:.6e}",
                                fiber.id,
                                limit.maximum_curvature()
                            ),
                        ));
                    }
                }
            }
            if let Some(placed) = self.geometry.placed.positions.get(range) {
                if placed.iter().flatten().any(|value| !value.is_finite()) {
                    issues.push(ValidationIssue::new(
                        "fiber.non-finite-placed",
                        format!("fiber {:?} has a non-finite placed position", fiber.id),
                    ));
                }
                if placed
                    .windows(2)
                    .any(|pair| distance(pair[0], pair[1]) <= f64::EPSILON)
                {
                    issues.push(ValidationIssue::new(
                        "fiber.collapsed-segment",
                        format!("fiber {:?} has a collapsed placed segment", fiber.id),
                    ));
                }
            }
        }

        for (index, anchor) in self.junctions.anchors.iter().copied().enumerate() {
            if !anchor.rest_arc_length.is_finite() || anchor.rest_arc_length < 0.0 {
                issues.push(ValidationIssue::new(
                    "anchor.invalid-coordinate",
                    format!("anchor {index} has an invalid rest arc length"),
                ));
                continue;
            }
            if anchor
                .section_offset
                .is_some_and(|offset| offset.iter().any(|value| !value.is_finite()))
            {
                issues.push(ValidationIssue::new(
                    "anchor.invalid-offset",
                    format!("anchor {index} has a non-finite section offset"),
                ));
            }
            if let Err(error) = self.resolve_anchor(anchor) {
                issues.push(ValidationIssue::new(
                    "anchor.unresolvable",
                    format!("anchor {index} cannot be resolved: {error}"),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{FiberAssembly, FiberId, PeriodicCell, Section};

    #[test]
    fn valid_assembly_passes_geometric_validation() {
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

        assembly.validate().unwrap();
    }
}
