//! Solver-neutral geometric characterization of fiber assemblies.

#![warn(missing_docs)]

use std::f64::consts::PI;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{maximum_polyline_curvature, FiberAssembly, Section, Vec3};

/// Geometric summary of one assembly state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssemblyMetrics {
    /// Number of fibers.
    pub fibers: usize,
    /// Number of piecewise-linear centerline segments.
    pub segments: usize,
    /// Total placed centerline length.
    pub total_centerline_length: f64,
    /// Swept fiber volume divided by cell volume.
    pub solid_volume_fraction: f64,
    /// Length-weighted second-order orientation tensor.
    ///
    /// Each segment contributes `length * tangent tensor tangent`. The trace
    /// is one for every nonempty assembly and is insensitive to fiber sign.
    pub orientation_tensor: [[f64; 3]; 3],
    /// Largest current centerline curvature.
    pub maximum_curvature: f64,
    /// Largest current curvature divided by its fiber's admissible maximum.
    pub maximum_curvature_ratio: f64,
}

/// Initial and final metrics captured around assembly relaxation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssemblyCharacterizationReport {
    /// State produced by the generator before relaxation.
    pub initial: Option<AssemblyMetrics>,
    /// State presented to export after relaxation.
    pub final_state: Option<AssemblyMetrics>,
}

/// Measures volume fraction, orientation, length, and curvature utilization.
pub fn characterize_assembly(assembly: &FiberAssembly) -> AssemblyMetrics {
    let mut metrics = AssemblyMetrics {
        fibers: assembly.topology.fibers.len(),
        ..AssemblyMetrics::default()
    };
    let mut swept_volume = 0.0;

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let start = fiber.vertices.start as usize;
        let Some(end) = fiber.vertices.checked_end().map(|end| end as usize) else {
            continue;
        };
        let Some(points) = assembly.geometry.placed.positions.get(start..end) else {
            continue;
        };
        let area = assembly
            .sections
            .entries
            .get(fiber.section.0 as usize)
            .map(section_area)
            .unwrap_or(0.0);
        for segment in points.windows(2) {
            let delta = sub(segment[1], segment[0]);
            let length = norm(delta);
            if length <= f64::EPSILON {
                continue;
            }
            metrics.segments += 1;
            metrics.total_centerline_length += length;
            swept_volume += area * length;
            let tangent = scale(delta, length.recip());
            for row in 0..3 {
                for column in 0..3 {
                    metrics.orientation_tensor[row][column] +=
                        length * tangent[row] * tangent[column];
                }
            }
        }
        let fiber_maximum_curvature = maximum_polyline_curvature(points);
        metrics.maximum_curvature = metrics.maximum_curvature.max(fiber_maximum_curvature);
        if let Some(limit) = assembly
            .admissibility
            .bend_limits
            .get(fiber_index)
            .copied()
            .flatten()
        {
            metrics.maximum_curvature_ratio = metrics
                .maximum_curvature_ratio
                .max(fiber_maximum_curvature / limit.maximum_curvature());
        }
    }

    if metrics.total_centerline_length > 0.0 {
        for value in metrics.orientation_tensor.iter_mut().flatten() {
            *value /= metrics.total_centerline_length;
        }
    }
    let cell_volume = assembly.cell.signed_volume().abs();
    if cell_volume > f64::EPSILON {
        metrics.solid_volume_fraction = swept_volume / cell_volume;
    }
    metrics
}

/// Captures generator output and the final relaxed state using the standard
/// TANGLE workflow phases.
pub struct AssemblyCharacterizationPlugin;

impl Plugin for AssemblyCharacterizationPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(AssemblyCharacterizationReport::default())
            .add_update_system(
                capture_initial.run_if(in_state(TangleStage::Generate)),
                TanglePhase::Observe,
            )
            .add_update_system(
                capture_final.run_if(in_state(TangleStage::Export)),
                TanglePhase::Export,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_CHARACTERIZATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

fn capture_initial(
    assembly: Res<FiberAssembly>,
    mut report: ResMut<AssemblyCharacterizationReport>,
) {
    if report.initial.is_none() && !assembly.topology.fibers.is_empty() {
        report.initial = Some(characterize_assembly(&assembly));
    }
}

fn capture_final(assembly: Res<FiberAssembly>, mut report: ResMut<AssemblyCharacterizationReport>) {
    report.final_state = Some(characterize_assembly(&assembly));
}

fn section_area(section: &Section) -> f64 {
    match section {
        Section::Circular { radius } => PI * radius * radius,
        Section::Elliptical { semi_axes } => PI * semi_axes[0] * semi_axes[1],
    }
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(value: Vec3, factor: f64) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn norm(value: Vec3) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell};

    #[test]
    fn orientation_tensor_is_length_weighted_and_sign_invariant() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[1.5, 1.0, 1.0], [0.5, 1.0, 1.0]],
            )
            .unwrap();
        let metrics = characterize_assembly(&assembly);
        assert_eq!(metrics.orientation_tensor[0][0], 1.0);
        assert_eq!(metrics.orientation_tensor[1][1], 0.0);
        assert_eq!(metrics.orientation_tensor[2][2], 0.0);
        assert!((metrics.solid_volume_fraction - PI * 0.01 / 8.0).abs() < 1.0e-12);
    }

    #[test]
    fn orthogonal_equal_length_fibers_split_orientation_diagonal() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        for (id, placed) in [
            (1, [[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]]),
            (2, [[1.0, 0.5, 1.0], [1.0, 1.5, 1.0]]),
        ] {
            assembly
                .add_fiber(
                    FiberId(id),
                    material,
                    section,
                    &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    &placed,
                )
                .unwrap();
        }
        let tensor = characterize_assembly(&assembly).orientation_tensor;
        assert!((tensor[0][0] - 0.5).abs() < 1.0e-12);
        assert!((tensor[1][1] - 0.5).abs() < 1.0e-12);
        assert_eq!(tensor[2][2], 0.0);
    }
}
