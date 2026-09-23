//! Solver-neutral geometric characterization of fiber assemblies.

#![warn(missing_docs)]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
mod distribution;
mod entanglement;
mod graph;
mod neighbors;
mod phases;
mod scorecard;
mod shape;
mod slices;

pub use distribution::Distribution;
pub use entanglement::{
    analyze_entanglement, EntanglementConfig, EntanglementError, EntanglementMetrics,
    FiberEntanglementMetrics, ENTANGLEMENT_SCHEMA_VERSION,
};
pub use graph::{analyze_contact_graph, ContactGraphMetrics, CONTACT_GRAPH_SCHEMA_VERSION};
pub use neighbors::{
    analyze_neighbors, ContactEvent, FiberNeighborMetrics, NeighborAnalysisConfig,
    NeighborAnalysisError, NeighborMetrics, NEIGHBOR_SCHEMA_VERSION,
};
pub use phases::{
    analyze_phases, PhaseAnalysisConfig, PhaseAnalysisError, PhaseMetrics, PHASE_SCHEMA_VERSION,
};
pub use scorecard::{
    crop_assembly, profile_structure, score_structure, ScoreRow, Scorecard, ScorecardConfig,
    ScorecardError, StructureProfile, SCORECARD_SCHEMA_VERSION,
};
pub use shape::{
    analyze_shape, FiberShapeMetrics, ShapeAnalysisConfig, ShapeAnalysisError, ShapeMetrics,
    SHAPE_SCHEMA_VERSION,
};
pub use slices::{
    analyze_slices, SliceAnalysisConfig, SliceAnalysisError, SliceMetrics, SLICE_SCHEMA_VERSION,
};

use tangle_core::{
    maximum_polyline_curvature, FiberAssembly, FiberId, MaterialId, Section, SectionId, Vec3,
};

/// Schema version written by [`write_analysis_json`].
pub const ANALYSIS_SCHEMA_VERSION: u32 = 1;

/// Cell metadata included with every characterization report.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CellMetrics {
    /// World-space cell origin.
    pub origin: Vec3,
    /// Cell edge vectors.
    pub basis: [Vec3; 3],
    /// Length of each cell edge vector.
    pub lengths: Vec3,
    /// Per-axis periodicity.
    pub periodic: [bool; 3],
    /// Absolute cell volume.
    pub volume: f64,
}

/// Exact centerline-derived metrics for one source fiber.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberMetrics {
    /// Stable source fiber identifier.
    pub fiber_id: FiberId,
    /// Material identifier referenced by the fiber.
    pub material_id: MaterialId,
    /// Cross-section identifier referenced by the fiber.
    pub section_id: SectionId,
    /// Optional manufacturing layer.
    pub formation_layer: Option<u32>,
    /// Manufacturing step at which the fiber becomes active.
    pub formation_step: u32,
    /// Current number of piecewise-linear segments.
    pub segments: usize,
    /// Current number of centerline vertices.
    pub vertices: usize,
    /// Intrinsic isolated-fiber centerline length.
    pub intrinsic_length: f64,
    /// Current placed centerline length.
    pub placed_length: f64,
    /// Placed length divided by intrinsic length.
    pub stretch_ratio: f64,
    /// Cross-sectional area.
    pub section_area: f64,
    /// Diameter of a circle with the same cross-sectional area.
    pub equivalent_diameter: f64,
    /// Nominal swept volume, section area times placed centerline length.
    pub nominal_swept_volume: f64,
    /// Maximum current polyline curvature.
    pub maximum_curvature: f64,
    /// Maximum curvature divided by the configured admissible curvature.
    pub maximum_curvature_ratio: Option<f64>,
}

/// Centerline-derived metrics accumulated for one material.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaterialMetrics {
    /// Dense material identifier.
    pub material_id: MaterialId,
    /// Human-readable material name.
    pub material_name: String,
    /// Number of fibers assigned to this material.
    pub fibers: usize,
    /// Number of placed centerline segments assigned to this material.
    pub segments: usize,
    /// Total placed centerline length.
    pub total_centerline_length: f64,
    /// Nominal swept volume.
    pub nominal_swept_volume: f64,
    /// Nominal swept volume divided by cell volume.
    pub nominal_swept_volume_fraction: f64,
    /// Length-weighted second-order orientation tensor.
    pub length_weighted_orientation_tensor: [[f64; 3]; 3],
    /// Nominal-volume-weighted second-order orientation tensor.
    pub volume_weighted_orientation_tensor: [[f64; 3]; 3],
}

impl Default for MaterialMetrics {
    fn default() -> Self {
        Self {
            material_id: MaterialId(0),
            material_name: String::new(),
            fibers: 0,
            segments: 0,
            total_centerline_length: 0.0,
            nominal_swept_volume: 0.0,
            nominal_swept_volume_fraction: 0.0,
            length_weighted_orientation_tensor: [[0.0; 3]; 3],
            volume_weighted_orientation_tensor: [[0.0; 3]; 3],
        }
    }
}

/// Geometric summary of one assembly state.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssemblyMetrics {
    /// Version of the serialized analysis schema.
    pub schema_version: u32,
    /// Cell metadata used to normalize dimensional quantities.
    pub cell: CellMetrics,
    /// Number of fibers.
    pub fibers: usize,
    /// Number of piecewise-linear centerline segments.
    pub segments: usize,
    /// Number of centerline vertices.
    pub vertices: usize,
    /// Number of persistent junction objects.
    pub junctions: usize,
    /// Total intrinsic isolated-fiber centerline length.
    pub total_intrinsic_length: f64,
    /// Total placed centerline length.
    pub total_centerline_length: f64,
    /// Nominal swept volume, section area times placed centerline length.
    pub nominal_swept_volume: f64,
    /// Nominal swept fiber volume divided by cell volume.
    ///
    /// This compatibility name predates the PuMA interoperability work. It is
    /// the same value as [`Self::nominal_swept_volume_fraction`], not a union
    /// volume of the swept fiber surfaces.
    pub solid_volume_fraction: f64,
    /// Nominal swept fiber volume divided by cell volume.
    pub nominal_swept_volume_fraction: f64,
    /// Length-weighted second-order orientation tensor.
    ///
    /// Each segment contributes `length * tangent tensor tangent`. The trace
    /// is one for every nonempty assembly and is insensitive to fiber sign.
    pub orientation_tensor: [[f64; 3]; 3],
    /// Nominal-volume-weighted second-order orientation tensor.
    ///
    /// This is the appropriate native comparison for an occupied-voxel
    /// orientation tensor when fiber cross-sections differ.
    pub volume_weighted_orientation_tensor: [[f64; 3]; 3],
    /// Largest current centerline curvature.
    pub maximum_curvature: f64,
    /// Largest current curvature divided by its fiber's admissible maximum.
    pub maximum_curvature_ratio: f64,
    /// Number of fibers whose current curvature exceeds the hard bend limit.
    pub bend_limit_violations: usize,
    /// Per-material metrics in dense material identifier order.
    pub materials: Vec<MaterialMetrics>,
    /// Per-fiber metrics in topology order.
    pub fiber_metrics: Vec<FiberMetrics>,
}

/// Initial and final metrics captured around assembly relaxation.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssemblyCharacterizationReport {
    /// State produced by the generator before relaxation.
    pub initial: Option<AssemblyMetrics>,
    /// State presented to export after relaxation.
    pub final_state: Option<AssemblyMetrics>,
}

/// Measures volume fraction, orientation, length, and curvature utilization.
pub fn characterize_assembly(assembly: &FiberAssembly) -> AssemblyMetrics {
    let cell_volume = assembly.cell.signed_volume().abs();
    let mut metrics = AssemblyMetrics {
        schema_version: ANALYSIS_SCHEMA_VERSION,
        cell: CellMetrics {
            origin: assembly.cell.origin,
            basis: assembly.cell.basis,
            lengths: assembly.cell.basis.map(norm),
            periodic: assembly.cell.periodic,
            volume: cell_volume,
        },
        fibers: assembly.topology.fibers.len(),
        junctions: assembly.junctions.junctions.len(),
        materials: assembly
            .materials
            .entries
            .iter()
            .enumerate()
            .map(|(index, material)| MaterialMetrics {
                material_id: MaterialId(index as u32),
                material_name: material.name.clone(),
                ..MaterialMetrics::default()
            })
            .collect(),
        ..AssemblyMetrics::default()
    };
    let mut volume_orientation_weight = 0.0;

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let start = fiber.vertices.start as usize;
        let Some(end) = fiber.vertices.checked_end().map(|end| end as usize) else {
            continue;
        };
        let Some(points) = assembly.geometry.placed.positions.get(start..end) else {
            continue;
        };
        let intrinsic = assembly
            .geometry
            .intrinsic
            .positions
            .get(start..end)
            .unwrap_or(&[]);
        let section = assembly
            .sections
            .entries
            .get(fiber.section.0 as usize)
            .copied();
        let area = section.as_ref().map(section_area).unwrap_or(0.0);
        let intrinsic_length = polyline_length(intrinsic);
        let mut placed_length = 0.0;
        let mut nominal_swept_volume = 0.0;
        for segment in points.windows(2) {
            let delta = sub(segment[1], segment[0]);
            let length = norm(delta);
            if length <= f64::EPSILON {
                continue;
            }
            metrics.segments += 1;
            metrics.total_centerline_length += length;
            placed_length += length;
            let segment_volume = area * length;
            nominal_swept_volume += segment_volume;
            metrics.nominal_swept_volume += segment_volume;
            volume_orientation_weight += segment_volume;
            let tangent = scale(delta, length.recip());
            accumulate_orientation(&mut metrics.orientation_tensor, tangent, length);
            accumulate_orientation(
                &mut metrics.volume_weighted_orientation_tensor,
                tangent,
                segment_volume,
            );

            if let Some(material) = metrics.materials.get_mut(fiber.material.0 as usize) {
                material.segments += 1;
                material.total_centerline_length += length;
                material.nominal_swept_volume += segment_volume;
                accumulate_orientation(
                    &mut material.length_weighted_orientation_tensor,
                    tangent,
                    length,
                );
                accumulate_orientation(
                    &mut material.volume_weighted_orientation_tensor,
                    tangent,
                    segment_volume,
                );
            }
        }
        metrics.vertices += points.len();
        metrics.total_intrinsic_length += intrinsic_length;
        let fiber_maximum_curvature = maximum_polyline_curvature(points);
        metrics.maximum_curvature = metrics.maximum_curvature.max(fiber_maximum_curvature);
        let fiber_curvature_ratio = assembly
            .admissibility
            .bend_limits
            .get(fiber_index)
            .copied()
            .flatten()
            .map(|limit| fiber_maximum_curvature / limit.maximum_curvature());
        if let Some(ratio) = fiber_curvature_ratio {
            metrics.maximum_curvature_ratio = metrics.maximum_curvature_ratio.max(ratio);
            if ratio > 1.0 {
                metrics.bend_limit_violations += 1;
            }
        }
        if let Some(material) = metrics.materials.get_mut(fiber.material.0 as usize) {
            material.fibers += 1;
        }
        metrics.fiber_metrics.push(FiberMetrics {
            fiber_id: fiber.id,
            material_id: fiber.material,
            section_id: fiber.section,
            formation_layer: fiber.formation_layer,
            formation_step: fiber.formation_step,
            segments: points.len().saturating_sub(1),
            vertices: points.len(),
            intrinsic_length,
            placed_length,
            stretch_ratio: if intrinsic_length > f64::EPSILON {
                placed_length / intrinsic_length
            } else {
                0.0
            },
            section_area: area,
            equivalent_diameter: (4.0 * area / PI).sqrt(),
            nominal_swept_volume,
            maximum_curvature: fiber_maximum_curvature,
            maximum_curvature_ratio: fiber_curvature_ratio,
        });
    }

    normalize_orientation(
        &mut metrics.orientation_tensor,
        metrics.total_centerline_length,
    );
    normalize_orientation(
        &mut metrics.volume_weighted_orientation_tensor,
        volume_orientation_weight,
    );
    if cell_volume > f64::EPSILON {
        metrics.nominal_swept_volume_fraction = metrics.nominal_swept_volume / cell_volume;
        metrics.solid_volume_fraction = metrics.nominal_swept_volume_fraction;
    }
    for material in &mut metrics.materials {
        normalize_orientation(
            &mut material.length_weighted_orientation_tensor,
            material.total_centerline_length,
        );
        normalize_orientation(
            &mut material.volume_weighted_orientation_tensor,
            material.nominal_swept_volume,
        );
        if cell_volume > f64::EPSILON {
            material.nominal_swept_volume_fraction = material.nominal_swept_volume / cell_volume;
        }
    }
    metrics
}

/// Writes a native characterization report as stable, pretty-printed JSON.
pub fn write_analysis_json(
    metrics: &AssemblyMetrics,
    path: impl AsRef<Path>,
) -> Result<(), AnalysisWriteError> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, metrics)?;
    Ok(())
}

/// Failure while serializing a native characterization report.
#[derive(Debug)]
pub enum AnalysisWriteError {
    /// Filesystem failure.
    Io(std::io::Error),
    /// JSON serialization failure.
    Json(serde_json::Error),
}

impl fmt::Display for AnalysisWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "analysis output failed: {error}"),
            Self::Json(error) => write!(formatter, "analysis serialization failed: {error}"),
        }
    }
}

impl Error for AnalysisWriteError {}

impl From<std::io::Error> for AnalysisWriteError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for AnalysisWriteError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
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

fn polyline_length(points: &[Vec3]) -> f64 {
    points
        .windows(2)
        .map(|segment| norm(sub(segment[1], segment[0])))
        .sum()
}

fn accumulate_orientation(tensor: &mut [[f64; 3]; 3], tangent: Vec3, weight: f64) {
    for row in 0..3 {
        for column in 0..3 {
            tensor[row][column] += weight * tangent[row] * tangent[column];
        }
    }
}

fn normalize_orientation(tensor: &mut [[f64; 3]; 3], weight: f64) {
    if weight > f64::EPSILON {
        for value in tensor.iter_mut().flatten() {
            *value /= weight;
        }
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

    #[test]
    fn volume_weighting_and_material_reports_preserve_diameter_effects() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let thick = assembly.materials.add("thick");
        let thin = assembly.materials.add("thin");
        let thick_section = assembly.sections.add(Section::Circular { radius: 0.2 });
        let thin_section = assembly.sections.add(Section::Circular { radius: 0.1 });
        assembly
            .add_fiber(
                FiberId(1),
                thick,
                thick_section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]],
            )
            .unwrap();
        assembly
            .add_fiber(
                FiberId(2),
                thin,
                thin_section,
                &[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                &[[1.0, 0.5, 1.0], [1.0, 1.5, 1.0]],
            )
            .unwrap();

        let metrics = characterize_assembly(&assembly);
        assert_eq!(metrics.schema_version, ANALYSIS_SCHEMA_VERSION);
        assert_eq!(metrics.vertices, 4);
        assert!((metrics.orientation_tensor[0][0] - 0.5).abs() < 1.0e-12);
        assert!((metrics.orientation_tensor[1][1] - 0.5).abs() < 1.0e-12);
        assert!((metrics.volume_weighted_orientation_tensor[0][0] - 0.8).abs() < 1.0e-12);
        assert!((metrics.volume_weighted_orientation_tensor[1][1] - 0.2).abs() < 1.0e-12);
        assert_eq!(metrics.materials[0].material_name, "thick");
        assert_eq!(metrics.materials[0].fibers, 1);
        assert_eq!(metrics.materials[1].material_name, "thin");
        assert_eq!(metrics.fiber_metrics[0].fiber_id, FiberId(1));
        assert!((metrics.fiber_metrics[0].equivalent_diameter - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn json_uses_the_explicit_nominal_volume_fraction_name() {
        let assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let json = serde_json::to_value(characterize_assembly(&assembly)).unwrap();
        assert_eq!(json["schema_version"], ANALYSIS_SCHEMA_VERSION);
        assert!(json.get("nominal_swept_volume_fraction").is_some());
        assert!(json.get("cell").is_some());
        assert!(json.get("materials").is_some());
    }
}
