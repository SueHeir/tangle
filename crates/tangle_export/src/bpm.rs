//! Unified bonded-particle discretization and export.

use std::ops::Range;
use std::path::Path;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, MaterialId, Section, Vec3};

use crate::{
    add, distance, nearest_anchor_capsule, nearest_anchor_particle, orthorhombic_bounds, scale,
    sub, write_capsule_lammps_data, write_lammps_data, BpmExportConfig, BpmExportMode, BpmModel,
    DemAtomType, DemBond, DemBpmModel, DemCapsule, DemCapsuleBpmExportConfig, DemCapsuleBpmModel,
    DemParticle, ExportError,
};

/// Summary of a completed solver-neutral BPM export.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BpmExportReport {
    /// Geometry representation and sampling policy.
    pub mode: BpmExportMode,
    /// Number of written sphere or spherocylinder particles.
    pub particles: usize,
    /// Number of written bonds.
    pub bonds: usize,
    /// Inter-fiber bonds derived from persistent junctions.
    pub junction_bonds: usize,
    /// Material-to-atom-type mappings written to the data file.
    pub atom_types: Vec<DemAtomType>,
    /// Written data file.
    pub data_path: std::path::PathBuf,
}

/// GRASS plugin that discretizes the final assembly and writes a BPM model.
pub struct BpmExportPlugin {
    /// Export configuration.
    pub config: BpmExportConfig,
}

impl Plugin for BpmExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(self.config.clone())
            .add_resource(BpmExportReport::default())
            .add_update_system(
                export_bpm_system.run_if(in_state(TangleStage::Export)),
                TanglePhase::Export,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_EXPORT.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone(), TANGLE_ASSEMBLY.clone()]
    }
}

/// Builds one of the four supported BPM centerline discretizations.
pub fn build_bpm_model(
    assembly: &FiberAssembly,
    config: &BpmExportConfig,
) -> Result<BpmModel, ExportError> {
    validate_common(config)?;
    match config.mode {
        BpmExportMode::SpheresExact | BpmExportMode::SpheresDynamic => {
            validate_sphere_spacing(config.sphere_spacing_over_radius)?;
            build_sphere_model(assembly, config).map(BpmModel::Spheres)
        }
        BpmExportMode::SpherocylindersExact => {
            let legacy = DemCapsuleBpmExportConfig {
                data_path: config.data_path.clone(),
                density: config.density,
                atom_type: config.atom_type,
                bond_type: config.bond_type,
                maximum_length_over_diameter: None,
            };
            crate::build_dem_capsule_bpm_model(assembly, &legacy).map(BpmModel::Spherocylinders)
        }
        BpmExportMode::SpherocylindersConstant => {
            build_constant_spherocylinder_model(assembly, config).map(BpmModel::Spherocylinders)
        }
    }
}

/// Writes a BPM model using the LAMMPS sphere or extended spherocylinder data
/// representation selected during discretization.
pub fn write_bpm_lammps_data(model: &BpmModel, path: &Path) -> Result<(), ExportError> {
    match model {
        BpmModel::Spheres(model) => write_lammps_data(model, path),
        BpmModel::Spherocylinders(model) => write_capsule_lammps_data(model, path),
    }
}

fn validate_common(config: &BpmExportConfig) -> Result<(), ExportError> {
    if !config.density.is_finite() || config.density <= 0.0 {
        return Err(ExportError::InvalidDensity(config.density));
    }
    if config.atom_type == 0 {
        return Err(ExportError::InvalidAtomType(config.atom_type));
    }
    Ok(())
}

fn validate_sphere_spacing(spacing: f64) -> Result<(), ExportError> {
    if !spacing.is_finite() || !(1.0 / 3.0..=1.0).contains(&spacing) {
        return Err(ExportError::InvalidSphereSpacingOverRadius(spacing));
    }
    Ok(())
}

fn build_sphere_model(
    assembly: &FiberAssembly,
    config: &BpmExportConfig,
) -> Result<DemBpmModel, ExportError> {
    let (box_low, box_high) = orthorhombic_bounds(assembly)?;
    let mut model = DemBpmModel {
        box_low,
        box_high,
        atom_types: atom_types(assembly, config.atom_type)?,
        ..DemBpmModel::default()
    };
    let mut fiber_particle_ranges = Vec::with_capacity(assembly.topology.fibers.len());

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let atom_type = model
            .atom_types
            .get(fiber.material.0 as usize)
            .ok_or(ExportError::MissingMaterial(fiber.id.0))?
            .atom_type;
        let radius = circular_radius(assembly, fiber.section.0 as usize, fiber.id.0)?;
        let points = fiber_points(assembly, fiber_index)?;
        let spacing = radius * config.sphere_spacing_over_radius;
        let samples = match config.mode {
            BpmExportMode::SpheresExact => exact_sphere_samples(points, spacing, fiber.id.0)?,
            BpmExportMode::SpheresDynamic => dynamic_sphere_samples(points, spacing, fiber.id.0)?,
            _ => unreachable!("sphere builder is only called for sphere modes"),
        };
        let first_particle = model.particles.len();
        for position in samples {
            let id =
                u32::try_from(model.particles.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
            model.particles.push(DemParticle {
                id,
                molecule: u32::try_from(fiber_index + 1).map_err(|_| ExportError::IndexOverflow)?,
                atom_type,
                diameter: 2.0 * radius,
                density: config.density,
                position,
            });
        }
        model.atom_types[fiber.material.0 as usize].particles +=
            model.particles.len() - first_particle;
        append_consecutive_bonds(
            &mut model.bonds,
            config.bond_type,
            model.particles[first_particle..]
                .iter()
                .map(|particle| particle.id),
        )?;
        fiber_particle_ranges.push(first_particle..model.particles.len());
    }

    append_sphere_junction_bonds(assembly, &mut model, &fiber_particle_ranges)?;
    wrap_sphere_centers(assembly, &mut model);
    Ok(model)
}

fn exact_sphere_samples(
    points: &[Vec3],
    spacing: f64,
    fiber_id: u32,
) -> Result<Vec<Vec3>, ExportError> {
    let total_length = polyline_length(points);
    if total_length <= f64::EPSILON {
        return Err(ExportError::InvalidFiberSpan(fiber_id));
    }
    let interval_ratio = total_length / spacing;
    let nearest_integer = interval_ratio.round();
    let intervals =
        if (interval_ratio - nearest_integer).abs() <= 1.0e-12 * interval_ratio.abs().max(1.0) {
            nearest_integer as usize
        } else {
            interval_ratio.floor() as usize
        };
    if intervals == 0 {
        return Ok(vec![crate::sample_polyline(points, 0.5 * total_length)]);
    }
    let unused = (total_length - intervals as f64 * spacing).max(0.0);
    let offset = 0.5 * unused;
    Ok((0..=intervals)
        .map(|index| crate::sample_polyline(points, offset + index as f64 * spacing))
        .collect())
}

fn dynamic_sphere_samples(
    points: &[Vec3],
    spacing: f64,
    fiber_id: u32,
) -> Result<Vec<Vec3>, ExportError> {
    let mut samples = Vec::new();
    let mut found_segment = false;
    samples.push(points[0]);
    for pair in points.windows(2) {
        let length = distance(pair[0], pair[1]);
        if length <= f64::EPSILON {
            continue;
        }
        found_segment = true;
        let intervals = (length / spacing).round().max(1.0) as usize;
        for index in 1..=intervals {
            let coordinate = index as f64 / intervals as f64;
            samples.push(add(pair[0], scale(sub(pair[1], pair[0]), coordinate)));
        }
    }
    if !found_segment {
        return Err(ExportError::InvalidFiberSpan(fiber_id));
    }
    Ok(samples)
}

fn build_constant_spherocylinder_model(
    assembly: &FiberAssembly,
    config: &BpmExportConfig,
) -> Result<DemCapsuleBpmModel, ExportError> {
    let target_length = shortest_active_segment(assembly)?;
    let (box_low, box_high) = orthorhombic_bounds(assembly)?;
    let mut model = DemCapsuleBpmModel {
        box_low,
        box_high,
        atom_types: atom_types(assembly, config.atom_type)?,
        ..DemCapsuleBpmModel::default()
    };
    let mut fiber_capsule_ranges = Vec::with_capacity(assembly.topology.fibers.len());

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let atom_type = model
            .atom_types
            .get(fiber.material.0 as usize)
            .ok_or(ExportError::MissingMaterial(fiber.id.0))?
            .atom_type;
        let radius = circular_radius(assembly, fiber.section.0 as usize, fiber.id.0)?;
        let points = fiber_points(assembly, fiber_index)?;
        let first_capsule = model.capsules.len();
        for pair in points.windows(2) {
            append_constant_capsules(
                &mut model.capsules,
                pair[0],
                pair[1],
                target_length,
                fiber_index,
                atom_type,
                radius,
                config.density,
            )?;
        }
        if model.capsules.len() == first_capsule {
            return Err(ExportError::InvalidFiberSpan(fiber.id.0));
        }
        model.atom_types[fiber.material.0 as usize].particles +=
            model.capsules.len() - first_capsule;
        append_consecutive_bonds(
            &mut model.bonds,
            config.bond_type,
            model.capsules[first_capsule..]
                .iter()
                .map(|capsule| capsule.id),
        )?;
        fiber_capsule_ranges.push(first_capsule..model.capsules.len());
    }

    append_capsule_junction_bonds(assembly, &mut model, &fiber_capsule_ranges)?;
    wrap_capsule_centers(assembly, &mut model);
    Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn append_constant_capsules(
    capsules: &mut Vec<DemCapsule>,
    start: Vec3,
    end: Vec3,
    target_length: f64,
    fiber_index: usize,
    atom_type: u32,
    radius: f64,
    density: f64,
) -> Result<(), ExportError> {
    let delta = sub(end, start);
    let segment_length = distance(start, end);
    if segment_length <= f64::EPSILON {
        return Ok(());
    }
    let axis = scale(delta, 1.0 / segment_length);
    let tolerance = target_length * 1.0e-10;
    let mut consumed = 0.0;
    while segment_length - consumed > tolerance {
        let remaining = segment_length - consumed;
        let piece_length = remaining.min(target_length);
        let center_distance = consumed + 0.5 * piece_length;
        let id = u32::try_from(capsules.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
        capsules.push(DemCapsule {
            id,
            molecule: u32::try_from(fiber_index + 1).map_err(|_| ExportError::IndexOverflow)?,
            atom_type,
            diameter: 2.0 * radius,
            density,
            position: add(start, scale(axis, center_distance)),
            half_length: 0.5 * piece_length,
            axis,
        });
        consumed += piece_length;
    }
    // Absorb only roundoff-sized residuals so the last capsule ends exactly at
    // the source vertex without introducing a microscopic extra particle.
    let residual = segment_length - consumed;
    if residual.abs() > 0.0 {
        if let Some(last) = capsules.last_mut() {
            last.half_length += 0.5 * residual;
            last.position = add(last.position, scale(axis, 0.5 * residual));
        }
    }
    Ok(())
}

fn shortest_active_segment(assembly: &FiberAssembly) -> Result<f64, ExportError> {
    let mut shortest = f64::INFINITY;
    for fiber_index in 0..assembly.topology.fibers.len() {
        for pair in fiber_points(assembly, fiber_index)?.windows(2) {
            let length = distance(pair[0], pair[1]);
            if length > f64::EPSILON {
                shortest = shortest.min(length);
            }
        }
    }
    if shortest.is_finite() {
        Ok(shortest)
    } else {
        Err(ExportError::NoActiveSegments)
    }
}

fn atom_types(
    assembly: &FiberAssembly,
    first_atom_type: u32,
) -> Result<Vec<DemAtomType>, ExportError> {
    assembly
        .materials
        .entries
        .iter()
        .enumerate()
        .map(|(material, descriptor)| {
            let material = u32::try_from(material).map_err(|_| ExportError::IndexOverflow)?;
            Ok(DemAtomType {
                atom_type: first_atom_type
                    .checked_add(material)
                    .ok_or(ExportError::IndexOverflow)?,
                material: MaterialId(material),
                material_name: descriptor.name.clone(),
                particles: 0,
            })
        })
        .collect()
}

fn circular_radius(
    assembly: &FiberAssembly,
    section_index: usize,
    fiber_id: u32,
) -> Result<f64, ExportError> {
    match assembly
        .sections
        .entries
        .get(section_index)
        .ok_or(ExportError::MissingSection(fiber_id))?
    {
        Section::Circular { radius } => Ok(*radius),
        Section::Elliptical { .. } => Err(ExportError::NonCircularSection(fiber_id)),
    }
}

fn fiber_points(assembly: &FiberAssembly, fiber_index: usize) -> Result<&[Vec3], ExportError> {
    let fiber = &assembly.topology.fibers[fiber_index];
    let start = fiber.vertices.start as usize;
    let end = fiber
        .vertices
        .checked_end()
        .ok_or(ExportError::InvalidFiberSpan(fiber.id.0))? as usize;
    let points = assembly
        .geometry
        .placed
        .positions
        .get(start..end)
        .ok_or(ExportError::InvalidFiberSpan(fiber.id.0))?;
    if points.len() < 2 {
        return Err(ExportError::InvalidFiberSpan(fiber.id.0));
    }
    Ok(points)
}

fn polyline_length(points: &[Vec3]) -> f64 {
    points
        .windows(2)
        .map(|pair| distance(pair[0], pair[1]))
        .sum()
}

fn append_consecutive_bonds(
    bonds: &mut Vec<DemBond>,
    bond_type: u32,
    particle_ids: impl Iterator<Item = u32>,
) -> Result<(), ExportError> {
    let mut previous = None;
    for particle in particle_ids {
        if let Some(previous) = previous {
            bonds.push(DemBond {
                id: u32::try_from(bonds.len() + 1).map_err(|_| ExportError::IndexOverflow)?,
                bond_type,
                particle_a: previous,
                particle_b: particle,
            });
        }
        previous = Some(particle);
    }
    Ok(())
}

fn append_sphere_junction_bonds(
    assembly: &FiberAssembly,
    model: &mut DemBpmModel,
    ranges: &[Range<usize>],
) -> Result<(), ExportError> {
    for junction in &assembly.junctions.junctions {
        let start = junction.anchors.start as usize;
        let end = start + junction.anchors.len as usize;
        let anchors = &assembly.junctions.anchors[start..end];
        let first = nearest_anchor_particle(assembly, model, ranges, anchors[0])?;
        for anchor in &anchors[1..] {
            let other = nearest_anchor_particle(assembly, model, ranges, *anchor)?;
            if first != other {
                model.bonds.push(DemBond {
                    id: u32::try_from(model.bonds.len() + 1)
                        .map_err(|_| ExportError::IndexOverflow)?,
                    bond_type: junction.law.0.saturating_add(2),
                    particle_a: first,
                    particle_b: other,
                });
            }
        }
    }
    Ok(())
}

fn append_capsule_junction_bonds(
    assembly: &FiberAssembly,
    model: &mut DemCapsuleBpmModel,
    ranges: &[Range<usize>],
) -> Result<(), ExportError> {
    for junction in &assembly.junctions.junctions {
        let start = junction.anchors.start as usize;
        let end = start + junction.anchors.len as usize;
        let anchors = &assembly.junctions.anchors[start..end];
        let first = nearest_anchor_capsule(assembly, model, ranges, anchors[0])?;
        for anchor in &anchors[1..] {
            let other = nearest_anchor_capsule(assembly, model, ranges, *anchor)?;
            if first != other {
                model.bonds.push(DemBond {
                    id: u32::try_from(model.bonds.len() + 1)
                        .map_err(|_| ExportError::IndexOverflow)?,
                    bond_type: junction.law.0.saturating_add(2),
                    particle_a: first,
                    particle_b: other,
                });
            }
        }
    }
    Ok(())
}

fn wrap_sphere_centers(assembly: &FiberAssembly, model: &mut DemBpmModel) {
    for particle in &mut model.particles {
        wrap_position(
            assembly.cell.periodic,
            model.box_low,
            model.box_high,
            &mut particle.position,
        );
    }
}

fn wrap_capsule_centers(assembly: &FiberAssembly, model: &mut DemCapsuleBpmModel) {
    for capsule in &mut model.capsules {
        wrap_position(
            assembly.cell.periodic,
            model.box_low,
            model.box_high,
            &mut capsule.position,
        );
    }
}

fn wrap_position(periodic: [bool; 3], low: Vec3, high: Vec3, position: &mut Vec3) {
    for axis in 0..3 {
        if periodic[axis] {
            let length = high[axis] - low[axis];
            position[axis] = low[axis] + (position[axis] - low[axis]).rem_euclid(length);
        }
    }
}

fn export_bpm_system(
    assembly: Res<FiberAssembly>,
    config: Res<BpmExportConfig>,
    mut report: ResMut<BpmExportReport>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    let model = build_bpm_model(&assembly, &config)
        .unwrap_or_else(|error| panic!("BPM discretization failed: {error}"));
    write_bpm_lammps_data(&model, &config.data_path)
        .unwrap_or_else(|error| panic!("BPM data export failed: {error}"));
    *report = BpmExportReport {
        mode: config.mode,
        particles: model.particles(),
        bonds: model.bonds(),
        junction_bonds: assembly
            .junctions
            .junctions
            .iter()
            .map(|junction| junction.anchors.len.saturating_sub(1) as usize)
            .sum(),
        atom_types: model.atom_types().to_vec(),
        data_path: config.data_path.clone(),
    };
    next.set(TangleStage::Done);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell};

    fn one_fiber(points: &[Vec3], radius: f64) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius });
        assembly
            .add_fiber(FiberId(1), material, section, points, points)
            .unwrap();
        assembly
    }

    fn config(mode: BpmExportMode) -> BpmExportConfig {
        BpmExportConfig::new("")
            .with_mode(mode)
            .with_density(1_800.0)
            .with_sphere_spacing_over_radius(1.0)
    }

    #[test]
    fn exact_spheres_keep_constant_spacing_and_share_end_remainder() {
        let assembly = one_fiber(&[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]], 0.1);
        let BpmModel::Spheres(model) =
            build_bpm_model(&assembly, &config(BpmExportMode::SpheresExact)).unwrap()
        else {
            panic!("expected spheres");
        };

        assert_eq!(model.particles.len(), 10);
        assert!((model.particles[0].position[0] - 0.025).abs() < 1.0e-12);
        assert!((model.particles[9].position[0] - 0.925).abs() < 1.0e-12);
        for pair in model.particles.windows(2) {
            assert!((distance(pair[0].position, pair[1].position) - 0.1).abs() < 1.0e-12);
        }
    }

    #[test]
    fn dynamic_spheres_preserve_each_segment_endpoint() {
        let assembly = one_fiber(&[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.18, 0.0]], 0.1);
        let BpmModel::Spheres(model) =
            build_bpm_model(&assembly, &config(BpmExportMode::SpheresDynamic)).unwrap()
        else {
            panic!("expected spheres");
        };

        assert_eq!(model.particles.len(), 6);
        assert_eq!(model.particles[3].position, [0.25, 0.0, 0.0]);
        assert_eq!(model.particles[5].position, [0.25, 0.18, 0.0]);
        assert!(
            (distance(model.particles[0].position, model.particles[1].position) - 0.25 / 3.0).abs()
                < 1.0e-12
        );
        assert!(
            (distance(model.particles[3].position, model.particles[4].position) - 0.09).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn sphere_spacing_is_limited_to_one_third_through_one_radius() {
        let assembly = one_fiber(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.1);
        for spacing in [1.0 / 3.0, 1.0] {
            let valid =
                config(BpmExportMode::SpheresExact).with_sphere_spacing_over_radius(spacing);
            assert!(build_bpm_model(&assembly, &valid).is_ok());
        }
        for spacing in [0.32, 1.01] {
            let invalid =
                config(BpmExportMode::SpheresExact).with_sphere_spacing_over_radius(spacing);
            assert!(matches!(
                build_bpm_model(&assembly, &invalid),
                Err(ExportError::InvalidSphereSpacingOverRadius(value)) if value == spacing
            ));
        }
    }

    #[test]
    fn exact_spherocylinders_follow_active_segments() {
        let assembly = one_fiber(&[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.65, 0.0, 0.0]], 0.05);
        let BpmModel::Spherocylinders(model) =
            build_bpm_model(&assembly, &config(BpmExportMode::SpherocylindersExact)).unwrap()
        else {
            panic!("expected spherocylinders");
        };

        assert_eq!(model.capsules.len(), 2);
        assert!((2.0 * model.capsules[0].half_length - 0.2).abs() < 1.0e-12);
        assert!((2.0 * model.capsules[1].half_length - 0.45).abs() < 1.0e-12);
    }

    #[test]
    fn constant_spherocylinders_use_the_shortest_active_segment() {
        let assembly = one_fiber(&[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.65, 0.0, 0.0]], 0.05);
        let BpmModel::Spherocylinders(model) =
            build_bpm_model(&assembly, &config(BpmExportMode::SpherocylindersConstant)).unwrap()
        else {
            panic!("expected spherocylinders");
        };

        let lengths = model
            .capsules
            .iter()
            .map(|capsule| 2.0 * capsule.half_length)
            .collect::<Vec<_>>();
        assert_eq!(lengths.len(), 4);
        assert!((lengths[0] - 0.2).abs() < 1.0e-12);
        assert!((lengths[1] - 0.2).abs() < 1.0e-12);
        assert!((lengths[2] - 0.2).abs() < 1.0e-12);
        assert!((lengths[3] - 0.05).abs() < 1.0e-12);
        let final_capsule = model.capsules.last().unwrap();
        assert!((final_capsule.position[0] + final_capsule.half_length - 0.65).abs() < 1.0e-12);
    }
}
