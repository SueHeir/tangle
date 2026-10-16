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
    DemAtomType, DemBond, DemBpmModel, DemCapsule, DemCapsuleBpmModel, DemParticle, ExportError,
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
            build_spherocylinder_model(assembly, config, None).map(BpmModel::Spherocylinders)
        }
        BpmExportMode::SpherocylindersConstant => {
            let target_length = shortest_active_segment(assembly)?;
            build_spherocylinder_model(assembly, config, Some(target_length))
                .map(BpmModel::Spherocylinders)
        }
    }
}

/// Writes a BPM model as a LAMMPS data file: `atom_style bpm/sphere` for
/// spheres, `atom_style bacillus` for spherocylinders.
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
    let ratio = config.min_capsule_length_over_diameter;
    if !ratio.is_finite() || ratio < 0.0 {
        return Err(ExportError::InvalidCapsuleLengthRatio(ratio));
    }
    if config.bond_type == 0 {
        return Err(ExportError::InvalidFiberTypes(
            "the first bond type must be positive".to_string(),
        ));
    }
    Ok(())
}

/// The export's fiber types (one per material, or the caller's) and each
/// fiber's index into them.
fn fiber_typing(
    assembly: &FiberAssembly,
    config: &BpmExportConfig,
) -> Result<(Vec<DemAtomType>, Vec<usize>), ExportError> {
    let fibers = &assembly.topology.fibers;
    let (names, materials, of_fiber): (Vec<String>, Vec<MaterialId>, Vec<usize>) =
        match &config.fiber_types {
            None => {
                let mut of_fiber = Vec::with_capacity(fibers.len());
                for fiber in fibers {
                    if fiber.material.0 as usize >= assembly.materials.entries.len() {
                        return Err(ExportError::MissingMaterial(fiber.id.0));
                    }
                    of_fiber.push(fiber.material.0 as usize);
                }
                let names = assembly
                    .materials
                    .entries
                    .iter()
                    .map(|descriptor| descriptor.name.clone())
                    .collect::<Vec<_>>();
                let materials = (0..names.len())
                    .map(|index| MaterialId(index as u32))
                    .collect();
                (names, materials, of_fiber)
            }
            Some(types) => {
                if types.of_fiber.len() != fibers.len() {
                    return Err(ExportError::InvalidFiberTypes(format!(
                        "{} fiber types for {} fibers",
                        types.of_fiber.len(),
                        fibers.len()
                    )));
                }
                if let Some(kind) = types
                    .of_fiber
                    .iter()
                    .find(|kind| **kind as usize >= types.names.len())
                {
                    return Err(ExportError::InvalidFiberTypes(format!(
                        "type {kind} of {} named types",
                        types.names.len()
                    )));
                }
                let of_fiber = types
                    .of_fiber
                    .iter()
                    .map(|kind| *kind as usize)
                    .collect::<Vec<_>>();
                let materials = (0..types.names.len())
                    .map(|kind| {
                        of_fiber
                            .iter()
                            .position(|of| *of == kind)
                            .map_or(MaterialId(0), |fiber| fibers[fiber].material)
                    })
                    .collect();
                (types.names.clone(), materials, of_fiber)
            }
        };
    let types = names
        .into_iter()
        .zip(materials)
        .enumerate()
        .map(|(index, (name, material))| {
            let offset = u32::try_from(index).map_err(|_| ExportError::IndexOverflow)?;
            Ok(DemAtomType {
                atom_type: config
                    .atom_type
                    .checked_add(offset)
                    .ok_or(ExportError::IndexOverflow)?,
                bond_type: config
                    .bond_type
                    .checked_add(offset)
                    .ok_or(ExportError::IndexOverflow)?,
                material,
                material_name: name,
                particles: 0,
            })
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    Ok((types, of_fiber))
}

/// Bond type of junction law zero: the first after the fiber types' bond types.
fn first_junction_bond_type(config: &BpmExportConfig, types: &[DemAtomType]) -> u32 {
    config.bond_type.saturating_add(types.len() as u32)
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
    let (atom_types, type_of_fiber) = fiber_typing(assembly, config)?;
    let mut model = DemBpmModel {
        box_low,
        box_high,
        atom_types,
        ..DemBpmModel::default()
    };
    let mut fiber_particle_ranges = Vec::with_capacity(assembly.topology.fibers.len());

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let kind = type_of_fiber[fiber_index];
        let atom_type = model.atom_types[kind].atom_type;
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
        model.atom_types[kind].particles += model.particles.len() - first_particle;
        append_consecutive_bonds(
            &mut model.bonds,
            model.atom_types[kind].bond_type,
            model.particles[first_particle..]
                .iter()
                .map(|particle| particle.id),
        )?;
        fiber_particle_ranges.push(first_particle..model.particles.len());
    }

    let junction_base = first_junction_bond_type(config, &model.atom_types);
    append_sphere_junction_bonds(assembly, &mut model, &fiber_particle_ranges, junction_base)?;
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

/// One capsule per active segment, or with `target_length`, each segment
/// tiled with capsules of that length and one shorter end remainder; segments
/// and remainders shorter than the minimum capsule length are merged first.
fn build_spherocylinder_model(
    assembly: &FiberAssembly,
    config: &BpmExportConfig,
    target_length: Option<f64>,
) -> Result<DemCapsuleBpmModel, ExportError> {
    let (box_low, box_high) = orthorhombic_bounds(assembly)?;
    let (atom_types, type_of_fiber) = fiber_typing(assembly, config)?;
    let mut model = DemCapsuleBpmModel {
        box_low,
        box_high,
        atom_types,
        ..DemCapsuleBpmModel::default()
    };
    let mut fiber_capsule_ranges = Vec::with_capacity(assembly.topology.fibers.len());

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let kind = type_of_fiber[fiber_index];
        let atom_type = model.atom_types[kind].atom_type;
        let radius = circular_radius(assembly, fiber.section.0 as usize, fiber.id.0)?;
        // A capsule shorter than its diameter overlaps the capsule two along
        // the chain, which it isn't bonded to, and LAMMPS's BPM bond styles
        // keep pair forces on between those two.
        let min_length = config.min_capsule_length_over_diameter * 2.0 * radius;
        let points = fiber_points(assembly, fiber_index)?;
        let points = merged_vertices(points, min_length);
        let target_length = target_length.map(|target| target.max(min_length));
        let first_capsule = model.capsules.len();
        for pair in points.windows(2) {
            append_constant_capsules(
                &mut model.capsules,
                pair[0],
                pair[1],
                target_length,
                min_length,
                fiber_index,
                atom_type,
                radius,
                config.density,
            )?;
        }
        if model.capsules.len() == first_capsule {
            return Err(ExportError::InvalidFiberSpan(fiber.id.0));
        }
        model.atom_types[kind].particles += model.capsules.len() - first_capsule;
        append_consecutive_bonds(
            &mut model.bonds,
            model.atom_types[kind].bond_type,
            model.capsules[first_capsule..]
                .iter()
                .map(|capsule| capsule.id),
        )?;
        fiber_capsule_ranges.push(first_capsule..model.capsules.len());
    }

    let junction_base = first_junction_bond_type(config, &model.atom_types);
    append_capsule_junction_bonds(assembly, &mut model, &fiber_capsule_ranges, junction_base)?;
    // TANGLE keeps long periodic fibers unwrapped so their centerlines remain
    // continuous across many cell traversals; LAMMPS needs the centers in the
    // box. Wrap only after junction lookup, which compares capsules and
    // anchors in the same unwrapped frame.
    wrap_capsule_centers(assembly, &mut model);
    Ok(model)
}

/// The fiber's vertices with the ones closer than `min_length` to the last
/// kept vertex dropped, so each segment between kept vertices is at least that
/// long. The fiber's two ends are always kept; a short remainder at the end
/// joins the last segment, and a fiber shorter than `min_length` is one
/// segment.
fn merged_vertices(points: &[Vec3], min_length: f64) -> Vec<Vec3> {
    let threshold = min_length * (1.0 - 1.0e-12);
    let mut kept = vec![points[0]];
    for &point in &points[1..] {
        if distance(*kept.last().expect("kept is never empty"), point) >= threshold {
            kept.push(point);
        }
    }
    let end = *points.last().expect("a fiber has at least two vertices");
    if kept.len() == 1 {
        kept.push(end);
    } else if *kept.last().expect("kept is never empty") != end {
        *kept.last_mut().expect("kept is never empty") = end;
    }
    kept
}

#[allow(clippy::too_many_arguments)]
fn append_constant_capsules(
    capsules: &mut Vec<DemCapsule>,
    start: Vec3,
    end: Vec3,
    target_length: Option<f64>,
    min_length: f64,
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
    let target_length = target_length.unwrap_or(segment_length);
    let tolerance = target_length * 1.0e-10;
    let mut consumed = 0.0;
    while segment_length - consumed > tolerance {
        let remaining = segment_length - consumed;
        let mut piece_length = remaining.min(target_length);
        // A remainder shorter than `min_length` joins this piece instead.
        if remaining - piece_length > tolerance && remaining - piece_length < min_length {
            piece_length = remaining;
        }
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
    first_bond_type: u32,
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
                    bond_type: first_bond_type.saturating_add(junction.law.0),
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
    first_bond_type: u32,
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
                    bond_type: first_bond_type.saturating_add(junction.law.0),
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
    use tangle_core::{FiberAnchor, FiberId, JunctionId, JunctionParameterId, PeriodicCell};

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
        // The 0.05 end remainder is shorter than the 0.1 diameter, so it joins
        // the last full piece.
        assert_eq!(lengths.len(), 3);
        assert!((lengths[0] - 0.2).abs() < 1.0e-12);
        assert!((lengths[1] - 0.2).abs() < 1.0e-12);
        assert!((lengths[2] - 0.25).abs() < 1.0e-12);
        let final_capsule = model.capsules.last().unwrap();
        assert!((final_capsule.position[0] + final_capsule.half_length - 0.65).abs() < 1.0e-12);
    }

    /// Two straight fibers of different radii, the first along x and the
    /// second along y, each of two segments; with `materials`, each its own
    /// material.
    fn two_fibers(materials: bool) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let fine = assembly.materials.add("fine 7.00um");
        let coarse = if materials {
            assembly.materials.add("coarse 19.00um")
        } else {
            fine
        };
        for (id, material, radius, points) in [
            (
                1,
                fine,
                0.05,
                [[0.2, 0.5, 1.0], [0.6, 0.5, 1.0], [1.4, 0.5, 1.0]],
            ),
            (
                2,
                coarse,
                0.1,
                [[1.0, 0.2, 1.2], [1.0, 1.0, 1.2], [1.0, 1.8, 1.2]],
            ),
        ] {
            let section = assembly.sections.add(Section::Circular { radius });
            assembly
                .add_fiber(FiberId(id), material, section, &points, &points)
                .unwrap();
        }
        assembly
    }

    fn sections(data: &str) -> std::collections::HashMap<String, Vec<Vec<f64>>> {
        let mut sections = std::collections::HashMap::new();
        let mut name = None;
        for line in data.lines().skip(1) {
            let words = line
                .split('#')
                .next()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>();
            if words.is_empty() {
                continue;
            }
            if words[0].chars().next().unwrap().is_alphabetic() {
                name = Some(words[0].to_string());
                sections.insert(words[0].to_string(), Vec::new());
            } else if let Some(name) = &name {
                sections
                    .get_mut(name)
                    .unwrap()
                    .push(words.iter().map(|word| word.parse().unwrap()).collect());
            }
        }
        sections
    }

    fn header_count(data: &str, what: &str) -> Option<u64> {
        data.lines().find_map(|line| {
            let (count, rest) = line.split_once(' ')?;
            (rest == what).then(|| count.parse().ok())?
        })
    }

    fn written(model: &BpmModel, name: &str) -> String {
        let path = std::env::temp_dir().join(format!("tangle-{name}-{}.data", std::process::id()));
        write_bpm_lammps_data(model, &path).unwrap();
        let data = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        data
    }

    #[test]
    fn capsules_are_written_for_atom_style_bacillus() {
        let assembly = two_fibers(false);
        let config = BpmExportConfig::new("").with_density(1_500.0);
        let model = build_bpm_model(&assembly, &config).unwrap();
        let data = written(&model, "bacillus");
        let lines = data.lines().collect::<Vec<_>>();
        assert_eq!(header_count(&data, "atoms"), Some(4));
        assert_eq!(header_count(&data, "bacilli"), Some(4));
        assert_eq!(header_count(&data, "bonds"), Some(2));
        assert_eq!(header_count(&data, "atom types"), Some(1));
        assert_eq!(header_count(&data, "bond types"), Some(1));
        assert!(lines.contains(&"Atoms # bacillus"));
        // Nothing but rows after a section's title and blank line: LAMMPS
        // reads every line there as data.
        let first = lines
            .iter()
            .position(|line| *line == "Atoms # bacillus")
            .unwrap();
        assert!(lines[first..].iter().all(|line| !line.starts_with('#')));
        for title in ["Atoms # bacillus", "Bacilli", "Bonds"] {
            let at = lines.iter().position(|line| *line == title).unwrap();
            assert_eq!(lines[at + 1], "");
        }

        let sections = sections(&data);
        let atoms = &sections["Atoms"];
        let bacilli = &sections["Bacilli"];
        assert_eq!(atoms.len(), 4);
        assert_eq!(bacilli.len(), 4);
        let close = |got: &[f64], want: &[f64]| {
            got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1.0e-12)
        };
        // id molecule type bacillus-flag density x y z
        assert!(close(
            &atoms[0],
            &[1.0, 1.0, 1.0, 1.0, 1_500.0, 0.4, 0.5, 1.0]
        ));
        assert!(close(
            &atoms[3],
            &[4.0, 2.0, 1.0, 1.0, 1_500.0, 1.0, 1.4, 1.2]
        ));
        // id Ixx Iyy Izz Ixy Ixz Iyz pole diameter
        for (row, (pole, diameter)) in bacilli.iter().zip([
            ([0.2, 0.0, 0.0], 0.1),
            ([0.4, 0.0, 0.0], 0.1),
            ([0.0, 0.4, 0.0], 0.2),
            ([0.0, 0.4, 0.0], 0.2),
        ]) {
            assert!(close(&row[7..10], &pole), "{row:?}");
            assert_eq!(row[10], diameter);
            assert!(row[1..4].iter().all(|moment| *moment > 0.0));
        }
        assert_eq!(
            sections["Bonds"],
            [vec![1.0, 1.0, 1.0, 2.0], vec![2.0, 1.0, 3.0, 4.0]]
        );
    }

    #[test]
    fn capsule_inertia_matches_a_cylinder_and_two_hemispheres() {
        use std::f64::consts::PI;
        let (r, length, density) = (0.5, 4.0, 1.0);
        let axis = [0.6, 0.0, 0.8];
        let capsule = DemCapsule {
            id: 1,
            molecule: 1,
            atom_type: 1,
            diameter: 2.0 * r,
            density,
            position: [0.0; 3],
            half_length: 0.5 * length,
            axis,
        };
        let tensor = crate::capsule_inertia(&capsule);
        // Moments about the axis and across it, from the textbook capsule:
        // a cylinder plus a sphere split in two and moved to the ends.
        let cylinder = density * PI * r * r * length;
        let sphere = density * 4.0 / 3.0 * PI * r * r * r;
        let axial = 0.5 * cylinder * r * r + 0.4 * sphere * r * r;
        let transverse = cylinder * (length * length / 12.0 + r * r / 4.0)
            + sphere * (0.4 * r * r + length * length / 4.0 + 3.0 * length * r / 8.0);
        let along = (0..3)
            .map(|a| {
                (0..3)
                    .map(|b| axis[a] * tensor[a][b] * axis[b])
                    .sum::<f64>()
            })
            .sum::<f64>();
        let across_axis = [0.8, 0.0, -0.6];
        let across = (0..3)
            .map(|a| {
                (0..3)
                    .map(|b| across_axis[a] * tensor[a][b] * across_axis[b])
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert!((along - axial).abs() < 1.0e-12 * axial);
        assert!((across - transverse).abs() < 1.0e-12 * transverse);
        assert!((tensor[1][1] - transverse).abs() < 1.0e-12 * transverse);
        assert!(tensor[0][1].abs() < 1.0e-15 && tensor[1][2].abs() < 1.0e-15);
    }

    #[test]
    fn each_material_is_a_fiber_type_with_its_own_atom_and_bond_type() {
        let assembly = two_fibers(true);
        let model = build_bpm_model(&assembly, &BpmExportConfig::new("")).unwrap();
        let types = model.atom_types();
        assert_eq!(types.len(), 2);
        assert_eq!(
            (types[0].atom_type, types[0].bond_type, types[0].particles),
            (1, 1, 2)
        );
        assert_eq!(
            (types[1].atom_type, types[1].bond_type, types[1].particles),
            (2, 2, 2)
        );
        let data = written(&model, "two-materials");
        assert_eq!(header_count(&data, "atom types"), Some(2));
        assert_eq!(header_count(&data, "bond types"), Some(2));
        assert!(data.contains(
            "# atom type 2 and bond type 2 = fiber type \"coarse 19.00um\" (2 capsules)"
        ));
        let sections = sections(&data);
        let atom_types = sections["Atoms"]
            .iter()
            .map(|row| row[2])
            .collect::<Vec<_>>();
        assert_eq!(atom_types, [1.0, 1.0, 2.0, 2.0]);
        assert_eq!(
            sections["Bonds"],
            [vec![1.0, 1.0, 1.0, 2.0], vec![2.0, 2.0, 3.0, 4.0]]
        );

        let BpmModel::Spheres(spheres) =
            build_bpm_model(&assembly, &config(BpmExportMode::SpheresExact)).unwrap()
        else {
            panic!("expected spheres");
        };
        let fiber_two = spheres
            .particles
            .iter()
            .find(|p| p.molecule == 2)
            .unwrap()
            .id;
        assert!(spheres
            .bonds
            .iter()
            .all(|bond| bond.bond_type == if bond.particle_a >= fiber_two { 2 } else { 1 }));
    }

    #[test]
    fn caller_fiber_types_group_fibers_across_materials() {
        // Three materials (diameters within one type differ), two fiber types.
        let mut assembly = two_fibers(true);
        let other = assembly.materials.add("fine 7.10um");
        let section = assembly.sections.add(Section::Circular { radius: 0.0355 });
        let points = [[0.2, 1.5, 0.5], [1.4, 1.5, 0.5]];
        assembly
            .add_fiber(FiberId(3), other, section, &points, &points)
            .unwrap();
        let config = BpmExportConfig::new("").with_fiber_types(
            vec![0, 1, 0],
            vec!["fine".to_string(), "coarse".to_string()],
        );
        let model = build_bpm_model(&assembly, &config).unwrap();
        let types = model.atom_types();
        assert_eq!(types.len(), 2);
        assert_eq!(types[0].material_name, "fine");
        assert_eq!(
            (types[0].atom_type, types[0].bond_type, types[0].particles),
            (1, 1, 3)
        );
        assert_eq!(
            (types[1].atom_type, types[1].bond_type, types[1].particles),
            (2, 2, 2)
        );
        let BpmModel::Spherocylinders(capsules) = &model else {
            panic!("expected spherocylinders");
        };
        let atom_types = capsules
            .capsules
            .iter()
            .map(|c| c.atom_type)
            .collect::<Vec<_>>();
        assert_eq!(atom_types, [1, 1, 2, 2, 1]);
        let bond_types = capsules
            .bonds
            .iter()
            .map(|b| b.bond_type)
            .collect::<Vec<_>>();
        assert_eq!(bond_types, [1, 2]);

        for (of_fiber, names) in [(vec![0, 1], 2), (vec![0, 2, 1], 2)] {
            let names = (0..names).map(|k| format!("type {k}")).collect();
            assert!(matches!(
                build_bpm_model(
                    &assembly,
                    &BpmExportConfig::new("").with_fiber_types(of_fiber, names)
                ),
                Err(ExportError::InvalidFiberTypes(_))
            ));
        }
    }

    #[test]
    fn junction_bonds_take_the_bond_types_after_the_fiber_types() {
        let mut assembly = two_fibers(true);
        let law = assembly.junction_laws.add("welded");
        let anchor = |fiber, rest_arc_length| FiberAnchor {
            fiber: FiberId(fiber),
            rest_arc_length,
            section_offset: None,
        };
        assembly
            .add_junction(
                JunctionId(1),
                law,
                JunctionParameterId(0),
                &[anchor(1, 0.8), anchor(2, 0.7)],
            )
            .unwrap();
        let model = build_bpm_model(&assembly, &BpmExportConfig::new("")).unwrap();
        let BpmModel::Spherocylinders(capsules) = &model else {
            panic!("expected spherocylinders");
        };
        let junction = capsules.bonds.last().unwrap();
        assert_eq!(capsules.bonds.len(), 3);
        assert_eq!(junction.bond_type, 3);
        assert_eq!((junction.particle_a, junction.particle_b), (2, 3));
        assert_eq!(
            header_count(&written(&model, "junction"), "bond types"),
            Some(3)
        );
    }

    #[test]
    fn a_fiber_of_one_segment_writes_no_bonds_section() {
        let assembly = one_fiber(&[[0.2, 1.0, 1.0], [1.2, 1.0, 1.0]], 0.05);
        let model = build_bpm_model(&assembly, &BpmExportConfig::new("")).unwrap();
        let data = written(&model, "no-bonds");
        assert_eq!(header_count(&data, "bonds"), Some(0));
        assert!(!data.lines().any(|line| line == "Bonds"));
        assert_eq!(sections(&data)["Bacilli"].len(), 1);
    }

    #[test]
    fn capsule_export_wraps_periodic_centers_into_the_primary_cell() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic(
            [1.0, 2.0, 3.0],
            [true, true, false],
        ));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.05 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                &[[-2.25, 4.25, 1.0], [-1.75, 4.25, 1.0]],
            )
            .unwrap();
        let BpmModel::Spherocylinders(model) =
            build_bpm_model(&assembly, &BpmExportConfig::new("")).unwrap()
        else {
            panic!("expected spherocylinders");
        };
        assert_eq!(model.capsules.len(), 1);
        let position = model.capsules[0].position;
        assert!((position[0] - 0.0).abs() < 1.0e-12, "{position:?}");
        assert!((position[1] - 0.25).abs() < 1.0e-12, "{position:?}");
        assert!((position[2] - 1.0).abs() < 1.0e-12, "{position:?}");
    }

    #[test]
    fn capsules_are_at_least_a_diameter_long() {
        // Diameter 0.2: segments of 0.05 merge until they reach it, and the
        // short end remainder joins the last capsule.
        let points = [
            [0.0, 1.0, 1.0],
            [0.05, 1.0, 1.0],
            [0.1, 1.0, 1.0],
            [0.3, 1.0, 1.0],
            [0.32, 1.0, 1.0],
            [0.6, 1.0, 1.0],
            [0.65, 1.0, 1.0],
        ];
        let assembly = one_fiber(&points, 0.1);
        let lengths = |config: &BpmExportConfig| {
            let BpmModel::Spherocylinders(model) = build_bpm_model(&assembly, config).unwrap()
            else {
                panic!("expected spherocylinders");
            };
            let total = model.capsules.iter().map(|c| 2.0 * c.half_length).sum::<f64>();
            assert!((total - 0.65).abs() < 1.0e-12);
            model
                .capsules
                .iter()
                .map(|capsule| 2.0 * capsule.half_length)
                .collect::<Vec<_>>()
        };
        let merged = lengths(&BpmExportConfig::new(""));
        assert_eq!(merged.len(), 2);
        assert!((merged[0] - 0.3).abs() < 1.0e-12 && (merged[1] - 0.35).abs() < 1.0e-12);
        // Constant mode tiles with at least a diameter too.
        let constant = lengths(&config(BpmExportMode::SpherocylindersConstant));
        assert!(constant.iter().all(|length| *length >= 0.2 - 1.0e-12), "{constant:?}");
        // Off: every active segment.
        let every = lengths(&BpmExportConfig::new("").with_min_capsule_length_over_diameter(0.0));
        assert_eq!(every.len(), 6);
        // A fiber shorter than its diameter is one capsule.
        let short = one_fiber(&[[0.0, 1.0, 1.0], [0.05, 1.0, 1.0], [0.1, 1.0, 1.0]], 0.1);
        let BpmModel::Spherocylinders(model) =
            build_bpm_model(&short, &BpmExportConfig::new("")).unwrap()
        else {
            panic!("expected spherocylinders");
        };
        assert_eq!(model.capsules.len(), 1);
        assert!(model.bonds.is_empty());
    }
}
