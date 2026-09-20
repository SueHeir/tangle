//! Export TANGLE fiber assemblies into derived solver representations.

#![warn(missing_docs)]

mod ovito;
mod trajectory;

pub use ovito::{
    write_ovito_assembly_frame, write_ovito_dump_frame, write_ovito_view_script, OvitoColoring,
    OvitoRepresentation, OvitoTrajectoryConfig, OvitoTrajectoryReport,
};
pub use trajectory::OvitoTrajectoryPlugin;

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::{FiberAssembly, MaterialId, Section, Vec3};

/// One sphere in a bonded-particle discretization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemParticle {
    /// One-based particle identifier.
    pub id: u32,
    /// One-based source-fiber molecule identifier.
    pub molecule: u32,
    /// One-based atom type.
    pub atom_type: u32,
    /// Sphere diameter.
    pub diameter: f64,
    /// Material density.
    pub density: f64,
    /// Sphere center.
    pub position: Vec3,
}

/// One adaptive fiber segment represented as a rigid capsule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemCapsule {
    /// One-based particle identifier.
    pub id: u32,
    /// One-based source-fiber molecule identifier.
    pub molecule: u32,
    /// One-based atom type.
    pub atom_type: u32,
    /// Capsule diameter, including the hemispherical caps.
    pub diameter: f64,
    /// Material density.
    pub density: f64,
    /// Capsule center of mass.
    pub position: Vec3,
    /// Half of the cylindrical centerline length.
    pub half_length: f64,
    /// Unit centerline direction.
    pub axis: Vec3,
}

/// One intra-fiber or persistent-junction particle bond.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemBond {
    /// One-based bond identifier.
    pub id: u32,
    /// One-based bond type.
    pub bond_type: u32,
    /// First particle identifier.
    pub particle_a: u32,
    /// Second particle identifier.
    pub particle_b: u32,
}

/// Mapping from one TANGLE material to one DEM atom type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemAtomType {
    /// One-based atom type written to the LAMMPS data file.
    pub atom_type: u32,
    /// Solver-neutral TANGLE material identifier.
    pub material: MaterialId,
    /// Human-readable TANGLE material name.
    pub material_name: String,
    /// Number of particles exported with this type.
    pub particles: usize,
}

/// Complete bonded-sphere model derived from a fiber assembly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DemBpmModel {
    /// Exported particles.
    pub particles: Vec<DemParticle>,
    /// Consecutive intra-fiber bonds followed by persistent-junction bonds.
    pub bonds: Vec<DemBond>,
    /// Stable material-to-atom-type mappings used by the particles.
    pub atom_types: Vec<DemAtomType>,
    /// Lower orthorhombic box corner.
    pub box_low: Vec3,
    /// Upper orthorhombic box corner.
    pub box_high: Vec3,
}

/// Complete bonded-capsule model derived from the active fiber segments.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DemCapsuleBpmModel {
    /// One rigid capsule per exported centerline segment.
    pub capsules: Vec<DemCapsule>,
    /// Consecutive intra-fiber bonds followed by persistent-junction bonds.
    pub bonds: Vec<DemBond>,
    /// Stable material-to-atom-type mappings used by the capsules.
    pub atom_types: Vec<DemAtomType>,
    /// Lower orthorhombic box corner.
    pub box_low: Vec3,
    /// Upper orthorhombic box corner.
    pub box_high: Vec3,
}

/// DEM-BPM discretization and output configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DemBpmExportConfig {
    /// LAMMPS-data output path.
    pub data_path: PathBuf,
    /// Optional directly usable DIRT configuration output path.
    pub dirt_config_path: Option<PathBuf>,
    /// Sphere material density.
    pub density: f64,
    /// Maximum center spacing as a fraction of sphere diameter.
    pub spacing_ratio: f64,
    /// First atom type assigned to TANGLE material zero. Subsequent dense
    /// material identifiers receive consecutive atom types.
    pub atom_type: u32,
    /// Intra-fiber bond type written to the data file.
    pub bond_type: u32,
}

impl DemBpmExportConfig {
    /// Creates a conventional bonded-sphere export configuration.
    pub fn new(data_path: impl Into<PathBuf>) -> Self {
        Self {
            data_path: data_path.into(),
            dirt_config_path: None,
            density: 1.0,
            spacing_ratio: 0.9,
            atom_type: 1,
            bond_type: 1,
        }
    }

    /// Sets the sphere material density.
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Sets maximum particle spacing as a fraction of sphere diameter.
    pub fn with_spacing_ratio(mut self, spacing_ratio: f64) -> Self {
        self.spacing_ratio = spacing_ratio;
        self
    }
}

/// Adaptive-segment capsule discretization and output configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DemCapsuleBpmExportConfig {
    /// DIRT-oriented extended LAMMPS-data output path.
    pub data_path: PathBuf,
    /// Optional DIRT configuration written beside the capsule data.
    pub dirt_config_path: Option<PathBuf>,
    /// Capsule material density.
    pub density: f64,
    /// First atom type assigned to TANGLE material zero.
    pub atom_type: u32,
    /// Intra-fiber bond type written to the data file.
    pub bond_type: u32,
    /// Optional export-only maximum centerline length divided by diameter.
    /// Long adaptive segments are split without changing the TANGLE assembly.
    pub maximum_length_over_diameter: Option<f64>,
}

impl DemCapsuleBpmExportConfig {
    /// Creates an adaptive-capsule export configuration.
    pub fn new(data_path: impl Into<PathBuf>) -> Self {
        Self {
            data_path: data_path.into(),
            dirt_config_path: None,
            density: 1.0,
            atom_type: 1,
            bond_type: 1,
            maximum_length_over_diameter: None,
        }
    }

    /// Sets the capsule material density.
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Splits exported segments longer than this multiple of their diameter.
    pub fn with_maximum_length_over_diameter(mut self, ratio: f64) -> Self {
        self.maximum_length_over_diameter = Some(ratio);
        self
    }

    /// Requests a directly runnable DIRT loading configuration.
    pub fn with_dirt_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.dirt_config_path = Some(path.into());
        self
    }
}

/// Summary of a completed DEM-BPM export.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DemBpmExportReport {
    /// Number of written particles.
    pub particles: usize,
    /// Number of written bonds.
    pub bonds: usize,
    /// Inter-fiber bonds derived from persistent junctions.
    pub junction_bonds: usize,
    /// Material-to-atom-type mappings written to the data file.
    pub atom_types: Vec<DemAtomType>,
    /// Written data file.
    pub data_path: PathBuf,
    /// Written DIRT configuration, when requested.
    pub dirt_config_path: Option<PathBuf>,
}

/// Summary of a completed adaptive-capsule DEM-BPM export.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DemCapsuleBpmExportReport {
    /// Number of written capsule particles.
    pub capsules: usize,
    /// Number of written bonds.
    pub bonds: usize,
    /// Inter-fiber bonds derived from persistent junctions.
    pub junction_bonds: usize,
    /// Material-to-atom-type mappings written to the data file.
    pub atom_types: Vec<DemAtomType>,
    /// Written data file.
    pub data_path: PathBuf,
    /// Written DIRT loading configuration, when requested.
    pub dirt_config_path: Option<PathBuf>,
}

/// Failure while discretizing or writing a DEM-BPM model.
#[derive(Debug)]
pub enum ExportError {
    /// Density was not positive and finite.
    InvalidDensity(f64),
    /// Spacing ratio was outside the supported interval.
    InvalidSpacingRatio(f64),
    /// Capsule export length ratio was not positive and finite.
    InvalidCapsuleLengthRatio(f64),
    /// First atom type was zero; LAMMPS types are one-based.
    InvalidAtomType(u32),
    /// DEM sphere export currently requires circular fibers.
    NonCircularSection(u32),
    /// Fiber section lookup failed.
    MissingSection(u32),
    /// Fiber geometry lookup failed.
    InvalidFiberSpan(u32),
    /// Fiber material lookup failed.
    MissingMaterial(u32),
    /// The current LAMMPS-data writer requires an orthorhombic positive cell.
    NonOrthorhombicCell,
    /// The particle or bond count exceeded 32-bit identifiers.
    IndexOverflow,
    /// File-system output failed.
    Io(std::io::Error),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDensity(value) => write!(f, "density must be positive, got {value}"),
            Self::InvalidSpacingRatio(value) => {
                write!(f, "spacing_ratio must be in (0, 1], got {value}")
            }
            Self::InvalidCapsuleLengthRatio(value) => write!(
                f,
                "maximum_length_over_diameter must be positive, got {value}"
            ),
            Self::InvalidAtomType(value) => {
                write!(f, "first atom type must be positive, got {value}")
            }
            Self::NonCircularSection(id) => {
                write!(f, "fiber {id} uses a non-circular section")
            }
            Self::MissingSection(id) => write!(f, "fiber {id} references a missing section"),
            Self::InvalidFiberSpan(id) => write!(f, "fiber {id} has an invalid placed span"),
            Self::MissingMaterial(id) => write!(f, "fiber {id} references a missing material"),
            Self::NonOrthorhombicCell => {
                f.write_str("DEM-BPM LAMMPS export currently requires a positive orthorhombic cell")
            }
            Self::IndexOverflow => f.write_str("DEM-BPM model exceeds 32-bit identifier capacity"),
            Self::Io(error) => write!(f, "output failed: {error}"),
        }
    }
}

impl Error for ExportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ExportError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Discretizes placed centerlines into overlapping or touching bonded spheres.
pub fn build_dem_bpm_model(
    assembly: &FiberAssembly,
    config: &DemBpmExportConfig,
) -> Result<DemBpmModel, ExportError> {
    if !config.density.is_finite() || config.density <= 0.0 {
        return Err(ExportError::InvalidDensity(config.density));
    }
    if !config.spacing_ratio.is_finite()
        || config.spacing_ratio <= 0.0
        || config.spacing_ratio > 1.0
    {
        return Err(ExportError::InvalidSpacingRatio(config.spacing_ratio));
    }
    if config.atom_type == 0 {
        return Err(ExportError::InvalidAtomType(config.atom_type));
    }
    let (box_low, box_high) = orthorhombic_bounds(assembly)?;
    let mut model = DemBpmModel {
        box_low,
        box_high,
        atom_types: assembly
            .materials
            .entries
            .iter()
            .enumerate()
            .map(|(material, descriptor)| {
                let material = u32::try_from(material).map_err(|_| ExportError::IndexOverflow)?;
                let atom_type = config
                    .atom_type
                    .checked_add(material)
                    .ok_or(ExportError::IndexOverflow)?;
                Ok(DemAtomType {
                    atom_type,
                    material: MaterialId(material),
                    material_name: descriptor.name.clone(),
                    particles: 0,
                })
            })
            .collect::<Result<Vec<_>, ExportError>>()?,
        ..DemBpmModel::default()
    };
    let mut fiber_particle_ranges = Vec::with_capacity(assembly.topology.fibers.len());

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let material_type = model
            .atom_types
            .get(fiber.material.0 as usize)
            .ok_or(ExportError::MissingMaterial(fiber.id.0))?;
        let atom_type = material_type.atom_type;
        let section = assembly
            .sections
            .entries
            .get(fiber.section.0 as usize)
            .ok_or(ExportError::MissingSection(fiber.id.0))?;
        let radius = match section {
            Section::Circular { radius } => *radius,
            Section::Elliptical { .. } => {
                return Err(ExportError::NonCircularSection(fiber.id.0));
            }
        };
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
        let total_length: f64 = points
            .windows(2)
            .map(|pair| distance(pair[0], pair[1]))
            .sum();
        let max_spacing = 2.0 * radius * config.spacing_ratio;
        let intervals = (total_length / max_spacing).ceil().max(1.0) as usize;
        let first_particle = model.particles.len();
        for sample in 0..=intervals {
            let arc_length = total_length * sample as f64 / intervals as f64;
            let id =
                u32::try_from(model.particles.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
            model.particles.push(DemParticle {
                id,
                molecule: u32::try_from(fiber_index + 1).map_err(|_| ExportError::IndexOverflow)?,
                atom_type,
                diameter: 2.0 * radius,
                density: config.density,
                position: sample_polyline(points, arc_length),
            });
        }
        model.atom_types[fiber.material.0 as usize].particles +=
            model.particles.len() - first_particle;
        for index in first_particle..model.particles.len() - 1 {
            let id =
                u32::try_from(model.bonds.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
            model.bonds.push(DemBond {
                id,
                bond_type: config.bond_type,
                particle_a: model.particles[index].id,
                particle_b: model.particles[index + 1].id,
            });
        }
        fiber_particle_ranges.push(first_particle..model.particles.len());
    }

    for junction in &assembly.junctions.junctions {
        let anchor_start = junction.anchors.start as usize;
        let anchor_end = anchor_start + junction.anchors.len as usize;
        let anchors = &assembly.junctions.anchors[anchor_start..anchor_end];
        let first_particle =
            nearest_anchor_particle(assembly, &model, &fiber_particle_ranges, anchors[0])?;
        for anchor in &anchors[1..] {
            let other_particle =
                nearest_anchor_particle(assembly, &model, &fiber_particle_ranges, *anchor)?;
            if first_particle == other_particle {
                continue;
            }
            let id =
                u32::try_from(model.bonds.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
            model.bonds.push(DemBond {
                id,
                // Type one remains the intra-fiber bond. Junction-law IDs map
                // deterministically to subsequent LAMMPS bond types.
                bond_type: junction.law.0.saturating_add(2),
                particle_a: first_particle,
                particle_b: other_particle,
            });
        }
    }

    Ok(model)
}

/// Converts the active placed polyline segments into rigid capsule particles.
///
/// Unlike the bonded-sphere export, this preserves TANGLE's adaptive
/// segmentation. `maximum_length_over_diameter` can impose a downstream DEM
/// resolution ceiling without modifying the relaxed assembly.
pub fn build_dem_capsule_bpm_model(
    assembly: &FiberAssembly,
    config: &DemCapsuleBpmExportConfig,
) -> Result<DemCapsuleBpmModel, ExportError> {
    if !config.density.is_finite() || config.density <= 0.0 {
        return Err(ExportError::InvalidDensity(config.density));
    }
    if config.atom_type == 0 {
        return Err(ExportError::InvalidAtomType(config.atom_type));
    }
    if let Some(ratio) = config.maximum_length_over_diameter {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(ExportError::InvalidCapsuleLengthRatio(ratio));
        }
    }
    let (box_low, box_high) = orthorhombic_bounds(assembly)?;
    let mut model = DemCapsuleBpmModel {
        box_low,
        box_high,
        atom_types: assembly
            .materials
            .entries
            .iter()
            .enumerate()
            .map(|(material, descriptor)| {
                let material = u32::try_from(material).map_err(|_| ExportError::IndexOverflow)?;
                Ok(DemAtomType {
                    atom_type: config
                        .atom_type
                        .checked_add(material)
                        .ok_or(ExportError::IndexOverflow)?,
                    material: MaterialId(material),
                    material_name: descriptor.name.clone(),
                    particles: 0,
                })
            })
            .collect::<Result<Vec<_>, ExportError>>()?,
        ..DemCapsuleBpmModel::default()
    };
    let mut fiber_capsule_ranges = Vec::with_capacity(assembly.topology.fibers.len());

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let atom_type = model
            .atom_types
            .get(fiber.material.0 as usize)
            .ok_or(ExportError::MissingMaterial(fiber.id.0))?
            .atom_type;
        let section = assembly
            .sections
            .entries
            .get(fiber.section.0 as usize)
            .ok_or(ExportError::MissingSection(fiber.id.0))?;
        let radius = match section {
            Section::Circular { radius } => *radius,
            Section::Elliptical { .. } => {
                return Err(ExportError::NonCircularSection(fiber.id.0));
            }
        };
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
        let first_capsule = model.capsules.len();
        for pair in points.windows(2) {
            let delta = sub(pair[1], pair[0]);
            let segment_length = distance(pair[0], pair[1]);
            if segment_length <= f64::EPSILON {
                continue;
            }
            let maximum_length = config
                .maximum_length_over_diameter
                .map(|ratio| ratio * 2.0 * radius);
            let pieces = maximum_length
                .map(|maximum| (segment_length / maximum).ceil().max(1.0) as usize)
                .unwrap_or(1);
            let piece_length = segment_length / pieces as f64;
            let axis = scale(delta, 1.0 / segment_length);
            for piece in 0..pieces {
                let coordinate = (piece as f64 + 0.5) / pieces as f64;
                let id = u32::try_from(model.capsules.len() + 1)
                    .map_err(|_| ExportError::IndexOverflow)?;
                model.capsules.push(DemCapsule {
                    id,
                    molecule: u32::try_from(fiber_index + 1)
                        .map_err(|_| ExportError::IndexOverflow)?,
                    atom_type,
                    diameter: 2.0 * radius,
                    density: config.density,
                    position: add(pair[0], scale(delta, coordinate)),
                    half_length: 0.5 * piece_length,
                    axis,
                });
            }
        }
        model.atom_types[fiber.material.0 as usize].particles +=
            model.capsules.len() - first_capsule;
        for index in first_capsule..model.capsules.len().saturating_sub(1) {
            model.bonds.push(DemBond {
                id: u32::try_from(model.bonds.len() + 1).map_err(|_| ExportError::IndexOverflow)?,
                bond_type: config.bond_type,
                particle_a: model.capsules[index].id,
                particle_b: model.capsules[index + 1].id,
            });
        }
        fiber_capsule_ranges.push(first_capsule..model.capsules.len());
    }

    for junction in &assembly.junctions.junctions {
        let anchor_start = junction.anchors.start as usize;
        let anchor_end = anchor_start + junction.anchors.len as usize;
        let anchors = &assembly.junctions.anchors[anchor_start..anchor_end];
        let first = nearest_anchor_capsule(assembly, &model, &fiber_capsule_ranges, anchors[0])?;
        for anchor in &anchors[1..] {
            let other = nearest_anchor_capsule(assembly, &model, &fiber_capsule_ranges, *anchor)?;
            if first == other {
                continue;
            }
            model.bonds.push(DemBond {
                id: u32::try_from(model.bonds.len() + 1).map_err(|_| ExportError::IndexOverflow)?,
                bond_type: junction.law.0.saturating_add(2),
                particle_a: first,
                particle_b: other,
            });
        }
    }

    // TANGLE keeps long periodic fibers in an unwrapped representation so their
    // centerlines remain continuous across many cell traversals. DIRT, however,
    // expects every inserted particle center to begin inside the primary cell;
    // its exchange step is intentionally only a neighboring-rank migration, not
    // an arbitrary multi-box fold. Wrap only after junction-anchor lookup so the
    // latter can still compare capsules and anchors in the same unwrapped frame.
    for capsule in &mut model.capsules {
        for axis in 0..3 {
            if assembly.cell.periodic[axis] {
                let length = model.box_high[axis] - model.box_low[axis];
                debug_assert!(length > 0.0);
                capsule.position[axis] = model.box_low[axis]
                    + (capsule.position[axis] - model.box_low[axis]).rem_euclid(length);
            }
        }
    }
    Ok(model)
}

/// Writes a DIRT-compatible LAMMPS data file using the bpm/sphere atom style.
pub fn write_lammps_data(model: &DemBpmModel, path: &Path) -> Result<(), ExportError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "TANGLE DEM-BPM fiber export")?;
    for mapping in &model.atom_types {
        let name = mapping.material_name.replace(['\r', '\n'], " ");
        writeln!(
            writer,
            "# atom type {} = TANGLE material {} {:?} ({} particles)",
            mapping.atom_type, mapping.material.0, name, mapping.particles
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "{} atoms", model.particles.len())?;
    writeln!(writer, "{} bonds", model.bonds.len())?;
    let atom_types = model
        .atom_types
        .iter()
        .map(|mapping| mapping.atom_type)
        .chain(model.particles.iter().map(|particle| particle.atom_type))
        .max()
        .unwrap_or(1);
    writeln!(writer, "{atom_types} atom types")?;
    let bond_types = model
        .bonds
        .iter()
        .map(|bond| bond.bond_type)
        .max()
        .unwrap_or(1);
    writeln!(writer, "{bond_types} bond types")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "{:.17e} {:.17e} xlo xhi",
        model.box_low[0], model.box_high[0]
    )?;
    writeln!(
        writer,
        "{:.17e} {:.17e} ylo yhi",
        model.box_low[1], model.box_high[1]
    )?;
    writeln!(
        writer,
        "{:.17e} {:.17e} zlo zhi",
        model.box_low[2], model.box_high[2]
    )?;
    writeln!(writer)?;
    writeln!(writer, "Atoms # bpm/sphere")?;
    writeln!(writer)?;
    writeln!(writer, "# id molecule type diameter density x y z")?;
    for particle in &model.particles {
        writeln!(
            writer,
            "{} {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
            particle.id,
            particle.molecule,
            particle.atom_type,
            particle.diameter,
            particle.density,
            particle.position[0],
            particle.position[1],
            particle.position[2]
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "Bonds")?;
    writeln!(writer)?;
    writeln!(writer, "# id type atom1 atom2")?;
    for bond in &model.bonds {
        writeln!(
            writer,
            "{} {} {} {}",
            bond.id, bond.bond_type, bond.particle_a, bond.particle_b
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Writes an extended DIRT data file with `Atoms`, `Capsules`, and `Bonds` sections.
pub fn write_capsule_lammps_data(
    model: &DemCapsuleBpmModel,
    path: &Path,
) -> Result<(), ExportError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "TANGLE adaptive-capsule DEM-BPM fiber export")?;
    for mapping in &model.atom_types {
        let name = mapping.material_name.replace(['\r', '\n'], " ");
        writeln!(
            writer,
            "# atom type {} = TANGLE material {} {:?} ({} capsules)",
            mapping.atom_type, mapping.material.0, name, mapping.particles
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "{} atoms", model.capsules.len())?;
    writeln!(writer, "{} bonds", model.bonds.len())?;
    let atom_types = model
        .atom_types
        .iter()
        .map(|mapping| mapping.atom_type)
        .chain(model.capsules.iter().map(|capsule| capsule.atom_type))
        .max()
        .unwrap_or(1);
    writeln!(writer, "{atom_types} atom types")?;
    let bond_types = model
        .bonds
        .iter()
        .map(|bond| bond.bond_type)
        .max()
        .unwrap_or(1);
    writeln!(writer, "{bond_types} bond types")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "{:.17e} {:.17e} xlo xhi",
        model.box_low[0], model.box_high[0]
    )?;
    writeln!(
        writer,
        "{:.17e} {:.17e} ylo yhi",
        model.box_low[1], model.box_high[1]
    )?;
    writeln!(
        writer,
        "{:.17e} {:.17e} zlo zhi",
        model.box_low[2], model.box_high[2]
    )?;
    writeln!(writer)?;
    writeln!(writer, "Atoms # bpm/sphere")?;
    writeln!(writer)?;
    writeln!(writer, "# id molecule type diameter density x y z")?;
    for capsule in &model.capsules {
        writeln!(
            writer,
            "{} {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
            capsule.id,
            capsule.molecule,
            capsule.atom_type,
            capsule.diameter,
            capsule.density,
            capsule.position[0],
            capsule.position[1],
            capsule.position[2]
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "Capsules")?;
    writeln!(writer)?;
    writeln!(writer, "# atom-id half-length axis-x axis-y axis-z")?;
    for capsule in &model.capsules {
        writeln!(
            writer,
            "{} {:.17e} {:.17e} {:.17e} {:.17e}",
            capsule.id, capsule.half_length, capsule.axis[0], capsule.axis[1], capsule.axis[2]
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "Bonds")?;
    writeln!(writer)?;
    writeln!(writer, "# id type atom1 atom2")?;
    for bond in &model.bonds {
        writeln!(
            writer,
            "{} {} {} {}",
            bond.id, bond.bond_type, bond.particle_a, bond.particle_b
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Writes a zero-step DIRT loading configuration for an exported capsule model.
/// Material and bond coefficients are intentionally conservative placeholders;
/// loading and topology can be checked before a physical test law is selected.
pub fn write_dirt_capsule_config(
    model: &DemCapsuleBpmModel,
    data_path: &Path,
    config_path: &Path,
    periodic: [bool; 3],
) -> Result<(), ExportError> {
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(config_path)?);
    let data_path = data_path
        .canonicalize()
        .unwrap_or_else(|_| data_path.to_path_buf());
    let data_path = toml_string(&data_path.to_string_lossy());
    writeln!(writer, "# Generated by TANGLE for DIRT capsule loading.")?;
    writeln!(
        writer,
        "# Replace the placeholder material/bond laws before a physical test."
    )?;
    writeln!(writer, "[comm]")?;
    writeln!(writer, "processors_x = 1")?;
    writeln!(writer, "processors_y = 1")?;
    writeln!(writer, "processors_z = 1")?;
    writeln!(writer)?;
    writeln!(writer, "[domain]")?;
    for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
        writeln!(writer, "{name}_low = {:.17e}", model.box_low[axis])?;
        writeln!(writer, "{name}_high = {:.17e}", model.box_high[axis])?;
        writeln!(
            writer,
            "boundary_{name} = {:?}",
            if periodic[axis] { "periodic" } else { "fixed" }
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "[neighbor]")?;
    writeln!(writer, "skin_fraction = 1.2")?;
    // DIRT's scalar neighbor grid defaults to a 1 m bin, which degenerates to
    // one O(N^2) bucket for micron-scale specimens. A capsule's conservative
    // contact reach is COM-to-tip (half-length + radius); two largest capsules
    // plus the configured skin therefore define a safe, scale-aware bin size.
    let maximum_reach = model
        .capsules
        .iter()
        .map(|capsule| capsule.half_length + 0.5 * capsule.diameter)
        .fold(0.0_f64, f64::max);
    let bin_size = (2.0 * 1.2 * maximum_reach).max(f64::EPSILON);
    writeln!(writer, "bin_size = {bin_size:.17e}")?;
    writeln!(writer, "every = 20")?;
    writeln!(writer)?;
    writeln!(writer, "[dem]")?;
    writeln!(writer, "contact_model = {:?}", "hertz")?;
    writeln!(writer, "tangential_model = {:?}", "linear_nohistory")?;
    for mapping in &model.atom_types {
        writeln!(writer)?;
        writeln!(writer, "[[dem.materials]]")?;
        writeln!(writer, "name = {}", toml_string(&mapping.material_name))?;
        writeln!(writer, "youngs_mod = 1.0e9")?;
        writeln!(writer, "poisson_ratio = 0.25")?;
        writeln!(writer, "restitution = 0.3")?;
        writeln!(writer, "friction = 0.3")?;
    }
    let fallback_material = model
        .atom_types
        .first()
        .map(|mapping| mapping.material_name.as_str())
        .unwrap_or("fiber");
    writeln!(writer)?;
    writeln!(writer, "[[particles.insert]]")?;
    writeln!(writer, "source = {:?}", "file")?;
    writeln!(writer, "file = {data_path}")?;
    writeln!(writer, "format = {:?}", "lammps_data")?;
    writeln!(writer, "atom_style = {:?}", "bpm/sphere")?;
    writeln!(writer, "material = {}", toml_string(fallback_material))?;
    write!(writer, "type_map = {{ ")?;
    for (index, mapping) in model.atom_types.iter().enumerate() {
        if index > 0 {
            write!(writer, ", ")?;
        }
        write!(
            writer,
            "{:?} = {}",
            mapping.atom_type.to_string(),
            toml_string(&mapping.material_name)
        )?;
    }
    writeln!(writer, " }}")?;
    writeln!(writer)?;
    writeln!(writer, "[capsule]")?;
    writeln!(writer, "file = {data_path}")?;
    writeln!(writer)?;
    writeln!(writer, "[bonds]")?;
    writeln!(writer, "file = {data_path}")?;
    writeln!(writer, "format = {:?}", "lammps_data")?;
    writeln!(writer, "auto_bond = false")?;
    writeln!(writer, "bond_radius_ratio = 1.0")?;
    writeln!(writer, "youngs_modulus = 1.0e9")?;
    writeln!(writer, "shear_modulus = 4.0e8")?;
    writeln!(writer, "bending_modulus = 1.0e9")?;
    writeln!(writer, "twist_modulus = 4.0e8")?;
    writeln!(writer)?;
    writeln!(writer, "[run]")?;
    writeln!(writer, "steps = 0")?;
    writeln!(writer, "thermo = 1")?;
    writer.flush()?;
    Ok(())
}

fn toml_string(value: &str) -> String {
    format!("{:?}", value)
}

/// GRASS export plugin that writes a DEM-BPM model and finishes the workflow.
pub struct DemBpmExportPlugin {
    /// Export configuration.
    pub config: DemBpmExportConfig,
}

/// GRASS export plugin that writes an adaptive-capsule DEM-BPM model.
pub struct DemCapsuleBpmExportPlugin {
    /// Export configuration.
    pub config: DemCapsuleBpmExportConfig,
}

impl Plugin for DemCapsuleBpmExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(self.config.clone())
            .add_resource(DemCapsuleBpmExportReport::default())
            .add_update_system(
                export_capsules_system.run_if(in_state(TangleStage::Export)),
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

impl Plugin for DemBpmExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(self.config.clone())
            .add_resource(DemBpmExportReport::default())
            .add_update_system(
                export_system.run_if(in_state(TangleStage::Export)),
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

fn export_system(
    assembly: Res<FiberAssembly>,
    config: Res<DemBpmExportConfig>,
    mut report: ResMut<DemBpmExportReport>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    let model = build_dem_bpm_model(&assembly, &config)
        .unwrap_or_else(|error| panic!("DEM-BPM discretization failed: {error}"));
    write_lammps_data(&model, &config.data_path)
        .unwrap_or_else(|error| panic!("DEM-BPM data export failed: {error}"));

    *report = DemBpmExportReport {
        particles: model.particles.len(),
        bonds: model.bonds.len(),
        junction_bonds: assembly
            .junctions
            .junctions
            .iter()
            .map(|junction| junction.anchors.len.saturating_sub(1) as usize)
            .sum(),
        atom_types: model.atom_types.clone(),
        data_path: config.data_path.clone(),
        dirt_config_path: config.dirt_config_path.clone(),
    };
    next.set(TangleStage::Done);
}

fn export_capsules_system(
    assembly: Res<FiberAssembly>,
    config: Res<DemCapsuleBpmExportConfig>,
    mut report: ResMut<DemCapsuleBpmExportReport>,
    mut next: ResMut<NextState<TangleStage>>,
) {
    let model = build_dem_capsule_bpm_model(&assembly, &config)
        .unwrap_or_else(|error| panic!("capsule DEM-BPM discretization failed: {error}"));
    write_capsule_lammps_data(&model, &config.data_path)
        .unwrap_or_else(|error| panic!("capsule DEM-BPM data export failed: {error}"));
    if let Some(path) = config.dirt_config_path.as_deref() {
        write_dirt_capsule_config(&model, &config.data_path, path, assembly.cell.periodic)
            .unwrap_or_else(|error| panic!("DIRT capsule configuration export failed: {error}"));
    }
    *report = DemCapsuleBpmExportReport {
        capsules: model.capsules.len(),
        bonds: model.bonds.len(),
        junction_bonds: assembly
            .junctions
            .junctions
            .iter()
            .map(|junction| junction.anchors.len.saturating_sub(1) as usize)
            .sum(),
        atom_types: model.atom_types.clone(),
        data_path: config.data_path.clone(),
        dirt_config_path: config.dirt_config_path.clone(),
    };
    next.set(TangleStage::Done);
}

fn orthorhombic_bounds(assembly: &FiberAssembly) -> Result<(Vec3, Vec3), ExportError> {
    let basis = assembly.cell.basis;
    let off_diagonal = [
        basis[0][1],
        basis[0][2],
        basis[1][0],
        basis[1][2],
        basis[2][0],
        basis[2][1],
    ];
    if off_diagonal.iter().any(|value| value.abs() > 1.0e-14)
        || basis[0][0] <= 0.0
        || basis[1][1] <= 0.0
        || basis[2][2] <= 0.0
    {
        return Err(ExportError::NonOrthorhombicCell);
    }
    Ok((
        assembly.cell.origin,
        [
            assembly.cell.origin[0] + basis[0][0],
            assembly.cell.origin[1] + basis[1][1],
            assembly.cell.origin[2] + basis[2][2],
        ],
    ))
}

fn sample_polyline(points: &[Vec3], requested: f64) -> Vec3 {
    let mut traversed = 0.0;
    for pair in points.windows(2) {
        let segment_length = distance(pair[0], pair[1]);
        if requested <= traversed + segment_length {
            let coordinate = ((requested - traversed) / segment_length).clamp(0.0, 1.0);
            return add(pair[0], scale(sub(pair[1], pair[0]), coordinate));
        }
        traversed += segment_length;
    }
    *points.last().expect("a fiber has at least two vertices")
}

fn nearest_anchor_particle(
    assembly: &FiberAssembly,
    model: &DemBpmModel,
    fiber_particle_ranges: &[std::ops::Range<usize>],
    anchor: tangle_core::FiberAnchor,
) -> Result<u32, ExportError> {
    let resolved = assembly
        .resolve_anchor(anchor)
        .map_err(|_| ExportError::InvalidFiberSpan(anchor.fiber.0))?;
    let fiber = &assembly.topology.fibers[resolved.fiber_index as usize];
    let first_vertex = fiber.vertices.start as usize + resolved.local_segment as usize;
    let coordinate = resolved.coordinate;
    let target = add(
        assembly.geometry.placed.positions[first_vertex],
        scale(
            sub(
                assembly.geometry.placed.positions[first_vertex + 1],
                assembly.geometry.placed.positions[first_vertex],
            ),
            coordinate,
        ),
    );
    fiber_particle_ranges[resolved.fiber_index as usize]
        .clone()
        .min_by(|first, second| {
            distance(model.particles[*first].position, target)
                .total_cmp(&distance(model.particles[*second].position, target))
        })
        .map(|index| model.particles[index].id)
        .ok_or(ExportError::InvalidFiberSpan(anchor.fiber.0))
}

fn nearest_anchor_capsule(
    assembly: &FiberAssembly,
    model: &DemCapsuleBpmModel,
    fiber_capsule_ranges: &[std::ops::Range<usize>],
    anchor: tangle_core::FiberAnchor,
) -> Result<u32, ExportError> {
    let resolved = assembly
        .resolve_anchor(anchor)
        .map_err(|_| ExportError::InvalidFiberSpan(anchor.fiber.0))?;
    let fiber = &assembly.topology.fibers[resolved.fiber_index as usize];
    let first_vertex = fiber.vertices.start as usize + resolved.local_segment as usize;
    let target = add(
        assembly.geometry.placed.positions[first_vertex],
        scale(
            sub(
                assembly.geometry.placed.positions[first_vertex + 1],
                assembly.geometry.placed.positions[first_vertex],
            ),
            resolved.coordinate,
        ),
    );
    fiber_capsule_ranges[resolved.fiber_index as usize]
        .clone()
        .min_by(|first, second| {
            distance(model.capsules[*first].position, target)
                .total_cmp(&distance(model.capsules[*second].position, target))
        })
        .map(|index| model.capsules[index].id)
        .ok_or(ExportError::InvalidFiberSpan(anchor.fiber.0))
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(value: Vec3, factor: f64) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    let delta = sub(a, b);
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberAnchor, FiberId, JunctionId, JunctionParameterId, PeriodicCell};

    #[test]
    fn exports_consecutive_fiber_bonds_without_junctions() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        for (id, y) in [(1, 0.5), (2, 1.5)] {
            assembly
                .add_fiber(
                    FiberId(id),
                    material,
                    section,
                    &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    &[[0.5, y, 1.0], [1.5, y, 1.0]],
                )
                .unwrap();
        }
        let config = DemBpmExportConfig {
            data_path: PathBuf::new(),
            dirt_config_path: None,
            density: 1_000.0,
            spacing_ratio: 1.0,
            atom_type: 1,
            bond_type: 1,
        };
        let model = build_dem_bpm_model(&assembly, &config).unwrap();
        assert_eq!(model.particles.len(), 12);
        assert_eq!(model.bonds.len(), 10);
        assert_eq!(model.atom_types.len(), 1);
        assert_eq!(model.atom_types[0].atom_type, 1);
        assert_eq!(model.atom_types[0].material_name, "fiber");
        assert_eq!(model.atom_types[0].particles, 12);
        assert!(model
            .bonds
            .iter()
            .all(
                |bond| model.particles[(bond.particle_a - 1) as usize].molecule
                    == model.particles[(bond.particle_b - 1) as usize].molecule
            ));
    }

    #[test]
    fn maps_tangle_materials_to_distinct_atom_types() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let thin = assembly.materials.add("7 um stiff fiber");
        let thick = assembly.materials.add("19 um bendy fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        for (id, material, y) in [(1, thin, 0.5), (2, thick, 1.5)] {
            assembly
                .add_fiber(
                    FiberId(id),
                    material,
                    section,
                    &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    &[[0.5, y, 1.0], [1.5, y, 1.0]],
                )
                .unwrap();
        }
        let config = DemBpmExportConfig::new(PathBuf::new())
            .with_density(1_000.0)
            .with_spacing_ratio(1.0);
        let model = build_dem_bpm_model(&assembly, &config).unwrap();

        assert_eq!(model.atom_types.len(), 2);
        assert_eq!(model.atom_types[0].atom_type, 1);
        assert_eq!(model.atom_types[0].material, thin);
        assert_eq!(model.atom_types[0].particles, 6);
        assert_eq!(model.atom_types[1].atom_type, 2);
        assert_eq!(model.atom_types[1].material, thick);
        assert_eq!(model.atom_types[1].particles, 6);
        assert!(model.particles[..6]
            .iter()
            .all(|particle| particle.atom_type == 1));
        assert!(model.particles[6..]
            .iter()
            .all(|particle| particle.atom_type == 2));

        let path = std::env::temp_dir().join(format!(
            "tangle-multi-atom-types-{}.data",
            std::process::id()
        ));
        write_lammps_data(&model, &path).unwrap();
        let data = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(data.contains("2 atom types"));
        assert!(data.contains("# atom type 1 = TANGLE material 0 \"7 um stiff fiber\""));
        assert!(data.contains("# atom type 2 = TANGLE material 1 \"19 um bendy fiber\""));
        assert!(data.contains("1 1 1 "));
        assert!(data.contains("7 2 2 "));
    }

    #[test]
    fn exports_persistent_junctions_as_inter_fiber_bonds() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        for (id, points) in [
            (1, [[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]]),
            (2, [[1.0, 0.5, 1.0], [1.0, 1.5, 1.0]]),
        ] {
            assembly
                .add_fiber(FiberId(id), material, section, &points, &points)
                .unwrap();
        }
        let law = assembly.junction_laws.add("welded");
        assembly
            .add_junction(
                JunctionId(1),
                law,
                JunctionParameterId(0),
                &[
                    FiberAnchor {
                        fiber: FiberId(1),
                        rest_arc_length: 0.5,
                        section_offset: None,
                    },
                    FiberAnchor {
                        fiber: FiberId(2),
                        rest_arc_length: 0.5,
                        section_offset: None,
                    },
                ],
            )
            .unwrap();
        let model = build_dem_bpm_model(
            &assembly,
            &DemBpmExportConfig::new(PathBuf::new())
                .with_density(1_000.0)
                .with_spacing_ratio(1.0),
        )
        .unwrap();
        assert_eq!(model.bonds.len(), 11);
        let junction_bond = model.bonds.last().unwrap();
        assert_eq!(junction_bond.bond_type, 2);
        assert_ne!(
            model.particles[(junction_bond.particle_a - 1) as usize].molecule,
            model.particles[(junction_bond.particle_b - 1) as usize].molecule
        );
    }

    #[test]
    fn exports_active_segments_as_end_to_end_capsules() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
                &[[0.25, 1.0, 1.0], [0.75, 1.0, 1.0], [1.75, 1.0, 1.0]],
            )
            .unwrap();
        let config = DemCapsuleBpmExportConfig::new(PathBuf::new())
            .with_density(1_000.0)
            .with_maximum_length_over_diameter(2.0);
        let model = build_dem_capsule_bpm_model(&assembly, &config).unwrap();

        // 0.5 m becomes two pieces and 1.0 m becomes three at a 0.4 m ceiling.
        assert_eq!(model.capsules.len(), 5);
        assert_eq!(model.bonds.len(), 4);
        assert!(model
            .capsules
            .iter()
            .all(|capsule| capsule.half_length * 2.0 <= 0.4 + 1.0e-12));
        assert!(model
            .capsules
            .iter()
            .all(|capsule| capsule.axis == [1.0, 0.0, 0.0]));
        for pair in model.capsules.windows(2) {
            let center_distance = distance(pair[0].position, pair[1].position);
            assert!((center_distance - pair[0].half_length - pair[1].half_length).abs() < 1.0e-12);
        }

        let path =
            std::env::temp_dir().join(format!("tangle-capsules-{}.data", std::process::id()));
        write_capsule_lammps_data(&model, &path).unwrap();
        let data = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(data.contains("Atoms # bpm/sphere"));
        assert!(data.contains("\nCapsules\n"));
        assert!(data.contains("# atom-id half-length axis-x axis-y axis-z"));
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
        let model = build_dem_capsule_bpm_model(
            &assembly,
            &DemCapsuleBpmExportConfig::new(PathBuf::new()).with_density(1_000.0),
        )
        .unwrap();

        assert_eq!(model.capsules.len(), 1);
        let position = model.capsules[0].position;
        assert!((position[0] - 0.0).abs() < 1.0e-12, "{position:?}");
        assert!((position[1] - 0.25).abs() < 1.0e-12, "{position:?}");
        assert!((position[2] - 1.0).abs() < 1.0e-12, "{position:?}");
    }
}
