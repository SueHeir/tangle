//! Export TANGLE fiber assemblies into derived solver representations.

#![warn(missing_docs)]

mod bpm;
mod bridges;
mod ovito;
mod puma;
mod trajectory;

pub use bpm::{build_bpm_model, write_bpm_lammps_data, BpmExportPlugin, BpmExportReport};
pub use bridges::{junction_bridges, JunctionBridge, DEFAULT_BOND_RADIUS_RATIO};
pub use ovito::{
    write_ovito_assembly_frame, write_ovito_dump_frame, write_ovito_view_script, OvitoColoring,
    OvitoRepresentation, OvitoTrajectoryConfig, OvitoTrajectoryReport,
};
pub use puma::{
    voxelize_labels, write_puma_bundle, PumaExportError, PumaExportReport, PumaVoxelExportConfig,
    VoxelLabels, PUMA_BUNDLE_SCHEMA_VERSION,
};
pub use trajectory::OvitoTrajectoryPlugin;

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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

/// One fiber type of a DEM export: its atom type and the bond type joining
/// consecutive particles along its fibers.
///
/// By default every TANGLE material is a fiber type. With
/// [`BpmExportConfig::with_fiber_types`] the caller groups the fibers instead
/// (the CT pipeline's fiber types, say); `material` is then the material of
/// the type's first fiber and `material_name` the type's name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemAtomType {
    /// One-based atom type written to the LAMMPS data file.
    pub atom_type: u32,
    /// One-based bond type of the bonds along this type's fibers.
    pub bond_type: u32,
    /// Solver-neutral TANGLE material identifier.
    pub material: MaterialId,
    /// Human-readable TANGLE material or fiber-type name.
    pub material_name: String,
    /// Number of particles exported with this type.
    pub particles: usize,
}

/// Caller-defined fiber types for a BPM export: each fiber's zero-based type
/// and each type's name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BpmFiberTypes {
    /// One zero-based type per fiber, in assembly order.
    pub of_fiber: Vec<u32>,
    /// One name per type.
    pub names: Vec<String>,
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

/// Centerline discretization used by the solver-neutral BPM exporter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BpmExportMode {
    /// Constant sphere spacing along the complete fiber arc. Any unused arc
    /// length is divided equally between the two fiber ends.
    SpheresExact,
    /// Preserve every centerline vertex and adjust sphere spacing separately
    /// inside each source segment.
    SpheresDynamic,
    /// Export one spherocylinder for every active TANGLE segment.
    #[default]
    SpherocylindersExact,
    /// Tile every source segment with the shortest active segment length,
    /// followed by one shorter end remainder when necessary (one at least
    /// the minimum capsule length; a shorter one joins the piece before it).
    SpherocylindersConstant,
}

impl BpmExportMode {
    /// Stable kebab-case name used by Python and documentation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SpheresExact => "spheres-exact",
            Self::SpheresDynamic => "spheres-dynamic",
            Self::SpherocylindersExact => "spherocylinders-exact",
            Self::SpherocylindersConstant => "spherocylinders-constant",
        }
    }
}

/// Unified BPM discretization and LAMMPS-data output configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct BpmExportConfig {
    /// LAMMPS-data output path.
    pub data_path: PathBuf,
    /// Geometry representation and sampling policy.
    pub mode: BpmExportMode,
    /// Sphere center spacing divided by fiber radius. Sphere modes support
    /// values from one third through one radius, inclusive. Below two radii,
    /// spheres of one fiber overlap neighbours they are not bonded to (at one
    /// third, the next five), so the input script must exclude pair forces
    /// within a fiber, e.g. `neigh_modify exclude molecule/intra all`.
    pub sphere_spacing_over_radius: f64,
    /// Particle material density.
    pub density: f64,
    /// Atom type of the first fiber type; the others follow consecutively.
    pub atom_type: u32,
    /// Bond type along the fibers of the first fiber type; the others follow
    /// consecutively, then one bond type per junction law.
    pub bond_type: u32,
    /// Fiber types to export with their own atom and bond types. `None`: one
    /// fiber type per TANGLE material.
    pub fiber_types: Option<BpmFiberTypes>,
    /// Shortest spherocylinder, in fiber diameters (default 1). Shorter
    /// segments are merged with the next ones, so a capsule doesn't overlap
    /// the unbonded capsule two along its fiber. Zero keeps every segment.
    pub min_capsule_length_over_diameter: f64,
}

impl BpmExportConfig {
    /// Creates an exact active-segment spherocylinder export.
    pub fn new(data_path: impl Into<PathBuf>) -> Self {
        Self {
            data_path: data_path.into(),
            mode: BpmExportMode::SpherocylindersExact,
            sphere_spacing_over_radius: 1.0 / 3.0,
            density: 1.0,
            atom_type: 1,
            bond_type: 1,
            fiber_types: None,
            min_capsule_length_over_diameter: 1.0,
        }
    }

    /// Sets the shortest spherocylinder, in fiber diameters (zero: keep every
    /// segment).
    pub fn with_min_capsule_length_over_diameter(mut self, ratio: f64) -> Self {
        self.min_capsule_length_over_diameter = ratio;
        self
    }

    /// Gives each fiber a zero-based type (`of_fiber`, in assembly order) and
    /// each type a name. Each type gets its own atom type and bond type.
    pub fn with_fiber_types(mut self, of_fiber: Vec<u32>, names: Vec<String>) -> Self {
        self.fiber_types = Some(BpmFiberTypes { of_fiber, names });
        self
    }

    /// Selects the exported geometry representation and sampling policy.
    pub fn with_mode(mut self, mode: BpmExportMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets sphere center spacing as a multiple of fiber radius.
    pub fn with_sphere_spacing_over_radius(mut self, spacing: f64) -> Self {
        self.sphere_spacing_over_radius = spacing;
        self
    }

    /// Sets particle material density.
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }
}

/// Geometry produced by the unified BPM discretizer.
#[derive(Clone, Debug, PartialEq)]
pub enum BpmModel {
    /// Bonded spheres.
    Spheres(DemBpmModel),
    /// Bonded spherocylinders.
    Spherocylinders(DemCapsuleBpmModel),
}

impl BpmModel {
    /// Number of sphere or spherocylinder particles.
    pub fn particles(&self) -> usize {
        match self {
            Self::Spheres(model) => model.particles.len(),
            Self::Spherocylinders(model) => model.capsules.len(),
        }
    }

    /// Number of intra-fiber and junction bonds.
    pub fn bonds(&self) -> usize {
        match self {
            Self::Spheres(model) => model.bonds.len(),
            Self::Spherocylinders(model) => model.bonds.len(),
        }
    }

    /// Material-to-atom-type mappings used by the particles.
    pub fn atom_types(&self) -> &[DemAtomType] {
        match self {
            Self::Spheres(model) => &model.atom_types,
            Self::Spherocylinders(model) => &model.atom_types,
        }
    }
}

/// DEM-BPM discretization and output configuration.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DemBpmExportConfig {
    /// LAMMPS-data output path.
    pub data_path: PathBuf,
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

#[cfg(test)]
impl DemBpmExportConfig {
    /// Creates a conventional bonded-sphere export configuration.
    pub fn new(data_path: impl Into<PathBuf>) -> Self {
        Self {
            data_path: data_path.into(),
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

/// Failure while discretizing or writing a DEM-BPM model.
#[derive(Debug)]
pub enum ExportError {
    /// Density was not positive and finite.
    InvalidDensity(f64),
    /// Spacing ratio was outside the supported interval.
    InvalidSpacingRatio(f64),
    /// Radius-based sphere spacing was outside the supported interval.
    InvalidSphereSpacingOverRadius(f64),
    /// Shortest-capsule ratio was negative or not finite.
    InvalidCapsuleLengthRatio(f64),
    /// Caller-defined fiber types did not match the assembly's fibers.
    InvalidFiberTypes(String),
    /// First atom type was zero; LAMMPS types are one-based.
    InvalidAtomType(u32),
    /// DEM sphere export currently requires circular fibers.
    NonCircularSection(u32),
    /// Fiber section lookup failed.
    MissingSection(u32),
    /// Fiber geometry lookup failed.
    InvalidFiberSpan(u32),
    /// A constant spherocylinder length could not be found.
    NoActiveSegments,
    /// Binder-bridge radius ratio was not positive and finite.
    InvalidBondRadiusRatio(f64),
    /// Junction anchor lookup failed.
    InvalidJunction(u32),
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
            Self::InvalidSphereSpacingOverRadius(value) => write!(
                f,
                "sphere_spacing_over_radius must be in [1/3, 1], got {value}"
            ),
            Self::InvalidCapsuleLengthRatio(value) => write!(
                f,
                "min_capsule_length_over_diameter must be zero or more, got {value}"
            ),
            Self::InvalidFiberTypes(reason) => write!(f, "invalid fiber types: {reason}"),
            Self::InvalidAtomType(value) => {
                write!(f, "first atom type must be positive, got {value}")
            }
            Self::NonCircularSection(id) => {
                write!(f, "fiber {id} is oval, which BPM export does not support")
            }
            Self::MissingSection(id) => write!(f, "fiber {id} references a missing section"),
            Self::InvalidFiberSpan(id) => write!(f, "fiber {id} has an invalid placed span"),
            Self::NoActiveSegments => {
                f.write_str("constant spherocylinder export requires an active segment")
            }
            Self::InvalidBondRadiusRatio(value) => {
                write!(f, "bond radius ratio must be positive, got {value}")
            }
            Self::InvalidJunction(id) => write!(f, "junction {id} has an invalid anchor span"),
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
pub(crate) fn build_dem_bpm_model(
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
                    bond_type: config.bond_type,
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

/// Writes a LAMMPS data file using the `bpm/sphere` atom style.
pub(crate) fn write_lammps_data(model: &DemBpmModel, path: &Path) -> Result<(), ExportError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "TANGLE DEM-BPM fiber export")?;
    for mapping in &model.atom_types {
        let name = mapping.material_name.replace(['\r', '\n'], " ");
        writeln!(
            writer,
            "# atom type {} and bond type {} = fiber type {:?} ({} particles)",
            mapping.atom_type, mapping.bond_type, name, mapping.particles
        )?;
    }
    // The column legends live in the header: LAMMPS and OVITO read every
    // line after a section's blank line as data, so a comment line there
    // is rejected.
    writeln!(
        writer,
        "# Atoms columns: id molecule type diameter density x y z"
    )?;
    writeln!(writer, "# Bonds columns: id type atom1 atom2")?;
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
        .atom_types
        .iter()
        .map(|mapping| mapping.bond_type)
        .chain(model.bonds.iter().map(|bond| bond.bond_type))
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
    // LAMMPS refuses a Bonds section in a file that declares no bonds.
    if !model.bonds.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "Bonds")?;
        writeln!(writer)?;
        for bond in &model.bonds {
            writeln!(
                writer,
                "{} {} {} {}",
                bond.id, bond.bond_type, bond.particle_a, bond.particle_b
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Writes bonded capsules as a LAMMPS data file for `atom_style bacillus`.
///
/// Each capsule is one atom at its center of mass. Its `Atoms` row is
/// `id molecule type 1 density x y z` (the `1` marks it as a capsule, and
/// LAMMPS turns the density into the capsule's mass). Its `Bacilli` row is
/// `id Ixx Iyy Izz Ixy Ixz Iyz px py pz diameter`: the inertia tensor about
/// the center and the pole (half the cylinder axis, from the center to one
/// cap's center), both in the box frame. Bonds join the centers of
/// consecutive capsules along a fiber.
pub(crate) fn write_capsule_lammps_data(
    model: &DemCapsuleBpmModel,
    path: &Path,
) -> Result<(), ExportError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "TANGLE spherocylinder DEM-BPM fiber export (LAMMPS atom_style bacillus)"
    )?;
    for mapping in &model.atom_types {
        let name = mapping.material_name.replace(['\r', '\n'], " ");
        writeln!(
            writer,
            "# atom type {} and bond type {} = fiber type {:?} ({} capsules)",
            mapping.atom_type, mapping.bond_type, name, mapping.particles
        )?;
    }
    // Legends in the header, as in `write_lammps_data`.
    writeln!(
        writer,
        "# Atoms columns: id molecule type bacillus density x y z"
    )?;
    writeln!(
        writer,
        "# Bacilli columns: id Ixx Iyy Izz Ixy Ixz Iyz pole-x pole-y pole-z diameter (box frame; the pole is half the axis)"
    )?;
    writeln!(writer, "# Bonds columns: id type atom1 atom2")?;
    writeln!(writer)?;
    writeln!(writer, "{} atoms", model.capsules.len())?;
    writeln!(writer, "{} bacilli", model.capsules.len())?;
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
        .atom_types
        .iter()
        .map(|mapping| mapping.bond_type)
        .chain(model.bonds.iter().map(|bond| bond.bond_type))
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
    writeln!(writer, "Atoms # bacillus")?;
    writeln!(writer)?;
    for capsule in &model.capsules {
        writeln!(
            writer,
            "{} {} {} 1 {:.17e} {:.17e} {:.17e} {:.17e}",
            capsule.id,
            capsule.molecule,
            capsule.atom_type,
            capsule.density,
            capsule.position[0],
            capsule.position[1],
            capsule.position[2]
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "Bacilli")?;
    writeln!(writer)?;
    for capsule in &model.capsules {
        let inertia = capsule_inertia(capsule);
        let pole = scale(capsule.axis, capsule.half_length);
        writeln!(
            writer,
            "{} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
            capsule.id,
            inertia[0][0],
            inertia[1][1],
            inertia[2][2],
            inertia[0][1],
            inertia[0][2],
            inertia[1][2],
            pole[0],
            pole[1],
            pole[2],
            capsule.diameter
        )?;
    }
    // LAMMPS refuses a Bonds section in a file that declares no bonds.
    if !model.bonds.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "Bonds")?;
        writeln!(writer)?;
        for bond in &model.bonds {
            writeln!(
                writer,
                "{} {} {} {}",
                bond.id, bond.bond_type, bond.particle_a, bond.particle_b
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Inertia tensor of a capsule about its center of mass, in the box frame.
pub(crate) fn capsule_inertia(capsule: &DemCapsule) -> [[f64; 3]; 3] {
    use std::f64::consts::PI;
    let r = 0.5 * capsule.diameter;
    let length = 2.0 * capsule.half_length;
    let cylinder = capsule.density * PI * r * r * length;
    let cap = capsule.density * 2.0 / 3.0 * PI * r * r * r; // one hemisphere
    let axial = 0.5 * cylinder * r * r + 2.0 * 0.4 * cap * r * r;
    // Each cap about its own centroid (3r/8 from its flat face), moved to the
    // capsule's center.
    let cap_centroid = 0.5 * length + 3.0 * r / 8.0;
    let transverse = cylinder * (3.0 * r * r + length * length) / 12.0
        + 2.0 * (0.4 * cap * r * r - cap * (3.0 * r / 8.0).powi(2) + cap * cap_centroid.powi(2));
    let u = capsule.axis;
    let mut tensor = [[0.0; 3]; 3];
    for (a, row) in tensor.iter_mut().enumerate() {
        for (b, value) in row.iter_mut().enumerate() {
            let identity = if a == b { transverse } else { 0.0 };
            *value = identity + (axial - transverse) * u[a] * u[b];
        }
    }
    tensor
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
        if segment_length <= f64::EPSILON {
            continue;
        }
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
    // The capsule whose axis passes nearest the anchor holds it; a long
    // capsule's center can be farther away than a short neighbour's. At a
    // joint, where two capsules tie, the nearer center wins.
    let key = |index: usize| {
        let capsule = &model.capsules[index];
        (
            axis_distance(capsule, target),
            distance(capsule.position, target),
        )
    };
    fiber_capsule_ranges[resolved.fiber_index as usize]
        .clone()
        .min_by(|first, second| {
            let (first, second) = (key(*first), key(*second));
            first
                .0
                .total_cmp(&second.0)
                .then(first.1.total_cmp(&second.1))
        })
        .map(|index| model.capsules[index].id)
        .ok_or(ExportError::InvalidFiberSpan(anchor.fiber.0))
}

/// Distance from `point` to the capsule's axis segment, center ± half length.
fn axis_distance(capsule: &DemCapsule, point: Vec3) -> f64 {
    let offset = sub(point, capsule.position);
    let along =
        (offset[0] * capsule.axis[0] + offset[1] * capsule.axis[1] + offset[2] * capsule.axis[2])
            .clamp(-capsule.half_length, capsule.half_length);
    distance(offset, scale(capsule.axis, along))
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
        let thin = assembly.materials.add("fine_7um");
        let thick = assembly.materials.add("coarse_19um");
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
        assert!(data.contains("# atom type 1 and bond type 1 = fiber type \"fine_7um\""));
        assert!(data.contains("# atom type 2 and bond type 1 = fiber type \"coarse_19um\""));
        assert!(data.contains("1 1 1 "));
        assert!(data.contains("7 2 2 "));
        let lines: Vec<&str> = data.lines().collect();
        let first_section = lines
            .iter()
            .position(|line| *line == "Atoms # bpm/sphere")
            .unwrap();
        assert!(lines[first_section..]
            .iter()
            .all(|line| !line.starts_with('#')));
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
}
