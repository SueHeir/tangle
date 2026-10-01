//! Solver-neutral VTK ImageData bundles for direct use with PuMA.

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use tangle_characterize::{characterize_assembly, write_analysis_json, AnalysisWriteError};
use tangle_core::{FiberAssembly, FiberId, Section, Vec3};

use crate::{junction_bridges, JunctionBridge};

/// Version of the on-disk PuMA bundle schema.
pub const PUMA_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Options for rasterizing one TANGLE assembly onto a cubic voxel grid.
#[derive(Clone, Debug, PartialEq)]
pub struct PumaVoxelExportConfig {
    /// Directory that receives the complete bundle.
    pub output_directory: PathBuf,
    /// Cubic voxel edge length in the assembly's length units.
    pub voxel_size: f64,
    /// Whether to write the optional stable owner-fiber image.
    pub include_fiber_ids: bool,
    /// Whether to write the optional smooth capsule-interface image.
    pub include_interface: bool,
    /// Absolute signed-distance tolerance for ownership ties.
    pub ambiguity_tolerance: f64,
    /// Binder-bridge radius as a fraction of the thinner fiber's radius.
    /// `Some` voxelizes every persistent junction as binder (its own phase,
    /// plus the bond-ID and binder-occupancy images); `None` leaves binder
    /// out and the fiber images are exactly as without junctions.
    pub bond_radius_ratio: Option<f64>,
}

impl PumaVoxelExportConfig {
    /// Creates a bundle configuration with both diagnostic images enabled.
    pub fn new(output_directory: impl Into<PathBuf>, voxel_size: f64) -> Self {
        Self {
            output_directory: output_directory.into(),
            voxel_size,
            include_fiber_ids: true,
            include_interface: true,
            ambiguity_tolerance: voxel_size * 1.0e-6,
            bond_radius_ratio: None,
        }
    }

    /// Voxelizes persistent junctions as binder bridges of the given radius
    /// ratio, or leaves them out with `None`.
    pub fn with_bonds(mut self, radius_ratio: Option<f64>) -> Self {
        self.bond_radius_ratio = radius_ratio;
        self
    }

    /// Enables or disables the stable owner-fiber image.
    pub fn with_fiber_ids(mut self, include: bool) -> Self {
        self.include_fiber_ids = include;
        self
    }

    /// Enables or disables the smooth capsule-interface image.
    pub fn with_interface(mut self, include: bool) -> Self {
        self.include_interface = include;
        self
    }

    /// Sets the signed-distance tolerance used to identify ownership ties.
    pub fn with_ambiguity_tolerance(mut self, tolerance: f64) -> Self {
        self.ambiguity_tolerance = tolerance;
        self
    }
}

/// Files and diagnostics produced by [`write_puma_bundle`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PumaExportReport {
    /// Bundle directory.
    pub output_directory: PathBuf,
    /// Number of cells along x, y, and z.
    pub voxel_counts: [usize; 3],
    /// Total number of grid cells.
    pub total_voxels: usize,
    /// Number of cells assigned to a solid material.
    pub occupied_voxels: usize,
    /// Occupied-voxel volume fraction.
    pub voxel_volume_fraction: f64,
    /// Number of occupied cells with a cross-fiber ownership tie.
    pub ambiguous_voxels: usize,
    /// Number of cells assigned to the binder phase (included in
    /// `occupied_voxels`).
    #[serde(default)]
    pub binder_voxels: usize,
    /// Number of binder bridges voxelized.
    #[serde(default)]
    pub bonds: usize,
    /// Primary phase and orientation image.
    pub domain_path: PathBuf,
    /// Optional owner-fiber image.
    pub fiber_ids_path: Option<PathBuf>,
    /// Optional smooth interface image.
    pub interface_path: Option<PathBuf>,
    /// Owner-junction image, when bonds are exported.
    #[serde(default)]
    pub bond_ids_path: Option<PathBuf>,
    /// Smooth binder-occupancy image, when bonds are exported.
    #[serde(default)]
    pub binder_interface_path: Option<PathBuf>,
    /// Bundle manifest.
    pub manifest_path: PathBuf,
    /// Native TANGLE characterization.
    pub analysis_path: PathBuf,
}

/// Failure while validating, voxelizing, or writing a PuMA bundle.
#[derive(Debug)]
pub enum PumaExportError {
    /// The requested voxel size or tolerance is invalid.
    InvalidConfig(String),
    /// PuMA ImageData currently requires a diagonal orthorhombic cell.
    NonOrthorhombicCell,
    /// The voxel size does not tile one or more cell edges.
    IncommensurateGrid {
        /// Requested cubic voxel size.
        voxel_size: f64,
        /// Cell edge lengths.
        lengths: Vec3,
        /// Nearest integer voxel counts.
        suggested_counts: [usize; 3],
    },
    /// A section the voxel exporter cannot represent. Not produced since
    /// elliptical sections are voxelized with their directors.
    UnsupportedSection {
        /// Zero-based section-table index.
        section_index: usize,
    },
    /// A phase identifier cannot be represented by the VTI UInt16 image.
    TooManyMaterials(usize),
    /// Grid allocation overflowed the host index range.
    GridTooLarge([usize; 3]),
    /// The assembly's own invariants are invalid.
    InvalidAssembly(String),
    /// Filesystem or VTI output failure.
    Io(std::io::Error),
    /// JSON serialization failure.
    Json(serde_json::Error),
    /// Native analysis output failure.
    Analysis(AnalysisWriteError),
}

/// The voxel size nearest `voxel_size` that divides every cell length into
/// whole voxels, when the lengths share one (equal or commensurate edges).
fn tiling_voxel_size(lengths: Vec3, voxel_size: f64) -> Option<(f64, [usize; 3])> {
    let shortest = (0..3).min_by(|&a, &b| lengths[a].total_cmp(&lengths[b]))?;
    let base = (lengths[shortest] / voxel_size).round().max(1.0);
    // Try the shortest edge's nearest counts first, then its neighbours.
    for step in [0.0, 1.0, -1.0, 2.0, -2.0] {
        let count = base + step;
        if count < 1.0 {
            continue;
        }
        let size = lengths[shortest] / count;
        let mut counts = [0usize; 3];
        let tiles = (0..3).all(|axis| {
            let real = lengths[axis] / size;
            counts[axis] = real.round() as usize;
            (real - real.round()).abs() / real.max(1.0) <= 1.0e-9
        });
        if tiles {
            return Some((size, counts));
        }
    }
    None
}

impl fmt::Display for PumaExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid PuMA export: {message}"),
            Self::NonOrthorhombicCell => write!(
                formatter,
                "PuMA VTI export currently requires a diagonal orthorhombic cell"
            ),
            Self::IncommensurateGrid {
                voxel_size,
                lengths,
                suggested_counts,
            } => {
                write!(
                    formatter,
                    "voxel size {voxel_size:.6e} does not divide the cell lengths [{:.6e}, {:.6e}, {:.6e}] into whole voxels; ",
                    lengths[0], lengths[1], lengths[2]
                )?;
                match tiling_voxel_size(*lengths, *voxel_size) {
                    Some((size, counts)) => write!(
                        formatter,
                        "try voxel_size={size:.6e} ({} x {} x {} voxels)",
                        counts[0], counts[1], counts[2]
                    ),
                    None => write!(
                        formatter,
                        "nearest counts are {suggested_counts:?}; make each cell length a multiple of the voxel size"
                    ),
                }
            }
            Self::UnsupportedSection { section_index } => write!(
                formatter,
                "section {section_index} cannot be voxelized"
            ),
            Self::TooManyMaterials(count) => {
                write!(formatter, "{count} materials exceed the UInt16 phase-ID limit")
            }
            Self::GridTooLarge(counts) => {
                write!(formatter, "voxel grid {counts:?} exceeds the host index range")
            }
            Self::InvalidAssembly(message) => write!(formatter, "invalid assembly: {message}"),
            Self::Io(error) => write!(formatter, "PuMA bundle output failed: {error}"),
            Self::Json(error) => write!(formatter, "PuMA manifest serialization failed: {error}"),
            Self::Analysis(error) => write!(formatter, "PuMA analysis output failed: {error}"),
        }
    }
}

impl Error for PumaExportError {}

impl From<std::io::Error> for PumaExportError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for PumaExportError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<AnalysisWriteError> for PumaExportError {
    fn from(error: AnalysisWriteError) -> Self {
        Self::Analysis(error)
    }
}

#[derive(Clone, Debug, Serialize)]
struct PumaManifest {
    schema_version: u32,
    exporter: &'static str,
    units: ManifestUnits,
    cell: ManifestCell,
    grid: ManifestGrid,
    arrays: Vec<ManifestArray>,
    materials: Vec<ManifestMaterial>,
    fibers: Vec<ManifestFiber>,
    #[serde(skip_serializing_if = "Option::is_none")]
    binder: Option<ManifestBinder>,
    voxelization: ManifestVoxelization,
    source: ManifestSource,
    files: Vec<ManifestFile>,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestUnits {
    length: &'static str,
    orientation: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestCell {
    origin: Vec3,
    basis: [Vec3; 3],
    lengths: Vec3,
    periodic: [bool; 3],
    volume: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestGrid {
    voxel_size: f64,
    voxel_counts: [usize; 3],
    total_voxels: usize,
    occupied_voxels: usize,
    voxel_volume_fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestArray {
    file: String,
    name: &'static str,
    vtk_type: &'static str,
    components: usize,
    association: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestMaterial {
    phase_id: u16,
    material_id: u32,
    name: String,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestFiber {
    export_id: u32,
    fiber_id: u32,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestBinder {
    phase_id: u16,
    name: &'static str,
    geometry: &'static str,
    radius_ratio: f64,
    occupancy_rule: &'static str,
    interface_rule: &'static str,
    bonds: Vec<ManifestBond>,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestBond {
    export_id: u32,
    junction_id: u32,
    law: String,
    radius: f64,
    length: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestVoxelization {
    geometry: &'static str,
    occupancy_rule: &'static str,
    ownership_rule: &'static str,
    orientation_rule: &'static str,
    periodic_image_policy: &'static str,
    ambiguity_tolerance: f64,
    ambiguous_voxels: usize,
    interface_threshold: Option<u8>,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestSource {
    assembly_hash_fnv1a64: String,
    provenance_generator: String,
    provenance_seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestFile {
    path: String,
    bytes: u64,
    hash_fnv1a64: String,
}

struct VoxelFields {
    phase: Vec<u16>,
    orientation: Vec<[f32; 3]>,
    owner_export_id: Vec<u32>,
    interface: Vec<u8>,
    ambiguous: Vec<bool>,
    /// Owner-junction export ID per cell; empty without bonds.
    bond_id: Vec<u32>,
    /// Smooth binder occupancy outside the fibers; empty without bonds.
    binder_interface: Vec<u8>,
}

/// Per-voxel labels on the PuMA grid, computed in memory without writing a
/// bundle.
///
/// The grid, ownership and binder rules are exactly those of
/// [`write_puma_bundle`]. Every image is flat with x fastest, then y, then z.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelLabels {
    /// Number of cells along x, y, and z.
    pub voxel_counts: [usize; 3],
    /// Cubic voxel edge length.
    pub voxel_size: f64,
    /// Corner of the grid, which is the cell origin.
    pub origin: Vec3,
    /// Phase per cell: material index + 1, the binder phase after every
    /// material, or 0 for void.
    pub phase: Vec<u16>,
    /// Owner fiber per cell as dense fiber index + 1, or 0 for void and
    /// binder.
    pub fiber: Vec<u32>,
    /// Owner junction per binder cell as junction index + 1, or 0. Empty
    /// when bonds are off.
    pub bond: Vec<u32>,
}

/// Rasterizes an assembly onto the PuMA grid and returns the labels in
/// memory. `bond_radius_ratio` voxelizes persistent junctions as binder, as
/// [`PumaVoxelExportConfig::with_bonds`] does.
pub fn voxelize_labels(
    assembly: &FiberAssembly,
    voxel_size: f64,
    bond_radius_ratio: Option<f64>,
) -> Result<VoxelLabels, PumaExportError> {
    let config =
        PumaVoxelExportConfig::new(PathBuf::new(), voxel_size).with_bonds(bond_radius_ratio);
    let (_, counts, total_voxels, bridges) = prepare_grid(assembly, &config)?;
    let fields = voxelize_with_bonds(assembly, &config, counts, total_voxels, &bridges)?;
    Ok(VoxelLabels {
        voxel_counts: counts,
        voxel_size,
        origin: assembly.cell.origin,
        phase: fields.phase,
        fiber: fields.owner_export_id,
        bond: fields.bond_id,
    })
}

/// Characterizes and rasterizes an assembly into a versioned bundle that can
/// be imported with `pumapy.import_vti` without a TANGLE-specific adapter.
pub fn write_puma_bundle(
    assembly: &FiberAssembly,
    config: &PumaVoxelExportConfig,
) -> Result<PumaExportReport, PumaExportError> {
    let (lengths, counts, total_voxels, bridges) = prepare_grid(assembly, config)?;
    fs::create_dir_all(&config.output_directory)?;
    let fields = voxelize_with_bonds(assembly, config, counts, total_voxels, &bridges)?;
    let binder_phase = binder_phase_id(assembly);
    let occupied_voxels = fields.phase.iter().filter(|phase| **phase != 0).count();
    let binder_voxels = if config.bond_radius_ratio.is_some() {
        fields
            .phase
            .iter()
            .filter(|phase| **phase == binder_phase)
            .count()
    } else {
        0
    };
    let ambiguous_voxels = fields.ambiguous.iter().filter(|value| **value).count();
    let voxel_volume_fraction = occupied_voxels as f64 / total_voxels as f64;

    let domain_path = config.output_directory.join("domain.vti");
    write_domain_vti(
        &domain_path,
        assembly.cell.origin,
        config.voxel_size,
        counts,
        &fields.phase,
        &fields.orientation,
    )?;

    let fiber_ids_path = config
        .include_fiber_ids
        .then(|| config.output_directory.join("fiber_ids.vti"));
    if let Some(path) = &fiber_ids_path {
        write_scalar_vti_u32(
            path,
            assembly.cell.origin,
            config.voxel_size,
            counts,
            "fiber_id",
            &fields.owner_export_id,
        )?;
    }

    let interface_path = config
        .include_interface
        .then(|| config.output_directory.join("interface.vti"));
    if let Some(path) = &interface_path {
        write_scalar_vti_u8(
            path,
            assembly.cell.origin,
            config.voxel_size,
            counts,
            "interface_grayscale",
            &fields.interface,
        )?;
    }

    let bonds_enabled = config.bond_radius_ratio.is_some();
    let bond_ids_path = bonds_enabled.then(|| config.output_directory.join("bond_ids.vti"));
    if let Some(path) = &bond_ids_path {
        write_scalar_vti_u32(
            path,
            assembly.cell.origin,
            config.voxel_size,
            counts,
            "bond_id",
            &fields.bond_id,
        )?;
    }
    let binder_interface_path =
        bonds_enabled.then(|| config.output_directory.join("binder_interface.vti"));
    if let Some(path) = &binder_interface_path {
        write_scalar_vti_u8(
            path,
            assembly.cell.origin,
            config.voxel_size,
            counts,
            "binder_grayscale",
            &fields.binder_interface,
        )?;
    }

    let analysis_path = config.output_directory.join("tangle_analysis.json");
    write_analysis_json(&characterize_assembly(assembly), &analysis_path)?;

    let mut arrays = vec![
        ManifestArray {
            file: "domain.vti".into(),
            name: "phase_id",
            vtk_type: "UInt16",
            components: 1,
            association: "CellData",
        },
        ManifestArray {
            file: "domain.vti".into(),
            name: "orientation",
            vtk_type: "Float32",
            components: 3,
            association: "CellData",
        },
    ];
    if fiber_ids_path.is_some() {
        arrays.push(ManifestArray {
            file: "fiber_ids.vti".into(),
            name: "fiber_id",
            vtk_type: "UInt32",
            components: 1,
            association: "CellData",
        });
    }
    if interface_path.is_some() {
        arrays.push(ManifestArray {
            file: "interface.vti".into(),
            name: "interface_grayscale",
            vtk_type: "UInt8",
            components: 1,
            association: "CellData",
        });
    }

    if bonds_enabled {
        arrays.push(ManifestArray {
            file: "bond_ids.vti".into(),
            name: "bond_id",
            vtk_type: "UInt32",
            components: 1,
            association: "CellData",
        });
        arrays.push(ManifestArray {
            file: "binder_interface.vti".into(),
            name: "binder_grayscale",
            vtk_type: "UInt8",
            components: 1,
            association: "CellData",
        });
    }

    let mut bundle_paths = vec![domain_path.clone(), analysis_path.clone()];
    bundle_paths.extend(fiber_ids_path.iter().cloned());
    bundle_paths.extend(interface_path.iter().cloned());
    bundle_paths.extend(bond_ids_path.iter().cloned());
    bundle_paths.extend(binder_interface_path.iter().cloned());
    let files = bundle_paths
        .iter()
        .map(|path| manifest_file(path, &config.output_directory))
        .collect::<Result<Vec<_>, _>>()?;
    let assembly_bytes = serde_json::to_vec(assembly)?;
    let manifest = PumaManifest {
        schema_version: PUMA_BUNDLE_SCHEMA_VERSION,
        exporter: "tangle_export::write_puma_bundle",
        units: ManifestUnits {
            length: "assembly_length_unit",
            orientation: "dimensionless_unit_vector",
        },
        cell: ManifestCell {
            origin: assembly.cell.origin,
            basis: assembly.cell.basis,
            lengths,
            periodic: assembly.cell.periodic,
            volume: assembly.cell.signed_volume().abs(),
        },
        grid: ManifestGrid {
            voxel_size: config.voxel_size,
            voxel_counts: counts,
            total_voxels,
            occupied_voxels,
            voxel_volume_fraction,
        },
        arrays,
        materials: assembly
            .materials
            .entries
            .iter()
            .enumerate()
            .map(|(index, material)| ManifestMaterial {
                phase_id: (index + 1) as u16,
                material_id: index as u32,
                name: material.name.clone(),
            })
            .collect(),
        fibers: assembly
            .topology
            .fibers
            .iter()
            .enumerate()
            .map(|(index, fiber)| ManifestFiber {
                export_id: index as u32 + 1,
                fiber_id: fiber.id.0,
            })
            .collect(),
        binder: config.bond_radius_ratio.map(|radius_ratio| ManifestBinder {
            phase_id: binder_phase,
            name: "binder",
            geometry: "round capsule between the two anchored centerline points of each junction; radius is radius_ratio times the thinner fiber's radius (short semi-axis for ovals)",
            radius_ratio,
            occupancy_rule: "voxel center inside a bridge and outside every fiber; fibers win where both overlap",
            interface_rule: "binder_grayscale is the bridge's smooth occupancy capped at 255 minus interface_grayscale, so fiber and binder occupancies add to at most 255",
            bonds: bridges
                .iter()
                .map(|bridge| ManifestBond {
                    export_id: bridge.junction_index as u32 + 1,
                    junction_id: bridge.junction_id,
                    law: assembly
                        .junction_laws
                        .entries
                        .get(bridge.law as usize)
                        .map(|law| law.name.clone())
                        .unwrap_or_default(),
                    radius: bridge.radius,
                    length: bridge.length(),
                })
                .collect(),
        }),
        voxelization: ManifestVoxelization {
            geometry:
                "piecewise-linear centerline swept by a circular radius with spherical end caps",
            occupancy_rule: "voxel center lies inside at least one swept segment",
            ownership_rule: "minimum signed capsule distance, then stable source FiberId",
            orientation_rule:
                "closest segment tangent; incident same-fiber ties are sign-aligned and averaged",
            periodic_image_policy:
                "all translated images whose expanded AABB intersects the fundamental cell",
            ambiguity_tolerance: config.ambiguity_tolerance,
            ambiguous_voxels,
            interface_threshold: config.include_interface.then_some(128),
        },
        source: ManifestSource {
            assembly_hash_fnv1a64: format!("{:016x}", fnv1a64(&assembly_bytes)),
            provenance_generator: assembly.provenance.source.clone(),
            provenance_seed: assembly.provenance.seed,
        },
        files,
    };
    let manifest_path = config.output_directory.join("manifest.json");
    let writer = BufWriter::new(File::create(&manifest_path)?);
    serde_json::to_writer_pretty(writer, &manifest)?;

    Ok(PumaExportReport {
        output_directory: config.output_directory.clone(),
        voxel_counts: counts,
        total_voxels,
        occupied_voxels,
        voxel_volume_fraction,
        ambiguous_voxels,
        binder_voxels,
        bonds: bridges.len(),
        domain_path,
        fiber_ids_path,
        interface_path,
        bond_ids_path,
        binder_interface_path,
        manifest_path,
        analysis_path,
    })
}

/// Validates the assembly and grid and builds the binder bridges: the
/// lengths, voxel counts, voxel total and bridges shared by
/// [`write_puma_bundle`] and [`voxelize_labels`].
fn prepare_grid(
    assembly: &FiberAssembly,
    config: &PumaVoxelExportConfig,
) -> Result<(Vec3, [usize; 3], usize, Vec<JunctionBridge>), PumaExportError> {
    assembly
        .validate()
        .map_err(|error| PumaExportError::InvalidAssembly(error.to_string()))?;
    let (lengths, counts) = validate_grid(assembly, config)?;
    let total_voxels = counts
        .iter()
        .try_fold(1usize, |product, count| product.checked_mul(*count))
        .ok_or(PumaExportError::GridTooLarge(counts))?;
    let bridges = match config.bond_radius_ratio {
        Some(ratio) => junction_bridges(assembly, ratio)
            .map_err(|error| PumaExportError::InvalidConfig(error.to_string()))?,
        None => Vec::new(),
    };
    Ok((lengths, counts, total_voxels, bridges))
}

fn validate_grid(
    assembly: &FiberAssembly,
    config: &PumaVoxelExportConfig,
) -> Result<(Vec3, [usize; 3]), PumaExportError> {
    if !config.voxel_size.is_finite() || config.voxel_size <= 0.0 {
        return Err(PumaExportError::InvalidConfig(
            "voxel_size must be finite and positive".into(),
        ));
    }
    if !config.ambiguity_tolerance.is_finite() || config.ambiguity_tolerance < 0.0 {
        return Err(PumaExportError::InvalidConfig(
            "ambiguity_tolerance must be finite and nonnegative".into(),
        ));
    }
    let phases = assembly.materials.entries.len() + usize::from(config.bond_radius_ratio.is_some());
    if phases > u16::MAX as usize {
        return Err(PumaExportError::TooManyMaterials(phases));
    }
    let basis = assembly.cell.basis;
    let scale = basis
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let diagonal_tolerance = scale.max(1.0) * 1.0e-12;
    for (row, vector) in basis.iter().enumerate() {
        for (column, value) in vector.iter().enumerate() {
            if row != column && value.abs() > diagonal_tolerance {
                return Err(PumaExportError::NonOrthorhombicCell);
            }
        }
        if vector[row] <= 0.0 || !vector[row].is_finite() {
            return Err(PumaExportError::NonOrthorhombicCell);
        }
    }
    let lengths = [basis[0][0], basis[1][1], basis[2][2]];
    let mut counts = [0usize; 3];
    let mut commensurate = true;
    for axis in 0..3 {
        let real_count = lengths[axis] / config.voxel_size;
        let rounded = real_count.round();
        counts[axis] = rounded.max(1.0) as usize;
        let relative_error = (real_count - rounded).abs() / real_count.max(1.0);
        commensurate &= relative_error <= 1.0e-9;
    }
    if !commensurate {
        return Err(PumaExportError::IncommensurateGrid {
            voxel_size: config.voxel_size,
            lengths,
            suggested_counts: counts,
        });
    }
    Ok((lengths, counts))
}

/// Binder phase ID: the one after every material's.
fn binder_phase_id(assembly: &FiberAssembly) -> u16 {
    (assembly.materials.entries.len() + 1).min(u16::MAX as usize) as u16
}

#[cfg(test)]
fn voxelize(
    assembly: &FiberAssembly,
    config: &PumaVoxelExportConfig,
    counts: [usize; 3],
    total_voxels: usize,
) -> Result<VoxelFields, PumaExportError> {
    voxelize_with_bonds(assembly, config, counts, total_voxels, &[])
}

fn voxelize_with_bonds(
    assembly: &FiberAssembly,
    config: &PumaVoxelExportConfig,
    counts: [usize; 3],
    total_voxels: usize,
    bridges: &[JunctionBridge],
) -> Result<VoxelFields, PumaExportError> {
    let mut phase = vec![0u16; total_voxels];
    let mut orientation = vec![[0.0f32; 3]; total_voxels];
    let mut owner_export_id = vec![0u32; total_voxels];
    let mut interface = vec![0u8; total_voxels];
    let mut ambiguous = vec![false; total_voxels];
    let mut best_signed_distance = vec![f64::INFINITY; total_voxels];
    let mut orientation_sum = vec![[0.0f64; 3]; total_voxels];
    let interface_half_width = config.voxel_size * 0.5;
    let origin = assembly.cell.origin;
    let lengths = [
        assembly.cell.basis[0][0],
        assembly.cell.basis[1][1],
        assembly.cell.basis[2][2],
    ];

    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let section_index = fiber.section.0 as usize;
        let (radius, oval) = match assembly.sections.entries.get(section_index) {
            Some(Section::Circular { radius }) => (*radius, None),
            Some(section @ Section::Elliptical { semi_axes }) if section.is_circular() => {
                (semi_axes[0], None)
            }
            Some(Section::Elliptical { semi_axes }) => (
                semi_axes[0].max(semi_axes[1]),
                Some((*semi_axes, assembly.fiber_directors(fiber_index))),
            ),
            None => continue,
        };
        let start = fiber.vertices.start as usize;
        let Some(end) = fiber.vertices.checked_end().map(|value| value as usize) else {
            continue;
        };
        let Some(points) = assembly.geometry.placed.positions.get(start..end) else {
            continue;
        };
        for (local_segment, segment) in points.windows(2).enumerate() {
            let base_a = segment[0];
            let base_b = segment[1];
            let delta = sub(base_b, base_a);
            let segment_length = norm(delta);
            if segment_length <= f64::EPSILON {
                continue;
            }
            let tangent = scale(delta, segment_length.recip());
            let image_ranges = periodic_image_ranges(
                base_a,
                base_b,
                radius + interface_half_width,
                origin,
                lengths,
                assembly.cell.periodic,
            );
            for image_x in image_ranges[0].0..=image_ranges[0].1 {
                for image_y in image_ranges[1].0..=image_ranges[1].1 {
                    for image_z in image_ranges[2].0..=image_ranges[2].1 {
                        let shift = [
                            image_x as f64 * lengths[0],
                            image_y as f64 * lengths[1],
                            image_z as f64 * lengths[2],
                        ];
                        let a = add(base_a, shift);
                        let b = add(base_b, shift);
                        let ranges = voxel_ranges(
                            a,
                            b,
                            radius + interface_half_width,
                            origin,
                            config.voxel_size,
                            counts,
                        );
                        for z in ranges[2].0..ranges[2].1 {
                            for y in ranges[1].0..ranges[1].1 {
                                for x in ranges[0].0..ranges[0].1 {
                                    let index = x + counts[0] * (y + counts[1] * z);
                                    let center = [
                                        origin[0] + (x as f64 + 0.5) * config.voxel_size,
                                        origin[1] + (y as f64 + 0.5) * config.voxel_size,
                                        origin[2] + (z as f64 + 0.5) * config.voxel_size,
                                    ];
                                    let signed_distance = match &oval {
                                        None => point_segment_distance(center, a, b) - radius,
                                        Some((semi_axes, directors)) => {
                                            oval_segment_signed_distance(
                                                center,
                                                a,
                                                b,
                                                directors[local_segment],
                                                directors[local_segment + 1],
                                                *semi_axes,
                                            )
                                        }
                                    };
                                    if signed_distance > interface_half_width {
                                        continue;
                                    }
                                    interface[index] = interface[index].max(interface_value(
                                        signed_distance,
                                        interface_half_width,
                                    ));
                                    if signed_distance > 0.0 {
                                        continue;
                                    }
                                    let candidate_export_id = fiber_index as u32 + 1;
                                    let candidate_phase = fiber.material.0 as u16 + 1;
                                    let best = best_signed_distance[index];
                                    if signed_distance < best - config.ambiguity_tolerance {
                                        best_signed_distance[index] = signed_distance;
                                        phase[index] = candidate_phase;
                                        owner_export_id[index] = candidate_export_id;
                                        orientation_sum[index] = scale(tangent, segment_length);
                                    } else if (signed_distance - best).abs()
                                        <= config.ambiguity_tolerance
                                    {
                                        let old_owner = owner_export_id[index];
                                        if old_owner == candidate_export_id {
                                            add_sign_aligned(
                                                &mut orientation_sum[index],
                                                scale(tangent, segment_length),
                                            );
                                        } else if old_owner != 0 {
                                            ambiguous[index] = true;
                                            let old_fiber =
                                                &assembly.topology.fibers[(old_owner - 1) as usize];
                                            if stable_fiber_key(fiber.id, candidate_export_id)
                                                < stable_fiber_key(old_fiber.id, old_owner)
                                            {
                                                phase[index] = candidate_phase;
                                                owner_export_id[index] = candidate_export_id;
                                                orientation_sum[index] =
                                                    scale(tangent, segment_length);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let (bond_id, binder_interface) = if config.bond_radius_ratio.is_some() {
        let mut bond_id = vec![0u32; total_voxels];
        let mut binder_interface = vec![0u8; total_voxels];
        let binder_phase = binder_phase_id(assembly);
        for bridge in bridges {
            let delta = sub(bridge.end, bridge.start);
            let length = norm(delta);
            let axis = if length > f64::EPSILON {
                scale(delta, length.recip())
            } else {
                [0.0; 3]
            };
            let export_id = bridge.junction_index as u32 + 1;
            let padding = bridge.radius + interface_half_width;
            let image_ranges = periodic_image_ranges(
                bridge.start,
                bridge.end,
                padding,
                origin,
                lengths,
                assembly.cell.periodic,
            );
            for image_x in image_ranges[0].0..=image_ranges[0].1 {
                for image_y in image_ranges[1].0..=image_ranges[1].1 {
                    for image_z in image_ranges[2].0..=image_ranges[2].1 {
                        let shift = [
                            image_x as f64 * lengths[0],
                            image_y as f64 * lengths[1],
                            image_z as f64 * lengths[2],
                        ];
                        let a = add(bridge.start, shift);
                        let b = add(bridge.end, shift);
                        let ranges = voxel_ranges(a, b, padding, origin, config.voxel_size, counts);
                        for z in ranges[2].0..ranges[2].1 {
                            for y in ranges[1].0..ranges[1].1 {
                                for x in ranges[0].0..ranges[0].1 {
                                    let index = x + counts[0] * (y + counts[1] * z);
                                    let center = [
                                        origin[0] + (x as f64 + 0.5) * config.voxel_size,
                                        origin[1] + (y as f64 + 0.5) * config.voxel_size,
                                        origin[2] + (z as f64 + 0.5) * config.voxel_size,
                                    ];
                                    let signed_distance =
                                        point_segment_distance(center, a, b) - bridge.radius;
                                    if signed_distance > interface_half_width {
                                        continue;
                                    }
                                    let outside_fibers = 255 - interface[index];
                                    binder_interface[index] = binder_interface[index].max(
                                        interface_value(signed_distance, interface_half_width)
                                            .min(outside_fibers),
                                    );
                                    let free = phase[index] == 0 || phase[index] == binder_phase;
                                    if signed_distance > 0.0
                                        || !free
                                        || signed_distance >= best_signed_distance[index]
                                    {
                                        continue;
                                    }
                                    best_signed_distance[index] = signed_distance;
                                    phase[index] = binder_phase;
                                    bond_id[index] = export_id;
                                    orientation_sum[index] = axis;
                                }
                            }
                        }
                    }
                }
            }
        }
        (bond_id, binder_interface)
    } else {
        (Vec::new(), Vec::new())
    };
    for (output, sum) in orientation.iter_mut().zip(orientation_sum) {
        let magnitude = norm(sum);
        if magnitude > f64::EPSILON {
            *output = [
                (sum[0] / magnitude) as f32,
                (sum[1] / magnitude) as f32,
                (sum[2] / magnitude) as f32,
            ];
        }
    }
    Ok(VoxelFields {
        phase,
        orientation,
        owner_export_id,
        interface,
        ambiguous,
        bond_id,
        binder_interface,
    })
}

fn periodic_image_ranges(
    a: Vec3,
    b: Vec3,
    padding: f64,
    origin: Vec3,
    lengths: Vec3,
    periodic: [bool; 3],
) -> [(i64, i64); 3] {
    let mut ranges = [(0, 0); 3];
    for axis in 0..3 {
        if periodic[axis] {
            let segment_min = a[axis].min(b[axis]) - padding;
            let segment_max = a[axis].max(b[axis]) + padding;
            ranges[axis].0 = ((origin[axis] - segment_max) / lengths[axis]).ceil() as i64;
            ranges[axis].1 =
                ((origin[axis] + lengths[axis] - segment_min) / lengths[axis]).floor() as i64;
        }
    }
    ranges
}

fn voxel_ranges(
    a: Vec3,
    b: Vec3,
    padding: f64,
    origin: Vec3,
    voxel_size: f64,
    counts: [usize; 3],
) -> [(usize, usize); 3] {
    let mut ranges = [(0, 0); 3];
    for axis in 0..3 {
        let lower = ((a[axis].min(b[axis]) - padding - origin[axis]) / voxel_size).floor();
        let upper = ((a[axis].max(b[axis]) + padding - origin[axis]) / voxel_size).ceil();
        ranges[axis].0 = lower.max(0.0).min(counts[axis] as f64) as usize;
        ranges[axis].1 = upper.max(0.0).min(counts[axis] as f64) as usize;
    }
    ranges
}

fn point_segment_distance(point: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = sub(b, a);
    let denominator = dot(ab, ab);
    let parameter = if denominator > f64::EPSILON {
        (dot(sub(point, a), ab) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    norm(sub(point, add(a, scale(ab, parameter))))
}

/// Approximate signed distance from a point to an elliptical tube around one
/// segment: the first semi-axis lies along the director (interpolated between
/// the segment's two vertices), the second across it. Past the segment ends
/// the tube closes with an ellipsoidal cap whose third semi-axis is the short
/// one, as round fibers close with hemispheres. The first-order estimate
/// f / |grad f| is exact in sign and accurate within the interface band;
/// deep inside it is limited to the short semi-axis.
fn oval_segment_signed_distance(
    point: Vec3,
    a: Vec3,
    b: Vec3,
    director_a: Vec3,
    director_b: Vec3,
    semi_axes: [f64; 2],
) -> f64 {
    let delta = sub(b, a);
    let length_squared = dot(delta, delta);
    let length = length_squared.sqrt();
    let raw = dot(sub(point, a), delta) / length_squared;
    let along = raw.clamp(0.0, 1.0);
    let tangent = scale(delta, length.recip());
    let offset = sub(point, add(a, scale(delta, along)));
    let overshoot = (raw - along) * length;
    let aligned_b = if dot(director_a, director_b) < 0.0 {
        scale(director_b, -1.0)
    } else {
        director_b
    };
    let blended = add(scale(director_a, 1.0 - along), scale(aligned_b, along));
    let director = sub(blended, scale(tangent, dot(blended, tangent)));
    let director = scale(director, norm(director).max(f64::MIN_POSITIVE).recip());
    let binormal = [
        tangent[1] * director[2] - tangent[2] * director[1],
        tangent[2] * director[0] - tangent[0] * director[2],
        tangent[0] * director[1] - tangent[1] * director[0],
    ];
    let [long, across] = semi_axes;
    let cap = long.min(across);
    let u = dot(offset, director) / long;
    let v = dot(offset, binormal) / across;
    let w = overshoot / cap;
    let level = u * u + v * v + w * w - 1.0;
    let gradient = 2.0 * ((u / long).powi(2) + (v / across).powi(2) + (w / cap).powi(2)).sqrt();
    if gradient <= f64::MIN_POSITIVE {
        return -cap;
    }
    (level / gradient).max(-cap)
}

fn interface_value(signed_distance: f64, half_width: f64) -> u8 {
    if signed_distance <= -half_width {
        255
    } else if signed_distance >= half_width {
        0
    } else {
        ((0.5 - 0.5 * signed_distance / half_width) * 255.0).round() as u8
    }
}

fn stable_fiber_key(id: FiberId, export_id: u32) -> (u32, u32) {
    (id.0, export_id)
}

fn add_sign_aligned(sum: &mut Vec3, tangent: Vec3) {
    let aligned = if dot(*sum, tangent) < 0.0 {
        scale(tangent, -1.0)
    } else {
        tangent
    };
    *sum = add(*sum, aligned);
}

fn write_domain_vti(
    path: &Path,
    origin: Vec3,
    spacing: f64,
    counts: [usize; 3],
    phase: &[u16],
    orientation: &[[f32; 3]],
) -> Result<(), std::io::Error> {
    let phase_bytes = u16_bytes(phase);
    let orientation_bytes = f32x3_bytes(orientation);
    let offset_orientation = 8 + phase_bytes.len();
    let arrays = format!(
        "        <DataArray type=\"UInt16\" Name=\"phase_id\" format=\"appended\" offset=\"0\"/>\n        <DataArray type=\"Float32\" Name=\"orientation\" NumberOfComponents=\"3\" format=\"appended\" offset=\"{offset_orientation}\"/>\n"
    );
    write_vti(
        path,
        origin,
        spacing,
        counts,
        &arrays,
        &[phase_bytes, orientation_bytes],
    )
}

fn write_scalar_vti_u32(
    path: &Path,
    origin: Vec3,
    spacing: f64,
    counts: [usize; 3],
    name: &str,
    values: &[u32],
) -> Result<(), std::io::Error> {
    let arrays = format!(
        "        <DataArray type=\"UInt32\" Name=\"{name}\" format=\"appended\" offset=\"0\"/>\n"
    );
    write_vti(path, origin, spacing, counts, &arrays, &[u32_bytes(values)])
}

fn write_scalar_vti_u8(
    path: &Path,
    origin: Vec3,
    spacing: f64,
    counts: [usize; 3],
    name: &str,
    values: &[u8],
) -> Result<(), std::io::Error> {
    let arrays = format!(
        "        <DataArray type=\"UInt8\" Name=\"{name}\" format=\"appended\" offset=\"0\"/>\n"
    );
    write_vti(path, origin, spacing, counts, &arrays, &[values.to_vec()])
}

fn write_vti(
    path: &Path,
    origin: Vec3,
    spacing: f64,
    counts: [usize; 3],
    arrays: &str,
    blocks: &[Vec<u8>],
) -> Result<(), std::io::Error> {
    let mut writer = BufWriter::new(File::create(path)?);
    write!(
        writer,
        "<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n  <ImageData WholeExtent=\"0 {} 0 {} 0 {}\" Origin=\"{:.17e} {:.17e} {:.17e}\" Spacing=\"{:.17e} {:.17e} {:.17e}\">\n    <Piece Extent=\"0 {} 0 {} 0 {}\">\n      <PointData/>\n      <CellData>\n{}      </CellData>\n    </Piece>\n  </ImageData>\n  <AppendedData encoding=\"raw\">\n_",
        counts[0],
        counts[1],
        counts[2],
        origin[0],
        origin[1],
        origin[2],
        spacing,
        spacing,
        spacing,
        counts[0],
        counts[1],
        counts[2],
        arrays
    )?;
    for block in blocks {
        writer.write_all(&(block.len() as u64).to_le_bytes())?;
        writer.write_all(block)?;
    }
    writer.write_all(b"\n  </AppendedData>\n</VTKFile>\n")?;
    Ok(())
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32x3_bytes(values: &[[f32; 3]]) -> Vec<u8> {
    values
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn manifest_file(path: &Path, root: &Path) -> Result<ManifestFile, std::io::Error> {
    let bytes = fs::read(path)?;
    Ok(ManifestFile {
        path: path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned(),
        bytes: bytes.len() as u64,
        hash_fnv1a64: format!("{:016x}", fnv1a64(&bytes)),
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(value: Vec3) -> f64 {
    dot(value, value).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell};

    fn temporary_bundle(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tangle-{name}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    #[test]
    fn exports_multimaterial_domain_and_manifest() {
        let directory = temporary_bundle("puma-multimaterial");
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let first = assembly.materials.add("first");
        let second = assembly.materials.add("second");
        let section = assembly.sections.add(Section::Circular { radius: 0.15 });
        assembly
            .add_fiber(
                FiberId(11),
                first,
                section,
                &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                &[[0.25, 0.25, 0.5], [0.75, 0.25, 0.5]],
            )
            .unwrap();
        assembly
            .add_fiber(
                FiberId(22),
                second,
                section,
                &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                &[[0.25, 0.75, 0.5], [0.75, 0.75, 0.5]],
            )
            .unwrap();

        let report =
            write_puma_bundle(&assembly, &PumaVoxelExportConfig::new(&directory, 0.1)).unwrap();
        assert_eq!(report.voxel_counts, [10, 10, 10]);
        assert!(report.occupied_voxels > 0);
        assert!(report.domain_path.is_file());
        assert!(report.fiber_ids_path.as_ref().unwrap().is_file());
        assert!(report.interface_path.as_ref().unwrap().is_file());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(report.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["schema_version"], PUMA_BUNDLE_SCHEMA_VERSION);
        assert_eq!(manifest["materials"][0]["phase_id"], 1);
        assert_eq!(manifest["materials"][1]["phase_id"], 2);
        assert_eq!(manifest["fibers"][0]["fiber_id"], 11);
        let domain = fs::read(&report.domain_path).unwrap();
        let marker = b"<AppendedData encoding=\"raw\">\n_";
        let appended = domain
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap()
            + marker.len();
        let phase_bytes = u64::from_le_bytes(domain[appended..appended + 8].try_into().unwrap());
        assert_eq!(phase_bytes, 2_000);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_voxel_size_that_does_not_tile_suggests_one_that_does() {
        let message = PumaExportError::IncommensurateGrid {
            voxel_size: 1.3e-6,
            lengths: [150.0e-6; 3],
            suggested_counts: [115; 3],
        }
        .to_string();
        assert!(message.contains("try voxel_size=1.304348e-6 (115 x 115 x 115 voxels)"), "{message}");
    }

    #[test]
    fn periodic_image_occupies_both_cell_faces() {
        let directory = temporary_bundle("puma-periodic");
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [true, false, false]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.12 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                &[[0.9, 0.5, 0.5], [1.1, 0.5, 0.5]],
            )
            .unwrap();
        let (_, counts) =
            validate_grid(&assembly, &PumaVoxelExportConfig::new(&directory, 0.1)).unwrap();
        let fields = voxelize(
            &assembly,
            &PumaVoxelExportConfig::new(&directory, 0.1),
            counts,
            1000,
        )
        .unwrap();
        let occupied_at_x =
            |x: usize| (0..10).any(|z| (0..10).any(|y| fields.phase[x + 10 * (y + 10 * z)] != 0));
        assert!(occupied_at_x(0));
        assert!(occupied_at_x(9));
    }

    #[test]
    fn rejects_a_non_tiling_voxel_size() {
        let assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let error =
            validate_grid(&assembly, &PumaVoxelExportConfig::new("unused", 0.3)).unwrap_err();
        assert!(matches!(error, PumaExportError::IncommensurateGrid { .. }));
    }

    #[test]
    fn equal_distance_ownership_uses_stable_fiber_id() {
        let directory = temporary_bundle("puma-tie");
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let first_material = assembly.materials.add("first");
        let second_material = assembly.materials.add("second");
        let section = assembly.sections.add(Section::Circular { radius: 0.15 });
        let placed = [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]];
        assembly
            .add_fiber(FiberId(10), first_material, section, &placed, &placed)
            .unwrap();
        assembly
            .add_fiber(FiberId(5), second_material, section, &placed, &placed)
            .unwrap();

        let config = PumaVoxelExportConfig::new(&directory, 0.1);
        let fields = voxelize(&assembly, &config, [10; 3], 1_000).unwrap();
        let occupied = fields
            .phase
            .iter()
            .zip(&fields.owner_export_id)
            .filter(|(phase, _)| **phase != 0)
            .collect::<Vec<_>>();
        assert!(!occupied.is_empty());
        assert!(occupied
            .iter()
            .all(|(phase, owner)| **phase == 2 && **owner == 2));
        assert!(fields.ambiguous.iter().any(|value| *value));
    }

    #[test]
    fn oval_distance_is_zero_on_the_ellipse() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let director = [0.0, 1.0, 0.0];
        let flipped = [0.0, -1.0, 0.0];
        let distance =
            |point: Vec3| oval_segment_signed_distance(point, a, b, director, flipped, [0.3, 0.15]);
        assert!(distance([0.5, 0.3, 0.0]).abs() < 1.0e-12);
        assert!(distance([0.5, 0.0, 0.15]).abs() < 1.0e-12);
        assert!(distance([0.5, 0.35, 0.0]) > 0.0);
        assert!(distance([0.5, 0.0, 0.2]) > 0.0);
        assert!(distance([0.5, 0.2, 0.1]) < 0.0);
        assert!((distance([0.5, 0.0, 0.0]) + 0.15).abs() < 1.0e-12);
        // Ellipsoidal caps past the ends.
        assert!(distance([1.1, 0.0, 0.0]) < 0.0);
        assert!(distance([1.2, 0.0, 0.0]) > 0.0);
    }

    #[test]
    fn oval_fiber_voxelizes_to_its_volume() {
        let directory = temporary_bundle("puma-oval");
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("oval");
        let section = assembly.sections.add(Section::Elliptical {
            semi_axes: [0.3, 0.15],
        });
        let points = [[0.2, 0.5, 0.5], [0.5, 0.5, 0.5], [0.8, 0.5, 0.5]];
        assembly
            .add_fiber(FiberId(1), material, section, &points, &points)
            .unwrap();
        let voxel = 0.02;
        let report =
            write_puma_bundle(&assembly, &PumaVoxelExportConfig::new(&directory, voxel)).unwrap();
        // Elliptical tube plus two half-ellipsoid caps (third semi-axis 0.15).
        let volume = std::f64::consts::PI * 0.3 * 0.15 * 0.6
            + 4.0 / 3.0 * std::f64::consts::PI * 0.3 * 0.15 * 0.15;
        let expected = volume / voxel.powi(3);
        let occupied = report.occupied_voxels as f64;
        assert!(
            (occupied / expected - 1.0).abs() < 0.03,
            "{occupied} voxels, expected {expected}"
        );
        fs::remove_dir_all(directory).unwrap();
    }

    fn read_file(path: &Option<PathBuf>) -> Vec<u8> {
        fs::read(path.as_ref().unwrap()).unwrap()
    }

    #[test]
    fn bonds_off_leave_every_image_as_without_junctions() {
        use crate::bridges::tests::touching_cross;
        let plain_directory = temporary_bundle("puma-bonds-plain");
        let joined_directory = temporary_bundle("puma-bonds-joined");
        let plain = write_puma_bundle(
            &touching_cross(false, false),
            &PumaVoxelExportConfig::new(&plain_directory, 0.05),
        )
        .unwrap();
        let joined = write_puma_bundle(
            &touching_cross(false, true),
            &PumaVoxelExportConfig::new(&joined_directory, 0.05),
        )
        .unwrap();
        assert_eq!(
            fs::read(&plain.domain_path).unwrap(),
            fs::read(&joined.domain_path).unwrap()
        );
        assert_eq!(
            read_file(&plain.fiber_ids_path),
            read_file(&joined.fiber_ids_path)
        );
        assert_eq!(
            read_file(&plain.interface_path),
            read_file(&joined.interface_path)
        );
        assert!(joined.bond_ids_path.is_none());
        assert_eq!(joined.binder_voxels, 0);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&joined.manifest_path).unwrap()).unwrap();
        assert!(manifest.get("binder").is_none());
        fs::remove_dir_all(plain_directory).unwrap();
        fs::remove_dir_all(joined_directory).unwrap();
    }

    #[test]
    fn bonds_on_add_a_binder_phase_outside_the_fibers() {
        use crate::bridges::tests::touching_cross;
        let assembly = touching_cross(false, true);
        let plain_directory = temporary_bundle("puma-binder-plain");
        let directory = temporary_bundle("puma-binder");
        let plain = write_puma_bundle(
            &assembly,
            &PumaVoxelExportConfig::new(&plain_directory, 0.05),
        )
        .unwrap();
        let config = PumaVoxelExportConfig::new(&directory, 0.05).with_bonds(Some(0.8));
        let report = write_puma_bundle(&assembly, &config).unwrap();
        assert_eq!(report.bonds, 1);
        assert!(report.binder_voxels > 0);
        assert!(report.bond_ids_path.as_ref().unwrap().is_file());
        assert!(report.binder_interface_path.as_ref().unwrap().is_file());
        assert_eq!(
            report.occupied_voxels,
            plain.occupied_voxels + report.binder_voxels
        );
        // The fiber images do not change when binder is added.
        assert_eq!(
            read_file(&plain.fiber_ids_path),
            read_file(&report.fiber_ids_path)
        );
        assert_eq!(
            read_file(&plain.interface_path),
            read_file(&report.interface_path)
        );

        let bridges = junction_bridges(&assembly, 0.8).unwrap();
        let fields = voxelize_with_bonds(&assembly, &config, [20; 3], 8_000, &bridges).unwrap();
        let binder_phase = binder_phase_id(&assembly);
        assert_eq!(binder_phase, 2);
        for index in 0..8_000 {
            if fields.phase[index] == binder_phase {
                assert_eq!(fields.owner_export_id[index], 0);
                assert_eq!(fields.bond_id[index], 1);
            } else {
                assert_eq!(fields.bond_id[index], 0);
            }
            assert!(
                u16::from(fields.binder_interface[index]) + u16::from(fields.interface[index])
                    <= 255
            );
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&report.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["binder"]["phase_id"], 2);
        assert_eq!(manifest["binder"]["bonds"][0]["junction_id"], 9);
        assert_eq!(manifest["binder"]["bonds"][0]["law"], "bond");
        fs::remove_dir_all(plain_directory).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn binder_across_a_periodic_wall_stays_near_the_wall() {
        use crate::bridges::tests::touching_cross;
        let assembly = touching_cross(true, true);
        let config = PumaVoxelExportConfig::new("unused", 0.05).with_bonds(Some(0.8));
        let bridges = junction_bridges(&assembly, 0.8).unwrap();
        let fields = voxelize_with_bonds(&assembly, &config, [20; 3], 8_000, &bridges).unwrap();
        let binder_phase = binder_phase_id(&assembly);
        let mut binder_columns = Vec::new();
        for (index, phase) in fields.phase.iter().enumerate() {
            if *phase == binder_phase {
                binder_columns.push(index % 20);
            }
        }
        assert!(!binder_columns.is_empty());
        // The gap between the fibers through the wall is x in [0.95, 1) and
        // [0, 0.05); the binder never reaches the middle of the cell.
        assert!(binder_columns.iter().all(|x| *x <= 2 || *x >= 17));
    }

    #[test]
    fn in_memory_labels_match_the_bundle_rasterization() {
        use crate::bridges::tests::touching_cross;
        let assembly = touching_cross(true, true);
        let labels = voxelize_labels(&assembly, 0.05, Some(0.8)).unwrap();
        assert_eq!(labels.voxel_counts, [20; 3]);
        assert_eq!(labels.voxel_size, 0.05);
        assert_eq!(labels.origin, assembly.cell.origin);

        let config = PumaVoxelExportConfig::new("unused", 0.05).with_bonds(Some(0.8));
        let bridges = junction_bridges(&assembly, 0.8).unwrap();
        let fields = voxelize_with_bonds(&assembly, &config, [20; 3], 8_000, &bridges).unwrap();
        assert_eq!(labels.phase, fields.phase);
        assert_eq!(labels.fiber, fields.owner_export_id);
        assert_eq!(labels.bond, fields.bond_id);
        assert!(labels.phase.contains(&binder_phase_id(&assembly)));
        assert!(labels.fiber.contains(&1) && labels.fiber.contains(&2));
    }

    #[test]
    fn in_memory_labels_without_bonds_have_no_binder() {
        use crate::bridges::tests::touching_cross;
        let assembly = touching_cross(false, true);
        let labels = voxelize_labels(&assembly, 0.05, None).unwrap();
        assert!(labels.bond.is_empty());
        assert!(!labels.phase.contains(&binder_phase_id(&assembly)));
        for (phase, fiber) in labels.phase.iter().zip(&labels.fiber) {
            assert_eq!(*phase == 0, *fiber == 0);
        }
    }

    #[test]
    fn in_memory_labels_reject_a_non_tiling_voxel_size() {
        let assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let error = voxelize_labels(&assembly, 0.3, None).unwrap_err();
        assert!(matches!(error, PumaExportError::IncommensurateGrid { .. }));
    }
}
