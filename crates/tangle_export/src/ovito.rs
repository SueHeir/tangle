use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use tangle_core::{measure_vertex_curvature, FiberAssembly, Section, Vec3};

use crate::{build_dem_bpm_model, DemBpmExportConfig, DemBpmModel, ExportError};

/// Geometry written to an OVITO relaxation trajectory.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum OvitoRepresentation {
    /// One spherocylinder for every piecewise-linear centerline segment.
    ///
    /// This is the compact representation of TANGLE's geometric contact
    /// model: a straight two-point fiber becomes one capsule, while a curved
    /// fiber becomes one capsule per centerline segment.
    #[default]
    FiberSegments,
    /// An explicit connected capsule view: one cylindrical body per segment
    /// plus one spherical joint at every centerline vertex.
    ///
    /// This uses per-type OVITO shapes instead of relying on OVITO's compound
    /// spherocylinder glyph, making shared fiber endpoints visually exact.
    ConnectedFiberSegments,
    /// The overlapping-sphere discretization used by the DEM-BPM exporter.
    DemParticles {
        /// Maximum center spacing as a fraction of sphere diameter.
        spacing_ratio: f64,
    },
}

/// Property used to color objects in the generated OVITO viewing recipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OvitoColoring {
    /// Assign a distinct color from each fiber's molecule identifier.
    #[default]
    Fiber,
    /// Color by current curvature divided by maximum admissible curvature.
    ///
    /// Zero is straight and one is the bend limit. Values above one are
    /// violations and saturate at the high end of the color map.
    CurvatureRatio,
    /// Color by dyadic centerline subdivision level.
    RefinementLevel,
}

/// Configuration for an OVITO-readable relaxation trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct OvitoTrajectoryConfig {
    /// Multi-frame LAMMPS dump output path.
    pub dump_path: PathBuf,
    /// Optional generated OVITO Python viewing recipe.
    pub view_script_path: Option<PathBuf>,
    /// Session path written when the viewing recipe is run with `ovitos` or
    /// the OVITO Python package.
    pub session_path: Option<PathBuf>,
    /// Write every Nth relaxation step. The final state is always written.
    pub frame_interval: usize,
    /// Whether to write the host-side generated assembly before GPU formation.
    /// Disable this for staged insertion recipes whose later fiber groups are
    /// prepacked but not yet physically present.
    pub write_initial_frame: bool,
    /// Geometry used for each trajectory frame.
    pub representation: OvitoRepresentation,
    /// Property selected by the generated viewing recipe.
    pub coloring: OvitoColoring,
    /// Particle type stored in the dump.
    pub atom_type: u32,
}

impl OvitoTrajectoryConfig {
    /// Creates a trajectory that renders one spherocylinder per fiber segment.
    pub fn fiber_segments(dump_path: impl Into<PathBuf>, frame_interval: usize) -> Self {
        Self {
            dump_path: dump_path.into(),
            view_script_path: None,
            session_path: None,
            frame_interval,
            write_initial_frame: true,
            representation: OvitoRepresentation::FiberSegments,
            coloring: OvitoColoring::Fiber,
            atom_type: 1,
        }
    }

    /// Includes or omits the pre-relaxation host assembly frame.
    pub fn with_initial_frame(mut self, write: bool) -> Self {
        self.write_initial_frame = write;
        self
    }

    /// Selects the scalar or identifier used by the generated viewing recipe.
    pub fn with_coloring(mut self, coloring: OvitoColoring) -> Self {
        self.coloring = coloring;
        self
    }

    /// Requests a portable viewing recipe and its target OVITO session path.
    pub fn with_viewing_files(
        mut self,
        view_script_path: impl Into<PathBuf>,
        session_path: impl Into<PathBuf>,
    ) -> Self {
        self.view_script_path = Some(view_script_path.into());
        self.session_path = Some(session_path.into());
        self
    }
}

/// Summary of an active or completed OVITO trajectory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OvitoTrajectoryReport {
    /// Number of frames written so far.
    pub frames: usize,
    /// Most recent relaxation timestep written.
    pub last_timestep: Option<usize>,
    /// Trajectory output path.
    pub dump_path: PathBuf,
    /// Generated viewing-recipe path, when requested.
    pub view_script_path: Option<PathBuf>,
    /// Session path targeted by the generated viewing recipe.
    pub session_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FiberCapsule {
    id: u32,
    fiber_id: u32,
    atom_type: u32,
    radius: f64,
    length: f64,
    position: Vec3,
    orientation: [f64; 4],
    local_segment: u32,
    fiber_points: u32,
    natural_curvature: f64,
    current_curvature: f64,
    curvature_ratio: f64,
    curvature_excess: f64,
    refinement_level: f64,
}

/// Writes or appends one fiber-assembly frame using the configured OVITO
/// representation.
///
/// Device solvers use this entry point after explicitly downloading sparse
/// debug snapshots.
pub fn write_ovito_assembly_frame(
    assembly: &FiberAssembly,
    config: &OvitoTrajectoryConfig,
    timestep: usize,
    append: bool,
) -> Result<(), ExportError> {
    match config.representation {
        OvitoRepresentation::FiberSegments => write_fiber_segment_frame(
            assembly,
            config.atom_type,
            &config.dump_path,
            timestep,
            append,
        ),
        OvitoRepresentation::ConnectedFiberSegments => {
            write_connected_fiber_frame(assembly, config, timestep, append)
        }
        OvitoRepresentation::DemParticles { spacing_ratio } => {
            let model = build_dem_bpm_model(
                assembly,
                &DemBpmExportConfig {
                    data_path: PathBuf::new(),
                    density: 1.0,
                    spacing_ratio,
                    atom_type: config.atom_type,
                    bond_type: 1,
                },
            )?;
            write_ovito_dump_frame(
                &model,
                assembly.cell.periodic,
                &config.dump_path,
                timestep,
                append,
            )
        }
    }
}

fn write_connected_fiber_frame(
    assembly: &FiberAssembly,
    config: &OvitoTrajectoryConfig,
    timestep: usize,
    append: bool,
) -> Result<(), ExportError> {
    let (box_low, box_high) = crate::orthorhombic_bounds(assembly)?;
    let capsules = build_fiber_capsules(assembly, config.atom_type)?;
    let mut vertex_count = 0;
    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        vertex_count +=
            fiber.vertices.len as usize * SectionLanes::of_fiber(assembly, fiber_index)?.count();
    }
    let mut writer = open_frame_writer(&config.dump_path, append)?;
    write_frame_header(
        &mut writer,
        timestep,
        capsules.len() + vertex_count,
        box_low,
        box_high,
        assembly.cell.periodic,
    )?;
    writeln!(
        writer,
        "ITEM: ATOMS id mol type AsphericalShape.X AsphericalShape.Y AsphericalShape.Z quati quatj quatk quatw x y z segment fiber_points natural_curvature current_curvature curvature_ratio curvature_excess refinement_level"
    )?;

    for capsule in &capsules {
        write_fiber_glyph(
            &mut writer,
            capsule.id,
            capsule.fiber_id,
            config.atom_type,
            [capsule.radius, capsule.radius, capsule.length],
            capsule.orientation,
            capsule.position,
            capsule.local_segment,
            capsule.fiber_points,
            capsule.natural_curvature,
            capsule.current_curvature,
            capsule.curvature_ratio,
            capsule.curvature_excess,
            capsule.refinement_level,
        )?;
    }

    let sphere_type = config
        .atom_type
        .checked_add(1)
        .ok_or(ExportError::IndexOverflow)?;
    let mut next_id = u32::try_from(capsules.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let lanes = SectionLanes::of_fiber(assembly, fiber_index)?;
        let radius = lanes.radius;
        let start = fiber.vertices.start as usize;
        let end = fiber
            .vertices
            .checked_end()
            .ok_or(ExportError::InvalidFiberSpan(fiber.id.0))? as usize;
        let placed = assembly
            .geometry
            .placed
            .positions
            .get(start..end)
            .ok_or(ExportError::InvalidFiberSpan(fiber.id.0))?;
        let intrinsic = assembly
            .geometry
            .intrinsic
            .positions
            .get(start..end)
            .ok_or(ExportError::InvalidFiberSpan(fiber.id.0))?;
        let natural_curvatures = vertex_curvatures(intrinsic);
        let current_curvatures = vertex_curvatures(placed);
        let total_rest_length = intrinsic
            .windows(2)
            .map(|pair| norm(sub(pair[1], pair[0])))
            .sum::<f64>();
        let maximum_curvature = assembly
            .admissibility
            .bend_limits
            .get(fiber_index)
            .copied()
            .flatten()
            .map(|limit| limit.maximum_curvature());
        let fiber_points = u32::try_from(placed.len()).map_err(|_| ExportError::IndexOverflow)?;
        for (local_vertex, position) in placed.iter().copied().enumerate() {
            let natural_curvature = natural_curvatures[local_vertex];
            let current_curvature = current_curvatures[local_vertex];
            let curvature_ratio = maximum_curvature
                .map(|maximum| current_curvature / maximum)
                .unwrap_or(0.0);
            let curvature_excess = maximum_curvature
                .map(|maximum| (current_curvature - maximum).max(0.0))
                .unwrap_or(0.0);
            let previous_length = local_vertex
                .checked_sub(1)
                .map(|previous| norm(sub(intrinsic[local_vertex], intrinsic[previous])));
            let next_length = intrinsic
                .get(local_vertex + 1)
                .map(|next| norm(sub(*next, intrinsic[local_vertex])));
            let local_rest_length = match (previous_length, next_length) {
                (Some(previous), Some(next)) => previous.min(next),
                (Some(length), None) | (None, Some(length)) => length,
                (None, None) => total_rest_length,
            };
            let refinement_level = if local_rest_length > 0.0 {
                (total_rest_length / local_rest_length).log2().max(0.0)
            } else {
                0.0
            };
            for lane in 0..lanes.count() {
                write_fiber_glyph(
                    &mut writer,
                    next_id,
                    fiber.id.0,
                    sphere_type,
                    [radius, radius, radius],
                    [0.0, 0.0, 0.0, 1.0],
                    lanes.point(lane, local_vertex, position),
                    u32::try_from(local_vertex).map_err(|_| ExportError::IndexOverflow)?,
                    fiber_points,
                    natural_curvature,
                    current_curvature,
                    curvature_ratio,
                    curvature_excess,
                    refinement_level,
                )?;
                next_id = next_id.checked_add(1).ok_or(ExportError::IndexOverflow)?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_fiber_glyph(
    writer: &mut impl Write,
    id: u32,
    fiber_id: u32,
    atom_type: u32,
    shape: [f64; 3],
    orientation: [f64; 4],
    position: Vec3,
    local_index: u32,
    fiber_points: u32,
    natural_curvature: f64,
    current_curvature: f64,
    curvature_ratio: f64,
    curvature_excess: f64,
    refinement_level: f64,
) -> Result<(), std::io::Error> {
    writeln!(
        writer,
        "{id} {fiber_id} {atom_type} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {local_index} {fiber_points} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e}",
        shape[0],
        shape[1],
        shape[2],
        orientation[0],
        orientation[1],
        orientation[2],
        orientation[3],
        position[0],
        position[1],
        position[2],
        natural_curvature,
        current_curvature,
        curvature_ratio,
        curvature_excess,
        refinement_level,
    )
}

fn write_fiber_segment_frame(
    assembly: &FiberAssembly,
    atom_type: u32,
    path: &Path,
    timestep: usize,
    append: bool,
) -> Result<(), ExportError> {
    let (box_low, box_high) = crate::orthorhombic_bounds(assembly)?;
    let capsules = build_fiber_capsules(assembly, atom_type)?;
    let mut writer = open_frame_writer(path, append)?;
    write_frame_header(
        &mut writer,
        timestep,
        capsules.len(),
        box_low,
        box_high,
        assembly.cell.periodic,
    )?;
    writeln!(
        writer,
        "ITEM: ATOMS id mol type AsphericalShape.X AsphericalShape.Y AsphericalShape.Z quati quatj quatk quatw x y z segment fiber_points natural_curvature current_curvature curvature_ratio curvature_excess refinement_level"
    )?;
    for capsule in capsules {
        writeln!(
            writer,
            "{} {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
            capsule.id,
            capsule.fiber_id,
            capsule.atom_type,
            capsule.radius,
            capsule.radius,
            capsule.length,
            capsule.orientation[0],
            capsule.orientation[1],
            capsule.orientation[2],
            capsule.orientation[3],
            capsule.position[0],
            capsule.position[1],
            capsule.position[2],
            capsule.local_segment,
            capsule.fiber_points,
            capsule.natural_curvature,
            capsule.current_curvature,
            capsule.curvature_ratio,
            capsule.curvature_excess,
            capsule.refinement_level
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn build_fiber_capsules(
    assembly: &FiberAssembly,
    atom_type: u32,
) -> Result<Vec<FiberCapsule>, ExportError> {
    let mut capsules = Vec::new();
    for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
        let lanes = SectionLanes::of_fiber(assembly, fiber_index)?;
        let radius = lanes.radius;
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
        let intrinsic = assembly
            .geometry
            .intrinsic
            .positions
            .get(start..end)
            .ok_or(ExportError::InvalidFiberSpan(fiber.id.0))?;
        let natural_curvatures = vertex_curvatures(intrinsic);
        let current_curvatures = vertex_curvatures(points);
        let total_rest_length = intrinsic
            .windows(2)
            .map(|pair| norm(sub(pair[1], pair[0])))
            .sum::<f64>();
        let maximum_curvature = assembly
            .admissibility
            .bend_limits
            .get(fiber_index)
            .copied()
            .flatten()
            .map(|limit| limit.maximum_curvature());
        let fiber_points = u32::try_from(points.len()).map_err(|_| ExportError::IndexOverflow)?;
        for (local_segment, points) in points.windows(2).enumerate() {
            let natural_curvature =
                natural_curvatures[local_segment].max(natural_curvatures[local_segment + 1]);
            let current_curvature =
                current_curvatures[local_segment].max(current_curvatures[local_segment + 1]);
            let curvature_ratio = maximum_curvature
                .map(|maximum| current_curvature / maximum)
                .unwrap_or(0.0);
            let curvature_excess = maximum_curvature
                .map(|maximum| (current_curvature - maximum).max(0.0))
                .unwrap_or(0.0);
            let segment_rest_length =
                norm(sub(intrinsic[local_segment + 1], intrinsic[local_segment]));
            let refinement_level = if segment_rest_length > 0.0 {
                (total_rest_length / segment_rest_length).log2().max(0.0)
            } else {
                0.0
            };
            for lane in 0..lanes.count() {
                let first = lanes.point(lane, local_segment, points[0]);
                let second = lanes.point(lane, local_segment + 1, points[1]);
                let lane_delta = sub(second, first);
                let lane_length = norm(lane_delta);
                let id =
                    u32::try_from(capsules.len() + 1).map_err(|_| ExportError::IndexOverflow)?;
                capsules.push(FiberCapsule {
                    id,
                    fiber_id: fiber.id.0,
                    atom_type,
                    radius,
                    length: lane_length,
                    position: scale(add(first, second), 0.5),
                    orientation: z_axis_orientation(lane_delta, lane_length),
                    local_segment: u32::try_from(local_segment)
                        .map_err(|_| ExportError::IndexOverflow)?,
                    fiber_points,
                    natural_curvature,
                    current_curvature,
                    curvature_ratio,
                    curvature_excess,
                    refinement_level,
                });
            }
        }
    }
    Ok(capsules)
}

/// The round lanes that draw one fiber: a single lane on the centerline for
/// round fibers, and for ovals the same row of lanes along the long-axis
/// directors that the relaxation solver uses for contact.
struct SectionLanes {
    /// Radius of every lane.
    radius: f64,
    /// Signed lane offsets along the director; empty for a round fiber.
    offsets: Vec<f64>,
    /// Long-axis director per fiber vertex; empty for a round fiber.
    directors: Vec<Vec3>,
}

impl SectionLanes {
    fn of_fiber(assembly: &FiberAssembly, fiber_index: usize) -> Result<Self, ExportError> {
        let fiber = &assembly.topology.fibers[fiber_index];
        let section = assembly
            .sections
            .entries
            .get(fiber.section.0 as usize)
            .ok_or(ExportError::MissingSection(fiber.id.0))?;
        let (radius, offsets) = oval_lanes(section);
        if offsets.is_empty() {
            return Ok(Self {
                radius,
                offsets,
                directors: Vec::new(),
            });
        }
        let mut directors = assembly.fiber_directors(fiber_index);
        if let Section::Elliptical { semi_axes } = section {
            if semi_axes[1] > semi_axes[0] {
                // The long axis is the second one: turn a quarter turn.
                let start = fiber.vertices.start as usize;
                let placed = &assembly.geometry.placed.positions
                    [start..start + fiber.vertices.len as usize];
                for (director, tangent) in directors
                    .iter_mut()
                    .zip(tangle_core::polyline_tangents(placed))
                {
                    *director = [
                        tangent[1] * director[2] - tangent[2] * director[1],
                        tangent[2] * director[0] - tangent[0] * director[2],
                        tangent[0] * director[1] - tangent[1] * director[0],
                    ];
                }
            }
        }
        Ok(Self {
            radius,
            offsets,
            directors,
        })
    }

    fn count(&self) -> usize {
        self.offsets.len().max(1)
    }

    fn point(&self, lane: usize, local_vertex: usize, position: Vec3) -> Vec3 {
        match self.offsets.get(lane) {
            Some(offset) => add(position, scale(self.directors[local_vertex], *offset)),
            None => position,
        }
    }
}

/// Lane radius and signed lane offsets of a section, matching the solver's
/// lane model; no offsets for a round section.
pub(crate) fn oval_lanes(section: &Section) -> (f64, Vec<f64>) {
    match *section {
        Section::Circular { radius } => (radius, Vec::new()),
        Section::Elliptical { semi_axes } => {
            let long = semi_axes[0].max(semi_axes[1]);
            let short = semi_axes[0].min(semi_axes[1]);
            let offset = long - short;
            if offset <= 0.0 {
                return (long, Vec::new());
            }
            let lanes = tangle_relax::oval_lane_count(offset as f32, short as f32) as usize;
            let offsets = (0..lanes)
                .map(|lane| offset * (2.0 * lane as f64 / (lanes - 1) as f64 - 1.0))
                .collect();
            (short, offsets)
        }
    }
}

fn vertex_curvatures(points: &[Vec3]) -> Vec<f64> {
    let mut curvatures = vec![0.0; points.len()];
    for (offset, triple) in points.windows(3).enumerate() {
        if let Some(measurement) = measure_vertex_curvature(triple[0], triple[1], triple[2]) {
            curvatures[offset + 1] = measurement.curvature;
        }
    }
    curvatures
}

fn z_axis_orientation(direction: Vec3, length: f64) -> [f64; 4] {
    if length <= 1.0e-15 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let unit = scale(direction, length.recip());
    if unit[2] < -1.0 + 1.0e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let mut quaternion = [-unit[1], unit[0], 0.0, 1.0 + unit[2]];
    let magnitude = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    for value in &mut quaternion {
        *value /= magnitude;
    }
    quaternion
}

/// Writes or appends one DEM-particle frame in the LAMMPS custom dump format.
///
/// The `diameter` and `mol` columns are recognized by OVITO as particle radius
/// and molecule identifier, respectively. Set `append` to `false` for the
/// first frame to replace any trajectory left by an earlier run.
pub fn write_ovito_dump_frame(
    model: &DemBpmModel,
    periodic: [bool; 3],
    path: &Path,
    timestep: usize,
    append: bool,
) -> Result<(), ExportError> {
    let mut writer = open_frame_writer(path, append)?;
    write_frame_header(
        &mut writer,
        timestep,
        model.particles.len(),
        model.box_low,
        model.box_high,
        periodic,
    )?;
    writeln!(writer, "ITEM: ATOMS id mol type diameter x y z")?;
    for particle in &model.particles {
        writeln!(
            writer,
            "{} {} {} {:.17e} {:.17e} {:.17e} {:.17e}",
            particle.id,
            particle.molecule,
            particle.atom_type,
            particle.diameter,
            particle.position[0],
            particle.position[1],
            particle.position[2]
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn open_frame_writer(path: &Path, append: bool) -> Result<BufWriter<File>, std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = if append {
        OpenOptions::new().create(true).append(true).open(path)?
    } else {
        File::create(path)?
    };
    Ok(BufWriter::new(file))
}

fn write_frame_header(
    writer: &mut impl Write,
    timestep: usize,
    particles: usize,
    box_low: Vec3,
    box_high: Vec3,
    periodic: [bool; 3],
) -> Result<(), std::io::Error> {
    writeln!(writer, "ITEM: TIMESTEP")?;
    writeln!(writer, "{timestep}")?;
    writeln!(writer, "ITEM: NUMBER OF ATOMS")?;
    writeln!(writer, "{particles}")?;
    writeln!(
        writer,
        "ITEM: BOX BOUNDS {} {} {}",
        boundary_flag(periodic[0]),
        boundary_flag(periodic[1]),
        boundary_flag(periodic[2])
    )?;
    for axis in 0..3 {
        writeln!(writer, "{:.17e} {:.17e}", box_low[axis], box_high[axis])?;
    }
    Ok(())
}

/// Writes a reproducible OVITO Python recipe for loading and styling a
/// trajectory, then saving it as an `.ovito` session.
pub fn write_ovito_view_script(
    config: &OvitoTrajectoryConfig,
    script_path: &Path,
) -> Result<(), ExportError> {
    if let Some(parent) = script_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let dump_expression = python_path_expression(&config.dump_path, script_path);
    let session_expression = config
        .session_path
        .as_ref()
        .map(|path| python_path_expression(path, script_path));
    let shape = match config.representation {
        OvitoRepresentation::FiberSegments => "Spherocylinder",
        OvitoRepresentation::ConnectedFiberSegments => "Sphere",
        OvitoRepresentation::DemParticles { .. } => "Sphere",
    };
    let mut writer = BufWriter::new(File::create(script_path)?);
    writeln!(
        writer,
        "# Generated by TANGLE. Run with: ovitos {}",
        script_path.display()
    )?;
    writeln!(writer, "from pathlib import Path")?;
    writeln!(writer, "import ovito")?;
    writeln!(writer, "from ovito.io import import_file")?;
    writeln!(
        writer,
        "from ovito.modifiers import AssignColorModifier, ColorCodingModifier, ExpressionSelectionModifier"
    )?;
    writeln!(writer, "from ovito.vis import ParticlesVis")?;
    writeln!(writer)?;
    writeln!(writer, "HERE = Path(__file__).resolve().parent")?;
    writeln!(writer, "trajectory_path = {dump_expression}")?;
    writeln!(
        writer,
        "pipeline = import_file(str(trajectory_path), sort_particles=True)"
    )?;
    writeln!(
        writer,
        "print(f'Loaded {{pipeline.num_frames}} OVITO frames from {{trajectory_path}}')"
    )?;
    if config.representation == OvitoRepresentation::ConnectedFiberSegments {
        let sphere_type = config
            .atom_type
            .checked_add(1)
            .ok_or(ExportError::IndexOverflow)?;
        writeln!(writer, "def setup_connected_fibers(frame, data):")?;
        writeln!(writer, "    types = data.particles_.particle_types_")?;
        writeln!(
            writer,
            "    types.type_by_id_({}).shape = ParticlesVis.Shape.Cylinder",
            config.atom_type
        )?;
        writeln!(
            writer,
            "    types.type_by_id_({sphere_type}).shape = ParticlesVis.Shape.Sphere"
        )?;
        writeln!(writer, "pipeline.modifiers.append(setup_connected_fibers)")?;
    }
    match config.coloring {
        OvitoColoring::Fiber => writeln!(
            writer,
            "pipeline.modifiers.append(ColorCodingModifier(property='Molecule Identifier'))"
        )?,
        OvitoColoring::CurvatureRatio => {
            writeln!(
                writer,
                "# 0 = straight; 1 = admissible bend limit; >1 = violation"
            )?;
            writeln!(
                writer,
                "pipeline.modifiers.append(ColorCodingModifier(property='curvature_ratio', start_value=0.0, end_value=1.0))"
            )?;
            writeln!(
                writer,
                "pipeline.modifiers.append(ExpressionSelectionModifier(expression='curvature_ratio > 1.0'))"
            )?;
            writeln!(
                writer,
                "pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.0, 0.0)))"
            )?;
        }
        OvitoColoring::RefinementLevel => writeln!(
            writer,
            "pipeline.modifiers.append(ColorCodingModifier(property='refinement_level'))"
        )?,
    }
    writeln!(writer, "pipeline.add_to_scene(name='TANGLE relaxation')")?;
    writeln!(writer, "data = pipeline.compute()")?;
    writeln!(
        writer,
        "data.particles.vis.shape = ParticlesVis.Shape.{shape}"
    )?;
    writeln!(writer, "data.cell.vis.enabled = True")?;
    if let Some(session_expression) = session_expression {
        writeln!(writer, "session_path = {session_expression}")?;
        writeln!(writer, "ovito.scene.save(str(session_path))")?;
        writeln!(writer, "print(f'Saved OVITO session: {{session_path}}')")?;
    }
    writer.flush()?;
    Ok(())
}

fn python_path_expression(path: &Path, script_path: &Path) -> String {
    if path.parent() == script_path.parent() {
        if let Some(name) = path.file_name() {
            return format!("HERE / '{}'", escape_python_string(&name.to_string_lossy()));
        }
    }
    format!("Path('{}')", escape_python_string(&path.to_string_lossy()))
}

fn escape_python_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

fn boundary_flag(periodic: bool) -> &'static str {
    if periodic {
        "pp"
    } else {
        "ff"
    }
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

fn norm(value: Vec3) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DemParticle;
    use tangle_core::{FiberId, PeriodicCell};

    #[test]
    fn writes_ovito_recognized_dem_particle_columns() {
        let model = DemBpmModel {
            particles: vec![DemParticle {
                id: 1,
                molecule: 7,
                atom_type: 2,
                diameter: 0.25,
                density: 1.0,
                position: [0.1, 0.2, 0.3],
            }],
            bonds: Vec::new(),
            atom_types: Vec::new(),
            box_low: [0.0; 3],
            box_high: [1.0; 3],
        };
        let mut output = Vec::new();
        write_frame_header(
            &mut output,
            12,
            model.particles.len(),
            model.box_low,
            model.box_high,
            [true, false, false],
        )
        .unwrap();
        writeln!(output, "ITEM: ATOMS id mol type diameter x y z").unwrap();
        let text = String::from_utf8(output).unwrap();

        assert!(text.contains("ITEM: TIMESTEP\n12"));
        assert!(text.contains("ITEM: BOX BOUNDS pp ff ff"));
        assert!(text.contains("ITEM: ATOMS id mol type diameter x y z"));
    }

    #[test]
    fn straight_fiber_becomes_one_oriented_capsule() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        assembly
            .add_fiber(
                FiberId(7),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.25, 1.0, 1.0], [1.75, 1.0, 1.0]],
            )
            .unwrap();

        let capsules = build_fiber_capsules(&assembly, 3).unwrap();
        assert_eq!(capsules.len(), 1);
        assert_eq!(capsules[0].fiber_id, 7);
        assert_eq!(capsules[0].fiber_points, 2);
        assert!((capsules[0].length - 1.5).abs() < 1.0e-12);
        assert_eq!(capsules[0].position, [1.0; 3]);
        assert_eq!(capsules[0].natural_curvature, 0.0);
        assert_eq!(capsules[0].current_curvature, 0.0);
        assert!((capsules[0].orientation[1] - 0.5_f64.sqrt()).abs() < 1.0e-12);
        assert!((capsules[0].orientation[3] - 0.5_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn connected_view_writes_one_body_and_shared_vertex_spheres() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        assembly
            .add_fiber(
                FiberId(7),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.25, 1.0, 1.0], [1.75, 1.0, 1.0]],
            )
            .unwrap();
        let directory = std::env::temp_dir().join(format!(
            "tangle-ovito-connected-frame-{}",
            std::process::id()
        ));
        let path = directory.join("frame.dump");
        let config = OvitoTrajectoryConfig {
            dump_path: path.clone(),
            view_script_path: None,
            session_path: None,
            frame_interval: 1,
            write_initial_frame: true,
            representation: OvitoRepresentation::ConnectedFiberSegments,
            coloring: OvitoColoring::Fiber,
            atom_type: 3,
        };

        write_ovito_assembly_frame(&assembly, &config, 0, false).unwrap();
        let dump = std::fs::read_to_string(&path).unwrap();
        assert!(dump.contains("ITEM: NUMBER OF ATOMS\n3\n"));
        assert_eq!(
            dump.lines()
                .filter(|line| line.starts_with("1 7 3 "))
                .count(),
            1
        );
        assert_eq!(
            dump.lines()
                .filter(|line| line.starts_with("2 7 4 ") || line.starts_with("3 7 4 "))
                .count(),
            2
        );
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(directory);
    }

    #[test]
    fn native_spherocylinder_writes_centerline_distance_as_body_length() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        assembly
            .add_fiber(
                FiberId(7),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.25, 1.0, 1.0], [1.75, 1.0, 1.0]],
            )
            .unwrap();
        let directory = std::env::temp_dir().join(format!(
            "tangle-ovito-spherocylinder-frame-{}",
            std::process::id()
        ));
        let dump_path = directory.join("frame.dump");
        let script_path = directory.join("view.py");
        let config = OvitoTrajectoryConfig::fiber_segments(&dump_path, 1)
            .with_viewing_files(&script_path, directory.join("frame.ovito"));

        write_ovito_assembly_frame(&assembly, &config, 0, false).unwrap();
        write_ovito_view_script(&config, &script_path).unwrap();

        let dump = std::fs::read_to_string(&dump_path).unwrap();
        assert!(dump.contains("AsphericalShape.X AsphericalShape.Y AsphericalShape.Z"));
        assert!(!dump.contains("shapex shapey shapez"));
        let particle = dump
            .lines()
            .find(|line| line.starts_with("1 7 1 "))
            .unwrap();
        let fields = particle.split_whitespace().collect::<Vec<_>>();
        assert!((fields[5].parse::<f64>().unwrap() - 1.5).abs() < 1.0e-12);
        let script = std::fs::read_to_string(&script_path).unwrap();
        assert!(script.contains("ParticlesVis.Shape.Spherocylinder"));

        let _ = std::fs::remove_file(dump_path);
        let _ = std::fs::remove_file(script_path);
        let _ = std::fs::remove_dir(directory);
    }

    #[test]
    fn generated_view_recipe_uses_portable_sibling_paths() {
        let config = OvitoTrajectoryConfig {
            dump_path: PathBuf::from("output/relaxation.dump"),
            view_script_path: Some(PathBuf::from("output/relaxation_view.py")),
            session_path: Some(PathBuf::from("output/relaxation.ovito")),
            frame_interval: 1,
            write_initial_frame: true,
            representation: OvitoRepresentation::FiberSegments,
            coloring: OvitoColoring::Fiber,
            atom_type: 1,
        };
        assert_eq!(
            python_path_expression(&config.dump_path, config.view_script_path.as_ref().unwrap()),
            "HERE / 'relaxation.dump'"
        );
        assert_eq!(
            python_path_expression(
                config.session_path.as_ref().unwrap(),
                config.view_script_path.as_ref().unwrap()
            ),
            "HERE / 'relaxation.ovito'"
        );
    }

    #[test]
    fn curvature_coloring_recipe_selects_exported_ratio() {
        let config = OvitoTrajectoryConfig {
            dump_path: PathBuf::from("output/relaxation.dump"),
            view_script_path: Some(PathBuf::from("output/relaxation_view.py")),
            session_path: None,
            frame_interval: 1,
            write_initial_frame: true,
            representation: OvitoRepresentation::FiberSegments,
            coloring: OvitoColoring::CurvatureRatio,
            atom_type: 1,
        };
        let directory = std::env::temp_dir().join(format!(
            "tangle-ovito-curvature-script-{}",
            std::process::id()
        ));
        let path = directory.join("view.py");
        write_ovito_view_script(&config, &path).unwrap();
        let script = std::fs::read_to_string(&path).unwrap();
        assert!(script.contains("property='curvature_ratio'"));
        assert!(script.contains("start_value=0.0"));
        assert!(script.contains("end_value=1.0"));
        assert!(script.contains("expression='curvature_ratio > 1.0'"));
        assert!(script.contains("AssignColorModifier(color=(1.0, 0.0, 0.0))"));
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(directory);
    }

    #[test]
    fn oval_fibers_draw_as_rows_of_lane_capsules() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("oval");
        let section = assembly.sections.add(Section::Elliptical {
            semi_axes: [0.06, 0.04],
        });
        let points = [[0.2, 0.5, 0.5], [0.5, 0.5, 0.5], [0.8, 0.5, 0.5]];
        assembly
            .add_fiber(FiberId(3), material, section, &points, &points)
            .unwrap();

        let capsules = build_fiber_capsules(&assembly, 1).unwrap();
        // Offset 0.02 over lane radius 0.04 gives two lane gaps, three lanes.
        assert_eq!(capsules.len(), 2 * 3);
        assert!(capsules.iter().all(|capsule| capsule.radius == 0.04));
        let widths = capsules[0..3]
            .iter()
            .map(|capsule| capsule.position[1] - 0.5)
            .collect::<Vec<_>>();
        assert!((widths[0] + 0.02).abs() < 1.0e-12);
        assert!(widths[1].abs() < 1.0e-12);
        assert!((widths[2] - 0.02).abs() < 1.0e-12);
    }
}

