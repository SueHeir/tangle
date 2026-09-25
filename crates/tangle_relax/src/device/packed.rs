use std::error::Error;
use std::fmt;

use tangle_core::{polyline_tangents, FiberAssembly, GeometryState, Section, Span, Vec3};

/// Device-resident midpoint subdivision parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveSegmentationConfig {
    /// Desired segment length near contact, divided by fiber diameter.
    pub contact_length_over_diameter: f32,
    /// Hard lower segment-length bound, divided by fiber diameter.
    pub minimum_length_over_diameter: f32,
    /// Maximum midpoint subdivision depth reserved for each fiber.
    pub maximum_refinement_levels: u32,
    /// Solver iterations between device-side refinement decisions.
    ///
    /// This cadence is independent of GRASS batch boundaries.
    pub refinement_interval: usize,
    /// Consecutive adaptation checks with excessive penetration required
    /// before a segment may split.
    pub refinement_persistence: u32,
    /// Consecutive contact-free adaptation checks required before a sibling
    /// pair may merge. Zero disables coarsening.
    pub coarsening_persistence: u32,
    /// Maximum midpoint-to-parent-chord error, divided by fiber diameter,
    /// accepted when merging two child segments.
    pub coarsening_error_over_diameter: f32,
    /// Maximum admissible-curvature ratio at a midpoint that may be removed.
    pub coarsening_curvature_ratio: f32,
}

impl Default for AdaptiveSegmentationConfig {
    fn default() -> Self {
        Self {
            contact_length_over_diameter: 2.0,
            minimum_length_over_diameter: 0.5,
            maximum_refinement_levels: 8,
            refinement_interval: 8,
            refinement_persistence: 3,
            coarsening_persistence: 32,
            coarsening_error_over_diameter: 0.1,
            coarsening_curvature_ratio: 0.25,
        }
    }
}

/// Flat structure-of-arrays input uploaded to a CubeCL runtime.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PackedAssembly {
    /// Interleaved xyz coordinates, three values per vertex.
    pub positions: Vec<f32>,
    /// Interleaved intrinsic xyz coordinates, three values per vertex.
    pub intrinsic_positions: Vec<f32>,
    /// Two global vertex indices per segment.
    pub segment_vertices: Vec<u32>,
    /// Dense owner-fiber index per segment.
    pub segment_fibers: Vec<u32>,
    /// Bounding radius per segment: the capsule radius of a round fiber, or
    /// the long semi-axis of an oval one.
    pub segment_radii: Vec<f32>,
    /// Oval sections only: distance from the centerline to the outermost of
    /// the round lanes that model the oval in contact, along the vertex
    /// director. Each lane's radius is `segment_radii - segment_lane_offsets`.
    /// Zero for round fibers; empty in checkpoints written before ovals.
    #[serde(default)]
    pub segment_lane_offsets: Vec<f32>,
    /// Oval sections only: unit long-axis direction per vertex, interleaved
    /// xyz. Zero for round fibers; empty in checkpoints written before ovals.
    #[serde(default)]
    pub directors: Vec<f32>,
    /// Intrinsic length per segment.
    pub segment_rest_lengths: Vec<f32>,
    /// One for active leaf segments and zero for inactive tree nodes.
    pub segment_active: Vec<u32>,
    /// Left and right child segment indices, or `u32::MAX` for leaves.
    pub segment_children: Vec<u32>,
    /// Refinement epoch in which a segment became active.
    pub segment_birth_epochs: Vec<u32>,
    /// Consecutive adaptation checks with unresolved contact per segment.
    pub segment_contact_epochs: Vec<u32>,
    /// Consecutive contact-free, geometrically mergeable checks per parent.
    pub segment_quiet_epochs: Vec<u32>,
    /// Dyadic refinement level of every reserved segment.
    pub segment_refinement_levels: Vec<u32>,
    /// First segment and segment count for each fiber.
    pub fiber_segment_spans: Vec<u32>,
    /// First vertex and vertex count for each fiber.
    pub fiber_vertex_spans: Vec<u32>,
    /// Dense owner-fiber index per vertex.
    pub vertex_fibers: Vec<u32>,
    /// Optional manufacturing layer per fiber, or `u32::MAX` when absent.
    pub fiber_formation_layers: Vec<u32>,
    /// Manufacturing recipe step at which each fiber becomes active.
    pub fiber_formation_steps: Vec<u32>,
    /// Previous and next segment per vertex, or `u32::MAX` when absent.
    pub vertex_segments: Vec<u32>,
    /// Intrinsic chord across each internal vertex; zero at endpoints.
    pub vertex_rest_chords: Vec<f32>,
    /// Maximum admissible curvature per vertex; zero means unrestricted.
    pub vertex_max_curvature: Vec<f32>,
    /// One for vertices currently belonging to the active centerline.
    pub vertex_active: Vec<u32>,
    /// One for vertices whose positions are kinematically pinned.
    pub vertex_pinned: Vec<u32>,
    /// Dyadic refinement level at which each vertex becomes active.
    pub vertex_refinement_levels: Vec<u32>,
    /// Lower xyz bounds of the orthorhombic cell.
    pub cell_lower: [f32; 3],
    /// Upper xyz bounds of the orthorhombic cell.
    pub cell_upper: [f32; 3],
    /// One for periodic cell axes and zero for bounded axes.
    pub cell_periodic: [u32; 3],
    /// Whether this layout contains inactive adaptive tree capacity.
    pub adaptive: bool,
    /// Whether any fiber has an oval section, so directors are updated.
    #[serde(default)]
    pub has_ovals: bool,
    /// Whether adaptive coarsening is safe for this topology.
    ///
    /// Assemblies containing persistent junctions remain refined because the
    /// current device solver does not yet carry junction-anchor protection.
    pub coarsening_safe: bool,
}

impl PackedAssembly {
    /// Converts a solver-neutral assembly to the device layout.
    pub fn from_assembly(assembly: &FiberAssembly) -> Result<Self, PackingError> {
        Self::from_assembly_with_options(assembly, None, false)
    }

    /// Converts an assembly using optional device-resident adaptive capacity.
    pub fn from_assembly_with_options(
        assembly: &FiberAssembly,
        adaptive: Option<AdaptiveSegmentationConfig>,
        pin_fiber_ends: bool,
    ) -> Result<Self, PackingError> {
        assembly
            .validate()
            .map_err(|error| PackingError::InvalidAssembly(error.to_string()))?;
        let origin = assembly.cell.origin;
        let basis = assembly.cell.basis;
        if basis[0][1] != 0.0
            || basis[0][2] != 0.0
            || basis[1][0] != 0.0
            || basis[1][2] != 0.0
            || basis[2][0] != 0.0
            || basis[2][1] != 0.0
            || basis[0][0] <= 0.0
            || basis[1][1] <= 0.0
            || basis[2][2] <= 0.0
        {
            return Err(PackingError::NonOrthorhombicCellUnsupported);
        }

        if let Some(config) = adaptive {
            return Self::from_adaptive_fibers(assembly, config, pin_fiber_ends, origin, basis);
        }

        let positions: Vec<f32> = assembly
            .geometry
            .placed
            .positions
            .iter()
            .flat_map(|point| point.iter().map(|value| *value as f32))
            .collect();
        let intrinsic_positions = assembly
            .geometry
            .intrinsic
            .positions
            .iter()
            .flat_map(|point| point.iter().map(|value| *value as f32))
            .collect();
        let mut segment_vertices = Vec::new();
        let mut segment_fibers = Vec::new();
        let mut segment_radii = Vec::new();
        let mut segment_lane_offsets = Vec::new();
        let mut directors = vec![0.0_f32; 3 * assembly.geometry.placed.positions.len()];
        let mut has_ovals = false;
        let mut segment_rest_lengths = Vec::new();
        let mut fiber_segment_spans = Vec::with_capacity(2 * assembly.topology.fibers.len());
        let mut fiber_vertex_spans = Vec::with_capacity(2 * assembly.topology.fibers.len());
        let mut vertex_fibers = vec![0; assembly.geometry.placed.positions.len()];
        let mut fiber_formation_layers = Vec::with_capacity(assembly.topology.fibers.len());
        let mut fiber_formation_steps = Vec::with_capacity(assembly.topology.fibers.len());
        let mut vertex_segments = vec![u32::MAX; 2 * assembly.geometry.placed.positions.len()];
        let mut vertex_rest_chords = vec![0.0; assembly.geometry.placed.positions.len()];
        let mut vertex_max_curvature = vec![0.0; assembly.geometry.placed.positions.len()];
        let mut vertex_pinned = vec![0; assembly.geometry.placed.positions.len()];

        for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
            fiber_formation_layers.push(fiber.formation_layer.unwrap_or(u32::MAX));
            fiber_formation_steps.push(fiber.formation_step);
            let section = assembly
                .sections
                .entries
                .get(fiber.section.0 as usize)
                .ok_or(PackingError::MissingSection(fiber.id.0))?;
            let lanes = LaneSection::from_section(section);
            let radius = lanes.radius;
            if lanes.offset > 0.0 {
                has_ovals = true;
                let start = fiber.vertices.start as usize;
                let placed =
                    &assembly.geometry.placed.positions[start..start + fiber.vertices.len as usize];
                let fiber_directors =
                    lanes.device_directors(placed, assembly.fiber_directors(fiber_index));
                for (local, director) in fiber_directors.iter().enumerate() {
                    for axis in 0..3 {
                        directors[3 * (start + local) + axis] = director[axis] as f32;
                    }
                }
            }
            let segment_start = segment_fibers.len() as u32;
            let vertex_start = fiber.vertices.start;
            let vertex_count = fiber.vertices.len;
            fiber_segment_spans.push(segment_start);
            fiber_segment_spans.push(vertex_count - 1);
            fiber_vertex_spans.push(vertex_start);
            fiber_vertex_spans.push(vertex_count);
            for vertex in vertex_start..vertex_start + vertex_count {
                vertex_fibers[vertex as usize] = fiber_index as u32;
            }
            if pin_fiber_ends {
                vertex_pinned[vertex_start as usize] = 1;
                vertex_pinned[(vertex_start + vertex_count - 1) as usize] = 1;
            }
            for local in 0..vertex_count - 1 {
                let segment_index = segment_fibers.len() as u32;
                segment_vertices.push(vertex_start + local);
                segment_vertices.push(vertex_start + local + 1);
                segment_fibers.push(fiber_index as u32);
                segment_radii.push(radius);
                segment_lane_offsets.push(lanes.offset);
                let intrinsic = &assembly.geometry.intrinsic.positions;
                segment_rest_lengths.push(distance(
                    intrinsic[(vertex_start + local) as usize],
                    intrinsic[(vertex_start + local + 1) as usize],
                ));
                vertex_segments[2 * (vertex_start + local) as usize + 1] = segment_index;
                vertex_segments[2 * (vertex_start + local + 1) as usize] = segment_index;
            }
            for local in 1..vertex_count - 1 {
                let intrinsic = &assembly.geometry.intrinsic.positions;
                vertex_rest_chords[(vertex_start + local) as usize] = distance(
                    intrinsic[(vertex_start + local - 1) as usize],
                    intrinsic[(vertex_start + local + 1) as usize],
                );
                if let Some(limit) = assembly.admissibility.bend_limits[fiber_index] {
                    vertex_max_curvature[(vertex_start + local) as usize] =
                        limit.maximum_curvature() as f32;
                }
            }
        }

        let segment_count = segment_fibers.len();
        let vertex_count = vertex_fibers.len();

        Ok(Self {
            positions,
            intrinsic_positions,
            segment_vertices,
            segment_fibers,
            segment_radii,
            segment_lane_offsets,
            directors,
            segment_rest_lengths,
            segment_active: vec![1; segment_count],
            segment_children: vec![u32::MAX; 2 * segment_count],
            segment_birth_epochs: vec![0; segment_count],
            segment_contact_epochs: vec![0; segment_count],
            segment_quiet_epochs: vec![0; segment_count],
            segment_refinement_levels: vec![0; segment_count],
            fiber_segment_spans,
            fiber_vertex_spans,
            vertex_fibers,
            fiber_formation_layers,
            fiber_formation_steps,
            vertex_segments,
            vertex_rest_chords,
            vertex_max_curvature,
            vertex_active: vec![1; vertex_count],
            vertex_pinned,
            vertex_refinement_levels: vec![0; vertex_count],
            cell_lower: origin.map(|value| value as f32),
            cell_upper: [
                (origin[0] + basis[0][0]) as f32,
                (origin[1] + basis[1][1]) as f32,
                (origin[2] + basis[2][2]) as f32,
            ],
            cell_periodic: assembly.cell.periodic.map(u32::from),
            adaptive: false,
            has_ovals,
            coarsening_safe: assembly.junctions.junctions.is_empty(),
        })
    }

    fn from_adaptive_fibers(
        assembly: &FiberAssembly,
        config: AdaptiveSegmentationConfig,
        pin_fiber_ends: bool,
        origin: [f64; 3],
        basis: [[f64; 3]; 3],
    ) -> Result<Self, PackingError> {
        let mut positions = Vec::new();
        let mut intrinsic_positions = Vec::new();
        let mut segment_vertices = Vec::new();
        let mut segment_fibers = Vec::new();
        let mut segment_radii = Vec::new();
        let mut segment_rest_lengths = Vec::new();
        let mut segment_active = Vec::new();
        let mut segment_children = Vec::new();
        let mut segment_birth_epochs = Vec::new();
        let mut segment_contact_epochs = Vec::new();
        let mut segment_quiet_epochs = Vec::new();
        let mut segment_refinement_levels = Vec::new();
        let mut fiber_segment_spans = Vec::new();
        let mut fiber_vertex_spans = Vec::new();
        let mut fiber_formation_layers = Vec::new();
        let mut fiber_formation_steps = Vec::new();
        let mut vertex_fibers = Vec::new();
        let mut vertex_segments = Vec::new();
        let mut vertex_rest_chords = Vec::new();
        let mut vertex_max_curvature = Vec::new();
        let mut vertex_active = Vec::new();
        let mut vertex_pinned = Vec::new();
        let mut vertex_refinement_levels = Vec::new();

        for (fiber_index, fiber) in assembly.topology.fibers.iter().enumerate() {
            let section = assembly
                .sections
                .entries
                .get(fiber.section.0 as usize)
                .ok_or(PackingError::MissingSection(fiber.id.0))?;
            if !section.is_circular() {
                return Err(PackingError::NonCircularSection(fiber.id.0));
            }
            let radius = LaneSection::from_section(section).radius;
            let diameter = 2.0 * radius;
            let minimum_length = config.minimum_length_over_diameter * diameter;
            let vertex_start = positions.len() as u32 / 3;
            let segment_start = segment_fibers.len() as u32;
            fiber_formation_layers.push(fiber.formation_layer.unwrap_or(u32::MAX));
            fiber_formation_steps.push(fiber.formation_step);
            let maximum_curvature = assembly.admissibility.bend_limits[fiber_index]
                .map(|limit| limit.maximum_curvature() as f32)
                .unwrap_or(0.0);
            let source_start = fiber.vertices.start as usize;
            let source_count = fiber.vertices.len as usize;
            let mut roots = Vec::with_capacity(source_count - 1);

            for base_segment in 0..source_count - 1 {
                let intrinsic_a =
                    assembly.geometry.intrinsic.positions[source_start + base_segment];
                let intrinsic_b =
                    assembly.geometry.intrinsic.positions[source_start + base_segment + 1];
                let placed_a = assembly.geometry.placed.positions[source_start + base_segment];
                let placed_b = assembly.geometry.placed.positions[source_start + base_segment + 1];
                let rest_length = distance(intrinsic_a, intrinsic_b);
                let mut levels = 0_u32;
                while levels < config.maximum_refinement_levels
                    && rest_length / (1_u32 << (levels + 1)) as f32 >= minimum_length
                {
                    levels += 1;
                }
                let leaf_count = 1_u32 << levels;
                let first_vertex = positions.len() as u32 / 3 - u32::from(base_segment > 0);
                let first_local = usize::from(base_segment > 0);
                for local in first_local..=leaf_count as usize {
                    let fraction = local as f64 / leaf_count as f64;
                    for axis in 0..3 {
                        positions.push(
                            (placed_a[axis] + fraction * (placed_b[axis] - placed_a[axis])) as f32,
                        );
                        intrinsic_positions.push(
                            (intrinsic_a[axis] + fraction * (intrinsic_b[axis] - intrinsic_a[axis]))
                                as f32,
                        );
                    }
                    vertex_fibers.push(fiber_index as u32);
                    vertex_segments.extend([u32::MAX, u32::MAX]);
                    vertex_rest_chords.push(0.0);
                    vertex_max_curvature.push(maximum_curvature);
                    let original = local == leaf_count as usize || base_segment == 0 && local == 0;
                    vertex_active.push(u32::from(original));
                    let global_endpoint = (base_segment == 0 && local == 0)
                        || (base_segment == source_count - 2 && local == leaf_count as usize);
                    vertex_pinned.push(u32::from(pin_fiber_ends && global_endpoint));
                    let level = if original {
                        0
                    } else {
                        levels - (local as u32).trailing_zeros()
                    };
                    vertex_refinement_levels.push(level);
                }
                roots.push((first_vertex, leaf_count, levels, rest_length));
            }

            for (first_vertex, leaf_count, levels, rest_length) in roots {
                let root = segment_fibers.len() as u32;
                let node_count = 2 * leaf_count - 1;
                for local_node in 0..node_count {
                    let one_based = local_node + 1;
                    let level = 31 - one_based.leading_zeros();
                    let level_start = 1_u32 << level;
                    let offset = one_based - level_start;
                    let stride = leaf_count >> level;
                    let first = first_vertex + offset * stride;
                    let second = first + stride;
                    segment_vertices.extend([first, second]);
                    segment_fibers.push(fiber_index as u32);
                    segment_radii.push(radius);
                    segment_rest_lengths.push(rest_length / (1_u32 << level) as f32);
                    segment_active.push(u32::from(local_node == 0));
                    if level < levels {
                        segment_children
                            .extend([root + 2 * local_node + 1, root + 2 * local_node + 2]);
                    } else {
                        segment_children.extend([u32::MAX, u32::MAX]);
                    }
                    segment_birth_epochs.push(if local_node == 0 { 0 } else { u32::MAX });
                    segment_contact_epochs.push(0);
                    segment_quiet_epochs.push(0);
                    segment_refinement_levels.push(level);
                }
                vertex_segments[2 * first_vertex as usize + 1] = root;
                vertex_segments[2 * (first_vertex + leaf_count) as usize] = root;
            }

            let vertex_count = positions.len() as u32 / 3 - vertex_start;
            let segment_count = segment_fibers.len() as u32 - segment_start;
            fiber_vertex_spans.extend([vertex_start, vertex_count]);
            fiber_segment_spans.extend([segment_start, segment_count]);
        }

        Ok(Self {
            directors: vec![0.0; positions.len()],
            positions,
            intrinsic_positions,
            segment_vertices,
            segment_fibers,
            segment_lane_offsets: vec![0.0; segment_radii.len()],
            segment_radii,
            segment_rest_lengths,
            segment_active,
            segment_children,
            segment_birth_epochs,
            segment_contact_epochs,
            segment_quiet_epochs,
            segment_refinement_levels,
            fiber_segment_spans,
            fiber_vertex_spans,
            vertex_fibers,
            fiber_formation_layers,
            fiber_formation_steps,
            vertex_segments,
            vertex_rest_chords,
            vertex_max_curvature,
            vertex_active,
            vertex_pinned,
            vertex_refinement_levels,
            cell_lower: origin.map(|value| value as f32),
            cell_upper: [
                (origin[0] + basis[0][0]) as f32,
                (origin[1] + basis[1][1]) as f32,
                (origin[2] + basis[2][2]) as f32,
            ],
            cell_periodic: assembly.cell.periodic.map(u32::from),
            adaptive: true,
            has_ovals: false,
            coarsening_safe: assembly.junctions.junctions.is_empty(),
        })
    }

    /// Fills lane offsets and directors with zeros (round fibers) when a
    /// checkpoint written before oval sections lacks them.
    pub fn ensure_lane_buffers(&mut self) {
        if self.segment_lane_offsets.len() != self.segment_count() {
            self.segment_lane_offsets = vec![0.0; self.segment_count()];
        }
        if self.directors.len() != self.positions.len() {
            self.directors = vec![0.0; self.positions.len()];
        }
    }

    /// Per-vertex, per-axis distance from the centerline to a wall the fiber
    /// touches: the radius for round fibers, and for ovals the lane radius
    /// plus the outer lane's offset along that axis.
    pub fn vertex_wall_extents(&self) -> Vec<f32> {
        let mut extents = vec![0.0_f32; self.positions.len()];
        for fiber in 0..self.fiber_count() {
            let segment = self.fiber_segment_spans[2 * fiber] as usize;
            let radius = self.segment_radii[segment];
            let offset = self
                .segment_lane_offsets
                .get(segment)
                .copied()
                .unwrap_or(0.0);
            let start = self.fiber_vertex_spans[2 * fiber] as usize;
            let count = self.fiber_vertex_spans[2 * fiber + 1] as usize;
            for vertex in start..start + count {
                for axis in 0..3 {
                    extents[3 * vertex + axis] = if offset > 0.0 {
                        radius - offset + offset * self.directors[3 * vertex + axis].abs()
                    } else {
                        radius
                    };
                }
            }
        }
        extents
    }

    /// Number of vertices in the packed representation.
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Number of segments in the packed representation.
    pub fn segment_count(&self) -> usize {
        self.segment_fibers.len()
    }

    /// Number of fibers in the packed representation.
    pub fn fiber_count(&self) -> usize {
        self.fiber_vertex_spans.len() / 2
    }

    /// Replaces an assembly's placed positions with a downloaded device state.
    /// Downloaded directors, given for layouts with oval fibers, replace the
    /// placed directors.
    pub fn unpack_positions(
        &self,
        positions: &[f32],
        directors: Option<&[f32]>,
        assembly: &mut FiberAssembly,
    ) -> Result<(), PackingError> {
        if positions.len() != self.positions.len()
            || assembly.geometry.placed.positions.len() != self.vertex_count()
            || directors.is_some_and(|directors| directors.len() != positions.len())
        {
            return Err(PackingError::PositionLengthMismatch);
        }
        for (target, xyz) in assembly
            .geometry
            .placed
            .positions
            .iter_mut()
            .zip(positions.chunks_exact(3))
        {
            *target = [xyz[0] as f64, xyz[1] as f64, xyz[2] as f64];
        }
        if let Some(directors) = directors.filter(|_| self.has_ovals) {
            assembly.ensure_directors();
            for fiber in &assembly.topology.fibers {
                let section = assembly.sections.entries[fiber.section.0 as usize];
                let lanes = LaneSection::from_section(&section);
                if lanes.offset <= 0.0 {
                    continue;
                }
                let start = fiber.vertices.start as usize;
                let range = start..start + fiber.vertices.len as usize;
                let device = directors[3 * range.start..3 * range.end]
                    .chunks_exact(3)
                    .map(|xyz| [xyz[0] as f64, xyz[1] as f64, xyz[2] as f64])
                    .collect::<Vec<_>>();
                let host = lanes
                    .host_directors(&assembly.geometry.placed.positions[range.clone()], device);
                assembly.geometry.placed.directors[range].copy_from_slice(&host);
            }
        }
        Ok(())
    }

    /// Rebuilds an assembly from the active vertices of an adaptive device layout.
    pub fn unpack_active_positions(
        &self,
        positions: &[f32],
        directors: Option<&[f32]>,
        active: &[u32],
        assembly: &mut FiberAssembly,
    ) -> Result<(), PackingError> {
        if positions.len() != self.positions.len()
            || active.len() != self.vertex_count()
            || directors.is_some_and(|directors| directors.len() != positions.len())
        {
            return Err(PackingError::PositionLengthMismatch);
        }
        // Staged fibers of a uniform layout may be oval; keep their directors.
        // Adaptive layouts hold round fibers only.
        let directors = directors.filter(|_| self.has_ovals);
        let mut placed_directors = Vec::new();
        let mut placed = Vec::new();
        let mut intrinsic = Vec::new();
        let source_fibers = std::mem::take(&mut assembly.topology.fibers);
        let source_limits = std::mem::take(&mut assembly.admissibility.bend_limits);
        let source_fiber_count = source_fibers.len();
        let mut fibers = Vec::with_capacity(source_fiber_count);
        let mut bend_limits = Vec::with_capacity(source_fiber_count);
        for (fiber_index, (mut fiber, bend_limit)) in
            source_fibers.into_iter().zip(source_limits).enumerate()
        {
            let start = self.fiber_vertex_spans[2 * fiber_index] as usize;
            let count = self.fiber_vertex_spans[2 * fiber_index + 1] as usize;
            let new_start = u32::try_from(placed.len()).map_err(|_| PackingError::IndexOverflow)?;
            for vertex in start..start + count {
                if active[vertex] == 0 {
                    continue;
                }
                placed.push([
                    positions[3 * vertex] as f64,
                    positions[3 * vertex + 1] as f64,
                    positions[3 * vertex + 2] as f64,
                ]);
                intrinsic.push([
                    self.intrinsic_positions[3 * vertex] as f64,
                    self.intrinsic_positions[3 * vertex + 1] as f64,
                    self.intrinsic_positions[3 * vertex + 2] as f64,
                ]);
                if let Some(directors) = directors {
                    placed_directors.push([
                        directors[3 * vertex] as f64,
                        directors[3 * vertex + 1] as f64,
                        directors[3 * vertex + 2] as f64,
                    ]);
                }
            }
            let len = u32::try_from(placed.len() - new_start as usize)
                .map_err(|_| PackingError::IndexOverflow)?;
            if len == 0 {
                continue;
            }
            if directors.is_some() {
                let section = assembly.sections.entries[fiber.section.0 as usize];
                let range = new_start as usize..placed.len();
                let device = placed_directors[range.clone()].to_vec();
                placed_directors[range.clone()].copy_from_slice(
                    &LaneSection::from_section(&section).host_directors(&placed[range], device),
                );
            }
            if len < 2 {
                return Err(PackingError::ActiveFiberTooShort {
                    id: fiber.id.0,
                    vertices: len,
                });
            }
            fiber.vertices = Span {
                start: new_start,
                len,
            };
            fibers.push(fiber);
            bend_limits.push(bend_limit);
        }
        assembly.topology.fibers = fibers;
        assembly.admissibility.bend_limits = bend_limits;
        assembly.geometry.placed = GeometryState {
            positions: placed,
            directors: placed_directors,
        };
        assembly.geometry.intrinsic = GeometryState {
            positions: intrinsic,
            directors: Vec::new(),
        };
        assembly.geometry.assembled_reference = None;
        if assembly.topology.fibers.len() != source_fiber_count {
            assembly.junctions = Default::default();
        }
        Ok(())
    }
}

/// Geometry that the initial CubeCL capsule backend cannot represent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackingError {
    /// The source assembly failed core validation.
    InvalidAssembly(String),
    /// The first backend accepts only axis-aligned cells.
    NonOrthorhombicCellUnsupported,
    /// A fiber referenced a missing section.
    MissingSection(u32),
    /// Adaptive segmentation accepts only circular sections.
    NonCircularSection(u32),
    /// Downloaded and host geometry lengths differ.
    PositionLengthMismatch,
    /// An activity mask exposed only part of one fiber centerline.
    ActiveFiberTooShort {
        /// Stable fiber identifier.
        id: u32,
        /// Number of active vertices.
        vertices: u32,
    },
    /// A packed index exceeded `u32` capacity.
    IndexOverflow,
}

impl fmt::Display for PackingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAssembly(message) => write!(formatter, "invalid assembly: {message}"),
            Self::NonOrthorhombicCellUnsupported => {
                formatter.write_str("CubeCL relaxation currently requires an orthorhombic cell")
            }
            Self::MissingSection(id) => {
                write!(formatter, "fiber {id} references a missing section")
            }
            Self::NonCircularSection(id) => write!(
                formatter,
                "fiber {id} has an oval section, which adaptive segmentation does not support yet"
            ),
            Self::PositionLengthMismatch => {
                formatter.write_str("downloaded position buffer has the wrong length")
            }
            Self::ActiveFiberTooShort { id, vertices } => write!(
                formatter,
                "active fiber {id} has only {vertices} active centerline vertices"
            ),
            Self::IndexOverflow => {
                formatter.write_str("adaptive packed index exceeded u32 capacity")
            }
        }
    }
}

impl Error for PackingError {}

fn distance(first: [f64; 3], second: [f64; 3]) -> f32 {
    let x = second[0] - first[0];
    let y = second[1] - first[1];
    let z = second[2] - first[2];
    (x * x + y * y + z * z).sqrt() as f32
}

/// Most lanes one oval is split into for contact.
pub const MAXIMUM_OVAL_LANES: u32 = 5;
/// Largest spacing between neighbouring lanes, as a fraction of the lane
/// radius. At 0.6 the boundary between two lanes dips at most 4.6 % of the
/// lane radius below a flat side.
pub const OVAL_LANE_SPACING: f32 = 0.6;

/// Number of round lanes that model an oval of the given outer lane offset
/// and lane radius; one for a round fiber. The contact kernel computes the
/// same count.
pub fn oval_lane_count(offset: f32, lane_radius: f32) -> u32 {
    if offset <= 0.0 {
        return 1;
    }
    let gaps = (2.0 * offset / (OVAL_LANE_SPACING * lane_radius)).ceil() as u32;
    (gaps + 1).clamp(2, MAXIMUM_OVAL_LANES)
}

/// How one section is packed: a bounding radius plus, for an oval, the
/// offset of its outermost lane and which semi-axis is the long one.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LaneSection {
    radius: f32,
    offset: f32,
    /// The long axis is the second semi-axis, so device directors are the
    /// stored directors turned a quarter turn about the tangent.
    turned: bool,
}

impl LaneSection {
    fn from_section(section: &Section) -> Self {
        match *section {
            Section::Circular { radius } => Self {
                radius: radius as f32,
                offset: 0.0,
                turned: false,
            },
            Section::Elliptical { semi_axes } => {
                let long = semi_axes[0].max(semi_axes[1]);
                let short = semi_axes[0].min(semi_axes[1]);
                Self {
                    radius: long as f32,
                    offset: (long - short) as f32,
                    turned: semi_axes[1] > semi_axes[0],
                }
            }
        }
    }

    /// Stored directors point along the first semi-axis; device directors
    /// always point along the long one.
    fn device_directors(&self, placed: &[Vec3], directors: Vec<Vec3>) -> Vec<Vec3> {
        if self.turned {
            quarter_turn(placed, directors)
        } else {
            directors
        }
    }

    fn host_directors(&self, placed: &[Vec3], directors: Vec<Vec3>) -> Vec<Vec3> {
        let mut directors = if self.turned {
            quarter_turn(placed, directors)
        } else {
            directors
        };
        tangle_core::orthonormalize_directors(placed, &mut directors);
        directors
    }
}

/// Turns each director a quarter turn about the local tangent. Ellipses are
/// symmetric, so turning either way swaps the two semi-axes.
fn quarter_turn(placed: &[Vec3], directors: Vec<Vec3>) -> Vec<Vec3> {
    polyline_tangents(placed)
        .into_iter()
        .zip(directors)
        .map(|(t, d)| {
            [
                t[1] * d[2] - t[2] * d[1],
                t[2] * d[0] - t[0] * d[2],
                t[0] * d[1] - t[1] * d[0],
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell};

    #[test]
    fn preserves_optional_formation_layers() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.01 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                &[[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]],
            )
            .unwrap();
        assembly
            .set_fiber_formation_layer(FiberId(1), Some(3))
            .unwrap();
        assembly.set_fiber_formation_step(FiberId(1), 2).unwrap();

        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        assert_eq!(packed.fiber_formation_layers, vec![3]);
        assert_eq!(packed.fiber_formation_steps, vec![2]);
    }

    #[test]
    fn reserves_a_diameter_limited_midpoint_tree() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.025 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
                &[[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]],
            )
            .unwrap();

        let packed = PackedAssembly::from_assembly_with_options(
            &assembly,
            Some(AdaptiveSegmentationConfig {
                contact_length_over_diameter: 2.0,
                minimum_length_over_diameter: 1.0,
                maximum_refinement_levels: 8,
                refinement_interval: 8,
                ..AdaptiveSegmentationConfig::default()
            }),
            true,
        )
        .unwrap();

        // 0.8 / 2^4 = 0.05 = one diameter, so the complete tree has
        // 16 leaves, 31 segment nodes, and 17 vertex slots.
        assert_eq!(packed.segment_count(), 31);
        assert_eq!(packed.vertex_count(), 17);
        assert_eq!(packed.segment_active.iter().sum::<u32>(), 1);
        assert_eq!(packed.vertex_active.iter().sum::<u32>(), 2);
        assert_eq!(packed.vertex_pinned.iter().sum::<u32>(), 2);
        assert_eq!(packed.segment_children[0..2], [1, 2]);
        assert_eq!(packed.segment_refinement_levels[0], 0);
        assert_eq!(packed.segment_refinement_levels[30], 4);
    }

    #[test]
    fn reserves_independent_trees_for_a_multisegment_fiber() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.025 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[-0.4, 0.0, 0.0], [0.0, 0.1, 0.0], [0.4, 0.0, 0.0]],
                &[[0.1, 0.5, 0.5], [0.5, 0.6, 0.5], [0.9, 0.5, 0.5]],
            )
            .unwrap();

        let packed = PackedAssembly::from_assembly_with_options(
            &assembly,
            Some(AdaptiveSegmentationConfig {
                contact_length_over_diameter: 2.0,
                minimum_length_over_diameter: 1.0,
                maximum_refinement_levels: 8,
                refinement_interval: 8,
                ..AdaptiveSegmentationConfig::default()
            }),
            false,
        )
        .unwrap();

        // Each approximately 0.412-long source segment reserves an eight-leaf
        // tree. The shared source vertex appears only once in material order.
        assert_eq!(packed.segment_count(), 30);
        assert_eq!(packed.vertex_count(), 17);
        assert_eq!(packed.segment_active.iter().sum::<u32>(), 2);
        assert_eq!(packed.vertex_active.iter().sum::<u32>(), 3);
        assert_eq!(packed.fiber_segment_spans, vec![0, 30]);
        assert_eq!(packed.fiber_vertex_spans, vec![0, 17]);
        assert_eq!(packed.vertex_segments[2 * 8], 0);
        assert_eq!(packed.vertex_segments[2 * 8 + 1], 15);
    }

    fn oval_assembly(semi_axes: [f64; 2]) -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("oval");
        let section = assembly.sections.add(Section::Elliptical { semi_axes });
        let points = [[0.2, 0.5, 0.5], [0.5, 0.5, 0.5], [0.8, 0.5, 0.5]];
        assembly
            .add_fiber(FiberId(1), material, section, &points, &points)
            .unwrap();
        assembly
    }

    #[test]
    fn packs_ovals_as_lanes_along_the_long_axis() {
        let packed = PackedAssembly::from_assembly(&oval_assembly([0.03, 0.02])).unwrap();
        assert!(packed.has_ovals);
        assert_eq!(packed.segment_radii, vec![0.03; 2]);
        assert!((packed.segment_lane_offsets[0] - 0.01).abs() < 1.0e-7);
        // Default long axis lies flat, across the fiber.
        assert_eq!(&packed.directors[0..3], &[0.0, 1.0, 0.0]);
        let extents = packed.vertex_wall_extents();
        assert!((extents[1] - 0.03).abs() < 1.0e-7);
        assert!((extents[2] - 0.02).abs() < 1.0e-7);

        // Listing the short semi-axis first turns the device director so it
        // still points along the long axis.
        let turned = PackedAssembly::from_assembly(&oval_assembly([0.02, 0.03])).unwrap();
        assert_eq!(turned.segment_radii, packed.segment_radii);
        assert!(turned.directors[1].abs() < 1.0e-7);
        assert!((turned.directors[2].abs() - 1.0).abs() < 1.0e-7);
    }

    #[test]
    fn round_fibers_pack_without_lanes() {
        let mut assembly = oval_assembly([0.03, 0.03]);
        assembly.sections.entries[0] = Section::Circular { radius: 0.03 };
        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        assert!(!packed.has_ovals);
        assert_eq!(packed.segment_lane_offsets, vec![0.0; 2]);
        assert!(packed.directors.iter().all(|value| *value == 0.0));
        assert!(packed
            .vertex_wall_extents()
            .iter()
            .all(|value| *value == 0.03));
    }

    #[test]
    fn lane_count_grows_with_flatness() {
        assert_eq!(oval_lane_count(0.0, 1.0), 1);
        assert_eq!(oval_lane_count(0.1, 1.0), 2);
        assert_eq!(oval_lane_count(0.5, 1.0), 3);
        assert_eq!(oval_lane_count(1.0, 1.0), 5);
        assert_eq!(oval_lane_count(3.0, 1.0), MAXIMUM_OVAL_LANES);
    }

    #[test]
    fn unpacking_turns_device_directors_back() {
        let mut assembly = oval_assembly([0.02, 0.03]);
        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        let directors = packed.directors.clone();
        packed
            .unpack_positions(&packed.positions, Some(&directors), &mut assembly)
            .unwrap();
        // Stored directors point along the first (short) semi-axis again.
        let stored = assembly.geometry.placed.directors[1];
        assert!(stored[0].abs() < 1.0e-7 && stored[2].abs() < 1.0e-7);
        assert!((stored[1].abs() - 1.0).abs() < 1.0e-7);
    }
}
