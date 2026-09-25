mod cell_list;
mod image_force;
mod neighbor_list;
mod pinning;

pub use image_force::ImageForceSettings;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::{
    activate_formation_step, apply_coarsening_candidates, apply_contact_corrections,
    apply_formation_layer_targets, apply_internal_corrections, apply_refinement_candidates,
    apply_rigid_contact_corrections, apply_vertex_targets, assess_reduced_metrics,
    begin_relaxation_batch, clear_active_index_counts, clear_adaptation_epoch,
    clear_reduced_metrics, clear_wall_reactions, compact_active_indices, compact_affine_vertices,
    compact_moving_walls, compact_rigid_fiber_centers, find_internal_corrections,
    finish_adaptation_epoch, initialize_vertex_displacement_targets, mark_coarsening_candidates,
    mark_refinement_candidates, mask_active_list_weights, measure_compaction_metrics,
    measure_curvature_ratio, measure_layer_target_error, measure_vertex_target_error,
    project_fiber_curvature_in_place, reduce_active_segment_penetration,
    reduce_active_vertex_metrics, refine_vertex_paths, total_active_list_weight,
    update_fiber_directors,
};
use super::{
    AdaptiveSegmentationConfig, FiberMotion, PackedAssembly, PackingError, RelaxationConfig,
};
use crate::CellListConfig;
use crate::{CompactionEnergyModel, CompactionKinematics, CompactionMetrics};
use tangle_contact::device::{capture_segment_contacts, find_segment_corrections};

const CELL_SCAN_BLOCK_SIZE: usize = 256;
/// Largest neighbor-list buffer, below WGPU's default 128 MiB binding limit.
const MAXIMUM_NEIGHBOR_LIST_BYTES: usize = 120 << 20;
/// Cell room for proxy pieces of segments stretched past their rest length.
const PROXY_STRETCH_ALLOWANCE: f32 = 1.05;
/// Largest director turn, in radians, applied to one vertex per iteration.
const MAXIMUM_DIRECTOR_TURN: f32 = 0.1;

/// One unique inter-fiber capsule contact captured from the resident GPU world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentContactCandidate {
    /// First packed segment index.
    pub first_segment: u32,
    /// Second packed segment index.
    pub second_segment: u32,
    /// Closest-point coordinate on the first segment.
    pub first_coordinate: f32,
    /// Closest-point coordinate on the second segment.
    pub second_coordinate: f32,
    /// Signed surface separation; negative values indicate penetration.
    pub surface_gap: f32,
    /// Acute angle between the two segment tangents, in radians.
    pub crossing_angle: f32,
}

/// Result of an explicitly requested GPU contact-capture checkpoint.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContactCapture {
    /// Unique inter-fiber segment pairs within the requested surface gap.
    pub candidates: Vec<SegmentContactCandidate>,
    /// Whether the requested candidate output capacity was exceeded.
    pub overflow: bool,
}

/// Maximum residuals for persistent device-resident formation targets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FormationTargetError {
    /// Maximum center-coordinate error among fibers carrying a layer target.
    pub layer: Option<f32>,
    /// Maximum coordinate error among vertices carrying a needle target.
    pub vertex: Option<f32>,
}

impl FormationTargetError {
    /// Returns the largest active target error, or `None` when no target exists.
    pub fn maximum(self) -> Option<f32> {
        match (self.layer, self.vertex) {
            (Some(layer), Some(vertex)) => Some(layer.max(vertex)),
            (Some(layer), None) => Some(layer),
            (None, Some(vertex)) => Some(vertex),
            (None, None) => None,
        }
    }
}

/// Scalar status downloaded at one GRASS-controlled GPU batch boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchStatus {
    /// Corrections applied over the lifetime of this device world.
    pub total_iterations: usize,
    /// Corrections applied by the most recently requested batch.
    pub batch_iterations: usize,
    /// Maximum capsule penetration in the final contact pass.
    pub max_penetration: f32,
    /// Largest contact displacement in the final active iteration.
    pub max_displacement: f32,
    /// Largest current curvature divided by its admissible maximum.
    pub max_curvature_ratio: f32,
    /// Whether the current constraints are converged.
    pub converged: bool,
    /// Reserved compatibility flag; compact cell lists do not overflow.
    pub cell_list_overflow: bool,
    /// Total parent segments split over the lifetime of this device world.
    pub segment_splits: usize,
    /// Total refinement epochs that activated at least one midpoint.
    pub refinement_passes: usize,
    /// Total sibling-pair merges over the lifetime of this device world.
    pub segment_merges: usize,
    /// Total adaptation epochs that removed at least one midpoint.
    pub coarsening_passes: usize,
}

/// Persistent CubeCL buffers for one packed fiber assembly.
///
/// The geometry and topology stay device-resident across calls to
/// [`run_batch`](Self::run_batch). GRASS plugins can therefore choose and
/// sequence macro operations without re-uploading the fiber world.
pub struct DeviceFiberWorld<R: Runtime> {
    client: ComputeClient<R>,
    packed: PackedAssembly,
    positions: Handle,
    intrinsic_positions: Handle,
    segment_vertices: Handle,
    segment_fibers: Handle,
    segment_radii: Handle,
    segment_lane_offsets: Handle,
    directors: Handle,
    director_corrections: Handle,
    vertex_wall_extents: Handle,
    segment_rest_lengths: Handle,
    segment_active: Handle,
    segment_children: Handle,
    segment_birth_epochs: Handle,
    segment_contact_epochs: Handle,
    segment_quiet_epochs: Handle,
    segment_refinement_levels: Handle,
    fiber_segment_spans: Handle,
    fiber_vertex_spans: Handle,
    fiber_formation_layers: Handle,
    fiber_formation_steps: Handle,
    vertex_segments: Handle,
    vertex_max_curvature: Handle,
    vertex_active: Handle,
    vertex_pinned: Handle,
    vertex_refinement_levels: Handle,
    cell_lower: Handle,
    cell_upper: Handle,
    cell_periodic: Handle,
    control: Handle,
    metrics: Handle,
    cell_counts: Handle,
    cell_offsets: Handle,
    cell_cursors: Handle,
    cell_segments: Handle,
    // Scatter scratch: run-order slots and each slot's cell, ranked into
    // `cell_segments` in segment order.
    scattered_cell_segments: Handle,
    scattered_cell_proxies: Handle,
    cell_slot_cells: Handle,
    cell_overflow: Handle,
    cell_scan_block_size: usize,
    cell_scan_block_sums: Vec<Handle>,
    cell_scan_block_offsets: Vec<Handle>,
    // Pending-rebuild flag and completed-rebuild count.
    neighbor_state: Handle,
    neighbor_counts: Handle,
    neighbor_segments: Handle,
    neighbor_reference_positions: Handle,
    // Cell-sorted copies of each slot's segment geometry (eight floats) and
    // topology (vertex ids, fiber, proxy piece and proxy count), refreshed at
    // every list build.
    slot_geometry: Handle,
    slot_topology: Handle,
    // Proxy pieces per packed segment, the piece each cell-list slot holds,
    // and the slot capacity (the sum of all segments' pieces).
    segment_proxies: Handle,
    cell_proxies: Handle,
    proxy_capacity: usize,
    neighbor_skin: f32,
    // Configured slots per list block, and the block size in use: smaller
    // when the active segments' blocks would not fit in `neighbor_slots`.
    neighbor_capacity: u32,
    neighbor_block: u32,
    neighbor_slots: usize,
    // Static blocks per packed segment, the active ones' weights, their
    // exclusive-scan offsets (in blocks), and the scan's total and scratch.
    list_weights: Handle,
    active_list_weights: Handle,
    list_offsets: Handle,
    list_weight_total: Handle,
    list_scan_block_sums: Vec<Handle>,
    list_scan_block_offsets: Vec<Handle>,
    // A control word that is always 1, for scans outside the relaxation loop.
    always_run: Handle,
    corrections: Handle,
    segment_max: Handle,
    curvature_ratio: Handle,
    internal_corrections: Handle,
    vertex_step: Handle,
    wall_reactions: Handle,
    refinement_count: Handle,
    coarsening_candidates: Handle,
    compaction_metrics: Handle,
    formation_target_error: Handle,
    active_segment_indices: Handle,
    active_vertex_indices: Handle,
    active_index_counts: Handle,
    active_segment_count: usize,
    active_vertex_count: usize,
    reduced_metrics: Handle,
    active_layer_targets: Option<Handle>,
    active_layer_target_count: usize,
    active_layer_axis: usize,
    active_layer_first: u32,
    active_layer_last: u32,
    active_layer_stiffness: f32,
    active_layer_max_translation: f32,
    active_vertex_targets: Option<ActiveVertexTargets>,
    image_force: Option<image_force::ImageForce>,
    total_iterations: usize,
    cell_size: f32,
    segment_cell_size: f32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
    cell_count: usize,
    cell_lower_host: [f32; 3],
    cell_upper_host: [f32; 3],
}

struct ActiveVertexTargets {
    indices: Handle,
    coordinates: Handle,
    command_coordinates: Handle,
    count: usize,
    axis: usize,
    stiffness: f32,
    max_translation: f32,
}

/// Persistent manufacturing-layer command stored in a restart checkpoint.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerTargetCheckpoint {
    /// Target coordinate for every generated layer.
    pub coordinates: Vec<f32>,
    /// Cartesian axis normal to the layers.
    pub axis: usize,
    /// First manufacturing layer affected by this target.
    pub first_layer: u32,
    /// Last manufacturing layer affected by this target.
    pub last_layer: u32,
    /// Fraction of target error corrected per solver iteration.
    pub stiffness: f32,
    /// Maximum per-iteration translation.
    pub max_translation: f32,
}

/// Persistent per-vertex command stored in a restart checkpoint.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VertexTargetCheckpoint {
    /// Packed vertex indices carrying targets.
    pub indices: Vec<u32>,
    /// Absolute target coordinate corresponding to every index.
    pub coordinates: Vec<f32>,
    /// Current prescribed coordinate reached by the moving actuator.
    pub command_coordinates: Vec<f32>,
    /// Cartesian target axis.
    pub axis: usize,
    /// Fraction of target error corrected per solver iteration.
    pub stiffness: f32,
    /// Maximum per-iteration translation.
    pub max_translation: f32,
}

/// Complete device-resident state required to restart a relaxation run.
///
/// Scratch broad-phase and correction buffers are intentionally omitted: they
/// are rebuilt by the first batch after resume. Persistent geometry, adaptive
/// topology masks, manufacturing targets, wall reactions, and counters are
/// retained exactly.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceWorldCheckpoint {
    /// Packed topology with current positions, masks, cell, and refinement links.
    pub packed: PackedAssembly,
    /// Lifetime device iteration count.
    pub total_iterations: usize,
    /// Cumulative/current split and merge counts plus productive pass counts.
    ///
    /// Entries are cumulative splits, current splits, split passes, cumulative
    /// merges, current merges, and merge passes.
    pub refinement_count: [u32; 6],
    /// Last per-vertex wall reactions used by compaction metrics.
    pub wall_reactions: Vec<f32>,
    /// Active layer target, when a formation operation is holding one.
    pub layer_target: Option<LayerTargetCheckpoint>,
    /// Active needle or prescribed vertex targets.
    pub vertex_target: Option<VertexTargetCheckpoint>,
}

impl<R: Runtime> DeviceFiberWorld<R> {
    /// Uploads a packed assembly and allocates all reusable solver buffers.
    pub fn upload(
        device: &R::Device,
        mut packed: PackedAssembly,
        cell_list: CellListConfig,
        max_step: f32,
    ) -> Self {
        assert!(packed.segment_count() > 0);
        assert!(packed.fiber_count() > 0);
        assert!(cell_list.cell_size_scale >= 1.0);
        assert!(cell_list.neighbor_skin_scale.is_finite() && cell_list.neighbor_skin_scale >= 0.0);
        assert!(cell_list.neighbor_capacity > 0);
        assert!(max_step > 0.0);
        packed.ensure_lane_buffers();

        let mut fiber_has_active_segment = vec![false; packed.fiber_count()];
        for (segment, active) in packed.segment_active.iter().copied().enumerate() {
            if active != 0 {
                fiber_has_active_segment[packed.segment_fibers[segment] as usize] = true;
            }
        }
        // Inactive dyadic parents must not keep the broad phase permanently
        // coarse after refinement. Fibers not yet inserted still contribute
        // all reserved segments so later staged activation remains safe.
        let participates_in_cell_scale = |segment: usize| {
            packed.segment_active[segment] != 0
                || !fiber_has_active_segment[packed.segment_fibers[segment] as usize]
        };
        let maximum_rest_length = packed
            .segment_rest_lengths
            .iter()
            .copied()
            .enumerate()
            .filter(|(segment, _)| participates_in_cell_scale(*segment))
            .map(|(_, length)| length)
            .fold(0.0_f32, f32::max);
        let initial_length = |segment: usize| {
            let first = packed.segment_vertices[2 * segment] as usize;
            let second = packed.segment_vertices[2 * segment + 1] as usize;
            let dx = packed.positions[3 * second] - packed.positions[3 * first];
            let dy = packed.positions[3 * second + 1] - packed.positions[3 * first + 1];
            let dz = packed.positions[3 * second + 2] - packed.positions[3 * first + 2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let maximum_initial_length = (0..packed.segment_count())
            .filter(|segment| participates_in_cell_scale(*segment))
            .map(initial_length)
            .fold(0.0_f32, f32::max);
        let maximum_length = maximum_rest_length.max(maximum_initial_length);
        let maximum_radius = packed.segment_radii.iter().copied().fold(0.0_f32, f32::max);
        let neighbor_skin = cell_list.neighbor_skin_scale * maximum_radius;
        // Neighbor-list room grows with segment length, counted in blocks of
        // `neighbor_capacity` slots. A uniform layout's longest segment is a
        // leaf, so every segment gets one block as before; an adaptive layout
        // gives an unrefined parent one block per leaf length (or twice the
        // interaction margin) of its length, since its neighbor count grows
        // with it.
        let segment_length = |segment: usize| {
            let first = packed.segment_vertices[2 * segment] as usize;
            let second = packed.segment_vertices[2 * segment + 1] as usize;
            let dx = packed.positions[3 * second] - packed.positions[3 * first];
            let dy = packed.positions[3 * second + 1] - packed.positions[3 * first + 1];
            let dz = packed.positions[3 * second + 2] - packed.positions[3 * first + 2];
            packed.segment_rest_lengths[segment].max((dx * dx + dy * dy + dz * dz).sqrt())
        };
        let interaction_margin = 2.0 * maximum_radius + (2.0 * max_step).max(neighbor_skin);
        let maximum_leaf_length = (0..packed.segment_count())
            .filter(|segment| packed.segment_children[2 * segment] == u32::MAX)
            .map(segment_length)
            .fold(0.0_f32, f32::max);
        let list_block_length =
            maximum_length.min(maximum_leaf_length.max(2.0 * interaction_margin));
        let list_weights: Vec<u32> = (0..packed.segment_count())
            .map(|segment| ((segment_length(segment) / list_block_length).ceil() as u32).max(1))
            .collect();
        // Segments longer than one list block are also binned as that many
        // equal proxy pieces, so the grid follows the short segments: a
        // uniform layout bins every segment whole as before, while an adaptive
        // layout bins unrefined parents as leaf-length pieces. Neighbor lists
        // are built from this grid, so a cell must also span the skin around
        // the widest capsule pair.
        let proxy_length = list_block_length;
        let segment_proxies = list_weights.clone();
        let proxy_capacity = segment_proxies
            .iter()
            .map(|&proxies| proxies as usize)
            .sum();
        // Pieces of a segment stretched past its rest and initial length are
        // longer than `proxy_length`; in a dyadic tree every root of the
        // longest length has zero slack, so pieces get 5 % of room to stretch.
        // A uniform layout keeps its previous grid.
        let piece_length = if proxy_length < maximum_length {
            PROXY_STRETCH_ALLOWANCE * proxy_length
        } else {
            proxy_length
        };
        let cell_size = (piece_length + interaction_margin) * cell_list.cell_size_scale;
        // Contact capture bins whole segments.
        let segment_cell_size = (maximum_length + interaction_margin) * cell_list.cell_size_scale;
        let cell_extent = [
            packed.cell_upper[0] - packed.cell_lower[0],
            packed.cell_upper[1] - packed.cell_lower[1],
            packed.cell_upper[2] - packed.cell_lower[2],
        ];
        // Floor keeps each exact per-axis grid width at least as large as the
        // conservative broad-phase size. Device kernels derive widths from
        // the physical extent so a periodic hash repeats after one box vector.
        let cells_x = (cell_extent[0] / cell_size).floor().max(1.0) as u32;
        let cells_y = (cell_extent[1] / cell_size).floor().max(1.0) as u32;
        let cells_z = (cell_extent[2] / cell_size).floor().max(1.0) as u32;
        let cell_count = cells_x as usize * cells_y as usize * cells_z as usize;
        let client = R::client(device);
        let positions = client.create_from_slice(f32::as_bytes(&packed.positions));
        let intrinsic_positions =
            client.create_from_slice(f32::as_bytes(&packed.intrinsic_positions));
        let segment_vertices = client.create_from_slice(u32::as_bytes(&packed.segment_vertices));
        let segment_fibers = client.create_from_slice(u32::as_bytes(&packed.segment_fibers));
        let segment_radii = client.create_from_slice(f32::as_bytes(&packed.segment_radii));
        let segment_lane_offsets =
            client.create_from_slice(f32::as_bytes(&packed.segment_lane_offsets));
        let directors = client.create_from_slice(f32::as_bytes(&packed.directors));
        let director_corrections =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; 2 * packed.segment_count()]));
        let vertex_wall_extents =
            client.create_from_slice(f32::as_bytes(&packed.vertex_wall_extents()));
        let segment_rest_lengths =
            client.create_from_slice(f32::as_bytes(&packed.segment_rest_lengths));
        let segment_active = client.create_from_slice(u32::as_bytes(&packed.segment_active));
        let segment_children = client.create_from_slice(u32::as_bytes(&packed.segment_children));
        let segment_birth_epochs =
            client.create_from_slice(u32::as_bytes(&packed.segment_birth_epochs));
        let segment_contact_epochs =
            client.create_from_slice(u32::as_bytes(&packed.segment_contact_epochs));
        let segment_quiet_epochs =
            client.create_from_slice(u32::as_bytes(&packed.segment_quiet_epochs));
        let segment_refinement_levels =
            client.create_from_slice(u32::as_bytes(&packed.segment_refinement_levels));
        let fiber_segment_spans =
            client.create_from_slice(u32::as_bytes(&packed.fiber_segment_spans));
        let fiber_vertex_spans =
            client.create_from_slice(u32::as_bytes(&packed.fiber_vertex_spans));
        let fiber_formation_layers =
            client.create_from_slice(u32::as_bytes(&packed.fiber_formation_layers));
        let fiber_formation_steps =
            client.create_from_slice(u32::as_bytes(&packed.fiber_formation_steps));
        let vertex_segments = client.create_from_slice(u32::as_bytes(&packed.vertex_segments));
        let vertex_max_curvature =
            client.create_from_slice(f32::as_bytes(&packed.vertex_max_curvature));
        let vertex_active = client.create_from_slice(u32::as_bytes(&packed.vertex_active));
        let vertex_pinned = client.create_from_slice(u32::as_bytes(&packed.vertex_pinned));
        let vertex_refinement_levels =
            client.create_from_slice(u32::as_bytes(&packed.vertex_refinement_levels));
        let cell_lower = client.create_from_slice(f32::as_bytes(&packed.cell_lower));
        let cell_upper = client.create_from_slice(f32::as_bytes(&packed.cell_upper));
        let cell_periodic = client.create_from_slice(u32::as_bytes(&packed.cell_periodic));
        // active, lifetime iterations, batch iterations, terminal failure
        let control = client.create_from_slice(u32::as_bytes(&[0_u32, 0, 0, 0]));
        let metrics = client.create_from_slice(f32::as_bytes(&[f32::INFINITY, 0.0, f32::INFINITY]));
        let cell_counts = client.create_from_slice(u32::as_bytes(&vec![0_u32; cell_count]));
        let cell_offsets = client.create_from_slice(u32::as_bytes(&vec![0_u32; cell_count]));
        let cell_cursors = client.create_from_slice(u32::as_bytes(&vec![0_u32; cell_count]));
        let cell_segments = client.empty(proxy_capacity * core::mem::size_of::<u32>());
        let cell_proxies = client.empty(proxy_capacity * core::mem::size_of::<u32>());
        let scattered_cell_segments = client.empty(proxy_capacity * core::mem::size_of::<u32>());
        let scattered_cell_proxies = client.empty(proxy_capacity * core::mem::size_of::<u32>());
        let cell_slot_cells = client.empty(proxy_capacity * core::mem::size_of::<u32>());
        let segment_proxies = client.create_from_slice(u32::as_bytes(&segment_proxies));
        let cell_overflow = client.create_from_slice(u32::as_bytes(&[0_u32]));
        let maximum_scan_block_size = CELL_SCAN_BLOCK_SIZE
            .min(client.properties().hardware.max_cube_dim.0 as usize)
            .min(client.properties().hardware.max_units_per_cube as usize)
            .max(1);
        let cell_scan_block_size = 1usize << maximum_scan_block_size.ilog2();
        let mut cell_scan_block_sums = Vec::new();
        let mut cell_scan_block_offsets = Vec::new();
        let mut scan_length = cell_count;
        loop {
            let blocks = scan_length.div_ceil(cell_scan_block_size);
            cell_scan_block_sums.push(client.empty(blocks * core::mem::size_of::<u32>()));
            cell_scan_block_offsets.push(client.empty(blocks * core::mem::size_of::<u32>()));
            if blocks == 1 {
                break;
            }
            scan_length = blocks;
        }
        // Room for every segment's full lists at once, kept within a single
        // binding on every backend. When the active segments need more, the
        // block size shrinks (see `update_neighbor_list_layout`); an
        // overflowing segment only falls back to scanning its cells.
        let neighbor_capacity = cell_list.neighbor_capacity;
        // A parent's weight never exceeds its two children's, so the leaves'
        // total bounds every active set of an adaptive tree.
        let total_list_weight: usize = (0..packed.segment_count())
            .filter(|&segment| packed.segment_children[2 * segment] == u32::MAX)
            .map(|segment| list_weights[segment] as usize)
            .sum();
        let neighbor_slots = (neighbor_capacity as usize * total_list_weight)
            .min(MAXIMUM_NEIGHBOR_LIST_BYTES / core::mem::size_of::<u32>())
            .max(1);
        // Lists start stale so the first contact pass builds them.
        let neighbor_state = client.create_from_slice(u32::as_bytes(&[1_u32, 0]));
        let neighbor_counts =
            client.create_from_slice(u32::as_bytes(&vec![0_u32; packed.segment_count()]));
        let neighbor_segments = client.empty(neighbor_slots * core::mem::size_of::<u32>());
        let list_weights = client.create_from_slice(u32::as_bytes(&list_weights));
        let active_list_weights =
            client.empty(packed.segment_count() * core::mem::size_of::<u32>());
        let list_offsets = client.empty(packed.segment_count() * core::mem::size_of::<u32>());
        let list_weight_total = client.create_from_slice(u32::as_bytes(&[0_u32]));
        let mut list_scan_block_sums = Vec::new();
        let mut list_scan_block_offsets = Vec::new();
        let mut scan_length = packed.segment_count();
        loop {
            let blocks = scan_length.div_ceil(cell_scan_block_size);
            list_scan_block_sums.push(client.empty(blocks * core::mem::size_of::<u32>()));
            list_scan_block_offsets.push(client.empty(blocks * core::mem::size_of::<u32>()));
            if blocks == 1 {
                break;
            }
            scan_length = blocks;
        }
        let neighbor_reference_positions =
            client.create_from_slice(f32::as_bytes(&packed.positions));
        let slot_geometry = client.empty(8 * proxy_capacity * core::mem::size_of::<f32>());
        let slot_topology = client.empty(5 * proxy_capacity * core::mem::size_of::<u32>());
        let corrections =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; 6 * packed.segment_count()]));
        let segment_max =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; packed.segment_count()]));
        let curvature_ratio =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; packed.vertex_count()]));
        let internal_corrections =
            client.empty(3 * packed.vertex_count() * core::mem::size_of::<f32>());
        let vertex_step =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; packed.vertex_count()]));
        let wall_reactions =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; 3 * packed.vertex_count()]));
        // cumulative/current splits, split passes, cumulative/current merges,
        // and merge passes
        let refinement_count = client.create_from_slice(u32::as_bytes(&[0_u32; 6]));
        let coarsening_candidates =
            client.create_from_slice(u32::as_bytes(&vec![0_u32; packed.segment_count()]));
        let compaction_metrics = client.create_from_slice(f32::as_bytes(&[0.0_f32; 13]));
        let formation_target_error = client.create_from_slice(f32::as_bytes(&[0.0_f32; 2]));
        let initial_active_segments = packed
            .segment_active
            .iter()
            .enumerate()
            .filter_map(|(index, active)| (*active != 0).then_some(index as u32))
            .collect::<Vec<_>>();
        let initial_active_vertices = packed
            .vertex_active
            .iter()
            .enumerate()
            .filter_map(|(index, active)| (*active != 0).then_some(index as u32))
            .collect::<Vec<_>>();
        let mut active_segment_storage = vec![u32::MAX; packed.segment_count()];
        active_segment_storage[..initial_active_segments.len()]
            .copy_from_slice(&initial_active_segments);
        let mut active_vertex_storage = vec![u32::MAX; packed.vertex_count()];
        active_vertex_storage[..initial_active_vertices.len()]
            .copy_from_slice(&initial_active_vertices);
        let active_segment_indices =
            client.create_from_slice(u32::as_bytes(&active_segment_storage));
        let active_vertex_indices = client.create_from_slice(u32::as_bytes(&active_vertex_storage));
        let active_index_counts = client.create_from_slice(u32::as_bytes(&[
            initial_active_segments.len() as u32,
            initial_active_vertices.len() as u32,
        ]));
        let reduced_metrics = client.create_from_slice(u32::as_bytes(&[0_u32; 3]));
        let cell_lower_host = packed.cell_lower;
        let cell_upper_host = packed.cell_upper;

        let always_run = client.create_from_slice(u32::as_bytes(&[1_u32]));
        let mut world = Self {
            always_run,
            client,
            packed,
            positions,
            intrinsic_positions,
            segment_vertices,
            segment_fibers,
            segment_radii,
            segment_lane_offsets,
            directors,
            director_corrections,
            vertex_wall_extents,
            segment_rest_lengths,
            segment_active,
            segment_children,
            segment_birth_epochs,
            segment_contact_epochs,
            segment_quiet_epochs,
            segment_refinement_levels,
            fiber_segment_spans,
            fiber_vertex_spans,
            fiber_formation_layers,
            fiber_formation_steps,
            vertex_segments,
            vertex_max_curvature,
            vertex_active,
            vertex_pinned,
            vertex_refinement_levels,
            cell_lower,
            cell_upper,
            cell_periodic,
            control,
            metrics,
            cell_counts,
            cell_offsets,
            cell_cursors,
            cell_segments,
            scattered_cell_segments,
            scattered_cell_proxies,
            cell_slot_cells,
            cell_overflow,
            cell_scan_block_size,
            cell_scan_block_sums,
            cell_scan_block_offsets,
            neighbor_state,
            neighbor_counts,
            neighbor_segments,
            neighbor_reference_positions,
            slot_geometry,
            slot_topology,
            segment_proxies,
            cell_proxies,
            proxy_capacity,
            neighbor_skin,
            neighbor_capacity,
            neighbor_block: neighbor_capacity,
            neighbor_slots,
            list_weights,
            active_list_weights,
            list_offsets,
            list_weight_total,
            list_scan_block_sums,
            list_scan_block_offsets,
            corrections,
            segment_max,
            curvature_ratio,
            internal_corrections,
            vertex_step,
            wall_reactions,
            refinement_count,
            coarsening_candidates,
            compaction_metrics,
            formation_target_error,
            active_segment_indices,
            active_vertex_indices,
            active_index_counts,
            active_segment_count: initial_active_segments.len(),
            active_vertex_count: initial_active_vertices.len(),
            reduced_metrics,
            active_layer_targets: None,
            active_layer_target_count: 0,
            active_layer_axis: 0,
            active_layer_first: 0,
            active_layer_last: u32::MAX,
            active_layer_stiffness: 1.0,
            active_layer_max_translation: 1.0,
            active_vertex_targets: None,
            image_force: None,
            total_iterations: 0,
            cell_size,
            segment_cell_size,
            cells_x,
            cells_y,
            cells_z,
            cell_count,
            cell_lower_host,
            cell_upper_host,
        };
        world.update_neighbor_list_layout();
        world
    }

    /// Recreates a resident world from a previously downloaded checkpoint.
    pub fn restore(
        device: &R::Device,
        checkpoint: DeviceWorldCheckpoint,
        cell_list: CellListConfig,
        max_step: f32,
    ) -> Self {
        let total_iterations = checkpoint.total_iterations;
        let refinement_count = checkpoint.refinement_count;
        let wall_reactions = checkpoint.wall_reactions;
        let layer_target = checkpoint.layer_target;
        let vertex_target = checkpoint.vertex_target;
        let mut world = Self::upload(device, checkpoint.packed, cell_list, max_step);
        assert_eq!(wall_reactions.len(), 3 * world.packed.vertex_count());
        world.control = world.client.create_from_slice(u32::as_bytes(&[
            0,
            u32::try_from(total_iterations).expect("checkpoint iteration count exceeds u32"),
            0,
            0,
        ]));
        world.refinement_count = world
            .client
            .create_from_slice(u32::as_bytes(&refinement_count));
        world.wall_reactions = world
            .client
            .create_from_slice(f32::as_bytes(&wall_reactions));
        world.total_iterations = total_iterations;
        if let Some(target) = layer_target {
            world.active_layer_target_count = target.coordinates.len();
            world.active_layer_targets = Some(
                world
                    .client
                    .create_from_slice(f32::as_bytes(&target.coordinates)),
            );
            world.active_layer_axis = target.axis;
            world.active_layer_first = target.first_layer;
            world.active_layer_last = target.last_layer;
            world.active_layer_stiffness = target.stiffness;
            world.active_layer_max_translation = target.max_translation;
        }
        if let Some(target) = vertex_target {
            assert_eq!(target.indices.len(), target.coordinates.len());
            assert_eq!(target.indices.len(), target.command_coordinates.len());
            world.active_vertex_targets = Some(ActiveVertexTargets {
                indices: world
                    .client
                    .create_from_slice(u32::as_bytes(&target.indices)),
                coordinates: world
                    .client
                    .create_from_slice(f32::as_bytes(&target.coordinates)),
                command_coordinates: world
                    .client
                    .create_from_slice(f32::as_bytes(&target.command_coordinates)),
                count: target.indices.len(),
                axis: target.axis,
                stiffness: target.stiffness,
                max_translation: target.max_translation,
            });
        }
        world
    }

    /// Downloads the persistent subset of this world needed for restart.
    pub fn checkpoint(&self) -> DeviceWorldCheckpoint {
        let read_f32 = |handle: &Handle| {
            let bytes = self
                .client
                .read_one(handle.clone())
                .expect("CubeCL checkpoint f32 readback failed");
            f32::from_bytes(&bytes).to_vec()
        };
        let read_u32 = |handle: &Handle| {
            let bytes = self
                .client
                .read_one(handle.clone())
                .expect("CubeCL checkpoint u32 readback failed");
            u32::from_bytes(&bytes).to_vec()
        };

        let mut packed = self.packed.clone();
        packed.positions = read_f32(&self.positions);
        if packed.has_ovals {
            packed.directors = read_f32(&self.directors);
        }
        packed.segment_active = read_u32(&self.segment_active);
        packed.segment_birth_epochs = read_u32(&self.segment_birth_epochs);
        packed.segment_contact_epochs = read_u32(&self.segment_contact_epochs);
        packed.segment_quiet_epochs = read_u32(&self.segment_quiet_epochs);
        packed.vertex_active = read_u32(&self.vertex_active);
        packed.vertex_segments = read_u32(&self.vertex_segments);
        packed.cell_lower = self.cell_lower_host;
        packed.cell_upper = self.cell_upper_host;
        let refinement = read_u32(&self.refinement_count);
        let layer_target =
            self.active_layer_targets
                .as_ref()
                .map(|targets| LayerTargetCheckpoint {
                    coordinates: read_f32(targets),
                    axis: self.active_layer_axis,
                    first_layer: self.active_layer_first,
                    last_layer: self.active_layer_last,
                    stiffness: self.active_layer_stiffness,
                    max_translation: self.active_layer_max_translation,
                });
        let vertex_target =
            self.active_vertex_targets
                .as_ref()
                .map(|targets| VertexTargetCheckpoint {
                    indices: read_u32(&targets.indices),
                    coordinates: read_f32(&targets.coordinates),
                    command_coordinates: read_f32(&targets.command_coordinates),
                    axis: targets.axis,
                    stiffness: targets.stiffness,
                    max_translation: targets.max_translation,
                });

        DeviceWorldCheckpoint {
            packed,
            total_iterations: self.total_iterations,
            refinement_count: [
                refinement[0],
                refinement[1],
                refinement[2],
                refinement[3],
                refinement[4],
                refinement[5],
            ],
            wall_reactions: read_f32(&self.wall_reactions),
            layer_target,
            vertex_target,
        }
    }

    fn rebuild_active_indices(&mut self) {
        let cube_dim = CubeDim::new_1d(64);
        let capacity = self.packed.segment_count().max(self.packed.vertex_count());
        unsafe {
            clear_active_index_counts::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
            );
            compact_active_indices::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(capacity.div_ceil(64) as u32, 1, 1),
                cube_dim.clone(),
                BufferArg::from_raw_parts(
                    self.segment_active.clone(),
                    self.packed.segment_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_segment_indices.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.active_vertex_indices.clone(),
                    self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
            );
        }
        let bytes = self
            .client
            .read_one(self.active_index_counts.clone())
            .expect("CubeCL active-index count readback failed");
        let counts = u32::from_bytes(&bytes);
        self.active_segment_count = counts[0] as usize;
        self.active_vertex_count = counts[1] as usize;
        self.update_neighbor_list_layout();
    }

    /// Places each active segment's neighbor list after the lists of all
    /// lower-indexed active segments and fits the block size to the buffer.
    ///
    /// The offsets follow segment order, not the order of the atomically
    /// compacted active list, so an unchanged active set keeps its layout.
    /// Every caller that changes the active set also requests a list rebuild.
    fn update_neighbor_list_layout(&mut self) {
        let segments = self.packed.segment_count();
        let segment_cubes = CubeCount::Static(segments.div_ceil(64) as u32, 1, 1);
        unsafe {
            mask_active_list_weights::launch_unchecked::<R>(
                &self.client,
                segment_cubes,
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.segment_active.clone(), segments),
                BufferArg::from_raw_parts(self.list_weights.clone(), segments),
                BufferArg::from_raw_parts(self.active_list_weights.clone(), segments),
            );
        }
        self.scan_u32(
            self.active_list_weights.clone(),
            self.list_offsets.clone(),
            segments,
            0,
            self.always_run.clone(),
            1,
            &self.list_scan_block_sums,
            &self.list_scan_block_offsets,
        );
        unsafe {
            total_active_list_weight::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.active_list_weights.clone(), segments),
                BufferArg::from_raw_parts(self.list_offsets.clone(), segments),
                BufferArg::from_raw_parts(self.list_weight_total.clone(), 1),
            );
        }
        let bytes = self
            .client
            .read_one(self.list_weight_total.clone())
            .expect("CubeCL neighbor-list weight readback failed");
        let total = (u32::from_bytes(&bytes)[0] as usize).max(1);
        // Zero only past ~31 M active blocks: then every segment takes the
        // exact overflow scan instead of a list.
        self.neighbor_block = self
            .neighbor_capacity
            .min((self.neighbor_slots / total).min(u32::MAX as usize) as u32);
    }

    /// Projects every active fiber back inside its curvature limit with
    /// forward/backward Gauss–Seidel sweeps, one thread per fiber.
    ///
    /// Each projection sees the corrections already applied to its neighbors,
    /// so adjacent bend constraints reinforce each other instead of being
    /// averaged as in a Jacobi pass, and one launch covers every sweep.
    fn launch_curvature_cleanup(&self, config: &RelaxationConfig) {
        unsafe {
            project_fiber_curvature_in_place::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.wall_reactions.clone(),
                    3 * self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(
                    self.fiber_vertex_spans.clone(),
                    self.packed.fiber_vertex_spans.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_wall_extents.clone(),
                    3 * self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_max_curvature.clone(),
                    self.packed.vertex_max_curvature.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_pinned.clone(),
                    self.packed.vertex_pinned.len(),
                ),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(self.control.clone(), 4),
                config.curvature_cleanup_sweeps as u32,
                config.curvature_limit_stiffness,
                config.curvature_limit_safety_margin,
            );
        }
    }

    /// Turns oval directors by the contact turns found this iteration, keeps
    /// them perpendicular to the moved centerlines, smooths their twist and
    /// refreshes the wall extents that depend on them.
    fn launch_director_update(&self, config: &RelaxationConfig) {
        unsafe {
            update_fiber_directors::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(self.directors.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.director_corrections.clone(),
                    2 * self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_wall_extents.clone(),
                    3 * self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(
                    self.fiber_segment_spans.clone(),
                    self.packed.fiber_segment_spans.len(),
                ),
                BufferArg::from_raw_parts(
                    self.fiber_vertex_spans.clone(),
                    self.packed.fiber_vertex_spans.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_radii.clone(),
                    self.packed.segment_radii.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_lane_offsets.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(self.control.clone(), 4),
                config.twist_stiffness,
                MAXIMUM_DIRECTOR_TURN,
            );
        }
    }

    /// Executes up to `iterations` relaxation corrections and downloads only
    /// the small batch-boundary status buffers.
    pub fn run_batch(&mut self, config: &RelaxationConfig, iterations: usize) -> BatchStatus {
        assert!(iterations > 0);
        let cube_dim = CubeDim::new_1d(64);
        let vertex_cubes = CubeCount::Static(self.packed.vertex_count().div_ceil(64) as u32, 1, 1);
        unsafe {
            begin_relaxation_batch::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.control.clone(), 4),
            );
        }

        // Host-side operations between batches (layer targets, needling
        // commands) may have moved vertices without passing through the
        // per-iteration displacement check.
        self.flag_neighbor_list_displacement();
        let starting_iteration = self.total_iterations;
        for step in 0..=iterations {
            self.rebuild_neighbor_lists_if_requested();
            unsafe {
                find_segment_corrections::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(self.active_segment_count.div_ceil(64) as u32, 1, 1),
                    cube_dim.clone(),
                    BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                    BufferArg::from_raw_parts(
                        self.segment_vertices.clone(),
                        self.packed.segment_vertices.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_fibers.clone(),
                        self.packed.segment_fibers.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_radii.clone(),
                        self.packed.segment_radii.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_lane_offsets.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.directors.clone(), self.packed.positions.len()),
                    BufferArg::from_raw_parts(
                        self.active_segment_indices.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                    BufferArg::from_raw_parts(self.cell_counts.clone(), self.cell_count),
                    BufferArg::from_raw_parts(self.cell_offsets.clone(), self.cell_count),
                    BufferArg::from_raw_parts(self.cell_segments.clone(), self.proxy_capacity),
                    BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                    BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                    BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                    BufferArg::from_raw_parts(self.control.clone(), 4),
                    BufferArg::from_raw_parts(
                        self.neighbor_counts.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.neighbor_segments.clone(), self.neighbor_slots),
                    BufferArg::from_raw_parts(
                        self.neighbor_reference_positions.clone(),
                        self.packed.positions.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_proxies.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.cell_proxies.clone(), self.proxy_capacity),
                    BufferArg::from_raw_parts(
                        self.corrections.clone(),
                        6 * self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(
                        self.director_corrections.clone(),
                        2 * self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_max.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(
                        self.list_offsets.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(
                        self.list_weights.clone(),
                        self.packed.segment_count(),
                    ),
                    self.neighbor_block,
                    config.correction_fraction,
                    config.contact_aggregation as u32,
                    self.cells_x,
                    self.cells_y,
                    self.cells_z,
                );
                measure_curvature_ratio::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                    cube_dim.clone(),
                    BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                    BufferArg::from_raw_parts(
                        self.vertex_max_curvature.clone(),
                        self.packed.vertex_max_curvature.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.active_vertex_indices.clone(),
                        self.packed.vertex_count(),
                    ),
                    BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                    BufferArg::from_raw_parts(
                        self.vertex_active.clone(),
                        self.packed.vertex_active.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.vertex_segments.clone(),
                        self.packed.vertex_segments.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_vertices.clone(),
                        self.packed.segment_vertices.len(),
                    ),
                    BufferArg::from_raw_parts(self.control.clone(), 4),
                    BufferArg::from_raw_parts(
                        self.curvature_ratio.clone(),
                        self.packed.vertex_count(),
                    ),
                );
                clear_reduced_metrics::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts(self.reduced_metrics.clone(), 3),
                );
                reduce_active_segment_penetration::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(self.active_segment_count.div_ceil(64) as u32, 1, 1),
                    cube_dim.clone(),
                    BufferArg::from_raw_parts(
                        self.active_segment_indices.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                    BufferArg::from_raw_parts(
                        self.segment_max.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.reduced_metrics.clone(), 3),
                );
                reduce_active_vertex_metrics::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                    cube_dim.clone(),
                    BufferArg::from_raw_parts(
                        self.active_vertex_indices.clone(),
                        self.packed.vertex_count(),
                    ),
                    BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                    BufferArg::from_raw_parts(
                        self.curvature_ratio.clone(),
                        self.packed.vertex_count(),
                    ),
                    BufferArg::from_raw_parts(self.vertex_step.clone(), self.packed.vertex_count()),
                    BufferArg::from_raw_parts(self.reduced_metrics.clone(), 3),
                );
                assess_reduced_metrics::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts(self.reduced_metrics.clone(), 3),
                    BufferArg::from_raw_parts(self.cell_overflow.clone(), 1),
                    BufferArg::from_raw_parts(self.control.clone(), 4),
                    BufferArg::from_raw_parts(self.metrics.clone(), 3),
                    config.penetration_tolerance,
                    config.curvature_ratio_tolerance,
                    iterations as u32,
                    u32::from(config.force_full_iterations),
                );
                clear_wall_reactions::launch_unchecked::<R>(
                    &self.client,
                    vertex_cubes.clone(),
                    cube_dim.clone(),
                    BufferArg::from_raw_parts(
                        self.wall_reactions.clone(),
                        3 * self.packed.vertex_count(),
                    ),
                    BufferArg::from_raw_parts(self.control.clone(), 4),
                );
                // A rigid-only settling stage must preserve both the fiber
                // shape and the discretization on which its accepted
                // curvature was measured. Refinement is resumed by the next
                // flexible stage.
                if config.motion_model == FiberMotion::Flexible {
                    if let Some(adaptive) = config.adaptive_segmentation {
                        let completed_iteration = starting_iteration + step + 1;
                        if step < iterations
                            && completed_iteration % adaptive.refinement_interval == 0
                        {
                            self.launch_adaptation_epoch(
                                adaptive,
                                config.penetration_tolerance,
                                (completed_iteration / adaptive.refinement_interval) as u32,
                            );
                            self.rebuild_active_indices();
                            self.request_neighbor_list_rebuild_after_adaptation();
                        }
                    }
                }
                if config.motion_model == FiberMotion::Flexible {
                    for _ in 0..config.constraint_iterations {
                        find_internal_corrections::launch_unchecked::<R>(
                            &self.client,
                            CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                            cube_dim.clone(),
                            BufferArg::from_raw_parts(
                                self.positions.clone(),
                                self.packed.positions.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.intrinsic_positions.clone(),
                                self.packed.intrinsic_positions.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.segment_vertices.clone(),
                                self.packed.segment_vertices.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.segment_rest_lengths.clone(),
                                self.packed.segment_rest_lengths.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.active_vertex_indices.clone(),
                                self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                            BufferArg::from_raw_parts(
                                self.vertex_active.clone(),
                                self.packed.vertex_active.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_pinned.clone(),
                                self.packed.vertex_pinned.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_segments.clone(),
                                self.packed.vertex_segments.len(),
                            ),
                            BufferArg::from_raw_parts(self.control.clone(), 4),
                            BufferArg::from_raw_parts(
                                self.internal_corrections.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            config.stretch_stiffness,
                            config.bend_stiffness,
                        );
                        apply_internal_corrections::launch_unchecked::<R>(
                            &self.client,
                            CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                            cube_dim.clone(),
                            BufferArg::from_raw_parts(
                                self.positions.clone(),
                                self.packed.positions.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.internal_corrections.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_wall_extents.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(
                                self.active_vertex_indices.clone(),
                                self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                            BufferArg::from_raw_parts(
                                self.vertex_active.clone(),
                                self.packed.vertex_active.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_pinned.clone(),
                                self.packed.vertex_pinned.len(),
                            ),
                            BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                            BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                            BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                            BufferArg::from_raw_parts(
                                self.wall_reactions.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.control.clone(), 4),
                        );
                    }
                }
                match config.motion_model {
                    FiberMotion::Flexible => {
                        apply_contact_corrections::launch_unchecked::<R>(
                            &self.client,
                            CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                            cube_dim.clone(),
                            BufferArg::from_raw_parts(
                                self.positions.clone(),
                                self.packed.positions.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.corrections.clone(),
                                6 * self.packed.segment_count(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_segments.clone(),
                                self.packed.vertex_segments.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_wall_extents.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(
                                self.active_vertex_indices.clone(),
                                self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                            BufferArg::from_raw_parts(
                                self.vertex_active.clone(),
                                self.packed.vertex_active.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_pinned.clone(),
                                self.packed.vertex_pinned.len(),
                            ),
                            BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                            BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                            BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                            BufferArg::from_raw_parts(
                                self.wall_reactions.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.control.clone(), 4),
                            BufferArg::from_raw_parts(
                                self.vertex_step.clone(),
                                self.packed.vertex_count(),
                            ),
                            config.max_step,
                        );
                    }
                    FiberMotion::RigidTranslation => {
                        apply_rigid_contact_corrections::launch_unchecked::<R>(
                            &self.client,
                            CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                            cube_dim.clone(),
                            BufferArg::from_raw_parts(
                                self.positions.clone(),
                                self.packed.positions.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.corrections.clone(),
                                6 * self.packed.segment_count(),
                            ),
                            BufferArg::from_raw_parts(
                                self.fiber_segment_spans.clone(),
                                self.packed.fiber_segment_spans.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.fiber_vertex_spans.clone(),
                                self.packed.fiber_vertex_spans.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_wall_extents.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                            BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                            BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                            BufferArg::from_raw_parts(
                                self.wall_reactions.clone(),
                                3 * self.packed.vertex_count(),
                            ),
                            BufferArg::from_raw_parts(self.control.clone(), 4),
                            BufferArg::from_raw_parts(
                                self.vertex_step.clone(),
                                self.packed.vertex_count(),
                            ),
                            config.max_step,
                        );
                    }
                }
                // Contact is applied after the main stretch/bend sweep. Finish
                // flexible iterations with in-place Gauss–Seidel sweeps along
                // each fiber (race-free: one thread owns each fiber) so
                // neighboring bend constraints reinforce rather than cancel.
                if config.motion_model == FiberMotion::Flexible {
                    self.launch_curvature_cleanup(config);
                    if self.packed.has_ovals {
                        self.launch_director_update(config);
                    }
                }
                // Manufacturing targets are capped kinematic actuators. Apply
                // them after mechanics so each iteration ends at the commanded
                // position; the next contact pass resolves the small induced
                // displacement. The extra pass is measurement-only.
                if step < iterations {
                    if let Some(targets) = &self.active_layer_targets {
                        apply_formation_layer_targets::launch_unchecked::<R>(
                            &self.client,
                            CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                            cube_dim.clone(),
                            BufferArg::from_raw_parts(
                                self.positions.clone(),
                                self.packed.positions.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.fiber_vertex_spans.clone(),
                                self.packed.fiber_vertex_spans.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.fiber_formation_layers.clone(),
                                self.packed.fiber_formation_layers.len(),
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_active.clone(),
                                self.packed.vertex_active.len(),
                            ),
                            BufferArg::from_raw_parts(
                                targets.clone(),
                                self.active_layer_target_count,
                            ),
                            BufferArg::from_raw_parts(self.control.clone(), 4),
                            self.active_layer_axis as u32,
                            self.active_layer_first,
                            self.active_layer_last,
                            self.active_layer_stiffness,
                            self.active_layer_max_translation,
                        );
                    }
                    if let Some(targets) = &self.active_vertex_targets {
                        apply_vertex_targets::launch_unchecked::<R>(
                            &self.client,
                            CubeCount::Static(targets.count.div_ceil(64) as u32, 1, 1),
                            cube_dim.clone(),
                            BufferArg::from_raw_parts(
                                self.positions.clone(),
                                self.packed.positions.len(),
                            ),
                            BufferArg::from_raw_parts(targets.indices.clone(), targets.count),
                            BufferArg::from_raw_parts(targets.coordinates.clone(), targets.count),
                            BufferArg::from_raw_parts(
                                targets.command_coordinates.clone(),
                                targets.count,
                            ),
                            BufferArg::from_raw_parts(
                                self.vertex_active.clone(),
                                self.packed.vertex_active.len(),
                            ),
                            BufferArg::from_raw_parts(self.control.clone(), 4),
                            targets.axis as u32,
                            targets.stiffness,
                            targets.max_translation,
                        );
                    }
                    // CT image attraction: one self-contained find/apply
                    // launch pair, absent unless an image force is set.
                    if let Some(image) = &self.image_force {
                        self.launch_image_force(image, config.max_step);
                    }
                }
            }
            self.flag_neighbor_list_displacement();
        }
        let status = self.read_status(config);
        self.total_iterations = status.total_iterations;
        status
    }

    fn launch_adaptation_epoch(
        &self,
        config: AdaptiveSegmentationConfig,
        penetration_threshold: f32,
        epoch: u32,
    ) {
        if !self.packed.adaptive {
            return;
        }
        unsafe {
            clear_adaptation_epoch::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.refinement_count.clone(), 6),
            );
            // Decide every split before applying any, so no thread sees a
            // sibling's split from the same epoch.
            mark_refinement_candidates::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.segment_count().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(
                    self.segment_radii.clone(),
                    self.packed.segment_radii.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_rest_lengths.clone(),
                    self.packed.segment_rest_lengths.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_children.clone(),
                    self.packed.segment_children.len(),
                ),
                BufferArg::from_raw_parts(self.segment_max.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(
                    self.segment_active.clone(),
                    self.packed.segment_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_birth_epochs.clone(),
                    self.packed.segment_birth_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_contact_epochs.clone(),
                    self.packed.segment_contact_epochs.len(),
                ),
                BufferArg::from_raw_parts(self.control.clone(), 4),
                BufferArg::from_raw_parts(
                    self.coarsening_candidates.clone(),
                    self.packed.segment_count(),
                ),
                epoch,
                penetration_threshold,
                config.contact_length_over_diameter,
                config.minimum_length_over_diameter,
                config.refinement_persistence,
            );
            apply_refinement_candidates::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.segment_count().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_children.clone(),
                    self.packed.segment_children.len(),
                ),
                BufferArg::from_raw_parts(
                    self.coarsening_candidates.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_active.clone(),
                    self.packed.segment_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_birth_epochs.clone(),
                    self.packed.segment_birth_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_contact_epochs.clone(),
                    self.packed.segment_contact_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_quiet_epochs.clone(),
                    self.packed.segment_quiet_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_segments.clone(),
                    self.packed.vertex_segments.len(),
                ),
                BufferArg::from_raw_parts(self.control.clone(), 4),
                BufferArg::from_raw_parts(self.refinement_count.clone(), 6),
                epoch,
            );

            if config.coarsening_persistence > 0
                && self.packed.coarsening_safe
                && self.active_vertex_targets.is_none()
            {
                mark_coarsening_candidates::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(self.packed.segment_count().div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                    BufferArg::from_raw_parts(
                        self.segment_vertices.clone(),
                        self.packed.segment_vertices.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_radii.clone(),
                        self.packed.segment_radii.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_children.clone(),
                        self.packed.segment_children.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_max.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_active.clone(),
                        self.packed.segment_active.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_quiet_epochs.clone(),
                        self.packed.segment_quiet_epochs.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.vertex_active.clone(),
                        self.packed.vertex_active.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.vertex_pinned.clone(),
                        self.packed.vertex_pinned.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.curvature_ratio.clone(),
                        self.packed.vertex_count(),
                    ),
                    BufferArg::from_raw_parts(self.control.clone(), 4),
                    BufferArg::from_raw_parts(
                        self.coarsening_candidates.clone(),
                        self.packed.segment_count(),
                    ),
                    0.0,
                    config.coarsening_persistence,
                    config.coarsening_error_over_diameter,
                    config.coarsening_curvature_ratio,
                );
                apply_coarsening_candidates::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(self.packed.segment_count().div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(
                        self.segment_vertices.clone(),
                        self.packed.segment_vertices.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_children.clone(),
                        self.packed.segment_children.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_active.clone(),
                        self.packed.segment_active.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_birth_epochs.clone(),
                        self.packed.segment_birth_epochs.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_contact_epochs.clone(),
                        self.packed.segment_contact_epochs.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.segment_quiet_epochs.clone(),
                        self.packed.segment_quiet_epochs.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.vertex_active.clone(),
                        self.packed.vertex_active.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.vertex_segments.clone(),
                        self.packed.vertex_segments.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.coarsening_candidates.clone(),
                        self.packed.segment_count(),
                    ),
                    BufferArg::from_raw_parts(self.control.clone(), 4),
                    BufferArg::from_raw_parts(self.refinement_count.clone(), 6),
                    epoch,
                );
            }

            finish_adaptation_epoch::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.refinement_count.clone(), 6),
            );
        }
    }

    /// Installs a persistent manufacturing-layer target and applies its first
    /// correction without downloading geometry.
    pub fn apply_layer_targets(
        &mut self,
        axis: usize,
        targets: &[f32],
        stiffness: f32,
        max_translation: f32,
    ) {
        self.apply_layer_target_range(
            axis,
            targets,
            0,
            targets.len().saturating_sub(1) as u32,
            stiffness,
            max_translation,
        );
    }

    /// Installs a target for one newly deposited layer without retethering the
    /// already relaxed stack.
    pub fn apply_single_layer_target(
        &mut self,
        axis: usize,
        targets: &[f32],
        layer: u32,
        stiffness: f32,
        max_translation: f32,
    ) {
        assert!((layer as usize) < targets.len());
        self.apply_layer_target_range(axis, targets, layer, layer, stiffness, max_translation);
    }

    fn apply_layer_target_range(
        &mut self,
        axis: usize,
        targets: &[f32],
        first_layer: u32,
        last_layer: u32,
        stiffness: f32,
        max_translation: f32,
    ) {
        assert!(axis < 3);
        assert!(!targets.is_empty());
        assert!(stiffness > 0.0 && stiffness <= 1.0);
        assert!(max_translation > 0.0);
        let target_buffer = self.client.create_from_slice(f32::as_bytes(targets));
        let immediate_control = self.client.create_from_slice(u32::as_bytes(&[1_u32]));
        unsafe {
            apply_formation_layer_targets::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.fiber_vertex_spans.clone(),
                    self.packed.fiber_vertex_spans.len(),
                ),
                BufferArg::from_raw_parts(
                    self.fiber_formation_layers.clone(),
                    self.packed.fiber_formation_layers.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(target_buffer.clone(), targets.len()),
                BufferArg::from_raw_parts(immediate_control, 1),
                axis as u32,
                first_layer,
                last_layer,
                stiffness,
                max_translation,
            );
        }
        self.active_layer_targets = Some(target_buffer);
        self.active_layer_target_count = targets.len();
        self.active_layer_axis = axis;
        self.active_layer_first = first_layer;
        self.active_layer_last = last_layer;
        self.active_layer_stiffness = stiffness;
        self.active_layer_max_translation = max_translation;
    }

    /// Releases the persistent manufacturing-layer target.
    pub fn clear_layer_targets(&mut self) {
        self.active_layer_targets = None;
        self.active_layer_target_count = 0;
    }

    /// Installs persistent absolute targets for selected vertices by capturing
    /// their current coordinates and adding `displacement` entirely on-device.
    pub fn apply_vertex_displacement_targets(
        &mut self,
        axis: usize,
        vertex_indices: &[u32],
        displacement: f32,
        stiffness: f32,
        max_translation: f32,
    ) {
        assert!(axis < 3);
        assert!(!vertex_indices.is_empty());
        assert!(vertex_indices
            .iter()
            .all(|index| (*index as usize) < self.packed.vertex_count()));
        assert!(displacement.is_finite() && displacement != 0.0);
        assert!(stiffness > 0.0 && stiffness <= 1.0);
        assert!(max_translation > 0.0);
        let indices = self.client.create_from_slice(u32::as_bytes(vertex_indices));
        let coordinates = self
            .client
            .empty(vertex_indices.len() * std::mem::size_of::<f32>());
        let command_coordinates = self
            .client
            .empty(vertex_indices.len() * std::mem::size_of::<f32>());
        unsafe {
            initialize_vertex_displacement_targets::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(vertex_indices.len().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(indices.clone(), vertex_indices.len()),
                BufferArg::from_raw_parts(coordinates.clone(), vertex_indices.len()),
                BufferArg::from_raw_parts(command_coordinates.clone(), vertex_indices.len()),
                axis as u32,
                displacement,
            );
        }
        self.active_vertex_targets = Some(ActiveVertexTargets {
            indices,
            coordinates,
            command_coordinates,
            count: vertex_indices.len(),
            axis,
            stiffness,
            max_translation,
        });
    }

    /// Releases all persistent per-vertex formation targets.
    pub fn clear_vertex_targets(&mut self) {
        self.active_vertex_targets = None;
    }

    /// Measures persistent formation-target residuals without downloading the
    /// resident fiber geometry.
    pub fn formation_target_error(&self) -> FormationTargetError {
        let layer_slot = self.active_layer_targets.as_ref().map(|targets| {
            unsafe {
                measure_layer_target_error::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                    BufferArg::from_raw_parts(
                        self.fiber_vertex_spans.clone(),
                        self.packed.fiber_vertex_spans.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.fiber_formation_layers.clone(),
                        self.packed.fiber_formation_layers.len(),
                    ),
                    BufferArg::from_raw_parts(
                        self.vertex_active.clone(),
                        self.packed.vertex_active.len(),
                    ),
                    BufferArg::from_raw_parts(targets.clone(), self.active_layer_target_count),
                    BufferArg::from_raw_parts(self.formation_target_error.clone(), 2),
                    self.active_layer_axis as u32,
                    self.active_layer_first,
                    self.active_layer_last,
                );
            }
            0_usize
        });
        let vertex_slot = self.active_vertex_targets.as_ref().map(|targets| {
            unsafe {
                measure_vertex_target_error::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                    BufferArg::from_raw_parts(targets.indices.clone(), targets.count),
                    BufferArg::from_raw_parts(targets.coordinates.clone(), targets.count),
                    BufferArg::from_raw_parts(
                        self.vertex_active.clone(),
                        self.packed.vertex_active.len(),
                    ),
                    BufferArg::from_raw_parts(self.formation_target_error.clone(), 2),
                    targets.axis as u32,
                );
            }
            1_usize
        });
        if layer_slot.is_none() && vertex_slot.is_none() {
            return FormationTargetError::default();
        }
        let bytes = self
            .client
            .read_one(self.formation_target_error.clone())
            .expect("CubeCL formation-target error readback failed");
        let values = f32::from_bytes(&bytes);
        FormationTargetError {
            layer: layer_slot.map(|slot| values[slot]),
            vertex: vertex_slot.map(|slot| values[slot]),
        }
    }

    /// Changes the resident orthorhombic cell and applies the requested
    /// formation kinematics without downloading fiber geometry.
    pub fn compact_cell(
        &mut self,
        new_lower: [f32; 3],
        new_upper: [f32; 3],
        kinematics: CompactionKinematics,
    ) {
        let maximum_radius = self
            .packed
            .segment_radii
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        for axis in 0..3 {
            assert!(new_lower[axis].is_finite() && new_upper[axis].is_finite());
            assert!(new_upper[axis] - new_lower[axis] > 2.0 * maximum_radius);
            assert!(new_lower[axis] >= self.cell_lower_host[axis]);
            assert!(new_upper[axis] <= self.cell_upper_host[axis]);
        }
        let new_lower_buffer = self.client.create_from_slice(f32::as_bytes(&new_lower));
        let new_upper_buffer = self.client.create_from_slice(f32::as_bytes(&new_upper));
        unsafe {
            match kinematics {
                CompactionKinematics::RigidFiberCenters => {
                    compact_rigid_fiber_centers::launch_unchecked::<R>(
                        &self.client,
                        CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(
                            self.positions.clone(),
                            self.packed.positions.len(),
                        ),
                        BufferArg::from_raw_parts(
                            self.fiber_vertex_spans.clone(),
                            self.packed.fiber_vertex_spans.len(),
                        ),
                        BufferArg::from_raw_parts(
                            self.vertex_active.clone(),
                            self.packed.vertex_active.len(),
                        ),
                        BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                        BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                        BufferArg::from_raw_parts(new_lower_buffer.clone(), 3),
                        BufferArg::from_raw_parts(new_upper_buffer.clone(), 3),
                        BufferArg::from_raw_parts(
                            self.wall_reactions.clone(),
                            3 * self.packed.vertex_count(),
                        ),
                    );
                }
                CompactionKinematics::MovingWalls => {
                    compact_moving_walls::launch_unchecked::<R>(
                        &self.client,
                        CubeCount::Static(self.packed.vertex_count().div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(
                            self.positions.clone(),
                            self.packed.positions.len(),
                        ),
                        BufferArg::from_raw_parts(
                            self.vertex_wall_extents.clone(),
                            3 * self.packed.vertex_count(),
                        ),
                        BufferArg::from_raw_parts(
                            self.vertex_active.clone(),
                            self.packed.vertex_active.len(),
                        ),
                        BufferArg::from_raw_parts(new_lower_buffer.clone(), 3),
                        BufferArg::from_raw_parts(new_upper_buffer.clone(), 3),
                        BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                        BufferArg::from_raw_parts(
                            self.wall_reactions.clone(),
                            3 * self.packed.vertex_count(),
                        ),
                    );
                }
                CompactionKinematics::AffineVertices => {
                    compact_affine_vertices::launch_unchecked::<R>(
                        &self.client,
                        CubeCount::Static(self.packed.vertex_count().div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(
                            self.positions.clone(),
                            self.packed.positions.len(),
                        ),
                        BufferArg::from_raw_parts(
                            self.vertex_active.clone(),
                            self.packed.vertex_active.len(),
                        ),
                        BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                        BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                        BufferArg::from_raw_parts(new_lower_buffer.clone(), 3),
                        BufferArg::from_raw_parts(new_upper_buffer.clone(), 3),
                        BufferArg::from_raw_parts(
                            self.wall_reactions.clone(),
                            3 * self.packed.vertex_count(),
                        ),
                    );
                }
            }
        }
        self.cell_lower = new_lower_buffer;
        self.cell_upper = new_upper_buffer;
        self.cell_lower_host = new_lower;
        self.cell_upper_host = new_upper;
        self.packed.cell_lower = new_lower;
        self.packed.cell_upper = new_upper;
        self.cells_x = ((new_upper[0] - new_lower[0]) / self.cell_size)
            .floor()
            .max(1.0) as u32;
        self.cells_y = ((new_upper[1] - new_lower[1]) / self.cell_size)
            .floor()
            .max(1.0) as u32;
        self.cells_z = ((new_upper[2] - new_lower[2]) / self.cell_size)
            .floor()
            .max(1.0) as u32;
        assert!(
            self.cells_x as usize * self.cells_y as usize * self.cells_z as usize
                <= self.cell_count
        );
        // The grid and every vertex moved with the cell.
        self.request_neighbor_list_rebuild();
    }

    /// Reduces directional pressure and penalty-energy measures from the
    /// current device state without downloading geometry.
    pub fn compaction_metrics(
        &self,
        correction_fraction: f32,
        model: CompactionEnergyModel,
    ) -> CompactionMetrics {
        assert!(correction_fraction > 0.0);
        assert!(model.contact_stiffness >= 0.0);
        assert!(model.stretch_stiffness >= 0.0);
        assert!(model.bending_stiffness >= 0.0);
        unsafe {
            measure_compaction_metrics::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_rest_lengths.clone(),
                    self.packed.segment_rest_lengths.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_active.clone(),
                    self.packed.segment_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.corrections.clone(),
                    6 * self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(self.segment_max.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(self.curvature_ratio.clone(), self.packed.vertex_count()),
                BufferArg::from_raw_parts(
                    self.wall_reactions.clone(),
                    3 * self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(self.compaction_metrics.clone(), 13),
                correction_fraction,
                model.contact_stiffness,
                model.stretch_stiffness,
                model.bending_stiffness,
            );
        }
        let bytes = self
            .client
            .read_one(self.compaction_metrics.clone())
            .expect("CubeCL compaction-metric readback failed");
        let values = f32::from_bytes(&bytes);
        CompactionMetrics {
            pressure: [values[0], values[1], values[2]],
            contact_energy: values[3],
            stretch_energy: values[4],
            bending_energy: values[5],
            total_energy: values[6],
        }
    }

    /// Lower/upper pressure on each cell-axis face from the most recent
    /// compaction-metric reduction. Each row is `[lower, upper]`.
    pub fn wall_face_pressures(&self) -> [[f32; 2]; 3] {
        let bytes = self
            .client
            .read_one(self.compaction_metrics.clone())
            .expect("CubeCL wall-face metric readback failed");
        let values = f32::from_bytes(&bytes);
        [
            [values[7], values[8]],
            [values[9], values[10]],
            [values[11], values[12]],
        ]
    }

    /// Current resident orthorhombic cell bounds.
    pub fn cell_bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.cell_lower_host, self.cell_upper_host)
    }

    /// Makes every prepacked fiber through `maximum_step` participate in
    /// subsequent device kernels without replacing the resident world.
    pub fn activate_formation_step(&mut self, maximum_step: u32) -> (usize, usize) {
        unsafe {
            activate_formation_step::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.fiber_count().div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(
                    self.fiber_formation_steps.clone(),
                    self.packed.fiber_formation_steps.len(),
                ),
                BufferArg::from_raw_parts(
                    self.fiber_segment_spans.clone(),
                    self.packed.fiber_segment_spans.len(),
                ),
                BufferArg::from_raw_parts(
                    self.fiber_vertex_spans.clone(),
                    self.packed.fiber_vertex_spans.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_refinement_levels.clone(),
                    self.packed.segment_refinement_levels.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_refinement_levels.clone(),
                    self.packed.vertex_refinement_levels.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_active.clone(),
                    self.packed.segment_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                maximum_step,
            );
        }
        self.rebuild_active_indices();
        // Newly active fibers are missing from the lists.
        self.request_neighbor_list_rebuild();
        (self.active_segment_count, self.active_vertex_count)
    }

    /// Replaces per-fiber curvature limits without downloading resident
    /// geometry. Values are maximum curvature in inverse-length units; zero
    /// disables the limit for that fiber.
    pub fn set_fiber_maximum_curvatures(&mut self, maximum_curvatures: &[f32]) {
        assert_eq!(maximum_curvatures.len(), self.packed.fiber_count());
        for (vertex, fiber) in self.packed.vertex_fibers.iter().copied().enumerate() {
            self.packed.vertex_max_curvature[vertex] = maximum_curvatures[fiber as usize];
        }
        self.vertex_max_curvature = self
            .client
            .create_from_slice(f32::as_bytes(&self.packed.vertex_max_curvature));
    }

    /// Captures a compact list of current inter-fiber contacts on the GPU.
    ///
    /// Geometry remains resident. Only the accepted segment pairs and their
    /// closest-point coordinates are downloaded at this explicit checkpoint.
    pub fn capture_contacts(
        &mut self,
        maximum_surface_gap: f32,
        capacity: usize,
    ) -> ContactCapture {
        assert!(maximum_surface_gap.is_finite() && maximum_surface_gap >= 0.0);
        assert!(capacity > 0);
        self.rebuild_active_indices();
        // Whole segments need the segment-sized grid; never go finer than the
        // allocated neighbor grid, whose cell buffers the capture reuses.
        let capture_cell_size = self.segment_cell_size.max(self.cell_size) + maximum_surface_gap;
        let extents = [
            self.packed.cell_upper[0] - self.packed.cell_lower[0],
            self.packed.cell_upper[1] - self.packed.cell_lower[1],
            self.packed.cell_upper[2] - self.packed.cell_lower[2],
        ];
        let cells_x = (extents[0] / capture_cell_size).floor().max(1.0) as u32;
        let cells_y = (extents[1] / capture_cell_size).floor().max(1.0) as u32;
        let cells_z = (extents[2] / capture_cell_size).floor().max(1.0) as u32;
        let cell_count = cells_x as usize * cells_y as usize * cells_z as usize;
        let cube_dim = CubeDim::new_1d(64);
        let capture_control = self.client.create_from_slice(u32::as_bytes(&[1_u32]));
        let captured_count = self.client.create_from_slice(u32::as_bytes(&[0_u32, 0]));
        let captured_segments = self
            .client
            .empty(2 * capacity * core::mem::size_of::<u32>());
        let captured_coordinates = self
            .client
            .empty(2 * capacity * core::mem::size_of::<f32>());
        let captured_surface_gaps = self.client.empty(capacity * core::mem::size_of::<f32>());
        let captured_crossing_angles = self.client.empty(capacity * core::mem::size_of::<f32>());
        self.rebuild_cell_list(
            cell_count,
            cells_x,
            cells_y,
            cells_z,
            capture_control.clone(),
            1,
            false,
        );
        // The shared grid now uses the capture layout, which the overflow
        // path of the next contact pass must not read.
        self.request_neighbor_list_rebuild();
        unsafe {
            capture_segment_contacts::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.active_segment_count.div_ceil(64) as u32, 1, 1),
                cube_dim,
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_fibers.clone(),
                    self.packed.segment_fibers.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_radii.clone(),
                    self.packed.segment_radii.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_segment_indices.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.cell_counts.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_offsets.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_segments.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(captured_count.clone(), 2),
                BufferArg::from_raw_parts(captured_segments.clone(), 2 * capacity),
                BufferArg::from_raw_parts(captured_coordinates.clone(), 2 * capacity),
                BufferArg::from_raw_parts(captured_surface_gaps.clone(), capacity),
                BufferArg::from_raw_parts(captured_crossing_angles.clone(), capacity),
                maximum_surface_gap,
                cells_x,
                cells_y,
                cells_z,
            );
        }
        let count_bytes = self
            .client
            .read_one(captured_count)
            .expect("CubeCL contact-capture count readback failed");
        let count_values = u32::from_bytes(&count_bytes);
        let count = (count_values[0] as usize).min(capacity);
        let cell_overflow_bytes = self
            .client
            .read_one(self.cell_overflow.clone())
            .expect("CubeCL contact-capture overflow readback failed");
        let segments_bytes = self
            .client
            .read_one(captured_segments)
            .expect("CubeCL contact segment readback failed");
        let coordinates_bytes = self
            .client
            .read_one(captured_coordinates)
            .expect("CubeCL contact coordinate readback failed");
        let gaps_bytes = self
            .client
            .read_one(captured_surface_gaps)
            .expect("CubeCL contact gap readback failed");
        let angles_bytes = self
            .client
            .read_one(captured_crossing_angles)
            .expect("CubeCL contact angle readback failed");
        let segments = u32::from_bytes(&segments_bytes);
        let coordinates = f32::from_bytes(&coordinates_bytes);
        let gaps = f32::from_bytes(&gaps_bytes);
        let angles = f32::from_bytes(&angles_bytes);
        let mut candidates: Vec<_> = (0..count)
            .map(|index| SegmentContactCandidate {
                first_segment: segments[2 * index],
                second_segment: segments[2 * index + 1],
                first_coordinate: coordinates[2 * index],
                second_coordinate: coordinates[2 * index + 1],
                surface_gap: gaps[index],
                crossing_angle: angles[index],
            })
            .collect();
        // The device appends candidates in a run-dependent order; sort them so
        // junction capture sees the same sequence every run.
        candidates.sort_by(|left, right| {
            (left.first_segment, left.second_segment)
                .cmp(&(right.first_segment, right.second_segment))
                .then(left.first_coordinate.total_cmp(&right.first_coordinate))
                .then(left.second_coordinate.total_cmp(&right.second_coordinate))
        });
        ContactCapture {
            candidates,
            overflow: count_values[1] != 0 || u32::from_bytes(&cell_overflow_bytes)[0] != 0,
        }
    }

    /// Downloads scalar convergence state without downloading geometry.
    pub fn read_status(&self, config: &RelaxationConfig) -> BatchStatus {
        // One batched readback: each separate read is a full device sync.
        let [control_bytes, metric_bytes, overflow_bytes, refinement_bytes]: [_; 4] =
            cubecl::future::reader::read_sync(self.client.read_async(vec![
                self.control.clone(),
                self.metrics.clone(),
                self.cell_overflow.clone(),
                self.refinement_count.clone(),
            ]))
            .expect("CubeCL batch-status readback failed")
            .try_into()
            .unwrap_or_else(|_| panic!("CubeCL returned the wrong number of status buffers"));
        let control = u32::from_bytes(&control_bytes);
        let metrics = f32::from_bytes(&metric_bytes);
        let overflow = u32::from_bytes(&overflow_bytes)[0] != 0 || control[3] != 0;
        let refinement = u32::from_bytes(&refinement_bytes);
        BatchStatus {
            total_iterations: control[1] as usize,
            batch_iterations: control[2] as usize,
            max_penetration: metrics[0],
            max_displacement: metrics[1],
            max_curvature_ratio: metrics[2],
            converged: !config.force_full_iterations
                && metrics[0] <= config.penetration_tolerance
                && metrics[2] <= 1.0 + config.curvature_ratio_tolerance
                && !overflow,
            cell_list_overflow: overflow,
            segment_splits: refinement[0] as usize,
            refinement_passes: refinement[2] as usize,
            segment_merges: refinement[3] as usize,
            coarsening_passes: refinement[5] as usize,
        }
    }

    /// Downloads the current positions and installs them into a host assembly.
    pub fn download_into(
        &self,
        assembly: &mut tangle_core::FiberAssembly,
    ) -> Result<(), PackingError> {
        let bytes = self
            .client
            .read_one(self.positions.clone())
            .expect("CubeCL position readback failed");
        self.unpack_positions(f32::from_bytes(&bytes), assembly)
    }

    /// Downloads the current interleaved xyz position buffer.
    pub fn download_positions(&self) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(self.positions.clone())
            .expect("CubeCL position readback failed");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Installs an already downloaded position buffer into a host assembly.
    pub fn unpack_positions(
        &self,
        positions: &[f32],
        assembly: &mut tangle_core::FiberAssembly,
    ) -> Result<(), PackingError> {
        let directors = self.packed.has_ovals.then(|| self.download_directors());
        if self.packed.adaptive
            || self
                .packed
                .fiber_formation_steps
                .iter()
                .any(|step| *step > 0)
        {
            let active = self.download_vertex_active();
            self.packed
                .unpack_active_positions(positions, directors.as_deref(), &active, assembly)
        } else {
            self.packed
                .unpack_positions(positions, directors.as_deref(), assembly)
        }
    }

    /// Downloads the current interleaved long-axis directors (zero for round
    /// fibers).
    pub fn download_directors(&self) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(self.directors.clone())
            .expect("CubeCL director readback failed");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Downloads the current active-vertex flags.
    pub fn download_vertex_active(&self) -> Vec<u32> {
        let bytes = self
            .client
            .read_one(self.vertex_active.clone())
            .expect("CubeCL active-vertex readback failed");
        u32::from_bytes(&bytes).to_vec()
    }

    /// Downloads the segment activity mask at an explicit topology checkpoint.
    pub fn download_segment_active(&self) -> Vec<u32> {
        let bytes = self
            .client
            .read_one(self.segment_active.clone())
            .expect("CubeCL segment activity readback failed");
        u32::from_bytes(&bytes).to_vec()
    }

    /// Activates the dyadic ancestor paths needed to expose selected reserved
    /// material vertices to an external formation constraint.
    ///
    /// This sparse macro-operation downloads topology masks, uploads compact
    /// path indices, and performs geometry interpolation and topology edits on
    /// the device. Targets must belong to distinct fibers so paths cannot
    /// overlap.
    pub fn activate_refinement_vertices(
        &mut self,
        vertices: &[u32],
        refinement_interval: usize,
    ) -> usize {
        assert!(refinement_interval > 0);
        if vertices.is_empty() || !self.packed.adaptive {
            return 0;
        }
        let active_vertices = self.download_vertex_active();
        let active_segments = self.download_segment_active();
        let mut fibers = Vec::with_capacity(vertices.len());
        let mut path_spans = Vec::with_capacity(2 * vertices.len());
        let mut path_segments = Vec::new();
        for vertex in vertices.iter().copied() {
            let vertex_index = vertex as usize;
            assert!(vertex_index < self.packed.vertex_count());
            let fiber = self.packed.vertex_fibers[vertex_index] as usize;
            assert!(
                !fibers.contains(&fiber),
                "forced refinement targets must belong to distinct fibers"
            );
            fibers.push(fiber);
            if active_vertices[vertex_index] != 0 {
                continue;
            }
            let segment_start = self.packed.fiber_segment_spans[2 * fiber] as usize;
            let segment_count = self.packed.fiber_segment_spans[2 * fiber + 1] as usize;
            let mut current = (segment_start..segment_start + segment_count)
                .find(|segment| {
                    if active_segments[*segment] == 0 {
                        return false;
                    }
                    let first = self.packed.segment_vertices[2 * *segment];
                    let second = self.packed.segment_vertices[2 * *segment + 1];
                    first < vertex && vertex < second
                })
                .expect("reserved target vertex must lie inside one active segment");
            let path_start = path_segments.len();
            loop {
                let first = self.packed.segment_vertices[2 * current];
                let second = self.packed.segment_vertices[2 * current + 1];
                let midpoint = (first + second) / 2;
                path_segments.push(current as u32);
                if midpoint == vertex {
                    break;
                }
                let child_offset = usize::from(vertex > midpoint);
                let child = self.packed.segment_children[2 * current + child_offset];
                assert_ne!(
                    child,
                    u32::MAX,
                    "target vertex has no reserved refinement path"
                );
                current = child as usize;
            }
            path_spans.extend([path_start as u32, (path_segments.len() - path_start) as u32]);
        }
        if path_spans.is_empty() {
            return 0;
        }
        let target_count = path_spans.len() / 2;
        let split_count = path_segments.len();
        let spans = self.client.create_from_slice(u32::as_bytes(&path_spans));
        let segments = self.client.create_from_slice(u32::as_bytes(&path_segments));
        let epoch = u32::try_from(self.total_iterations / refinement_interval + 1)
            .expect("adaptive refinement epoch exceeds u32");
        unsafe {
            refine_vertex_paths::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(target_count.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_children.clone(),
                    self.packed.segment_children.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_active.clone(),
                    self.packed.segment_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_birth_epochs.clone(),
                    self.packed.segment_birth_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_contact_epochs.clone(),
                    self.packed.segment_contact_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_quiet_epochs.clone(),
                    self.packed.segment_quiet_epochs.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_segments.clone(),
                    self.packed.vertex_segments.len(),
                ),
                BufferArg::from_raw_parts(spans, path_spans.len()),
                BufferArg::from_raw_parts(segments, path_segments.len()),
                BufferArg::from_raw_parts(self.refinement_count.clone(), 6),
                epoch,
            );
        }
        // The split parents are now inactive and the path midpoints active;
        // every per-iteration kernel walks the active lists, so refresh them
        // now rather than at the next adaptation epoch.
        self.rebuild_active_indices();
        self.request_neighbor_list_rebuild();
        split_count
    }

    /// Returns the immutable packed topology used to create this world.
    pub fn packed(&self) -> &PackedAssembly {
        &self.packed
    }

    /// Width of one broad-phase cell.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Number of broad-phase cells.
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }
}

#[cfg(all(test, any(feature = "wgpu", feature = "cpu")))]
#[path = "world_tests.rs"]
mod tests;
