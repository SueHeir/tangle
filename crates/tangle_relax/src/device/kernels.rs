use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn begin_relaxation_batch(control: &mut [u32]) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    control[2] = 0;
    if control[3] == 0 {
        control[0] = 1;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn clear_wall_reactions(wall_reactions: &mut [f32], control: &[u32]) {
    let vertex = ABSOLUTE_POS;
    if vertex >= wall_reactions.len() / 3 || control[0] == 0 {
        terminate!();
    }
    let index = vertex as usize;
    wall_reactions[3 * index] = 0.0;
    wall_reactions[3 * index + 1] = 0.0;
    wall_reactions[3 * index + 2] = 0.0;
}

#[cube(launch_unchecked)]
pub(crate) fn clear_adaptation_epoch(count: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        count[1].store(0);
        count[4].store(0);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn finish_adaptation_epoch(count: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        if count[1].load() > 0 {
            count[2].fetch_add(1);
        }
        if count[4].load() > 0 {
            count[5].fetch_add(1);
        }
    }
}

/// Marks contact neighbor lists stale when the adaptation epoch that just
/// finished split or merged any segment.
#[cube(launch_unchecked)]
pub(crate) fn request_neighbor_list_rebuild_after_adaptation(
    refinement_count: &[u32],
    neighbor_state: &mut [u32],
) {
    if ABSOLUTE_POS == 0 && (refinement_count[1] != 0 || refinement_count[4] != 0) {
        neighbor_state[0] = 1;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn clear_active_index_counts(counts: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        counts[0].store(0);
        counts[1].store(0);
    }
}

/// Compacts sparse adaptive topology masks into dense device-resident lists.
#[cube(launch_unchecked)]
pub(crate) fn compact_active_indices(
    segment_active: &[u32],
    vertex_active: &[u32],
    active_segments: &mut [u32],
    active_vertices: &mut [u32],
    counts: &mut [Atomic<u32>],
) {
    let index = ABSOLUTE_POS;
    if index < segment_active.len() && segment_active[index as usize] != 0 {
        let slot = counts[0].fetch_add(1);
        active_segments[slot as usize] = index as u32;
    }
    if index < vertex_active.len() && vertex_active[index as usize] != 0 {
        let slot = counts[1].fetch_add(1);
        active_vertices[slot as usize] = index as u32;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn clear_reduced_metrics(metrics: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        metrics[0].store(0);
        metrics[1].store(0);
        metrics[2].store(0);
    }
}

/// Atomically reduces nonnegative contact, curvature, and displacement values.
/// Positive IEEE-754 bit patterns have the same ordering as `u32`.
#[cube(launch_unchecked)]
pub(crate) fn reduce_active_segment_penetration(
    active_segments: &[u32],
    counts: &[Atomic<u32>],
    segment_max_penetration: &[f32],
    metrics: &mut [Atomic<u32>],
) {
    let work = ABSOLUTE_POS;
    if work >= counts[0].load() as usize {
        terminate!();
    }
    let segment = active_segments[work as usize] as usize;
    metrics[0].fetch_max(segment_max_penetration[segment].max(0.0).to_bits());
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_active_vertex_metrics(
    active_vertices: &[u32],
    counts: &[Atomic<u32>],
    curvature_ratio: &[f32],
    vertex_step: &[f32],
    metrics: &mut [Atomic<u32>],
) {
    let work = ABSOLUTE_POS;
    if work >= counts[1].load() as usize {
        terminate!();
    }
    let vertex = active_vertices[work as usize] as usize;
    metrics[1].fetch_max(vertex_step[vertex].max(0.0).to_bits());
    metrics[2].fetch_max(curvature_ratio[vertex].max(0.0).to_bits());
}

#[cube(launch_unchecked)]
pub(crate) fn assess_reduced_metrics(
    reduced: &[Atomic<u32>],
    cell_overflow: &[Atomic<u32>],
    control: &mut [u32],
    metrics: &mut [f32],
    penetration_tolerance: f32,
    curvature_ratio_tolerance: f32,
    max_iterations: u32,
    force_full_iterations: u32,
) {
    if ABSOLUTE_POS != 0 || control[0] == 0 {
        terminate!();
    }
    let maximum_penetration = f32::from_bits(reduced[0].load());
    let maximum_curvature_ratio = f32::from_bits(reduced[2].load());
    metrics[0] = maximum_penetration;
    metrics[1] = f32::from_bits(reduced[1].load());
    metrics[2] = maximum_curvature_ratio;
    if cell_overflow[0].load() != 0
        || (force_full_iterations == 0
            && maximum_penetration <= penetration_tolerance
            && maximum_curvature_ratio <= 1.0 + curvature_ratio_tolerance)
        || control[2] >= max_iterations
    {
        if cell_overflow[0].load() != 0 {
            control[3] = 1;
        }
        control[0] = 0;
    } else {
        control[1] += 1;
        control[2] += 1;
    }
}

/// Activates all prepacked fibers assigned to this or an earlier recipe step.
///
/// Formation activation is intentionally limited to fixed centerline layouts;
/// adaptive refinement owns these same topology masks after relaxation starts.
#[cube(launch_unchecked)]
pub(crate) fn activate_formation_step(
    fiber_steps: &[u32],
    fiber_segment_spans: &[u32],
    fiber_vertex_spans: &[u32],
    segment_refinement_levels: &[u32],
    vertex_refinement_levels: &[u32],
    segment_active: &mut [u32],
    vertex_active: &mut [u32],
    maximum_step: u32,
) {
    let fiber = ABSOLUTE_POS;
    if fiber >= fiber_steps.len() {
        terminate!();
    }
    let fiber_index = fiber as usize;
    let segment_start = fiber_segment_spans[2 * fiber_index] as usize;
    let segment_count = fiber_segment_spans[2 * fiber_index + 1] as usize;
    let vertex_start = fiber_vertex_spans[2 * fiber_index] as usize;
    let vertex_count = fiber_vertex_spans[2 * fiber_index + 1] as usize;
    if fiber_steps[fiber_index] > maximum_step {
        for local in 0..segment_count {
            segment_active[segment_start + local] = 0;
        }
        for local in 0..vertex_count {
            vertex_active[vertex_start + local] = 0;
        }
    } else {
        let mut already_active = false;
        for local in 0..segment_count {
            if segment_active[segment_start + local] != 0 {
                already_active = true;
            }
        }
        if !already_active {
            for local in 0..segment_count {
                let segment = segment_start + local;
                let mut active = 0_u32;
                if segment_refinement_levels[segment] == 0 {
                    active = 1;
                }
                segment_active[segment] = active;
            }
            for local in 0..vertex_count {
                let vertex = vertex_start + local;
                let mut active = 0_u32;
                if vertex_refinement_levels[vertex] == 0 {
                    active = 1;
                }
                vertex_active[vertex] = active;
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn refine_contact_segments(
    positions: &mut [f32],
    segment_vertices: &[u32],
    segment_radii: &[f32],
    segment_rest_lengths: &[f32],
    segment_children: &[u32],
    segment_max_penetration: &[f32],
    segment_active: &mut [u32],
    segment_birth_epochs: &mut [u32],
    segment_contact_epochs: &mut [u32],
    segment_quiet_epochs: &mut [u32],
    vertex_active: &mut [u32],
    vertex_segments: &mut [u32],
    control: &[u32],
    refinement_count: &mut [Atomic<u32>],
    epoch: u32,
    penetration_threshold: f32,
    contact_length_over_diameter: f32,
    minimum_length_over_diameter: f32,
    required_contact_epochs: u32,
) {
    let segment = ABSOLUTE_POS;
    if segment >= segment_active.len() || control[0] == 0 {
        terminate!();
    }
    let index = segment as usize;
    if segment_active[index] == 0 {
        segment_contact_epochs[index] = 0;
        terminate!();
    }
    if segment_birth_epochs[index] >= epoch {
        terminate!();
    }
    if segment_max_penetration[index] <= penetration_threshold {
        segment_contact_epochs[index] = 0;
        terminate!();
    }
    let left = segment_children[2 * index];
    let right = segment_children[2 * index + 1];
    if left == u32::MAX || right == u32::MAX {
        segment_contact_epochs[index] = 0;
        terminate!();
    }
    let rest_length = segment_rest_lengths[index];
    let diameter = 2.0 * segment_radii[index];
    if rest_length <= contact_length_over_diameter * diameter
        || 0.5 * rest_length < minimum_length_over_diameter * diameter
    {
        segment_contact_epochs[index] = 0;
        terminate!();
    }
    let persistence = segment_contact_epochs[index] + 1;
    segment_contact_epochs[index] = persistence;
    if persistence < required_contact_epochs {
        terminate!();
    }

    let first = segment_vertices[2 * index] as usize;
    let second = segment_vertices[2 * index + 1] as usize;
    let midpoint = (first + second) / 2;
    positions[3 * midpoint] = 0.5 * (positions[3 * first] + positions[3 * second]);
    positions[3 * midpoint + 1] = 0.5 * (positions[3 * first + 1] + positions[3 * second + 1]);
    positions[3 * midpoint + 2] = 0.5 * (positions[3 * first + 2] + positions[3 * second + 2]);
    vertex_active[midpoint] = 1;
    vertex_segments[2 * first + 1] = left;
    vertex_segments[2 * midpoint] = left;
    vertex_segments[2 * midpoint + 1] = right;
    vertex_segments[2 * second] = right;
    segment_active[index] = 0;
    segment_active[left as usize] = 1;
    segment_active[right as usize] = 1;
    segment_birth_epochs[left as usize] = epoch;
    segment_birth_epochs[right as usize] = epoch;
    segment_contact_epochs[index] = 0;
    segment_contact_epochs[left as usize] = 0;
    segment_contact_epochs[right as usize] = 0;
    segment_quiet_epochs[index] = 0;
    segment_quiet_epochs[left as usize] = 0;
    segment_quiet_epochs[right as usize] = 0;
    refinement_count[0].fetch_add(1);
    refinement_count[1].fetch_add(1);
}

/// Activates precomputed ancestor paths to material vertices required by an
/// external geometric operation such as a needle constraint.
///
/// Each path belongs to a different fiber, so its thread may update the path
/// sequentially without conflicting with another target thread.
#[cube(launch_unchecked)]
pub(crate) fn refine_vertex_paths(
    positions: &mut [f32],
    segment_vertices: &[u32],
    segment_children: &[u32],
    segment_active: &mut [u32],
    segment_birth_epochs: &mut [u32],
    segment_contact_epochs: &mut [u32],
    segment_quiet_epochs: &mut [u32],
    vertex_active: &mut [u32],
    vertex_segments: &mut [u32],
    path_spans: &[u32],
    path_segments: &[u32],
    refinement_count: &mut [Atomic<u32>],
    epoch: u32,
) {
    let target = ABSOLUTE_POS;
    if target >= path_spans.len() / 2 {
        terminate!();
    }
    let target_index = target as usize;
    let start = path_spans[2 * target_index] as usize;
    let count = path_spans[2 * target_index + 1] as usize;
    for local in 0..count {
        let parent = path_segments[start + local] as usize;
        if segment_active[parent] != 0 {
            let left = segment_children[2 * parent] as usize;
            let right = segment_children[2 * parent + 1] as usize;
            let first = segment_vertices[2 * parent] as usize;
            let second = segment_vertices[2 * parent + 1] as usize;
            let midpoint = (first + second) / 2;
            positions[3 * midpoint] = 0.5 * (positions[3 * first] + positions[3 * second]);
            positions[3 * midpoint + 1] =
                0.5 * (positions[3 * first + 1] + positions[3 * second + 1]);
            positions[3 * midpoint + 2] =
                0.5 * (positions[3 * first + 2] + positions[3 * second + 2]);
            vertex_active[midpoint] = 1;
            vertex_segments[2 * first + 1] = left as u32;
            vertex_segments[2 * midpoint] = left as u32;
            vertex_segments[2 * midpoint + 1] = right as u32;
            vertex_segments[2 * second] = right as u32;
            segment_active[parent] = 0;
            segment_active[left] = 1;
            segment_active[right] = 1;
            segment_birth_epochs[left] = epoch;
            segment_birth_epochs[right] = epoch;
            segment_contact_epochs[parent] = 0;
            segment_contact_epochs[left] = 0;
            segment_contact_epochs[right] = 0;
            segment_quiet_epochs[parent] = 0;
            segment_quiet_epochs[left] = 0;
            segment_quiet_epochs[right] = 0;
            refinement_count[0].fetch_add(1);
        }
    }
    if target == 0 && path_segments.len() > 0 {
        refinement_count[2].fetch_add(1);
    }
}

/// Marks inactive parents whose two active children have remained contact-free
/// and whose removable midpoint carries negligible geometric information.
/// This read-only topology pass prevents parent/child races during merging.
#[cube(launch_unchecked)]
pub(crate) fn mark_coarsening_candidates(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_radii: &[f32],
    segment_children: &[u32],
    segment_max_penetration: &[f32],
    segment_active: &[u32],
    segment_quiet_epochs: &mut [u32],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    curvature_ratio: &[f32],
    control: &[u32],
    candidates: &mut [u32],
    penetration_threshold: f32,
    required_quiet_epochs: u32,
    maximum_error_over_diameter: f32,
    maximum_curvature_ratio: f32,
) {
    let segment = ABSOLUTE_POS;
    if segment >= segment_active.len() {
        terminate!();
    }
    let index = segment as usize;
    candidates[index] = 0;
    if control[0] == 0 || segment_active[index] != 0 {
        segment_quiet_epochs[index] = 0;
        terminate!();
    }
    let left = segment_children[2 * index];
    let right = segment_children[2 * index + 1];
    if left == u32::MAX
        || right == u32::MAX
        || segment_active[left as usize] == 0
        || segment_active[right as usize] == 0
        || segment_max_penetration[left as usize] > penetration_threshold
        || segment_max_penetration[right as usize] > penetration_threshold
    {
        segment_quiet_epochs[index] = 0;
        terminate!();
    }

    let first = segment_vertices[2 * index] as usize;
    let second = segment_vertices[2 * index + 1] as usize;
    let midpoint = (first + second) / 2;
    if vertex_active[midpoint] == 0
        || vertex_pinned[midpoint] != 0
        || curvature_ratio[midpoint] > maximum_curvature_ratio
    {
        segment_quiet_epochs[index] = 0;
        terminate!();
    }
    let error_x = positions[3 * midpoint] - 0.5 * (positions[3 * first] + positions[3 * second]);
    let error_y =
        positions[3 * midpoint + 1] - 0.5 * (positions[3 * first + 1] + positions[3 * second + 1]);
    let error_z =
        positions[3 * midpoint + 2] - 0.5 * (positions[3 * first + 2] + positions[3 * second + 2]);
    let error = (error_x * error_x + error_y * error_y + error_z * error_z).sqrt();
    let diameter = 2.0 * segment_radii[index];
    if error > maximum_error_over_diameter * diameter {
        segment_quiet_epochs[index] = 0;
        terminate!();
    }

    let quiet = segment_quiet_epochs[index] + 1;
    segment_quiet_epochs[index] = quiet;
    if quiet >= required_quiet_epochs {
        candidates[index] = 1;
    }
}

/// Applies a race-free set of sibling merges selected by
/// [`mark_coarsening_candidates`]. Candidate parents cannot be ancestors of
/// one another because each candidate required both children to be active in
/// the preceding read-only pass.
#[cube(launch_unchecked)]
pub(crate) fn apply_coarsening_candidates(
    segment_vertices: &[u32],
    segment_children: &[u32],
    segment_active: &mut [u32],
    segment_birth_epochs: &mut [u32],
    segment_contact_epochs: &mut [u32],
    segment_quiet_epochs: &mut [u32],
    vertex_active: &mut [u32],
    vertex_segments: &mut [u32],
    candidates: &[u32],
    control: &[u32],
    refinement_count: &mut [Atomic<u32>],
    epoch: u32,
) {
    let segment = ABSOLUTE_POS;
    if segment >= segment_active.len() || control[0] == 0 {
        terminate!();
    }
    let index = segment as usize;
    if candidates[index] == 0 {
        terminate!();
    }
    let left = segment_children[2 * index] as usize;
    let right = segment_children[2 * index + 1] as usize;
    let first = segment_vertices[2 * index] as usize;
    let second = segment_vertices[2 * index + 1] as usize;
    let midpoint = (first + second) / 2;

    segment_active[index] = 1;
    segment_active[left] = 0;
    segment_active[right] = 0;
    segment_birth_epochs[index] = epoch;
    segment_contact_epochs[index] = 0;
    segment_contact_epochs[left] = 0;
    segment_contact_epochs[right] = 0;
    segment_quiet_epochs[index] = 0;
    segment_quiet_epochs[left] = 0;
    segment_quiet_epochs[right] = 0;
    vertex_active[midpoint] = 0;
    vertex_segments[2 * first + 1] = segment as u32;
    vertex_segments[2 * midpoint] = u32::MAX;
    vertex_segments[2 * midpoint + 1] = u32::MAX;
    vertex_segments[2 * second] = segment as u32;
    refinement_count[3].fetch_add(1);
    refinement_count[4].fetch_add(1);
}

#[cube(launch_unchecked)]
pub(crate) fn apply_formation_layer_targets(
    positions: &mut [f32],
    fiber_vertex_spans: &[u32],
    fiber_layers: &[u32],
    vertex_active: &[u32],
    layer_targets: &[f32],
    control: &[u32],
    axis: u32,
    first_layer: u32,
    last_layer: u32,
    stiffness: f32,
    max_translation: f32,
) {
    let fiber = ABSOLUTE_POS;
    if fiber >= fiber_layers.len() || control[0] == 0 {
        terminate!();
    }
    let fiber_index = fiber as usize;
    let layer = fiber_layers[fiber_index];
    if layer == u32::MAX
        || layer >= layer_targets.len() as u32
        || layer < first_layer
        || layer > last_layer
    {
        terminate!();
    }
    let start = fiber_vertex_spans[2 * fiber_index] as usize;
    let count = fiber_vertex_spans[2 * fiber_index + 1] as usize;
    if count == 0 || vertex_active[start] == 0 {
        terminate!();
    }
    let coordinate_axis = axis as usize;
    let mut center = 0.0_f32;
    for local in 0..count {
        center += positions[3 * (start + local) + coordinate_axis];
    }
    center /= count as f32;
    let translation = ((layer_targets[layer as usize] - center) * stiffness)
        .clamp(-max_translation, max_translation);
    for local in 0..count {
        positions[3 * (start + local) + coordinate_axis] += translation;
    }
}

/// Captures an absolute target coordinate from a selected vertex's current
/// device-resident position plus a prescribed displacement.
#[cube(launch_unchecked)]
pub(crate) fn initialize_vertex_displacement_targets(
    positions: &[f32],
    vertex_indices: &[u32],
    target_coordinates: &mut [f32],
    command_coordinates: &mut [f32],
    axis: u32,
    displacement: f32,
) {
    let target = ABSOLUTE_POS;
    if target >= vertex_indices.len() {
        terminate!();
    }
    let index = target as usize;
    let vertex = vertex_indices[index] as usize;
    let coordinate = positions[3 * vertex + axis as usize];
    target_coordinates[index] = coordinate + displacement;
    command_coordinates[index] = coordinate;
}

/// Pulls selected active vertices toward absolute formation targets. The
/// target buffers remain resident and can be reapplied once per solver
/// iteration while contacts and fiber constraints relax around the needle.
#[cube(launch_unchecked)]
pub(crate) fn apply_vertex_targets(
    positions: &mut [f32],
    vertex_indices: &[u32],
    target_coordinates: &[f32],
    command_coordinates: &mut [f32],
    vertex_active: &[u32],
    control: &[u32],
    axis: u32,
    stiffness: f32,
    max_translation: f32,
) {
    let target = ABSOLUTE_POS;
    if target >= vertex_indices.len() || control[0] == 0 {
        terminate!();
    }
    let index = target as usize;
    let vertex = vertex_indices[index] as usize;
    if vertex_active[vertex] == 0 {
        terminate!();
    }
    let coordinate = 3 * vertex + axis as usize;
    let command = command_coordinates[index];
    let translation = ((target_coordinates[index] - command) * stiffness)
        .clamp(-max_translation, max_translation);
    command_coordinates[index] = command + translation;
    positions[coordinate] = command + translation;
}

/// Reduces the active manufacturing-layer target to one maximum center error.
///
/// Layer populations are small relative to the contact problem, so one device
/// invocation performs this infrequent batch-boundary diagnostic without
/// downloading the resident geometry.
#[cube(launch_unchecked)]
pub(crate) fn measure_layer_target_error(
    positions: &[f32],
    fiber_vertex_spans: &[u32],
    fiber_layers: &[u32],
    vertex_active: &[u32],
    layer_targets: &[f32],
    output: &mut [f32],
    axis: u32,
    first_layer: u32,
    last_layer: u32,
) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let coordinate_axis = axis as usize;
    let mut maximum = 0.0_f32;
    for fiber in 0..fiber_layers.len() {
        let layer = fiber_layers[fiber];
        if layer != u32::MAX
            && layer < layer_targets.len() as u32
            && layer >= first_layer
            && layer <= last_layer
        {
            let start = fiber_vertex_spans[2 * fiber] as usize;
            let count = fiber_vertex_spans[2 * fiber + 1] as usize;
            if count > 0 && vertex_active[start] != 0 {
                let mut center = 0.0_f32;
                for local in 0..count {
                    center += positions[3 * (start + local) + coordinate_axis];
                }
                center /= count as f32;
                let error = (layer_targets[layer as usize] - center).abs();
                if error > maximum {
                    maximum = error;
                }
            }
        }
    }
    output[0] = maximum;
}

/// Reduces all active needle targets to one maximum coordinate error.
#[cube(launch_unchecked)]
pub(crate) fn measure_vertex_target_error(
    positions: &[f32],
    vertex_indices: &[u32],
    target_coordinates: &[f32],
    vertex_active: &[u32],
    output: &mut [f32],
    axis: u32,
) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let coordinate_axis = axis as usize;
    let mut maximum = 0.0_f32;
    for target in 0..vertex_indices.len() {
        let vertex = vertex_indices[target] as usize;
        if vertex_active[vertex] != 0 {
            let coordinate = positions[3 * vertex + coordinate_axis];
            let error = (target_coordinates[target] - coordinate).abs();
            if error > maximum {
                maximum = error;
            }
        }
    }
    output[1] = maximum;
}

/// Moves active fibers with the changing cell while preserving each complete
/// centerline as a rigid shape.
#[cube(launch_unchecked)]
pub(crate) fn compact_rigid_fiber_centers(
    positions: &mut [f32],
    fiber_vertex_spans: &[u32],
    vertex_active: &[u32],
    old_lower: &[f32],
    old_upper: &[f32],
    new_lower: &[f32],
    new_upper: &[f32],
    wall_reactions: &mut [f32],
) {
    let fiber = ABSOLUTE_POS;
    if fiber >= fiber_vertex_spans.len() / 2 {
        terminate!();
    }
    let fiber_index = fiber as usize;
    let start = fiber_vertex_spans[2 * fiber_index] as usize;
    let count = fiber_vertex_spans[2 * fiber_index + 1] as usize;
    if count == 0 || vertex_active[start] == 0 {
        terminate!();
    }
    let mut center_x = 0.0_f32;
    let mut center_y = 0.0_f32;
    let mut center_z = 0.0_f32;
    for local in 0..count {
        let vertex = start + local;
        center_x += positions[3 * vertex];
        center_y += positions[3 * vertex + 1];
        center_z += positions[3 * vertex + 2];
    }
    center_x /= count as f32;
    center_y /= count as f32;
    center_z /= count as f32;
    let fraction_x = (center_x - old_lower[0]) / (old_upper[0] - old_lower[0]);
    let fraction_y = (center_y - old_lower[1]) / (old_upper[1] - old_lower[1]);
    let fraction_z = (center_z - old_lower[2]) / (old_upper[2] - old_lower[2]);
    let dx = new_lower[0] + fraction_x * (new_upper[0] - new_lower[0]) - center_x;
    let dy = new_lower[1] + fraction_y * (new_upper[1] - new_lower[1]) - center_y;
    let dz = new_lower[2] + fraction_z * (new_upper[2] - new_lower[2]) - center_z;
    for local in 0..count {
        let vertex = start + local;
        positions[3 * vertex] += dx;
        positions[3 * vertex + 1] += dy;
        positions[3 * vertex + 2] += dz;
        wall_reactions[3 * vertex] = 0.0;
        wall_reactions[3 * vertex + 1] = 0.0;
        wall_reactions[3 * vertex + 2] = 0.0;
    }
}

/// Applies the cell deformation gradient to each active vertex.
#[cube(launch_unchecked)]
pub(crate) fn compact_affine_vertices(
    positions: &mut [f32],
    vertex_active: &[u32],
    old_lower: &[f32],
    old_upper: &[f32],
    new_lower: &[f32],
    new_upper: &[f32],
    wall_reactions: &mut [f32],
) {
    let vertex = ABSOLUTE_POS;
    if vertex >= vertex_active.len() || vertex_active[vertex as usize] == 0 {
        terminate!();
    }
    let index = vertex as usize;
    for axis in 0..3 {
        let coordinate = 3 * index + axis;
        let fraction =
            (positions[coordinate] - old_lower[axis]) / (old_upper[axis] - old_lower[axis]);
        positions[coordinate] = new_lower[axis] + fraction * (new_upper[axis] - new_lower[axis]);
        wall_reactions[coordinate] = 0.0;
    }
}

/// Advances hard planar walls and projects active fiber material inside them.
#[cube(launch_unchecked)]
pub(crate) fn compact_moving_walls(
    positions: &mut [f32],
    segment_radii: &[f32],
    vertex_fibers: &[u32],
    fiber_segment_spans: &[u32],
    vertex_active: &[u32],
    new_lower: &[f32],
    new_upper: &[f32],
    cell_periodic: &[u32],
    wall_reactions: &mut [f32],
) {
    let vertex = ABSOLUTE_POS;
    if vertex >= vertex_active.len() || vertex_active[vertex as usize] == 0 {
        terminate!();
    }
    let index = vertex as usize;
    let fiber = vertex_fibers[index] as usize;
    let segment = fiber_segment_spans[2 * fiber] as usize;
    let radius = segment_radii[segment];
    for axis in 0..3 {
        let coordinate = 3 * index + axis;
        if cell_periodic[axis] == 0 {
            let previous = positions[coordinate];
            let projected = previous.clamp(new_lower[axis] + radius, new_upper[axis] - radius);
            positions[coordinate] = projected;
            // Negative is the lower face and positive is the upper face.
            wall_reactions[coordinate] = previous - projected;
        } else {
            wall_reactions[coordinate] = 0.0;
        }
    }
}

/// Reduces the current constraint state to directional pressure and penalty
/// energy measures used by the formation controller.
#[cube(launch_unchecked)]
pub(crate) fn measure_compaction_metrics(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_rest_lengths: &[f32],
    segment_active: &[u32],
    corrections: &[f32],
    segment_max_penetration: &[f32],
    curvature_ratio: &[f32],
    wall_reactions: &[f32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    output: &mut [f32],
    correction_fraction: f32,
    contact_stiffness: f32,
    stretch_stiffness: f32,
    bending_stiffness: f32,
) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let mut load_x = 0.0_f32;
    let mut load_y = 0.0_f32;
    let mut load_z = 0.0_f32;
    let mut contact_energy = 0.0_f32;
    let mut stretch_energy = 0.0_f32;
    for segment in 0..segment_active.len() {
        if segment_active[segment] != 0 {
            load_x += corrections[6 * segment].abs() + corrections[6 * segment + 3].abs();
            load_y += corrections[6 * segment + 1].abs() + corrections[6 * segment + 4].abs();
            load_z += corrections[6 * segment + 2].abs() + corrections[6 * segment + 5].abs();
            let penetration = segment_max_penetration[segment];
            contact_energy += 0.5 * contact_stiffness * penetration * penetration;
            let first = segment_vertices[2 * segment] as usize;
            let second = segment_vertices[2 * segment + 1] as usize;
            let dx = positions[3 * second] - positions[3 * first];
            let dy = positions[3 * second + 1] - positions[3 * first + 1];
            let dz = positions[3 * second + 2] - positions[3 * first + 2];
            let length = (dx * dx + dy * dy + dz * dz).sqrt();
            let rest = segment_rest_lengths[segment];
            if rest > 1.0e-7_f32 {
                let extension = length - rest;
                stretch_energy += 0.5 * stretch_stiffness * extension * extension / rest;
            }
        }
    }
    let mut bending_energy = 0.0_f32;
    for vertex in 0..curvature_ratio.len() {
        let excess = (curvature_ratio[vertex] - 1.0).max(0.0);
        bending_energy += 0.5 * bending_stiffness * excess * excess;
    }
    let length_x = cell_upper[0] - cell_lower[0];
    let length_y = cell_upper[1] - cell_lower[1];
    let length_z = cell_upper[2] - cell_lower[2];
    let mut lower_x = 0.0_f32;
    let mut upper_x = 0.0_f32;
    let mut lower_y = 0.0_f32;
    let mut upper_y = 0.0_f32;
    let mut lower_z = 0.0_f32;
    let mut upper_z = 0.0_f32;
    for vertex in 0..wall_reactions.len() / 3 {
        let rx = wall_reactions[3 * vertex];
        let ry = wall_reactions[3 * vertex + 1];
        let rz = wall_reactions[3 * vertex + 2];
        lower_x += (-rx).max(0.0);
        upper_x += rx.max(0.0);
        lower_y += (-ry).max(0.0);
        upper_y += ry.max(0.0);
        lower_z += (-rz).max(0.0);
        upper_z += rz.max(0.0);
    }
    let inverse_projection = 1.0 / correction_fraction.max(1.0e-7_f32);
    let pressure_load_x = if cell_periodic[0] != 0 {
        load_x
    } else {
        lower_x + upper_x
    };
    let pressure_load_y = if cell_periodic[1] != 0 {
        load_y
    } else {
        lower_y + upper_y
    };
    let pressure_load_z = if cell_periodic[2] != 0 {
        load_z
    } else {
        lower_z + upper_z
    };
    output[0] = contact_stiffness * pressure_load_x * inverse_projection / (length_y * length_z);
    output[1] = contact_stiffness * pressure_load_y * inverse_projection / (length_x * length_z);
    output[2] = contact_stiffness * pressure_load_z * inverse_projection / (length_x * length_y);
    output[3] = contact_energy;
    output[4] = stretch_energy;
    output[5] = bending_energy;
    output[6] = contact_energy + stretch_energy + bending_energy;
    output[7] = contact_stiffness * lower_x * inverse_projection / (length_y * length_z);
    output[8] = contact_stiffness * upper_x * inverse_projection / (length_y * length_z);
    output[9] = contact_stiffness * lower_y * inverse_projection / (length_x * length_z);
    output[10] = contact_stiffness * upper_y * inverse_projection / (length_x * length_z);
    output[11] = contact_stiffness * lower_z * inverse_projection / (length_x * length_y);
    output[12] = contact_stiffness * upper_z * inverse_projection / (length_x * length_y);
}

#[cube(launch_unchecked)]
pub(crate) fn measure_curvature_ratio(
    positions: &[f32],
    vertex_max_curvature: &[f32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    vertex_active: &[u32],
    vertex_segments: &[u32],
    segment_vertices: &[u32],
    control: &[u32],
    curvature_ratio: &mut [f32],
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[1].load() as usize || control[0] == 0 {
        terminate!();
    }
    let index = active_vertices[work as usize] as usize;
    if vertex_active[index] == 0 {
        curvature_ratio[index] = 0.0;
        terminate!();
    }
    let maximum = vertex_max_curvature[index];
    let mut ratio = 0.0_f32;
    let previous_segment = vertex_segments[2 * index];
    let next_segment = vertex_segments[2 * index + 1];
    if maximum > 0.0 && previous_segment != u32::MAX && next_segment != u32::MAX {
        let previous_segment_index = previous_segment as usize;
        let next_segment_index = next_segment as usize;
        let previous_a = segment_vertices[2 * previous_segment_index] as usize;
        let previous_b = segment_vertices[2 * previous_segment_index + 1] as usize;
        let previous = if previous_a == index {
            previous_b
        } else {
            previous_a
        };
        let next_a = segment_vertices[2 * next_segment_index] as usize;
        let next_b = segment_vertices[2 * next_segment_index + 1] as usize;
        let next = if next_a == index { next_b } else { next_a };
        let ax = positions[3 * index] - positions[3 * previous];
        let ay = positions[3 * index + 1] - positions[3 * previous + 1];
        let az = positions[3 * index + 2] - positions[3 * previous + 2];
        let bx = positions[3 * next] - positions[3 * index];
        let by = positions[3 * next + 1] - positions[3 * index + 1];
        let bz = positions[3 * next + 2] - positions[3 * index + 2];
        let first_length = (ax * ax + ay * ay + az * az).sqrt();
        let second_length = (bx * bx + by * by + bz * bz).sqrt();
        if first_length > 1.0e-7_f32 && second_length > 1.0e-7_f32 {
            let mut cosine = (ax * bx + ay * by + az * bz) / (first_length * second_length);
            cosine = cosine.clamp(-1.0, 1.0);
            let dual_length = 0.5 * (first_length + second_length);
            let curvature = 2.0 * (0.5 * (1.0 - cosine)).sqrt() / dual_length;
            ratio = curvature / maximum;
        }
    }
    curvature_ratio[index] = ratio;
}

#[cube(launch_unchecked)]
pub(crate) fn assess_contacts(
    segment_max_penetration: &[f32],
    curvature_ratio: &[f32],
    cell_overflow: &[Atomic<u32>],
    control: &mut [u32],
    metrics: &mut [f32],
    penetration_tolerance: f32,
    curvature_ratio_tolerance: f32,
    max_iterations: u32,
    force_full_iterations: u32,
) {
    if ABSOLUTE_POS != 0 || control[0] == 0 {
        terminate!();
    }
    let mut maximum = 0.0_f32;
    for segment in 0..segment_max_penetration.len() {
        if segment_max_penetration[segment] > maximum {
            maximum = segment_max_penetration[segment];
        }
    }
    metrics[0] = maximum;
    let mut maximum_curvature_ratio = 0.0_f32;
    for vertex in 0..curvature_ratio.len() {
        if curvature_ratio[vertex] > maximum_curvature_ratio {
            maximum_curvature_ratio = curvature_ratio[vertex];
        }
    }
    metrics[2] = maximum_curvature_ratio;
    if cell_overflow[0].load() != 0
        || (force_full_iterations == 0
            && maximum <= penetration_tolerance
            && maximum_curvature_ratio <= 1.0 + curvature_ratio_tolerance)
        || control[2] >= max_iterations
    {
        if cell_overflow[0].load() != 0 {
            control[3] = 1;
        }
        control[0] = 0;
    } else {
        control[1] += 1;
        control[2] += 1;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn find_internal_corrections(
    positions: &[f32],
    intrinsic_positions: &[f32],
    segment_vertices: &[u32],
    segment_rest_lengths: &[f32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    vertex_segments: &[u32],
    control: &[u32],
    internal_corrections: &mut [f32],
    stretch_stiffness: f32,
    bend_stiffness: f32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[1].load() as usize || control[0] == 0 {
        terminate!();
    }
    let index = active_vertices[work as usize] as usize;
    if vertex_active[index] == 0 || vertex_pinned[index] != 0 {
        internal_corrections[3 * index] = 0.0;
        internal_corrections[3 * index + 1] = 0.0;
        internal_corrections[3 * index + 2] = 0.0;
        terminate!();
    }
    let x = positions[3 * index];
    let y = positions[3 * index + 1];
    let z = positions[3 * index + 2];
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    let mut cz = 0.0_f32;

    let previous_segment = vertex_segments[2 * index];
    let next_segment = vertex_segments[2 * index + 1];

    if previous_segment != u32::MAX {
        let segment = previous_segment as usize;
        let first = segment_vertices[2 * segment] as usize;
        let second = segment_vertices[2 * segment + 1] as usize;
        let other = if first == index { second } else { first };
        let dx = positions[3 * other] - x;
        let dy = positions[3 * other + 1] - y;
        let dz = positions[3 * other + 2] - z;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();
        if length > 1.0e-7_f32 {
            let magnitude =
                0.5 * stretch_stiffness * (length - segment_rest_lengths[segment]) / length;
            cx += dx * magnitude;
            cy += dy * magnitude;
            cz += dz * magnitude;
        }
    }
    if next_segment != u32::MAX {
        let segment = next_segment as usize;
        let first = segment_vertices[2 * segment] as usize;
        let second = segment_vertices[2 * segment + 1] as usize;
        let other = if first == index { second } else { first };
        let dx = positions[3 * other] - x;
        let dy = positions[3 * other + 1] - y;
        let dz = positions[3 * other + 2] - z;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();
        if length > 1.0e-7_f32 {
            let magnitude =
                0.5 * stretch_stiffness * (length - segment_rest_lengths[segment]) / length;
            cx += dx * magnitude;
            cy += dy * magnitude;
            cz += dz * magnitude;
        }
    }
    if previous_segment != u32::MAX {
        let adjacent = previous_segment as usize;
        let first = segment_vertices[2 * adjacent] as usize;
        let second = segment_vertices[2 * adjacent + 1] as usize;
        let middle = if first == index { second } else { first };
        let outer_segment = vertex_segments[2 * middle];
        if outer_segment != u32::MAX {
            let outer_index = outer_segment as usize;
            let outer_first = segment_vertices[2 * outer_index] as usize;
            let outer_second = segment_vertices[2 * outer_index + 1] as usize;
            let other = if outer_first == middle {
                outer_second
            } else {
                outer_first
            };
            let dx = positions[3 * other] - x;
            let dy = positions[3 * other + 1] - y;
            let dz = positions[3 * other + 2] - z;
            let length = (dx * dx + dy * dy + dz * dz).sqrt();
            if length > 1.0e-7_f32 {
                let rx = intrinsic_positions[3 * other] - intrinsic_positions[3 * index];
                let ry = intrinsic_positions[3 * other + 1] - intrinsic_positions[3 * index + 1];
                let rz = intrinsic_positions[3 * other + 2] - intrinsic_positions[3 * index + 2];
                let rest_chord = (rx * rx + ry * ry + rz * rz).sqrt();
                let magnitude = 0.5 * bend_stiffness * (length - rest_chord) / length;
                cx += dx * magnitude;
                cy += dy * magnitude;
                cz += dz * magnitude;
            }
        }
    }
    if next_segment != u32::MAX {
        let adjacent = next_segment as usize;
        let first = segment_vertices[2 * adjacent] as usize;
        let second = segment_vertices[2 * adjacent + 1] as usize;
        let middle = if first == index { second } else { first };
        let outer_segment = vertex_segments[2 * middle + 1];
        if outer_segment != u32::MAX {
            let outer_index = outer_segment as usize;
            let outer_first = segment_vertices[2 * outer_index] as usize;
            let outer_second = segment_vertices[2 * outer_index + 1] as usize;
            let other = if outer_first == middle {
                outer_second
            } else {
                outer_first
            };
            let dx = positions[3 * other] - x;
            let dy = positions[3 * other + 1] - y;
            let dz = positions[3 * other + 2] - z;
            let length = (dx * dx + dy * dy + dz * dz).sqrt();
            if length > 1.0e-7_f32 {
                let rx = intrinsic_positions[3 * other] - intrinsic_positions[3 * index];
                let ry = intrinsic_positions[3 * other + 1] - intrinsic_positions[3 * index + 1];
                let rz = intrinsic_positions[3 * other + 2] - intrinsic_positions[3 * index + 2];
                let rest_chord = (rx * rx + ry * ry + rz * rz).sqrt();
                let magnitude = 0.5 * bend_stiffness * (length - rest_chord) / length;
                cx += dx * magnitude;
                cy += dy * magnitude;
                cz += dz * magnitude;
            }
        }
    }
    internal_corrections[3 * index] = cx;
    internal_corrections[3 * index + 1] = cy;
    internal_corrections[3 * index + 2] = cz;
}

/// Computes one three-point bend-limit projection per internal vertex.
///
/// The nine outputs are corrections for the previous, center, and next
/// vertices. Pinned vertices have zero inverse mass, so the remaining free
/// vertices receive the complete projection.
#[cube(launch_unchecked)]
pub(crate) fn find_curvature_limit_corrections(
    positions: &[f32],
    segment_vertices: &[u32],
    vertex_max_curvature: &[f32],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    vertex_segments: &[u32],
    control: &[u32],
    bend_corrections: &mut [f32],
    stiffness: f32,
    safety_margin: f32,
) {
    let vertex = ABSOLUTE_POS;
    if vertex >= vertex_active.len() || control[0] == 0 {
        terminate!();
    }
    let center = vertex as usize;
    for component in 0..9 {
        bend_corrections[9 * center + component] = 0.0;
    }
    if vertex_active[center] == 0 {
        terminate!();
    }
    let maximum_curvature = vertex_max_curvature[center] * (1.0 - safety_margin);
    let previous_segment = vertex_segments[2 * center];
    let next_segment = vertex_segments[2 * center + 1];
    if maximum_curvature <= 0.0 || previous_segment == u32::MAX || next_segment == u32::MAX {
        terminate!();
    }

    let previous_segment_index = previous_segment as usize;
    let previous_a = segment_vertices[2 * previous_segment_index] as usize;
    let previous_b = segment_vertices[2 * previous_segment_index + 1] as usize;
    let previous = if previous_a == center {
        previous_b
    } else {
        previous_a
    };
    let next_segment_index = next_segment as usize;
    let next_a = segment_vertices[2 * next_segment_index] as usize;
    let next_b = segment_vertices[2 * next_segment_index + 1] as usize;
    let next = if next_a == center { next_b } else { next_a };

    let ux = positions[3 * center] - positions[3 * previous];
    let uy = positions[3 * center + 1] - positions[3 * previous + 1];
    let uz = positions[3 * center + 2] - positions[3 * previous + 2];
    let vx = positions[3 * next] - positions[3 * center];
    let vy = positions[3 * next + 1] - positions[3 * center + 1];
    let vz = positions[3 * next + 2] - positions[3 * center + 2];
    let first_length = (ux * ux + uy * uy + uz * uz).sqrt();
    let second_length = (vx * vx + vy * vy + vz * vz).sqrt();
    if first_length <= 1.0e-7_f32 || second_length <= 1.0e-7_f32 {
        terminate!();
    }

    let unx = ux / first_length;
    let uny = uy / first_length;
    let unz = uz / first_length;
    let vnx = vx / second_length;
    let vny = vy / second_length;
    let vnz = vz / second_length;
    let cosine = (unx * vnx + uny * vny + unz * vnz).clamp(-1.0, 1.0);
    let sine_half = 0.25 * maximum_curvature * (first_length + second_length);
    if sine_half >= 1.0 {
        terminate!();
    }
    let minimum_cosine = 1.0 - 2.0 * sine_half * sine_half;
    let violation = minimum_cosine - cosine;
    if violation <= 0.0 {
        terminate!();
    }

    // Derivatives of dot(normalize(u), normalize(v)) with respect to u and v.
    let gux = (vnx - cosine * unx) / first_length;
    let guy = (vny - cosine * uny) / first_length;
    let guz = (vnz - cosine * unz) / first_length;
    let gvx = (unx - cosine * vnx) / second_length;
    let gvy = (uny - cosine * vny) / second_length;
    let gvz = (unz - cosine * vnz) / second_length;
    let center_gx = gux - gvx;
    let center_gy = guy - gvy;
    let center_gz = guz - gvz;
    let mut previous_weight = 1.0_f32;
    let mut center_weight = 1.0_f32;
    let mut next_weight = 1.0_f32;
    if vertex_pinned[previous] != 0 {
        previous_weight = 0.0;
    }
    if vertex_pinned[center] != 0 {
        center_weight = 0.0;
    }
    if vertex_pinned[next] != 0 {
        next_weight = 0.0;
    }
    let denominator = previous_weight * (gux * gux + guy * guy + guz * guz)
        + center_weight * (center_gx * center_gx + center_gy * center_gy + center_gz * center_gz)
        + next_weight * (gvx * gvx + gvy * gvy + gvz * gvz);
    if denominator <= 1.0e-12_f32 {
        terminate!();
    }
    let scale = stiffness * violation / denominator;
    bend_corrections[9 * center] = -previous_weight * scale * gux;
    bend_corrections[9 * center + 1] = -previous_weight * scale * guy;
    bend_corrections[9 * center + 2] = -previous_weight * scale * guz;
    bend_corrections[9 * center + 3] = center_weight * scale * center_gx;
    bend_corrections[9 * center + 4] = center_weight * scale * center_gy;
    bend_corrections[9 * center + 5] = center_weight * scale * center_gz;
    bend_corrections[9 * center + 6] = next_weight * scale * gvx;
    bend_corrections[9 * center + 7] = next_weight * scale * gvy;
    bend_corrections[9 * center + 8] = next_weight * scale * gvz;
}

#[cube]
fn project_curvature_triplet_in_place(
    positions: &mut [f32],
    vertex_max_curvature: &[f32],
    vertex_pinned: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    previous: usize,
    center: usize,
    next: usize,
    radius: f32,
    stiffness: f32,
    safety_margin: f32,
) {
    let maximum_curvature = vertex_max_curvature[center] * (1.0 - safety_margin);
    if maximum_curvature > 0.0 {
        let ux = positions[3 * center] - positions[3 * previous];
        let uy = positions[3 * center + 1] - positions[3 * previous + 1];
        let uz = positions[3 * center + 2] - positions[3 * previous + 2];
        let vx = positions[3 * next] - positions[3 * center];
        let vy = positions[3 * next + 1] - positions[3 * center + 1];
        let vz = positions[3 * next + 2] - positions[3 * center + 2];
        let first_length = (ux * ux + uy * uy + uz * uz).sqrt();
        let second_length = (vx * vx + vy * vy + vz * vz).sqrt();
        if first_length > 1.0e-7_f32 && second_length > 1.0e-7_f32 {
            let unx = ux / first_length;
            let uny = uy / first_length;
            let unz = uz / first_length;
            let vnx = vx / second_length;
            let vny = vy / second_length;
            let vnz = vz / second_length;
            let cosine = (unx * vnx + uny * vny + unz * vnz).clamp(-1.0, 1.0);
            let sine_half = 0.25 * maximum_curvature * (first_length + second_length);
            if sine_half < 1.0 {
                let minimum_cosine = 1.0 - 2.0 * sine_half * sine_half;
                let violation = minimum_cosine - cosine;
                if violation > 0.0 {
                    let gux = (vnx - cosine * unx) / first_length;
                    let guy = (vny - cosine * uny) / first_length;
                    let guz = (vnz - cosine * unz) / first_length;
                    let gvx = (unx - cosine * vnx) / second_length;
                    let gvy = (uny - cosine * vny) / second_length;
                    let gvz = (unz - cosine * vnz) / second_length;
                    let center_gx = gux - gvx;
                    let center_gy = guy - gvy;
                    let center_gz = guz - gvz;
                    let mut previous_weight = 1.0_f32;
                    let mut center_weight = 1.0_f32;
                    let mut next_weight = 1.0_f32;
                    if vertex_pinned[previous] != 0 {
                        previous_weight = 0.0;
                    }
                    if vertex_pinned[center] != 0 {
                        center_weight = 0.0;
                    }
                    if vertex_pinned[next] != 0 {
                        next_weight = 0.0;
                    }
                    let denominator = previous_weight * (gux * gux + guy * guy + guz * guz)
                        + center_weight
                            * (center_gx * center_gx
                                + center_gy * center_gy
                                + center_gz * center_gz)
                        + next_weight * (gvx * gvx + gvy * gvy + gvz * gvz);
                    if denominator > 1.0e-12_f32 {
                        let scale = stiffness * violation / denominator;
                        for axis in 0..3 {
                            let previous_gradient = if axis == 0 {
                                gux
                            } else if axis == 1 {
                                guy
                            } else {
                                guz
                            };
                            let center_gradient = if axis == 0 {
                                center_gx
                            } else if axis == 1 {
                                center_gy
                            } else {
                                center_gz
                            };
                            let next_gradient = if axis == 0 {
                                gvx
                            } else if axis == 1 {
                                gvy
                            } else {
                                gvz
                            };
                            let previous_coordinate = 3 * previous + axis;
                            let center_coordinate = 3 * center + axis;
                            let next_coordinate = 3 * next + axis;
                            let mut previous_value = positions[previous_coordinate]
                                - previous_weight * scale * previous_gradient;
                            let mut center_value = positions[center_coordinate]
                                + center_weight * scale * center_gradient;
                            let mut next_value =
                                positions[next_coordinate] + next_weight * scale * next_gradient;
                            if cell_periodic[axis] == 0 {
                                previous_value = previous_value
                                    .clamp(cell_lower[axis] + radius, cell_upper[axis] - radius);
                                center_value = center_value
                                    .clamp(cell_lower[axis] + radius, cell_upper[axis] - radius);
                                next_value = next_value
                                    .clamp(cell_lower[axis] + radius, cell_upper[axis] - radius);
                            }
                            if previous_weight > 0.0 {
                                positions[previous_coordinate] = previous_value;
                            }
                            if center_weight > 0.0 {
                                positions[center_coordinate] = center_value;
                            }
                            if next_weight > 0.0 {
                                positions[next_coordinate] = next_value;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Performs nonlinear forward/backward curvature projections independently
/// along every fiber. Fibers own disjoint position ranges, so in-place updates
/// are race-free while adjacent constraints observe each other's corrections.
#[cube(launch_unchecked)]
pub(crate) fn project_fiber_curvature_in_place(
    positions: &mut [f32],
    fiber_segment_spans: &[u32],
    fiber_vertex_spans: &[u32],
    segment_radii: &[f32],
    vertex_max_curvature: &[f32],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
    sweeps: u32,
    stiffness: f32,
    safety_margin: f32,
) {
    let fiber = ABSOLUTE_POS;
    if fiber >= fiber_vertex_spans.len() / 2 || control[0] == 0 {
        terminate!();
    }
    let fiber_index = fiber as usize;
    let start = fiber_vertex_spans[2 * fiber_index] as usize;
    let count = fiber_vertex_spans[2 * fiber_index + 1] as usize;
    if count < 3 || vertex_active[start] == 0 {
        terminate!();
    }
    let segment_start = fiber_segment_spans[2 * fiber_index] as usize;
    let radius = segment_radii[segment_start];
    for _ in 0..sweeps {
        let mut previous = start;
        let mut center = start + 1;
        while center < start + count && vertex_active[center] == 0 {
            center += 1;
        }
        while center < start + count {
            let mut next = center + 1;
            while next < start + count && vertex_active[next] == 0 {
                next += 1;
            }
            if next >= start + count {
                break;
            }
            project_curvature_triplet_in_place(
                positions,
                vertex_max_curvature,
                vertex_pinned,
                cell_lower,
                cell_upper,
                cell_periodic,
                previous,
                center,
                next,
                radius,
                stiffness,
                safety_margin,
            );
            previous = center;
            center = next;
        }

        let mut next = start + count - 1;
        let mut center_index = next as i32 - 1;
        while center_index >= start as i32 && vertex_active[center_index as usize] == 0 {
            center_index -= 1;
        }
        while center_index >= start as i32 {
            let center_vertex = center_index as usize;
            let mut previous_index = center_index - 1;
            while previous_index >= start as i32 && vertex_active[previous_index as usize] == 0 {
                previous_index -= 1;
            }
            if previous_index < start as i32 {
                break;
            }
            let previous_vertex = previous_index as usize;
            project_curvature_triplet_in_place(
                positions,
                vertex_max_curvature,
                vertex_pinned,
                cell_lower,
                cell_upper,
                cell_periodic,
                previous_vertex,
                center_vertex,
                next,
                radius,
                stiffness,
                safety_margin,
            );
            next = center_vertex;
            center_index = previous_index;
        }
    }
}

/// Gathers the at-most-three bend constraints incident to each vertex.
#[cube(launch_unchecked)]
pub(crate) fn gather_curvature_limit_corrections(
    segment_vertices: &[u32],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    vertex_segments: &[u32],
    control: &[u32],
    bend_corrections: &[f32],
    vertex_corrections: &mut [f32],
) {
    let vertex = ABSOLUTE_POS;
    if vertex >= vertex_active.len() || control[0] == 0 {
        terminate!();
    }
    let index = vertex as usize;
    if vertex_active[index] == 0 || vertex_pinned[index] != 0 {
        vertex_corrections[3 * index] = 0.0;
        vertex_corrections[3 * index + 1] = 0.0;
        vertex_corrections[3 * index + 2] = 0.0;
        terminate!();
    }

    let mut cx = bend_corrections[9 * index + 3];
    let mut cy = bend_corrections[9 * index + 4];
    let mut cz = bend_corrections[9 * index + 5];
    let previous_segment = vertex_segments[2 * index];
    if previous_segment != u32::MAX {
        let segment = previous_segment as usize;
        let first = segment_vertices[2 * segment] as usize;
        let second = segment_vertices[2 * segment + 1] as usize;
        let previous = if first == index { second } else { first };
        cx += bend_corrections[9 * previous + 6];
        cy += bend_corrections[9 * previous + 7];
        cz += bend_corrections[9 * previous + 8];
    }
    let next_segment = vertex_segments[2 * index + 1];
    if next_segment != u32::MAX {
        let segment = next_segment as usize;
        let first = segment_vertices[2 * segment] as usize;
        let second = segment_vertices[2 * segment + 1] as usize;
        let next = if first == index { second } else { first };
        cx += bend_corrections[9 * next];
        cy += bend_corrections[9 * next + 1];
        cz += bend_corrections[9 * next + 2];
    }
    vertex_corrections[3 * index] = cx;
    vertex_corrections[3 * index + 1] = cy;
    vertex_corrections[3 * index + 2] = cz;
}

#[cube(launch_unchecked)]
pub(crate) fn apply_internal_corrections(
    positions: &mut [f32],
    internal_corrections: &[f32],
    segment_radii: &[f32],
    vertex_fibers: &[u32],
    fiber_segment_spans: &[u32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    wall_reactions: &mut [f32],
    control: &[u32],
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[1].load() as usize || control[0] == 0 {
        terminate!();
    }
    let index = active_vertices[work as usize] as usize;
    if vertex_active[index] == 0 || vertex_pinned[index] != 0 {
        terminate!();
    }
    let fiber = vertex_fibers[index] as usize;
    let segment_start = fiber_segment_spans[2 * fiber] as usize;
    let radius = segment_radii[segment_start];
    for axis in 0..3 {
        let coordinate = 3 * index + axis;
        let moved = positions[coordinate] + internal_corrections[coordinate];
        positions[coordinate] = if cell_periodic[axis] != 0 {
            moved
        } else {
            moved.clamp(cell_lower[axis] + radius, cell_upper[axis] - radius)
        };
        if cell_periodic[axis] == 0 {
            wall_reactions[coordinate] += moved - positions[coordinate];
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn apply_contact_corrections(
    positions: &mut [f32],
    corrections: &[f32],
    vertex_segments: &[u32],
    segment_radii: &[f32],
    vertex_fibers: &[u32],
    fiber_segment_spans: &[u32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    wall_reactions: &mut [f32],
    control: &[u32],
    vertex_step: &mut [f32],
    max_step: f32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[1].load() as usize || control[0] == 0 {
        terminate!();
    }
    let index = active_vertices[work as usize] as usize;
    if vertex_active[index] == 0 || vertex_pinned[index] != 0 {
        vertex_step[index] = 0.0;
        terminate!();
    }
    let previous = vertex_segments[2 * index];
    let next = vertex_segments[2 * index + 1];
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    let mut cz = 0.0_f32;
    if previous != u32::MAX {
        let segment = previous as usize;
        cx += corrections[6 * segment + 3];
        cy += corrections[6 * segment + 4];
        cz += corrections[6 * segment + 5];
    }
    if next != u32::MAX {
        let segment = next as usize;
        cx += corrections[6 * segment];
        cy += corrections[6 * segment + 1];
        cz += corrections[6 * segment + 2];
    }
    let length = (cx * cx + cy * cy + cz * cz).sqrt();
    if length > max_step {
        let scale = max_step / length;
        cx *= scale;
        cy *= scale;
        cz *= scale;
    }
    let fiber = vertex_fibers[index] as usize;
    let segment_start = fiber_segment_spans[2 * fiber] as usize;
    let radius = segment_radii[segment_start];
    let moved_x = positions[3 * index] + cx;
    let moved_y = positions[3 * index + 1] + cy;
    let moved_z = positions[3 * index + 2] + cz;
    positions[3 * index] = if cell_periodic[0] != 0 {
        moved_x
    } else {
        moved_x.clamp(cell_lower[0] + radius, cell_upper[0] - radius)
    };
    if cell_periodic[0] == 0 {
        wall_reactions[3 * index] += moved_x - positions[3 * index];
    }
    positions[3 * index + 1] = if cell_periodic[1] != 0 {
        moved_y
    } else {
        moved_y.clamp(cell_lower[1] + radius, cell_upper[1] - radius)
    };
    if cell_periodic[1] == 0 {
        wall_reactions[3 * index + 1] += moved_y - positions[3 * index + 1];
    }
    positions[3 * index + 2] = if cell_periodic[2] != 0 {
        moved_z
    } else {
        moved_z.clamp(cell_lower[2] + radius, cell_upper[2] - radius)
    };
    if cell_periodic[2] == 0 {
        wall_reactions[3 * index + 2] += moved_z - positions[3 * index + 2];
    }
    vertex_step[index] = (cx * cx + cy * cy + cz * cz).sqrt();
}

#[cube(launch_unchecked)]
pub(crate) fn apply_rigid_contact_corrections(
    positions: &mut [f32],
    corrections: &[f32],
    fiber_segment_spans: &[u32],
    fiber_vertex_spans: &[u32],
    segment_radii: &[f32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    wall_reactions: &mut [f32],
    control: &[u32],
    vertex_step: &mut [f32],
    max_step: f32,
) {
    let fiber = ABSOLUTE_POS;
    if fiber >= fiber_vertex_spans.len() / 2 || control[0] == 0 {
        terminate!();
    }
    let fiber_index = fiber as usize;
    let segment_start = fiber_segment_spans[2 * fiber_index] as usize;
    let segment_count = fiber_segment_spans[2 * fiber_index + 1] as usize;
    let vertex_start = fiber_vertex_spans[2 * fiber_index] as usize;
    let vertex_count = fiber_vertex_spans[2 * fiber_index + 1] as usize;
    let mut tx = 0.0_f32;
    let mut ty = 0.0_f32;
    let mut tz = 0.0_f32;
    let mut deepest_squared = 0.0_f32;
    for local in 0..segment_count {
        let segment = segment_start + local;
        let cx = 0.5 * (corrections[6 * segment] + corrections[6 * segment + 3]);
        let cy = 0.5 * (corrections[6 * segment + 1] + corrections[6 * segment + 4]);
        let cz = 0.5 * (corrections[6 * segment + 2] + corrections[6 * segment + 5]);
        let squared = cx * cx + cy * cy + cz * cz;
        if squared > deepest_squared {
            deepest_squared = squared;
            tx = cx;
            ty = cy;
            tz = cz;
        }
    }
    // Use the fiber's deepest segment contact. Averaging can cancel contacts
    // from opposite sides of a long fiber even though its maximum penetration
    // remains inadmissible; deepest-first translation is a deterministic
    // coordinate-descent step on the hard acceptance metric.
    let length = (tx * tx + ty * ty + tz * tz).sqrt();
    if length > max_step {
        let scale = max_step / length;
        tx *= scale;
        ty *= scale;
        tz *= scale;
    }
    let requested_tx = tx;
    let requested_ty = ty;
    let requested_tz = tz;

    let radius = segment_radii[segment_start];
    let mut minimum_x = positions[3 * vertex_start];
    let mut maximum_x = minimum_x;
    let mut minimum_y = positions[3 * vertex_start + 1];
    let mut maximum_y = minimum_y;
    let mut minimum_z = positions[3 * vertex_start + 2];
    let mut maximum_z = minimum_z;
    for local in 1..vertex_count {
        let vertex = vertex_start + local;
        minimum_x = minimum_x.min(positions[3 * vertex]);
        maximum_x = maximum_x.max(positions[3 * vertex]);
        minimum_y = minimum_y.min(positions[3 * vertex + 1]);
        maximum_y = maximum_y.max(positions[3 * vertex + 1]);
        minimum_z = minimum_z.min(positions[3 * vertex + 2]);
        maximum_z = maximum_z.max(positions[3 * vertex + 2]);
    }
    if cell_periodic[0] == 0 {
        tx = tx.clamp(
            cell_lower[0] + radius - minimum_x,
            cell_upper[0] - radius - maximum_x,
        );
    }
    if cell_periodic[1] == 0 {
        ty = ty.clamp(
            cell_lower[1] + radius - minimum_y,
            cell_upper[1] - radius - maximum_y,
        );
    }
    if cell_periodic[2] == 0 {
        tz = tz.clamp(
            cell_lower[2] + radius - minimum_z,
            cell_upper[2] - radius - maximum_z,
        );
    }
    let applied = (tx * tx + ty * ty + tz * tz).sqrt();
    let reaction_x = (requested_tx - tx) / vertex_count as f32;
    let reaction_y = (requested_ty - ty) / vertex_count as f32;
    let reaction_z = (requested_tz - tz) / vertex_count as f32;
    for local in 0..vertex_count {
        let vertex = vertex_start + local;
        positions[3 * vertex] += tx;
        positions[3 * vertex + 1] += ty;
        positions[3 * vertex + 2] += tz;
        vertex_step[vertex] = applied;
        wall_reactions[3 * vertex] += reaction_x;
        wall_reactions[3 * vertex + 1] += reaction_y;
        wall_reactions[3 * vertex + 2] += reaction_z;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn measure_step(vertex_step: &[f32], control: &[u32], metrics: &mut [f32]) {
    if ABSOLUTE_POS != 0 || control[0] == 0 {
        terminate!();
    }
    let mut maximum = 0.0_f32;
    for vertex in 0..vertex_step.len() {
        if vertex_step[vertex] > maximum {
            maximum = vertex_step[vertex];
        }
    }
    metrics[1] = maximum;
}
