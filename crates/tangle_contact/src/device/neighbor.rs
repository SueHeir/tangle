//! Verlet neighbor lists for capsule contacts.
//!
//! A list stores, for every active segment, the segments whose capsules were
//! within `skin` of touching when the list was built. Because the distance
//! between two segments changes by at most twice the largest endpoint
//! displacement, the list contains every contact until some vertex has moved
//! more than half the skin since the build. `neighbor_state` carries that
//! rebuild request between kernels: entry 0 is the pending-rebuild flag and
//! entry 1 counts completed rebuilds.

#![allow(missing_docs)]

use cubecl::prelude::*;

/// Distance between two segment axes `p1 + s d1` and `p2 + t d2`.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn segment_axis_distance(
    p1x: f32,
    p1y: f32,
    p1z: f32,
    d1x: f32,
    d1y: f32,
    d1z: f32,
    p2x: f32,
    p2y: f32,
    p2z: f32,
    d2x: f32,
    d2y: f32,
    d2z: f32,
) -> f32 {
    let length_epsilon = 1.0e-20_f32;
    let parallel_relative_epsilon = 1.0e-6_f32;
    let rx = p1x - p2x;
    let ry = p1y - p2y;
    let rz = p1z - p2z;
    let a = d1x * d1x + d1y * d1y + d1z * d1z;
    let e = d2x * d2x + d2y * d2y + d2z * d2z;
    let f = d2x * rx + d2y * ry + d2z * rz;
    let mut s = 0.0_f32;
    let mut t = 0.0_f32;
    if a <= length_epsilon && e > length_epsilon {
        t = (f / e).clamp(0.0, 1.0);
    } else if a > length_epsilon {
        let c = d1x * rx + d1y * ry + d1z * rz;
        if e <= length_epsilon {
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1x * d2x + d1y * d2y + d1z * d2z;
            let denominator = a * e - b * b;
            if denominator.abs() > parallel_relative_epsilon * a * e {
                s = ((b * f - c * e) / denominator).clamp(0.0, 1.0);
            }
            let projected = (b * s + f) / e;
            if projected < 0.0 {
                s = (-c / a).clamp(0.0, 1.0);
            } else if projected > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = projected;
            }
        }
    }
    let delta_x = (p2x + d2x * t) - (p1x + d1x * s);
    let delta_y = (p2y + d2y * t) - (p1y + d1y * s);
    let delta_z = (p2z + d2z * t) - (p1z + d1z * s);
    (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt()
}

/// Builds each active segment's neighbor list from the current cell list.
///
/// Segments with more than `capacity` neighbors store their true count, which
/// tells the contact kernel to fall back to the cell-list traversal around the
/// recorded home cell for that segment.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn build_segment_neighbor_lists(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_fibers: &[u32],
    segment_radii: &[f32],
    active_segments: &[u32],
    active_counts: &[Atomic<u32>],
    cell_counts: &[u32],
    cell_offsets: &[u32],
    cell_segments: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    neighbor_state: &[u32],
    neighbor_counts: &mut [u32],
    neighbor_segments: &mut [u32],
    neighbor_home_cells: &mut [u32],
    skin: f32,
    capacity: u32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[0].load() as usize || neighbor_state[0] == 0 {
        terminate!();
    }
    let segment = active_segments[work as usize];
    let segment_index = segment as usize;
    let first_vertex = segment_vertices[2 * segment_index] as usize;
    let second_vertex = segment_vertices[2 * segment_index + 1] as usize;
    let p1x = positions[3 * first_vertex];
    let p1y = positions[3 * first_vertex + 1];
    let p1z = positions[3 * first_vertex + 2];
    let q1x = positions[3 * second_vertex];
    let q1y = positions[3 * second_vertex + 1];
    let q1z = positions[3 * second_vertex + 2];
    let d1x = q1x - p1x;
    let d1y = q1y - p1y;
    let d1z = q1z - p1z;
    let owner = segment_fibers[segment_index];
    let radius = segment_radii[segment_index];
    let half_length = 0.5 * (d1x * d1x + d1y * d1y + d1z * d1z).sqrt();
    let length_x = cell_upper[0] - cell_lower[0];
    let length_y = cell_upper[1] - cell_lower[1];
    let length_z = cell_upper[2] - cell_lower[2];

    let midpoint_x = 0.5 * (p1x + q1x);
    let midpoint_y = 0.5 * (p1y + q1y);
    let midpoint_z = 0.5 * (p1z + q1z);
    let width_x = length_x / cells_x as f32;
    let width_y = length_y / cells_y as f32;
    let width_z = length_z / cells_z as f32;
    let raw_home_x = ((midpoint_x - cell_lower[0]) / width_x).floor() as i32;
    let raw_home_y = ((midpoint_y - cell_lower[1]) / width_y).floor() as i32;
    let raw_home_z = ((midpoint_z - cell_lower[2]) / width_z).floor() as i32;
    let mut home_x = raw_home_x.clamp(0, cells_x as i32 - 1) as u32;
    let mut home_y = raw_home_y.clamp(0, cells_y as i32 - 1) as u32;
    let mut home_z = raw_home_z.clamp(0, cells_z as i32 - 1) as u32;
    if cell_periodic[0] != 0 {
        home_x = ((raw_home_x % cells_x as i32 + cells_x as i32) % cells_x as i32) as u32;
    }
    if cell_periodic[1] != 0 {
        home_y = ((raw_home_y % cells_y as i32 + cells_y as i32) % cells_y as i32) as u32;
    }
    if cell_periodic[2] != 0 {
        home_z = ((raw_home_z % cells_z as i32 + cells_z as i32) % cells_z as i32) as u32;
    }
    neighbor_home_cells[segment_index] = (home_z * cells_y + home_y) * cells_x + home_x;

    let list_start = segment_index * capacity as usize;
    let mut count = 0_u32;
    for neighbor in 0..27_u32 {
        let offset_x = (neighbor % 3) as i32 - 1;
        let offset_y = ((neighbor / 3) % 3) as i32 - 1;
        let offset_z = (neighbor / 9) as i32 - 1;
        let raw_neighbor_x = home_x as i32 + offset_x;
        let raw_neighbor_y = home_y as i32 + offset_y;
        let raw_neighbor_z = home_z as i32 + offset_z;
        let valid_x =
            cell_periodic[0] != 0 || (raw_neighbor_x >= 0 && raw_neighbor_x < cells_x as i32);
        let valid_y =
            cell_periodic[1] != 0 || (raw_neighbor_y >= 0 && raw_neighbor_y < cells_y as i32);
        let valid_z =
            cell_periodic[2] != 0 || (raw_neighbor_z >= 0 && raw_neighbor_z < cells_z as i32);
        let unique_x = cell_periodic[0] == 0
            || !((cells_x == 1 && offset_x != 0)
                || (cells_x == 2
                    && ((home_x == 0 && offset_x > 0) || (home_x == 1 && offset_x < 0))));
        let unique_y = cell_periodic[1] == 0
            || !((cells_y == 1 && offset_y != 0)
                || (cells_y == 2
                    && ((home_y == 0 && offset_y > 0) || (home_y == 1 && offset_y < 0))));
        let unique_z = cell_periodic[2] == 0
            || !((cells_z == 1 && offset_z != 0)
                || (cells_z == 2
                    && ((home_z == 0 && offset_z > 0) || (home_z == 1 && offset_z < 0))));
        if valid_x && valid_y && valid_z && unique_x && unique_y && unique_z {
            let neighbor_x =
                ((raw_neighbor_x % cells_x as i32 + cells_x as i32) % cells_x as i32) as u32;
            let neighbor_y =
                ((raw_neighbor_y % cells_y as i32 + cells_y as i32) % cells_y as i32) as u32;
            let neighbor_z =
                ((raw_neighbor_z % cells_z as i32 + cells_z as i32) % cells_z as i32) as u32;
            let cell = ((neighbor_z * cells_y + neighbor_y) * cells_x + neighbor_x) as usize;
            let cell_count = cell_counts[cell];
            let start = cell_offsets[cell];
            for slot in 0..cell_count {
                let other = cell_segments[(start + slot) as usize] as usize;
                let third_vertex = segment_vertices[2 * other] as usize;
                let fourth_vertex = segment_vertices[2 * other + 1] as usize;
                let adjacent_same_fiber = segment_fibers[other] == owner
                    && (first_vertex == third_vertex
                        || first_vertex == fourth_vertex
                        || second_vertex == third_vertex
                        || second_vertex == fourth_vertex);
                if other != segment_index && !adjacent_same_fiber {
                    let mut p2x = positions[3 * third_vertex];
                    let mut p2y = positions[3 * third_vertex + 1];
                    let mut p2z = positions[3 * third_vertex + 2];
                    let mut q2x = positions[3 * fourth_vertex];
                    let mut q2y = positions[3 * fourth_vertex + 1];
                    let mut q2z = positions[3 * fourth_vertex + 2];
                    if cell_periodic[0] != 0 {
                        let image_shift =
                            (0.5 * (p2x + q2x - p1x - q1x) / length_x).round() * length_x;
                        p2x -= image_shift;
                        q2x -= image_shift;
                    }
                    if cell_periodic[1] != 0 {
                        let image_shift =
                            (0.5 * (p2y + q2y - p1y - q1y) / length_y).round() * length_y;
                        p2y -= image_shift;
                        q2y -= image_shift;
                    }
                    if cell_periodic[2] != 0 {
                        let image_shift =
                            (0.5 * (p2z + q2z - p1z - q1z) / length_z).round() * length_z;
                        p2z -= image_shift;
                        q2z -= image_shift;
                    }
                    let d2x = q2x - p2x;
                    let d2y = q2y - p2y;
                    let d2z = q2z - p2z;
                    // A small relative slack keeps rounding from dropping a
                    // pair that sits exactly on the skin boundary.
                    let interaction = (radius + segment_radii[other] + skin) * (1.0 + 1.0e-4_f32);
                    let midpoint_dx = 0.5 * (p2x + q2x - p1x - q1x);
                    let midpoint_dy = 0.5 * (p2y + q2y - p1y - q1y);
                    let midpoint_dz = 0.5 * (p2z + q2z - p1z - q1z);
                    let reach = half_length
                        + 0.5 * (d2x * d2x + d2y * d2y + d2z * d2z).sqrt()
                        + interaction;
                    if midpoint_dx * midpoint_dx
                        + midpoint_dy * midpoint_dy
                        + midpoint_dz * midpoint_dz
                        <= reach * reach
                        && segment_axis_distance(
                            p1x, p1y, p1z, d1x, d1y, d1z, p2x, p2y, p2z, d2x, d2y, d2z,
                        ) <= interaction
                    {
                        if count < capacity {
                            neighbor_segments[list_start + count as usize] = other as u32;
                        }
                        count += 1;
                    }
                }
            }
        }
    }
    neighbor_counts[segment_index] = count;
}

/// Records the positions a freshly built neighbor list is valid for.
#[cube(launch_unchecked)]
pub fn snapshot_neighbor_reference_positions(
    positions: &[f32],
    reference_positions: &mut [f32],
    neighbor_state: &[u32],
) {
    let index = ABSOLUTE_POS;
    if index >= positions.len() || neighbor_state[0] == 0 {
        terminate!();
    }
    reference_positions[index] = positions[index];
}

/// Clears a pending rebuild request once the lists and reference are current.
#[cube(launch_unchecked)]
pub fn finish_neighbor_list_rebuild(neighbor_state: &mut [u32]) {
    if ABSOLUTE_POS != 0 || neighbor_state[0] == 0 {
        terminate!();
    }
    neighbor_state[0] = 0;
    neighbor_state[1] += 1;
}

/// Requests a rebuild before the next contact pass.
#[cube(launch_unchecked)]
pub fn request_neighbor_list_rebuild(neighbor_state: &mut [u32]) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    neighbor_state[0] = 1;
}

/// Requests a rebuild once any active vertex has moved more than half the
/// skin since the lists were built.
#[cube(launch_unchecked)]
pub fn flag_neighbor_list_displacement(
    positions: &[f32],
    reference_positions: &[f32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    neighbor_state: &mut [Atomic<u32>],
    maximum_displacement: f32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[1].load() as usize {
        terminate!();
    }
    let vertex = active_vertices[work as usize] as usize;
    let dx = positions[3 * vertex] - reference_positions[3 * vertex];
    let dy = positions[3 * vertex + 1] - reference_positions[3 * vertex + 1];
    let dz = positions[3 * vertex + 2] - reference_positions[3 * vertex + 2];
    if dx * dx + dy * dy + dz * dz > maximum_displacement * maximum_displacement {
        neighbor_state[0].store(1);
    }
}
