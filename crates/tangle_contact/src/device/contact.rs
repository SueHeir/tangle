//! Exact capsule contact capture and projection kernels.

#![allow(missing_docs)]

use cubecl::prelude::*;

use super::cell_list::{proxy_cell, proxy_of};

#[cube(launch_unchecked)]
#[allow(unused_assignments)]
pub fn capture_segment_contacts(
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
    captured_count: &mut [Atomic<u32>],
    captured_segments: &mut [u32],
    captured_coordinates: &mut [f32],
    captured_surface_gaps: &mut [f32],
    captured_crossing_angles: &mut [f32],
    maximum_surface_gap: f32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[0].load() as usize {
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
    let length_epsilon = 1.0e-20_f32;
    let parallel_relative_epsilon = 1.0e-6_f32;

    let midpoint_x = 0.5 * (p1x + q1x);
    let midpoint_y = 0.5 * (p1y + q1y);
    let midpoint_z = 0.5 * (p1z + q1z);
    let width_x = (cell_upper[0] - cell_lower[0]) / cells_x as f32;
    let width_y = (cell_upper[1] - cell_lower[1]) / cells_y as f32;
    let width_z = (cell_upper[2] - cell_lower[2]) / cells_z as f32;
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
            let mut shift_x = 0.0_f32;
            let mut shift_y = 0.0_f32;
            let mut shift_z = 0.0_f32;
            if raw_neighbor_x < 0 {
                shift_x = -(cell_upper[0] - cell_lower[0]);
            } else if raw_neighbor_x >= cells_x as i32 {
                shift_x = cell_upper[0] - cell_lower[0];
            }
            if raw_neighbor_y < 0 {
                shift_y = -(cell_upper[1] - cell_lower[1]);
            } else if raw_neighbor_y >= cells_y as i32 {
                shift_y = cell_upper[1] - cell_lower[1];
            }
            if raw_neighbor_z < 0 {
                shift_z = -(cell_upper[2] - cell_lower[2]);
            } else if raw_neighbor_z >= cells_z as i32 {
                shift_z = cell_upper[2] - cell_lower[2];
            }
            let cell = ((neighbor_z * cells_y + neighbor_y) * cells_x + neighbor_x) as usize;
            let count = cell_counts[cell];
            let start = cell_offsets[cell];
            for slot in 0..count {
                let other = cell_segments[(start + slot) as usize] as usize;
                if other > segment_index && segment_fibers[other] != owner {
                    let third_vertex = segment_vertices[2 * other] as usize;
                    let fourth_vertex = segment_vertices[2 * other + 1] as usize;
                    let mut p2x = positions[3 * third_vertex] + shift_x;
                    let mut p2y = positions[3 * third_vertex + 1] + shift_y;
                    let mut p2z = positions[3 * third_vertex + 2] + shift_z;
                    let mut q2x = positions[3 * fourth_vertex] + shift_x;
                    let mut q2y = positions[3 * fourth_vertex + 1] + shift_y;
                    let mut q2z = positions[3 * fourth_vertex + 2] + shift_z;
                    let length_x = cell_upper[0] - cell_lower[0];
                    let length_y = cell_upper[1] - cell_lower[1];
                    let length_z = cell_upper[2] - cell_lower[2];
                    let image_dx = 0.5 * (p2x + q2x - p1x - q1x);
                    let image_dy = 0.5 * (p2y + q2y - p1y - q1y);
                    let image_dz = 0.5 * (p2z + q2z - p1z - q1z);
                    if cell_periodic[0] != 0 {
                        let image_shift = (image_dx / length_x).round() * length_x;
                        p2x -= image_shift;
                        q2x -= image_shift;
                    }
                    if cell_periodic[1] != 0 {
                        let image_shift = (image_dy / length_y).round() * length_y;
                        p2y -= image_shift;
                        q2y -= image_shift;
                    }
                    if cell_periodic[2] != 0 {
                        let image_shift = (image_dz / length_z).round() * length_z;
                        p2z -= image_shift;
                        q2z -= image_shift;
                    }
                    let d2x = q2x - p2x;
                    let d2y = q2y - p2y;
                    let d2z = q2z - p2z;
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
                    let center_distance =
                        (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();
                    let surface_gap = center_distance - radius - segment_radii[other];
                    if surface_gap <= maximum_surface_gap {
                        let capture = captured_count[0].fetch_add(1);
                        if capture < captured_surface_gaps.len() as u32 {
                            let capture_index = capture as usize;
                            captured_segments[2 * capture_index] = segment as u32;
                            captured_segments[2 * capture_index + 1] = other as u32;
                            captured_coordinates[2 * capture_index] = s;
                            captured_coordinates[2 * capture_index + 1] = t;
                            captured_surface_gaps[capture_index] = surface_gap;
                            let length_product = (a * e).sqrt();
                            let mut crossing_angle = 0.0_f32;
                            if length_product > length_epsilon {
                                let absolute_cosine = ((d1x * d2x + d1y * d2y + d1z * d2z)
                                    / length_product)
                                    .abs()
                                    .clamp(0.0, 1.0);
                                crossing_angle = absolute_cosine.acos();
                            }
                            captured_crossing_angles[capture_index] = crossing_angle;
                        } else {
                            captured_count[1].store(1);
                        }
                    }
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
#[allow(unused_assignments)]
pub fn find_segment_corrections(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_fibers: &[u32],
    segment_radii: &[f32],
    segment_lane_offsets: &[f32],
    directors: &[f32],
    active_segments: &[u32],
    active_counts: &[Atomic<u32>],
    cell_counts: &[u32],
    cell_offsets: &[u32],
    cell_segments: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
    neighbor_counts: &[u32],
    neighbor_segments: &[u32],
    neighbor_reference_positions: &[f32],
    segment_proxies: &[u32],
    cell_proxies: &[u32],
    corrections: &mut [f32],
    director_corrections: &mut [f32],
    segment_max_penetration: &mut [f32],
    list_offsets: &[u32],
    list_weights: &[u32],
    neighbor_capacity: u32,
    correction_fraction: f32,
    contact_aggregation: u32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[0].load() as usize || control[0] == 0 {
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
    // An oval is a row of round lanes side by side along the vertex
    // directors; `radius` bounds the whole row. Round fibers have one lane
    // at zero offset, which reduces every lane term below to the plain
    // capsule contact. Lane counts match `oval_lane_count` in tangle_relax.
    let offset1 = segment_lane_offsets[segment_index];
    let lane_radius1 = radius - offset1;
    let mut lanes1 = 1_u32;
    let mut a1x = 0.0_f32;
    let mut a1y = 0.0_f32;
    let mut a1z = 0.0_f32;
    let mut b1x = 0.0_f32;
    let mut b1y = 0.0_f32;
    let mut b1z = 0.0_f32;
    let mut ea1x = 0.0_f32;
    let mut ea1y = 0.0_f32;
    let mut ea1z = 0.0_f32;
    let mut eb1x = 0.0_f32;
    let mut eb1y = 0.0_f32;
    let mut eb1z = 0.0_f32;
    // Radius of gyration of the section: turning a vertex's director by an
    // angle moves material by this much times the angle.
    let gyration1 = 0.5 * (radius * radius + lane_radius1 * lane_radius1).sqrt();
    if offset1 > 0.0 {
        lanes1 = (2.0 * offset1 / (0.6 * lane_radius1)).ceil() as u32 + 1;
        if lanes1 < 2 {
            lanes1 = 2;
        }
        if lanes1 > 5 {
            lanes1 = 5;
        }
        a1x = directors[3 * first_vertex];
        a1y = directors[3 * first_vertex + 1];
        a1z = directors[3 * first_vertex + 2];
        b1x = directors[3 * second_vertex];
        b1y = directors[3 * second_vertex + 1];
        b1z = directors[3 * second_vertex + 2];
        let inverse_length = 1.0 / (2.0 * half_length).max(1.0e-20_f32);
        let t1x = d1x * inverse_length;
        let t1y = d1y * inverse_length;
        let t1z = d1z * inverse_length;
        ea1x = t1y * a1z - t1z * a1y;
        ea1y = t1z * a1x - t1x * a1z;
        ea1z = t1x * a1y - t1y * a1x;
        eb1x = t1y * b1z - t1z * b1y;
        eb1y = t1z * b1x - t1x * b1z;
        eb1z = t1x * b1y - t1y * b1x;
    }
    let mut twist_0 = 0.0_f32;
    let mut twist_1 = 0.0_f32;
    let length_epsilon = 1.0e-20_f32;
    let parallel_relative_epsilon = 1.0e-6_f32;
    let mut correction_0x = 0.0_f32;
    let mut correction_0y = 0.0_f32;
    let mut correction_0z = 0.0_f32;
    let mut correction_1x = 0.0_f32;
    let mut correction_1y = 0.0_f32;
    let mut correction_1z = 0.0_f32;
    let mut maximum_penetration = 0.0_f32;
    let mut correction_weight = 0.0_f32;

    // Candidates come from the segment's neighbor list. A segment whose list
    // overflowed its capacity instead scans the cells around each of its
    // proxy pieces, located from the positions the lists were built from, and
    // takes a candidate only from the proxy pair holding the closest points,
    // so a pair of long segments is not counted twice. Both sets are valid
    // until a rebuild is requested.
    let length_x = cell_upper[0] - cell_lower[0];
    let length_y = cell_upper[1] - cell_lower[1];
    let length_z = cell_upper[2] - cell_lower[2];
    let listed_count = neighbor_counts[segment_index];
    let listed = listed_count <= neighbor_capacity * list_weights[segment_index];
    let proxies = segment_proxies[segment_index];
    let mut ranges = 27 * proxies;
    if listed {
        ranges = 1;
    }

    for range in 0..ranges {
        let proxy = range / 27;
        let neighbor = range % 27;
        let mut count = 0_u32;
        let mut start = 0_usize;
        if listed {
            count = listed_count;
            start = (neighbor_capacity * list_offsets[segment_index]) as usize;
        } else {
            let home_cell = proxy_cell(
                neighbor_reference_positions[3 * first_vertex],
                neighbor_reference_positions[3 * first_vertex + 1],
                neighbor_reference_positions[3 * first_vertex + 2],
                neighbor_reference_positions[3 * second_vertex],
                neighbor_reference_positions[3 * second_vertex + 1],
                neighbor_reference_positions[3 * second_vertex + 2],
                proxy,
                proxies,
                cell_lower,
                cell_upper,
                cell_periodic,
                cells_x,
                cells_y,
                cells_z,
            );
            let home_x = home_cell % cells_x;
            let home_y = (home_cell / cells_x) % cells_y;
            let home_z = home_cell / (cells_x * cells_y);
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
                count = cell_counts[cell];
                start = cell_offsets[cell] as usize;
            }
        }
        for slot in 0..count {
            let mut other = 0_usize;
            let mut other_proxy = 0_u32;
            if listed {
                other = neighbor_segments[start + slot as usize] as usize;
            } else {
                other = cell_segments[start + slot as usize] as usize;
                other_proxy = cell_proxies[start + slot as usize];
            }
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
                let image_dx = 0.5 * (p2x + q2x - p1x - q1x);
                let image_dy = 0.5 * (p2y + q2y - p1y - q1y);
                let image_dz = 0.5 * (p2z + q2z - p1z - q1z);
                if cell_periodic[0] != 0 {
                    let image_shift = (image_dx / length_x).round() * length_x;
                    p2x -= image_shift;
                    q2x -= image_shift;
                }
                if cell_periodic[1] != 0 {
                    let image_shift = (image_dy / length_y).round() * length_y;
                    p2y -= image_shift;
                    q2y -= image_shift;
                }
                if cell_periodic[2] != 0 {
                    let image_shift = (image_dz / length_z).round() * length_z;
                    p2z -= image_shift;
                    q2z -= image_shift;
                }
                let d2x = q2x - p2x;
                let d2y = q2y - p2y;
                let d2z = q2z - p2z;
                // Every point of a segment lies within half its length of
                // its midpoint, so capsules whose midpoints are farther
                // apart than both half-lengths plus both radii cannot
                // touch. Rejecting them here skips the exact solve for the
                // vast majority of broad-phase candidates.
                let midpoint_dx = 0.5 * (p2x + q2x - p1x - q1x);
                let midpoint_dy = 0.5 * (p2y + q2y - p1y - q1y);
                let midpoint_dz = 0.5 * (p2z + q2z - p1z - q1z);
                let other_radius = segment_radii[other];
                let reach = (half_length
                    + 0.5 * (d2x * d2x + d2y * d2y + d2z * d2z).sqrt()
                    + radius
                    + other_radius)
                    * (1.0 + 1.0e-4_f32);
                if midpoint_dx * midpoint_dx + midpoint_dy * midpoint_dy + midpoint_dz * midpoint_dz
                    <= reach * reach
                {
                    let offset2 = segment_lane_offsets[other];
                    let lane_radius2 = other_radius - offset2;
                    let gyration2 =
                        0.5 * (other_radius * other_radius + lane_radius2 * lane_radius2).sqrt();
                    let mut lanes2 = 1_u32;
                    let mut a2x = 0.0_f32;
                    let mut a2y = 0.0_f32;
                    let mut a2z = 0.0_f32;
                    let mut b2x = 0.0_f32;
                    let mut b2y = 0.0_f32;
                    let mut b2z = 0.0_f32;
                    let mut ea2x = 0.0_f32;
                    let mut ea2y = 0.0_f32;
                    let mut ea2z = 0.0_f32;
                    let mut eb2x = 0.0_f32;
                    let mut eb2y = 0.0_f32;
                    let mut eb2z = 0.0_f32;
                    if offset2 > 0.0 {
                        lanes2 = (2.0 * offset2 / (0.6 * lane_radius2)).ceil() as u32 + 1;
                        if lanes2 < 2 {
                            lanes2 = 2;
                        }
                        if lanes2 > 5 {
                            lanes2 = 5;
                        }
                        a2x = directors[3 * third_vertex];
                        a2y = directors[3 * third_vertex + 1];
                        a2z = directors[3 * third_vertex + 2];
                        b2x = directors[3 * fourth_vertex];
                        b2y = directors[3 * fourth_vertex + 1];
                        b2z = directors[3 * fourth_vertex + 2];
                        let inverse_length =
                            1.0 / (d2x * d2x + d2y * d2y + d2z * d2z).sqrt().max(1.0e-20_f32);
                        let t2x = d2x * inverse_length;
                        let t2y = d2y * inverse_length;
                        let t2z = d2z * inverse_length;
                        ea2x = t2y * a2z - t2z * a2y;
                        ea2y = t2z * a2x - t2x * a2z;
                        ea2z = t2x * a2y - t2y * a2x;
                        eb2x = t2y * b2z - t2z * b2y;
                        eb2y = t2z * b2x - t2x * b2z;
                        eb2z = t2x * b2y - t2y * b2x;
                    }
                    for lane1 in 0..lanes1 {
                        let mut c1 = 0.0_f32;
                        if lanes1 > 1 {
                            c1 = offset1 * (2.0 * lane1 as f32 / (lanes1 - 1) as f32 - 1.0);
                        }
                        let lp1x = p1x + c1 * a1x;
                        let lp1y = p1y + c1 * a1y;
                        let lp1z = p1z + c1 * a1z;
                        let ld1x = q1x + c1 * b1x - lp1x;
                        let ld1y = q1y + c1 * b1y - lp1y;
                        let ld1z = q1z + c1 * b1z - lp1z;
                        for lane2 in 0..lanes2 {
                            let mut c2 = 0.0_f32;
                            if lanes2 > 1 {
                                c2 = offset2 * (2.0 * lane2 as f32 / (lanes2 - 1) as f32 - 1.0);
                            }
                            let lp2x = p2x + c2 * a2x;
                            let lp2y = p2y + c2 * a2y;
                            let lp2z = p2z + c2 * a2z;
                            let ld2x = q2x + c2 * b2x - lp2x;
                            let ld2y = q2y + c2 * b2y - lp2y;
                            let ld2z = q2z + c2 * b2z - lp2z;
                            let rx = lp1x - lp2x;
                            let ry = lp1y - lp2y;
                            let rz = lp1z - lp2z;
                            let a = ld1x * ld1x + ld1y * ld1y + ld1z * ld1z;
                            let e = ld2x * ld2x + ld2y * ld2y + ld2z * ld2z;
                            let f = ld2x * rx + ld2y * ry + ld2z * rz;
                            let mut s = 0.0_f32;
                            let mut t = 0.0_f32;

                            if a <= length_epsilon && e > length_epsilon {
                                t = (f / e).clamp(0.0, 1.0);
                            } else if a > length_epsilon {
                                let c = ld1x * rx + ld1y * ry + ld1z * rz;
                                if e <= length_epsilon {
                                    s = (-c / a).clamp(0.0, 1.0);
                                } else {
                                    let b = ld1x * ld2x + ld1y * ld2y + ld1z * ld2z;
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

                            let point1x = lp1x + ld1x * s;
                            let point1y = lp1y + ld1y * s;
                            let point1z = lp1z + ld1z * s;
                            let point2x = lp2x + ld2x * t;
                            let point2y = lp2y + ld2y * t;
                            let point2z = lp2z + ld2z * t;
                            let delta_x = point2x - point1x;
                            let delta_y = point2y - point1y;
                            let delta_z = point2z - point1z;
                            let distance =
                                (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();
                            let penetration = lane_radius1 + lane_radius2 - distance;
                            let canonical = listed
                                || (proxy_of(s, proxies) == proxy
                                    && proxy_of(t, segment_proxies[other]) == other_proxy);
                            if penetration > 0.0 && canonical {
                                let deepest = penetration > maximum_penetration;
                                if deepest {
                                    maximum_penetration = penetration;
                                }

                                let mut nx = 0.0_f32;
                                let mut ny = 0.0_f32;
                                let mut nz = 0.0_f32;
                                if distance > 1.0e-7_f32 {
                                    let inverse_distance = 1.0 / distance;
                                    nx = delta_x * inverse_distance;
                                    ny = delta_y * inverse_distance;
                                    nz = delta_z * inverse_distance;
                                } else {
                                    nx = ld1y * ld2z - ld1z * ld2y;
                                    ny = ld1z * ld2x - ld1x * ld2z;
                                    nz = ld1x * ld2y - ld1y * ld2x;
                                    let mut normal_length = (nx * nx + ny * ny + nz * nz).sqrt();
                                    if normal_length <= 1.0e-7_f32 {
                                        if ld1x.abs() <= ld1y.abs() && ld1x.abs() <= ld1z.abs() {
                                            nx = 0.0;
                                            ny = ld1z;
                                            nz = -ld1y;
                                        } else if ld1y.abs() <= ld1z.abs() {
                                            nx = -ld1z;
                                            ny = 0.0;
                                            nz = ld1x;
                                        } else {
                                            nx = ld1y;
                                            ny = -ld1x;
                                            nz = 0.0;
                                        }
                                        normal_length = (nx * nx + ny * ny + nz * nz).sqrt();
                                    }
                                    if normal_length > 1.0e-7_f32 {
                                        nx /= normal_length;
                                        ny /= normal_length;
                                        nz /= normal_length;
                                    } else {
                                        nx = 1.0;
                                    }
                                    let dominant = if nx.abs() >= ny.abs() && nx.abs() >= nz.abs() {
                                        nx
                                    } else if ny.abs() >= nz.abs() {
                                        ny
                                    } else {
                                        nz
                                    };
                                    if dominant < 0.0 {
                                        nx = -nx;
                                        ny = -ny;
                                        nz = -nz;
                                    }
                                    if segment_index > other {
                                        nx = -nx;
                                        ny = -ny;
                                        nz = -nz;
                                    }
                                }
                                let weight_0 = 1.0 - s;
                                let weight_1 = s;
                                let other_weight_0 = 1.0 - t;
                                // Turning a director moves its lane by the lane
                                // offset times the angle, in the direction
                                // tangent × director. Angles are measured in
                                // lengths of the radius of gyration, so the
                                // turn and the shift of a vertex share one
                                // generalized mass. All four terms are zero for
                                // round fibers.
                                let turn_0 = weight_0
                                    * (c1 / gyration1)
                                    * (nx * ea1x + ny * ea1y + nz * ea1z);
                                let turn_1 = weight_1
                                    * (c1 / gyration1)
                                    * (nx * eb1x + ny * eb1y + nz * eb1z);
                                let other_turn_0 = other_weight_0
                                    * (c2 / gyration2)
                                    * (nx * ea2x + ny * ea2y + nz * ea2z);
                                let other_turn_1 =
                                    t * (c2 / gyration2) * (nx * eb2x + ny * eb2y + nz * eb2z);
                                let denominator = weight_0 * weight_0
                                    + weight_1 * weight_1
                                    + other_weight_0 * other_weight_0
                                    + t * t
                                    + (turn_0 * turn_0
                                        + turn_1 * turn_1
                                        + other_turn_0 * other_turn_0
                                        + other_turn_1 * other_turn_1);
                                let magnitude = -correction_fraction * penetration / denominator;
                                let mut aggregate_weight = 1.0_f32;
                                if contact_aggregation == 1 {
                                    aggregate_weight = penetration;
                                }
                                if contact_aggregation == 2 {
                                    if deepest {
                                        correction_0x = nx * magnitude * weight_0;
                                        correction_0y = ny * magnitude * weight_0;
                                        correction_0z = nz * magnitude * weight_0;
                                        correction_1x = nx * magnitude * weight_1;
                                        correction_1y = ny * magnitude * weight_1;
                                        correction_1z = nz * magnitude * weight_1;
                                        twist_0 = magnitude * turn_0 / gyration1;
                                        twist_1 = magnitude * turn_1 / gyration1;
                                        correction_weight = 1.0;
                                    }
                                } else {
                                    correction_0x += nx * magnitude * weight_0 * aggregate_weight;
                                    correction_0y += ny * magnitude * weight_0 * aggregate_weight;
                                    correction_0z += nz * magnitude * weight_0 * aggregate_weight;
                                    correction_1x += nx * magnitude * weight_1 * aggregate_weight;
                                    correction_1y += ny * magnitude * weight_1 * aggregate_weight;
                                    correction_1z += nz * magnitude * weight_1 * aggregate_weight;
                                    twist_0 += magnitude * turn_0 / gyration1 * aggregate_weight;
                                    twist_1 += magnitude * turn_1 / gyration1 * aggregate_weight;
                                    correction_weight += aggregate_weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Jacobi contact projections sharing one segment must be averaged. A raw
    // sum grows with local coordination number, immediately hits max_step in
    // dense networks, and oscillates rather than reducing every constraint.
    if contact_aggregation != 2 && correction_weight > 1.0e-20_f32 {
        correction_0x /= correction_weight;
        correction_0y /= correction_weight;
        correction_0z /= correction_weight;
        correction_1x /= correction_weight;
        correction_1y /= correction_weight;
        correction_1z /= correction_weight;
        twist_0 /= correction_weight;
        twist_1 /= correction_weight;
    }

    corrections[6 * segment_index] = correction_0x;
    corrections[6 * segment_index + 1] = correction_0y;
    corrections[6 * segment_index + 2] = correction_0z;
    corrections[6 * segment_index + 3] = correction_1x;
    corrections[6 * segment_index + 4] = correction_1y;
    corrections[6 * segment_index + 5] = correction_1z;
    director_corrections[2 * segment_index] = twist_0;
    director_corrections[2 * segment_index + 1] = twist_1;
    segment_max_penetration[segment_index] = maximum_penetration;
}
