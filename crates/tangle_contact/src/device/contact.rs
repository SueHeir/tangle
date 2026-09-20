//! Exact capsule contact capture and projection kernels.

#![allow(missing_docs)]

use cubecl::prelude::*;

#[cube(launch_unchecked)]
#[allow(unused_assignments)]
pub fn capture_segment_contacts(
    positions: &Array<f32>,
    segment_vertices: &Array<u32>,
    segment_fibers: &Array<u32>,
    segment_radii: &Array<f32>,
    active_segments: &Array<u32>,
    active_counts: &Array<Atomic<u32>>,
    cell_counts: &Array<u32>,
    cell_offsets: &Array<u32>,
    cell_segments: &Array<u32>,
    cell_lower: &Array<f32>,
    cell_upper: &Array<f32>,
    cell_periodic: &Array<u32>,
    captured_count: &mut Array<Atomic<u32>>,
    captured_segments: &mut Array<u32>,
    captured_coordinates: &mut Array<f32>,
    captured_surface_gaps: &mut Array<f32>,
    captured_crossing_angles: &mut Array<f32>,
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
    positions: &Array<f32>,
    segment_vertices: &Array<u32>,
    segment_fibers: &Array<u32>,
    segment_radii: &Array<f32>,
    active_segments: &Array<u32>,
    active_counts: &Array<Atomic<u32>>,
    cell_counts: &Array<u32>,
    cell_offsets: &Array<u32>,
    cell_segments: &Array<u32>,
    cell_lower: &Array<f32>,
    cell_upper: &Array<f32>,
    cell_periodic: &Array<u32>,
    control: &Array<u32>,
    corrections: &mut Array<f32>,
    segment_max_penetration: &mut Array<f32>,
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
                let third_vertex = segment_vertices[2 * other] as usize;
                let fourth_vertex = segment_vertices[2 * other + 1] as usize;
                let adjacent_same_fiber = segment_fibers[other] == owner
                    && (first_vertex == third_vertex
                        || first_vertex == fourth_vertex
                        || second_vertex == third_vertex
                        || second_vertex == fourth_vertex);
                if other != segment_index && !adjacent_same_fiber {
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

                    let point1x = p1x + d1x * s;
                    let point1y = p1y + d1y * s;
                    let point1z = p1z + d1z * s;
                    let point2x = p2x + d2x * t;
                    let point2y = p2y + d2y * t;
                    let point2z = p2z + d2z * t;
                    let delta_x = point2x - point1x;
                    let delta_y = point2y - point1y;
                    let delta_z = point2z - point1z;
                    let distance =
                        (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();
                    let penetration = radius + segment_radii[other] - distance;
                    if penetration > 0.0 {
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
                            nx = d1y * d2z - d1z * d2y;
                            ny = d1z * d2x - d1x * d2z;
                            nz = d1x * d2y - d1y * d2x;
                            let mut normal_length = (nx * nx + ny * ny + nz * nz).sqrt();
                            if normal_length <= 1.0e-7_f32 {
                                if d1x.abs() <= d1y.abs() && d1x.abs() <= d1z.abs() {
                                    nx = 0.0;
                                    ny = d1z;
                                    nz = -d1y;
                                } else if d1y.abs() <= d1z.abs() {
                                    nx = -d1z;
                                    ny = 0.0;
                                    nz = d1x;
                                } else {
                                    nx = d1y;
                                    ny = -d1x;
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
                        let denominator = weight_0 * weight_0
                            + weight_1 * weight_1
                            + other_weight_0 * other_weight_0
                            + t * t;
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
                                correction_weight = 1.0;
                            }
                        } else {
                            correction_0x += nx * magnitude * weight_0 * aggregate_weight;
                            correction_0y += ny * magnitude * weight_0 * aggregate_weight;
                            correction_0z += nz * magnitude * weight_0 * aggregate_weight;
                            correction_1x += nx * magnitude * weight_1 * aggregate_weight;
                            correction_1y += ny * magnitude * weight_1 * aggregate_weight;
                            correction_1z += nz * magnitude * weight_1 * aggregate_weight;
                            correction_weight += aggregate_weight;
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
    }

    corrections[6 * segment_index] = correction_0x;
    corrections[6 * segment_index + 1] = correction_0y;
    corrections[6 * segment_index + 2] = correction_0z;
    corrections[6 * segment_index + 3] = correction_1x;
    corrections[6 * segment_index + 4] = correction_1y;
    corrections[6 * segment_index + 5] = correction_1z;
    segment_max_penetration[segment_index] = maximum_penetration;
}
