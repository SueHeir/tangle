//! Image-attraction kernels used to fit centerlines to a CT volume.
//!
//! Each active, unpinned vertex samples a normalized image on a polar grid in
//! its cross-section plane, keeps the samples it owns under the
//! nearest-capsule-surface rule, and moves laterally toward their weighted
//! centroid. Contact, stretch and bend constraints of the ordinary relaxation
//! act on the same positions, so the fit stays admissible while it follows
//! the image.
//!
//! The polar grid is fixed by host-side tables in units of the vertex radius:
//! for every ring `[radius, gaussian weight × area, area, support flag]`,
//! followed by `[cos, sin]` for every spoke. No trigonometry or exponentials
//! run on the device.

use cubecl::prelude::*;

/// Entries per ring in the polar sampling table.
pub(crate) const RING_TABLE_STRIDE: usize = 4;

/// Other-fiber segments kept per vertex for the ownership test. A vertex
/// with more candidates falls back to scanning its neighbor lists per sample.
const MAX_CANDIDATES: usize = 32;

/// Trilinear image value at grid coordinate `(gx, gy, gz)`, zero outside.
///
/// Grid coordinates are voxel-center based: voxel `(i, j, k)` sits at
/// `(i, j, k)`. The image is stored `(z, y, x)` with `x` fastest.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sample_image(image: &[f32], nx: u32, ny: u32, nz: u32, gx: f32, gy: f32, gz: f32) -> f32 {
    let fx = gx.floor();
    let fy = gy.floor();
    let fz = gz.floor();
    let tx = gx - fx;
    let ty = gy - fy;
    let tz = gz - fz;
    let i0 = fx as i32;
    let j0 = fy as i32;
    let k0 = fz as i32;
    let mut value = 0.0_f32;
    for corner in 0..8_u32 {
        let di = (corner & 1) as i32;
        let dj = ((corner >> 1) & 1) as i32;
        let dk = ((corner >> 2) & 1) as i32;
        let i = i0 + di;
        let j = j0 + dj;
        let k = k0 + dk;
        if i >= 0 && j >= 0 && k >= 0 && i < nx as i32 && j < ny as i32 && k < nz as i32 {
            let mut wx = 1.0 - tx;
            if di == 1 {
                wx = tx;
            }
            let mut wy = 1.0 - ty;
            if dj == 1 {
                wy = ty;
            }
            let mut wz = 1.0 - tz;
            if dk == 1 {
                wz = tz;
            }
            let offset = ((k as u32 * ny + j as u32) * nx + i as u32) as usize;
            value += wx * wy * wz * image[offset];
        }
    }
    value
}

/// Surface distance from point `(sx, sy, sz)` to capsule segment `other`,
/// using the nearest periodic image of the segment.
#[cube]
#[allow(clippy::too_many_arguments)]
fn capsule_surface_distance(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_radii: &[f32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    other: usize,
    sx: f32,
    sy: f32,
    sz: f32,
) -> f32 {
    let first = segment_vertices[2 * other] as usize;
    let second = segment_vertices[2 * other + 1] as usize;
    let mut ax = positions[3 * first];
    let mut ay = positions[3 * first + 1];
    let mut az = positions[3 * first + 2];
    let dx = positions[3 * second] - ax;
    let dy = positions[3 * second + 1] - ay;
    let dz = positions[3 * second + 2] - az;
    if cell_periodic[0] != 0 {
        let length = cell_upper[0] - cell_lower[0];
        ax += ((sx - ax - 0.5 * dx) / length).round() * length;
    }
    if cell_periodic[1] != 0 {
        let length = cell_upper[1] - cell_lower[1];
        ay += ((sy - ay - 0.5 * dy) / length).round() * length;
    }
    if cell_periodic[2] != 0 {
        let length = cell_upper[2] - cell_lower[2];
        az += ((sz - az - 0.5 * dz) / length).round() * length;
    }
    let rx = sx - ax;
    let ry = sy - ay;
    let rz = sz - az;
    let squared = dx * dx + dy * dy + dz * dz;
    let mut t = 0.0_f32;
    if squared > 1.0e-30_f32 {
        t = ((rx * dx + ry * dy + rz * dz) / squared).clamp(0.0, 1.0);
    }
    let ex = rx - t * dx;
    let ey = ry - t * dy;
    let ez = rz - t * dz;
    (ex * ex + ey * ey + ez * ez).sqrt() - segment_radii[other]
}

/// Records `other` as an ownership candidate of the vertex at `(x, y, z)`
/// when it belongs to another fiber and its surface is within `limit` of the
/// vertex. Returns the updated candidate count, which keeps counting past
/// the array capacity so overflow can be detected.
#[cube]
#[allow(clippy::too_many_arguments)]
fn consider_candidate(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_fibers: &[u32],
    segment_radii: &[f32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    candidates: &mut Array<u32>,
    count: u32,
    other: u32,
    owner: u32,
    limit: f32,
    x: f32,
    y: f32,
    z: f32,
) -> u32 {
    let mut updated = count;
    if segment_fibers[other as usize] != owner {
        let mut duplicate = false;
        let mut stored = count;
        if stored > MAX_CANDIDATES as u32 {
            stored = MAX_CANDIDATES as u32;
        }
        for slot in 0..stored {
            if candidates[slot as usize] == other {
                duplicate = true;
            }
        }
        if !duplicate {
            let gap = capsule_surface_distance(
                positions,
                segment_vertices,
                segment_radii,
                cell_lower,
                cell_upper,
                cell_periodic,
                other as usize,
                x,
                y,
                z,
            );
            if gap < limit {
                if count < MAX_CANDIDATES as u32 {
                    candidates[count as usize] = other;
                }
                updated = count + 1;
            }
        }
    }
    updated
}

/// Adds the ownership candidates listed around `segment` (its neighbor list,
/// or the 27 cells around its home cell when the list overflowed).
#[cube]
#[allow(clippy::too_many_arguments)]
fn collect_candidates(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_fibers: &[u32],
    segment_radii: &[f32],
    neighbor_counts: &[u32],
    neighbor_segments: &[u32],
    neighbor_home_cells: &[u32],
    cell_counts: &[u32],
    cell_offsets: &[u32],
    cell_segments: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    candidates: &mut Array<u32>,
    count: u32,
    segment: usize,
    owner: u32,
    limit: f32,
    x: f32,
    y: f32,
    z: f32,
    neighbor_capacity: u32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) -> u32 {
    let mut updated = count;
    let listed_count = neighbor_counts[segment];
    if listed_count <= neighbor_capacity {
        let start = segment * neighbor_capacity as usize;
        for slot in 0..listed_count {
            updated = consider_candidate(
                positions,
                segment_vertices,
                segment_fibers,
                segment_radii,
                cell_lower,
                cell_upper,
                cell_periodic,
                candidates,
                updated,
                neighbor_segments[start + slot as usize],
                owner,
                limit,
                x,
                y,
                z,
            );
        }
    } else {
        let home_cell = neighbor_home_cells[segment];
        let home_x = home_cell % cells_x;
        let home_y = (home_cell / cells_x) % cells_y;
        let home_z = home_cell / (cells_x * cells_y);
        for neighbor in 0..27_u32 {
            let raw_x = home_x as i32 + (neighbor % 3) as i32 - 1;
            let raw_y = home_y as i32 + ((neighbor / 3) % 3) as i32 - 1;
            let raw_z = home_z as i32 + (neighbor / 9) as i32 - 1;
            let valid_x = cell_periodic[0] != 0 || (raw_x >= 0 && raw_x < cells_x as i32);
            let valid_y = cell_periodic[1] != 0 || (raw_y >= 0 && raw_y < cells_y as i32);
            let valid_z = cell_periodic[2] != 0 || (raw_z >= 0 && raw_z < cells_z as i32);
            if valid_x && valid_y && valid_z {
                let cx = ((raw_x % cells_x as i32 + cells_x as i32) % cells_x as i32) as u32;
                let cy = ((raw_y % cells_y as i32 + cells_y as i32) % cells_y as i32) as u32;
                let cz = ((raw_z % cells_z as i32 + cells_z as i32) % cells_z as i32) as u32;
                let cell = ((cz * cells_y + cy) * cells_x + cx) as usize;
                let cell_count = cell_counts[cell];
                let start = cell_offsets[cell] as usize;
                for slot in 0..cell_count {
                    updated = consider_candidate(
                        positions,
                        segment_vertices,
                        segment_fibers,
                        segment_radii,
                        cell_lower,
                        cell_upper,
                        cell_periodic,
                        candidates,
                        updated,
                        cell_segments[start + slot as usize],
                        owner,
                        limit,
                        x,
                        y,
                        z,
                    );
                }
            }
        }
    }
    updated
}

/// Whether a segment of another fiber listed around `segment` has its surface
/// closer to the sample than `own_gap` (the sample's distance to the vertex
/// surface). Falls back to the 27 cells around the recorded home cell when
/// the segment's neighbor list overflowed, as the contact kernel does.
#[cube]
#[allow(clippy::too_many_arguments)]
fn claimed_by_other_fiber(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_fibers: &[u32],
    segment_radii: &[f32],
    neighbor_counts: &[u32],
    neighbor_segments: &[u32],
    neighbor_home_cells: &[u32],
    cell_counts: &[u32],
    cell_offsets: &[u32],
    cell_segments: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    segment: usize,
    owner: u32,
    own_gap: f32,
    sx: f32,
    sy: f32,
    sz: f32,
    neighbor_capacity: u32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) -> bool {
    let mut claimed = false;
    let listed_count = neighbor_counts[segment];
    if listed_count <= neighbor_capacity {
        let start = segment * neighbor_capacity as usize;
        for slot in 0..listed_count {
            if !claimed {
                let other = neighbor_segments[start + slot as usize] as usize;
                if segment_fibers[other] != owner {
                    let gap = capsule_surface_distance(
                        positions,
                        segment_vertices,
                        segment_radii,
                        cell_lower,
                        cell_upper,
                        cell_periodic,
                        other,
                        sx,
                        sy,
                        sz,
                    );
                    if gap < own_gap {
                        claimed = true;
                    }
                }
            }
        }
    } else {
        let home_cell = neighbor_home_cells[segment];
        let home_x = home_cell % cells_x;
        let home_y = (home_cell / cells_x) % cells_y;
        let home_z = home_cell / (cells_x * cells_y);
        for neighbor in 0..27_u32 {
            let offset_x = (neighbor % 3) as i32 - 1;
            let offset_y = ((neighbor / 3) % 3) as i32 - 1;
            let offset_z = (neighbor / 9) as i32 - 1;
            let raw_x = home_x as i32 + offset_x;
            let raw_y = home_y as i32 + offset_y;
            let raw_z = home_z as i32 + offset_z;
            let valid_x = cell_periodic[0] != 0 || (raw_x >= 0 && raw_x < cells_x as i32);
            let valid_y = cell_periodic[1] != 0 || (raw_y >= 0 && raw_y < cells_y as i32);
            let valid_z = cell_periodic[2] != 0 || (raw_z >= 0 && raw_z < cells_z as i32);
            if valid_x && valid_y && valid_z && !claimed {
                let cx = ((raw_x % cells_x as i32 + cells_x as i32) % cells_x as i32) as u32;
                let cy = ((raw_y % cells_y as i32 + cells_y as i32) % cells_y as i32) as u32;
                let cz = ((raw_z % cells_z as i32 + cells_z as i32) % cells_z as i32) as u32;
                let cell = ((cz * cells_y + cy) * cells_x + cx) as usize;
                let count = cell_counts[cell];
                let start = cell_offsets[cell] as usize;
                for slot in 0..count {
                    if !claimed {
                        let other = cell_segments[start + slot as usize] as usize;
                        if segment_fibers[other] != owner {
                            let gap = capsule_surface_distance(
                                positions,
                                segment_vertices,
                                segment_radii,
                                cell_lower,
                                cell_upper,
                                cell_periodic,
                                other,
                                sx,
                                sy,
                                sz,
                            );
                            if gap < own_gap {
                                claimed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    claimed
}

/// Computes each active vertex's lateral image correction and its image
/// statistics.
///
/// `corrections` receives three values per vertex (zero where no step is
/// taken). `stats` receives the owned mass (Σ clip(I, ±1.5) · dA over owned
/// samples, in voxel² units) and the support (mean I over the central
/// samples) per vertex. With `gate == 0` the kernel ignores the batch
/// control flag, so the statistics can be measured between batches.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_image_corrections(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_fibers: &[u32],
    segment_radii: &[f32],
    vertex_segments: &[u32],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    control: &[u32],
    neighbor_counts: &[u32],
    neighbor_segments: &[u32],
    neighbor_home_cells: &[u32],
    cell_counts: &[u32],
    cell_offsets: &[u32],
    cell_segments: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    image: &[f32],
    tables: &[f32],
    corrections: &mut [f32],
    stats: &mut [f32],
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    inverse_voxel: f32,
    rate: f32,
    reach_radii: f32,
    max_step: f32,
    image_nx: u32,
    image_ny: u32,
    image_nz: u32,
    rings: u32,
    spokes: u32,
    gate: u32,
    neighbor_capacity: u32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) {
    let work = ABSOLUTE_POS;
    if work >= active_counts[1].load() as usize || (gate != 0 && control[0] == 0) {
        terminate!();
    }
    let index = active_vertices[work as usize] as usize;
    corrections[3 * index] = 0.0;
    corrections[3 * index + 1] = 0.0;
    corrections[3 * index + 2] = 0.0;
    stats[2 * index] = 0.0;
    stats[2 * index + 1] = 0.0;
    if vertex_active[index] == 0 {
        terminate!();
    }
    let previous_segment = vertex_segments[2 * index];
    let next_segment = vertex_segments[2 * index + 1];
    if previous_segment == u32::MAX && next_segment == u32::MAX {
        terminate!();
    }
    let x = positions[3 * index];
    let y = positions[3 * index + 1];
    let z = positions[3 * index + 2];

    // Tangent from the neighboring vertices (one-sided at fiber ends).
    let mut owner = 0_u32;
    let mut radius = 0.0_f32;
    let mut previous_x = x;
    let mut previous_y = y;
    let mut previous_z = z;
    let mut next_x = x;
    let mut next_y = y;
    let mut next_z = z;
    if previous_segment != u32::MAX {
        let segment = previous_segment as usize;
        let first = segment_vertices[2 * segment] as usize;
        let second = segment_vertices[2 * segment + 1] as usize;
        let other = if first == index { second } else { first };
        previous_x = positions[3 * other];
        previous_y = positions[3 * other + 1];
        previous_z = positions[3 * other + 2];
        owner = segment_fibers[segment];
        radius = segment_radii[segment];
    }
    if next_segment != u32::MAX {
        let segment = next_segment as usize;
        let first = segment_vertices[2 * segment] as usize;
        let second = segment_vertices[2 * segment + 1] as usize;
        let other = if first == index { second } else { first };
        next_x = positions[3 * other];
        next_y = positions[3 * other + 1];
        next_z = positions[3 * other + 2];
        owner = segment_fibers[segment];
        radius = segment_radii[segment];
    }
    let mut tx = next_x - previous_x;
    let mut ty = next_y - previous_y;
    let mut tz = next_z - previous_z;
    let tangent_length = (tx * tx + ty * ty + tz * tz).sqrt();
    if tangent_length <= 1.0e-12_f32 || radius <= 0.0 {
        terminate!();
    }
    tx /= tangent_length;
    ty /= tangent_length;
    tz /= tangent_length;

    // Perpendicular basis (u, w) of the cross-section plane.
    let mut ax = 1.0_f32;
    let mut ay = 0.0_f32;
    if tx.abs() > 0.9 {
        ax = 0.0;
        ay = 1.0;
    }
    let along = ax * tx + ay * ty;
    let mut ux = ax - along * tx;
    let mut uy = ay - along * ty;
    let mut uz = -along * tz;
    let u_length = (ux * ux + uy * uy + uz * uz).sqrt();
    ux /= u_length;
    uy /= u_length;
    uz /= u_length;
    let wx = ty * uz - tz * uy;
    let wy = tz * ux - tx * uz;
    let wz = tx * uy - ty * ux;

    // Other-fiber segments that can own any sample of this vertex: a sample
    // at distance rho <= reach * r is claimed only by a surface closer than
    // rho - r, so the segment's surface lies within (2 reach - 1) r of the
    // vertex.
    let limit = (2.0 * reach_radii - 1.0) * radius;
    let mut candidates = Array::<u32>::new(MAX_CANDIDATES);
    let mut candidate_count = 0_u32;
    if previous_segment != u32::MAX {
        candidate_count = collect_candidates(
            positions,
            segment_vertices,
            segment_fibers,
            segment_radii,
            neighbor_counts,
            neighbor_segments,
            neighbor_home_cells,
            cell_counts,
            cell_offsets,
            cell_segments,
            cell_lower,
            cell_upper,
            cell_periodic,
            &mut candidates,
            candidate_count,
            previous_segment as usize,
            owner,
            limit,
            x,
            y,
            z,
            neighbor_capacity,
            cells_x,
            cells_y,
            cells_z,
        );
    }
    if next_segment != u32::MAX {
        candidate_count = collect_candidates(
            positions,
            segment_vertices,
            segment_fibers,
            segment_radii,
            neighbor_counts,
            neighbor_segments,
            neighbor_home_cells,
            cell_counts,
            cell_offsets,
            cell_segments,
            cell_lower,
            cell_upper,
            cell_periodic,
            &mut candidates,
            candidate_count,
            next_segment as usize,
            owner,
            limit,
            x,
            y,
            z,
            neighbor_capacity,
            cells_x,
            cells_y,
            cells_z,
        );
    }
    let overflowed = candidate_count > MAX_CANDIDATES as u32;

    let voxel_radius = radius * inverse_voxel;
    let area_scale = voxel_radius * voxel_radius;
    let spoke_table = RING_TABLE_STRIDE * rings as usize;
    let mut weight_sum = 0.0_f32;
    let mut shift_x = 0.0_f32;
    let mut shift_y = 0.0_f32;
    let mut shift_z = 0.0_f32;
    let mut total_weight = 0.0_f32;
    let mut mass = 0.0_f32;
    let mut support_sum = 0.0_f32;
    let mut support_count = 0.0_f32;
    for ring in 0..rings {
        let entry = RING_TABLE_STRIDE * ring as usize;
        let rho = tables[entry] * radius;
        let ring_weight = tables[entry + 1] * area_scale;
        let ring_area = tables[entry + 2] * area_scale;
        let is_support = tables[entry + 3] > 0.5;
        let own_gap = rho - radius;
        for spoke in 0..spokes {
            let cosine = tables[spoke_table + 2 * spoke as usize];
            let sine = tables[spoke_table + 2 * spoke as usize + 1];
            let ox = rho * (cosine * ux + sine * wx);
            let oy = rho * (cosine * uy + sine * wy);
            let oz = rho * (cosine * uz + sine * wz);
            let sx = x + ox;
            let sy = y + oy;
            let sz = z + oz;
            let intensity = sample_image(
                image,
                image_nx,
                image_ny,
                image_nz,
                (sx - origin_x) * inverse_voxel - 0.5,
                (sy - origin_y) * inverse_voxel - 0.5,
                (sz - origin_z) * inverse_voxel - 0.5,
            );
            if is_support {
                support_sum += intensity;
                support_count += 1.0;
            }
            let mut claimed = false;
            if !overflowed {
                for slot in 0..candidate_count {
                    if !claimed {
                        let gap = capsule_surface_distance(
                            positions,
                            segment_vertices,
                            segment_radii,
                            cell_lower,
                            cell_upper,
                            cell_periodic,
                            candidates[slot as usize] as usize,
                            sx,
                            sy,
                            sz,
                        );
                        if gap < own_gap {
                            claimed = true;
                        }
                    }
                }
            }
            if overflowed && previous_segment != u32::MAX {
                claimed = claimed_by_other_fiber(
                    positions,
                    segment_vertices,
                    segment_fibers,
                    segment_radii,
                    neighbor_counts,
                    neighbor_segments,
                    neighbor_home_cells,
                    cell_counts,
                    cell_offsets,
                    cell_segments,
                    cell_lower,
                    cell_upper,
                    cell_periodic,
                    previous_segment as usize,
                    owner,
                    own_gap,
                    sx,
                    sy,
                    sz,
                    neighbor_capacity,
                    cells_x,
                    cells_y,
                    cells_z,
                );
            }
            if overflowed && !claimed && next_segment != u32::MAX {
                claimed = claimed_by_other_fiber(
                    positions,
                    segment_vertices,
                    segment_fibers,
                    segment_radii,
                    neighbor_counts,
                    neighbor_segments,
                    neighbor_home_cells,
                    cell_counts,
                    cell_offsets,
                    cell_segments,
                    cell_lower,
                    cell_upper,
                    cell_periodic,
                    next_segment as usize,
                    owner,
                    own_gap,
                    sx,
                    sy,
                    sz,
                    neighbor_capacity,
                    cells_x,
                    cells_y,
                    cells_z,
                );
            }
            total_weight += ring_weight;
            if !claimed {
                let weight = intensity.clamp(0.0, 1.5) * ring_weight;
                weight_sum += weight;
                shift_x += weight * ox;
                shift_y += weight * oy;
                shift_z += weight * oz;
                mass += intensity.clamp(-1.5, 1.5) * ring_area;
            }
        }
    }
    let mut support = 0.0_f32;
    if support_count > 0.0 {
        support = support_sum / support_count;
    }
    stats[2 * index] = mass;
    stats[2 * index + 1] = support;
    if vertex_pinned[index] != 0 || rate <= 0.0 || weight_sum <= 1.0e-6_f32 * total_weight {
        terminate!();
    }
    let scale = rate * support.clamp(0.0, 1.0) / weight_sum;
    let mut cx = shift_x * scale;
    let mut cy = shift_y * scale;
    let mut cz = shift_z * scale;
    let length = (cx * cx + cy * cy + cz * cz).sqrt();
    if length > max_step {
        let clamp = max_step / length;
        cx *= clamp;
        cy *= clamp;
        cz *= clamp;
    }
    corrections[3 * index] = cx;
    corrections[3 * index + 1] = cy;
    corrections[3 * index + 2] = cz;
}

/// Adds the image corrections to active, unpinned vertices, keeping bounded
/// axes inside the cell exactly as the contact projection does.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_image_corrections(
    positions: &mut [f32],
    corrections: &[f32],
    vertex_segments: &[u32],
    segment_radii: &[f32],
    active_vertices: &[u32],
    active_counts: &[Atomic<u32>],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
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
    let previous = vertex_segments[2 * index];
    let next = vertex_segments[2 * index + 1];
    let mut radius = 0.0_f32;
    if previous != u32::MAX {
        radius = segment_radii[previous as usize];
    }
    if next != u32::MAX {
        radius = segment_radii[next as usize];
    }
    for axis in 0..3_u32 {
        let coordinate = 3 * index + axis as usize;
        let moved = positions[coordinate] + corrections[coordinate];
        if cell_periodic[axis as usize] != 0 {
            positions[coordinate] = moved;
        } else {
            positions[coordinate] = moved.clamp(
                cell_lower[axis as usize] + radius,
                cell_upper[axis as usize] - radius,
            );
        }
    }
}
