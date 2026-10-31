//! Junction tether kernels: bonded anchors pulled back together.
//!
//! A tether joins two anchor points, each interpolated on one segment of a
//! fiber. It acts only in tension: once the anchors drift further apart than
//! the tether's rest length, each anchor moves a share of the excess back
//! toward the other, capped at the iteration's step. Contact still keeps the
//! fibers apart, because a rest length is never shorter than the two radii.
//!
//! The work is split in two deterministic launches, as for the image force:
//! one thread per tether finds its correction, then one thread per touched
//! vertex sums the corrections it receives through a host-built list, so no
//! two threads write the same vertex.

use cubecl::prelude::*;

/// Finds the correction of every tether: the step its first anchor takes
/// toward its second. The second anchor takes the opposite step.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_junction_corrections(
    positions: &[f32],
    junction_vertices: &[u32],
    junction_coordinates: &[f32],
    junction_rest_lengths: &[f32],
    vertex_active: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
    corrections: &mut [f32],
    stiffness: f32,
    max_step: f32,
) {
    let tether = ABSOLUTE_POS;
    if tether >= junction_rest_lengths.len() || control[0] == 0 {
        terminate!();
    }
    let index = tether as usize;
    corrections[3 * index] = 0.0;
    corrections[3 * index + 1] = 0.0;
    corrections[3 * index + 2] = 0.0;
    let a0 = junction_vertices[4 * index] as usize;
    let a1 = junction_vertices[4 * index + 1] as usize;
    let b0 = junction_vertices[4 * index + 2] as usize;
    let b1 = junction_vertices[4 * index + 3] as usize;
    // A fiber not yet inserted, or a vertex coarsened away, holds no anchor.
    if vertex_active[a0] == 0
        || vertex_active[a1] == 0
        || vertex_active[b0] == 0
        || vertex_active[b1] == 0
    {
        terminate!();
    }
    let ta = junction_coordinates[2 * index];
    let tb = junction_coordinates[2 * index + 1];
    let mut dx = 0.0_f32;
    let mut dy = 0.0_f32;
    let mut dz = 0.0_f32;
    for axis in 0..3_u32 {
        let k = axis as usize;
        let pa = positions[3 * a0 + k] + ta * (positions[3 * a1 + k] - positions[3 * a0 + k]);
        let pb = positions[3 * b0 + k] + tb * (positions[3 * b1 + k] - positions[3 * b0 + k]);
        let mut delta = pb - pa;
        if cell_periodic[k] != 0 {
            let length = cell_upper[k] - cell_lower[k];
            delta -= (delta / length).round() * length;
        }
        if axis == 0 {
            dx = delta;
        } else if axis == 1 {
            dy = delta;
        } else {
            dz = delta;
        }
    }
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    let excess = length - junction_rest_lengths[index];
    if excess <= 0.0 || length <= 1.0e-30_f32 {
        terminate!();
    }
    // Each anchor closes half of the excess, scaled by the stiffness.
    let mut step = 0.5 * stiffness * excess;
    if step > max_step {
        step = max_step;
    }
    let scale = step / length;
    corrections[3 * index] = scale * dx;
    corrections[3 * index + 1] = scale * dy;
    corrections[3 * index + 2] = scale * dz;
}

/// Moves every vertex that holds an anchor by the corrections it receives.
///
/// Entries `entry_offsets[i]..entry_offsets[i + 1]` belong to vertex
/// `touched_vertices[i]`; each names a tether and a signed coefficient that
/// spreads the anchor's step over the two ends of its segment, so the
/// interpolated anchor point moves by exactly that step.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_junction_corrections(
    positions: &mut [f32],
    corrections: &[f32],
    touched_vertices: &[u32],
    entry_offsets: &[u32],
    entry_tethers: &[u32],
    entry_coefficients: &[f32],
    vertex_segments: &[u32],
    segment_radii: &[f32],
    vertex_active: &[u32],
    vertex_pinned: &[u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
    max_step: f32,
) {
    let work = ABSOLUTE_POS;
    if work >= touched_vertices.len() || control[0] == 0 {
        terminate!();
    }
    let index = touched_vertices[work] as usize;
    if vertex_active[index] == 0 || vertex_pinned[index] != 0 {
        terminate!();
    }
    let first = entry_offsets[work] as usize;
    let last = entry_offsets[work + 1] as usize;
    let mut mx = 0.0_f32;
    let mut my = 0.0_f32;
    let mut mz = 0.0_f32;
    for entry in first..last {
        let tether = entry_tethers[entry] as usize;
        let coefficient = entry_coefficients[entry];
        mx += coefficient * corrections[3 * tether];
        my += coefficient * corrections[3 * tether + 1];
        mz += coefficient * corrections[3 * tether + 2];
    }
    let magnitude = (mx * mx + my * my + mz * mz).sqrt();
    if magnitude > max_step {
        let scale = max_step / magnitude;
        mx *= scale;
        my *= scale;
        mz *= scale;
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
        let k = axis as usize;
        let mut step = mx;
        if axis == 1 {
            step = my;
        } else if axis == 2 {
            step = mz;
        }
        let moved = positions[3 * index + k] + step;
        if cell_periodic[k] != 0 {
            positions[3 * index + k] = moved;
        } else {
            positions[3 * index + k] = moved.clamp(cell_lower[k] + radius, cell_upper[k] - radius);
        }
    }
}
