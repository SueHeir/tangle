//! Compact count/scan/scatter uniform-grid construction.

#![allow(missing_docs)]

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn clear_cell_list(
    cell_counts: &mut [Atomic<u32>],
    cell_cursors: &mut [Atomic<u32>],
    overflow: &mut [Atomic<u32>],
    control: &[u32],
) {
    let cell = ABSOLUTE_POS;
    if cell >= cell_counts.len() || control[0] == 0 {
        terminate!();
    }
    cell_counts[cell as usize].store(0);
    cell_cursors[cell as usize].store(0);
    if cell == 0 {
        overflow[0].store(0);
    }
}

#[cube(launch_unchecked)]
pub fn count_cell_segments(
    positions: &[f32],
    segment_vertices: &[u32],
    active_segments: &[u32],
    active_counts: &[Atomic<u32>],
    cell_counts: &mut [Atomic<u32>],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
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
    let first = segment_vertices[2 * segment_index] as usize;
    let second = segment_vertices[2 * segment_index + 1] as usize;
    let midpoint_x = 0.5 * (positions[3 * first] + positions[3 * second]);
    let midpoint_y = 0.5 * (positions[3 * first + 1] + positions[3 * second + 1]);
    let midpoint_z = 0.5 * (positions[3 * first + 2] + positions[3 * second + 2]);
    // Make the numerical hash period exactly equal to the physical cell.
    let width_x = (cell_upper[0] - cell_lower[0]) / cells_x as f32;
    let width_y = (cell_upper[1] - cell_lower[1]) / cells_y as f32;
    let width_z = (cell_upper[2] - cell_lower[2]) / cells_z as f32;
    let raw_x = ((midpoint_x - cell_lower[0]) / width_x).floor() as i32;
    let raw_y = ((midpoint_y - cell_lower[1]) / width_y).floor() as i32;
    let raw_z = ((midpoint_z - cell_lower[2]) / width_z).floor() as i32;
    let mut x = raw_x.clamp(0, cells_x as i32 - 1) as u32;
    let mut y = raw_y.clamp(0, cells_y as i32 - 1) as u32;
    let mut z = raw_z.clamp(0, cells_z as i32 - 1) as u32;
    if cell_periodic[0] != 0 {
        x = ((raw_x % cells_x as i32 + cells_x as i32) % cells_x as i32) as u32;
    }
    if cell_periodic[1] != 0 {
        y = ((raw_y % cells_y as i32 + cells_y as i32) % cells_y as i32) as u32;
    }
    if cell_periodic[2] != 0 {
        z = ((raw_z % cells_z as i32 + cells_z as i32) % cells_z as i32) as u32;
    }
    let cell = (z * cells_y + y) * cells_x + x;
    cell_counts[cell as usize].fetch_add(1);
}

/// Scans one 256-element block and emits the block total for recursive scans.
#[cube(launch_unchecked)]
pub fn exclusive_scan_cell_blocks(
    input: &[u32],
    output: &mut [u32],
    block_sums: &mut [u32],
    control: &[u32],
    length: u32,
    #[comptime] block_size: usize,
) {
    // Every unit reads the same flag, so the whole cube leaves together and
    // no unit is left waiting at a barrier.
    if control[0] == 0 {
        terminate!();
    }
    let mut shared = Shared::new_slice(block_size);
    let lane = UNIT_POS as usize;
    let index = CUBE_POS * block_size + lane;
    let mut value = 0_u32;
    if index < length as usize {
        value = input[index];
    }
    shared[lane] = value;
    sync_cube();
    let mut addend = 0_u32;
    if UNIT_POS >= 1 {
        addend = shared[(UNIT_POS - 1) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 2 {
        addend = shared[(UNIT_POS - 2) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 4 {
        addend = shared[(UNIT_POS - 4) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 8 {
        addend = shared[(UNIT_POS - 8) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 16 {
        addend = shared[(UNIT_POS - 16) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 32 {
        addend = shared[(UNIT_POS - 32) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 64 {
        addend = shared[(UNIT_POS - 64) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    addend = 0;
    if UNIT_POS >= 128 {
        addend = shared[(UNIT_POS - 128) as usize];
    }
    sync_cube();
    shared[lane] += addend;
    sync_cube();
    if index < length as usize {
        let mut exclusive = 0_u32;
        if UNIT_POS > 0 {
            exclusive = shared[lane - 1];
        }
        output[index] = exclusive;
    }
    if UNIT_POS as usize == block_size - 1 {
        block_sums[CUBE_POS] = shared[lane];
    }
}

/// Adds recursively scanned block totals to one scan level.
#[cube(launch_unchecked)]
pub fn add_cell_block_offsets(
    output: &mut [u32],
    block_offsets: &[u32],
    control: &[u32],
    length: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= length as usize || control[0] == 0 {
        terminate!();
    }
    output[index as usize] += block_offsets[CUBE_POS];
}

#[cube(launch_unchecked)]
pub fn scatter_cell_segments(
    positions: &[f32],
    segment_vertices: &[u32],
    active_segments: &[u32],
    active_counts: &[Atomic<u32>],
    cell_offsets: &[u32],
    cell_cursors: &mut [Atomic<u32>],
    cell_segments: &mut [u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
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
    let first = segment_vertices[2 * segment_index] as usize;
    let second = segment_vertices[2 * segment_index + 1] as usize;
    let midpoint_x = 0.5 * (positions[3 * first] + positions[3 * second]);
    let midpoint_y = 0.5 * (positions[3 * first + 1] + positions[3 * second + 1]);
    let midpoint_z = 0.5 * (positions[3 * first + 2] + positions[3 * second + 2]);
    let width_x = (cell_upper[0] - cell_lower[0]) / cells_x as f32;
    let width_y = (cell_upper[1] - cell_lower[1]) / cells_y as f32;
    let width_z = (cell_upper[2] - cell_lower[2]) / cells_z as f32;
    let raw_x = ((midpoint_x - cell_lower[0]) / width_x).floor() as i32;
    let raw_y = ((midpoint_y - cell_lower[1]) / width_y).floor() as i32;
    let raw_z = ((midpoint_z - cell_lower[2]) / width_z).floor() as i32;
    let mut x = raw_x.clamp(0, cells_x as i32 - 1) as u32;
    let mut y = raw_y.clamp(0, cells_y as i32 - 1) as u32;
    let mut z = raw_z.clamp(0, cells_z as i32 - 1) as u32;
    if cell_periodic[0] != 0 {
        x = ((raw_x % cells_x as i32 + cells_x as i32) % cells_x as i32) as u32;
    }
    if cell_periodic[1] != 0 {
        y = ((raw_y % cells_y as i32 + cells_y as i32) % cells_y as i32) as u32;
    }
    if cell_periodic[2] != 0 {
        z = ((raw_z % cells_z as i32 + cells_z as i32) % cells_z as i32) as u32;
    }
    let cell = (z * cells_y + y) * cells_x + x;
    let slot = cell_cursors[cell as usize].fetch_add(1);
    let start = cell_offsets[cell as usize];
    cell_segments[(start + slot) as usize] = segment;
}
