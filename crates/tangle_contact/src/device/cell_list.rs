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

/// Grid cell holding the midpoint of piece `proxy` when the segment from
/// `p` to `q` is split into `proxies` equal pieces.
///
/// Binning long segments as several short proxies keeps the grid sized for
/// the short (refined) segments. With one proxy this is the segment midpoint,
/// computed exactly as before proxies existed. Every kernel that locates a
/// proxy uses this function, so they agree bit for bit.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn proxy_cell(
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    proxy: u32,
    proxies: u32,
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
) -> u32 {
    let mut midpoint_x = 0.5 * (px + qx);
    let mut midpoint_y = 0.5 * (py + qy);
    let mut midpoint_z = 0.5 * (pz + qz);
    if proxies > 1 {
        let fraction = (proxy as f32 + 0.5) / proxies as f32;
        midpoint_x = px + (qx - px) * fraction;
        midpoint_y = py + (qy - py) * fraction;
        midpoint_z = pz + (qz - pz) * fraction;
    }
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
    (z * cells_y + y) * cells_x + x
}

/// Proxy piece of a `proxies`-piece segment containing the point at
/// `parameter` in `[0, 1]` along it.
#[cube]
pub fn proxy_of(parameter: f32, proxies: u32) -> u32 {
    let piece = (parameter * proxies as f32).floor() as u32;
    piece.min(proxies - 1)
}

/// Counts the proxies binned into each cell. `use_proxies == 0` bins every
/// segment whole (one proxy), as the contact-capture grid requires.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn count_cell_segments(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_proxies: &[u32],
    active_segments: &[u32],
    active_counts: &[Atomic<u32>],
    cell_counts: &mut [Atomic<u32>],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
    use_proxies: u32,
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
    let mut proxies = 1_u32;
    if use_proxies != 0 {
        proxies = segment_proxies[segment_index];
    }
    for proxy in 0..proxies {
        let cell = proxy_cell(
            positions[3 * first],
            positions[3 * first + 1],
            positions[3 * first + 2],
            positions[3 * second],
            positions[3 * second + 1],
            positions[3 * second + 2],
            proxy,
            proxies,
            cell_lower,
            cell_upper,
            cell_periodic,
            cells_x,
            cells_y,
            cells_z,
        );
        cell_counts[cell as usize].fetch_add(1);
    }
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

/// Places every proxy in its cell's slot range. The atomic cursor fills a
/// cell in a run-dependent order, so this writes to scratch slots (the
/// segment in `scattered_segments`, the piece in `scattered_proxies`, the
/// cell in `slot_cells`) that [`rank_cell_segments`] then puts in order.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_cell_segments(
    positions: &[f32],
    segment_vertices: &[u32],
    segment_proxies: &[u32],
    active_segments: &[u32],
    active_counts: &[Atomic<u32>],
    cell_offsets: &[u32],
    cell_cursors: &mut [Atomic<u32>],
    scattered_segments: &mut [u32],
    scattered_proxies: &mut [u32],
    slot_cells: &mut [u32],
    cell_lower: &[f32],
    cell_upper: &[f32],
    cell_periodic: &[u32],
    control: &[u32],
    use_proxies: u32,
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
    let mut proxies = 1_u32;
    if use_proxies != 0 {
        proxies = segment_proxies[segment_index];
    }
    for proxy in 0..proxies {
        let cell = proxy_cell(
            positions[3 * first],
            positions[3 * first + 1],
            positions[3 * first + 2],
            positions[3 * second],
            positions[3 * second + 1],
            positions[3 * second + 2],
            proxy,
            proxies,
            cell_lower,
            cell_upper,
            cell_periodic,
            cells_x,
            cells_y,
            cells_z,
        );
        let slot = cell_cursors[cell as usize].fetch_add(1);
        let index = (cell_offsets[cell as usize] + slot) as usize;
        scattered_segments[index] = segment;
        scattered_proxies[index] = proxy;
        slot_cells[index] = cell;
    }
}

/// Writes each cell's slots to `cell_segments`/`cell_proxies` in ascending
/// (segment, piece) order.
///
/// Each scratch slot counts the smaller keys in its own cell and moves to
/// that rank, so neighbor lists, the overflow cell scan and contact capture
/// see the same candidate order in every run, and float sums over candidates
/// are bitwise repeatable. `cell_count` is the allocated cell count; the
/// filled slots end at its last offset plus count.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn rank_cell_segments(
    cell_counts: &[u32],
    cell_offsets: &[u32],
    scattered_segments: &[u32],
    scattered_proxies: &[u32],
    slot_cells: &[u32],
    cell_segments: &mut [u32],
    cell_proxies: &mut [u32],
    control: &[u32],
    cell_count: u32,
) {
    let slot = ABSOLUTE_POS;
    let last = (cell_count - 1) as usize;
    if control[0] == 0 || slot >= (cell_offsets[last] + cell_counts[last]) as usize {
        terminate!();
    }
    let cell = slot_cells[slot] as usize;
    let start = cell_offsets[cell] as usize;
    let segment = scattered_segments[slot];
    let proxy = scattered_proxies[slot];
    let mut rank = 0_usize;
    for other in 0..cell_counts[cell] {
        let other_slot = start + other as usize;
        let other_segment = scattered_segments[other_slot];
        if other_segment < segment
            || (other_segment == segment && scattered_proxies[other_slot] < proxy)
        {
            rank += 1;
        }
    }
    cell_segments[start + rank] = segment;
    cell_proxies[start + rank] = proxy;
}
