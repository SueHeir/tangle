//! Compact device-resident cell-list construction.

use cubecl::prelude::*;
use cubecl::server::Handle;
use tangle_contact::device::{
    add_cell_block_offsets, clear_cell_list, count_cell_segments, exclusive_scan_cell_blocks,
    scatter_cell_segments,
};

use super::{DeviceFiberWorld, CELL_SCAN_BLOCK_SIZE};

impl<R: Runtime> DeviceFiberWorld<R> {
    fn scan_cell_counts(&self, input: Handle, output: Handle, length: usize, level: usize) {
        let blocks = length.div_ceil(CELL_SCAN_BLOCK_SIZE);
        unsafe {
            exclusive_scan_cell_blocks::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(blocks as u32, 1, 1),
                CubeDim::new_1d(CELL_SCAN_BLOCK_SIZE as u32),
                ArrayArg::from_raw_parts(input, length),
                ArrayArg::from_raw_parts(output.clone(), length),
                ArrayArg::from_raw_parts(self.cell_scan_block_sums[level].clone(), blocks),
                length as u32,
            );
        }
        if blocks == 1 {
            return;
        }

        self.scan_cell_counts(
            self.cell_scan_block_sums[level].clone(),
            self.cell_scan_block_offsets[level].clone(),
            blocks,
            level + 1,
        );
        unsafe {
            add_cell_block_offsets::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(blocks as u32, 1, 1),
                CubeDim::new_1d(CELL_SCAN_BLOCK_SIZE as u32),
                ArrayArg::from_raw_parts(output, length),
                ArrayArg::from_raw_parts(self.cell_scan_block_offsets[level].clone(), blocks),
                length as u32,
            );
        }
    }

    pub(super) fn rebuild_cell_list(
        &self,
        cell_count: usize,
        cells_x: u32,
        cells_y: u32,
        cells_z: u32,
        control: Handle,
        control_len: usize,
    ) {
        let cube_dim = CubeDim::new_1d(64);
        let cell_cubes = CubeCount::Static(cell_count.div_ceil(64) as u32, 1, 1);
        let segment_cubes = CubeCount::Static(self.active_segment_count.div_ceil(64) as u32, 1, 1);
        unsafe {
            clear_cell_list::launch_unchecked::<R>(
                &self.client,
                cell_cubes.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(self.cell_counts.clone(), cell_count),
                ArrayArg::from_raw_parts(self.cell_cursors.clone(), cell_count),
                ArrayArg::from_raw_parts(self.cell_overflow.clone(), 1),
                ArrayArg::from_raw_parts(control.clone(), control_len),
            );
            count_cell_segments::launch_unchecked::<R>(
                &self.client,
                segment_cubes.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                ArrayArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                ArrayArg::from_raw_parts(
                    self.active_segment_indices.clone(),
                    self.packed.segment_count(),
                ),
                ArrayArg::from_raw_parts(self.active_index_counts.clone(), 2),
                ArrayArg::from_raw_parts(self.cell_counts.clone(), cell_count),
                ArrayArg::from_raw_parts(self.cell_lower.clone(), 3),
                ArrayArg::from_raw_parts(self.cell_upper.clone(), 3),
                ArrayArg::from_raw_parts(self.cell_periodic.clone(), 3),
                ArrayArg::from_raw_parts(control.clone(), control_len),
                cells_x,
                cells_y,
                cells_z,
            );
        }
        self.scan_cell_counts(
            self.cell_counts.clone(),
            self.cell_offsets.clone(),
            cell_count,
            0,
        );
        unsafe {
            scatter_cell_segments::launch_unchecked::<R>(
                &self.client,
                segment_cubes,
                cube_dim.clone(),
                ArrayArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                ArrayArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                ArrayArg::from_raw_parts(
                    self.active_segment_indices.clone(),
                    self.packed.segment_count(),
                ),
                ArrayArg::from_raw_parts(self.active_index_counts.clone(), 2),
                ArrayArg::from_raw_parts(self.cell_offsets.clone(), cell_count),
                ArrayArg::from_raw_parts(self.cell_cursors.clone(), cell_count),
                ArrayArg::from_raw_parts(self.cell_segments.clone(), self.packed.segment_count()),
                ArrayArg::from_raw_parts(self.cell_lower.clone(), 3),
                ArrayArg::from_raw_parts(self.cell_upper.clone(), 3),
                ArrayArg::from_raw_parts(self.cell_periodic.clone(), 3),
                ArrayArg::from_raw_parts(control.clone(), control_len),
                cells_x,
                cells_y,
                cells_z,
            );
        }
    }
}
