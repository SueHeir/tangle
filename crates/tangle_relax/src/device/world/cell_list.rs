//! Compact device-resident cell-list construction.

use cubecl::prelude::*;
use cubecl::server::Handle;
use tangle_contact::device::{
    add_cell_block_offsets, clear_cell_list, count_cell_segments, exclusive_scan_cell_blocks,
    scatter_cell_segments,
};

use super::DeviceFiberWorld;

impl<R: Runtime> DeviceFiberWorld<R> {
    fn scan_cell_counts(
        &self,
        input: Handle,
        output: Handle,
        length: usize,
        level: usize,
        control: Handle,
        control_len: usize,
    ) {
        let blocks = length.div_ceil(self.cell_scan_block_size);
        unsafe {
            exclusive_scan_cell_blocks::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(blocks as u32, 1, 1),
                CubeDim::new_1d(self.cell_scan_block_size as u32),
                BufferArg::from_raw_parts(input, length),
                BufferArg::from_raw_parts(output.clone(), length),
                BufferArg::from_raw_parts(self.cell_scan_block_sums[level].clone(), blocks),
                BufferArg::from_raw_parts(control.clone(), control_len),
                length as u32,
                self.cell_scan_block_size,
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
            control.clone(),
            control_len,
        );
        unsafe {
            add_cell_block_offsets::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(blocks as u32, 1, 1),
                CubeDim::new_1d(self.cell_scan_block_size as u32),
                BufferArg::from_raw_parts(output, length),
                BufferArg::from_raw_parts(self.cell_scan_block_offsets[level].clone(), blocks),
                BufferArg::from_raw_parts(control, control_len),
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
                BufferArg::from_raw_parts(self.cell_counts.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_cursors.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_overflow.clone(), 1),
                BufferArg::from_raw_parts(control.clone(), control_len),
            );
            count_cell_segments::launch_unchecked::<R>(
                &self.client,
                segment_cubes.clone(),
                cube_dim.clone(),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_segment_indices.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.cell_counts.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(control.clone(), control_len),
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
            control.clone(),
            control_len,
        );
        unsafe {
            scatter_cell_segments::launch_unchecked::<R>(
                &self.client,
                segment_cubes,
                cube_dim.clone(),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_segment_indices.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.cell_offsets.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_cursors.clone(), cell_count),
                BufferArg::from_raw_parts(self.cell_segments.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(control.clone(), control_len),
                cells_x,
                cells_y,
                cells_z,
            );
        }
    }
}
