//! Device-resident Verlet neighbor lists for the contact pass.

use cubecl::prelude::*;
use tangle_contact::device::{
    build_segment_neighbor_lists, finish_neighbor_list_rebuild, flag_neighbor_list_displacement,
    gather_cell_slot_geometry, request_neighbor_list_rebuild,
    snapshot_neighbor_reference_positions,
};

use super::DeviceFiberWorld;
use crate::device::kernels::request_neighbor_list_rebuild_after_adaptation;

impl<R: Runtime> DeviceFiberWorld<R> {
    /// Rebuilds the cell list and neighbor lists when a rebuild is pending.
    ///
    /// Every kernel here checks the device-side request flag, so the host
    /// never waits on it; when no rebuild is pending they return immediately.
    pub(super) fn rebuild_neighbor_lists_if_requested(&self) {
        self.rebuild_cell_list(
            self.cell_count,
            self.cells_x,
            self.cells_y,
            self.cells_z,
            self.neighbor_state.clone(),
            2,
        );
        let cube_dim = CubeDim::new_1d(64);
        let segment_cubes = CubeCount::Static(self.active_segment_count.div_ceil(64) as u32, 1, 1);
        unsafe {
            gather_cell_slot_geometry::launch_unchecked::<R>(
                &self.client,
                segment_cubes.clone(),
                cube_dim.clone(),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.segment_vertices.clone(),
                    self.packed.segment_vertices.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_fibers.clone(),
                    self.packed.segment_fibers.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_radii.clone(),
                    self.packed.segment_radii.len(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.cell_segments.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
                BufferArg::from_raw_parts(
                    self.slot_geometry.clone(),
                    8 * self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.slot_topology.clone(),
                    3 * self.packed.segment_count(),
                ),
            );
            build_segment_neighbor_lists::launch_unchecked::<R>(
                &self.client,
                segment_cubes,
                cube_dim.clone(),
                BufferArg::from_raw_parts(
                    self.slot_geometry.clone(),
                    8 * self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.slot_topology.clone(),
                    3 * self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.cell_counts.clone(), self.cell_count),
                BufferArg::from_raw_parts(self.cell_offsets.clone(), self.cell_count),
                BufferArg::from_raw_parts(self.cell_segments.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
                BufferArg::from_raw_parts(
                    self.neighbor_counts.clone(),
                    self.packed.segment_count(),
                ),
                BufferArg::from_raw_parts(
                    self.neighbor_segments.clone(),
                    self.packed.segment_count() * self.neighbor_capacity as usize,
                ),
                BufferArg::from_raw_parts(
                    self.neighbor_home_cells.clone(),
                    self.packed.segment_count(),
                ),
                self.neighbor_skin,
                self.neighbor_capacity,
                self.cells_x,
                self.cells_y,
                self.cells_z,
            );
            snapshot_neighbor_reference_positions::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.packed.positions.len().div_ceil(64) as u32, 1, 1),
                cube_dim,
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.neighbor_reference_positions.clone(),
                    self.packed.positions.len(),
                ),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
            );
            finish_neighbor_list_rebuild::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
            );
        }
    }

    /// Marks the neighbor lists stale after geometry or activity changed
    /// outside the displacement-tracked relaxation loop.
    pub(super) fn request_neighbor_list_rebuild(&self) {
        unsafe {
            request_neighbor_list_rebuild::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
            );
        }
    }

    /// Requests a rebuild when the last adaptation epoch split or merged any
    /// segment, without a host readback.
    pub(super) fn request_neighbor_list_rebuild_after_adaptation(&self) {
        unsafe {
            request_neighbor_list_rebuild_after_adaptation::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(self.refinement_count.clone(), 6),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
            );
        }
    }

    /// Requests a rebuild once any active vertex has moved half the skin
    /// since the lists were built.
    pub(super) fn flag_neighbor_list_displacement(&self) {
        unsafe {
            flag_neighbor_list_displacement::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    self.neighbor_reference_positions.clone(),
                    self.packed.positions.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_vertex_indices.clone(),
                    self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.neighbor_state.clone(), 2),
                0.5 * self.neighbor_skin,
            );
        }
    }

    /// Number of neighbor-list rebuilds performed since upload.
    ///
    /// This is a diagnostic readback: it synchronizes with the device.
    pub fn neighbor_list_rebuilds(&self) -> u32 {
        let bytes = self
            .client
            .read_one(self.neighbor_state.clone())
            .expect("CubeCL neighbor-list state readback failed");
        u32::from_bytes(&bytes)[1]
    }
}
