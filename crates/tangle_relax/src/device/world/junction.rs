//! Junction tethers of a resident world and their step in `run_batch`.

use std::collections::BTreeMap;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::DeviceFiberWorld;
use crate::device::cube_grid;
use crate::device::junction::{apply_junction_corrections, find_junction_corrections};
use crate::device::PackedAssembly;

/// Resident tether buffers plus the per-vertex list that applies them.
pub(crate) struct JunctionTethers {
    count: usize,
    vertices: Handle,
    coordinates: Handle,
    rest_lengths: Handle,
    corrections: Handle,
    touched_count: usize,
    touched_vertices: Handle,
    entry_offsets: Handle,
    entry_count: usize,
    entry_tethers: Handle,
    entry_coefficients: Handle,
}

/// Per-vertex view of the tethers: every vertex that holds an anchor, and
/// for each one the tethers that move it with their signed coefficients.
pub(crate) struct TetherEntries {
    pub touched_vertices: Vec<u32>,
    pub entry_offsets: Vec<u32>,
    pub entry_tethers: Vec<u32>,
    pub entry_coefficients: Vec<f32>,
}

impl TetherEntries {
    /// An anchor at fraction `t` of its segment moves by `δ` when the
    /// segment's ends move by `w δ / (w₀² + w₁²)`, with weights `w₀ = 1 − t`
    /// and `w₁ = t`: the smallest end motion that carries the anchor. The
    /// first anchor of a tether steps toward the second (sign +1), the second
    /// back toward the first (sign −1).
    pub fn build(packed: &PackedAssembly) -> Self {
        let mut by_vertex: BTreeMap<u32, Vec<(u32, f32)>> = BTreeMap::new();
        for tether in 0..packed.junction_count() {
            for (side, sign) in [(0, 1.0_f32), (1, -1.0_f32)] {
                let t = packed.junction_coordinates[2 * tether + side];
                let ends = [
                    packed.junction_vertices[4 * tether + 2 * side],
                    packed.junction_vertices[4 * tether + 2 * side + 1],
                ];
                let weights = [1.0 - t, t];
                let norm = weights[0] * weights[0] + weights[1] * weights[1];
                for (vertex, weight) in ends.into_iter().zip(weights) {
                    if weight > 0.0 {
                        by_vertex
                            .entry(vertex)
                            .or_default()
                            .push((tether as u32, sign * weight / norm));
                    }
                }
            }
        }
        let mut entries = Self {
            touched_vertices: Vec::with_capacity(by_vertex.len()),
            entry_offsets: vec![0],
            entry_tethers: Vec::new(),
            entry_coefficients: Vec::new(),
        };
        for (vertex, list) in by_vertex {
            entries.touched_vertices.push(vertex);
            for (tether, coefficient) in list {
                entries.entry_tethers.push(tether);
                entries.entry_coefficients.push(coefficient);
            }
            entries
                .entry_offsets
                .push(entries.entry_tethers.len() as u32);
        }
        entries
    }
}

impl JunctionTethers {
    /// Uploads the tethers of a packed layout, or `None` when it has none.
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, packed: &PackedAssembly) -> Option<Self> {
        let count = packed.junction_count();
        if count == 0 {
            return None;
        }
        assert_eq!(packed.junction_vertices.len(), 4 * count);
        assert_eq!(packed.junction_coordinates.len(), 2 * count);
        let entries = TetherEntries::build(packed);
        Some(Self {
            count,
            vertices: client.create_from_slice(u32::as_bytes(&packed.junction_vertices)),
            coordinates: client.create_from_slice(f32::as_bytes(&packed.junction_coordinates)),
            rest_lengths: client.create_from_slice(f32::as_bytes(&packed.junction_rest_lengths)),
            corrections: client.create_from_slice(f32::as_bytes(&vec![0.0; 3 * count])),
            touched_count: entries.touched_vertices.len(),
            touched_vertices: client.create_from_slice(u32::as_bytes(&entries.touched_vertices)),
            entry_offsets: client.create_from_slice(u32::as_bytes(&entries.entry_offsets)),
            entry_count: entries.entry_tethers.len(),
            entry_tethers: client.create_from_slice(u32::as_bytes(&entries.entry_tethers)),
            entry_coefficients: client
                .create_from_slice(f32::as_bytes(&entries.entry_coefficients)),
        })
    }
}

impl<R: Runtime> DeviceFiberWorld<R> {
    /// One tether correction: find each tether's step, then move the
    /// vertices that hold anchors. Does nothing at zero stiffness or
    /// without tethers.
    pub(super) fn launch_junction_tethers(&self, stiffness: f32, max_step: f32) {
        let Some(tethers) = &self.junction_tethers else {
            return;
        };
        if stiffness <= 0.0 {
            return;
        }
        unsafe {
            find_junction_corrections::launch_unchecked::<R>(
                &self.client,
                cube_grid(tethers.count.div_ceil(64)),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(tethers.vertices.clone(), 4 * tethers.count),
                BufferArg::from_raw_parts(tethers.coordinates.clone(), 2 * tethers.count),
                BufferArg::from_raw_parts(tethers.rest_lengths.clone(), tethers.count),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(self.control.clone(), 4),
                BufferArg::from_raw_parts(tethers.corrections.clone(), 3 * tethers.count),
                stiffness,
                max_step,
            );
            apply_junction_corrections::launch_unchecked::<R>(
                &self.client,
                cube_grid(tethers.touched_count.div_ceil(64)),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(tethers.corrections.clone(), 3 * tethers.count),
                BufferArg::from_raw_parts(tethers.touched_vertices.clone(), tethers.touched_count),
                BufferArg::from_raw_parts(tethers.entry_offsets.clone(), tethers.touched_count + 1),
                BufferArg::from_raw_parts(tethers.entry_tethers.clone(), tethers.entry_count),
                BufferArg::from_raw_parts(tethers.entry_coefficients.clone(), tethers.entry_count),
                BufferArg::from_raw_parts(
                    self.vertex_segments.clone(),
                    self.packed.vertex_segments.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_radii.clone(),
                    self.packed.segment_radii.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_pinned.clone(),
                    self.packed.vertex_pinned.len(),
                ),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(self.control.clone(), 4),
                max_step,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TetherEntries;
    use crate::device::PackedAssembly;
    use tangle_core::{FiberAnchor, FiberAssembly, FiberId, JunctionId, PeriodicCell, Section};

    /// Two parallel fibers 0.1 apart in y, bonded at their midpoints.
    fn bonded_pair() -> FiberAssembly {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.02 });
        for (id, y) in [(1, 0.45), (2, 0.55)] {
            let intrinsic = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]];
            let placed = [[0.3, y, 0.5], [0.5, y, 0.5], [0.7, y, 0.5]];
            assembly
                .add_fiber(FiberId(id), material, section, &intrinsic, &placed)
                .unwrap();
        }
        let law = assembly.junction_laws.add("binder");
        assembly
            .add_junction(
                JunctionId(1),
                law,
                tangle_core::JunctionParameterId(0),
                &[
                    FiberAnchor {
                        fiber: FiberId(1),
                        rest_arc_length: 0.15,
                        section_offset: None,
                    },
                    FiberAnchor {
                        fiber: FiberId(2),
                        rest_arc_length: 0.25,
                        section_offset: None,
                    },
                ],
            )
            .unwrap();
        assembly
    }

    #[test]
    fn packs_one_tether_per_bonded_pair() {
        let packed = PackedAssembly::from_assembly(&bonded_pair()).unwrap();
        assert_eq!(packed.junction_count(), 1);
        // Arc 0.15 sits 3/4 along fiber 1's first segment; arc 0.25 sits
        // 1/4 along fiber 2's second segment (vertices 4 and 5).
        assert_eq!(packed.junction_vertices, vec![0, 1, 4, 5]);
        assert!((packed.junction_coordinates[0] - 0.75).abs() < 1.0e-6);
        assert!((packed.junction_coordinates[1] - 0.25).abs() < 1.0e-6);
        // Anchors at x = 0.45 and 0.55, y 0.45 and 0.55: √0.02 apart.
        assert!((packed.junction_rest_lengths[0] - 0.02_f32.sqrt()).abs() < 1.0e-6);
        assert!(!packed.coarsening_safe);
    }

    #[test]
    fn rest_length_never_pulls_fibers_into_contact() {
        let mut assembly = bonded_pair();
        // Fiber 2 slid onto fiber 1 so both anchors sit at x = 0.45, y = 0.45:
        // the rest length is the two radii, not zero.
        for point in &mut assembly.geometry.placed.positions[3..6] {
            point[0] -= 0.1;
            point[1] = 0.45;
        }
        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        assert!(packed.junction_rest_lengths[0] >= 0.04 - 1.0e-7);
    }

    #[test]
    fn rest_length_leaves_room_to_part_an_overlap_near_the_bond() {
        // Fiber 1 along x, fiber 2 along y 0.03 above it: they cross with
        // their surfaces overlapping by 0.01. The bond holds fiber 1 at
        // x = 0.55, beside the crossing, and fiber 2 at the crossing.
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.02 });
        let intrinsic = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]];
        let lines = [
            [[0.3, 0.5, 0.5], [0.5, 0.5, 0.5], [0.7, 0.5, 0.5]],
            [[0.5, 0.3, 0.53], [0.5, 0.5, 0.53], [0.5, 0.7, 0.53]],
        ];
        for (id, placed) in [(1, lines[0]), (2, lines[1])] {
            assembly
                .add_fiber(FiberId(id), material, section, &intrinsic, &placed)
                .unwrap();
        }
        let law = assembly.junction_laws.add("binder");
        let anchor = |fiber, s| FiberAnchor {
            fiber: FiberId(fiber),
            rest_arc_length: s,
            section_offset: None,
        };
        assembly
            .add_junction(
                JunctionId(1),
                law,
                tangle_core::JunctionParameterId(0),
                &[anchor(1, 0.25), anchor(2, 0.2)],
            )
            .unwrap();
        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        // Anchors √(0.05² + 0.03²) apart, plus the 0.01 contact must open.
        let expected = (0.05_f32.powi(2) + 0.03_f32.powi(2)).sqrt() + 0.01;
        assert!((packed.junction_rest_lengths[0] - expected).abs() < 1.0e-5);
    }

    #[test]
    fn coefficients_move_each_anchor_by_its_step() {
        let packed = PackedAssembly::from_assembly(&bonded_pair()).unwrap();
        let entries = TetherEntries::build(&packed);
        assert_eq!(entries.touched_vertices, vec![0, 1, 4, 5]);
        let coefficient = |vertex: u32| {
            let i = entries
                .touched_vertices
                .iter()
                .position(|&v| v == vertex)
                .unwrap();
            entries.entry_coefficients[entries.entry_offsets[i] as usize]
        };
        // Anchor point motion = Σ wᵢ · coefficientᵢ · δ = ±δ.
        let first = 0.25 * coefficient(0) + 0.75 * coefficient(1);
        let second = 0.75 * coefficient(4) + 0.25 * coefficient(5);
        assert!((first - 1.0).abs() < 1.0e-6, "{first}");
        assert!((second + 1.0).abs() < 1.0e-6, "{second}");
    }

    #[test]
    fn adaptive_tethers_ride_on_the_source_segment_ends() {
        let packed = PackedAssembly::from_assembly_with_options(
            &bonded_pair(),
            Some(crate::device::AdaptiveSegmentationConfig {
                minimum_length_over_diameter: 1.0,
                maximum_refinement_levels: 2,
                ..Default::default()
            }),
            false,
        )
        .unwrap();
        assert_eq!(packed.junction_count(), 1);
        let ends = &packed.junction_vertices;
        for &vertex in ends {
            assert_eq!(packed.vertex_active[vertex as usize], 1);
        }
        // Each source segment of 0.2 splits into 4 leaves (0.05 ≥ 0.04), so
        // fiber 1 spans vertices 0..=8 and fiber 2 starts at 9.
        assert_eq!(ends, &vec![0, 4, 13, 17]);
    }
}
