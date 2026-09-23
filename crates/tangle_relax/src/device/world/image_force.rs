//! Device-resident CT image and the image-attraction step of `run_batch`.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::DeviceFiberWorld;
use crate::device::image_force::{
    apply_image_corrections, find_image_corrections, RING_TABLE_STRIDE,
};

/// Largest image buffer, below WGPU's default 128 MiB binding limit.
pub(crate) const MAXIMUM_IMAGE_BYTES: usize = 120 << 20;

/// Polar sampling and step parameters of the image force.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageForceSettings {
    /// Fraction of the lateral centroid offset applied per iteration; zero
    /// disables the force.
    pub rate: f32,
    /// Outer sampling radius as a multiple of the vertex radius.
    pub reach_radii: f32,
    /// Gaussian sample-weight width as a multiple of the vertex radius.
    pub sigma_radii: f32,
    /// Number of concentric sampling rings.
    pub rings: u32,
    /// Number of samples on every ring.
    pub spokes: u32,
}

impl Default for ImageForceSettings {
    fn default() -> Self {
        Self {
            rate: 0.0,
            reach_radii: 1.4,
            sigma_radii: 0.8,
            rings: 4,
            spokes: 12,
        }
    }
}

impl ImageForceSettings {
    /// Host-side polar table: per ring `[radius, gaussian × area, area,
    /// support]` in units of the vertex radius, then `[cos, sin]` per spoke.
    fn tables(&self) -> Vec<f32> {
        let rings = self.rings as usize;
        let spokes = self.spokes as usize;
        let reach = self.reach_radii as f64;
        let sigma = self.sigma_radii as f64;
        let mut table = Vec::with_capacity(RING_TABLE_STRIDE * rings + 2 * spokes);
        for ring in 0..rings {
            let inner = reach * ring as f64 / rings as f64;
            let outer = reach * (ring + 1) as f64 / rings as f64;
            let radius = 0.5 * (inner + outer);
            let area = std::f64::consts::PI * (outer * outer - inner * inner) / spokes as f64;
            let gaussian = (-radius * radius / (2.0 * sigma * sigma)).exp();
            // The innermost ring always counts toward the support, so it is
            // defined for any reach; other rings count within half a radius.
            let support = ring == 0 || radius <= 0.5;
            table.extend_from_slice(&[
                radius as f32,
                (gaussian * area) as f32,
                area as f32,
                if support { 1.0 } else { 0.0 },
            ]);
        }
        for spoke in 0..spokes {
            let angle = 2.0 * std::f64::consts::PI * spoke as f64 / spokes as f64;
            table.extend_from_slice(&[angle.cos() as f32, angle.sin() as f32]);
        }
        table
    }
}

/// A resident normalized CT volume plus the scratch buffers of its force.
pub(crate) struct ImageForce {
    image: Handle,
    image_len: usize,
    shape_zyx: [u32; 3],
    origin: [f32; 3],
    inverse_voxel: f32,
    settings: ImageForceSettings,
    tables: Handle,
    tables_len: usize,
    corrections: Handle,
    stats: Handle,
}

impl<R: Runtime> DeviceFiberWorld<R> {
    /// Uploads a normalized CT volume for the image force, replacing any
    /// previous one. The force starts disabled (rate zero).
    ///
    /// `image` is stored `(z, y, x)` with `x` fastest; voxel `[k, j, i]` is
    /// centered at `origin + (i + ½, j + ½, k + ½) · voxel_size` in the same
    /// frame as the fiber positions. Samples outside the volume read zero.
    pub fn set_image(
        &mut self,
        image: &[f32],
        shape_zyx: [usize; 3],
        voxel_size: f32,
        origin: [f32; 3],
    ) {
        let voxels = shape_zyx[0] * shape_zyx[1] * shape_zyx[2];
        assert!(voxels > 0, "image must not be empty");
        assert_eq!(image.len(), voxels, "image length must match its shape");
        assert!(
            image.len() * std::mem::size_of::<f32>() <= MAXIMUM_IMAGE_BYTES,
            "image exceeds the {MAXIMUM_IMAGE_BYTES}-byte device binding limit"
        );
        assert!(voxel_size.is_finite() && voxel_size > 0.0);
        assert!(origin.iter().all(|value| value.is_finite()));
        let settings = self
            .image_force
            .as_ref()
            .map(|force| force.settings)
            .unwrap_or_default();
        let tables = settings.tables();
        let vertices = self.packed.vertex_count();
        self.image_force = Some(ImageForce {
            image: self.client.create_from_slice(f32::as_bytes(image)),
            image_len: image.len(),
            shape_zyx: shape_zyx.map(|n| n as u32),
            origin,
            inverse_voxel: 1.0 / voxel_size,
            settings,
            tables_len: tables.len(),
            tables: self.client.create_from_slice(f32::as_bytes(&tables)),
            corrections: self
                .client
                .create_from_slice(f32::as_bytes(&vec![0.0; 3 * vertices])),
            stats: self
                .client
                .create_from_slice(f32::as_bytes(&vec![0.0; 2 * vertices])),
        });
    }

    /// Sets the image-force parameters; a rate of zero disables the force.
    ///
    /// # Panics
    ///
    /// Panics when no image has been uploaded with [`Self::set_image`].
    pub fn set_image_force(&mut self, settings: ImageForceSettings) {
        assert!(settings.rate.is_finite() && settings.rate >= 0.0);
        assert!(settings.reach_radii.is_finite() && settings.reach_radii > 0.0);
        assert!(settings.sigma_radii.is_finite() && settings.sigma_radii > 0.0);
        assert!(settings.rings > 0 && settings.spokes > 0);
        let tables = settings.tables();
        let tables = self.client.create_from_slice(f32::as_bytes(&tables));
        let force = self
            .image_force
            .as_mut()
            .expect("set_image must be called before set_image_force");
        force.tables_len =
            RING_TABLE_STRIDE * settings.rings as usize + 2 * settings.spokes as usize;
        force.tables = tables;
        force.settings = settings;
    }

    /// Current image-force parameters, if an image is resident.
    pub fn image_force_settings(&self) -> Option<ImageForceSettings> {
        self.image_force.as_ref().map(|force| force.settings)
    }

    /// Whether an image is resident and its force is enabled.
    pub fn image_force_active(&self) -> bool {
        self.image_force
            .as_ref()
            .is_some_and(|force| force.settings.rate > 0.0)
    }

    /// Releases the resident image and disables its force.
    pub fn clear_image(&mut self) {
        self.image_force = None;
    }

    /// Measures per-vertex image statistics at the current positions:
    /// interleaved `(owned mass in voxel², support)` for every packed vertex,
    /// zero for inactive vertices.
    ///
    /// # Panics
    ///
    /// Panics when no image has been uploaded with [`Self::set_image`].
    pub fn image_vertex_stats(&self) -> Vec<f32> {
        let force = self
            .image_force
            .as_ref()
            .expect("set_image must be called before image_vertex_stats");
        self.rebuild_neighbor_lists_if_requested();
        self.launch_find_image_corrections(force, 0.0, 1.0, 0);
        let bytes = self
            .client
            .read_one(force.stats.clone())
            .expect("CubeCL image-statistics readback failed");
        f32::from_bytes(&bytes).to_vec()
    }

    /// One image-force correction: find the lateral steps, then apply them.
    /// Returns immediately when the force is disabled.
    pub(super) fn launch_image_force(&self, force: &ImageForce, max_step: f32) {
        if force.settings.rate <= 0.0 {
            return;
        }
        self.launch_find_image_corrections(force, force.settings.rate, max_step, 1);
        unsafe {
            apply_image_corrections::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(self.positions.clone(), self.packed.positions.len()),
                BufferArg::from_raw_parts(
                    force.corrections.clone(),
                    3 * self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_segments.clone(),
                    self.packed.vertex_segments.len(),
                ),
                BufferArg::from_raw_parts(
                    self.segment_radii.clone(),
                    self.packed.segment_radii.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_vertex_indices.clone(),
                    self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
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
            );
        }
    }

    fn launch_find_image_corrections(
        &self,
        force: &ImageForce,
        rate: f32,
        max_step: f32,
        gate: u32,
    ) {
        unsafe {
            find_image_corrections::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(self.active_vertex_count.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
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
                BufferArg::from_raw_parts(
                    self.vertex_segments.clone(),
                    self.packed.vertex_segments.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_active.clone(),
                    self.packed.vertex_active.len(),
                ),
                BufferArg::from_raw_parts(
                    self.vertex_pinned.clone(),
                    self.packed.vertex_pinned.len(),
                ),
                BufferArg::from_raw_parts(
                    self.active_vertex_indices.clone(),
                    self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(self.active_index_counts.clone(), 2),
                BufferArg::from_raw_parts(self.control.clone(), 4),
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
                BufferArg::from_raw_parts(self.cell_counts.clone(), self.cell_count),
                BufferArg::from_raw_parts(self.cell_offsets.clone(), self.cell_count),
                BufferArg::from_raw_parts(self.cell_segments.clone(), self.packed.segment_count()),
                BufferArg::from_raw_parts(self.cell_lower.clone(), 3),
                BufferArg::from_raw_parts(self.cell_upper.clone(), 3),
                BufferArg::from_raw_parts(self.cell_periodic.clone(), 3),
                BufferArg::from_raw_parts(force.image.clone(), force.image_len),
                BufferArg::from_raw_parts(force.tables.clone(), force.tables_len),
                BufferArg::from_raw_parts(
                    force.corrections.clone(),
                    3 * self.packed.vertex_count(),
                ),
                BufferArg::from_raw_parts(force.stats.clone(), 2 * self.packed.vertex_count()),
                force.origin[0],
                force.origin[1],
                force.origin[2],
                force.inverse_voxel,
                rate,
                force.settings.reach_radii,
                max_step,
                force.shape_zyx[2],
                force.shape_zyx[1],
                force.shape_zyx[0],
                force.settings.rings,
                force.settings.spokes,
                gate,
                self.neighbor_capacity,
                self.cells_x,
                self.cells_y,
                self.cells_z,
            );
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use cubecl::cpu::{CpuDevice, CpuRuntime};
    use tangle_core::{FiberAssembly, FiberId, PeriodicCell, Section};

    use super::ImageForceSettings;
    use crate::{CellListConfig, DeviceFiberWorld, PackedAssembly, RelaxationConfig};

    const SIDE: usize = 24;

    /// Soft bright tubes along x at the given (y, z) centers, voxel size 1.
    fn tube_image(centers: &[[f32; 2]], radius: f32) -> Vec<f32> {
        let mut image = vec![0.0_f32; SIDE * SIDE * SIDE];
        for k in 0..SIDE {
            for j in 0..SIDE {
                let y = j as f32 + 0.5;
                let z = k as f32 + 0.5;
                let value = centers
                    .iter()
                    .map(|[cy, cz]| {
                        let distance = ((y - cy).powi(2) + (z - cz).powi(2)).sqrt();
                        (radius + 0.5 - distance).clamp(0.0, 1.0)
                    })
                    .fold(0.0_f32, f32::max);
                for i in 0..SIDE {
                    image[(k * SIDE + j) * SIDE + i] = value;
                }
            }
        }
        image
    }

    fn straight_fibers(offsets: &[[f64; 2]], radius: f64) -> FiberAssembly {
        let mut assembly =
            FiberAssembly::new(PeriodicCell::orthorhombic([SIDE as f64; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius });
        for (index, [y, z]) in offsets.iter().copied().enumerate() {
            // Segments longer than a diameter: shorter ones put non-adjacent
            // segments of the same fiber in contact, which stretches it.
            let placed: Vec<[f64; 3]> = (0..5).map(|n| [3.0 + 4.5 * n as f64, y, z]).collect();
            assembly
                .add_fiber(
                    FiberId(index as u32 + 1),
                    material,
                    section,
                    &placed,
                    &placed,
                )
                .unwrap();
        }
        assembly
    }

    fn world(assembly: &FiberAssembly) -> DeviceFiberWorld<CpuRuntime> {
        let packed = PackedAssembly::from_assembly(assembly).unwrap();
        DeviceFiberWorld::<CpuRuntime>::upload(
            &CpuDevice::default(),
            packed,
            CellListConfig::default(),
            0.25,
        )
    }

    fn config() -> RelaxationConfig {
        RelaxationConfig {
            force_full_iterations: true,
            max_step: 0.25,
            max_iterations: 1_000,
            iterations_per_batch: 64,
            ..RelaxationConfig::default()
        }
    }

    fn vertex_yz(world: &DeviceFiberWorld<CpuRuntime>) -> Vec<[f32; 2]> {
        world
            .download_positions()
            .chunks_exact(3)
            .map(|xyz| [xyz[1], xyz[2]])
            .collect()
    }

    #[test]
    fn offset_fiber_converges_onto_a_bright_tube() {
        let assembly = straight_fibers(&[[13.0, 12.0]], 2.0);
        let mut world = world(&assembly);
        world.set_image(&tube_image(&[[12.0, 12.0]], 2.0), [SIDE; 3], 1.0, [0.0; 3]);
        world.set_image_force(ImageForceSettings {
            rate: 0.5,
            ..ImageForceSettings::default()
        });
        let config = config();
        let first = world.run_batch(&config, 30);
        let second = world.run_batch(&config, 30);
        assert_eq!(first.batch_iterations, 30);
        assert_eq!(second.total_iterations, 60, "{second:?}");
        for [y, z] in vertex_yz(&world) {
            assert!((y - 12.0).abs() < 0.05, "y = {y}");
            assert!((z - 12.0).abs() < 0.05, "z = {z}");
        }
        let stats = world.image_vertex_stats();
        for vertex in stats.chunks_exact(2) {
            let (mass, support) = (vertex[0], vertex[1]);
            assert!(support > 0.95, "support = {support}");
            // Owned cross-section of a radius-2 tube with a one-voxel soft edge.
            assert!(mass > 10.0 && mass < 16.0, "mass = {mass}");
        }
    }

    #[test]
    fn zero_rate_leaves_positions_unchanged() {
        let assembly = straight_fibers(&[[13.0, 12.0]], 2.0);
        let mut world = world(&assembly);
        world.set_image(&tube_image(&[[12.0, 12.0]], 2.0), [SIDE; 3], 1.0, [0.0; 3]);
        assert!(!world.image_force_active());
        let before = world.download_positions();
        world.run_batch(&config(), 10);
        assert_eq!(world.download_positions(), before);
    }

    #[test]
    fn neighboring_fibers_each_keep_their_own_tube() {
        // Tubes 4.2 apart (a 0.2 gap between radius-2 surfaces); each fiber
        // starts 0.6 toward the other tube and must move back out.
        let assembly = straight_fibers(&[[10.5, 12.0], [13.5, 12.0]], 2.0);
        let mut world = world(&assembly);
        world.set_image(
            &tube_image(&[[9.9, 12.0], [14.1, 12.0]], 2.0),
            [SIDE; 3],
            1.0,
            [0.0; 3],
        );
        world.set_image_force(ImageForceSettings {
            rate: 0.5,
            ..ImageForceSettings::default()
        });
        world.run_batch(&config(), 80);
        let positions = vertex_yz(&world);
        let (first, second) = positions.split_at(positions.len() / 2);
        for [y, _] in first {
            assert!((y - 9.9).abs() < 0.15, "first fiber y = {y}");
        }
        for [y, _] in second {
            assert!((y - 14.1).abs() < 0.15, "second fiber y = {y}");
        }
    }
}
