//! Relaxation with CT image attraction, for fitting fibers to a scan.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tangle_relax::{
    DeviceWorld, FiberMotion, ImageForceSettings, PackedAssembly, RelaxationConfig,
};

use crate::collection::PyAssembly;
use crate::settings::PyRelaxationSettings;

/// Largest image the device can bind (WGPU's default limit is 128 MiB).
const MAXIMUM_IMAGE_BYTES: usize = 120 << 20;

/// A resident fiber world plus a normalized CT volume whose image force pulls
/// every vertex toward the bright voxels it owns.
///
/// The world and image are uploaded once; `run` relaxes contact, stretch and
/// bend exactly as `Recipe.run` does, with the image force added each
/// iteration when its rate is positive.
#[pyclass(name = "ImageRelaxer", module = "tangle._tangle")]
pub(crate) struct PyImageRelaxer {
    world: DeviceWorld,
    config: RelaxationConfig,
}

#[pymethods]
impl PyImageRelaxer {
    #[new]
    #[pyo3(signature = (assembly, settings, image, shape_zyx, voxel_size, origin=(0.0, 0.0, 0.0)))]
    fn new(
        assembly: PyRef<'_, PyAssembly>,
        settings: Option<PyRef<'_, PyRelaxationSettings>>,
        image: &[u8],
        shape_zyx: (usize, usize, usize),
        voxel_size: f64,
        origin: (f64, f64, f64),
    ) -> PyResult<Self> {
        let mut config = settings
            .map(|settings| settings.to_rust())
            .transpose()?
            .unwrap_or_else(|| PyRelaxationSettings::default().to_rust().unwrap());
        if config.adaptive_segmentation.is_some() {
            return Err(PyValueError::new_err(
                "ImageRelaxer does not support adaptive_segmentation",
            ));
        }
        if config.motion_model != FiberMotion::Flexible {
            return Err(PyValueError::new_err(
                "ImageRelaxer requires motion_model='flexible'",
            ));
        }
        // Fitting moves every vertex, ends included.
        config.pin_fiber_ends = false;

        let (nz, ny, nx) = shape_zyx;
        let voxels = nz
            .checked_mul(ny)
            .and_then(|n| n.checked_mul(nx))
            .filter(|n| *n > 0)
            .ok_or_else(|| PyValueError::new_err("shape_zyx must be positive"))?;
        if image.len() != 4 * voxels {
            return Err(PyValueError::new_err(format!(
                "image has {} bytes; shape_zyx {:?} needs {} float32 values ({} bytes)",
                image.len(),
                shape_zyx,
                voxels,
                4 * voxels
            )));
        }
        if image.len() > MAXIMUM_IMAGE_BYTES {
            return Err(PyValueError::new_err(format!(
                "image is {} bytes; the device limit is {MAXIMUM_IMAGE_BYTES}",
                image.len()
            )));
        }
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Err(PyValueError::new_err(
                "voxel_size must be positive and finite",
            ));
        }
        if ![origin.0, origin.1, origin.2]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(PyValueError::new_err("origin must be finite"));
        }
        let values: Vec<f32> = image
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();

        let model = assembly.model.lock().expect("assembly lock poisoned");
        if model.assembly.topology.fibers.is_empty() {
            return Err(PyValueError::new_err("assembly contains no fibers"));
        }
        let packed = PackedAssembly::from_assembly_with_options(&model.assembly, None, false)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        drop(model);

        let mut world = DeviceWorld::new(&config, packed);
        world.set_image(
            &values,
            [nz, ny, nx],
            voxel_size as f32,
            [origin.0 as f32, origin.1 as f32, origin.2 as f32],
        );
        Ok(Self { world, config })
    }

    /// Sets the image force; a rate of zero turns it off.
    ///
    /// `rate` is the fraction of the lateral offset to the owned-intensity
    /// centroid applied per iteration (scaled by the axis support).
    #[pyo3(signature = (rate, reach_radii=1.4, sigma_radii=0.8, rings=4, spokes=12))]
    fn set_image_force(
        &mut self,
        rate: f64,
        reach_radii: f64,
        sigma_radii: f64,
        rings: u32,
        spokes: u32,
    ) -> PyResult<()> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(PyValueError::new_err(
                "rate must be finite and non-negative",
            ));
        }
        if !reach_radii.is_finite() || reach_radii <= 0.0 {
            return Err(PyValueError::new_err("reach_radii must be positive"));
        }
        if !sigma_radii.is_finite() || sigma_radii <= 0.0 {
            return Err(PyValueError::new_err("sigma_radii must be positive"));
        }
        if rings == 0 || spokes == 0 {
            return Err(PyValueError::new_err("rings and spokes must be positive"));
        }
        self.world.set_image_force(ImageForceSettings {
            rate: rate as f32,
            reach_radii: reach_radii as f32,
            sigma_radii: sigma_radii as f32,
            rings,
            spokes,
        });
        Ok(())
    }

    /// Pins vertices: one list of flags per fiber, one flag per centerline
    /// node as the fiber was passed in. A pinned node keeps its position
    /// through every correction (contact, stretch, bend, curvature, image)
    /// but still takes part in contact, so pinned fibers are fixed obstacles
    /// for the others. Pass all ``False`` to release every pin.
    fn set_pinned(&mut self, flags: Vec<Vec<bool>>) -> PyResult<()> {
        let spans: Vec<(usize, usize)> = self.fiber_spans().collect();
        if flags.len() != spans.len() {
            return Err(PyValueError::new_err(format!(
                "set_pinned needs one list per fiber ({}), got {}",
                spans.len(),
                flags.len()
            )));
        }
        let mut pinned = vec![0_u32; self.world.packed().vertex_count()];
        for (fiber, ((start, count), fiber_flags)) in spans.iter().zip(&flags).enumerate() {
            if fiber_flags.len() != *count {
                return Err(PyValueError::new_err(format!(
                    "fiber {fiber} has {count} nodes but {} pin flags",
                    fiber_flags.len()
                )));
            }
            for (offset, &flag) in fiber_flags.iter().enumerate() {
                pinned[start + offset] = u32::from(flag);
            }
        }
        self.world.set_vertex_pinned(&pinned);
        Ok(())
    }

    /// Number of pinned nodes.
    #[getter]
    fn pinned_count(&self) -> usize {
        self.world.pinned_vertex_count()
    }

    /// Whether the image force is on.
    #[getter]
    fn image_force_active(&self) -> bool {
        self.world.image_force_active()
    }

    /// Number of fibers in the resident world.
    #[getter]
    fn fiber_count(&self) -> usize {
        self.world.packed().fiber_count()
    }

    /// Runs `iterations` solver iterations and returns a dict with
    /// `iterations`, `converged` (penetration and curvature within the
    /// settings' tolerances), `max_penetration` (meters) and
    /// `max_curvature_ratio`.
    ///
    /// With the image force on, every requested iteration runs; otherwise
    /// the run stops early once contact and curvature are converged, as in
    /// `Recipe.run`.
    fn run(&mut self, py: Python<'_>, iterations: usize) -> PyResult<Py<PyDict>> {
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be positive"));
        }
        let mut config = self.config;
        config.force_full_iterations |= self.world.image_force_active();
        let world = &mut self.world;
        let (done, status) = py.detach(|| {
            let mut done = 0;
            let mut status = None;
            while done < iterations {
                let batch = (iterations - done).min(config.iterations_per_batch.max(1));
                let result = world.run_batch(&config, batch);
                done += result.batch_iterations.min(batch);
                let stop = result.converged || result.batch_iterations < batch;
                status = Some(result);
                if stop {
                    break;
                }
            }
            (done, status.expect("at least one batch runs"))
        });
        let converged = status.max_penetration <= config.penetration_tolerance
            && status.max_curvature_ratio <= 1.0 + config.curvature_ratio_tolerance
            && !status.cell_list_overflow;
        let output = PyDict::new(py);
        output.set_item("iterations", done)?;
        output.set_item("converged", converged)?;
        output.set_item("max_penetration", status.max_penetration as f64)?;
        output.set_item("max_curvature_ratio", status.max_curvature_ratio as f64)?;
        Ok(output.unbind())
    }

    /// Placed centerlines of all fibers in fiber order, in meters.
    fn centerlines(&self) -> Vec<Vec<[f64; 3]>> {
        let positions = self.world.download_positions();
        self.fiber_spans()
            .map(|(start, count)| {
                positions[3 * start..3 * (start + count)]
                    .chunks_exact(3)
                    .map(|xyz| [xyz[0] as f64, xyz[1] as f64, xyz[2] as f64])
                    .collect()
            })
            .collect()
    }

    /// Per-fiber lists of `(mass, support)` for every vertex at the current
    /// positions. `mass` is Σ clip(I, ±1.5) · dA over the cross-section
    /// samples the vertex owns, in voxel² (an area; multiply by the local
    /// vertex spacing in voxels for owned intensity); `support` is the mean
    /// image value near the axis.
    fn vertex_image_stats(&self) -> Vec<Vec<(f64, f64)>> {
        let stats = self.world.image_vertex_stats();
        self.fiber_spans()
            .map(|(start, count)| {
                stats[2 * start..2 * (start + count)]
                    .chunks_exact(2)
                    .map(|pair| (pair[0] as f64, pair[1] as f64))
                    .collect()
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageRelaxer(fibers={}, image_force_active={})",
            self.world.packed().fiber_count(),
            self.world.image_force_active()
        )
    }
}

impl PyImageRelaxer {
    fn fiber_spans(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.world
            .packed()
            .fiber_vertex_spans
            .chunks_exact(2)
            .map(|span| (span[0] as usize, span[1] as usize))
    }
}
