//! Native helper behind `tangle.fem`, which builds the meshes in Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tangle_export::{voxelize_labels, PumaExportError};

use crate::collection::PyAssembly;

/// Voxel labels on the PuMA grid for `tangle.fem.hex_mesh`: the voxel counts
/// and the phase (u16), owner-fiber (u32) and binder-junction (u32) images as
/// little-endian bytes with x fastest. The junction image is empty without
/// bonds.
#[pyfunction]
#[pyo3(signature = (assembly, voxel_size, bond_radius_ratio=None))]
#[allow(clippy::type_complexity)]
pub(crate) fn fem_voxel_labels<'py>(
    py: Python<'py>,
    assembly: PyRef<'_, PyAssembly>,
    voxel_size: f64,
    bond_radius_ratio: Option<f64>,
) -> PyResult<(
    [usize; 3],
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
)> {
    let model = assembly.model.clone();
    let labels = py
        .detach(move || {
            let model = model.lock().expect("assembly lock poisoned");
            voxelize_labels(&model.assembly, voxel_size, bond_radius_ratio)
        })
        .map_err(|error| match error {
            PumaExportError::InvalidConfig(_)
            | PumaExportError::IncommensurateGrid { .. }
            | PumaExportError::NonOrthorhombicCell => PyValueError::new_err(error.to_string()),
            _ => PyRuntimeError::new_err(error.to_string()),
        })?;
    let phase: Vec<u8> = labels.phase.iter().flat_map(|v| v.to_le_bytes()).collect();
    let fiber: Vec<u8> = labels.fiber.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bond: Vec<u8> = labels.bond.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((
        labels.voxel_counts,
        PyBytes::new(py, &phase),
        PyBytes::new(py, &fiber),
        PyBytes::new(py, &bond),
    ))
}
