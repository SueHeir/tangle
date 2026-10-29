//! Native helper behind `tangle.fem`, which builds the meshes in Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tangle_core::AnchorError;
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

/// The bonds of every persistent junction for `tangle.fem.sweep_mesh`: one
/// from a junction's first anchor to each other anchor, as the binder
/// bridges are, as `(junction id, fiber index, segment, coordinate, fiber
/// index, segment, coordinate)`. Each anchor is a fiber index, the segment
/// of that fiber's centerline it lies on and the coordinate along that
/// segment (0 at its first vertex, 1 at its second), so it follows the
/// placed, relaxed, centerline.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn fem_bond_anchors(
    assembly: PyRef<'_, PyAssembly>,
) -> PyResult<Vec<(u32, u32, u32, f64, u32, u32, f64)>> {
    let model = assembly.model.lock().expect("assembly lock poisoned");
    let assembly = &model.assembly;
    let invalid = |error: AnchorError| {
        PyValueError::new_err(format!("a junction anchor is invalid: {error}"))
    };
    let mut bonds = Vec::new();
    for junction in &assembly.junctions.junctions {
        let start = junction.anchors.start as usize;
        let anchors = junction
            .anchors
            .checked_end()
            .and_then(|end| assembly.junctions.anchors.get(start..end as usize))
            .ok_or_else(|| {
                PyValueError::new_err(format!("junction {} has no anchors", junction.id.0))
            })?;
        let Some((first, others)) = anchors.split_first() else {
            continue;
        };
        let first = assembly.resolve_anchor(*first).map_err(invalid)?;
        for other in others {
            let other = assembly.resolve_anchor(*other).map_err(invalid)?;
            bonds.push((
                junction.id.0,
                first.fiber_index,
                first.local_segment,
                first.coordinate,
                other.fiber_index,
                other.local_segment,
                other.coordinate,
            ));
        }
    }
    Ok(bonds)
}
