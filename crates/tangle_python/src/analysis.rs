use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tangle_characterize::{write_analysis_json, AssemblyMetrics};
use tangle_export::PumaExportReport;

#[pyclass(name = "AnalysisReport", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyAnalysisReport {
    pub(crate) inner: AssemblyMetrics,
}

#[pymethods]
impl PyAnalysisReport {
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    #[getter]
    fn fiber_count(&self) -> usize {
        self.inner.fibers
    }

    #[getter]
    fn segment_count(&self) -> usize {
        self.inner.segments
    }

    #[getter]
    fn vertex_count(&self) -> usize {
        self.inner.vertices
    }

    #[getter]
    fn junction_count(&self) -> usize {
        self.inner.junctions
    }

    #[getter]
    fn total_centerline_length(&self) -> f64 {
        self.inner.total_centerline_length
    }

    #[getter]
    fn nominal_swept_volume_fraction(&self) -> f64 {
        self.inner.nominal_swept_volume_fraction
    }

    #[getter]
    fn length_weighted_orientation_tensor(&self) -> [[f64; 3]; 3] {
        self.inner.orientation_tensor
    }

    #[getter]
    fn volume_weighted_orientation_tensor(&self) -> [[f64; 3]; 3] {
        self.inner.volume_weighted_orientation_tensor
    }

    #[getter]
    fn maximum_curvature(&self) -> f64 {
        self.inner.maximum_curvature
    }

    #[getter]
    fn maximum_curvature_ratio(&self) -> f64 {
        self.inner.maximum_curvature_ratio
    }

    #[getter]
    fn bend_limit_violations(&self) -> usize {
        self.inner.bend_limit_violations
    }

    #[pyo3(signature = (pretty=true))]
    fn to_json(&self, pretty: bool) -> PyResult<String> {
        if pretty {
            serde_json::to_string_pretty(&self.inner)
        } else {
            serde_json::to_string(&self.inner)
        }
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let encoded = serde_json::to_string(&self.inner)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(py
            .import("json")?
            .call_method1("loads", (encoded,))?
            .unbind())
    }

    fn write_json(&self, path: PathBuf) -> PyResult<()> {
        write_analysis_json(&self.inner, path)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "AnalysisReport(fibers={}, segments={}, nominal_swept_volume_fraction={:.6})",
            self.inner.fibers, self.inner.segments, self.inner.nominal_swept_volume_fraction
        )
    }
}

#[pyclass(name = "PumaExportReport", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyPumaExportReport {
    pub(crate) inner: PumaExportReport,
}

#[pymethods]
impl PyPumaExportReport {
    #[getter]
    fn output_directory(&self) -> PathBuf {
        self.inner.output_directory.clone()
    }

    #[getter]
    fn voxel_counts(&self) -> [usize; 3] {
        self.inner.voxel_counts
    }

    #[getter]
    fn total_voxels(&self) -> usize {
        self.inner.total_voxels
    }

    #[getter]
    fn occupied_voxels(&self) -> usize {
        self.inner.occupied_voxels
    }

    #[getter]
    fn voxel_volume_fraction(&self) -> f64 {
        self.inner.voxel_volume_fraction
    }

    #[getter]
    fn ambiguous_voxels(&self) -> usize {
        self.inner.ambiguous_voxels
    }

    #[getter]
    fn domain_path(&self) -> PathBuf {
        self.inner.domain_path.clone()
    }

    #[getter]
    fn fiber_ids_path(&self) -> Option<PathBuf> {
        self.inner.fiber_ids_path.clone()
    }

    #[getter]
    fn interface_path(&self) -> Option<PathBuf> {
        self.inner.interface_path.clone()
    }

    #[getter]
    fn manifest_path(&self) -> PathBuf {
        self.inner.manifest_path.clone()
    }

    #[getter]
    fn analysis_path(&self) -> PathBuf {
        self.inner.analysis_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PumaExportReport(voxel_counts={:?}, occupied_voxels={}, ambiguous_voxels={})",
            self.inner.voxel_counts, self.inner.occupied_voxels, self.inner.ambiguous_voxels
        )
    }
}
