use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tangle_characterize::{
    analyze_neighbors, write_analysis_json, AssemblyMetrics, NeighborAnalysisConfig,
    NeighborMetrics,
};
use tangle_core::FiberAssembly;
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
    fn max_curvature(&self) -> f64 {
        self.inner.maximum_curvature
    }

    #[getter]
    fn max_curvature_ratio(&self) -> f64 {
        self.inner.maximum_curvature_ratio
    }

    /// Vertices whose curvature exceeds their material's bend limit.
    #[getter]
    fn curvature_limit_violations(&self) -> usize {
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

/// Runs the contact and neighbor analysis with Python keyword settings.
#[allow(clippy::too_many_arguments)]
pub(crate) fn characterize_neighbors(
    assembly: &FiberAssembly,
    contact_gap: f64,
    neighbor_gap: Option<f64>,
    in_axis_angle_degrees: f64,
    sample_spacing: Option<f64>,
    max_lag: Option<f64>,
    lag_count: usize,
) -> PyResult<PyNeighborReport> {
    let mut config = NeighborAnalysisConfig::new(contact_gap);
    config.neighbor_gap = neighbor_gap;
    config.in_axis_angle = in_axis_angle_degrees.to_radians();
    config.sample_spacing = sample_spacing;
    config.maximum_lag = max_lag;
    config.lag_count = lag_count;
    analyze_neighbors(assembly, &config)
        .map(|inner| PyNeighborReport { inner })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(name = "NeighborReport", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyNeighborReport {
    pub(crate) inner: NeighborMetrics,
}

#[pymethods]
impl PyNeighborReport {
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    #[getter]
    fn contact_gap(&self) -> f64 {
        self.inner.contact_gap
    }

    #[getter]
    fn neighbor_gap(&self) -> f64 {
        self.inner.neighbor_gap
    }

    #[getter]
    fn sample_spacing(&self) -> f64 {
        self.inner.sample_spacing
    }

    #[getter]
    fn contact_count(&self) -> usize {
        self.inner.contacts
    }

    #[getter]
    fn contacts_per_length(&self) -> f64 {
        self.inner.contacts_per_length
    }

    #[getter]
    fn random_baseline_contacts_per_length(&self) -> Option<f64> {
        self.inner.random_baseline_contacts_per_length
    }

    #[getter]
    fn contact_ratio_to_random(&self) -> Option<f64> {
        self.inner.contact_ratio_to_random
    }

    #[getter]
    fn contact_count_dispersion(&self) -> Option<f64> {
        self.inner.contact_count_dispersion
    }

    #[getter]
    fn in_axis_contact_fraction(&self) -> f64 {
        self.inner.in_axis_contact_fraction
    }

    #[getter]
    fn median_crossing_angle_degrees(&self) -> Option<f64> {
        self.inner.median_crossing_angle.map(f64::to_degrees)
    }

    #[getter]
    fn median_excess_persistence(&self) -> Option<f64> {
        self.inner.median_excess_persistence
    }

    #[getter]
    fn median_in_axis_contact_length(&self) -> Option<f64> {
        self.inner.median_in_axis_contact_length
    }

    #[getter]
    fn mean_free_length(&self) -> Option<f64> {
        self.inner.mean_free_length
    }

    #[getter]
    fn mean_neighbors(&self) -> f64 {
        self.inner.mean_neighbors
    }

    #[getter]
    fn mean_in_axis_neighbors(&self) -> f64 {
        self.inner.mean_in_axis_neighbors
    }

    #[getter]
    fn neighbor_count_histogram(&self) -> Vec<usize> {
        self.inner.neighbor_count_histogram.clone()
    }

    #[getter]
    fn in_axis_neighbor_count_histogram(&self) -> Vec<usize> {
        self.inner.in_axis_neighbor_count_histogram.clone()
    }

    #[getter]
    fn turnover_lags(&self) -> Vec<f64> {
        self.inner.turnover_lags.clone()
    }

    #[getter]
    fn neighbor_turnover(&self) -> Vec<Option<f64>> {
        self.inner.neighbor_turnover.clone()
    }

    #[getter]
    fn in_axis_neighbor_turnover(&self) -> Vec<Option<f64>> {
        self.inner.in_axis_neighbor_turnover.clone()
    }

    #[getter]
    fn neighbor_correlation_length(&self) -> Option<f64> {
        self.inner.neighbor_correlation_length
    }

    #[getter]
    fn in_axis_correlation_length(&self) -> Option<f64> {
        self.inner.in_axis_correlation_length
    }

    #[getter]
    fn free_lengths(&self) -> Vec<f64> {
        self.inner.free_lengths.clone()
    }

    /// Crossing angle of every contact event, in degrees.
    #[getter]
    fn crossing_angles_degrees(&self) -> Vec<f64> {
        self.inner
            .events
            .iter()
            .map(|event| event.crossing_angle.to_degrees())
            .collect()
    }

    /// Excess persistence of every out-of-axis contact event.
    #[getter]
    fn excess_persistence(&self) -> Vec<f64> {
        self.inner
            .events
            .iter()
            .filter(|event| !event.in_axis)
            .map(|event| event.excess_persistence())
            .collect()
    }

    /// Contact events per unit length for every fiber.
    #[getter]
    fn contacts_per_fiber_length(&self) -> Vec<f64> {
        self.inner
            .fibers
            .iter()
            .map(|fiber| fiber.contacts as f64 / fiber.length)
            .collect()
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
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
        let encoded = serde_json::to_string_pretty(&self.inner)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        std::fs::write(path, encoded).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborReport(contacts={}, contacts_per_length={:.6e}, in_axis_contact_fraction={:.3})",
            self.inner.contacts, self.inner.contacts_per_length, self.inner.in_axis_contact_fraction
        )
    }
}
