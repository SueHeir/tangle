use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tangle_characterize::{
    analyze_neighbors, analyze_shape, score_structure, write_analysis_json, AssemblyMetrics,
    Distribution, NeighborAnalysisConfig, NeighborMetrics, Scorecard, ScorecardConfig,
    ShapeAnalysisConfig, ShapeMetrics,
};
use tangle_core::FiberAssembly;
use tangle_export::PumaExportReport;

use crate::collection::PyAssembly;

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

/// Runs the fiber shape analysis with Python keyword settings.
pub(crate) fn characterize_shape(
    assembly: &FiberAssembly,
    sample_spacing: Option<f64>,
    max_lag: Option<f64>,
    lag_count: usize,
    quantile_count: usize,
    orientation_axis: [f64; 3],
    min_torsion_curvature: Option<f64>,
) -> PyResult<PyShapeReport> {
    let config = ShapeAnalysisConfig {
        sample_spacing,
        maximum_lag: max_lag,
        lag_count,
        quantile_count,
        orientation_axis,
        minimum_torsion_curvature: min_torsion_curvature,
    };
    analyze_shape(assembly, &config)
        .map(|inner| PyShapeReport { inner })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Decodes JSON into plain Python objects.
fn json_to_python(py: Python<'_>, encoded: serde_json::Result<String>) -> PyResult<Py<PyAny>> {
    let encoded = encoded.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(py
        .import("json")?
        .call_method1("loads", (encoded,))?
        .unbind())
}

#[pyclass(name = "ShapeReport", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyShapeReport {
    pub(crate) inner: ShapeMetrics,
}

impl PyShapeReport {
    fn distribution(py: Python<'_>, value: &Option<Distribution>) -> PyResult<Py<PyAny>> {
        json_to_python(py, serde_json::to_string(value))
    }
}

#[pymethods]
impl PyShapeReport {
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    #[getter]
    fn sample_spacing(&self) -> f64 {
        self.inner.sample_spacing
    }

    #[getter]
    fn min_torsion_curvature(&self) -> f64 {
        self.inner.minimum_torsion_curvature
    }

    #[getter]
    fn orientation_axis(&self) -> [f64; 3] {
        self.inner.orientation_axis
    }

    #[getter]
    fn fiber_count(&self) -> usize {
        self.inner.fiber_count
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.samples
    }

    #[getter]
    fn total_length(&self) -> f64 {
        self.inner.total_length
    }

    /// Sampled curvature: ``count``, ``mean``, ``standard_deviation`` and
    /// evenly spaced ``quantiles`` (minimum first, maximum last).
    #[getter]
    fn curvature(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::distribution(py, &self.inner.curvature)
    }

    /// Signed torsion where it is defined; positive is right-handed.
    #[getter]
    fn torsion(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::distribution(py, &self.inner.torsion)
    }

    #[getter]
    fn absolute_torsion(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::distribution(py, &self.inner.absolute_torsion)
    }

    #[getter]
    fn torsion_defined_fraction(&self) -> Option<f64> {
        self.inner.torsion_defined_fraction
    }

    /// Curl index (contour over end-to-end length, minus one), one per fiber.
    #[getter]
    fn curl_index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::distribution(py, &self.inner.curl_index)
    }

    #[getter]
    fn fiber_length(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::distribution(py, &self.inner.fiber_length)
    }

    /// ``|cos θ|`` between sampled chords and the orientation axis.
    #[getter]
    fn axis_cosine(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::distribution(py, &self.inner.axis_cosine)
    }

    #[getter]
    fn mean_squared_axis_cosine(&self) -> Option<f64> {
        self.inner.mean_squared_axis_cosine
    }

    /// Maximum-likelihood Schladitz β: 1 is isotropic, below 1 aligns fibers
    /// with the axis, above 1 lays them in the plane normal to it.
    #[getter]
    fn schladitz_beta(&self) -> Option<f64> {
        self.inner.schladitz_beta
    }

    #[getter]
    fn schladitz_fit_distance(&self) -> Option<f64> {
        self.inner.schladitz_fit_distance
    }

    #[getter]
    fn tangent_correlation_lags(&self) -> Vec<f64> {
        self.inner.tangent_correlation_lags.clone()
    }

    #[getter]
    fn tangent_correlation(&self) -> Vec<Option<f64>> {
        self.inner.tangent_correlation.clone()
    }

    #[getter]
    fn tangent_correlation_length(&self) -> Option<f64> {
        self.inner.tangent_correlation_length
    }

    #[getter]
    fn persistence_length(&self) -> Option<f64> {
        self.inner.persistence_length
    }

    /// Curl index of every fiber in topology order.
    #[getter]
    fn fiber_curl_indices(&self) -> Vec<Option<f64>> {
        self.inner.fibers.iter().map(|f| f.curl_index).collect()
    }

    /// Mean sampled curvature of every fiber in topology order.
    #[getter]
    fn fiber_mean_curvatures(&self) -> Vec<Option<f64>> {
        self.inner.fibers.iter().map(|f| f.mean_curvature).collect()
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
        json_to_python(py, serde_json::to_string(&self.inner))
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
        let format = |value: Option<f64>| value.map_or("None".to_owned(), |v| format!("{v:.4}"));
        format!(
            "ShapeReport(fibers={}, median_curvature={}, persistence_length={}, schladitz_beta={})",
            self.inner.fiber_count,
            format(self.inner.curvature.as_ref().map(Distribution::median)),
            format(self.inner.persistence_length),
            format(self.inner.schladitz_beta),
        )
    }
}

/// Scores a candidate structure against a reference, metric by metric.
#[pyfunction]
#[pyo3(name = "score_structure", signature = (candidate, reference, contact_gap, *, subdivisions=[2, 2, 2], candidate_region=None, reference_region=None, max_candidate_subvolumes=64, min_piece_length=None, sample_spacing=None, neighbor_sample_spacing=None, neighbor_gap=None, in_axis_angle_degrees=20.0, orientation_axis=[0.0, 0.0, 1.0], quantile_count=101))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_structure_py(
    candidate: PyRef<'_, PyAssembly>,
    reference: PyRef<'_, PyAssembly>,
    contact_gap: f64,
    subdivisions: [usize; 3],
    candidate_region: Option<([f64; 3], [f64; 3])>,
    reference_region: Option<([f64; 3], [f64; 3])>,
    max_candidate_subvolumes: usize,
    min_piece_length: Option<f64>,
    sample_spacing: Option<f64>,
    neighbor_sample_spacing: Option<f64>,
    neighbor_gap: Option<f64>,
    in_axis_angle_degrees: f64,
    orientation_axis: [f64; 3],
    quantile_count: usize,
) -> PyResult<PyScorecard> {
    // Copy each assembly out of its lock in turn, so passing the same
    // assembly twice cannot deadlock.
    let candidate = candidate
        .model
        .lock()
        .expect("assembly lock poisoned")
        .assembly
        .clone();
    let reference = reference
        .model
        .lock()
        .expect("assembly lock poisoned")
        .assembly
        .clone();
    let mut config = ScorecardConfig::new(contact_gap);
    config.subdivisions = subdivisions;
    config.candidate_region = candidate_region;
    config.reference_region = reference_region;
    config.maximum_candidate_subvolumes = max_candidate_subvolumes;
    config.minimum_piece_length = min_piece_length;
    config.shape.sample_spacing = sample_spacing;
    config.shape.orientation_axis = orientation_axis;
    config.shape.quantile_count = quantile_count;
    config.neighbors.sample_spacing = neighbor_sample_spacing;
    config.neighbors.neighbor_gap = neighbor_gap;
    config.neighbors.in_axis_angle = in_axis_angle_degrees.to_radians();
    score_structure(&candidate, &reference, &config)
        .map(|inner| PyScorecard { inner })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(name = "Scorecard", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyScorecard {
    pub(crate) inner: Scorecard,
}

#[pymethods]
impl PyScorecard {
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    #[getter]
    fn subvolume_size(&self) -> [f64; 3] {
        self.inner.subvolume_size
    }

    #[getter]
    fn reference_subvolume_count(&self) -> usize {
        self.inner.reference_subvolumes
    }

    #[getter]
    fn candidate_subvolume_count(&self) -> usize {
        self.inner.candidate_subvolumes
    }

    #[getter]
    fn min_piece_length(&self) -> f64 {
        self.inner.minimum_piece_length
    }

    /// One dict per metric: ``metric``, ``distribution``, ``candidate``,
    /// ``reference``, ``distance``, ``reference_spread`` and ``score``.
    #[getter]
    fn rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_python(py, serde_json::to_string(&self.inner.rows))
    }

    /// Score of every metric, keyed by name (``None`` where undefined).
    #[getter]
    fn scores(&self) -> std::collections::BTreeMap<String, Option<f64>> {
        self.inner
            .rows
            .iter()
            .map(|row| (row.metric.clone(), row.score))
            .collect()
    }

    /// Metrics of the whole candidate region.
    #[getter]
    fn candidate_profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_python(py, serde_json::to_string(&self.inner.candidate))
    }

    /// Metrics of the whole reference region.
    #[getter]
    fn reference_profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_python(py, serde_json::to_string(&self.inner.reference))
    }

    /// The rows as a fixed-width text table, worst score first.
    fn table(&self) -> String {
        let mut rows: Vec<_> = self.inner.rows.iter().collect();
        rows.sort_by(|a, b| {
            let key = |score: Option<f64>| score.unwrap_or(f64::NEG_INFINITY);
            key(b.score)
                .partial_cmp(&key(a.score))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let format = |value: Option<f64>| {
            value.map_or("-".to_owned(), |v| {
                if v == 0.0 || (1e-3..1e5).contains(&v.abs()) {
                    format!("{v:.4}")
                } else {
                    format!("{v:.3e}")
                }
            })
        };
        let mut text = format!(
            "{:<28} {:>10} {:>10} {:>10} {:>10} {:>8}\n",
            "metric", "candidate", "reference", "distance", "spread", "score"
        );
        for row in rows {
            text.push_str(&format!(
                "{:<28} {:>10} {:>10} {:>10} {:>10} {:>8}\n",
                row.metric,
                format(row.candidate),
                format(row.reference),
                format(row.distance),
                format(row.reference_spread),
                format(row.score),
            ));
        }
        text
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
        json_to_python(py, serde_json::to_string(&self.inner))
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
        let scored = self
            .inner
            .rows
            .iter()
            .filter(|row| row.score.is_some())
            .count();
        let worst = self
            .inner
            .rows
            .iter()
            .filter_map(|row| row.score.map(|score| (score, &row.metric)))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        match worst {
            Some((score, metric)) => {
                format!("Scorecard(metrics_scored={scored}, worst={metric} at {score:.2})")
            }
            None => format!("Scorecard(metrics_scored={scored})"),
        }
    }
}
