use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tangle_core::JunctionParameterId;
use tangle_generate::{JunctionCapturePolicy, JunctionMaterialPair};

/// Deterministic policy for promoting current contacts to persistent junctions.
#[pyclass(name = "JunctionPolicy", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyJunctionPolicy {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub law_name: String,
    #[pyo3(get, set)]
    pub parameter_set: u32,
    #[pyo3(get, set)]
    pub maximum_surface_gap: f32,
    #[pyo3(get, set)]
    pub minimum_crossing_angle: f32,
    #[pyo3(get, set)]
    pub maximum_crossing_angle: f32,
    #[pyo3(get, set)]
    pub probability: f32,
    #[pyo3(get, set)]
    pub seed: u64,
    #[pyo3(get, set)]
    pub material_pairs: Vec<(String, String)>,
    #[pyo3(get, set)]
    pub maximum_per_fiber_pair: usize,
    #[pyo3(get, set)]
    pub minimum_anchor_separation: f64,
    #[pyo3(get, set)]
    pub candidate_capacity: usize,
}

#[pymethods]
impl PyJunctionPolicy {
    #[new]
    #[pyo3(signature = (name="touching contacts".to_string(), law_name="bond".to_string()))]
    fn new(name: String, law_name: String) -> Self {
        let policy = JunctionCapturePolicy::touching(name, law_name);
        Self {
            name: policy.name,
            law_name: policy.law_name,
            parameter_set: policy.parameters.0,
            maximum_surface_gap: policy.maximum_surface_gap,
            minimum_crossing_angle: policy.minimum_crossing_angle,
            maximum_crossing_angle: policy.maximum_crossing_angle,
            probability: policy.probability,
            seed: policy.seed,
            material_pairs: Vec::new(),
            maximum_per_fiber_pair: policy.maximum_per_fiber_pair,
            minimum_anchor_separation: policy.minimum_anchor_separation,
            candidate_capacity: policy.candidate_capacity,
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "JunctionPolicy(name={:?}, law_name={:?}, probability={}, material_pairs={})",
            self.name,
            self.law_name,
            self.probability,
            self.material_pairs.len()
        )
    }
}

impl PyJunctionPolicy {
    pub(crate) fn to_rust(&self) -> PyResult<JunctionCapturePolicy> {
        if self.name.trim().is_empty() || self.law_name.trim().is_empty() {
            return Err(PyValueError::new_err(
                "junction policy and law names must not be empty",
            ));
        }
        if !self.maximum_surface_gap.is_finite() || self.maximum_surface_gap < 0.0 {
            return Err(PyValueError::new_err(
                "maximum_surface_gap must be nonnegative and finite",
            ));
        }
        if !self.minimum_crossing_angle.is_finite()
            || !self.maximum_crossing_angle.is_finite()
            || self.minimum_crossing_angle < 0.0
            || self.maximum_crossing_angle > std::f32::consts::FRAC_PI_2
            || self.minimum_crossing_angle > self.maximum_crossing_angle
        {
            return Err(PyValueError::new_err(
                "crossing angles must be ordered within [0, pi/2] radians",
            ));
        }
        if !self.probability.is_finite() || !(0.0..=1.0).contains(&self.probability) {
            return Err(PyValueError::new_err(
                "junction probability must be in [0, 1]",
            ));
        }
        if self.maximum_per_fiber_pair == 0 || self.candidate_capacity == 0 {
            return Err(PyValueError::new_err(
                "junction count and candidate capacity must be positive",
            ));
        }
        if !self.minimum_anchor_separation.is_finite() || self.minimum_anchor_separation < 0.0 {
            return Err(PyValueError::new_err(
                "minimum_anchor_separation must be nonnegative and finite",
            ));
        }
        Ok(JunctionCapturePolicy {
            name: self.name.clone(),
            law_name: self.law_name.clone(),
            parameters: JunctionParameterId(self.parameter_set),
            maximum_surface_gap: self.maximum_surface_gap,
            minimum_crossing_angle: self.minimum_crossing_angle,
            maximum_crossing_angle: self.maximum_crossing_angle,
            probability: self.probability,
            seed: self.seed,
            material_pairs: self
                .material_pairs
                .iter()
                .map(|(first, second)| JunctionMaterialPair::new(first, second))
                .collect(),
            maximum_per_fiber_pair: self.maximum_per_fiber_pair,
            minimum_anchor_separation: self.minimum_anchor_separation,
            candidate_capacity: self.candidate_capacity,
        })
    }
}
