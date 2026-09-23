use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tangle_checkpoint::CheckpointConfig;

/// Periodic restart checkpoint and optional resume settings.
#[pyclass(name = "CheckpointSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyCheckpointSettings {
    #[pyo3(get, set)]
    pub case_id: String,
    #[pyo3(get, set)]
    pub path: PathBuf,
    #[pyo3(get, set)]
    pub interval_iterations: usize,
    #[pyo3(get, set)]
    pub resume: bool,
    #[pyo3(get, set)]
    pub resume_path: Option<PathBuf>,
    #[pyo3(get, set)]
    pub resume_case_id: Option<String>,
    #[pyo3(get, set)]
    pub fresh_formation_on_resume: bool,
}

#[pymethods]
impl PyCheckpointSettings {
    #[new]
    #[pyo3(signature = (case_id, path, *, interval_iterations=500, resume=false))]
    fn new(case_id: String, path: PathBuf, interval_iterations: usize, resume: bool) -> Self {
        Self {
            case_id,
            path,
            interval_iterations,
            resume,
            resume_path: None,
            resume_case_id: None,
            fresh_formation_on_resume: false,
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckpointSettings(case_id={:?}, path={:?}, interval_iterations={}, resume={})",
            self.case_id, self.path, self.interval_iterations, self.resume
        )
    }
}

impl PyCheckpointSettings {
    pub(crate) fn to_rust(&self) -> PyResult<CheckpointConfig> {
        if self.case_id.trim().is_empty() {
            return Err(PyValueError::new_err(
                "checkpoint case_id must not be empty",
            ));
        }
        if self.interval_iterations == 0 {
            return Err(PyValueError::new_err(
                "checkpoint interval_iterations must be positive",
            ));
        }
        if self
            .resume_case_id
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(PyValueError::new_err(
                "resume_case_id must not be empty when supplied",
            ));
        }
        Ok(CheckpointConfig {
            case_id: self.case_id.clone(),
            path: self.path.clone(),
            interval_iterations: self.interval_iterations,
            resume: self.resume,
            resume_path: self.resume_path.clone(),
            resume_case_id: self.resume_case_id.clone(),
            fresh_formation_on_resume: self.fresh_formation_on_resume,
        })
    }
}
