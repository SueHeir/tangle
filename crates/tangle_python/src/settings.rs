use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use tangle_app::TangleStage;
use tangle_generate::{
    AcceptanceLimit, LimitEnforcement, RelaxationAcceptance, RelaxationTargets, SolveExhaustion,
    SolvePolicy,
};
use tangle_relax::{
    AdaptiveSegmentationConfig, CellListConfig, ContactAggregation, FiberMotion, RelaxationBackend,
    RelaxationConfig, RelaxationOverrides,
};

#[pyclass(name = "CellListSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyCellListSettings {
    #[pyo3(get, set)]
    pub cell_size_scale: f32,
}

impl Default for PyCellListSettings {
    fn default() -> Self {
        let config = CellListConfig::default();
        Self {
            cell_size_scale: config.cell_size_scale,
        }
    }
}

#[pymethods]
impl PyCellListSettings {
    #[new]
    #[pyo3(signature = (cell_size_scale=None))]
    fn new(cell_size_scale: Option<f32>) -> Self {
        let mut settings = Self::default();
        if let Some(value) = cell_size_scale {
            settings.cell_size_scale = value;
        }
        settings
    }

    fn __repr__(&self) -> String {
        format!("CellListSettings(cell_size_scale={})", self.cell_size_scale)
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        output.set_item("cell_size_scale", self.cell_size_scale)?;
        Ok(output.unbind())
    }
}

impl PyCellListSettings {
    fn to_rust(&self) -> PyResult<CellListConfig> {
        if !self.cell_size_scale.is_finite() || self.cell_size_scale < 1.0 {
            return Err(PyValueError::new_err(
                "cell_size_scale must be finite and at least 1",
            ));
        }
        Ok(CellListConfig {
            cell_size_scale: self.cell_size_scale,
            ..CellListConfig::default()
        })
    }
}

#[pyclass(name = "AdaptiveSegmentationSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyAdaptiveSegmentationSettings {
    #[pyo3(get, set)]
    pub contact_length_over_diameter: f32,
    #[pyo3(get, set)]
    pub minimum_length_over_diameter: f32,
    #[pyo3(get, set)]
    pub maximum_refinement_levels: u32,
    #[pyo3(get, set)]
    pub refinement_interval: usize,
    #[pyo3(get, set)]
    pub refinement_persistence: u32,
    #[pyo3(get, set)]
    pub coarsening_persistence: u32,
    #[pyo3(get, set)]
    pub coarsening_error_over_diameter: f32,
    #[pyo3(get, set)]
    pub coarsening_curvature_ratio: f32,
}

impl Default for PyAdaptiveSegmentationSettings {
    fn default() -> Self {
        let config = AdaptiveSegmentationConfig::default();
        Self {
            contact_length_over_diameter: config.contact_length_over_diameter,
            minimum_length_over_diameter: config.minimum_length_over_diameter,
            maximum_refinement_levels: config.maximum_refinement_levels,
            refinement_interval: config.refinement_interval,
            refinement_persistence: config.refinement_persistence,
            coarsening_persistence: config.coarsening_persistence,
            coarsening_error_over_diameter: config.coarsening_error_over_diameter,
            coarsening_curvature_ratio: config.coarsening_curvature_ratio,
        }
    }
}

#[pymethods]
impl PyAdaptiveSegmentationSettings {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[classmethod]
    fn profile(_class: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        let mut settings = Self::default();
        match name {
            "balanced" => {}
            "fast" => {
                settings.contact_length_over_diameter = 3.0;
                settings.minimum_length_over_diameter = 1.0;
                settings.maximum_refinement_levels = 6;
                settings.refinement_interval = 16;
                settings.refinement_persistence = 4;
                settings.coarsening_persistence = 16;
            }
            "strict" => {
                settings.contact_length_over_diameter = 1.0;
                settings.minimum_length_over_diameter = 0.25;
                settings.maximum_refinement_levels = 10;
                settings.refinement_interval = 4;
                settings.refinement_persistence = 2;
                settings.coarsening_persistence = 64;
                settings.coarsening_error_over_diameter = 0.05;
                settings.coarsening_curvature_ratio = 0.1;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown adaptive segmentation profile {other:?}; expected 'balanced', 'fast', or 'strict'"
                )))
            }
        }
        Ok(settings)
    }

    fn __repr__(&self) -> String {
        format!(
            concat!(
                "AdaptiveSegmentationSettings(contact_length_over_diameter={}, ",
                "minimum_length_over_diameter={}, maximum_refinement_levels={}, ",
                "refinement_interval={}, refinement_persistence={}, ",
                "coarsening_persistence={}, coarsening_error_over_diameter={}, ",
                "coarsening_curvature_ratio={})"
            ),
            self.contact_length_over_diameter,
            self.minimum_length_over_diameter,
            self.maximum_refinement_levels,
            self.refinement_interval,
            self.refinement_persistence,
            self.coarsening_persistence,
            self.coarsening_error_over_diameter,
            self.coarsening_curvature_ratio,
        )
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        output.set_item(
            "contact_length_over_diameter",
            self.contact_length_over_diameter,
        )?;
        output.set_item(
            "minimum_length_over_diameter",
            self.minimum_length_over_diameter,
        )?;
        output.set_item("maximum_refinement_levels", self.maximum_refinement_levels)?;
        output.set_item("refinement_interval", self.refinement_interval)?;
        output.set_item("refinement_persistence", self.refinement_persistence)?;
        output.set_item("coarsening_persistence", self.coarsening_persistence)?;
        output.set_item(
            "coarsening_error_over_diameter",
            self.coarsening_error_over_diameter,
        )?;
        output.set_item(
            "coarsening_curvature_ratio",
            self.coarsening_curvature_ratio,
        )?;
        Ok(output.unbind())
    }
}

impl PyAdaptiveSegmentationSettings {
    fn to_rust(&self) -> PyResult<AdaptiveSegmentationConfig> {
        if !self.contact_length_over_diameter.is_finite()
            || self.contact_length_over_diameter <= 0.0
        {
            return Err(PyValueError::new_err(
                "contact_length_over_diameter must be positive and finite",
            ));
        }
        if !self.minimum_length_over_diameter.is_finite()
            || self.minimum_length_over_diameter <= 0.0
            || self.minimum_length_over_diameter > self.contact_length_over_diameter
        {
            return Err(PyValueError::new_err(
                "minimum_length_over_diameter must be positive and no larger than contact_length_over_diameter",
            ));
        }
        if self.maximum_refinement_levels == 0 || self.maximum_refinement_levels >= 31 {
            return Err(PyValueError::new_err(
                "maximum_refinement_levels must be between 1 and 30",
            ));
        }
        if self.refinement_interval == 0 || self.refinement_persistence == 0 {
            return Err(PyValueError::new_err(
                "refinement_interval and refinement_persistence must be positive",
            ));
        }
        if !self.coarsening_error_over_diameter.is_finite()
            || self.coarsening_error_over_diameter < 0.0
            || !self.coarsening_curvature_ratio.is_finite()
            || self.coarsening_curvature_ratio < 0.0
        {
            return Err(PyValueError::new_err(
                "coarsening thresholds must be nonnegative and finite",
            ));
        }
        Ok(AdaptiveSegmentationConfig {
            contact_length_over_diameter: self.contact_length_over_diameter,
            minimum_length_over_diameter: self.minimum_length_over_diameter,
            maximum_refinement_levels: self.maximum_refinement_levels,
            refinement_interval: self.refinement_interval,
            refinement_persistence: self.refinement_persistence,
            coarsening_persistence: self.coarsening_persistence,
            coarsening_error_over_diameter: self.coarsening_error_over_diameter,
            coarsening_curvature_ratio: self.coarsening_curvature_ratio,
        })
    }
}

#[pyclass(name = "RelaxationSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyRelaxationSettings {
    #[pyo3(get, set)]
    pub backend: String,
    #[pyo3(get, set)]
    pub motion_model: String,
    #[pyo3(get, set)]
    pub pin_fiber_ends: bool,
    #[pyo3(get, set)]
    pub penetration_tolerance: f32,
    #[pyo3(get, set)]
    pub force_full_iterations: bool,
    #[pyo3(get, set)]
    pub correction_fraction: f32,
    #[pyo3(get, set)]
    pub contact_aggregation: String,
    #[pyo3(get, set)]
    pub stretch_stiffness: f32,
    #[pyo3(get, set)]
    pub bend_stiffness: f32,
    #[pyo3(get, set)]
    pub curvature_limit_stiffness: f32,
    #[pyo3(get, set)]
    pub curvature_limit_safety_margin: f32,
    #[pyo3(get, set)]
    pub curvature_ratio_tolerance: f32,
    #[pyo3(get, set)]
    pub constraint_iterations: usize,
    #[pyo3(get, set)]
    pub curvature_cleanup_sweeps: usize,
    #[pyo3(get, set)]
    pub max_step: f32,
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub iterations_per_batch: usize,
    #[pyo3(get, set)]
    pub debug_snapshot_interval: Option<usize>,
    #[pyo3(get, set)]
    pub save_assembled_reference: bool,
    cell_list: PyCellListSettings,
    adaptive_segmentation: Option<PyAdaptiveSegmentationSettings>,
}

impl Default for PyRelaxationSettings {
    fn default() -> Self {
        let config = RelaxationConfig::default();
        Self {
            backend: "wgpu".to_string(),
            motion_model: motion_model_name(config.motion_model).to_string(),
            pin_fiber_ends: config.pin_fiber_ends,
            penetration_tolerance: config.penetration_tolerance,
            force_full_iterations: config.force_full_iterations,
            correction_fraction: config.correction_fraction,
            contact_aggregation: aggregation_name(config.contact_aggregation).to_string(),
            stretch_stiffness: config.stretch_stiffness,
            bend_stiffness: config.bend_stiffness,
            curvature_limit_stiffness: config.curvature_limit_stiffness,
            curvature_limit_safety_margin: config.curvature_limit_safety_margin,
            curvature_ratio_tolerance: config.curvature_ratio_tolerance,
            constraint_iterations: config.constraint_iterations,
            curvature_cleanup_sweeps: config.curvature_cleanup_sweeps,
            max_step: config.max_step,
            max_iterations: config.max_iterations,
            iterations_per_batch: config.iterations_per_batch,
            debug_snapshot_interval: config.debug_snapshot_interval,
            save_assembled_reference: config.save_assembled_reference,
            cell_list: PyCellListSettings {
                cell_size_scale: config.cell_list.cell_size_scale,
            },
            adaptive_segmentation: config.adaptive_segmentation.map(|adaptive| {
                PyAdaptiveSegmentationSettings {
                    contact_length_over_diameter: adaptive.contact_length_over_diameter,
                    minimum_length_over_diameter: adaptive.minimum_length_over_diameter,
                    maximum_refinement_levels: adaptive.maximum_refinement_levels,
                    refinement_interval: adaptive.refinement_interval,
                    refinement_persistence: adaptive.refinement_persistence,
                    coarsening_persistence: adaptive.coarsening_persistence,
                    coarsening_error_over_diameter: adaptive.coarsening_error_over_diameter,
                    coarsening_curvature_ratio: adaptive.coarsening_curvature_ratio,
                }
            }),
        }
    }
}

#[pymethods]
impl PyRelaxationSettings {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[getter]
    fn cell_list(&self) -> PyCellListSettings {
        self.cell_list.clone()
    }

    #[setter]
    fn set_cell_list(&mut self, value: PyCellListSettings) {
        self.cell_list = value;
    }

    #[getter]
    fn cell_size_scale(&self) -> f32 {
        self.cell_list.cell_size_scale
    }

    #[setter]
    fn set_cell_size_scale(&mut self, value: f32) {
        self.cell_list.cell_size_scale = value;
    }

    #[getter]
    fn adaptive_segmentation(&self) -> Option<PyAdaptiveSegmentationSettings> {
        self.adaptive_segmentation.clone()
    }

    #[setter]
    fn set_adaptive_segmentation(&mut self, value: Option<PyAdaptiveSegmentationSettings>) {
        self.adaptive_segmentation = value;
    }

    fn enable_adaptive_segmentation(&mut self) {
        self.adaptive_segmentation = Some(PyAdaptiveSegmentationSettings::default());
    }

    fn disable_adaptive_segmentation(&mut self) {
        self.adaptive_segmentation = None;
    }

    fn __repr__(&self) -> String {
        format!(
            concat!(
                "RelaxationSettings(backend={:?}, motion_model={:?}, ",
                "penetration_tolerance={}, max_iterations={}, ",
                "iterations_per_batch={}, adaptive_segmentation={})"
            ),
            self.backend,
            self.motion_model,
            self.penetration_tolerance,
            self.max_iterations,
            self.iterations_per_batch,
            if self.adaptive_segmentation.is_some() {
                "enabled"
            } else {
                "disabled"
            }
        )
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        output.set_item("backend", &self.backend)?;
        output.set_item("motion_model", &self.motion_model)?;
        output.set_item("pin_fiber_ends", self.pin_fiber_ends)?;
        output.set_item("penetration_tolerance", self.penetration_tolerance)?;
        output.set_item("force_full_iterations", self.force_full_iterations)?;
        output.set_item("correction_fraction", self.correction_fraction)?;
        output.set_item("contact_aggregation", &self.contact_aggregation)?;
        output.set_item("stretch_stiffness", self.stretch_stiffness)?;
        output.set_item("bend_stiffness", self.bend_stiffness)?;
        output.set_item("curvature_limit_stiffness", self.curvature_limit_stiffness)?;
        output.set_item(
            "curvature_limit_safety_margin",
            self.curvature_limit_safety_margin,
        )?;
        output.set_item("curvature_ratio_tolerance", self.curvature_ratio_tolerance)?;
        output.set_item("constraint_iterations", self.constraint_iterations)?;
        output.set_item("curvature_cleanup_sweeps", self.curvature_cleanup_sweeps)?;
        output.set_item("max_step", self.max_step)?;
        output.set_item("max_iterations", self.max_iterations)?;
        output.set_item("iterations_per_batch", self.iterations_per_batch)?;
        output.set_item("cell_list", self.cell_list.to_dict(py)?)?;
        output.set_item(
            "adaptive_segmentation",
            self.adaptive_segmentation
                .as_ref()
                .map(|adaptive| adaptive.to_dict(py))
                .transpose()?,
        )?;
        output.set_item("debug_snapshot_interval", self.debug_snapshot_interval)?;
        output.set_item("save_assembled_reference", self.save_assembled_reference)?;
        Ok(output.unbind())
    }
}

impl PyRelaxationSettings {
    pub(crate) fn to_rust(&self) -> PyResult<RelaxationConfig> {
        let backend = match self.backend.as_str() {
            "wgpu" => RelaxationBackend::Wgpu,
            "cpu" => RelaxationBackend::Cpu,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported backend {other:?} in this Python build; expected 'wgpu' or 'cpu'"
                )))
            }
        };
        let motion_model = match self.motion_model.as_str() {
            "flexible" => FiberMotion::Flexible,
            "rigid_translation" => FiberMotion::RigidTranslation,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown motion_model {other:?}; expected 'flexible' or 'rigid_translation'"
                )))
            }
        };
        let contact_aggregation = match self.contact_aggregation.as_str() {
            "uniform_average" => ContactAggregation::UniformAverage,
            "penetration_weighted" => ContactAggregation::PenetrationWeighted,
            "deepest_only" => ContactAggregation::DeepestOnly,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown contact_aggregation {other:?}; expected 'uniform_average', 'penetration_weighted', or 'deepest_only'"
                )))
            }
        };
        if !self.penetration_tolerance.is_finite() || self.penetration_tolerance < 0.0 {
            return Err(PyValueError::new_err(
                "penetration_tolerance must be nonnegative and finite",
            ));
        }
        for (name, value) in [
            ("correction_fraction", self.correction_fraction),
            ("stretch_stiffness", self.stretch_stiffness),
            ("bend_stiffness", self.bend_stiffness),
            ("curvature_limit_stiffness", self.curvature_limit_stiffness),
        ] {
            if !value.is_finite()
                || !(0.0..=1.0).contains(&value)
                || value == 0.0 && name == "correction_fraction"
            {
                return Err(PyValueError::new_err(format!(
                    "{name} must be finite and within its [0, 1] range"
                )));
            }
        }
        if !self.curvature_limit_safety_margin.is_finite()
            || !(0.0..0.1).contains(&self.curvature_limit_safety_margin)
        {
            return Err(PyValueError::new_err(
                "curvature_limit_safety_margin must be in [0, 0.1)",
            ));
        }
        if !self.curvature_ratio_tolerance.is_finite() || self.curvature_ratio_tolerance < 0.0 {
            return Err(PyValueError::new_err(
                "curvature_ratio_tolerance must be nonnegative and finite",
            ));
        }
        if self.constraint_iterations == 0
            || self.curvature_cleanup_sweeps == 0
            || self.max_iterations == 0
            || self.iterations_per_batch == 0
        {
            return Err(PyValueError::new_err(
                "iteration and sweep counts must be positive",
            ));
        }
        if !self.max_step.is_finite() || self.max_step <= 0.0 {
            return Err(PyValueError::new_err(
                "max_step must be positive and finite",
            ));
        }
        if self.debug_snapshot_interval == Some(0) {
            return Err(PyValueError::new_err(
                "debug_snapshot_interval must be positive when supplied",
            ));
        }
        let adaptive_segmentation = self
            .adaptive_segmentation
            .as_ref()
            .map(PyAdaptiveSegmentationSettings::to_rust)
            .transpose()?;
        if adaptive_segmentation.is_some() && motion_model != FiberMotion::Flexible {
            return Err(PyValueError::new_err(
                "adaptive segmentation requires motion_model='flexible'",
            ));
        }
        Ok(RelaxationConfig {
            backend,
            motion_model,
            adaptive_segmentation,
            pin_fiber_ends: self.pin_fiber_ends,
            penetration_tolerance: self.penetration_tolerance,
            force_full_iterations: self.force_full_iterations,
            correction_fraction: self.correction_fraction,
            contact_aggregation,
            stretch_stiffness: self.stretch_stiffness,
            bend_stiffness: self.bend_stiffness,
            curvature_limit_stiffness: self.curvature_limit_stiffness,
            curvature_limit_safety_margin: self.curvature_limit_safety_margin,
            curvature_ratio_tolerance: self.curvature_ratio_tolerance,
            constraint_iterations: self.constraint_iterations,
            curvature_cleanup_sweeps: self.curvature_cleanup_sweeps,
            max_step: self.max_step,
            max_iterations: self.max_iterations,
            iterations_per_batch: self.iterations_per_batch,
            cell_list: self.cell_list.to_rust()?,
            debug_snapshot_interval: self.debug_snapshot_interval,
            save_assembled_reference: self.save_assembled_reference,
            next_stage: TangleStage::Done,
        })
    }
}

fn motion_model_name(value: FiberMotion) -> &'static str {
    match value {
        FiberMotion::Flexible => "flexible",
        FiberMotion::RigidTranslation => "rigid_translation",
    }
}

fn aggregation_name(value: ContactAggregation) -> &'static str {
    match value {
        ContactAggregation::UniformAverage => "uniform_average",
        ContactAggregation::PenetrationWeighted => "penetration_weighted",
        ContactAggregation::DeepestOnly => "deepest_only",
    }
}

#[pyclass(name = "RelaxationOverrides", module = "tangle._tangle")]
#[derive(Clone, Debug, Default)]
pub(crate) struct PyRelaxationOverrides {
    #[pyo3(get, set)]
    pub motion_model: Option<String>,
    #[pyo3(get, set)]
    pub correction_fraction: Option<f32>,
    #[pyo3(get, set)]
    pub contact_aggregation: Option<String>,
    #[pyo3(get, set)]
    pub stretch_stiffness: Option<f32>,
    #[pyo3(get, set)]
    pub bend_stiffness: Option<f32>,
    #[pyo3(get, set)]
    pub curvature_limit_stiffness: Option<f32>,
    #[pyo3(get, set)]
    pub constraint_iterations: Option<usize>,
    #[pyo3(get, set)]
    pub curvature_cleanup_sweeps: Option<usize>,
}

#[pymethods]
impl PyRelaxationOverrides {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RelaxationOverrides(motion_model={:?}, correction_fraction={:?}, bend_stiffness={:?})",
            self.motion_model, self.correction_fraction, self.bend_stiffness
        )
    }
}

impl PyRelaxationOverrides {
    pub(crate) fn to_rust(&self) -> PyResult<RelaxationOverrides> {
        let motion_model = self
            .motion_model
            .as_deref()
            .map(|value| match value {
                "flexible" => Ok(FiberMotion::Flexible),
                "rigid_translation" => Ok(FiberMotion::RigidTranslation),
                other => Err(PyValueError::new_err(format!(
                    "unknown override motion_model {other:?}"
                ))),
            })
            .transpose()?;
        let contact_aggregation = self
            .contact_aggregation
            .as_deref()
            .map(|value| match value {
                "uniform_average" => Ok(ContactAggregation::UniformAverage),
                "penetration_weighted" => Ok(ContactAggregation::PenetrationWeighted),
                "deepest_only" => Ok(ContactAggregation::DeepestOnly),
                other => Err(PyValueError::new_err(format!(
                    "unknown override contact_aggregation {other:?}"
                ))),
            })
            .transpose()?;
        for (name, value) in [
            ("correction_fraction", self.correction_fraction),
            ("stretch_stiffness", self.stretch_stiffness),
            ("bend_stiffness", self.bend_stiffness),
            ("curvature_limit_stiffness", self.curvature_limit_stiffness),
        ] {
            if value.is_some_and(|value| {
                !value.is_finite()
                    || !(0.0..=1.0).contains(&value)
                    || name == "correction_fraction" && value == 0.0
            }) {
                return Err(PyValueError::new_err(format!(
                    "{name} override must be within its [0, 1] range"
                )));
            }
        }
        if self.constraint_iterations == Some(0) || self.curvature_cleanup_sweeps == Some(0) {
            return Err(PyValueError::new_err(
                "override iteration and sweep counts must be positive",
            ));
        }
        Ok(RelaxationOverrides {
            motion_model,
            correction_fraction: self.correction_fraction,
            contact_aggregation,
            stretch_stiffness: self.stretch_stiffness,
            bend_stiffness: self.bend_stiffness,
            curvature_limit_stiffness: self.curvature_limit_stiffness,
            constraint_iterations: self.constraint_iterations,
            curvature_cleanup_sweeps: self.curvature_cleanup_sweeps,
        })
    }
}

#[pyclass(name = "SolvePolicy", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PySolvePolicy {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub solver_penetration: f32,
    #[pyo3(get, set)]
    pub solver_curvature_ratio: f32,
    #[pyo3(get, set)]
    pub acceptance_penetration: f32,
    #[pyo3(get, set)]
    pub penetration_enforcement: String,
    #[pyo3(get, set)]
    pub acceptance_curvature_ratio: f32,
    #[pyo3(get, set)]
    pub curvature_enforcement: String,
    #[pyo3(get, set)]
    pub maximum_iterations: usize,
    #[pyo3(get, set)]
    pub on_exhaustion: String,
}

#[pymethods]
impl PySolvePolicy {
    #[new]
    #[pyo3(signature = (
        name="relaxation".to_string(),
        *,
        solver_penetration=1.0e-4,
        solver_curvature_ratio=1.00001,
        acceptance_penetration=1.0e-4,
        penetration_enforcement="hard".to_string(),
        acceptance_curvature_ratio=1.00001,
        curvature_enforcement="hard".to_string(),
        maximum_iterations=2000,
        on_exhaustion="reject".to_string()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        solver_penetration: f32,
        solver_curvature_ratio: f32,
        acceptance_penetration: f32,
        penetration_enforcement: String,
        acceptance_curvature_ratio: f32,
        curvature_enforcement: String,
        maximum_iterations: usize,
        on_exhaustion: String,
    ) -> Self {
        Self {
            name,
            solver_penetration,
            solver_curvature_ratio,
            acceptance_penetration,
            penetration_enforcement,
            acceptance_curvature_ratio,
            curvature_enforcement,
            maximum_iterations,
            on_exhaustion,
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SolvePolicy(name={:?}, maximum_iterations={}, penetration={}/{}, curvature_ratio={}/{})",
            self.name,
            self.maximum_iterations,
            self.solver_penetration,
            self.acceptance_penetration,
            self.solver_curvature_ratio,
            self.acceptance_curvature_ratio,
        )
    }
}

impl PySolvePolicy {
    pub(crate) fn to_rust(&self) -> PyResult<SolvePolicy> {
        if self.name.trim().is_empty() {
            return Err(PyValueError::new_err("solve policy name must not be empty"));
        }
        if self.maximum_iterations == 0 {
            return Err(PyValueError::new_err(
                "solve policy maximum_iterations must be positive",
            ));
        }
        for (name, value) in [
            ("solver_penetration", self.solver_penetration),
            ("acceptance_penetration", self.acceptance_penetration),
            ("solver_curvature_ratio", self.solver_curvature_ratio),
            (
                "acceptance_curvature_ratio",
                self.acceptance_curvature_ratio,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be nonnegative and finite"
                )));
            }
        }
        let enforcement = |value: &str, label: &str| match value {
            "hard" => Ok(LimitEnforcement::Hard),
            "soft" => Ok(LimitEnforcement::Soft),
            other => Err(PyValueError::new_err(format!(
                "unknown {label} {other:?}; expected 'hard' or 'soft'"
            ))),
        };
        let on_exhaustion = match self.on_exhaustion.as_str() {
            "reject" => SolveExhaustion::Reject,
            "continue_if_hard_limits_satisfied" => SolveExhaustion::ContinueIfHardLimitsSatisfied,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown on_exhaustion {other:?}"
                )))
            }
        };
        Ok(SolvePolicy {
            name: self.name.clone(),
            solver_targets: RelaxationTargets {
                penetration: self.solver_penetration,
                curvature_ratio: self.solver_curvature_ratio,
            },
            acceptance: RelaxationAcceptance {
                penetration: AcceptanceLimit {
                    maximum: self.acceptance_penetration,
                    enforcement: enforcement(
                        &self.penetration_enforcement,
                        "penetration_enforcement",
                    )?,
                },
                curvature_ratio: AcceptanceLimit {
                    maximum: self.acceptance_curvature_ratio,
                    enforcement: enforcement(&self.curvature_enforcement, "curvature_enforcement")?,
                },
            },
            maximum_iterations: self.maximum_iterations,
            on_exhaustion,
        })
    }
}
