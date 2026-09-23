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

use crate::common::{choice_name, parse_choice, repr_fields, widen, with_kwargs};

const BACKENDS: &[(&str, RelaxationBackend)] = &[
    ("wgpu", RelaxationBackend::Wgpu),
    ("cpu", RelaxationBackend::Cpu),
];
const MOTION_MODELS: &[(&str, FiberMotion)] = &[
    ("flexible", FiberMotion::Flexible),
    ("rigid_translation", FiberMotion::RigidTranslation),
];
const CONTACT_AGGREGATIONS: &[(&str, ContactAggregation)] = &[
    ("uniform_average", ContactAggregation::UniformAverage),
    (
        "penetration_weighted",
        ContactAggregation::PenetrationWeighted,
    ),
    ("deepest_only", ContactAggregation::DeepestOnly),
];
const BUDGET_EXHAUSTION: &[(&str, SolveExhaustion)] = &[
    ("fail", SolveExhaustion::Reject),
    (
        "continue_if_hard_ok",
        SolveExhaustion::ContinueIfHardLimitsSatisfied,
    ),
];

// --- Adaptive segmentation ---------------------------------------------------

#[pyclass(name = "AdaptiveSegmentationSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyAdaptiveSegmentationSettings {
    #[pyo3(get, set)]
    pub contact_length_over_diameter: f64,
    #[pyo3(get, set)]
    pub min_length_over_diameter: f64,
    #[pyo3(get, set)]
    pub max_refinement_levels: u32,
    #[pyo3(get, set)]
    pub refinement_interval: usize,
    #[pyo3(get, set)]
    pub refinement_persistence: u32,
    #[pyo3(get, set)]
    pub coarsening_persistence: u32,
    #[pyo3(get, set)]
    pub coarsening_error_over_diameter: f64,
    #[pyo3(get, set)]
    pub coarsening_curvature_ratio: f64,
}

impl Default for PyAdaptiveSegmentationSettings {
    fn default() -> Self {
        Self::from_rust(AdaptiveSegmentationConfig::default())
    }
}

#[pymethods]
impl PyAdaptiveSegmentationSettings {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let settings = with_kwargs(py, Self::default(), kwargs, "AdaptiveSegmentationSettings")?;
        settings.to_rust()?;
        Ok(settings)
    }

    /// Starts from a named profile, then applies any keyword changes.
    #[classmethod]
    #[pyo3(signature = (name, **changes))]
    fn profile(
        _class: &Bound<'_, PyType>,
        py: Python<'_>,
        name: &str,
        changes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        #[derive(Clone, Copy)]
        enum Profile {
            Balanced,
            Fast,
            Strict,
        }
        let mut settings = Self::default();
        match parse_choice(
            name,
            "adaptive segmentation profile",
            &[
                ("balanced", Profile::Balanced),
                ("fast", Profile::Fast),
                ("strict", Profile::Strict),
            ],
        )? {
            Profile::Balanced => {}
            Profile::Fast => {
                settings.contact_length_over_diameter = 3.0;
                settings.min_length_over_diameter = 1.0;
                settings.max_refinement_levels = 6;
                settings.refinement_interval = 16;
                settings.refinement_persistence = 4;
                settings.coarsening_persistence = 16;
            }
            Profile::Strict => {
                settings.contact_length_over_diameter = 1.0;
                settings.min_length_over_diameter = 0.25;
                settings.max_refinement_levels = 10;
                settings.refinement_interval = 4;
                settings.refinement_persistence = 2;
                settings.coarsening_persistence = 64;
                settings.coarsening_error_over_diameter = 0.05;
                settings.coarsening_curvature_ratio = 0.1;
            }
        }
        let settings = with_kwargs(py, settings, changes, "profile")?;
        settings.to_rust()?;
        Ok(settings)
    }

    fn __repr__(&self) -> String {
        format!(
            concat!(
                "AdaptiveSegmentationSettings(contact_length_over_diameter={}, ",
                "min_length_over_diameter={}, max_refinement_levels={}, ",
                "refinement_interval={}, refinement_persistence={}, ",
                "coarsening_persistence={}, coarsening_error_over_diameter={}, ",
                "coarsening_curvature_ratio={})"
            ),
            self.contact_length_over_diameter,
            self.min_length_over_diameter,
            self.max_refinement_levels,
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

    #[pyo3(signature = (**changes))]
    fn replace(&self, py: Python<'_>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let settings = with_kwargs(py, self.clone(), changes, "replace")?;
        settings.to_rust()?;
        Ok(settings)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        output.set_item(
            "contact_length_over_diameter",
            self.contact_length_over_diameter,
        )?;
        output.set_item("min_length_over_diameter", self.min_length_over_diameter)?;
        output.set_item("max_refinement_levels", self.max_refinement_levels)?;
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
    fn from_rust(config: AdaptiveSegmentationConfig) -> Self {
        Self {
            contact_length_over_diameter: widen(config.contact_length_over_diameter),
            min_length_over_diameter: widen(config.minimum_length_over_diameter),
            max_refinement_levels: config.maximum_refinement_levels,
            refinement_interval: config.refinement_interval,
            refinement_persistence: config.refinement_persistence,
            coarsening_persistence: config.coarsening_persistence,
            coarsening_error_over_diameter: widen(config.coarsening_error_over_diameter),
            coarsening_curvature_ratio: widen(config.coarsening_curvature_ratio),
        }
    }

    fn to_rust(&self) -> PyResult<AdaptiveSegmentationConfig> {
        if !self.contact_length_over_diameter.is_finite()
            || self.contact_length_over_diameter <= 0.0
        {
            return Err(PyValueError::new_err(
                "contact_length_over_diameter must be positive and finite",
            ));
        }
        if !self.min_length_over_diameter.is_finite()
            || self.min_length_over_diameter <= 0.0
            || self.min_length_over_diameter > self.contact_length_over_diameter
        {
            return Err(PyValueError::new_err(
                "min_length_over_diameter must be positive and no larger than contact_length_over_diameter",
            ));
        }
        if self.max_refinement_levels == 0 || self.max_refinement_levels >= 31 {
            return Err(PyValueError::new_err(
                "max_refinement_levels must be between 1 and 30",
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
            contact_length_over_diameter: self.contact_length_over_diameter as f32,
            minimum_length_over_diameter: self.min_length_over_diameter as f32,
            maximum_refinement_levels: self.max_refinement_levels,
            refinement_interval: self.refinement_interval,
            refinement_persistence: self.refinement_persistence,
            coarsening_persistence: self.coarsening_persistence,
            coarsening_error_over_diameter: self.coarsening_error_over_diameter as f32,
            coarsening_curvature_ratio: self.coarsening_curvature_ratio as f32,
        })
    }
}

// --- Relaxation settings -----------------------------------------------------

/// Run-wide solver settings, passed to `Recipe.run()`.
#[pyclass(name = "RelaxationSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyRelaxationSettings {
    backend: RelaxationBackend,
    motion_model: FiberMotion,
    #[pyo3(get, set)]
    pub pin_fiber_ends: bool,
    #[pyo3(get, set)]
    pub penetration_tolerance: f64,
    #[pyo3(get, set)]
    pub force_full_iterations: bool,
    #[pyo3(get, set)]
    pub correction_fraction: f64,
    contact_aggregation: ContactAggregation,
    #[pyo3(get, set)]
    pub stretch_stiffness: f64,
    #[pyo3(get, set)]
    pub bend_stiffness: f64,
    #[pyo3(get, set)]
    pub curvature_limit_stiffness: f64,
    #[pyo3(get, set)]
    pub curvature_limit_safety_margin: f64,
    #[pyo3(get, set)]
    pub curvature_ratio_tolerance: f64,
    #[pyo3(get, set)]
    pub constraint_iterations: usize,
    #[pyo3(get, set)]
    pub curvature_cleanup_sweeps: usize,
    #[pyo3(get, set)]
    pub max_step: f64,
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub iterations_per_batch: usize,
    #[pyo3(get, set)]
    pub debug_snapshot_interval: Option<usize>,
    #[pyo3(get, set)]
    pub save_assembled_reference: bool,
    #[pyo3(get, set)]
    pub cell_size_scale: f64,
    #[pyo3(get, set)]
    pub adaptive_segmentation: Option<PyAdaptiveSegmentationSettings>,
}

impl Default for PyRelaxationSettings {
    fn default() -> Self {
        let config = RelaxationConfig::default();
        Self {
            backend: RelaxationBackend::Wgpu,
            motion_model: config.motion_model,
            pin_fiber_ends: config.pin_fiber_ends,
            penetration_tolerance: widen(config.penetration_tolerance),
            force_full_iterations: config.force_full_iterations,
            correction_fraction: widen(config.correction_fraction),
            contact_aggregation: config.contact_aggregation,
            stretch_stiffness: widen(config.stretch_stiffness),
            bend_stiffness: widen(config.bend_stiffness),
            curvature_limit_stiffness: widen(config.curvature_limit_stiffness),
            curvature_limit_safety_margin: widen(config.curvature_limit_safety_margin),
            curvature_ratio_tolerance: widen(config.curvature_ratio_tolerance),
            constraint_iterations: config.constraint_iterations,
            curvature_cleanup_sweeps: config.curvature_cleanup_sweeps,
            max_step: widen(config.max_step),
            max_iterations: config.max_iterations,
            iterations_per_batch: config.iterations_per_batch,
            debug_snapshot_interval: config.debug_snapshot_interval,
            save_assembled_reference: config.save_assembled_reference,
            cell_size_scale: widen(config.cell_list.cell_size_scale),
            adaptive_segmentation: config
                .adaptive_segmentation
                .map(PyAdaptiveSegmentationSettings::from_rust),
        }
    }
}

#[pymethods]
impl PyRelaxationSettings {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let settings = with_kwargs(py, Self::default(), kwargs, "RelaxationSettings")?;
        settings.to_rust()?;
        Ok(settings)
    }

    #[getter]
    fn backend(&self) -> &'static str {
        choice_name(self.backend, BACKENDS)
    }

    #[setter]
    fn set_backend(&mut self, value: &str) -> PyResult<()> {
        self.backend = parse_choice(value, "backend", BACKENDS)?;
        Ok(())
    }

    #[getter]
    fn motion_model(&self) -> &'static str {
        choice_name(self.motion_model, MOTION_MODELS)
    }

    #[setter]
    fn set_motion_model(&mut self, value: &str) -> PyResult<()> {
        self.motion_model = parse_choice(value, "motion_model", MOTION_MODELS)?;
        Ok(())
    }

    #[getter]
    fn contact_aggregation(&self) -> &'static str {
        choice_name(self.contact_aggregation, CONTACT_AGGREGATIONS)
    }

    #[setter]
    fn set_contact_aggregation(&mut self, value: &str) -> PyResult<()> {
        self.contact_aggregation =
            parse_choice(value, "contact_aggregation", CONTACT_AGGREGATIONS)?;
        Ok(())
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
            self.backend(),
            self.motion_model(),
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

    #[pyo3(signature = (**changes))]
    fn replace(&self, py: Python<'_>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let settings = with_kwargs(py, self.clone(), changes, "replace")?;
        settings.to_rust()?;
        Ok(settings)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        output.set_item("backend", self.backend())?;
        output.set_item("motion_model", self.motion_model())?;
        output.set_item("pin_fiber_ends", self.pin_fiber_ends)?;
        output.set_item("penetration_tolerance", self.penetration_tolerance)?;
        output.set_item("force_full_iterations", self.force_full_iterations)?;
        output.set_item("correction_fraction", self.correction_fraction)?;
        output.set_item("contact_aggregation", self.contact_aggregation())?;
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
        output.set_item("cell_size_scale", self.cell_size_scale)?;
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
        if !self.cell_size_scale.is_finite() || self.cell_size_scale < 1.0 {
            return Err(PyValueError::new_err(
                "cell_size_scale must be finite and at least 1",
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
        if adaptive_segmentation.is_some() && self.motion_model != FiberMotion::Flexible {
            return Err(PyValueError::new_err(
                "adaptive segmentation requires motion_model='flexible'",
            ));
        }
        Ok(RelaxationConfig {
            backend: self.backend,
            motion_model: self.motion_model,
            adaptive_segmentation,
            pin_fiber_ends: self.pin_fiber_ends,
            penetration_tolerance: self.penetration_tolerance as f32,
            force_full_iterations: self.force_full_iterations,
            correction_fraction: self.correction_fraction as f32,
            contact_aggregation: self.contact_aggregation,
            stretch_stiffness: self.stretch_stiffness as f32,
            bend_stiffness: self.bend_stiffness as f32,
            curvature_limit_stiffness: self.curvature_limit_stiffness as f32,
            curvature_limit_safety_margin: self.curvature_limit_safety_margin as f32,
            curvature_ratio_tolerance: self.curvature_ratio_tolerance as f32,
            constraint_iterations: self.constraint_iterations,
            curvature_cleanup_sweeps: self.curvature_cleanup_sweeps,
            max_step: self.max_step as f32,
            max_iterations: self.max_iterations,
            iterations_per_batch: self.iterations_per_batch,
            cell_list: CellListConfig {
                cell_size_scale: self.cell_size_scale as f32,
            },
            debug_snapshot_interval: self.debug_snapshot_interval,
            save_assembled_reference: self.save_assembled_reference,
            next_stage: TangleStage::Done,
        })
    }
}

// --- Relaxation overrides ----------------------------------------------------

/// Temporary per-step changes to `RelaxationSettings`; `None` keeps the
/// run-wide value.
#[pyclass(name = "RelaxationOverrides", module = "tangle._tangle")]
#[derive(Clone, Debug, Default)]
pub(crate) struct PyRelaxationOverrides {
    motion_model: Option<FiberMotion>,
    #[pyo3(get, set)]
    pub correction_fraction: Option<f64>,
    contact_aggregation: Option<ContactAggregation>,
    #[pyo3(get, set)]
    pub stretch_stiffness: Option<f64>,
    #[pyo3(get, set)]
    pub bend_stiffness: Option<f64>,
    #[pyo3(get, set)]
    pub curvature_limit_stiffness: Option<f64>,
    #[pyo3(get, set)]
    pub constraint_iterations: Option<usize>,
    #[pyo3(get, set)]
    pub curvature_cleanup_sweeps: Option<usize>,
}

#[pymethods]
impl PyRelaxationOverrides {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let overrides = with_kwargs(py, Self::default(), kwargs, "RelaxationOverrides")?;
        overrides.to_rust()?;
        Ok(overrides)
    }

    /// Named cleanup presets used by the felt examples, with optional changes.
    ///
    /// - `contact_first`: loose bending and stretch so contacts resolve first.
    /// - `curvature_cleanup`: small contact corrections, strong curvature limit.
    /// - `contact_cleanup`: strong contact corrections, moderate curvature limit.
    #[classmethod]
    #[pyo3(signature = (name, **changes))]
    fn preset(
        _class: &Bound<'_, PyType>,
        py: Python<'_>,
        name: &str,
        changes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        #[derive(Clone, Copy)]
        enum Preset {
            ContactFirst,
            CurvatureCleanup,
            ContactCleanup,
        }
        let preset = parse_choice(
            name,
            "overrides preset",
            &[
                ("contact_first", Preset::ContactFirst),
                ("curvature_cleanup", Preset::CurvatureCleanup),
                ("contact_cleanup", Preset::ContactCleanup),
            ],
        )?;
        let (correction, stretch, curvature_limit, iterations, sweeps) = match preset {
            Preset::ContactFirst => (0.8, 0.05, 0.15, 1, 1),
            Preset::CurvatureCleanup => (0.05, 0.35, 1.0, 8, 16),
            Preset::ContactCleanup => (0.8, 0.35, 0.75, 8, 16),
        };
        let overrides = Self {
            motion_model: Some(FiberMotion::Flexible),
            correction_fraction: Some(correction),
            contact_aggregation: Some(ContactAggregation::DeepestOnly),
            stretch_stiffness: Some(stretch),
            bend_stiffness: Some(0.0),
            curvature_limit_stiffness: Some(curvature_limit),
            constraint_iterations: Some(iterations),
            curvature_cleanup_sweeps: Some(sweeps),
        };
        let overrides = with_kwargs(py, overrides, changes, "preset")?;
        overrides.to_rust()?;
        Ok(overrides)
    }

    #[getter]
    fn motion_model(&self) -> Option<&'static str> {
        self.motion_model
            .map(|value| choice_name(value, MOTION_MODELS))
    }

    #[setter]
    fn set_motion_model(&mut self, value: Option<&str>) -> PyResult<()> {
        self.motion_model = value
            .map(|value| parse_choice(value, "motion_model", MOTION_MODELS))
            .transpose()?;
        Ok(())
    }

    #[getter]
    fn contact_aggregation(&self) -> Option<&'static str> {
        self.contact_aggregation
            .map(|value| choice_name(value, CONTACT_AGGREGATIONS))
    }

    #[setter]
    fn set_contact_aggregation(&mut self, value: Option<&str>) -> PyResult<()> {
        self.contact_aggregation = value
            .map(|value| parse_choice(value, "contact_aggregation", CONTACT_AGGREGATIONS))
            .transpose()?;
        Ok(())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[pyo3(signature = (**changes))]
    fn replace(&self, py: Python<'_>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let overrides = with_kwargs(py, self.clone(), changes, "replace")?;
        overrides.to_rust()?;
        Ok(overrides)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "RelaxationOverrides",
            &[
                "motion_model",
                "correction_fraction",
                "contact_aggregation",
                "stretch_stiffness",
                "bend_stiffness",
                "curvature_limit_stiffness",
                "constraint_iterations",
                "curvature_cleanup_sweeps",
            ],
            true,
        )
    }
}

impl PyRelaxationOverrides {
    pub(crate) fn to_rust(&self) -> PyResult<RelaxationOverrides> {
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
            motion_model: self.motion_model,
            correction_fraction: self.correction_fraction.map(|value| value as f32),
            contact_aggregation: self.contact_aggregation,
            stretch_stiffness: self.stretch_stiffness.map(|value| value as f32),
            bend_stiffness: self.bend_stiffness.map(|value| value as f32),
            curvature_limit_stiffness: self.curvature_limit_stiffness.map(|value| value as f32),
            constraint_iterations: self.constraint_iterations,
            curvature_cleanup_sweeps: self.curvature_cleanup_sweeps,
        })
    }
}

// --- Solve policy ------------------------------------------------------------

/// A named relaxation step: the solver aims for the `target_*` values, and
/// the step passes when the result is within the `max_*` values.
///
/// `max_penetration` and `max_curvature_ratio` default to their targets.
/// A `hard_*` limit must hold for the step to pass; a soft one only warns.
#[pyclass(name = "SolvePolicy", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PySolvePolicy {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub target_penetration: f64,
    #[pyo3(get, set)]
    pub target_curvature_ratio: f64,
    max_penetration: Option<f64>,
    max_curvature_ratio: Option<f64>,
    #[pyo3(get, set)]
    pub hard_penetration: bool,
    #[pyo3(get, set)]
    pub hard_curvature: bool,
    #[pyo3(get, set)]
    pub max_iterations: usize,
    on_budget_exhausted: SolveExhaustion,
}

#[pymethods]
impl PySolvePolicy {
    #[new]
    #[pyo3(signature = (name="relaxation".to_string(), **kwargs))]
    fn new(py: Python<'_>, name: String, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let policy = Self {
            name,
            target_penetration: 1.0e-4,
            target_curvature_ratio: 1.00001,
            max_penetration: None,
            max_curvature_ratio: None,
            hard_penetration: true,
            hard_curvature: true,
            max_iterations: 2000,
            on_budget_exhausted: SolveExhaustion::Reject,
        };
        let policy = with_kwargs(py, policy, kwargs, "SolvePolicy")?;
        policy.to_rust()?;
        Ok(policy)
    }

    #[getter]
    fn max_penetration(&self) -> f64 {
        self.max_penetration.unwrap_or(self.target_penetration)
    }

    #[setter]
    fn set_max_penetration(&mut self, value: Option<f64>) {
        self.max_penetration = value;
    }

    #[getter]
    fn max_curvature_ratio(&self) -> f64 {
        self.max_curvature_ratio
            .unwrap_or(self.target_curvature_ratio)
    }

    #[setter]
    fn set_max_curvature_ratio(&mut self, value: Option<f64>) {
        self.max_curvature_ratio = value;
    }

    #[getter]
    fn on_budget_exhausted(&self) -> &'static str {
        choice_name(self.on_budget_exhausted, BUDGET_EXHAUSTION)
    }

    #[setter]
    fn set_on_budget_exhausted(&mut self, value: &str) -> PyResult<()> {
        self.on_budget_exhausted = parse_choice(value, "on_budget_exhausted", BUDGET_EXHAUSTION)?;
        Ok(())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[pyo3(signature = (**changes))]
    fn replace(&self, py: Python<'_>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let policy = with_kwargs(py, self.clone(), changes, "replace")?;
        policy.to_rust()?;
        Ok(policy)
    }

    fn __repr__(&self) -> String {
        format!(
            concat!(
                "SolvePolicy({:?}, target_penetration={}, max_penetration={}, ",
                "target_curvature_ratio={}, max_curvature_ratio={}, max_iterations={})"
            ),
            self.name,
            self.target_penetration,
            self.max_penetration(),
            self.target_curvature_ratio,
            self.max_curvature_ratio(),
            self.max_iterations,
        )
    }
}

impl PySolvePolicy {
    pub(crate) fn to_rust(&self) -> PyResult<SolvePolicy> {
        if self.name.trim().is_empty() {
            return Err(PyValueError::new_err("solve policy name must not be empty"));
        }
        if self.max_iterations == 0 {
            return Err(PyValueError::new_err(
                "solve policy max_iterations must be positive",
            ));
        }
        for (name, value) in [
            ("target_penetration", self.target_penetration),
            ("max_penetration", self.max_penetration()),
            ("target_curvature_ratio", self.target_curvature_ratio),
            ("max_curvature_ratio", self.max_curvature_ratio()),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be nonnegative and finite"
                )));
            }
        }
        let enforcement = |hard: bool| {
            if hard {
                LimitEnforcement::Hard
            } else {
                LimitEnforcement::Soft
            }
        };
        Ok(SolvePolicy {
            name: self.name.clone(),
            solver_targets: RelaxationTargets {
                penetration: self.target_penetration as f32,
                curvature_ratio: self.target_curvature_ratio as f32,
            },
            acceptance: RelaxationAcceptance {
                penetration: AcceptanceLimit {
                    maximum: self.max_penetration() as f32,
                    enforcement: enforcement(self.hard_penetration),
                },
                curvature_ratio: AcceptanceLimit {
                    maximum: self.max_curvature_ratio() as f32,
                    enforcement: enforcement(self.hard_curvature),
                },
            },
            maximum_iterations: self.max_iterations,
            on_exhaustion: self.on_budget_exhausted,
        })
    }
}
