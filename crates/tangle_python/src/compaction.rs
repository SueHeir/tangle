use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use tangle_generate::{
    AdaptiveCompactionIncrement, CompactionConfig, CompactionGuards, CompactionPath,
    CompactionTarget,
};
use tangle_relax::{CompactionEnergyModel, CompactionKinematics};

use crate::common::{
    choice_name, parse_axis_mask, parse_choice, positive_finite, repr_fields, unit_vector, widen,
    with_kwargs,
};

const KINEMATICS: &[(&str, CompactionKinematics)] = &[
    (
        "rigid_fiber_centers",
        CompactionKinematics::RigidFiberCenters,
    ),
    ("moving_walls", CompactionKinematics::MovingWalls),
    ("affine_vertices", CompactionKinematics::AffineVertices),
];

// --- Targets -----------------------------------------------------------------

macro_rules! scalar_target {
    ($rust:ident, $python:literal, $doc:literal) => {
        scalar_target!($rust, $python, $doc, |_value: f64| PyResult::Ok(()));
    };
    ($rust:ident, $python:literal, $doc:literal, $check:expr) => {
        #[doc = $doc]
        #[pyclass(name = $python, module = "tangle._tangle", frozen)]
        #[derive(Clone, Debug)]
        pub(crate) struct $rust {
            #[pyo3(get)]
            value: f64,
        }

        #[pymethods]
        impl $rust {
            #[new]
            fn new(value: f64) -> PyResult<Self> {
                positive_finite(value, "value")?;
                ($check)(value)?;
                Ok(Self { value })
            }

            fn __repr__(&self) -> String {
                format!("{}({})", $python, self.value)
            }
        }
    };
}

scalar_target!(
    PyVolumeFractionTarget,
    "VolumeFractionTarget",
    "Stop when nominal fiber volume over cell volume reaches `value` (< 1).",
    |value: f64| {
        if value < 1.0 {
            PyResult::Ok(())
        } else {
            Err(PyValueError::new_err(
                "volume fraction target must be below 1",
            ))
        }
    }
);
scalar_target!(
    PyCellVolumeTarget,
    "CellVolumeTarget",
    "Stop when the cell volume reaches `value`."
);
scalar_target!(
    PyMeanPressureTarget,
    "MeanPressureTarget",
    "Stop when the mean of the three directional penalty pressures reaches `value`."
);
scalar_target!(
    PyPenaltyEnergyTarget,
    "PenaltyEnergyTarget",
    "Stop when total contact, stretch, and excess-bending penalty energy reaches `value`."
);

/// Stop when the three cell edge lengths reach `lengths`.
#[pyclass(name = "CellLengthsTarget", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyCellLengthsTarget {
    #[pyo3(get)]
    lengths: [f64; 3],
}

#[pymethods]
impl PyCellLengthsTarget {
    #[new]
    fn new(lengths: [f64; 3]) -> PyResult<Self> {
        for value in lengths {
            positive_finite(value, "lengths")?;
        }
        Ok(Self { lengths })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(slf.as_any(), "CellLengthsTarget", &["lengths"], false)
    }
}

/// Stop when each axis's penalty pressure reaches `pressures`; zero disables
/// an axis.
#[pyclass(name = "DirectionalPressureTarget", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyDirectionalPressureTarget {
    #[pyo3(get)]
    pressures: [f64; 3],
}

#[pymethods]
impl PyDirectionalPressureTarget {
    #[new]
    fn new(pressures: [f64; 3]) -> PyResult<Self> {
        validate_nonnegative_active(pressures, "pressures")?;
        Ok(Self { pressures })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "DirectionalPressureTarget",
            &["pressures"],
            false,
        )
    }
}

fn target_from_py(value: &Bound<'_, PyAny>) -> PyResult<CompactionTarget> {
    if let Ok(target) = value.extract::<PyRef<'_, PyVolumeFractionTarget>>() {
        if target.value >= 1.0 {
            return Err(PyValueError::new_err(
                "volume fraction target must be less than 1",
            ));
        }
        Ok(CompactionTarget::NominalVolumeFraction(target.value))
    } else if let Ok(target) = value.extract::<PyRef<'_, PyCellVolumeTarget>>() {
        Ok(CompactionTarget::CellVolume(target.value))
    } else if let Ok(target) = value.extract::<PyRef<'_, PyCellLengthsTarget>>() {
        Ok(CompactionTarget::CellLengths(target.lengths))
    } else if let Ok(target) = value.extract::<PyRef<'_, PyMeanPressureTarget>>() {
        Ok(CompactionTarget::MeanPressure(target.value as f32))
    } else if let Ok(target) = value.extract::<PyRef<'_, PyDirectionalPressureTarget>>() {
        Ok(CompactionTarget::DirectionalPressure(
            target.pressures.map(|value| value as f32),
        ))
    } else if let Ok(target) = value.extract::<PyRef<'_, PyPenaltyEnergyTarget>>() {
        Ok(CompactionTarget::PenaltyEnergy(target.value as f32))
    } else {
        Err(PyTypeError::new_err(
            "target must be a VolumeFractionTarget, CellVolumeTarget, CellLengthsTarget, MeanPressureTarget, DirectionalPressureTarget, or PenaltyEnergyTarget",
        ))
    }
}

fn target_to_py(py: Python<'_>, target: CompactionTarget) -> PyResult<Py<PyAny>> {
    Ok(match target {
        CompactionTarget::NominalVolumeFraction(value) => {
            Py::new(py, PyVolumeFractionTarget { value })?.into_any()
        }
        CompactionTarget::CellVolume(value) => {
            Py::new(py, PyCellVolumeTarget { value })?.into_any()
        }
        CompactionTarget::CellLengths(lengths) => {
            Py::new(py, PyCellLengthsTarget { lengths })?.into_any()
        }
        CompactionTarget::MeanPressure(value) => Py::new(
            py,
            PyMeanPressureTarget {
                value: widen(value),
            },
        )?
        .into_any(),
        CompactionTarget::DirectionalPressure(pressures) => Py::new(
            py,
            PyDirectionalPressureTarget {
                pressures: pressures.map(widen),
            },
        )?
        .into_any(),
        CompactionTarget::PenaltyEnergy(value) => Py::new(
            py,
            PyPenaltyEnergyTarget {
                value: widen(value),
            },
        )?
        .into_any(),
    })
}

// --- Paths -------------------------------------------------------------------

/// Split each strain increment among axes with fixed weights. Accepts axis
/// letters (`"z"`, `"xy"`) or three weights; `None` compresses along the
/// recipe's stack axis.
#[pyclass(name = "AxisWeightsPath", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyAxisWeightsPath {
    #[pyo3(get)]
    weights: Option<[f64; 3]>,
}

#[pymethods]
impl PyAxisWeightsPath {
    #[new]
    #[pyo3(signature = (weights=None))]
    fn new(weights: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let weights = weights
            .map(|value| -> PyResult<[f64; 3]> {
                if let Ok(weights) = value.extract::<[f64; 3]>() {
                    validate_nonnegative_active(weights, "weights")?;
                    Ok(weights)
                } else {
                    Ok(parse_axis_mask(value, "weights")?
                        .map(|active| if active { 1.0 } else { 0.0 }))
                }
            })
            .transpose()?;
        if let Some(weights) = weights {
            validate_nonnegative_active(weights, "weights")?;
        }
        Ok(Self { weights })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(slf.as_any(), "AxisWeightsPath", &["weights"], false)
    }
}

/// Adapt strain rates so the selected axes approach equal pressure.
#[pyclass(name = "EqualPressurePath", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyEqualPressurePath {
    #[pyo3(get)]
    axes: [bool; 3],
    #[pyo3(get)]
    pressure_floor: f64,
}

#[pymethods]
impl PyEqualPressurePath {
    #[new]
    #[pyo3(signature = (axes, *, pressure_floor=1.0e-12))]
    fn new(axes: &Bound<'_, PyAny>, pressure_floor: f64) -> PyResult<Self> {
        let axes = parse_axis_mask(axes, "axes")?;
        require_active_axes(axes)?;
        positive_finite(pressure_floor, "pressure_floor")?;
        Ok(Self {
            axes,
            pressure_floor,
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "EqualPressurePath",
            &["axes", "pressure_floor"],
            false,
        )
    }
}

/// Adapt strain rates toward a directional pressure ratio; zero disables an
/// axis.
#[pyclass(name = "StressRatioPath", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyStressRatioPath {
    #[pyo3(get)]
    ratio: [f64; 3],
    #[pyo3(get)]
    pressure_floor: f64,
}

#[pymethods]
impl PyStressRatioPath {
    #[new]
    #[pyo3(signature = (ratio, *, pressure_floor=1.0e-12))]
    fn new(ratio: [f64; 3], pressure_floor: f64) -> PyResult<Self> {
        validate_nonnegative_active(ratio, "ratio")?;
        positive_finite(pressure_floor, "pressure_floor")?;
        Ok(Self {
            ratio,
            pressure_floor,
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "StressRatioPath",
            &["ratio", "pressure_floor"],
            false,
        )
    }
}

/// Put each increment on the currently least expensive selected axis.
#[pyclass(name = "MinimumWorkPath", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyMinimumWorkPath {
    #[pyo3(get)]
    axes: [bool; 3],
}

#[pymethods]
impl PyMinimumWorkPath {
    #[new]
    fn new(axes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let axes = parse_axis_mask(axes, "axes")?;
        require_active_axes(axes)?;
        Ok(Self { axes })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(slf.as_any(), "MinimumWorkPath", &["axes"], false)
    }
}

#[derive(Clone, Debug)]
enum Path {
    AxisWeights(Option<[f64; 3]>),
    Rust(CompactionPath),
}

impl Path {
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(path) = value.extract::<PyRef<'_, PyAxisWeightsPath>>() {
            Ok(Self::AxisWeights(path.weights))
        } else if let Ok(path) = value.extract::<PyRef<'_, PyEqualPressurePath>>() {
            Ok(Self::Rust(CompactionPath::EqualPressure {
                active_axes: path.axes,
                pressure_floor: path.pressure_floor as f32,
            }))
        } else if let Ok(path) = value.extract::<PyRef<'_, PyStressRatioPath>>() {
            Ok(Self::Rust(CompactionPath::StressRatio {
                ratio: path.ratio.map(|value| value as f32),
                pressure_floor: path.pressure_floor as f32,
            }))
        } else if let Ok(path) = value.extract::<PyRef<'_, PyMinimumWorkPath>>() {
            Ok(Self::Rust(CompactionPath::MinimumIncrementalWork {
                active_axes: path.axes,
            }))
        } else {
            Err(PyTypeError::new_err(
                "path must be an AxisWeightsPath, EqualPressurePath, StressRatioPath, or MinimumWorkPath",
            ))
        }
    }

    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match *self {
            Self::AxisWeights(weights) => Py::new(py, PyAxisWeightsPath { weights })?.into_any(),
            Self::Rust(CompactionPath::AxisWeights(weights)) => Py::new(
                py,
                PyAxisWeightsPath {
                    weights: Some(weights.map(widen)),
                },
            )?
            .into_any(),
            Self::Rust(CompactionPath::EqualPressure {
                active_axes,
                pressure_floor,
            }) => Py::new(
                py,
                PyEqualPressurePath {
                    axes: active_axes,
                    pressure_floor: widen(pressure_floor),
                },
            )?
            .into_any(),
            Self::Rust(CompactionPath::StressRatio {
                ratio,
                pressure_floor,
            }) => Py::new(
                py,
                PyStressRatioPath {
                    ratio: ratio.map(widen),
                    pressure_floor: widen(pressure_floor),
                },
            )?
            .into_any(),
            Self::Rust(CompactionPath::MinimumIncrementalWork { active_axes }) => {
                Py::new(py, PyMinimumWorkPath { axes: active_axes })?.into_any()
            }
        })
    }

    fn to_rust(&self, stack_axis: usize) -> CompactionPath {
        match *self {
            Self::AxisWeights(weights) => CompactionPath::AxisWeights(
                weights
                    .unwrap_or_else(|| unit_vector(stack_axis))
                    .map(|value| value as f32),
            ),
            Self::Rust(path) => path,
        }
    }
}

// --- Compaction settings -----------------------------------------------------

/// Closed-loop compaction: shrink the cell toward `target` along `path`,
/// relaxing between increments.
#[pyclass(name = "CompactionSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyCompactionSettings {
    target: CompactionTarget,
    path: Path,
    kinematics: CompactionKinematics,
    #[pyo3(get, set)]
    pub cell_anchor: [f64; 3],
    #[pyo3(get, set)]
    pub balance_opposing_faces: bool,
    #[pyo3(get, set)]
    pub face_pressure_floor: f64,
    #[pyo3(get, set)]
    pub face_balance_strength: f64,
    #[pyo3(get, set)]
    pub initial_log_strain: f64,
    #[pyo3(get, set)]
    pub min_log_strain: f64,
    #[pyo3(get, set)]
    pub max_log_strain: f64,
    #[pyo3(get, set)]
    pub growth_factor: f64,
    #[pyo3(get, set)]
    pub shrink_factor: f64,
    #[pyo3(get, set)]
    pub relax_iterations: usize,
    #[pyo3(get, set)]
    pub max_shortening_over_min_diameter: f64,
    #[pyo3(get, set)]
    pub max_penetration: f64,
    #[pyo3(get, set)]
    pub max_curvature_ratio: f64,
    #[pyo3(get, set)]
    pub max_pressure: f64,
    #[pyo3(get, set)]
    pub max_penalty_energy: f64,
    #[pyo3(get, set)]
    pub max_steps: usize,
    #[pyo3(get, set)]
    pub max_relax_windows: usize,
    #[pyo3(get, set)]
    pub contact_energy_stiffness: f64,
    #[pyo3(get, set)]
    pub stretch_energy_stiffness: f64,
    #[pyo3(get, set)]
    pub bending_energy_stiffness: f64,
    #[pyo3(get, set)]
    pub target_tolerance: f64,
}

#[pymethods]
impl PyCompactionSettings {
    #[new]
    #[pyo3(signature = (target=None, **kwargs))]
    fn new(
        py: Python<'_>,
        target: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut settings = Self::default_for(CompactionTarget::NominalVolumeFraction(0.3));
        if let Some(target) = target {
            settings.target = target_from_py(target)?;
        }
        let settings = with_kwargs(py, settings, kwargs, "CompactionSettings")?;
        settings.validate()?;
        Ok(settings)
    }

    /// Shortcut for `CompactionSettings(VolumeFractionTarget(target), **changes)`.
    #[classmethod]
    #[pyo3(signature = (target, **changes))]
    fn volume_fraction(
        _class: &Bound<'_, PyType>,
        py: Python<'_>,
        target: f64,
        changes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        positive_finite(target, "target")?;
        if target >= 1.0 {
            return Err(PyValueError::new_err(
                "volume fraction target must be less than 1",
            ));
        }
        let settings = Self::default_for(CompactionTarget::NominalVolumeFraction(target));
        let settings = with_kwargs(py, settings, changes, "volume_fraction")?;
        settings.validate()?;
        Ok(settings)
    }

    #[getter]
    fn target(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        target_to_py(py, self.target)
    }

    #[setter]
    fn set_target(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.target = target_from_py(value)?;
        Ok(())
    }

    #[getter]
    fn path(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.path.to_py(py)
    }

    #[setter]
    fn set_path(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.path = Path::from_py(value)?;
        Ok(())
    }

    #[getter]
    fn kinematics(&self) -> &'static str {
        choice_name(self.kinematics, KINEMATICS)
    }

    #[setter]
    fn set_kinematics(&mut self, value: &str) -> PyResult<()> {
        self.kinematics = parse_choice(value, "kinematics", KINEMATICS)?;
        Ok(())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[pyo3(signature = (**changes))]
    fn replace(&self, py: Python<'_>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let settings = with_kwargs(py, self.clone(), changes, "replace")?;
        settings.validate()?;
        Ok(settings)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "CompactionSettings",
            &["target", "path", "kinematics"],
            false,
        )
    }
}

impl PyCompactionSettings {
    fn default_for(target: CompactionTarget) -> Self {
        let config = CompactionConfig::volume_fraction(0.3, [0.0, 0.0, 1.0]);
        Self {
            target,
            path: Path::AxisWeights(None),
            kinematics: config.kinematics,
            cell_anchor: config.cell_anchor.map(widen),
            balance_opposing_faces: config.balance_opposing_faces,
            face_pressure_floor: widen(config.face_pressure_floor),
            face_balance_strength: widen(config.face_balance_strength),
            initial_log_strain: widen(config.increment.initial_log_strain),
            min_log_strain: widen(config.increment.minimum_log_strain),
            max_log_strain: widen(config.increment.maximum_log_strain),
            growth_factor: widen(config.increment.growth_factor),
            shrink_factor: widen(config.increment.shrink_factor),
            relax_iterations: config.increment.relax_iterations,
            max_shortening_over_min_diameter: widen(
                config.increment.maximum_shortening_over_minimum_diameter,
            ),
            max_penetration: widen(config.guards.maximum_penetration),
            max_curvature_ratio: widen(config.guards.maximum_bend_ratio),
            max_pressure: widen(config.guards.maximum_pressure),
            max_penalty_energy: widen(config.guards.maximum_penalty_energy),
            max_steps: config.guards.maximum_steps,
            max_relax_windows: config.guards.maximum_relax_windows,
            contact_energy_stiffness: widen(config.energy_model.contact_stiffness),
            stretch_energy_stiffness: widen(config.energy_model.stretch_stiffness),
            bending_energy_stiffness: widen(config.energy_model.bending_stiffness),
            target_tolerance: widen(config.target_tolerance),
        }
    }

    /// Describes the target for `Recipe.operations()`.
    pub(crate) fn target_description(&self, py: Python<'_>) -> PyResult<String> {
        Ok(target_to_py(py, self.target)?.bind(py).repr()?.to_string())
    }

    pub(crate) fn to_rust(&self, stack_axis: usize) -> PyResult<CompactionConfig> {
        self.validate()?;
        Ok(CompactionConfig {
            target: self.target,
            path: self.path.to_rust(stack_axis),
            kinematics: self.kinematics,
            cell_anchor: self.cell_anchor.map(|value| value as f32),
            balance_opposing_faces: self.balance_opposing_faces,
            face_pressure_floor: self.face_pressure_floor as f32,
            face_balance_strength: self.face_balance_strength as f32,
            increment: AdaptiveCompactionIncrement {
                initial_log_strain: self.initial_log_strain as f32,
                minimum_log_strain: self.min_log_strain as f32,
                maximum_log_strain: self.max_log_strain as f32,
                growth_factor: self.growth_factor as f32,
                shrink_factor: self.shrink_factor as f32,
                relax_iterations: self.relax_iterations,
                maximum_shortening_over_minimum_diameter: self.max_shortening_over_min_diameter
                    as f32,
            },
            guards: CompactionGuards {
                maximum_penetration: self.max_penetration as f32,
                maximum_bend_ratio: self.max_curvature_ratio as f32,
                maximum_pressure: self.max_pressure as f32,
                maximum_penalty_energy: self.max_penalty_energy as f32,
                maximum_steps: self.max_steps,
                maximum_relax_windows: self.max_relax_windows,
            },
            energy_model: CompactionEnergyModel {
                contact_stiffness: self.contact_energy_stiffness as f32,
                stretch_stiffness: self.stretch_energy_stiffness as f32,
                bending_stiffness: self.bending_energy_stiffness as f32,
            },
            target_tolerance: self.target_tolerance as f32,
        })
    }

    fn validate(&self) -> PyResult<()> {
        if self
            .cell_anchor
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(PyValueError::new_err(
                "cell_anchor values must be in [0, 1]",
            ));
        }
        positive_finite(self.face_pressure_floor, "face_pressure_floor")?;
        if !self.face_balance_strength.is_finite()
            || !(0.0..=1.0).contains(&self.face_balance_strength)
        {
            return Err(PyValueError::new_err(
                "face_balance_strength must be in [0, 1]",
            ));
        }
        for (name, value) in [
            ("initial_log_strain", self.initial_log_strain),
            ("min_log_strain", self.min_log_strain),
            ("max_log_strain", self.max_log_strain),
            ("growth_factor", self.growth_factor),
            ("shrink_factor", self.shrink_factor),
            (
                "max_shortening_over_min_diameter",
                self.max_shortening_over_min_diameter,
            ),
            ("max_penetration", self.max_penetration),
            ("max_curvature_ratio", self.max_curvature_ratio),
            ("contact_energy_stiffness", self.contact_energy_stiffness),
            ("stretch_energy_stiffness", self.stretch_energy_stiffness),
            ("bending_energy_stiffness", self.bending_energy_stiffness),
            ("target_tolerance", self.target_tolerance),
        ] {
            positive_finite(value, name)?;
        }
        if self.growth_factor < 1.0 {
            return Err(PyValueError::new_err("growth_factor must be at least 1"));
        }
        if self.shrink_factor >= 1.0 {
            return Err(PyValueError::new_err("shrink_factor must be in (0, 1)"));
        }
        if self.max_curvature_ratio < 1.0 {
            return Err(PyValueError::new_err(
                "max_curvature_ratio must be at least 1",
            ));
        }
        if self.target_tolerance >= 1.0 {
            return Err(PyValueError::new_err("target_tolerance must be below 1"));
        }
        if self.min_log_strain > self.initial_log_strain
            || self.initial_log_strain > self.max_log_strain
        {
            return Err(PyValueError::new_err(
                "log strain increments must satisfy min <= initial <= max",
            ));
        }
        if self.relax_iterations == 0 || self.max_steps == 0 || self.max_relax_windows == 0 {
            return Err(PyValueError::new_err(
                "compaction iteration and step counts must be positive",
            ));
        }
        for (name, value) in [
            ("max_pressure", self.max_pressure),
            ("max_penalty_energy", self.max_penalty_energy),
        ] {
            if value.is_nan() || value <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be positive; infinity is allowed"
                )));
            }
        }
        Ok(())
    }
}

fn validate_nonnegative_active(values: [f64; 3], name: &str) -> PyResult<()> {
    if values.iter().any(|v| !v.is_finite() || *v < 0.0) || !values.iter().any(|v| *v > 0.0) {
        Err(PyValueError::new_err(format!(
            "{name} must be nonnegative with at least one positive axis"
        )))
    } else {
        Ok(())
    }
}

fn require_active_axes(axes: [bool; 3]) -> PyResult<()> {
    if axes.iter().any(|v| *v) {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "at least one compaction axis must be active",
        ))
    }
}
