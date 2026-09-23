use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tangle_generate::{
    AdaptiveCompactionIncrement, CompactionConfig, CompactionGuards, CompactionPath,
    CompactionTarget,
};
use tangle_relax::{CompactionEnergyModel, CompactionKinematics};

/// Fully editable closed-loop compaction configuration.
#[pyclass(name = "CompactionSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyCompactionSettings {
    #[pyo3(get, set)]
    pub target_type: String,
    #[pyo3(get, set)]
    pub target_value: f64,
    #[pyo3(get, set)]
    pub target_values: [f64; 3],
    #[pyo3(get, set)]
    pub path: String,
    #[pyo3(get, set)]
    pub axis_weights: [f32; 3],
    #[pyo3(get, set)]
    pub active_axes: [bool; 3],
    #[pyo3(get, set)]
    pub stress_ratio: [f32; 3],
    #[pyo3(get, set)]
    pub pressure_floor: f32,
    #[pyo3(get, set)]
    pub kinematics: String,
    #[pyo3(get, set)]
    pub cell_anchor: [f32; 3],
    #[pyo3(get, set)]
    pub balance_opposing_faces: bool,
    #[pyo3(get, set)]
    pub face_pressure_floor: f32,
    #[pyo3(get, set)]
    pub face_balance_strength: f32,
    #[pyo3(get, set)]
    pub initial_log_strain: f32,
    #[pyo3(get, set)]
    pub minimum_log_strain: f32,
    #[pyo3(get, set)]
    pub maximum_log_strain: f32,
    #[pyo3(get, set)]
    pub growth_factor: f32,
    #[pyo3(get, set)]
    pub shrink_factor: f32,
    #[pyo3(get, set)]
    pub relax_iterations: usize,
    #[pyo3(get, set)]
    pub maximum_shortening_over_minimum_diameter: f32,
    #[pyo3(get, set)]
    pub maximum_penetration: f32,
    #[pyo3(get, set)]
    pub maximum_bend_ratio: f32,
    #[pyo3(get, set)]
    pub maximum_pressure: f32,
    #[pyo3(get, set)]
    pub maximum_penalty_energy: f32,
    #[pyo3(get, set)]
    pub maximum_steps: usize,
    #[pyo3(get, set)]
    pub maximum_relax_windows: usize,
    #[pyo3(get, set)]
    pub contact_energy_stiffness: f32,
    #[pyo3(get, set)]
    pub stretch_energy_stiffness: f32,
    #[pyo3(get, set)]
    pub bending_energy_stiffness: f32,
    #[pyo3(get, set)]
    pub target_tolerance: f32,
}

#[pymethods]
impl PyCompactionSettings {
    #[new]
    #[pyo3(signature = (target_volume_fraction=0.3, axis_weights=[0.0, 0.0, 1.0]))]
    fn new(target_volume_fraction: f64, axis_weights: [f32; 3]) -> Self {
        Self::from_rust(CompactionConfig::volume_fraction(
            target_volume_fraction,
            axis_weights,
        ))
    }

    #[classmethod]
    #[pyo3(signature = (target, *, axis_weights=[0.0, 0.0, 1.0]))]
    fn volume_fraction(
        _class: &Bound<'_, pyo3::types::PyType>,
        target: f64,
        axis_weights: [f32; 3],
    ) -> Self {
        Self::from_rust(CompactionConfig::volume_fraction(target, axis_weights))
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompactionSettings(target_type={:?}, target_value={}, path={:?}, kinematics={:?})",
            self.target_type, self.target_value, self.path, self.kinematics
        )
    }
}

impl PyCompactionSettings {
    fn from_rust(config: CompactionConfig) -> Self {
        let (target_type, target_value, target_values) = match config.target {
            CompactionTarget::NominalVolumeFraction(value) => ("volume_fraction", value, [0.0; 3]),
            CompactionTarget::CellVolume(value) => ("cell_volume", value, [0.0; 3]),
            CompactionTarget::CellLengths(values) => ("cell_lengths", 0.0, values),
            CompactionTarget::MeanPressure(value) => ("mean_pressure", value as f64, [0.0; 3]),
            CompactionTarget::DirectionalPressure(values) => {
                ("directional_pressure", 0.0, values.map(f64::from))
            }
            CompactionTarget::PenaltyEnergy(value) => ("penalty_energy", value as f64, [0.0; 3]),
        };
        let (path, axis_weights, active_axes, stress_ratio, pressure_floor) = match config.path {
            CompactionPath::AxisWeights(values) => {
                ("axis_weights", values, [false; 3], [0.0; 3], 1.0e-12)
            }
            CompactionPath::EqualPressure {
                active_axes,
                pressure_floor,
            } => (
                "equal_pressure",
                [0.0; 3],
                active_axes,
                [0.0; 3],
                pressure_floor,
            ),
            CompactionPath::StressRatio {
                ratio,
                pressure_floor,
            } => ("stress_ratio", [0.0; 3], [false; 3], ratio, pressure_floor),
            CompactionPath::MinimumIncrementalWork { active_axes } => (
                "minimum_incremental_work",
                [0.0; 3],
                active_axes,
                [0.0; 3],
                1.0e-12,
            ),
        };
        Self {
            target_type: target_type.to_string(),
            target_value,
            target_values,
            path: path.to_string(),
            axis_weights,
            active_axes,
            stress_ratio,
            pressure_floor,
            kinematics: match config.kinematics {
                CompactionKinematics::RigidFiberCenters => "rigid_fiber_centers",
                CompactionKinematics::MovingWalls => "moving_walls",
                CompactionKinematics::AffineVertices => "affine_vertices",
            }
            .to_string(),
            cell_anchor: config.cell_anchor,
            balance_opposing_faces: config.balance_opposing_faces,
            face_pressure_floor: config.face_pressure_floor,
            face_balance_strength: config.face_balance_strength,
            initial_log_strain: config.increment.initial_log_strain,
            minimum_log_strain: config.increment.minimum_log_strain,
            maximum_log_strain: config.increment.maximum_log_strain,
            growth_factor: config.increment.growth_factor,
            shrink_factor: config.increment.shrink_factor,
            relax_iterations: config.increment.relax_iterations,
            maximum_shortening_over_minimum_diameter: config
                .increment
                .maximum_shortening_over_minimum_diameter,
            maximum_penetration: config.guards.maximum_penetration,
            maximum_bend_ratio: config.guards.maximum_bend_ratio,
            maximum_pressure: config.guards.maximum_pressure,
            maximum_penalty_energy: config.guards.maximum_penalty_energy,
            maximum_steps: config.guards.maximum_steps,
            maximum_relax_windows: config.guards.maximum_relax_windows,
            contact_energy_stiffness: config.energy_model.contact_stiffness,
            stretch_energy_stiffness: config.energy_model.stretch_stiffness,
            bending_energy_stiffness: config.energy_model.bending_stiffness,
            target_tolerance: config.target_tolerance,
        }
    }

    pub(crate) fn to_rust(&self) -> PyResult<CompactionConfig> {
        let target = match self.target_type.as_str() {
            "volume_fraction" => {
                positive_f64(self.target_value, "target_value")?;
                if self.target_value >= 1.0 {
                    return Err(PyValueError::new_err(
                        "volume fraction target must be less than 1",
                    ));
                }
                CompactionTarget::NominalVolumeFraction(self.target_value)
            }
            "cell_volume" => {
                CompactionTarget::CellVolume(positive_f64(self.target_value, "target_value")?)
            }
            "cell_lengths" => {
                validate_positive_f64s(self.target_values, "target_values")?;
                CompactionTarget::CellLengths(self.target_values)
            }
            "mean_pressure" => CompactionTarget::MeanPressure(positive_f64(
                self.target_value,
                "target_value",
            )? as f32),
            "directional_pressure" => {
                let values = self.target_values.map(|value| value as f32);
                validate_nonnegative_active(values, "target_values")?;
                CompactionTarget::DirectionalPressure(values)
            }
            "penalty_energy" => CompactionTarget::PenaltyEnergy(positive_f64(
                self.target_value,
                "target_value",
            )? as f32),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown compaction target_type {other:?}"
                )))
            }
        };
        let path = match self.path.as_str() {
            "axis_weights" => {
                validate_nonnegative_active(self.axis_weights, "axis_weights")?;
                CompactionPath::AxisWeights(self.axis_weights)
            }
            "equal_pressure" => {
                require_active_axes(self.active_axes)?;
                CompactionPath::EqualPressure {
                    active_axes: self.active_axes,
                    pressure_floor: positive_f32(self.pressure_floor, "pressure_floor")?,
                }
            }
            "stress_ratio" => {
                validate_nonnegative_active(self.stress_ratio, "stress_ratio")?;
                CompactionPath::StressRatio {
                    ratio: self.stress_ratio,
                    pressure_floor: positive_f32(self.pressure_floor, "pressure_floor")?,
                }
            }
            "minimum_incremental_work" => {
                require_active_axes(self.active_axes)?;
                CompactionPath::MinimumIncrementalWork {
                    active_axes: self.active_axes,
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown compaction path {other:?}"
                )))
            }
        };
        let kinematics = match self.kinematics.as_str() {
            "rigid_fiber_centers" => CompactionKinematics::RigidFiberCenters,
            "moving_walls" => CompactionKinematics::MovingWalls,
            "affine_vertices" => CompactionKinematics::AffineVertices,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown compaction kinematics {other:?}"
                )))
            }
        };
        self.validate()?;
        Ok(CompactionConfig {
            target,
            path,
            kinematics,
            cell_anchor: self.cell_anchor,
            balance_opposing_faces: self.balance_opposing_faces,
            face_pressure_floor: self.face_pressure_floor,
            face_balance_strength: self.face_balance_strength,
            increment: AdaptiveCompactionIncrement {
                initial_log_strain: self.initial_log_strain,
                minimum_log_strain: self.minimum_log_strain,
                maximum_log_strain: self.maximum_log_strain,
                growth_factor: self.growth_factor,
                shrink_factor: self.shrink_factor,
                relax_iterations: self.relax_iterations,
                maximum_shortening_over_minimum_diameter: self
                    .maximum_shortening_over_minimum_diameter,
            },
            guards: CompactionGuards {
                maximum_penetration: self.maximum_penetration,
                maximum_bend_ratio: self.maximum_bend_ratio,
                maximum_pressure: self.maximum_pressure,
                maximum_penalty_energy: self.maximum_penalty_energy,
                maximum_steps: self.maximum_steps,
                maximum_relax_windows: self.maximum_relax_windows,
            },
            energy_model: CompactionEnergyModel {
                contact_stiffness: self.contact_energy_stiffness,
                stretch_stiffness: self.stretch_energy_stiffness,
                bending_stiffness: self.bending_energy_stiffness,
            },
            target_tolerance: self.target_tolerance,
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
        positive_f32(self.face_pressure_floor, "face_pressure_floor")?;
        if !self.face_balance_strength.is_finite()
            || !(0.0..=1.0).contains(&self.face_balance_strength)
        {
            return Err(PyValueError::new_err(
                "face_balance_strength must be in [0, 1]",
            ));
        }
        for (name, value) in [
            ("initial_log_strain", self.initial_log_strain),
            ("minimum_log_strain", self.minimum_log_strain),
            ("maximum_log_strain", self.maximum_log_strain),
            ("growth_factor", self.growth_factor),
            ("shrink_factor", self.shrink_factor),
            (
                "maximum_shortening_over_minimum_diameter",
                self.maximum_shortening_over_minimum_diameter,
            ),
            ("maximum_penetration", self.maximum_penetration),
            ("maximum_bend_ratio", self.maximum_bend_ratio),
            ("contact_energy_stiffness", self.contact_energy_stiffness),
            ("stretch_energy_stiffness", self.stretch_energy_stiffness),
            ("bending_energy_stiffness", self.bending_energy_stiffness),
            ("target_tolerance", self.target_tolerance),
        ] {
            positive_f32(value, name)?;
        }
        if self.minimum_log_strain > self.initial_log_strain
            || self.initial_log_strain > self.maximum_log_strain
        {
            return Err(PyValueError::new_err(
                "log strain increments must satisfy minimum <= initial <= maximum",
            ));
        }
        if self.relax_iterations == 0 || self.maximum_steps == 0 || self.maximum_relax_windows == 0
        {
            return Err(PyValueError::new_err(
                "compaction iteration and step counts must be positive",
            ));
        }
        for (name, value) in [
            ("maximum_pressure", self.maximum_pressure),
            ("maximum_penalty_energy", self.maximum_penalty_energy),
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

fn positive_f32(value: f32, name: &str) -> PyResult<f32> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive and finite"
        )))
    }
}
fn positive_f64(value: f64, name: &str) -> PyResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive and finite"
        )))
    }
}
fn validate_positive_f64s(values: [f64; 3], name: &str) -> PyResult<()> {
    for value in values {
        positive_f64(value, name)?;
    }
    Ok(())
}
fn validate_nonnegative_active(values: [f32; 3], name: &str) -> PyResult<()> {
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
