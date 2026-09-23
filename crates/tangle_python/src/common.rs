//! Shared argument parsing for the Python bindings.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyClass;
use tangle_generate::ScalarDistribution;

/// Stacking axis used when neither the cell nor the caller names one.
pub(crate) const DEFAULT_STACK_AXIS: usize = 2;

/// Parses an axis given as `"x"`, `"y"`, `"z"`, or `0`, `1`, `2`.
pub(crate) fn parse_axis(value: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    if let Ok(text) = value.extract::<String>() {
        return match text.to_ascii_lowercase().as_str() {
            "x" => Ok(0),
            "y" => Ok(1),
            "z" => Ok(2),
            other => Err(PyValueError::new_err(format!(
                "{name} must be 'x', 'y', 'z', 0, 1, or 2; got {other:?}"
            ))),
        };
    }
    match value.extract::<usize>() {
        Ok(axis) if axis < 3 => Ok(axis),
        _ => Err(PyValueError::new_err(format!(
            "{name} must be 'x', 'y', 'z', 0, 1, or 2"
        ))),
    }
}

pub(crate) fn axis_name(axis: usize) -> &'static str {
    ["x", "y", "z"][axis]
}

/// Parses a set of axes given as letters (`"xy"`), one axis, or a bool triple.
pub(crate) fn parse_axis_mask(value: &Bound<'_, PyAny>, name: &str) -> PyResult<[bool; 3]> {
    if let Ok(text) = value.extract::<String>() {
        let text = text.to_ascii_lowercase();
        let mut mask = [false; 3];
        for letter in text.chars() {
            match letter {
                'x' => mask[0] = true,
                'y' => mask[1] = true,
                'z' => mask[2] = true,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "{name} letters must be drawn from 'xyz'; got {text:?}"
                    )))
                }
            }
        }
        return Ok(mask);
    }
    if let Ok(mask) = value.extract::<[bool; 3]>() {
        return Ok(mask);
    }
    let axis = parse_axis(value, name)?;
    let mut mask = [false; 3];
    mask[axis] = true;
    Ok(mask)
}

/// Parses a direction given as an axis name/index or as a 3-vector.
pub(crate) fn parse_direction(value: &Bound<'_, PyAny>, name: &str) -> PyResult<[f64; 3]> {
    if let Ok(vector) = value.extract::<[f64; 3]>() {
        if vector.iter().any(|component| !component.is_finite())
            || vector.iter().all(|component| *component == 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "{name} must be a finite, nonzero vector"
            )));
        }
        return Ok(vector);
    }
    Ok(unit_vector(parse_axis(value, name)?))
}

pub(crate) fn unit_vector(axis: usize) -> [f64; 3] {
    let mut vector = [0.0; 3];
    vector[axis] = 1.0;
    vector
}

/// Converts a Python direction back to its most readable form.
pub(crate) fn direction_to_py(py: Python<'_>, vector: [f64; 3]) -> PyResult<Py<PyAny>> {
    for axis in 0..3 {
        if vector == unit_vector(axis) {
            return Ok(axis_name(axis).into_pyobject(py)?.into_any().unbind());
        }
    }
    Ok(vector.into_pyobject(py)?.into_any().unbind())
}

/// Parses a fixed value (`0.2`) or a uniform range (`(0.1, 0.3)`).
pub(crate) fn parse_range(value: &Bound<'_, PyAny>, name: &str) -> PyResult<ScalarDistribution> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(ScalarDistribution::Constant(scalar));
    }
    if let Ok((minimum, maximum)) = value.extract::<(f64, f64)>() {
        if minimum > maximum {
            return Err(PyValueError::new_err(format!(
                "{name} range must be ordered as (min, max)"
            )));
        }
        return Ok(if minimum == maximum {
            ScalarDistribution::Constant(minimum)
        } else {
            ScalarDistribution::Uniform { minimum, maximum }
        });
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a number or a (min, max) tuple"
    )))
}

pub(crate) fn range_to_py(py: Python<'_>, value: ScalarDistribution) -> PyResult<Py<PyAny>> {
    Ok(match value {
        ScalarDistribution::Constant(scalar) => scalar.into_pyobject(py)?.into_any().unbind(),
        ScalarDistribution::Uniform { minimum, maximum } => {
            (minimum, maximum).into_pyobject(py)?.into_any().unbind()
        }
    })
}

pub(crate) fn scale_range(value: ScalarDistribution, factor: f64) -> ScalarDistribution {
    match value {
        ScalarDistribution::Constant(scalar) => ScalarDistribution::Constant(scalar * factor),
        ScalarDistribution::Uniform { minimum, maximum } => ScalarDistribution::Uniform {
            minimum: minimum * factor,
            maximum: maximum * factor,
        },
    }
}

/// Applies keyword arguments as attribute assignments on a fresh copy of
/// `value`, so constructors and `replace()` share the setters' validation.
pub(crate) fn with_kwargs<T>(
    py: Python<'_>,
    value: T,
    kwargs: Option<&Bound<'_, PyDict>>,
    class_name: &str,
) -> PyResult<T>
where
    T: PyClass + Clone + Into<PyClassInitializer<T>>,
{
    let Some(kwargs) = kwargs else {
        return Ok(value);
    };
    let bound = Bound::new(py, value)?;
    let object = bound.as_any();
    for (key, item) in kwargs.iter() {
        let key: String = key.extract()?;
        if key.starts_with('_') || !object.hasattr(key.as_str())? {
            return Err(PyTypeError::new_err(format!(
                "{class_name}() got an unexpected keyword argument {key:?}"
            )));
        }
        object.setattr(key.as_str(), item)?;
    }
    let result = bound.borrow().clone();
    Ok(result)
}

/// Parses a string option against its allowed spellings.
pub(crate) fn parse_choice<T: Copy>(value: &str, name: &str, choices: &[(&str, T)]) -> PyResult<T> {
    choices
        .iter()
        .find(|(label, _)| *label == value)
        .map(|(_, parsed)| *parsed)
        .ok_or_else(|| {
            let expected = choices
                .iter()
                .map(|(label, _)| format!("'{label}'"))
                .collect::<Vec<_>>()
                .join(", ");
            PyValueError::new_err(format!(
                "unknown {name} {value:?}; expected one of {expected}"
            ))
        })
}

pub(crate) fn choice_name<T: Copy + PartialEq>(
    value: T,
    choices: &[(&'static str, T)],
) -> &'static str {
    choices
        .iter()
        .find(|(_, parsed)| *parsed == value)
        .map(|(label, _)| *label)
        .expect("every enum value has a Python spelling")
}

pub(crate) fn positive_finite(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive and finite"
        )))
    }
}

pub(crate) fn nonnegative_finite(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be nonnegative and finite"
        )))
    }
}

pub(crate) fn unit_fraction(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in (0, 1]")))
    }
}

/// Widens an `f32` default to the `f64` a Python user would have typed, so
/// `1e-4_f32` reads back as `0.0001` rather than `9.99999974e-05`.
pub(crate) fn widen(value: f32) -> f64 {
    value.to_string().parse().unwrap_or(value as f64)
}

/// Formats `Class(field=repr(value), ...)` from Python-visible attributes, so
/// reprs show Python values (`None`, lists) rather than Rust debug output.
pub(crate) fn repr_fields(
    object: &Bound<'_, PyAny>,
    class_name: &str,
    fields: &[&str],
    skip_none: bool,
) -> PyResult<String> {
    let mut parts = Vec::with_capacity(fields.len());
    for field in fields {
        let value = object.getattr(*field)?;
        if skip_none && value.is_none() {
            continue;
        }
        parts.push(format!("{field}={}", value.repr()?));
    }
    Ok(format!("{class_name}({})", parts.join(", ")))
}
