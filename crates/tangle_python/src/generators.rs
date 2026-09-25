use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tangle_core::FiberAssembly;
use tangle_generate::{
    generate_biased_fiber_population, generate_fiber_pair_crossing, generate_multisegment_crossing,
    generate_point_crossing, CenterlineShape, FiberPairCrossingConfig, FiberPopulationSpec,
    MultiSegmentCrossingConfig, OrientationDistribution, PointCrossingConfig, PositionDistribution,
    ScalarDistribution,
};

use crate::collection::{PyCell, PyFiberCollection, PyMaterial};
use crate::common::{
    direction_to_py, parse_axis, parse_choice, parse_direction, parse_range, range_to_py,
    repr_fields, scale_range, unit_vector, with_kwargs,
};

// --- Orientation variants ----------------------------------------------------

/// Uniform directions over the unit sphere.
#[pyclass(name = "IsotropicOrientation", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyIsotropicOrientation;

#[pymethods]
impl PyIsotropicOrientation {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "IsotropicOrientation()".to_string()
    }
}

/// Directions concentrated around a plane. `normal=None` uses the cell's
/// stack axis.
#[pyclass(name = "PlanarOrientation", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyPlanarOrientation {
    normal: Option<[f64; 3]>,
    #[pyo3(get)]
    max_tilt: f64,
}

#[pymethods]
impl PyPlanarOrientation {
    #[new]
    #[pyo3(signature = (*, normal=None, max_tilt=10.0_f64.to_radians()))]
    fn new(normal: Option<&Bound<'_, PyAny>>, max_tilt: f64) -> PyResult<Self> {
        Ok(Self {
            normal: normal
                .map(|value| parse_direction(value, "normal"))
                .transpose()?,
            max_tilt,
        })
    }

    #[getter]
    fn normal(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.normal
            .map(|normal| direction_to_py(py, normal))
            .transpose()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "PlanarOrientation",
            &["normal", "max_tilt"],
            false,
        )
    }
}

/// Layer-aware planar mixture: a seeded reference direction per layer, its
/// in-plane perpendicular, and a uniformly random remainder.
#[pyclass(name = "LayeredBiaxialOrientation", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyLayeredBiaxialOrientation {
    normal: Option<[f64; 3]>,
    #[pyo3(get)]
    primary_fraction: f64,
    #[pyo3(get)]
    cross_fraction: f64,
    #[pyo3(get)]
    max_in_plane_deviation: f64,
    #[pyo3(get)]
    max_tilt: f64,
    #[pyo3(get)]
    seed: u64,
}

#[pymethods]
impl PyLayeredBiaxialOrientation {
    #[new]
    #[pyo3(signature = (
        *,
        normal=None,
        primary_fraction=0.4,
        cross_fraction=0.4,
        max_in_plane_deviation=10.0_f64.to_radians(),
        max_tilt=10.0_f64.to_radians(),
        seed=1
    ))]
    fn new(
        normal: Option<&Bound<'_, PyAny>>,
        primary_fraction: f64,
        cross_fraction: f64,
        max_in_plane_deviation: f64,
        max_tilt: f64,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            normal: normal
                .map(|value| parse_direction(value, "normal"))
                .transpose()?,
            primary_fraction,
            cross_fraction,
            max_in_plane_deviation,
            max_tilt,
            seed,
        })
    }

    #[getter]
    fn normal(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.normal
            .map(|normal| direction_to_py(py, normal))
            .transpose()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "LayeredBiaxialOrientation",
            &[
                "normal",
                "primary_fraction",
                "cross_fraction",
                "max_in_plane_deviation",
                "max_tilt",
                "seed",
            ],
            false,
        )
    }
}

/// Directions within a cone of half-angle `max_angle` around `axis`.
#[pyclass(name = "AlignedOrientation", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyAlignedOrientation {
    axis: [f64; 3],
    #[pyo3(get)]
    max_angle: f64,
}

#[pymethods]
impl PyAlignedOrientation {
    #[new]
    #[pyo3(signature = (axis, *, max_angle=15.0_f64.to_radians()))]
    fn new(axis: &Bound<'_, PyAny>, max_angle: f64) -> PyResult<Self> {
        Ok(Self {
            axis: parse_direction(axis, "axis")?,
            max_angle,
        })
    }

    #[getter]
    fn axis(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        direction_to_py(py, self.axis)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "AlignedOrientation",
            &["axis", "max_angle"],
            false,
        )
    }
}

#[derive(Clone, Debug)]
enum Orientation {
    Isotropic,
    Planar(PyPlanarOrientation),
    LayeredBiaxial(PyLayeredBiaxialOrientation),
    Aligned(PyAlignedOrientation),
}

impl Orientation {
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.extract::<PyRef<'_, PyIsotropicOrientation>>().is_ok() {
            Ok(Self::Isotropic)
        } else if let Ok(planar) = value.extract::<PyRef<'_, PyPlanarOrientation>>() {
            Ok(Self::Planar(planar.clone()))
        } else if let Ok(layered) = value.extract::<PyRef<'_, PyLayeredBiaxialOrientation>>() {
            Ok(Self::LayeredBiaxial(layered.clone()))
        } else if let Ok(aligned) = value.extract::<PyRef<'_, PyAlignedOrientation>>() {
            Ok(Self::Aligned(aligned.clone()))
        } else {
            Err(PyTypeError::new_err(
                "orientation must be IsotropicOrientation, PlanarOrientation, LayeredBiaxialOrientation, or AlignedOrientation",
            ))
        }
    }

    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::Isotropic => Py::new(py, PyIsotropicOrientation)?.into_any(),
            Self::Planar(value) => Py::new(py, value.clone())?.into_any(),
            Self::LayeredBiaxial(value) => Py::new(py, value.clone())?.into_any(),
            Self::Aligned(value) => Py::new(py, value.clone())?.into_any(),
        })
    }

    fn to_rust(&self, stack_axis: usize) -> OrientationDistribution {
        let normal = |value: Option<[f64; 3]>| value.unwrap_or_else(|| unit_vector(stack_axis));
        match self {
            Self::Isotropic => OrientationDistribution::Isotropic3d,
            Self::Planar(value) => OrientationDistribution::Planar {
                normal: normal(value.normal),
                maximum_tilt: value.max_tilt,
            },
            Self::LayeredBiaxial(value) => OrientationDistribution::LayeredBiaxial {
                normal: normal(value.normal),
                primary_fraction: value.primary_fraction,
                cross_fraction: value.cross_fraction,
                maximum_in_plane_deviation: value.max_in_plane_deviation,
                maximum_tilt: value.max_tilt,
                layer_seed: value.seed,
            },
            Self::Aligned(value) => OrientationDistribution::Aligned {
                axis: value.axis,
                maximum_angle: value.max_angle,
            },
        }
    }
}

// --- Position variants -------------------------------------------------------

/// Uniform centers over each fiber's feasible contained region.
#[pyclass(name = "UniformPosition", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyUniformPosition;

#[pymethods]
impl PyUniformPosition {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "UniformPosition()".to_string()
    }
}

/// Centers concentrated on `layer_count` evenly spaced layers. Each fiber's
/// `formation_layer` records its layer. `axis=None` uses the cell's stack axis.
#[pyclass(name = "LayeredPosition", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyLayeredPosition {
    #[pyo3(get)]
    layer_count: usize,
    #[pyo3(get)]
    axis: Option<usize>,
    #[pyo3(get)]
    jitter_fraction: f64,
}

#[pymethods]
impl PyLayeredPosition {
    #[new]
    #[pyo3(signature = (layer_count, *, axis=None, jitter_fraction=0.25))]
    fn new(
        layer_count: usize,
        axis: Option<&Bound<'_, PyAny>>,
        jitter_fraction: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            layer_count,
            axis: axis.map(|value| parse_axis(value, "axis")).transpose()?,
            jitter_fraction,
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "LayeredPosition",
            &["layer_count", "axis", "jitter_fraction"],
            false,
        )
    }
}

/// Monotonic density gradient along one axis. `axis=None` uses the cell's
/// stack axis.
#[pyclass(name = "DensityGradientPosition", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyDensityGradientPosition {
    #[pyo3(get)]
    axis: Option<usize>,
    #[pyo3(get)]
    exponent: f64,
    #[pyo3(get)]
    toward_high: bool,
}

#[pymethods]
impl PyDensityGradientPosition {
    #[new]
    #[pyo3(signature = (*, axis=None, exponent=1.0, toward_high=true))]
    fn new(axis: Option<&Bound<'_, PyAny>>, exponent: f64, toward_high: bool) -> PyResult<Self> {
        Ok(Self {
            axis: axis.map(|value| parse_axis(value, "axis")).transpose()?,
            exponent,
            toward_high,
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "DensityGradientPosition",
            &["axis", "exponent", "toward_high"],
            false,
        )
    }
}

#[derive(Clone, Debug)]
enum Position {
    Uniform,
    Layered(PyLayeredPosition),
    DensityGradient(PyDensityGradientPosition),
}

impl Position {
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.extract::<PyRef<'_, PyUniformPosition>>().is_ok() {
            Ok(Self::Uniform)
        } else if let Ok(layered) = value.extract::<PyRef<'_, PyLayeredPosition>>() {
            Ok(Self::Layered(layered.clone()))
        } else if let Ok(gradient) = value.extract::<PyRef<'_, PyDensityGradientPosition>>() {
            Ok(Self::DensityGradient(gradient.clone()))
        } else {
            Err(PyTypeError::new_err(
                "position must be UniformPosition, LayeredPosition, or DensityGradientPosition",
            ))
        }
    }

    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::Uniform => Py::new(py, PyUniformPosition)?.into_any(),
            Self::Layered(value) => Py::new(py, value.clone())?.into_any(),
            Self::DensityGradient(value) => Py::new(py, value.clone())?.into_any(),
        })
    }

    fn to_rust(&self, stack_axis: usize) -> PositionDistribution {
        match self {
            Self::Uniform => PositionDistribution::Uniform,
            Self::Layered(value) => PositionDistribution::Layered {
                axis: value.axis.unwrap_or(stack_axis),
                layers: value.layer_count,
                jitter_fraction: value.jitter_fraction,
            },
            Self::DensityGradient(value) => PositionDistribution::DensityGradient {
                axis: value.axis.unwrap_or(stack_axis),
                exponent: value.exponent,
                toward_high: value.toward_high,
            },
        }
    }
}

// --- Population --------------------------------------------------------------

/// A reproducible description of a biased fiber population.
///
/// Every field can be passed to the constructor as a keyword. `diameter=None`
/// uses the material's diameter; ranges are `(min, max)` tuples.
#[pyclass(name = "FiberPopulation", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyFiberPopulation {
    #[pyo3(get, set)]
    pub material: PyMaterial,
    #[pyo3(get, set)]
    pub count: usize,
    #[pyo3(get, set)]
    pub segments_per_fiber: usize,
    #[pyo3(get, set)]
    pub seed: u64,
    length: ScalarDistribution,
    diameter: Option<ScalarDistribution>,
    curvature_amplitude: ScalarDistribution,
    /// Physical length of the parent fiber that each generated centerline
    /// represents a window of. Recorded in provenance; it does not change
    /// the generated geometry.
    #[pyo3(get, set)]
    pub nominal_parent_length: Option<f64>,
    orientation: Orientation,
    position: Position,
    #[pyo3(get, set)]
    pub max_attempts_per_fiber: usize,
}

impl Default for PyFiberPopulation {
    fn default() -> Self {
        let spec = FiberPopulationSpec::default();
        let mean_diameter = match spec.radius {
            ScalarDistribution::Constant(radius) => 2.0 * radius,
            ScalarDistribution::Uniform { minimum, maximum } => minimum + maximum,
        };
        Self {
            material: PyMaterial {
                name: spec.material_name,
                diameter: mean_diameter,
                min_bend_radius: spec.minimum_bend_radius,
                thickness: None,
            },
            count: spec.count,
            segments_per_fiber: spec.segments_per_fiber,
            seed: spec.seed,
            length: spec.length,
            diameter: None,
            curvature_amplitude: spec.intrinsic_curvature_amplitude,
            nominal_parent_length: spec.nominal_parent_length,
            orientation: Orientation::Isotropic,
            position: Position::Uniform,
            max_attempts_per_fiber: spec.max_attempts_per_fiber,
        }
    }
}

#[pymethods]
impl PyFiberPopulation {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let population = with_kwargs(py, Self::default(), kwargs, "FiberPopulation")?;
        population.check_combination()?;
        Ok(population)
    }

    #[getter]
    fn length(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        range_to_py(py, self.length)
    }

    #[setter]
    fn set_length(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.length = parse_range(value, "length")?;
        Ok(())
    }

    #[getter]
    fn diameter(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.diameter
            .map(|value| range_to_py(py, value))
            .transpose()
    }

    #[setter]
    fn set_diameter(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.diameter = value
            .map(|value| parse_range(value, "diameter"))
            .transpose()?;
        Ok(())
    }

    #[getter]
    fn curvature_amplitude(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        range_to_py(py, self.curvature_amplitude)
    }

    #[setter]
    fn set_curvature_amplitude(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.curvature_amplitude = parse_range(value, "curvature_amplitude")?;
        Ok(())
    }

    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.orientation.to_py(py)
    }

    #[setter]
    fn set_orientation(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.orientation = Orientation::from_py(value)?;
        Ok(())
    }

    #[getter]
    fn position(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.position.to_py(py)
    }

    #[setter]
    fn set_position(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.position = Position::from_py(value)?;
        Ok(())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[pyo3(signature = (**changes))]
    fn replace(&self, py: Python<'_>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let population = with_kwargs(py, self.clone(), changes, "replace")?;
        population.check_combination()?;
        Ok(population)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        repr_fields(
            slf.as_any(),
            "FiberPopulation",
            &[
                "material",
                "count",
                "segments_per_fiber",
                "seed",
                "length",
                "diameter",
                "orientation",
                "position",
            ],
            false,
        )
    }
}

impl PyFiberPopulation {
    /// Rejects variant pairs the generator cannot combine.
    fn check_combination(&self) -> PyResult<()> {
        if matches!(self.orientation, Orientation::LayeredBiaxial(_))
            && !matches!(self.position, Position::Layered(_))
        {
            return Err(PyValueError::new_err(
                "LayeredBiaxialOrientation picks a direction per layer, so it needs position=LayeredPosition(...)",
            ));
        }
        Ok(())
    }

    fn to_rust(&self, stack_axis: usize) -> FiberPopulationSpec {
        let diameter = self
            .diameter
            .unwrap_or(ScalarDistribution::Constant(self.material.diameter));
        FiberPopulationSpec {
            count: self.count,
            segments_per_fiber: self.segments_per_fiber,
            seed: self.seed,
            length: self.length,
            nominal_parent_length: self.nominal_parent_length,
            radius: scale_range(diameter, 0.5),
            intrinsic_curvature_amplitude: self.curvature_amplitude,
            orientation: self.orientation.to_rust(stack_axis),
            position: self.position.to_rust(stack_axis),
            minimum_bend_radius: self.material.min_bend_radius,
            boundary: Default::default(),
            max_attempts_per_fiber: self.max_attempts_per_fiber,
            material_name: self.material.name.clone(),
        }
    }
}

// --- Generators --------------------------------------------------------------

fn material_or_default(
    material: Option<PyRef<'_, PyMaterial>>,
    diameter: f64,
    generator: &str,
) -> PyResult<PyMaterial> {
    let material = material
        .map(|material| material.clone())
        .unwrap_or(PyMaterial {
            name: "fiber".to_string(),
            diameter,
            min_bend_radius: None,
            thickness: None,
        });
    material.require_round(generator)?;
    Ok(material)
}

/// Gives every generated fiber the caller's bend limit when the native
/// generator has no bend-limit input of its own.
fn apply_bend_limit(mut collection: PyFiberCollection, material: &PyMaterial) -> PyFiberCollection {
    for fiber in &mut collection.fibers {
        fiber.material.min_bend_radius = material.min_bend_radius;
    }
    collection
}

#[pyfunction]
#[pyo3(name = "generate_point_crossing", signature = (cell, *, material=None, count=8, length=0.7, name="point crossing".to_string()))]
pub(crate) fn generate_point_crossing_py(
    cell: PyRef<'_, PyCell>,
    material: Option<PyRef<'_, PyMaterial>>,
    count: usize,
    length: f64,
    name: String,
) -> PyResult<PyFiberCollection> {
    let material = material_or_default(material, 0.05, "generate_point_crossing")?;
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_point_crossing(
        &mut assembly,
        &PointCrossingConfig {
            count,
            length,
            radius: 0.5 * material.diameter,
            material_name: material.name.clone(),
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(apply_bend_limit(
        PyFiberCollection::from_assembly(name, &assembly)?,
        &material,
    ))
}

#[pyfunction]
#[pyo3(name = "generate_multisegment_crossing", signature = (cell, *, material=None, count=8, segments_per_fiber=8, length=0.7, placed_chord_fraction=1.0, rest_shape="straight", rest_amplitude=0.0, placed_shape="straight", placed_amplitude=0.0, name="multisegment crossing".to_string()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_multisegment_crossing_py(
    cell: PyRef<'_, PyCell>,
    material: Option<PyRef<'_, PyMaterial>>,
    count: usize,
    segments_per_fiber: usize,
    length: f64,
    placed_chord_fraction: f64,
    rest_shape: &str,
    rest_amplitude: f64,
    placed_shape: &str,
    placed_amplitude: f64,
    name: String,
) -> PyResult<PyFiberCollection> {
    let material = material_or_default(material, 0.036, "generate_multisegment_crossing")?;
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_multisegment_crossing(
        &mut assembly,
        &MultiSegmentCrossingConfig {
            count,
            segments_per_fiber,
            length,
            placed_chord_fraction,
            radius: 0.5 * material.diameter,
            intrinsic_shape: shape(rest_shape, rest_amplitude, "rest_shape")?,
            placed_shape: shape(placed_shape, placed_amplitude, "placed_shape")?,
            minimum_bend_radius: material.min_bend_radius,
            material_name: material.name.clone(),
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

#[pyfunction]
#[pyo3(name = "generate_fiber_pair_crossing", signature = (cell, *, material=None, segments_per_fiber=1, length=0.8, axis_separation=0.04, crossing_angle_degrees=90.0, name="fiber pair crossing".to_string()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_fiber_pair_crossing_py(
    cell: PyRef<'_, PyCell>,
    material: Option<PyRef<'_, PyMaterial>>,
    segments_per_fiber: usize,
    length: f64,
    axis_separation: f64,
    crossing_angle_degrees: f64,
    name: String,
) -> PyResult<PyFiberCollection> {
    let material = material_or_default(material, 0.05, "generate_fiber_pair_crossing")?;
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_fiber_pair_crossing(
        &mut assembly,
        &FiberPairCrossingConfig {
            segments_per_fiber,
            length,
            radius: 0.5 * material.diameter,
            axis_separation,
            crossing_angle_degrees,
            minimum_bend_radius: material.min_bend_radius,
            material_name: material.name.clone(),
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

#[pyfunction]
#[pyo3(name = "generate_fiber_population", signature = (cell, population, *, name="fiber population".to_string()))]
pub(crate) fn generate_fiber_population_py(
    cell: PyRef<'_, PyCell>,
    population: PyRef<'_, PyFiberPopulation>,
    name: String,
) -> PyResult<PyFiberCollection> {
    let mut assembly = FiberAssembly::new(cell.inner);
    population.material.require_round("generate_fiber_population")?;
    population.check_combination()?;
    generate_biased_fiber_population(&mut assembly, &population.to_rust(cell.stack_axis))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

fn shape(name: &str, amplitude: f64, label: &str) -> PyResult<CenterlineShape> {
    #[derive(Clone, Copy)]
    enum Kind {
        Straight,
        Curved,
    }
    Ok(
        match parse_choice(
            name,
            label,
            &[("straight", Kind::Straight), ("curved", Kind::Curved)],
        )? {
            Kind::Straight => CenterlineShape::Straight,
            Kind::Curved => CenterlineShape::Curved { amplitude },
        },
    )
}
