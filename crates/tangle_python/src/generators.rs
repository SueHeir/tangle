use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tangle_core::FiberAssembly;
use tangle_generate::{
    generate_biased_fiber_population, generate_fiber_pair_crossing, generate_multisegment_crossing,
    generate_point_crossing, CenterlineShape, FiberPairCrossingConfig, FiberPopulationSpec,
    MultiSegmentCrossingConfig, OrientationDistribution, PointCrossingConfig, PositionDistribution,
    ScalarDistribution,
};

use crate::collection::{PyCell, PyFiberCollection};

/// Fully editable native biased-population generator settings.
#[pyclass(name = "FiberPopulationSettings", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyFiberPopulationSettings {
    #[pyo3(get, set)]
    pub count: usize,
    #[pyo3(get, set)]
    pub segments_per_fiber: usize,
    #[pyo3(get, set)]
    pub seed: u64,
    #[pyo3(get, set)]
    pub length_minimum: f64,
    #[pyo3(get, set)]
    pub length_maximum: f64,
    #[pyo3(get, set)]
    pub nominal_parent_length: Option<f64>,
    #[pyo3(get, set)]
    pub radius_minimum: f64,
    #[pyo3(get, set)]
    pub radius_maximum: f64,
    #[pyo3(get, set)]
    pub curvature_amplitude_minimum: f64,
    #[pyo3(get, set)]
    pub curvature_amplitude_maximum: f64,
    #[pyo3(get, set)]
    pub orientation: String,
    #[pyo3(get, set)]
    pub orientation_axis: [f64; 3],
    #[pyo3(get, set)]
    pub maximum_angle: f64,
    #[pyo3(get, set)]
    pub maximum_tilt: f64,
    #[pyo3(get, set)]
    pub primary_fraction: f64,
    #[pyo3(get, set)]
    pub cross_fraction: f64,
    #[pyo3(get, set)]
    pub maximum_in_plane_deviation: f64,
    #[pyo3(get, set)]
    pub layer_orientation_seed: u64,
    #[pyo3(get, set)]
    pub position: String,
    #[pyo3(get, set)]
    pub position_axis: usize,
    #[pyo3(get, set)]
    pub layers: usize,
    #[pyo3(get, set)]
    pub jitter_fraction: f64,
    #[pyo3(get, set)]
    pub density_exponent: f64,
    #[pyo3(get, set)]
    pub density_toward_high: bool,
    #[pyo3(get, set)]
    pub minimum_bend_radius: Option<f64>,
    #[pyo3(get, set)]
    pub max_attempts_per_fiber: usize,
    #[pyo3(get, set)]
    pub material_name: String,
}

impl Default for PyFiberPopulationSettings {
    fn default() -> Self {
        let spec = FiberPopulationSpec::default();
        let (length_minimum, length_maximum) = scalar_bounds(spec.length);
        let (radius_minimum, radius_maximum) = scalar_bounds(spec.radius);
        let (curvature_amplitude_minimum, curvature_amplitude_maximum) =
            scalar_bounds(spec.intrinsic_curvature_amplitude);
        Self {
            count: spec.count,
            segments_per_fiber: spec.segments_per_fiber,
            seed: spec.seed,
            length_minimum,
            length_maximum,
            nominal_parent_length: spec.nominal_parent_length,
            radius_minimum,
            radius_maximum,
            curvature_amplitude_minimum,
            curvature_amplitude_maximum,
            orientation: "isotropic_3d".to_string(),
            orientation_axis: [0.0, 0.0, 1.0],
            maximum_angle: 15.0_f64.to_radians(),
            maximum_tilt: 10.0_f64.to_radians(),
            primary_fraction: 0.4,
            cross_fraction: 0.4,
            maximum_in_plane_deviation: 10.0_f64.to_radians(),
            layer_orientation_seed: 1,
            position: "uniform".to_string(),
            position_axis: 2,
            layers: 1,
            jitter_fraction: 0.25,
            density_exponent: 1.0,
            density_toward_high: true,
            minimum_bend_radius: spec.minimum_bend_radius,
            max_attempts_per_fiber: spec.max_attempts_per_fiber,
            material_name: spec.material_name,
        }
    }
}

#[pymethods]
impl PyFiberPopulationSettings {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FiberPopulationSettings(count={}, segments_per_fiber={}, orientation={:?}, position={:?}, material_name={:?})",
            self.count,
            self.segments_per_fiber,
            self.orientation,
            self.position,
            self.material_name
        )
    }
}

impl PyFiberPopulationSettings {
    fn to_rust(&self) -> PyResult<FiberPopulationSpec> {
        let orientation = match self.orientation.as_str() {
            "isotropic_3d" => OrientationDistribution::Isotropic3d,
            "planar" => OrientationDistribution::Planar {
                normal: self.orientation_axis,
                maximum_tilt: self.maximum_tilt,
            },
            "layered_biaxial" => OrientationDistribution::LayeredBiaxial {
                normal: self.orientation_axis,
                primary_fraction: self.primary_fraction,
                cross_fraction: self.cross_fraction,
                maximum_in_plane_deviation: self.maximum_in_plane_deviation,
                maximum_tilt: self.maximum_tilt,
                layer_seed: self.layer_orientation_seed,
            },
            "aligned" => OrientationDistribution::Aligned {
                axis: self.orientation_axis,
                maximum_angle: self.maximum_angle,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown orientation {other:?}; expected 'isotropic_3d', 'planar', 'layered_biaxial', or 'aligned'"
                )))
            }
        };
        let position = match self.position.as_str() {
            "uniform" => PositionDistribution::Uniform,
            "layered" => PositionDistribution::Layered {
                axis: self.position_axis,
                layers: self.layers,
                jitter_fraction: self.jitter_fraction,
            },
            "density_gradient" => PositionDistribution::DensityGradient {
                axis: self.position_axis,
                exponent: self.density_exponent,
                toward_high: self.density_toward_high,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                "unknown position {other:?}; expected 'uniform', 'layered', or 'density_gradient'"
            )))
            }
        };
        Ok(FiberPopulationSpec {
            count: self.count,
            segments_per_fiber: self.segments_per_fiber,
            seed: self.seed,
            length: scalar(self.length_minimum, self.length_maximum),
            nominal_parent_length: self.nominal_parent_length,
            radius: scalar(self.radius_minimum, self.radius_maximum),
            intrinsic_curvature_amplitude: scalar(
                self.curvature_amplitude_minimum,
                self.curvature_amplitude_maximum,
            ),
            orientation,
            position,
            minimum_bend_radius: self.minimum_bend_radius,
            boundary: Default::default(),
            max_attempts_per_fiber: self.max_attempts_per_fiber,
            material_name: self.material_name.clone(),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (cell, *, count=8, length=0.7, radius=0.025, material_name="fiber".to_string(), name="point crossing".to_string()))]
pub(crate) fn py_generate_point_crossing(
    cell: PyRef<'_, PyCell>,
    count: usize,
    length: f64,
    radius: f64,
    material_name: String,
    name: String,
) -> PyResult<PyFiberCollection> {
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_point_crossing(
        &mut assembly,
        &PointCrossingConfig {
            count,
            length,
            radius,
            material_name,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

#[pyfunction]
#[pyo3(signature = (cell, *, count=8, segments_per_fiber=8, length=0.7, placed_chord_fraction=1.0, radius=0.018, rest_shape="straight", rest_amplitude=0.0, placed_shape="straight", placed_amplitude=0.0, minimum_bend_radius=None, material_name="fiber".to_string(), name="multisegment crossing".to_string()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_generate_multisegment_crossing(
    cell: PyRef<'_, PyCell>,
    count: usize,
    segments_per_fiber: usize,
    length: f64,
    placed_chord_fraction: f64,
    radius: f64,
    rest_shape: &str,
    rest_amplitude: f64,
    placed_shape: &str,
    placed_amplitude: f64,
    minimum_bend_radius: Option<f64>,
    material_name: String,
    name: String,
) -> PyResult<PyFiberCollection> {
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_multisegment_crossing(
        &mut assembly,
        &MultiSegmentCrossingConfig {
            count,
            segments_per_fiber,
            length,
            placed_chord_fraction,
            radius,
            intrinsic_shape: shape(rest_shape, rest_amplitude)?,
            placed_shape: shape(placed_shape, placed_amplitude)?,
            minimum_bend_radius,
            material_name,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

#[pyfunction]
#[pyo3(signature = (cell, *, segments_per_fiber=1, length=0.8, radius=0.025, axis_separation=0.04, crossing_angle_degrees=90.0, minimum_bend_radius=None, material_name="fiber".to_string(), name="fiber pair crossing".to_string()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_generate_fiber_pair_crossing(
    cell: PyRef<'_, PyCell>,
    segments_per_fiber: usize,
    length: f64,
    radius: f64,
    axis_separation: f64,
    crossing_angle_degrees: f64,
    minimum_bend_radius: Option<f64>,
    material_name: String,
    name: String,
) -> PyResult<PyFiberCollection> {
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_fiber_pair_crossing(
        &mut assembly,
        &FiberPairCrossingConfig {
            segments_per_fiber,
            length,
            radius,
            axis_separation,
            crossing_angle_degrees,
            minimum_bend_radius,
            material_name,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

#[pyfunction]
#[pyo3(signature = (cell, settings, *, name="fiber population".to_string()))]
pub(crate) fn py_generate_fiber_population(
    cell: PyRef<'_, PyCell>,
    settings: PyRef<'_, PyFiberPopulationSettings>,
    name: String,
) -> PyResult<PyFiberCollection> {
    let mut assembly = FiberAssembly::new(cell.inner);
    generate_biased_fiber_population(&mut assembly, &settings.to_rust()?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    PyFiberCollection::from_assembly(name, &assembly)
}

fn shape(name: &str, amplitude: f64) -> PyResult<CenterlineShape> {
    match name {
        "straight" => Ok(CenterlineShape::Straight),
        "curved" => Ok(CenterlineShape::Curved { amplitude }),
        other => Err(PyValueError::new_err(format!(
            "unknown centerline shape {other:?}; expected 'straight' or 'curved'"
        ))),
    }
}

fn scalar(minimum: f64, maximum: f64) -> ScalarDistribution {
    if minimum == maximum {
        ScalarDistribution::Constant(minimum)
    } else {
        ScalarDistribution::Uniform { minimum, maximum }
    }
}

fn scalar_bounds(distribution: ScalarDistribution) -> (f64, f64) {
    match distribution {
        ScalarDistribution::Constant(value) => (value, value),
        ScalarDistribution::Uniform { minimum, maximum } => (minimum, maximum),
    }
}
