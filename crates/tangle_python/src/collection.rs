use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use tangle_characterize::characterize_assembly;
use tangle_core::{FiberAssembly, FiberBendLimit, FiberId, PeriodicCell, Section, Vec3};
use tangle_export::{write_puma_bundle, PumaVoxelExportConfig};

use crate::analysis::{
    characterize_entanglement, characterize_neighbors, characterize_phases, characterize_shape,
    characterize_slices, PyAnalysisReport, PyEntanglementReport, PyNeighborReport, PyPhaseReport,
    PyPumaExportReport, PyShapeReport, PySliceReport,
};
use crate::common::{axis_name, parse_axis, parse_axis_mask, DEFAULT_STACK_AXIS};

/// Orthorhombic simulation cell. `stack_axis` is the direction plies stack
/// along; recipes, generators, and compaction default to it.
#[pyclass(name = "Cell", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyCell {
    pub(crate) inner: PeriodicCell,
    pub(crate) stack_axis: usize,
}

#[pymethods]
impl PyCell {
    #[new]
    #[pyo3(signature = (lengths, periodic=None, origin=[0.0, 0.0, 0.0], *, stack_axis=None))]
    fn new(
        lengths: [f64; 3],
        periodic: Option<&Bound<'_, PyAny>>,
        origin: [f64; 3],
        stack_axis: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if lengths
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(
                "cell lengths must be positive and finite",
            ));
        }
        if origin.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("cell origin must be finite"));
        }
        let periodic = periodic
            .map(|value| parse_axis_mask(value, "periodic"))
            .transpose()?
            .unwrap_or([false; 3]);
        let stack_axis = match stack_axis {
            Some(value) => parse_axis(value, "stack_axis")?,
            None => infer_stack_axis(periodic),
        };
        let mut cell = PeriodicCell::orthorhombic(lengths, periodic);
        cell.origin = origin;
        Ok(Self {
            inner: cell,
            stack_axis,
        })
    }

    #[getter]
    fn lengths(&self) -> [f64; 3] {
        cell_lengths(&self.inner)
    }

    #[getter]
    fn origin(&self) -> Vec3 {
        self.inner.origin
    }

    #[getter]
    fn periodic(&self) -> [bool; 3] {
        self.inner.periodic
    }

    #[getter]
    fn stack_axis(&self) -> usize {
        self.stack_axis
    }

    fn __repr__(&self) -> String {
        format!(
            "Cell(lengths={:?}, periodic={:?}, origin={:?}, stack_axis={:?})",
            self.lengths(),
            self.inner.periodic,
            self.inner.origin,
            axis_name(self.stack_axis)
        )
    }
}

/// The single non-periodic axis when there is exactly one, otherwise z.
/// Plies stack along a bounded axis: z when it is bounded or when every axis
/// is periodic, otherwise the last bounded axis.
fn infer_stack_axis(periodic: [bool; 3]) -> usize {
    if !periodic[DEFAULT_STACK_AXIS] {
        return DEFAULT_STACK_AXIS;
    }
    (0..3)
        .rev()
        .find(|axis| !periodic[*axis])
        .unwrap_or(DEFAULT_STACK_AXIS)
}

pub(crate) fn cell_lengths(cell: &PeriodicCell) -> [f64; 3] {
    [cell.basis[0][0], cell.basis[1][1], cell.basis[2][2]]
}

#[pyclass(name = "Material", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyMaterial {
    #[pyo3(get)]
    pub(crate) name: String,
    #[pyo3(get)]
    pub(crate) diameter: f64,
    #[pyo3(get)]
    pub(crate) min_bend_radius: Option<f64>,
}

#[pymethods]
impl PyMaterial {
    #[new]
    #[pyo3(signature = (name, diameter, min_bend_radius=None))]
    fn new(name: String, diameter: f64, min_bend_radius: Option<f64>) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err("material name must not be empty"));
        }
        if !diameter.is_finite() || diameter <= 0.0 {
            return Err(PyValueError::new_err(
                "material diameter must be positive and finite",
            ));
        }
        if min_bend_radius.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(PyValueError::new_err(
                "min_bend_radius must be positive and finite",
            ));
        }
        Ok(Self {
            name,
            diameter,
            min_bend_radius,
        })
    }

    #[getter]
    fn radius(&self) -> f64 {
        0.5 * self.diameter
    }

    fn __repr__(&self) -> String {
        match self.min_bend_radius {
            Some(radius) => format!(
                "Material(name={:?}, diameter={}, min_bend_radius={})",
                self.name, self.diameter, radius
            ),
            None => format!("Material(name={:?}, diameter={})", self.name, self.diameter),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CollectionFiber {
    pub(crate) placed: Vec<Vec3>,
    pub(crate) intrinsic: Vec<Vec3>,
    pub(crate) material: PyMaterial,
    pub(crate) tags: BTreeMap<String, String>,
    pub(crate) formation_layer: Option<u32>,
}

#[pyclass(name = "FiberCollection", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyFiberCollection {
    #[pyo3(get, set)]
    pub(crate) name: String,
    pub(crate) fibers: Vec<CollectionFiber>,
}

#[pymethods]
impl PyFiberCollection {
    #[new]
    #[pyo3(signature = (name="collection".to_string()))]
    fn new(name: String) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err("collection name must not be empty"));
        }
        Ok(Self {
            name,
            fibers: Vec::new(),
        })
    }

    #[pyo3(signature = (centerline, material, *, rest_centerline=None, tags=None, formation_layer=None))]
    fn add_fiber(
        &mut self,
        centerline: Vec<Vec3>,
        material: PyRef<'_, PyMaterial>,
        rest_centerline: Option<Vec<Vec3>>,
        tags: Option<HashMap<String, String>>,
        formation_layer: Option<u32>,
    ) -> PyResult<usize> {
        validate_centerline(&centerline, "centerline")?;
        let intrinsic = rest_centerline.unwrap_or_else(|| centerline.clone());
        validate_centerline(&intrinsic, "rest_centerline")?;
        if intrinsic.len() != centerline.len() {
            return Err(PyValueError::new_err(
                "centerline and rest_centerline must contain the same number of vertices",
            ));
        }
        self.fibers.push(CollectionFiber {
            placed: centerline,
            intrinsic,
            material: material.clone(),
            tags: tags.unwrap_or_default().into_iter().collect(),
            formation_layer,
        });
        Ok(self.fibers.len() - 1)
    }

    #[classmethod]
    #[pyo3(signature = (centerlines, material, *, name="collection".to_string(), formation_layer=None))]
    fn from_centerlines(
        _class: &Bound<'_, PyType>,
        centerlines: Vec<Vec<Vec3>>,
        material: PyRef<'_, PyMaterial>,
        name: String,
        formation_layer: Option<u32>,
    ) -> PyResult<Self> {
        let mut collection = Self::new(name)?;
        for centerline in centerlines {
            validate_centerline(&centerline, "centerline")?;
            collection.fibers.push(CollectionFiber {
                intrinsic: centerline.clone(),
                placed: centerline,
                material: material.clone(),
                tags: BTreeMap::new(),
                formation_layer,
            });
        }
        Ok(collection)
    }

    fn centerlines(&self) -> Vec<Vec<Vec3>> {
        self.fibers
            .iter()
            .map(|fiber| fiber.placed.clone())
            .collect()
    }

    fn rest_centerlines(&self) -> Vec<Vec<Vec3>> {
        self.fibers
            .iter()
            .map(|fiber| fiber.intrinsic.clone())
            .collect()
    }

    /// Appends clones of every fiber in another detached collection.
    fn extend(&mut self, other: PyRef<'_, PyFiberCollection>) {
        self.fibers.extend(other.fibers.iter().cloned());
    }

    /// Returns a new collection holding this collection's fibers followed by
    /// `other`'s. The result keeps this collection's name.
    fn __add__(&self, other: PyRef<'_, PyFiberCollection>) -> Self {
        let mut combined = self.clone();
        combined.fibers.extend(other.fibers.iter().cloned());
        combined
    }

    /// Returns the sorted `formation_layer` labels present in this collection.
    fn layer_ids(&self) -> Vec<u32> {
        let mut layers = self
            .fibers
            .iter()
            .filter_map(|fiber| fiber.formation_layer)
            .collect::<Vec<_>>();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    /// Copies the fibers carrying one deposition-layer label.
    #[pyo3(signature = (layer, *, name=None, allow_empty=false))]
    fn select_layer(&self, layer: u32, name: Option<String>, allow_empty: bool) -> PyResult<Self> {
        let fibers = self
            .fibers
            .iter()
            .filter(|fiber| fiber.formation_layer == Some(layer))
            .cloned()
            .collect::<Vec<_>>();
        if fibers.is_empty() && !allow_empty {
            return Err(PyValueError::new_err(format!(
                "collection contains no fibers in layer {layer}"
            )));
        }
        Ok(Self {
            name: name.unwrap_or_else(|| format!("{} layer {layer}", self.name)),
            fibers,
        })
    }

    fn __len__(&self) -> usize {
        self.fibers.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FiberCollection(name={:?}, fibers={})",
            self.name,
            self.fibers.len()
        )
    }
}

impl PyFiberCollection {
    pub(crate) fn from_assembly(name: String, assembly: &FiberAssembly) -> PyResult<Self> {
        let mut fibers = Vec::with_capacity(assembly.topology.fibers.len());
        for (index, fiber) in assembly.topology.fibers.iter().enumerate() {
            let start = fiber.vertices.start as usize;
            let end = start + fiber.vertices.len as usize;
            let material_name = assembly.materials.entries[fiber.material.0 as usize]
                .name
                .clone();
            let diameter = match assembly.sections.entries[fiber.section.0 as usize] {
                Section::Circular { radius } => 2.0 * radius,
                Section::Elliptical { .. } => {
                    return Err(PyValueError::new_err(
                        "Python FiberCollection currently requires circular generated sections",
                    ));
                }
            };
            fibers.push(CollectionFiber {
                placed: assembly.geometry.placed.positions[start..end].to_vec(),
                intrinsic: assembly.geometry.intrinsic.positions[start..end].to_vec(),
                material: PyMaterial {
                    name: material_name,
                    diameter,
                    min_bend_radius: assembly.admissibility.bend_limits[index]
                        .map(|limit| limit.minimum_bend_radius),
                },
                tags: BTreeMap::new(),
                formation_layer: fiber.formation_layer,
            });
        }
        Ok(Self { name, fibers })
    }
}

#[pyclass(name = "FiberSelection", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyFiberSelection {
    #[pyo3(get)]
    pub(crate) name: String,
    #[pyo3(get)]
    pub(crate) fiber_ids: Vec<u32>,
    #[pyo3(get)]
    pub(crate) formation_step: u32,
}

#[pymethods]
impl PyFiberSelection {
    fn __len__(&self) -> usize {
        self.fiber_ids.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FiberSelection(name={:?}, fibers={}, formation_step={})",
            self.name,
            self.fiber_ids.len(),
            self.formation_step
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AssemblyModel {
    pub(crate) assembly: FiberAssembly,
    pub(crate) stack_axis: usize,
    pub(crate) tags: Vec<BTreeMap<String, String>>,
    next_fiber_id: u32,
}

#[pyclass(name = "Assembly", module = "tangle._tangle")]
#[derive(Clone, Debug)]
pub(crate) struct PyAssembly {
    pub(crate) model: Arc<Mutex<AssemblyModel>>,
}

#[pymethods]
impl PyAssembly {
    #[new]
    fn new(cell: PyRef<'_, PyCell>) -> Self {
        Self {
            model: Arc::new(Mutex::new(AssemblyModel::new(&cell))),
        }
    }

    #[getter]
    fn fiber_count(&self) -> usize {
        self.model
            .lock()
            .expect("assembly lock poisoned")
            .assembly
            .topology
            .fibers
            .len()
    }

    #[getter]
    fn cell(&self) -> PyCell {
        let model = self.model.lock().expect("assembly lock poisoned");
        PyCell {
            inner: model.assembly.cell,
            stack_axis: model.stack_axis,
        }
    }

    fn centerlines(&self) -> Vec<Vec<Vec3>> {
        let model = self.model.lock().expect("assembly lock poisoned");
        model
            .assembly
            .topology
            .fibers
            .iter()
            .map(|fiber| {
                let start = fiber.vertices.start as usize;
                let end = start + fiber.vertices.len as usize;
                model.assembly.geometry.placed.positions[start..end].to_vec()
            })
            .collect()
    }

    /// Adds a collection's fibers directly, without a recipe or relaxation.
    ///
    /// Use this for imported geometry, such as centerlines tracked from a CT
    /// scan, that only needs characterization or export. Fibers added here are
    /// present from formation step 0.
    #[pyo3(signature = (collection, *, name=None, translation=[0.0, 0.0, 0.0], rotation=None))]
    fn insert(
        &self,
        collection: PyRef<'_, PyFiberCollection>,
        name: Option<String>,
        translation: Vec3,
        rotation: Option<[[f64; 3]; 3]>,
    ) -> PyResult<PyFiberSelection> {
        if collection.fibers.is_empty() {
            return Err(PyValueError::new_err(
                "cannot insert an empty fiber collection",
            ));
        }
        let name = name.unwrap_or_else(|| collection.name.clone());
        self.model.lock().expect("assembly lock poisoned").insert(
            &collection,
            name,
            0,
            translation,
            rotation.unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    }

    /// Runs TANGLE's exact centerline characterization without a relaxation step.
    fn characterize(&self) -> PyAnalysisReport {
        let model = self.model.lock().expect("assembly lock poisoned");
        PyAnalysisReport {
            inner: characterize_assembly(&model.assembly),
        }
    }

    /// Measures fiber-to-fiber contacts, neighbor persistence, and turnover.
    #[pyo3(signature = (contact_gap, *, neighbor_gap=None, in_axis_angle_degrees=20.0, sample_spacing=None, max_lag=None, lag_count=24))]
    fn characterize_neighbors(
        &self,
        contact_gap: f64,
        neighbor_gap: Option<f64>,
        in_axis_angle_degrees: f64,
        sample_spacing: Option<f64>,
        max_lag: Option<f64>,
        lag_count: usize,
    ) -> PyResult<PyNeighborReport> {
        let model = self.model.lock().expect("assembly lock poisoned");
        characterize_neighbors(
            &model.assembly,
            contact_gap,
            neighbor_gap,
            in_axis_angle_degrees,
            sample_spacing,
            max_lag,
            lag_count,
        )
    }

    /// Measures fiber curvature, torsion, tangent correlation, curl and the
    /// Schladitz orientation fit.
    #[pyo3(signature = (*, sample_spacing=None, max_lag=None, lag_count=24, quantile_count=101, orientation_axis=[0.0, 0.0, 1.0], min_torsion_curvature=None))]
    fn characterize_shape(
        &self,
        sample_spacing: Option<f64>,
        max_lag: Option<f64>,
        lag_count: usize,
        quantile_count: usize,
        orientation_axis: [f64; 3],
        min_torsion_curvature: Option<f64>,
    ) -> PyResult<PyShapeReport> {
        let model = self.model.lock().expect("assembly lock poisoned");
        characterize_shape(
            &model.assembly,
            sample_spacing,
            max_lag,
            lag_count,
            quantile_count,
            orientation_axis,
            min_torsion_curvature,
        )
    }

    /// Measures fiber writhe and the Gauss linking of contacting fibers.
    #[pyo3(signature = (contact_gap, *, sample_spacing=None, window=None, neighbor_sample_spacing=None, quantile_count=101))]
    fn characterize_entanglement(
        &self,
        contact_gap: f64,
        sample_spacing: Option<f64>,
        window: Option<f64>,
        neighbor_sample_spacing: Option<f64>,
        quantile_count: usize,
    ) -> PyResult<PyEntanglementReport> {
        let model = self.model.lock().expect("assembly lock poisoned");
        characterize_entanglement(
            &model.assembly,
            contact_gap,
            sample_spacing,
            window,
            neighbor_sample_spacing,
            quantile_count,
        )
    }

    /// Measures nearest-neighbor distances and the pair correlation g(r) of
    /// fiber cross-sections in planes normal to one cell axis.
    #[pyo3(signature = (*, axis=2, slice_count=16, max_radius=None, bin_count=40, quantile_count=101))]
    fn characterize_slices(
        &self,
        axis: usize,
        slice_count: usize,
        max_radius: Option<f64>,
        bin_count: usize,
        quantile_count: usize,
    ) -> PyResult<PySliceReport> {
        let model = self.model.lock().expect("assembly lock poisoned");
        characterize_slices(
            &model.assembly,
            axis,
            slice_count,
            max_radius,
            bin_count,
            quantile_count,
        )
    }

    /// Casts test lines through the fiber capsules and measures the exact
    /// solid fraction, solid and void chord lengths, two-point correlation
    /// and solid-fraction profile.
    #[pyo3(signature = (*, line_count=64, max_lag=None, lag_count=32, profile_axis=2, quantile_count=101))]
    fn characterize_phases(
        &self,
        line_count: usize,
        max_lag: Option<f64>,
        lag_count: usize,
        profile_axis: usize,
        quantile_count: usize,
    ) -> PyResult<PyPhaseReport> {
        let model = self.model.lock().expect("assembly lock poisoned");
        characterize_phases(
            &model.assembly,
            line_count,
            max_lag,
            lag_count,
            profile_axis,
            quantile_count,
        )
    }

    #[pyo3(signature = (output_directory, voxel_size, *, include_fiber_ids=true, include_interface=true, ambiguity_tolerance=None))]
    fn export_puma(
        &self,
        output_directory: PathBuf,
        voxel_size: f64,
        include_fiber_ids: bool,
        include_interface: bool,
        ambiguity_tolerance: Option<f64>,
    ) -> PyResult<PyPumaExportReport> {
        let mut config = PumaVoxelExportConfig::new(output_directory, voxel_size)
            .with_fiber_ids(include_fiber_ids)
            .with_interface(include_interface);
        if let Some(tolerance) = ambiguity_tolerance {
            config = config.with_ambiguity_tolerance(tolerance);
        }
        let model = self.model.lock().expect("assembly lock poisoned");
        let report = write_puma_bundle(&model.assembly, &config)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(PyPumaExportReport { inner: report })
    }

    fn __repr__(&self) -> String {
        let model = self.model.lock().expect("assembly lock poisoned");
        format!(
            "Assembly(fibers={}, cell={:?})",
            model.assembly.topology.fibers.len(),
            [
                model.assembly.cell.basis[0][0],
                model.assembly.cell.basis[1][1],
                model.assembly.cell.basis[2][2],
            ]
        )
    }
}

impl AssemblyModel {
    pub(crate) fn new(cell: &PyCell) -> Self {
        Self {
            assembly: FiberAssembly::new(cell.inner),
            stack_axis: cell.stack_axis,
            tags: Vec::new(),
            next_fiber_id: 1,
        }
    }

    /// Wraps a finished assembly, for example a recipe result. Every fiber is
    /// already formed, so all are moved to formation step 0; a recipe that
    /// continues from here then keeps them active from its first operation.
    pub(crate) fn from_assembly(mut assembly: FiberAssembly, stack_axis: usize) -> Self {
        for fiber in &mut assembly.topology.fibers {
            fiber.formation_step = 0;
        }
        let next_fiber_id = assembly
            .topology
            .fibers
            .iter()
            .map(|fiber| fiber.id.0)
            .max()
            .map_or(1, |id| id + 1);
        let tags = vec![BTreeMap::new(); assembly.topology.fibers.len()];
        Self {
            assembly,
            stack_axis,
            tags,
            next_fiber_id,
        }
    }

    pub(crate) fn insert(
        &mut self,
        collection: &PyFiberCollection,
        name: String,
        formation_step: u32,
        translation: Vec3,
        rotation: [[f64; 3]; 3],
    ) -> PyResult<PyFiberSelection> {
        if translation.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("translation must be finite"));
        }
        validate_rotation(rotation)?;
        let mut fiber_ids = Vec::with_capacity(collection.fibers.len());
        for input in &collection.fibers {
            let material = self.assembly.materials.add(input.material.name.clone());
            let section = self.assembly.sections.add(Section::Circular {
                radius: 0.5 * input.material.diameter,
            });
            let placed = input
                .placed
                .iter()
                .map(|point| transform_point(*point, rotation, translation))
                .collect::<Vec<_>>();
            let intrinsic = input
                .intrinsic
                .iter()
                .map(|point| transform_point(*point, rotation, translation))
                .collect::<Vec<_>>();
            let id = FiberId(self.next_fiber_id);
            self.next_fiber_id = self
                .next_fiber_id
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("fiber identifier overflow"))?;
            self.assembly
                .add_fiber(id, material, section, &intrinsic, &placed)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            self.assembly
                .set_fiber_formation_step(id, formation_step)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            self.assembly
                .set_fiber_formation_layer(id, input.formation_layer)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            if let Some(minimum_bend_radius) = input.material.min_bend_radius {
                self.assembly
                    .set_fiber_bend_limit(
                        id,
                        Some(FiberBendLimit {
                            minimum_bend_radius,
                        }),
                    )
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
            }
            self.tags.push(input.tags.clone());
            fiber_ids.push(id.0);
        }
        Ok(PyFiberSelection {
            name,
            fiber_ids,
            formation_step,
        })
    }
}

fn transform_point(point: Vec3, rotation: [[f64; 3]; 3], translation: Vec3) -> Vec3 {
    [
        rotation[0][0] * point[0]
            + rotation[0][1] * point[1]
            + rotation[0][2] * point[2]
            + translation[0],
        rotation[1][0] * point[0]
            + rotation[1][1] * point[1]
            + rotation[1][2] * point[2]
            + translation[1],
        rotation[2][0] * point[0]
            + rotation[2][1] * point[1]
            + rotation[2][2] * point[2]
            + translation[2],
    ]
}

fn validate_rotation(rotation: [[f64; 3]; 3]) -> PyResult<()> {
    if rotation.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("rotation matrix must be finite"));
    }
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for row in rotation {
        if (dot(row, row) - 1.0).abs() > 1.0e-6 {
            return Err(PyValueError::new_err(
                "rotation matrix rows must have unit length",
            ));
        }
    }
    for (first, second) in [(0, 1), (0, 2), (1, 2)] {
        if dot(rotation[first], rotation[second]).abs() > 1.0e-6 {
            return Err(PyValueError::new_err(
                "rotation matrix rows must be orthogonal",
            ));
        }
    }
    let determinant = rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    if (determinant - 1.0).abs() > 1.0e-6 {
        return Err(PyValueError::new_err(
            "rotation matrix must be right-handed with determinant +1",
        ));
    }
    Ok(())
}

fn validate_centerline(centerline: &[Vec3], label: &str) -> PyResult<()> {
    if centerline.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "{label} must contain at least two vertices"
        )));
    }
    if centerline
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "{label} coordinates must be finite"
        )));
    }
    if centerline.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PyValueError::new_err(format!(
            "{label} contains a zero-length segment"
        )));
    }
    Ok(())
}
