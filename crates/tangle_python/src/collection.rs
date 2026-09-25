use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use tangle_characterize::characterize_assembly;
use tangle_core::{
    default_directors, orthonormalize_directors, polyline_tangents, FiberAssembly, FiberBendLimit,
    FiberId, PeriodicCell, Section, Vec3,
};
use tangle_export::{write_puma_bundle, PumaVoxelExportConfig};

use crate::analysis::{
    characterize_neighbors, characterize_shape, PyAnalysisReport, PyNeighborReport,
    PyPumaExportReport, PyShapeReport,
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
    /// Short-axis width of an oval section; `None` for a round fiber, whose
    /// `diameter` is then its only width.
    #[pyo3(get)]
    pub(crate) thickness: Option<f64>,
}

impl PyMaterial {
    /// The cross-section this material gives each fiber: a circle, or an
    /// ellipse with `diameter` along the long axis and `thickness` across it.
    pub(crate) fn section(&self) -> Section {
        match self.thickness {
            Some(thickness) if thickness < self.diameter => Section::Elliptical {
                semi_axes: [0.5 * self.diameter, 0.5 * thickness],
            },
            _ => Section::Circular {
                radius: 0.5 * self.diameter,
            },
        }
    }

    pub(crate) fn from_section(
        name: String,
        section: Section,
        min_bend_radius: Option<f64>,
    ) -> Self {
        let (diameter, thickness) = match section {
            Section::Circular { radius } => (2.0 * radius, None),
            Section::Elliptical { semi_axes } => {
                let long = semi_axes[0].max(semi_axes[1]);
                let short = semi_axes[0].min(semi_axes[1]);
                (2.0 * long, (short < long).then_some(2.0 * short))
            }
        };
        Self {
            name,
            diameter,
            min_bend_radius,
            thickness,
        }
    }

    /// Generators that only build round fibers call this first.
    pub(crate) fn require_round(&self, generator: &str) -> PyResult<()> {
        if self.thickness.is_some() {
            return Err(PyValueError::new_err(format!(
                "{generator} does not support oval materials yet; build oval fibers with FiberCollection.add_fiber"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyMaterial {
    #[new]
    #[pyo3(signature = (name, diameter, min_bend_radius=None, *, thickness=None))]
    fn new(
        name: String,
        diameter: f64,
        min_bend_radius: Option<f64>,
        thickness: Option<f64>,
    ) -> PyResult<Self> {
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
        if thickness.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > diameter) {
            return Err(PyValueError::new_err(
                "thickness must be positive, finite and no larger than diameter",
            ));
        }
        Ok(Self {
            name,
            diameter,
            min_bend_radius,
            // An oval as thick as it is wide is round.
            thickness: thickness.filter(|value| *value < diameter),
        })
    }

    #[getter]
    fn radius(&self) -> f64 {
        0.5 * self.diameter
    }

    #[getter]
    fn is_oval(&self) -> bool {
        self.thickness.is_some()
    }

    fn __repr__(&self) -> String {
        let mut text = format!("Material(name={:?}, diameter={}", self.name, self.diameter);
        if let Some(radius) = self.min_bend_radius {
            text.push_str(&format!(", min_bend_radius={radius}"));
        }
        if let Some(thickness) = self.thickness {
            text.push_str(&format!(", thickness={thickness}"));
        }
        text.push(')');
        text
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CollectionFiber {
    pub(crate) placed: Vec<Vec3>,
    pub(crate) intrinsic: Vec<Vec3>,
    pub(crate) material: PyMaterial,
    /// Placed long-axis directions per vertex; `None` uses the defaults.
    pub(crate) long_axes: Option<Vec<Vec3>>,
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

    #[pyo3(signature = (centerline, material, *, rest_centerline=None, tags=None, formation_layer=None, long_axis=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_fiber(
        &mut self,
        centerline: Vec<Vec3>,
        material: PyRef<'_, PyMaterial>,
        rest_centerline: Option<Vec<Vec3>>,
        tags: Option<HashMap<String, String>>,
        formation_layer: Option<u32>,
        long_axis: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        validate_centerline(&centerline, "centerline")?;
        let intrinsic = rest_centerline.unwrap_or_else(|| centerline.clone());
        validate_centerline(&intrinsic, "rest_centerline")?;
        if intrinsic.len() != centerline.len() {
            return Err(PyValueError::new_err(
                "centerline and rest_centerline must contain the same number of vertices",
            ));
        }
        let long_axes = long_axis
            .map(|value| parse_long_axes(value, centerline.len()))
            .transpose()?;
        self.fibers.push(CollectionFiber {
            placed: centerline,
            intrinsic,
            material: material.clone(),
            long_axes,
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
                long_axes: None,
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

    /// Unit long-axis direction at every vertex of every fiber, perpendicular
    /// to the centerline. Round fibers report the default direction too.
    fn long_axes(&self) -> Vec<Vec<Vec3>> {
        self.fibers.iter().map(CollectionFiber::long_axes).collect()
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
            let section = assembly.sections.entries[fiber.section.0 as usize];
            let placed = &assembly.geometry.placed.positions[start..end];
            let long_axes = (!section.is_circular())
                .then(|| long_axis_directors(section, placed, assembly.fiber_directors(index)));
            fibers.push(CollectionFiber {
                placed: assembly.geometry.placed.positions[start..end].to_vec(),
                intrinsic: assembly.geometry.intrinsic.positions[start..end].to_vec(),
                material: PyMaterial::from_section(
                    material_name,
                    section,
                    assembly.admissibility.bend_limits[index]
                        .map(|limit| limit.minimum_bend_radius),
                ),
                long_axes,
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

    /// Unit long-axis direction at every vertex of every fiber, along the
    /// section's longer width. Round fibers report the default direction.
    fn long_axes(&self) -> Vec<Vec<Vec3>> {
        let model = self.model.lock().expect("assembly lock poisoned");
        let assembly = &model.assembly;
        assembly
            .topology
            .fibers
            .iter()
            .enumerate()
            .map(|(index, fiber)| {
                let start = fiber.vertices.start as usize;
                let end = start + fiber.vertices.len as usize;
                long_axis_directors(
                    assembly.sections.entries[fiber.section.0 as usize],
                    &assembly.geometry.placed.positions[start..end],
                    assembly.fiber_directors(index),
                )
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
            let section = self.assembly.sections.add(input.material.section());
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
            if let Some(long_axes) = &input.long_axes {
                let rotated = long_axes
                    .iter()
                    .map(|axis| transform_point(*axis, rotation, [0.0; 3]))
                    .collect::<Vec<_>>();
                self.assembly
                    .set_fiber_directors(id, &rotated)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
            }
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

impl CollectionFiber {
    fn long_axes(&self) -> Vec<Vec3> {
        match &self.long_axes {
            Some(axes) => {
                let mut axes = axes.clone();
                orthonormalize_directors(&self.placed, &mut axes);
                axes
            }
            None => default_directors(&self.placed),
        }
    }
}

/// A collection fiber's long axis is always the ellipse's longer semi-axis.
/// When an assembly stores the shorter one first, its directors point across
/// the fiber, so they are turned a quarter turn about the tangent.
fn long_axis_directors(section: Section, placed: &[Vec3], directors: Vec<Vec3>) -> Vec<Vec3> {
    match section {
        Section::Elliptical { semi_axes } if semi_axes[0] < semi_axes[1] => {
            polyline_tangents(placed)
                .into_iter()
                .zip(directors)
                .map(|(t, d)| {
                    [
                        t[1] * d[2] - t[2] * d[1],
                        t[2] * d[0] - t[0] * d[2],
                        t[0] * d[1] - t[1] * d[0],
                    ]
                })
                .collect()
        }
        _ => directors,
    }
}

/// Reads `long_axis`: one vector for the whole fiber or one per vertex.
fn parse_long_axes(value: &Bound<'_, PyAny>, vertex_count: usize) -> PyResult<Vec<Vec3>> {
    let axes = if let Ok(single) = value.extract::<Vec3>() {
        vec![single; vertex_count]
    } else {
        value.extract::<Vec<Vec3>>().map_err(|_| {
            PyValueError::new_err("long_axis must be one 3-vector or one 3-vector per vertex")
        })?
    };
    if axes.len() != vertex_count {
        return Err(PyValueError::new_err(format!(
            "long_axis has {} vectors for {vertex_count} vertices",
            axes.len()
        )));
    }
    if axes.iter().any(|axis| {
        axis.iter().any(|value| !value.is_finite()) || axis.iter().all(|value| *value == 0.0)
    }) {
        return Err(PyValueError::new_err(
            "long_axis vectors must be finite and nonzero",
        ));
    }
    Ok(axes)
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
