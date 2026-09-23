use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use grass_app::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};
use tangle_app::{TanglePreparedAssemblyPlugin, TangleStage, TangleWorkflowPlugin};
use tangle_characterize::characterize_assembly;
use tangle_checkpoint::{CheckpointConfig, CheckpointPlugin, CheckpointReport};
use tangle_core::{FiberAssembly, Vec3};
use tangle_export::{
    build_bpm_model, write_bpm_lammps_data, write_ovito_assembly_frame, write_ovito_view_script,
    write_puma_bundle, BpmExportConfig, BpmExportMode, OvitoColoring, OvitoRepresentation,
    OvitoTrajectoryConfig, OvitoTrajectoryPlugin, OvitoTrajectoryReport, PumaVoxelExportConfig,
};
use tangle_generate::{
    random_footprint_center, FormationOperation, FormationRecipeConfig, FormationRecipePlugin,
    FormationRecipeState, NeedlingConfig, NeedlingSelection,
};
use tangle_relax::{RelaxationPlugin, RelaxationState};

use crate::analysis::{PyAnalysisReport, PyPumaExportReport};
use crate::checkpoint::PyCheckpointSettings;
use crate::collection::{
    cell_lengths, AssemblyModel, PyAssembly, PyCell, PyFiberCollection, PyFiberSelection,
    PyMaterial,
};
use crate::common::{
    axis_name, nonnegative_finite, parse_axis, parse_axis_mask, parse_choice, positive_finite,
    unit_fraction,
};
use crate::compaction::PyCompactionSettings;
use crate::junctions::PyJunctionPolicy;
use crate::settings::{PyRelaxationOverrides, PyRelaxationSettings, PySolvePolicy};

pyo3::create_exception!(
    tangle,
    RecipeError,
    PyRuntimeError,
    "A recipe operation failed while Recipe.run() executed it."
);

const OVITO_COLORINGS: &[(&str, OvitoColoring)] = &[
    ("fiber", OvitoColoring::Fiber),
    ("curvature_ratio", OvitoColoring::CurvatureRatio),
    ("refinement_level", OvitoColoring::RefinementLevel),
];
const BPM_MODES: &[(&str, BpmExportMode)] = &[
    ("spheres_exact", BpmExportMode::SpheresExact),
    ("spheres_dynamic", BpmExportMode::SpheresDynamic),
    ("spherocylinders_exact", BpmExportMode::SpherocylindersExact),
    (
        "spherocylinders_constant",
        BpmExportMode::SpherocylindersConstant,
    ),
];

// --- Needle footprints -------------------------------------------------------

/// Pull fibers whose centerline crosses a circle in the layer plane.
///
/// `CircularFootprint(center, diameter=...)` places the circle explicitly;
/// `CircularFootprint.random(diameter=..., seed=...)` picks a reproducible
/// center per layer, uniform over the cell footprint.
#[pyclass(name = "CircularFootprint", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyCircularFootprint {
    #[pyo3(get)]
    center: Option<[f64; 2]>,
    #[pyo3(get)]
    diameter: f64,
    #[pyo3(get)]
    seed: Option<u64>,
}

#[pymethods]
impl PyCircularFootprint {
    #[new]
    #[pyo3(signature = (center, *, diameter))]
    fn new(center: [f64; 2], diameter: f64) -> PyResult<Self> {
        if center.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("footprint center must be finite"));
        }
        positive_finite(diameter, "diameter")?;
        Ok(Self {
            center: Some(center),
            diameter,
            seed: None,
        })
    }

    #[classmethod]
    #[pyo3(signature = (*, diameter, seed))]
    fn random(_class: &Bound<'_, PyType>, diameter: f64, seed: u64) -> PyResult<Self> {
        positive_finite(diameter, "diameter")?;
        Ok(Self {
            center: None,
            diameter,
            seed: Some(seed),
        })
    }

    fn __repr__(&self) -> String {
        match (self.center, self.seed) {
            (Some(center), _) => {
                format!("CircularFootprint({center:?}, diameter={})", self.diameter)
            }
            (None, seed) => format!(
                "CircularFootprint.random(diameter={}, seed={})",
                self.diameter,
                seed.unwrap_or_default()
            ),
        }
    }
}

/// Pull a seeded random `fraction` of the layer's fibers.
#[pyclass(name = "RandomFiberFraction", module = "tangle._tangle", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyRandomFiberFraction {
    #[pyo3(get)]
    fraction: f64,
    #[pyo3(get)]
    seed: u64,
}

#[pymethods]
impl PyRandomFiberFraction {
    #[new]
    #[pyo3(signature = (fraction, *, seed=0))]
    fn new(fraction: f64, seed: u64) -> PyResult<Self> {
        unit_fraction(fraction, "fraction")?;
        Ok(Self { fraction, seed })
    }

    fn __repr__(&self) -> String {
        format!("RandomFiberFraction({}, seed={})", self.fraction, self.seed)
    }
}

// --- Held targets ------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum HeldKind {
    LayerPlacement,
    Needles,
}

/// Returned by operations that hold fibers on targets. Use it as a context
/// manager to release the targets when the block ends, or ignore it and call
/// the matching `release_*` method yourself.
#[pyclass(name = "HeldTargets", module = "tangle._tangle")]
pub(crate) struct PyHeldTargets {
    recipe: Py<PyRecipe>,
    kind: HeldKind,
}

#[pymethods]
impl PyHeldTargets {
    fn __enter__(&self, py: Python<'_>) -> Py<PyRecipe> {
        self.recipe.clone_ref(py)
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        let mut recipe = self.recipe.borrow_mut(py);
        match self.kind {
            HeldKind::LayerPlacement => recipe.release_layer_placement(),
            HeldKind::Needles => recipe.release_needles(),
        }
        false
    }

    fn __repr__(&self) -> String {
        match self.kind {
            HeldKind::LayerPlacement => "HeldTargets(layer placement)".to_string(),
            HeldKind::Needles => "HeldTargets(needles)".to_string(),
        }
    }
}

// --- Recipe ------------------------------------------------------------------

/// An ordered list of formation operations, executed by `run()`.
///
/// `insert()` adds fibers to the starting assembly immediately; every other
/// operation is recorded and runs later, in order, inside `run()`. `run()`
/// leaves the starting assembly unchanged and returns the result, so running
/// the same recipe twice starts from the same fibers both times.
#[pyclass(name = "Recipe", module = "tangle._tangle")]
pub(crate) struct PyRecipe {
    model: Arc<Mutex<AssemblyModel>>,
    operations: Vec<FormationOperation>,
    operation_descriptions: Vec<String>,
    next_formation_step: u32,
    stack_axis: usize,
}

#[pymethods]
impl PyRecipe {
    #[new]
    #[pyo3(signature = (cell, *, stack_axis=None))]
    fn new(cell: &Bound<'_, PyAny>, stack_axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let model = if let Ok(cell) = cell.extract::<PyRef<'_, PyCell>>() {
            Arc::new(Mutex::new(AssemblyModel::new(&cell)))
        } else if let Ok(assembly) = cell.extract::<PyRef<'_, PyAssembly>>() {
            assembly.model.clone()
        } else {
            return Err(PyTypeError::new_err(
                "Recipe expects a Cell or Assembly as its first argument",
            ));
        };
        let stack_axis = match stack_axis {
            Some(value) => parse_axis(value, "stack_axis")?,
            None => model.lock().expect("assembly lock poisoned").stack_axis,
        };
        Ok(Self {
            model,
            operations: Vec::new(),
            operation_descriptions: Vec::new(),
            next_formation_step: 0,
            stack_axis,
        })
    }

    /// The axis plies stack along (0, 1, or 2). Defaults to the cell's.
    #[getter]
    fn stack_axis(&self) -> usize {
        self.stack_axis
    }

    #[setter]
    fn set_stack_axis(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.stack_axis = parse_axis(value, "stack_axis")?;
        Ok(())
    }

    #[pyo3(signature = (collection, *, name=None, translation=[0.0, 0.0, 0.0], rotation=None))]
    fn insert(
        &mut self,
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
        let step = self.next_formation_step;
        self.next_formation_step = self
            .next_formation_step
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("formation step overflow"))?;
        let selection_name = name.unwrap_or_else(|| collection.name.clone());
        let selection = self.model.lock().expect("assembly lock poisoned").insert(
            &collection,
            selection_name.clone(),
            step,
            translation,
            rotation.unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )?;
        self.push(
            FormationOperation::ActivateFibersThrough(step),
            format!(
                "insert {:?} ({} fibers) as formation step {}",
                selection_name,
                selection.fiber_ids.len(),
                step
            ),
        );
        Ok(selection)
    }

    /// Relaxes until the run's `RelaxationSettings` tolerances are met.
    #[pyo3(signature = (*, max_iterations=2000))]
    fn relax_until_converged(&mut self, max_iterations: usize) -> PyResult<()> {
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        self.push(
            FormationOperation::RelaxUntilConverged {
                maximum_iterations: max_iterations,
            },
            format!("relax until converged (up to {max_iterations} iterations)"),
        );
        Ok(())
    }

    /// Relaxes for exactly `iterations` iterations.
    fn relax_for(&mut self, iterations: usize) -> PyResult<()> {
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be positive"));
        }
        self.push(
            FormationOperation::RelaxFor(iterations),
            format!("relax for exactly {iterations} iterations"),
        );
        Ok(())
    }

    /// Relaxes until held layers and needles are within `tolerance` of their
    /// targets.
    #[pyo3(signature = (*, tolerance, max_iterations))]
    fn settle_targets(&mut self, tolerance: f64, max_iterations: usize) -> PyResult<()> {
        nonnegative_finite(tolerance, "tolerance")?;
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        self.push(
            FormationOperation::RelaxUntilTargetsReached {
                tolerance: tolerance as f32,
                maximum_iterations: max_iterations,
            },
            format!("settle held targets to tolerance {tolerance}"),
        );
        Ok(())
    }

    /// Relaxes under a `SolvePolicy`, optionally with temporary overrides.
    #[pyo3(signature = (policy, overrides=None))]
    fn solve(
        &mut self,
        policy: PyRef<'_, PySolvePolicy>,
        overrides: Option<PyRef<'_, PyRelaxationOverrides>>,
    ) -> PyResult<()> {
        let policy = policy.to_rust()?;
        let description = format!("solve {:?}", policy.name);
        let operation = if let Some(overrides) = overrides {
            FormationOperation::RelaxWithOverrides {
                policy,
                overrides: overrides.to_rust()?,
            }
        } else {
            FormationOperation::RelaxWithPolicy(policy)
        };
        self.push(operation, description);
        Ok(())
    }

    /// Sets the minimum bend radius of every fiber made of `material`.
    fn set_min_bend_radius(
        &mut self,
        material: &Bound<'_, PyAny>,
        min_bend_radius: f64,
    ) -> PyResult<()> {
        let material_name = if let Ok(material) = material.extract::<PyRef<'_, PyMaterial>>() {
            material.name.clone()
        } else if let Ok(name) = material.extract::<String>() {
            name
        } else {
            return Err(PyTypeError::new_err(
                "material must be a Material or a material name",
            ));
        };
        if material_name.trim().is_empty() {
            return Err(PyValueError::new_err("material name must not be empty"));
        }
        positive_finite(min_bend_radius, "min_bend_radius")?;
        // A recipe that resumes from a checkpoint has no fibers yet, so the
        // name can only be checked here when fibers were inserted.
        {
            let model = self.model.lock().expect("assembly lock poisoned");
            let materials = &model.assembly.materials.entries;
            if !materials.is_empty() && !materials.iter().any(|entry| entry.name == material_name) {
                let known = materials
                    .iter()
                    .map(|entry| format!("{:?}", entry.name))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(PyValueError::new_err(format!(
                    "unknown material {material_name:?}; this recipe has {known}"
                )));
            }
        }
        self.push(
            FormationOperation::SetMaterialBendRadius {
                material_name: material_name.clone(),
                minimum_bend_radius: min_bend_radius,
            },
            format!("set {material_name:?} minimum bend radius to {min_bend_radius}"),
        );
        Ok(())
    }

    /// Moves every layer toward `factor` times its initial spacing from the
    /// cell center. Returns a context manager that releases the layers.
    #[pyo3(signature = (factor, *, stiffness=0.25, max_translation=1.0e-5))]
    fn scale_layer_spacing(
        slf: &Bound<'_, Self>,
        factor: f64,
        stiffness: f64,
        max_translation: f64,
    ) -> PyResult<PyHeldTargets> {
        positive_finite(factor, "factor")?;
        unit_fraction(stiffness, "stiffness")?;
        positive_finite(max_translation, "max_translation")?;
        slf.borrow_mut().push(
            FormationOperation::MoveLayers {
                spacing_scale: factor as f32,
                stiffness: stiffness as f32,
                max_translation: max_translation as f32,
            },
            format!("scale layer spacing by {factor}"),
        );
        Ok(held(slf, HeldKind::LayerPlacement))
    }

    /// Holds the fibers tagged `formation_layer=layer` at `gap` above layer
    /// `layer - 1`. Returns a context manager that releases the placement.
    #[pyo3(signature = (layer, *, gap, stiffness=0.25, max_translation=1.0e-5))]
    fn place_layer_above(
        slf: &Bound<'_, Self>,
        layer: u32,
        gap: f64,
        stiffness: f64,
        max_translation: f64,
    ) -> PyResult<PyHeldTargets> {
        if layer == 0 {
            return Err(PyValueError::new_err("layer must be greater than zero"));
        }
        positive_finite(gap, "gap")?;
        unit_fraction(stiffness, "stiffness")?;
        positive_finite(max_translation, "max_translation")?;
        slf.borrow_mut().push(
            FormationOperation::PlaceLayerAbove {
                layer,
                gap: gap as f32,
                stiffness: stiffness as f32,
                max_translation: max_translation as f32,
            },
            format!("place layer {layer} above layer {}", layer - 1),
        );
        Ok(held(slf, HeldKind::LayerPlacement))
    }

    /// Releases every layer held by `place_layer_above` or `scale_layer_spacing`.
    fn release_layer_placement(&mut self) {
        self.push(
            FormationOperation::ReleaseLayerTargets,
            "release layer placement".to_string(),
        );
    }

    /// Pulls fibers tagged `formation_layer=layer` through the stack by
    /// `depth`. Returns a context manager that releases the needles.
    #[pyo3(signature = (
        layer,
        *,
        footprint,
        depth,
        min_fiber_diameter=None,
        stiffness=0.25,
        max_translation=1.0e-5,
        max_translation_over_diameter=0.25
    ))]
    #[allow(clippy::too_many_arguments)]
    fn needle_layer(
        slf: &Bound<'_, Self>,
        layer: u32,
        footprint: &Bound<'_, PyAny>,
        depth: f64,
        min_fiber_diameter: Option<f64>,
        stiffness: f64,
        max_translation: f64,
        max_translation_over_diameter: f64,
    ) -> PyResult<PyHeldTargets> {
        positive_finite(depth, "depth")?;
        if let Some(value) = min_fiber_diameter {
            positive_finite(value, "min_fiber_diameter")?;
        }
        unit_fraction(stiffness, "stiffness")?;
        positive_finite(max_translation, "max_translation")?;
        positive_finite(
            max_translation_over_diameter,
            "max_translation_over_diameter",
        )?;
        let (selection, description) = {
            let recipe = slf.borrow();
            recipe.needle_selection(layer, footprint)?
        };
        slf.borrow_mut().push(
            FormationOperation::NeedleLayer(NeedlingConfig {
                layer,
                selection,
                minimum_fiber_diameter: min_fiber_diameter.map(|value| value as f32),
                depth: depth as f32,
                stiffness: stiffness as f32,
                max_translation: max_translation as f32,
                maximum_translation_over_fiber_diameter: max_translation_over_diameter as f32,
            }),
            format!("needle layer {layer} through {description}"),
        );
        Ok(held(slf, HeldKind::Needles))
    }

    /// Releases every needle held by `needle_layer`.
    fn release_needles(&mut self) {
        self.push(
            FormationOperation::ReleaseNeedles,
            "release needles".to_string(),
        );
    }

    /// Shrinks or grows the cell along `axes` (default: the stack axis) to
    /// fit the active fibers plus `padding`.
    #[pyo3(signature = (*, axes=None, padding=0.0))]
    fn fit_cell_to_active_fibers(
        &mut self,
        axes: Option<&Bound<'_, PyAny>>,
        padding: f64,
    ) -> PyResult<()> {
        let axes = match axes {
            Some(value) => parse_axis_mask(value, "axes")?,
            None => {
                let mut mask = [false; 3];
                mask[self.stack_axis] = true;
                mask
            }
        };
        if !axes.iter().any(|selected| *selected) {
            return Err(PyValueError::new_err(
                "at least one cell-fit axis must be selected",
            ));
        }
        nonnegative_finite(padding, "padding")?;
        self.push(
            FormationOperation::FitCellToActiveFibers {
                axes,
                padding: padding as f32,
            },
            format!("fit cell axes {} to active fibers", axes_label(axes)),
        );
        Ok(())
    }

    #[pyo3(signature = (settings, overrides=None))]
    fn compact(
        &mut self,
        settings: PyRef<'_, PyCompactionSettings>,
        overrides: Option<PyRef<'_, PyRelaxationOverrides>>,
    ) -> PyResult<()> {
        let config = settings.to_rust(self.stack_axis)?;
        let description = format!("compact toward {}", settings.target_description());
        let operation = if let Some(overrides) = overrides {
            FormationOperation::CompactWithOverrides {
                config,
                overrides: overrides.to_rust()?,
            }
        } else {
            FormationOperation::Compact(config)
        };
        self.push(operation, description);
        Ok(())
    }

    fn capture_junctions(&mut self, policy: PyRef<'_, PyJunctionPolicy>) -> PyResult<()> {
        let policy = policy.to_rust()?;
        let description = format!("capture junctions using policy {:?}", policy.name);
        self.push(FormationOperation::CaptureJunctions(policy), description);
        Ok(())
    }

    /// Relaxes for `iterations`, capturing junctions every `capture_every`.
    #[pyo3(signature = (*, iterations, capture_every, policy))]
    fn relax_and_capture(
        &mut self,
        iterations: usize,
        capture_every: usize,
        policy: PyRef<'_, PyJunctionPolicy>,
    ) -> PyResult<()> {
        if iterations == 0 || capture_every == 0 || capture_every > iterations {
            return Err(PyValueError::new_err(
                "iterations must be positive and capture_every must lie within 1..=iterations",
            ));
        }
        let policy = policy.to_rust()?;
        self.push(
            FormationOperation::RelaxAndCapture {
                iterations,
                every: capture_every,
                policy,
            },
            format!(
                "relax for {iterations} iterations and capture junctions every {capture_every}"
            ),
        );
        Ok(())
    }

    fn operations(&self) -> Vec<String> {
        self.operation_descriptions.clone()
    }

    fn centerlines(&self) -> Vec<Vec<Vec3>> {
        assembly_centerlines(&self.model.lock().expect("assembly lock poisoned").assembly)
    }

    #[pyo3(signature = (
        settings=None,
        *,
        checkpoint=None,
        debug_ovito_path=None,
        debug_ovito_view_script_path=None,
        debug_ovito_session_path=None,
        debug_ovito_coloring="curvature_ratio"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        settings: Option<PyRef<'_, PyRelaxationSettings>>,
        checkpoint: Option<PyRef<'_, PyCheckpointSettings>>,
        debug_ovito_path: Option<PathBuf>,
        debug_ovito_view_script_path: Option<PathBuf>,
        debug_ovito_session_path: Option<PathBuf>,
        debug_ovito_coloring: &str,
    ) -> PyResult<PyRunResult> {
        let checkpoint = checkpoint
            .map(|checkpoint| checkpoint.to_rust())
            .transpose()?;
        let model = self.model.lock().expect("assembly lock poisoned");
        if model.assembly.topology.fibers.is_empty()
            && !checkpoint.as_ref().is_some_and(|config| config.resume)
        {
            return Err(PyValueError::new_err("recipe contains no fibers"));
        }
        if self.operations.is_empty() {
            return Err(PyValueError::new_err("recipe contains no operations"));
        }
        if !model.assembly.topology.fibers.is_empty() {
            model
                .assembly
                .validate()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        let mut settings = settings
            .map(|settings| settings.clone())
            .unwrap_or_default();
        let debug_ovito = if let Some(path) = debug_ovito_path {
            // With no explicit interval, keep only recipe/formation milestone
            // snapshots. An integer interval opts into detailed relaxation
            // frames in addition to those keyframes.
            let interval = settings.debug_snapshot_interval.unwrap_or(usize::MAX);
            settings.debug_snapshot_interval = Some(interval);
            let mut config = OvitoTrajectoryConfig::fiber_segments(path, interval)
                .with_initial_frame(false)
                .with_coloring(parse_choice(
                    debug_ovito_coloring,
                    "debug_ovito_coloring",
                    OVITO_COLORINGS,
                )?);
            config.view_script_path = debug_ovito_view_script_path;
            config.session_path = debug_ovito_session_path;
            Some(config)
        } else {
            if debug_ovito_view_script_path.is_some() || debug_ovito_session_path.is_some() {
                return Err(PyValueError::new_err(
                    "debug OVITO viewing paths require debug_ovito_path",
                ));
            }
            None
        };
        let relaxation = settings.to_rust()?;
        let assembly = model.assembly.clone();
        drop(model);
        let recipe = FormationRecipeConfig {
            layer_axis: self.stack_axis,
            operations: self.operations.clone(),
        };
        let stack_axis = self.stack_axis;
        let result = py.detach(move || {
            run_native_recipe(
                assembly,
                recipe,
                relaxation,
                checkpoint,
                debug_ovito,
                stack_axis,
            )
        });
        match result {
            Ok(result) => Ok(result),
            Err(RunFailure::Operation {
                index,
                iteration,
                reason,
            }) => Err(recipe_error(
                py,
                index,
                self.operation_descriptions
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("operation {index}")),
                iteration,
                reason,
            )),
            Err(RunFailure::Other(message)) => Err(PyRuntimeError::new_err(message)),
        }
    }

    fn __repr__(&self) -> String {
        let model = self.model.lock().expect("assembly lock poisoned");
        format!(
            "Recipe(fibers={}, operations={}, stack_axis={:?})",
            model.assembly.topology.fibers.len(),
            self.operations.len(),
            axis_name(self.stack_axis)
        )
    }
}

impl PyRecipe {
    fn push(&mut self, operation: FormationOperation, description: String) {
        self.operations.push(operation);
        self.operation_descriptions.push(description);
    }

    fn needle_selection(
        &self,
        layer: u32,
        footprint: &Bound<'_, PyAny>,
    ) -> PyResult<(NeedlingSelection, String)> {
        if let Ok(fraction) = footprint.extract::<PyRef<'_, PyRandomFiberFraction>>() {
            return Ok((
                NeedlingSelection::RandomFiberFraction {
                    fraction: fraction.fraction as f32,
                    seed: fraction.seed,
                },
                format!("a random {} of its fibers", fraction.fraction),
            ));
        }
        let circle = footprint
            .extract::<PyRef<'_, PyCircularFootprint>>()
            .map_err(|_| {
                PyTypeError::new_err("footprint must be a CircularFootprint or RandomFiberFraction")
            })?;
        let center = match (circle.center, circle.seed) {
            (Some(center), _) => center.map(|value| value as f32),
            (None, seed) => {
                let model = self.model.lock().expect("assembly lock poisoned");
                let lengths = cell_lengths(&model.assembly.cell);
                let origin = model.assembly.cell.origin;
                let plane = (0..3)
                    .filter(|axis| *axis != self.stack_axis)
                    .collect::<Vec<_>>();
                random_footprint_center(
                    seed.unwrap_or_default(),
                    layer,
                    [origin[plane[0]] as f32, origin[plane[1]] as f32],
                    [lengths[plane[0]] as f32, lengths[plane[1]] as f32],
                )
            }
        };
        Ok((
            NeedlingSelection::CircularFootprint {
                center,
                diameter: circle.diameter as f32,
            },
            format!(
                "a {:.3e} circle at [{:.3e}, {:.3e}]",
                circle.diameter, center[0], center[1]
            ),
        ))
    }
}

fn held(recipe: &Bound<'_, PyRecipe>, kind: HeldKind) -> PyHeldTargets {
    PyHeldTargets {
        recipe: recipe.clone().unbind(),
        kind,
    }
}

fn axes_label(axes: [bool; 3]) -> String {
    (0..3)
        .filter(|axis| axes[*axis])
        .map(axis_name)
        .collect::<String>()
}

fn recipe_error(
    py: Python<'_>,
    index: usize,
    operation: String,
    iteration: usize,
    reason: String,
) -> PyErr {
    let error = RecipeError::new_err(format!(
        "recipe operation {index} ({operation}) failed at iteration {iteration}: {reason}"
    ));
    let value = error.value(py);
    for (name, item) in [
        (
            "operation_index",
            index.into_pyobject(py).map(|v| v.into_any()),
        ),
        (
            "operation",
            operation.into_pyobject(py).map(|v| v.into_any()),
        ),
        (
            "iteration",
            iteration.into_pyobject(py).map(|v| v.into_any()),
        ),
        ("reason", reason.into_pyobject(py).map(|v| v.into_any())),
    ] {
        let Ok(item) = item;
        let _ = value.setattr(name, item);
    }
    error
}

// --- Run result --------------------------------------------------------------

#[pyclass(name = "RunResult", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyRunResult {
    assembly: FiberAssembly,
    stack_axis: usize,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub max_penetration: f32,
    #[pyo3(get)]
    pub max_curvature_ratio: f32,
    #[pyo3(get)]
    pub active_segments: usize,
    #[pyo3(get)]
    pub active_vertices: usize,
    #[pyo3(get)]
    pub segment_splits: usize,
    #[pyo3(get)]
    pub segment_merges: usize,
    #[pyo3(get)]
    pub refinement_passes: usize,
    #[pyo3(get)]
    pub coarsening_passes: usize,
    #[pyo3(get)]
    pub uploaded_bytes: usize,
    #[pyo3(get)]
    pub downloaded_bytes: usize,
    #[pyo3(get)]
    pub cell_count: usize,
    #[pyo3(get)]
    pub events: Vec<String>,
    #[pyo3(get)]
    pub warnings: Vec<String>,
    #[pyo3(get)]
    pub junction_captures: Vec<String>,
    #[pyo3(get)]
    pub resumed: bool,
    #[pyo3(get)]
    pub resumed_iteration: Option<usize>,
    #[pyo3(get)]
    pub checkpoint_saves: usize,
    #[pyo3(get)]
    pub last_checkpoint_iteration: Option<usize>,
    #[pyo3(get)]
    pub last_checkpoint_bytes: Option<u64>,
    #[pyo3(get)]
    pub debug_ovito_frames: usize,
}

#[pymethods]
impl PyRunResult {
    #[getter]
    fn fiber_count(&self) -> usize {
        self.assembly.topology.fibers.len()
    }

    /// The final assembly, for inspection or as the start of another recipe.
    #[getter]
    fn assembly(&self) -> PyAssembly {
        PyAssembly {
            model: Arc::new(Mutex::new(AssemblyModel::from_assembly(
                self.assembly.clone(),
                self.stack_axis,
            ))),
        }
    }

    fn centerlines(&self) -> Vec<Vec<Vec3>> {
        assembly_centerlines(&self.assembly)
    }

    fn characterize(&self) -> PyAnalysisReport {
        PyAnalysisReport {
            inner: characterize_assembly(&self.assembly),
        }
    }

    #[getter]
    fn junction_count(&self) -> usize {
        self.assembly.junctions.junctions.len()
    }

    #[pyo3(signature = (path, *, view_script_path=None, session_path=None, coloring="fiber"))]
    fn write_ovito(
        &self,
        path: PathBuf,
        view_script_path: Option<PathBuf>,
        session_path: Option<PathBuf>,
        coloring: &str,
    ) -> PyResult<()> {
        let coloring = parse_choice(coloring, "coloring", OVITO_COLORINGS)?;
        let mut config = OvitoTrajectoryConfig::fiber_segments(path, 1).with_coloring(coloring);
        config.representation = OvitoRepresentation::FiberSegments;
        config.view_script_path = view_script_path.clone();
        config.session_path = session_path;
        write_ovito_assembly_frame(&self.assembly, &config, self.iterations, false)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        if let Some(script) = view_script_path {
            write_ovito_view_script(&config, &script)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
        Ok(())
    }

    #[pyo3(signature = (data_path, *, mode="spherocylinders_exact", sphere_spacing_over_radius=1.0 / 3.0, density=1.0, atom_type=1, bond_type=1))]
    fn export_bpm(
        &self,
        data_path: PathBuf,
        mode: &str,
        sphere_spacing_over_radius: f64,
        density: f64,
        atom_type: u32,
        bond_type: u32,
    ) -> PyResult<(usize, usize)> {
        // Hyphenated spellings from earlier releases are still accepted.
        let mode = mode.to_ascii_lowercase().replace('-', "_");
        let config = BpmExportConfig {
            data_path: data_path.clone(),
            mode: parse_choice(&mode, "BPM export mode", BPM_MODES)?,
            sphere_spacing_over_radius,
            density,
            atom_type,
            bond_type,
        };
        let model = build_bpm_model(&self.assembly, &config)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        write_bpm_lammps_data(&model, &data_path)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok((model.particles(), model.bonds()))
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
        let report = write_puma_bundle(&self.assembly, &config)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(PyPumaExportReport { inner: report })
    }

    fn __repr__(&self) -> String {
        format!(
            concat!(
                "RunResult(fibers={}, iterations={}, converged={}, ",
                "max_penetration={}, max_curvature_ratio={}, active_segments={})"
            ),
            self.assembly.topology.fibers.len(),
            self.iterations,
            self.converged,
            self.max_penetration,
            self.max_curvature_ratio,
            self.active_segments,
        )
    }
}

enum RunFailure {
    Operation {
        index: usize,
        iteration: usize,
        reason: String,
    },
    Other(String),
}

fn run_native_recipe(
    assembly: FiberAssembly,
    recipe: FormationRecipeConfig,
    relaxation_config: tangle_relax::RelaxationConfig,
    checkpoint_config: Option<CheckpointConfig>,
    debug_ovito_config: Option<OvitoTrajectoryConfig>,
    stack_axis: usize,
) -> Result<PyRunResult, RunFailure> {
    let mut app = App::new();
    app.add_plugins(TangleWorkflowPlugin {
        initial: TangleStage::Relax,
    })
    .add_plugins(TanglePreparedAssemblyPlugin { assembly })
    .add_plugins(RelaxationPlugin {
        config: relaxation_config,
    })
    .add_plugins(FormationRecipePlugin { config: recipe });
    if let Some(config) = checkpoint_config {
        app.add_plugins(CheckpointPlugin { config });
    }
    if let Some(config) = debug_ovito_config {
        app.add_plugins(OvitoTrajectoryPlugin { config });
    }
    app.start();

    let missing = |what: &str| RunFailure::Other(format!("{what} was not installed"));
    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .ok_or_else(|| missing("relaxation result"))?
        .clone();
    let recipe_state = app
        .get_resource_ref::<FormationRecipeState>()
        .ok_or_else(|| missing("recipe result"))?
        .clone();
    if let Some(failure) = &recipe_state.failure {
        return Err(RunFailure::Operation {
            index: failure.operation,
            iteration: failure.iteration,
            reason: failure.reason.clone(),
        });
    }
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .ok_or_else(|| missing("final fiber assembly"))?
        .clone();
    let checkpoint = app
        .get_resource_ref::<CheckpointReport>()
        .map(|report| report.clone())
        .unwrap_or_default();
    let debug_ovito_frames = app
        .get_resource_ref::<OvitoTrajectoryReport>()
        .map_or(0, |report| report.frames);
    Ok(PyRunResult {
        assembly,
        stack_axis,
        iterations: relaxation.iterations,
        converged: relaxation.converged,
        max_penetration: relaxation.max_penetration,
        max_curvature_ratio: relaxation.max_curvature_ratio,
        active_segments: relaxation.active_segments,
        active_vertices: relaxation.active_vertices,
        segment_splits: relaxation.segment_splits,
        segment_merges: relaxation.segment_merges,
        refinement_passes: relaxation.refinement_passes,
        coarsening_passes: relaxation.coarsening_passes,
        uploaded_bytes: relaxation.uploaded_bytes,
        downloaded_bytes: relaxation.downloaded_bytes,
        cell_count: relaxation.cell_count,
        events: recipe_state
            .events
            .iter()
            .map(|event| event.description.clone())
            .collect(),
        warnings: recipe_state
            .warnings
            .iter()
            .map(|warning| warning.reason.clone())
            .collect(),
        junction_captures: recipe_state
            .junction_captures
            .iter()
            .map(|report| {
                format!(
                    "{}: {} created from {} candidates ({} rejected)",
                    report.policy_name, report.created, report.candidates, report.rejected
                )
            })
            .collect(),
        resumed: checkpoint.resumed,
        resumed_iteration: checkpoint.resumed_iteration,
        checkpoint_saves: checkpoint.saves,
        last_checkpoint_iteration: checkpoint.last_saved_iteration,
        last_checkpoint_bytes: checkpoint.last_saved_bytes,
        debug_ovito_frames,
    })
}

fn assembly_centerlines(assembly: &FiberAssembly) -> Vec<Vec<Vec3>> {
    assembly
        .topology
        .fibers
        .iter()
        .map(|fiber| {
            let start = fiber.vertices.start as usize;
            let end = start + fiber.vertices.len as usize;
            assembly.geometry.placed.positions[start..end].to_vec()
        })
        .collect()
}
