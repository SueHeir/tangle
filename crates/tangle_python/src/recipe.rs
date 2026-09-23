use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use grass_app::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
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
    FormationOperation, FormationRecipeConfig, FormationRecipePlugin, FormationRecipeState,
    NeedlingConfig, NeedlingSelection,
};
use tangle_relax::{RelaxationPlugin, RelaxationState};

use crate::analysis::{PyAnalysisReport, PyPumaExportReport};
use crate::checkpoint::PyCheckpointSettings;
use crate::collection::{AssemblyModel, PyAssembly, PyCell, PyFiberCollection, PyFiberSelection};
use crate::compaction::PyCompactionSettings;
use crate::junctions::PyJunctionPolicy;
use crate::settings::{PyRelaxationOverrides, PyRelaxationSettings, PySolvePolicy};

#[pyclass(name = "Recipe", module = "tangle._tangle")]
pub(crate) struct PyRecipe {
    model: Arc<Mutex<AssemblyModel>>,
    operations: Vec<FormationOperation>,
    operation_descriptions: Vec<String>,
    next_formation_step: u32,
    #[pyo3(get, set)]
    layer_axis: usize,
}

#[pymethods]
impl PyRecipe {
    #[new]
    #[pyo3(signature = (cell, *, layer_axis=2))]
    fn new(cell: &Bound<'_, PyAny>, layer_axis: usize) -> PyResult<Self> {
        if layer_axis >= 3 {
            return Err(PyValueError::new_err("layer_axis must be 0, 1, or 2"));
        }
        let model = if let Ok(cell) = cell.extract::<PyRef<'_, PyCell>>() {
            Arc::new(Mutex::new(AssemblyModel::new(cell.inner)))
        } else if let Ok(assembly) = cell.extract::<PyRef<'_, PyAssembly>>() {
            assembly.model.clone()
        } else {
            return Err(PyValueError::new_err(
                "Recipe expects a Cell or Assembly as its first argument",
            ));
        };
        Ok(Self {
            model,
            operations: Vec::new(),
            operation_descriptions: Vec::new(),
            next_formation_step: 0,
            layer_axis,
        })
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
        self.operations
            .push(FormationOperation::ActivateFibersThrough(step));
        self.operation_descriptions.push(format!(
            "insert {:?} ({} fibers) through formation step {}",
            selection_name,
            selection.fiber_ids.len(),
            step
        ));
        Ok(selection)
    }

    #[pyo3(signature = (*, maximum_iterations=2000))]
    fn relax(&mut self, maximum_iterations: usize) -> PyResult<()> {
        if maximum_iterations == 0 {
            return Err(PyValueError::new_err("maximum_iterations must be positive"));
        }
        self.operations
            .push(FormationOperation::RelaxUntilConverged { maximum_iterations });
        self.operation_descriptions.push(format!(
            "relax until converged (up to {maximum_iterations} iterations)"
        ));
        Ok(())
    }

    fn relax_for(&mut self, iterations: usize) -> PyResult<()> {
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be positive"));
        }
        self.operations
            .push(FormationOperation::RelaxFor(iterations));
        self.operation_descriptions
            .push(format!("relax for exactly {iterations} iterations"));
        Ok(())
    }

    #[pyo3(signature = (policy, overrides=None))]
    fn relax_with_policy(
        &mut self,
        policy: PyRef<'_, PySolvePolicy>,
        overrides: Option<PyRef<'_, PyRelaxationOverrides>>,
    ) -> PyResult<()> {
        let policy = policy.to_rust()?;
        self.operation_descriptions
            .push(format!("relax with policy {:?}", policy.name));
        self.operations.push(if let Some(overrides) = overrides {
            FormationOperation::RelaxWithOverrides {
                policy,
                overrides: overrides.to_rust()?,
            }
        } else {
            FormationOperation::RelaxWithPolicy(policy)
        });
        Ok(())
    }

    fn set_material_bend_radius(
        &mut self,
        material_name: String,
        minimum_bend_radius: f64,
    ) -> PyResult<()> {
        if material_name.trim().is_empty() {
            return Err(PyValueError::new_err("material_name must not be empty"));
        }
        if !minimum_bend_radius.is_finite() || minimum_bend_radius <= 0.0 {
            return Err(PyValueError::new_err(
                "minimum_bend_radius must be positive and finite",
            ));
        }
        self.operations
            .push(FormationOperation::SetMaterialBendRadius {
                material_name: material_name.clone(),
                minimum_bend_radius,
            });
        self.operation_descriptions.push(format!(
            "set material {:?} minimum bend radius to {}",
            material_name, minimum_bend_radius
        ));
        Ok(())
    }

    fn relax_until_targets_reached(
        &mut self,
        tolerance: f32,
        maximum_iterations: usize,
    ) -> PyResult<()> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(PyValueError::new_err(
                "target tolerance must be nonnegative and finite",
            ));
        }
        if maximum_iterations == 0 {
            return Err(PyValueError::new_err("maximum_iterations must be positive"));
        }
        self.operations
            .push(FormationOperation::RelaxUntilTargetsReached {
                tolerance,
                maximum_iterations,
            });
        self.operation_descriptions.push(format!(
            "relax active manufacturing targets to tolerance {tolerance}"
        ));
        Ok(())
    }

    #[pyo3(signature = (spacing_scale, *, stiffness=0.25, max_translation=1.0e-5))]
    fn move_layers(
        &mut self,
        spacing_scale: f32,
        stiffness: f32,
        max_translation: f32,
    ) -> PyResult<()> {
        positive_finite(spacing_scale, "spacing_scale")?;
        unit_fraction(stiffness, "stiffness")?;
        positive_finite(max_translation, "max_translation")?;
        self.operations.push(FormationOperation::MoveLayers {
            spacing_scale,
            stiffness,
            max_translation,
        });
        self.operation_descriptions
            .push(format!("move layers to spacing scale {spacing_scale}"));
        Ok(())
    }

    #[pyo3(signature = (layer, gap, *, stiffness=0.25, max_translation=1.0e-5))]
    fn place_layer_above(
        &mut self,
        layer: u32,
        gap: f32,
        stiffness: f32,
        max_translation: f32,
    ) -> PyResult<()> {
        if layer == 0 {
            return Err(PyValueError::new_err("layer must be greater than zero"));
        }
        positive_finite(gap, "gap")?;
        unit_fraction(stiffness, "stiffness")?;
        positive_finite(max_translation, "max_translation")?;
        self.operations.push(FormationOperation::PlaceLayerAbove {
            layer,
            gap,
            stiffness,
            max_translation,
        });
        self.operation_descriptions
            .push(format!("place layer {layer} above layer {}", layer - 1));
        Ok(())
    }

    fn release_layer_targets(&mut self) {
        self.operations
            .push(FormationOperation::ReleaseLayerTargets);
        self.operation_descriptions
            .push("release layer targets".to_string());
    }

    #[pyo3(signature = (layer, center, diameter, depth, *, minimum_fiber_diameter=None, stiffness=0.25, max_translation=1.0e-5, maximum_translation_over_fiber_diameter=0.25))]
    fn needle_layer_circular(
        &mut self,
        layer: u32,
        center: [f32; 2],
        diameter: f32,
        depth: f32,
        minimum_fiber_diameter: Option<f32>,
        stiffness: f32,
        max_translation: f32,
        maximum_translation_over_fiber_diameter: f32,
    ) -> PyResult<()> {
        if center.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("needle center must be finite"));
        }
        validate_needling(
            diameter,
            depth,
            minimum_fiber_diameter,
            stiffness,
            max_translation,
            maximum_translation_over_fiber_diameter,
        )?;
        self.operations
            .push(FormationOperation::NeedleLayer(NeedlingConfig {
                layer,
                selection: NeedlingSelection::CircularFootprint { center, diameter },
                minimum_fiber_diameter,
                depth,
                stiffness,
                max_translation,
                maximum_translation_over_fiber_diameter,
            }));
        self.operation_descriptions
            .push(format!("needle layer {layer} through a circular footprint"));
        Ok(())
    }

    #[pyo3(signature = (layer, fraction, depth, *, seed=0, minimum_fiber_diameter=None, stiffness=0.25, max_translation=1.0e-5, maximum_translation_over_fiber_diameter=0.25))]
    fn needle_layer_random(
        &mut self,
        layer: u32,
        fraction: f32,
        depth: f32,
        seed: u64,
        minimum_fiber_diameter: Option<f32>,
        stiffness: f32,
        max_translation: f32,
        maximum_translation_over_fiber_diameter: f32,
    ) -> PyResult<()> {
        unit_fraction(fraction, "fraction")?;
        validate_needling(
            1.0,
            depth,
            minimum_fiber_diameter,
            stiffness,
            max_translation,
            maximum_translation_over_fiber_diameter,
        )?;
        self.operations
            .push(FormationOperation::NeedleLayer(NeedlingConfig {
                layer,
                selection: NeedlingSelection::RandomFiberFraction { fraction, seed },
                minimum_fiber_diameter,
                depth,
                stiffness,
                max_translation,
                maximum_translation_over_fiber_diameter,
            }));
        self.operation_descriptions
            .push(format!("needle a random fraction of layer {layer}"));
        Ok(())
    }

    fn release_needles(&mut self) {
        self.operations.push(FormationOperation::ReleaseNeedles);
        self.operation_descriptions
            .push("release needle targets".to_string());
    }

    #[pyo3(signature = (*, axes=[false, false, true], padding=0.0))]
    fn fit_cell_to_active_fibers(&mut self, axes: [bool; 3], padding: f32) -> PyResult<()> {
        if !axes.iter().any(|selected| *selected) {
            return Err(PyValueError::new_err(
                "at least one cell-fit axis must be selected",
            ));
        }
        if !padding.is_finite() || padding < 0.0 {
            return Err(PyValueError::new_err(
                "padding must be nonnegative and finite",
            ));
        }
        self.operations
            .push(FormationOperation::FitCellToActiveFibers { axes, padding });
        self.operation_descriptions
            .push(format!("fit cell axes {axes:?} to active fibers"));
        Ok(())
    }

    #[pyo3(signature = (settings, overrides=None))]
    fn compact(
        &mut self,
        settings: PyRef<'_, PyCompactionSettings>,
        overrides: Option<PyRef<'_, PyRelaxationOverrides>>,
    ) -> PyResult<()> {
        let config = settings.to_rust()?;
        self.operation_descriptions
            .push(format!("compact toward {} target", settings.target_type));
        self.operations.push(if let Some(overrides) = overrides {
            FormationOperation::CompactWithOverrides {
                config,
                overrides: overrides.to_rust()?,
            }
        } else {
            FormationOperation::Compact(config)
        });
        Ok(())
    }

    fn capture_junctions(&mut self, policy: PyRef<'_, PyJunctionPolicy>) -> PyResult<()> {
        let policy = policy.to_rust()?;
        self.operation_descriptions
            .push(format!("capture junctions using policy {:?}", policy.name));
        self.operations
            .push(FormationOperation::CaptureJunctions(policy));
        Ok(())
    }

    fn relax_and_capture(
        &mut self,
        iterations: usize,
        every: usize,
        policy: PyRef<'_, PyJunctionPolicy>,
    ) -> PyResult<()> {
        if iterations == 0 || every == 0 || every > iterations {
            return Err(PyValueError::new_err(
                "iterations must be positive and every must lie within 1..=iterations",
            ));
        }
        let policy = policy.to_rust()?;
        self.operation_descriptions.push(format!(
            "relax for {iterations} iterations and capture junctions every {every}"
        ));
        self.operations.push(FormationOperation::RelaxAndCapture {
            iterations,
            every,
            policy,
        });
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
                .with_coloring(parse_ovito_coloring(debug_ovito_coloring)?);
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
            layer_axis: self.layer_axis,
            operations: self.operations.clone(),
        };
        let result = py
            .detach(move || {
                run_native_recipe(assembly, recipe, relaxation, checkpoint, debug_ovito)
            })
            .map_err(PyRuntimeError::new_err)?;
        self.model.lock().expect("assembly lock poisoned").assembly = result.assembly.clone();
        Ok(result)
    }

    fn __repr__(&self) -> String {
        let model = self.model.lock().expect("assembly lock poisoned");
        format!(
            "Recipe(fibers={}, operations={}, layer_axis={})",
            model.assembly.topology.fibers.len(),
            self.operations.len(),
            self.layer_axis
        )
    }
}

#[pyclass(name = "RunResult", module = "tangle._tangle", frozen)]
#[derive(Clone)]
pub(crate) struct PyRunResult {
    assembly: FiberAssembly,
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
        let coloring = parse_ovito_coloring(coloring)?;
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

    #[pyo3(signature = (data_path, *, mode="spherocylinders-exact", sphere_spacing_over_radius=1.0 / 3.0, density=1.0, atom_type=1, bond_type=1))]
    fn export_bpm(
        &self,
        data_path: PathBuf,
        mode: &str,
        sphere_spacing_over_radius: f64,
        density: f64,
        atom_type: u32,
        bond_type: u32,
    ) -> PyResult<(usize, usize)> {
        let config = BpmExportConfig {
            data_path: data_path.clone(),
            mode: parse_bpm_export_mode(mode)?,
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

fn run_native_recipe(
    assembly: FiberAssembly,
    recipe: FormationRecipeConfig,
    relaxation_config: tangle_relax::RelaxationConfig,
    checkpoint_config: Option<CheckpointConfig>,
    debug_ovito_config: Option<OvitoTrajectoryConfig>,
) -> Result<PyRunResult, String> {
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

    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .ok_or_else(|| "relaxation result was not installed".to_string())?
        .clone();
    let recipe_state = app
        .get_resource_ref::<FormationRecipeState>()
        .ok_or_else(|| "recipe result was not installed".to_string())?
        .clone();
    if let Some(failure) = &recipe_state.failure {
        return Err(format!(
            "recipe operation {} failed at iteration {}: {}",
            failure.operation, failure.iteration, failure.reason
        ));
    }
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .ok_or_else(|| "final fiber assembly was not installed".to_string())?
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

fn parse_ovito_coloring(value: &str) -> PyResult<OvitoColoring> {
    match value {
        "fiber" => Ok(OvitoColoring::Fiber),
        "curvature_ratio" => Ok(OvitoColoring::CurvatureRatio),
        "refinement_level" => Ok(OvitoColoring::RefinementLevel),
        other => Err(PyValueError::new_err(format!(
            "unknown coloring {other:?}; expected 'fiber', 'curvature_ratio', or 'refinement_level'"
        ))),
    }
}

fn parse_bpm_export_mode(value: &str) -> PyResult<BpmExportMode> {
    let normalized = value.to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "spheres-exact" => Ok(BpmExportMode::SpheresExact),
        "spheres-dynamic" => Ok(BpmExportMode::SpheresDynamic),
        "spherocylinders-exact" | "sphero-cylinder-exact" => {
            Ok(BpmExportMode::SpherocylindersExact)
        }
        "spherocylinders-constant" | "sphero-cylinder-constant" => {
            Ok(BpmExportMode::SpherocylindersConstant)
        }
        other => Err(PyValueError::new_err(format!(
            "unknown BPM mode {other:?}; expected 'spheres-exact', 'spheres-dynamic', \
             'spherocylinders-exact', or 'spherocylinders-constant'"
        ))),
    }
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

fn positive_finite(value: f32, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive and finite"
        )))
    }
}

fn unit_fraction(value: f32, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in (0, 1]")))
    }
}

fn validate_needling(
    diameter: f32,
    depth: f32,
    minimum_fiber_diameter: Option<f32>,
    stiffness: f32,
    max_translation: f32,
    maximum_translation_over_fiber_diameter: f32,
) -> PyResult<()> {
    positive_finite(diameter, "diameter")?;
    positive_finite(depth, "depth")?;
    if let Some(value) = minimum_fiber_diameter {
        positive_finite(value, "minimum_fiber_diameter")?;
    }
    unit_fraction(stiffness, "stiffness")?;
    positive_finite(max_translation, "max_translation")?;
    positive_finite(
        maximum_translation_over_fiber_diameter,
        "maximum_translation_over_fiber_diameter",
    )?;
    Ok(())
}
