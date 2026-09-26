//! Python bindings for TANGLE.

mod analysis;
mod checkpoint;
mod collection;
mod common;
mod compaction;
mod ct;
mod generators;
mod image;
mod junctions;
mod recipe;
mod settings;

use pyo3::prelude::*;

use analysis::{PyAnalysisReport, PyNeighborReport, PyPumaExportReport, PyShapeReport};
use checkpoint::PyCheckpointSettings;
use collection::{PyAssembly, PyCell, PyFiberCollection, PyFiberSelection, PyMaterial};
use compaction::{
    PyAxisWeightsPath, PyCellLengthsTarget, PyCellVolumeTarget, PyCompactionSettings,
    PyDirectionalPressureTarget, PyEqualPressurePath, PyMeanPressureTarget, PyMinimumWorkPath,
    PyPenaltyEnergyTarget, PyStressRatioPath, PyVolumeFractionTarget,
};
use generators::{
    PyAlignedOrientation, PyDensityGradientPosition, PyFiberPopulation, PyIsotropicOrientation,
    PyLayeredBiaxialOrientation, PyLayeredPosition, PyPlanarOrientation, PyUniformPosition,
};
use image::PyImageRelaxer;
use junctions::PyJunctionPolicy;
use recipe::{
    PyCircularFootprint, PyHeldTargets, PyRandomFiberFraction, PyRecipe, PyRunResult, RecipeError,
};
use settings::{
    PyAdaptiveSegmentationSettings, PyRelaxationOverrides, PyRelaxationSettings, PySolvePolicy,
};

/// Native Python extension module.
#[pymodule]
fn _tangle(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RecipeError", py.get_type::<RecipeError>())?;

    module.add_class::<PyAnalysisReport>()?;
    module.add_class::<PyNeighborReport>()?;
    module.add_class::<PyPumaExportReport>()?;
    module.add_class::<PyShapeReport>()?;

    module.add_class::<PyCell>()?;
    module.add_class::<PyMaterial>()?;
    module.add_class::<PyAssembly>()?;
    module.add_class::<PyFiberCollection>()?;
    module.add_class::<PyFiberSelection>()?;

    module.add_class::<PyIsotropicOrientation>()?;
    module.add_class::<PyPlanarOrientation>()?;
    module.add_class::<PyLayeredBiaxialOrientation>()?;
    module.add_class::<PyAlignedOrientation>()?;
    module.add_class::<PyUniformPosition>()?;
    module.add_class::<PyLayeredPosition>()?;
    module.add_class::<PyDensityGradientPosition>()?;
    module.add_class::<PyFiberPopulation>()?;
    module.add_function(wrap_pyfunction!(
        generators::generate_point_crossing_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        generators::generate_multisegment_crossing_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        generators::generate_fiber_pair_crossing_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        generators::generate_fiber_population_py,
        module
    )?)?;

    module.add_class::<PyCheckpointSettings>()?;
    module.add_class::<PyAdaptiveSegmentationSettings>()?;
    module.add_class::<PyRelaxationSettings>()?;
    module.add_class::<PyRelaxationOverrides>()?;
    module.add_class::<PySolvePolicy>()?;
    module.add_class::<PyJunctionPolicy>()?;

    module.add_class::<PyVolumeFractionTarget>()?;
    module.add_class::<PyCellVolumeTarget>()?;
    module.add_class::<PyCellLengthsTarget>()?;
    module.add_class::<PyMeanPressureTarget>()?;
    module.add_class::<PyDirectionalPressureTarget>()?;
    module.add_class::<PyPenaltyEnergyTarget>()?;
    module.add_class::<PyAxisWeightsPath>()?;
    module.add_class::<PyEqualPressurePath>()?;
    module.add_class::<PyStressRatioPath>()?;
    module.add_class::<PyMinimumWorkPath>()?;
    module.add_class::<PyCompactionSettings>()?;

    module.add_class::<PyCircularFootprint>()?;
    module.add_class::<PyRandomFiberFraction>()?;
    module.add_class::<PyHeldTargets>()?;
    module.add_class::<PyRecipe>()?;
    module.add_class::<PyRunResult>()?;
    module.add_class::<PyImageRelaxer>()?;
    module.add_class::<ct::PyCtHessian>()?;
    module.add_function(wrap_pyfunction!(ct::ct_gaussian_filter, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_sample, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_foreground_depth, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_core_holes, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_rasterize, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_paint, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_trace_one_way, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_trace_fibers, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_owners, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_end_step, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_cut_void, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_resample, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_support, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_curvature_ratio, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_render_occupancy, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_local_residual, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_trim_duplicates, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_remove_unsupported, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_merge_fragments, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_split_kinks, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_resolve_side_by_side, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_render_grey, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_node_confidence, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_overlap, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_project, module)?)?;
    module.add_function(wrap_pyfunction!(ct::ct_back_project, module)?)?;
    Ok(())
}
