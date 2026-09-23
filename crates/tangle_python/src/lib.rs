//! Python bindings for TANGLE.

mod analysis;
mod checkpoint;
mod collection;
mod common;
mod compaction;
mod generators;
mod junctions;
mod recipe;
mod settings;

use pyo3::prelude::*;

use analysis::{
    PyAnalysisReport, PyEntanglementReport, PyNeighborReport, PyPhaseReport, PyPumaExportReport,
    PyScorecard, PyShapeReport, PySliceReport,
};
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
    module.add_class::<PyScorecard>()?;
    module.add_class::<PyEntanglementReport>()?;
    module.add_class::<PySliceReport>()?;
    module.add_class::<PyPhaseReport>()?;
    module.add_function(wrap_pyfunction!(analysis::score_structure_py, module)?)?;

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
    Ok(())
}
