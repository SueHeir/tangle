//! Python bindings for TANGLE.

mod analysis;
mod checkpoint;
mod collection;
mod compaction;
mod generators;
mod junctions;
mod recipe;
mod settings;

use pyo3::prelude::*;

use analysis::{PyAnalysisReport, PyPumaExportReport};
use checkpoint::PyCheckpointSettings;
use collection::{PyAssembly, PyCell, PyFiberCollection, PyFiberSelection, PyMaterial};
use compaction::PyCompactionSettings;
use generators::PyFiberPopulationSettings;
use junctions::PyJunctionPolicy;
use recipe::{PyRecipe, PyRunResult};
use settings::{
    PyAdaptiveSegmentationSettings, PyCellListSettings, PyRelaxationOverrides,
    PyRelaxationSettings, PySolvePolicy,
};

/// Native Python extension module.
#[pymodule]
fn _tangle(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAnalysisReport>()?;
    module.add_class::<PyPumaExportReport>()?;
    module.add_class::<PyCheckpointSettings>()?;
    module.add_class::<PyCell>()?;
    module.add_class::<PyAssembly>()?;
    module.add_class::<PyMaterial>()?;
    module.add_class::<PyFiberCollection>()?;
    module.add_class::<PyFiberSelection>()?;
    module.add_class::<PyCompactionSettings>()?;
    module.add_class::<PyFiberPopulationSettings>()?;
    module.add_class::<PyJunctionPolicy>()?;
    module.add_class::<PyCellListSettings>()?;
    module.add_class::<PyAdaptiveSegmentationSettings>()?;
    module.add_class::<PyRelaxationSettings>()?;
    module.add_class::<PyRelaxationOverrides>()?;
    module.add_class::<PySolvePolicy>()?;
    module.add_class::<PyRecipe>()?;
    module.add_class::<PyRunResult>()?;
    module.add_function(wrap_pyfunction!(
        generators::py_generate_point_crossing,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        generators::py_generate_multisegment_crossing,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        generators::py_generate_fiber_pair_crossing,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        generators::py_generate_fiber_population,
        module
    )?)?;
    Ok(())
}
