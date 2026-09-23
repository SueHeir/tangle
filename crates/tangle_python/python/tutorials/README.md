# TANGLE topic notebooks

These notebooks form a topic-oriented reference for the Python interface. Each
notebook covers one concept completely and uses small, editable configurations.
They complement the workflow-oriented examples in `../examples/`.

| Notebook | Topic |
| --- | --- |
| `00_installation_and_environment.ipynb` | Build the extension, register/select the kernel, and verify the import |
| `01_high_level_overview.ipynb` | A readable end-to-end two-ply manufacturing recipe and conceptual map |
| `02_cells_and_boundaries.ipynb` | Cell dimensions, origins, periodic axes, and hard walls |
| `03_materials_and_centerlines.ipynb` | Fiber diameters, bend limits, placed shapes, and rest shapes |
| `04_fiber_collections.ipynb` | Detached collections, metadata, layers, selection, and composition |
| `05_fiber_generation.ipynb` | All built-in generators and every population setting |
| `06_insertion_and_transforms.ipynb` | Recipe insertion, activation, translation, rotation, and selections |
| `07_relaxation_settings.ipynb` | Every relaxation, contact, cell-list, and backend setting |
| `08_adaptive_refinement.ipynb` | Every refinement/coarsening setting and the supplied profiles |
| `09_solve_policies_and_overrides.ipynb` | Relaxation operations, acceptance policies, and per-stage overrides |
| `10_layer_motion_and_needling.ipynb` | Layer placement, target release, needling, and cell fitting |
| `11_compaction.ipynb` | Every target, path, kinematic choice, increment, energy, and guard setting |
| `12_junction_capture.ipynb` | Every junction filter and explicit capture schedule |
| `13_checkpoints_and_resume.ipynb` | Every checkpoint, continuation, and branching option |
| `14_results_and_exports.ipynb` | Results, native analysis, OVITO, BPM, and PuMA export controls |

After installing the Python, Rust/Cargo, and native compiler prerequisites
described step-by-step in notebook 00, install the development extension,
register its kernel, and launch Jupyter:

```console
python3 -m venv .venv-tangle
source .venv-tangle/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
maturin develop --release --manifest-path crates/tangle_python/Cargo.toml
python -m ipykernel install --user --name tangle --display-name "Python (TANGLE)"
jupyter lab crates/tangle_python/python/tutorials
```

Select **Python (TANGLE)** from Jupyter's kernel menu. Notebook 00 diagnoses the
active interpreter and verifies that the extension is visible before the API
tutorials begin.

With Miniforge, replace the `venv` creation/activation above with
`conda create --name tangle python=3.12 -y` and `conda activate tangle`, then
run the same `pip`, Maturin, and IPykernel commands. Rust/Cargo and a native
compiler remain required; notebook 00 contains the complete Miniforge path.

The setup commands above are POSIX-shell commands. Tutorial 00 and the
[Python guide](../../README.md) also provide Windows PowerShell instructions.
After rebuilding the extension, restart the notebook kernel before using new
API arguments. Geometry uses consistent length units (meters in these tutorials),
not an automatic units-conversion system.

The notebooks are intentionally unexecuted in version control. Expensive solve
or export cells are guarded by `RUN_SOLVER` or `RUN_RECIPE`; change the guard
only when you want to run that calculation.

`generate.py` is the source used to rebuild the committed notebooks. After an
API documentation change, run:

```console
python crates/tangle_python/python/tutorials/generate.py
```
