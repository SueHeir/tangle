# TANGLE Python guide

Python is the primary user path for TANGLE. Python constructs fiber
collections and manufacturing recipes; the existing Rust implementation owns
validation, CubeCL device residency, relaxation, refinement, and export.

The extension is compiled code. A development install requires Python 3.11 or
newer, a stable [Rust toolchain](https://rustup.rs/) including Cargo, a native
C/C++ compiler/linker, and network access for the first dependency build.
CubeCL's build dependency downloads its matching bundled LLVM automatically;
users do not need a system LLVM installation or `llvm-config`.

Obtain the source with Git (or download and unpack the repository):

```console
git clone https://github.com/SueHeir/tangle.git
cd tangle
```

For a first Rust installation on macOS, Linux, or WSL, use the official Rustup
installer and verify both compiler and package manager:

```console
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install stable
rustup default stable
rustc --version
cargo --version
```

On Windows, run the official `rustup-init.exe` from
[rust-lang.org/tools/install](https://rust-lang.org/tools/install/), select the
stable MSVC toolchain, and install the Visual Studio Desktop C++ build tools if
prompted.

The commands below install Maturin into the active Python environment with
`pip`, which keeps it aligned with the selected interpreter. Users who prefer
a Rust-managed global CLI may instead run `cargo install --locked maturin` and
then activate the intended Python environment before `maturin develop`. Do not
install Maturin both ways unless you intentionally want both copies.

Then create and activate the Python environment and let Maturin invoke Cargo
and install the compiled extension, all from the repository root. The following
uses a POSIX shell; tutorial 00 also gives Windows PowerShell commands.

```console
python3 -m venv .venv-tangle
source .venv-tangle/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
maturin develop --release \
  --manifest-path crates/tangle_python/Cargo.toml
```

Activating the environment is important: it sets `VIRTUAL_ENV`, which tells
Maturin exactly where to install the compiled module even if another `.venv`
directory exists nearby.

Miniforge users can replace the `venv` commands with a named Conda environment.
Rust/Cargo and the native compiler are still separate prerequisites:

```console
conda create --name tangle python=3.12 -y
conda activate tangle
python -m pip install --upgrade pip
python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
maturin develop --release \
  --manifest-path crates/tangle_python/Cargo.toml
python -m ipykernel install --user \
  --name tangle --display-name "Python (TANGLE)"
```

Use the named `tangle` environment rather than Conda's `base` environment, then
select **Python (TANGLE)** in VS Code or Jupyter.

These are source-build instructions, not a request to install an unrelated
package from PyPI with the same name. Clone this repository first and run build
commands at its root. Prebuilt wheel distribution is not provided here yet.

For Windows PowerShell, after installing Rust and the native build tools:

```powershell
py -m venv .venv-tangle
.\.venv-tangle\Scripts\Activate.ps1
python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
maturin develop --release --manifest-path crates/tangle_python/Cargo.toml
python -m ipykernel install --user --name tangle --display-name "Python (TANGLE)"
```

[Tutorial 00](python/tutorials/00_installation_and_environment.ipynb) contains
cross-platform setup, kernel registration, verification, and rebuild commands.
After rebuilding a loaded extension, restart the notebook kernel. For a failed
import, check `import sys; print(sys.executable)` and select the environment in
which Maturin installed TANGLE. Windows setup is documented but is not currently
covered by platform CI.

See `python/examples/` for executable scripts and matching Jupyter notebooks.
The notebooks contain the same calculations split into explanatory, editable
cells. To explore them interactively:

```console
cd crates/tangle_python/python/examples
jupyter lab
```

For a setting-by-setting reference, use `python/tutorials/`. Its fifteen
focused notebooks cover cells, materials, centerlines, collections, generators,
insertion, relaxation, adaptive segmentation, solve policies, layer motion,
needling, compaction, junction capture, checkpoints, results, OVITO, and BPM
export. Each configuration notebook enumerates every exposed field and its
units or accepted choices.

`python/examples/native/` contains Python script/notebook pairs for the native
manufacturing configurations; the PuMA fixture is in `python/examples/`.
See the [progressive example index](../../examples/README.md). Output paths,
export modes, and checkpoint identities can differ between front ends. The larger felt
notebooks retain their full scientific configurations but guard the expensive
execution cell so opening a notebook does not accidentally start an hours-long
run.

## Collections and staged insertion

`FiberCollection` is a detached, reusable set of centerlines. It does not need
to be contact-free. Each fiber carries its placed centerline, optional distinct
rest centerline, material, optional formation layer, and arbitrary string tags.

```python
import tangle

from tangle.units import mm, um

small = tangle.Material("small", diameter=7 * um, min_bend_radius=35 * um)

layer = tangle.FiberCollection("ply_01")
layer.add_fiber(
    [[-0.4 * mm, 0, 0], [0.4 * mm, 0, 0]],
    small,
    tags={"family": "machine_direction"},
    formation_layer=1,
)

cell = tangle.Cell([1 * mm, 1 * mm, 2 * mm], periodic="xy")
recipe = tangle.Recipe(cell)
ply = recipe.insert(layer, translation=[0.5 * mm, 0.5 * mm, 0.8 * mm])
recipe.relax_until_converged()
```

`insert()` returns a stable `FiberSelection` containing the assigned fiber IDs
and formation step. Every collection is packed once; recipe activation and
relaxation then occur inside one Rust/CubeCL execution. Layers stack along the
cell's `stack_axis`: z when z is bounded or every axis is periodic, otherwise
the last bounded axis. Pass
`stack_axis="x"` to `Cell` or `Recipe` to choose another. Generators,
compaction and `fit_cell_to_active_fibers` default to the same axis. Recipe
operations are ordered manufacturing instructions, not physical timesteps.

`insert()` changes the recipe's starting assembly immediately; every other
operation is recorded and runs inside `run()`. `run()` does not modify the
input `Assembly`: read the result from `result.assembly`. A failed operation
raises `tangle.RecipeError`, which carries `operation_index`, `operation`,
`iteration` and `reason`.

## Defaults and overrides

Configuration classes follow three naming conventions. `*Settings` are
run-wide, a `*Policy` is a named rule applied at one recipe step, and
`*Overrides` are temporary deltas for the step they are passed to.

Calling a constructor with no arguments exposes the corresponding Rust
defaults. Every field is also a keyword argument, can be inspected or changed
before the run, and `replace(**changes)` returns a modified copy. An unknown
keyword raises `TypeError`, and an unknown option string raises `ValueError`
on the line that set it:

```python
settings = tangle.RelaxationSettings(
    penetration_tolerance=0.1e-6,
    max_iterations=20_000,
    cell_size_scale=1.25,
    adaptive_segmentation=tangle.AdaptiveSegmentationSettings.profile(
        "balanced", refinement_interval=4, coarsening_persistence=48
    ),
)
print(settings.to_dict())
quick = settings.replace(max_iterations=2_000)
```

WGPU remains the default backend. On a system without a usable GPU, select
CubeCL's native multithreaded CPU runtime; the recipe and numerical settings do
not otherwise change:

```python
result = recipe.run(settings.replace(backend="cpu"))
```

The packaged extension includes both `wgpu` and `cpu`. Backend selection is
explicit so a large run never silently falls back to a much slower device.

Available adaptive profiles are `balanced`, `fast`, and `strict`. Profiles are
ordinary settings objects, not hidden solver modes. Assign `None` to
`settings.adaptive_segmentation` to disable refinement and coarsening.

Recipe-stage controls use the same pattern. Compaction exposes its target,
axis-selection path, kinematics, adaptive increment, guards, face balancing,
and penalty-energy weights:

```python
compaction = tangle.CompactionSettings.volume_fraction(
    0.40,  # compresses along the stack axis unless path= says otherwise
    balance_opposing_faces=True,
    max_penetration=0.1e-6,
)
recipe.compact(compaction)

pressure = tangle.CompactionSettings(
    tangle.MeanPressureTarget(1.0e3),
    path=tangle.EqualPressurePath("xy"),
)
```

Targets (`VolumeFractionTarget`, `CellVolumeTarget`, `CellLengthsTarget`,
`MeanPressureTarget`, `DirectionalPressureTarget`, `PenaltyEnergyTarget`) and
paths (`AxisWeightsPath`, `EqualPressurePath`, `StressRatioPath`,
`MinimumWorkPath`) are typed objects, so each carries only the fields it uses.

`SolvePolicy` sets a step's convergence targets and hard/soft gates, and
`RelaxationOverrides` temporarily shifts contact-versus-bending emphasis:

```python
recipe.solve(
    tangle.SolvePolicy(
        "cleanup/1-curvature-coarse",
        target_penetration=2e-6,
        target_curvature_ratio=1.25,  # max_* limits default to the targets
        max_iterations=30_000,
    ),
    tangle.RelaxationOverrides.preset("curvature_cleanup"),
)
```

Layer placement and needling hold fibers on targets until released. Use them
as context managers to release at the end of the block:

```python
with recipe.needle_layer(
    2,
    footprint=tangle.CircularFootprint.random(diameter=150 * um, seed=7),
    depth=350 * um,
):
    recipe.settle_targets(tolerance=0.5 * um, max_iterations=3_000)
    recipe.relax_for(150)
```

`JunctionPolicy` provides explicit,
deterministic contact-to-junction capture; contacts remain transient unless a
recipe includes `capture_junctions()` or `relax_and_capture()`.
Captured junctions are not mechanically enforced by subsequent relaxation.
Prefer late capture when representing binder added after the material settles.

Long recipes can use the same rolling, atomic restart checkpoints as native
TANGLE applications:

```python
checkpoint = tangle.CheckpointSettings(
    "three-ply-study",
    "output/three_ply.restart",
    interval_iterations=500,
)
result = recipe.run(settings, checkpoint=checkpoint)
print(result.checkpoint_saves, result.last_checkpoint_iteration)
```

Use `checkpoint.replace(resume=True)` to continue the saved recipe. A separate
`resume_path` and `resume_case_id` can be used to branch a run into a new
checkpoint file.
Cross-revision checkpoint compatibility is not guaranteed. Preserve the source
commit, settings, backend, and a geometry export alongside long-lived studies.

## Acceptance and OVITO keyframes

Inspect `result.converged`, `max_penetration`, `max_curvature_ratio`, `warnings`,
and `events` before treating geometry as mechanically admissible. Export methods
do not themselves prove convergence. A soft intermediate recipe gate can be
useful during formation but is not a final hard acceptance check.

To record formation milestones rather than regular solver frames:

```python
from pathlib import Path

output = Path("output/my_study")
output.mkdir(parents=True, exist_ok=True)
settings.debug_snapshot_interval = None  # keyframes only when debug path is supplied
result = recipe.run(
    settings,
    debug_ovito_path=output / "formation.dump",
    debug_ovito_view_script_path=output / "formation_view.py",
    debug_ovito_session_path=output / "formation.ovito",
)
```

Without `debug_ovito_path`, no debug trajectory is requested. With it, `None`
selects keyframes; a positive `debug_snapshot_interval` adds periodic snapshots.
The viewing script must be run with OVITO Python/`ovitos` to create the session
file. `result.write_ovito(...)` instead exports one final-state frame. Dense
trajectories add device readbacks and can substantially slow a solve.

## BPM export

`RunResult.export_bpm()` writes solver-neutral bonded-particle geometry. Choose
`spheres_exact`, `spheres_dynamic`, `spherocylinders_exact`, or
`spherocylinders_constant`. Sphere spacing is measured in fiber radii and may
range from `1/3` through `1`:

```python
particles, bonds = result.export_bpm(
    "output/network.data",
    mode="spheres_dynamic",
    sphere_spacing_over_radius=1.0 / 3.0,
    density=1800.0,
)
```

The data can be consumed by DIRT or another compatible LAMMPS BPM workflow;
TANGLE does not generate the downstream runtime configuration.
Capsule atom styles require a compatible downstream implementation; they are
not automatically supported by a stock LAMMPS installation. `atom_type` is the
starting type, with materials assigned consecutive types; the Python export
currently takes one density for all materials. Constitutive laws and contact
exclusions belong to the downstream model. Intentional intra-fiber sphere
overlap must not be confused with unwanted inter-fiber penetration.

## Native analysis and PuMA export

`Assembly.characterize()` and `RunResult.characterize()` dispatch to the Rust
`tangle_characterize` crate. The returned report includes exact centerline
lengths, nominal swept-volume fraction, per-material and per-fiber summaries,
curvature, and both length- and volume-weighted orientation tensors.

`export_puma()` dispatches to the Rust `tangle_export` voxelizer and writes a
directory containing `domain.vti`, optional diagnostic images, a manifest, and
the same native analysis as JSON:

```python
analysis = result.characterize()
bundle = result.export_puma(
    "output/specimen.puma",
    voxel_size=2e-6,
    include_fiber_ids=True,
    include_interface=True,
)
print(analysis.nominal_swept_volume_fraction)
print(bundle.voxel_volume_fraction, bundle.ambiguous_voxels)
```

### Contacts and neighbors

`characterize_neighbors(contact_gap, ...)` on an `Assembly` or `RunResult`
measures how fibers touch and travel together, which volume fraction and
orientation tensors cannot show. It reports contacts per unit length and
per fiber, the ratio to a random-placement baseline, the split between
in-axis (side-by-side) and out-of-axis (crossing) contacts, how much longer
contacts last than a straight crossing would ("excess persistence"), free
lengths between contacts, neighbor counts, and how quickly each fiber's set
of neighbors changes along its length:

```python
neighbors = result.characterize_neighbors(
    contact_gap=0.3e-6,        # surface gap counted as touching
    neighbor_gap=10e-6,        # surface gap counted as a neighbor
    in_axis_angle_degrees=20,  # below this, fibers run side by side
)
print(neighbors.contacts_per_length, neighbors.contact_ratio_to_random)
print(neighbors.in_axis_contact_fraction, neighbors.neighbor_correlation_length)
neighbors.write_json("output/neighbors.json")
```

The analysis uses only centerlines and radii, so centerlines tracked from a
CT scan can go through exactly the same call. `Assembly.insert()` adds a
collection without a recipe or relaxation:

```python
fiber = tangle.Material("fiber", diameter=7e-6)
scan = tangle.Assembly(tangle.Cell([1e-3, 1e-3, 1e-3]))
scan.insert(tangle.FiberCollection.from_centerlines(tracked_centerlines, fiber))
reference = scan.characterize_neighbors(contact_gap=1e-6)
```

CT segmentation cannot resolve gaps below about one voxel, so use a contact
tolerance of that order on both sides of a comparison. See the
[analysis guide](../../docs/puma_interoperability.md#contacts-and-neighbors)
for definitions.

PuMA is optional and is called directly in user Python code. The
[`puma_cross_validation` example](python/examples/puma_cross_validation.py)
and its matching notebook demonstrate configuration → native analysis → VTI
export → direct `pumapy` analysis → comparison. TANGLE does not ship a
`tangle.interop.puma` adapter.
The voxelizer currently runs on the host and accepts circular sections in
orthorhombic cells. It validates geometry structure and grid compatibility,
not solver convergence. See the [analysis guide](../../docs/puma_interoperability.md)
for exact definitions, supported comparisons, and current limitations.

## Examples

- `crossed_fibers.py` / `crossed_fibers.ipynb` run the smallest complete
  Python-to-CubeCL calculation and write OVITO and BPM outputs.
- `adaptive_crossing.py` / `adaptive_crossing.ipynb` start with one segment per
  fiber and demonstrate contact-driven subdivision.
- `layered_recipe.py` / `layered_recipe.ipynb` generate three collections in
  Python and insert and relax them in sequence.
- `puma_cross_validation.py` / `puma_cross_validation.ipynb` compare native
  centerline metrics with an independently imported PuMA voxel workspace.
- `native/` mirrors the complete Rust example suite using the same native
  generators and solver. See `native/README.md` for the configuration map.
- `ct_examples.py` fits Tangle fibers to synthetic CT scans with `tangle.ct`
  and scores them against the truth. See the
  [CT fitting guide](../../docs/ct_fitting.md) and the
  [results, with pictures](python/examples/ct_results/README.md).

## Topic reference notebooks

See [`python/tutorials/README.md`](python/tutorials/README.md) for the ordered
curriculum. These notebooks are deliberately small and concept-oriented; use
them to understand or customize one part of an application before moving to
the full manufacturing workflows.
