"""Generate the topic-oriented TANGLE tutorial notebooks.

The notebooks are committed artifacts so they open directly in Jupyter. Keep
their prose and code here so the set can be regenerated consistently after an
API change.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip(),
    }


def table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Setting | Meaning | Choices / units |", "| --- | --- | --- |"]
    lines.extend(f"| `{name}` | {meaning} | {choices} |" for name, meaning, choices in rows)
    return "\n".join(lines)


def write_notebook(name: str, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (TANGLE)",
                "language": "python",
                "name": "tangle",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / name).write_text(json.dumps(payload, indent=1) + "\n")


def settings_cells(
    class_name: str,
    variable: str,
    rows: list[tuple[str, str, str]],
    extra: str = "",
) -> list[dict]:
    names = [name for name, _, _ in rows]
    return [
        md(table(rows)),
        code(
            f"""
            # Read defaults from the compiled extension instead of duplicating
            # them in documentation that could become stale.
            {variable} = tangle.{class_name}()
            fields = {names!r}
            {{name: getattr({variable}, name) for name in fields}}
            """
        ),
        *([md(extra)] if extra else []),
    ]


RELAXATION_FIELDS = [
    ("backend", "CubeCL execution backend.", "`\"wgpu\"` or `\"cpu\"`"),
    ("motion_model", "How contact corrections move a fiber.", "`\"flexible\"` or `\"rigid_translation\"`"),
    ("pin_fiber_ends", "Prevents the first and last vertex from moving.", "boolean"),
    ("penetration_tolerance", "Largest accepted capsule overlap.", "length, m"),
    ("force_full_iterations", "Runs the entire iteration budget instead of accepting early.", "boolean"),
    ("correction_fraction", "Fraction of each contact correction applied per pass.", "dimensionless"),
    ("contact_aggregation", "Combines multiple corrections at a vertex.", "`uniform_average`, `penetration_weighted`, `deepest_only`"),
    ("stretch_stiffness", "Rest-length projection strength.", "dimensionless"),
    ("bend_stiffness", "Rest-shape bending projection strength.", "dimensionless"),
    ("curvature_limit_stiffness", "Admissible-curvature projection strength.", "dimensionless"),
    ("curvature_limit_safety_margin", "Keeps projected bends inside the hard limit.", "ratio"),
    ("curvature_ratio_tolerance", "Excess ratio allowed above one: accepted bend utilization is at most 1 + this tolerance.", "dimensionless excess"),
    ("constraint_iterations", "Constraint sweeps in each solver iteration.", "count"),
    ("curvature_cleanup_sweeps", "Extra hard-curvature projections per iteration.", "count"),
    ("max_step", "Maximum vertex displacement in one correction.", "length, m"),
    ("max_iterations", "Global relaxation iteration budget.", "count"),
    ("iterations_per_batch", "Iterations in one scheduler/device batch.", "count"),
    ("debug_snapshot_interval", "Periodic OVITO cadence; `None` keeps only keyframes when run() receives a debug path.", "iterations or `None`"),
    ("save_assembled_reference", "Stores the converged geometry as an assembled reference.", "boolean"),
    ("cell_list", "Broad-phase cell-list configuration.", "`CellListSettings`"),
    ("cell_size_scale", "Convenience alias for `cell_list.cell_size_scale`.", "at least 1"),
    ("adaptive_segmentation", "Optional adaptive refinement configuration.", "settings or `None`"),
]

ADAPTIVE_FIELDS = [
    ("contact_length_over_diameter", "Refine contacted segments longer than this scale.", "L/D ratio"),
    ("minimum_length_over_diameter", "Hard lower segment-length scale.", "L/D ratio"),
    ("maximum_refinement_levels", "Depth of the preallocated dyadic split tree.", "1–30"),
    ("refinement_interval", "Cadence for evaluating possible splits.", "iterations"),
    ("refinement_persistence", "Repeated contact observations required before splitting.", "passes"),
    ("coarsening_persistence", "Quiet observations required before merging siblings.", "passes"),
    ("coarsening_error_over_diameter", "Maximum midpoint/chord error allowed for a merge.", "error/D"),
    ("coarsening_curvature_ratio", "Maximum curvature utilization allowed for a merge.", "ratio"),
]

POPULATION_FIELDS = [
    ("count", "Number of fibers to generate.", "count"),
    ("segments_per_fiber", "Initial uniform centerline resolution.", "count"),
    ("seed", "Deterministic sampling seed.", "integer"),
    ("length_minimum", "Shortest sampled fiber.", "m"),
    ("length_maximum", "Longest sampled fiber.", "m"),
    ("nominal_parent_length", "Optional physical parent length for periodic fragments.", "m or `None`"),
    ("radius_minimum", "Smallest sampled radius.", "m"),
    ("radius_maximum", "Largest sampled radius.", "m"),
    ("curvature_amplitude_minimum", "Minimum generated waviness amplitude.", "m"),
    ("curvature_amplitude_maximum", "Maximum generated waviness amplitude.", "m"),
    ("orientation", "Orientation distribution.", "`isotropic_3d`, `planar`, `layered_biaxial`, `aligned`"),
    ("orientation_axis", "Normal or preferred direction used by the orientation model.", "unit-like xyz vector"),
    ("maximum_angle", "Angular support around an aligned direction.", "rad"),
    ("maximum_tilt", "Out-of-plane support for planar distributions.", "rad"),
    ("primary_fraction", "Layered-biaxial fraction near the primary direction.", "0–1"),
    ("cross_fraction", "Layered-biaxial fraction near the transverse direction.", "0–1"),
    ("maximum_in_plane_deviation", "Biaxial directional scatter.", "rad"),
    ("layer_orientation_seed", "Independent seed for the layer basis.", "integer"),
    ("position", "Center-position distribution.", "`uniform`, `layered`, `density_gradient`"),
    ("position_axis", "Axis used for layers or density gradients.", "0, 1, or 2"),
    ("layers", "Number of placement layers.", "count"),
    ("jitter_fraction", "Random layer-position jitter relative to spacing.", "fraction"),
    ("density_exponent", "Shape of a density-gradient distribution.", "positive exponent"),
    ("density_toward_high", "Chooses the high-coordinate side of the gradient.", "boolean"),
    ("minimum_bend_radius", "Optional admissible bend radius for generated fibers.", "m or `None`"),
    ("max_attempts_per_fiber", "Rejection-sampling budget per fiber.", "count"),
    ("material_name", "Material-table name assigned to the population.", "string"),
]

COMPACTION_FIELDS = [
    ("target_type", "Stopping observable.", "`volume_fraction`, `cell_volume`, `cell_lengths`, `mean_pressure`, `directional_pressure`, `penalty_energy`"),
    ("target_value", "Scalar target for scalar target types.", "target-dependent"),
    ("target_values", "Three-component target for vector target types.", "target-dependent xyz"),
    ("path", "Rule for distributing cell motion among axes.", "`axis_weights`, `equal_pressure`, `stress_ratio`, `minimum_incremental_work`"),
    ("axis_weights", "Prescribed relative shortening for the axis-weight path.", "nonnegative xyz"),
    ("active_axes", "Axes available to feedback-controlled paths.", "three booleans"),
    ("stress_ratio", "Desired directional pressure ratio.", "nonnegative xyz"),
    ("pressure_floor", "Numerical floor in pressure-ratio calculations.", "pressure"),
    ("kinematics", "How geometry follows cell changes.", "`rigid_fiber_centers`, `moving_walls`, `affine_vertices`"),
    ("cell_anchor", "Stationary fractional point while each cell axis changes.", "xyz in [0,1]"),
    ("balance_opposing_faces", "Balances work between low and high faces.", "boolean"),
    ("face_pressure_floor", "Floor used by opposing-face balancing.", "pressure"),
    ("face_balance_strength", "Strength of opposing-face feedback.", "0–1"),
    ("initial_log_strain", "First attempted logarithmic strain increment.", "positive strain"),
    ("minimum_log_strain", "Smallest retry increment.", "positive strain"),
    ("maximum_log_strain", "Largest grown increment.", "positive strain"),
    ("growth_factor", "Increment multiplier after easy accepted steps.", "greater than 1"),
    ("shrink_factor", "Increment multiplier after rejected steps.", "between 0 and 1"),
    ("relax_iterations", "Relaxation work allotted to each increment window.", "count"),
    ("maximum_shortening_over_minimum_diameter", "Caps an increment by the thinnest fiber size.", "ratio"),
    ("maximum_penetration", "Rejects a trial exceeding this overlap.", "m"),
    ("maximum_bend_ratio", "Rejects a trial exceeding this curvature utilization.", "ratio"),
    ("maximum_pressure", "Pressure guard.", "pressure"),
    ("maximum_penalty_energy", "Formation-energy guard.", "energy"),
    ("maximum_steps", "Maximum accepted/retried compaction steps.", "count"),
    ("maximum_relax_windows", "Maximum windows spent settling one trial.", "count"),
    ("contact_energy_stiffness", "Contact contribution to the formation penalty.", "model stiffness"),
    ("stretch_energy_stiffness", "Stretch contribution to the formation penalty.", "model stiffness"),
    ("bending_energy_stiffness", "Bending contribution to the formation penalty.", "model stiffness"),
    ("target_tolerance", "Relative/absolute acceptance tolerance for the target.", "target-dependent"),
]

JUNCTION_FIELDS = [
    ("name", "Human-readable capture-policy name.", "nonempty string"),
    ("law_name", "Symbolic downstream junction law.", "nonempty string"),
    ("parameter_set", "Junction-law parameter-table identifier.", "integer"),
    ("maximum_surface_gap", "Largest surface gap eligible for capture.", "m"),
    ("minimum_crossing_angle", "Smallest accepted unsigned crossing angle.", "rad, 0 to pi/2"),
    ("maximum_crossing_angle", "Largest accepted unsigned crossing angle.", "rad, 0 to pi/2"),
    ("probability", "Deterministic seeded thinning probability.", "0–1"),
    ("seed", "Seed for probabilistic capture.", "integer"),
    ("material_pairs", "Optional allowed material-name pairs.", "list of pairs"),
    ("maximum_per_fiber_pair", "Maximum anchors captured between one fiber pair.", "positive count"),
    ("minimum_anchor_separation", "Required material-coordinate separation between anchors.", "m"),
    ("candidate_capacity", "Device candidate-buffer capacity.", "positive count"),
]

CHECKPOINT_FIELDS = [
    ("case_id", "Identity used to reject an incompatible restart.", "nonempty string"),
    ("path", "Rolling checkpoint output path.", "filesystem path"),
    ("interval_iterations", "Sparse save cadence.", "positive iteration count"),
    ("resume", "Loads a checkpoint before executing.", "boolean"),
    ("resume_path", "Optional source path distinct from the new output path.", "path or `None`"),
    ("resume_case_id", "Optional expected identity for the source checkpoint.", "string or `None`"),
    ("fresh_formation_on_resume", "Keeps geometry/history but restarts recipe operations.", "boolean"),
]


NOTEBOOKS: dict[str, list[dict]] = {
    "00_installation_and_environment.ipynb": [
        md("""
        # Install TANGLE and configure Jupyter

        TANGLE is not a pure-Python package. Its Python API wraps the Rust
        implementation with PyO3, so installation has two parts: Cargo compiles
        the Rust workspace, then Maturin installs the resulting extension into
        a Python environment. Jupyter must use that same environment as its
        kernel.

        ## Installation plan

        1. Install Python 3.10+, stable Rust/Cargo, and a native compiler.
        2. Create a virtual environment or named Conda environment and open
           this notebook with that kernel.
        3. Install Maturin and Jupyter tooling into the active environment.
        4. Let Maturin invoke Cargo and install TANGLE.
        5. Register the reusable **Python (TANGLE)** kernel and verify the import.

        CubeCL's CPU backend uses LLVM internally, but Cargo downloads CubeCL's
        matching LLVM bundle automatically. No separate LLVM installation or
        `llvm-config` is required.
        """),
        md("""
        ## One-time prerequisite: install Rust and Cargo

        If `rustc --version` and `cargo --version` already work, skip this
        section. Otherwise install Rust with the official Rustup installer.
        Cargo is installed alongside Rust.

        On macOS, Linux, or Windows Subsystem for Linux:

        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source "$HOME/.cargo/env"
        rustup toolchain install stable
        rustup default stable
        rustc --version
        cargo --version
        ```

        On Windows, download and run the official
        [`rustup-init.exe`](https://rust-lang.org/tools/install/), accept the
        stable MSVC toolchain, then open a new PowerShell window:

        ```powershell
        rustup toolchain install stable
        rustup default stable
        rustc --version
        cargo --version
        ```

        Rust also needs a native linker: Xcode command-line tools on macOS,
        GCC/Clang development tools on Linux, or Visual Studio Build Tools with
        the Desktop C++ workload on Windows. The Windows Rustup installer may
        prompt for those build tools. Restart VS Code after installing Rust so
        its notebook kernels inherit the updated command path.

        ### Where Maturin is installed

        The setup commands below install Maturin with `pip` inside the active
        Python environment. This is intentional: the `maturin` executable then
        follows the selected Jupyter kernel and installs TANGLE into that same
        environment.

        Maturin can instead be installed as a Rust CLI with Cargo:

        ```bash
        cargo install --locked maturin
        maturin --version
        ```

        If you choose the Cargo installation, `maturin` is normally placed in
        Cargo's executable directory (usually `~/.cargo/bin`). You still need
        to activate `.venv-tangle` or the named Conda environment before
        running `maturin develop`. The self-running notebook cell deliberately
        uses the environment-local `pip` installation, so there is no need to
        run both installation methods.
        """),
        md("""
        ## Manual equivalent: POSIX shell

        These are the plain shell commands performed by the setup cell below.
        Run them from the TANGLE repository root if you prefer a terminal.

        ```bash
        python3 --version
        rustc --version
        cargo --version
        python3 -m venv .venv-tangle
        source .venv-tangle/bin/activate
        python -m pip install --upgrade pip
        python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
        maturin develop --release --manifest-path crates/tangle_python/Cargo.toml
        python -m ipykernel install --user --name tangle --display-name "Python (TANGLE)"
        python -c "import tangle; print(tangle.__file__)"
        ```
        """),
        md("""
        ## Manual equivalent: Windows PowerShell

        ```powershell
        py -3 --version
        rustc --version
        cargo --version
        py -3 -m venv .venv-tangle
        .\\.venv-tangle\\Scripts\\Activate.ps1
        python -m pip install --upgrade pip
        python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
        maturin develop --release --manifest-path crates/tangle_python/Cargo.toml
        python -m ipykernel install --user --name tangle --display-name "Python (TANGLE)"
        python -c "import tangle; print(tangle.__file__)"
        ```

        These commands assume the one-time Rust/native-compiler prerequisite
        above has already been completed.
        """),
        md("""
        ## Miniforge / Conda alternative

        Miniforge can manage the Python environment instead of `venv`. It does
        not replace Rust/Cargo or the platform's native compiler. After
        [installing Miniforge](https://github.com/conda-forge/miniforge), open
        a Conda-enabled terminal and run these commands from the TANGLE
        repository root:

        ```bash
        conda create --name tangle python=3.12 -y
        conda activate tangle
        python -m pip install --upgrade pip
        python -m pip install "maturin>=1.8,<2" jupyterlab ipykernel
        maturin develop --release --manifest-path crates/tangle_python/Cargo.toml
        python -m ipykernel install --user --name tangle --display-name "Python (TANGLE)"
        python -c "import tangle; print(tangle.__file__)"
        ```

        Use a named environment rather than installing TANGLE into Conda's
        `base` environment. In VS Code or Jupyter, select **Python (TANGLE)**
        after registration. If Rust was installed while VS Code was open,
        restart VS Code before building so its kernel can find Cargo.
        """),
        md("""
        ## Run the installation from this notebook

        The notebook must already use a virtual-environment or named Conda
        kernel; a running kernel cannot replace its own Python interpreter. The
        cell finds the repository root, checks Rust/Cargo, installs the Python
        build tools, configures Maturin for the active environment, compiles
        TANGLE, and registers this exact interpreter as **Python (TANGLE)**.
        """),
        code("""
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        # A compiled extension must be installed into the same environment as
        # this running kernel; installing into system Python will not help it.
        active_virtualenv = bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != sys.base_prefix
        active_conda = bool(os.environ.get("CONDA_PREFIX")) or (
            Path(sys.prefix) / "conda-meta"
        ).is_dir()
        if not active_virtualenv and not active_conda:
            raise RuntimeError(
                "This kernel is not using a virtual environment or Conda "
                "environment. Create one with the commands above, then select "
                "its Python kernel."
            )
        if active_conda and os.environ.get("CONDA_DEFAULT_ENV") == "base":
            raise RuntimeError(
                "This kernel is using Conda's base environment. Create and "
                "activate the named 'tangle' environment shown above."
            )

        # Walk upward so this notebook works from Jupyter or VS Code without
        # assuming that either application chose the repository as its CWD.
        def find_repository_root(start: Path) -> Path:
            for candidate in (start, *start.parents):
                manifest = candidate / "crates/tangle_python/Cargo.toml"
                if manifest.is_file():
                    return candidate
            raise RuntimeError(
                "Could not find crates/tangle_python/Cargo.toml above "
                f"{start}. Open this notebook from the TANGLE checkout."
            )

        repository = find_repository_root(Path.cwd().resolve())
        print("Repository:", repository)
        print("Environment:", sys.prefix)

        # Maturin invokes Cargo, so fail early with a useful message when the
        # one-time Rust installation is missing from this kernel's PATH.
        for executable in ("rustc", "cargo"):
            if shutil.which(executable) is None:
                raise RuntimeError(
                    f"{executable} is not available. Install stable Rust with "
                    "rustup, restart the shell/VS Code, and try again."
                )

        def run(command: list[str], *, environment=None) -> None:
            print("\\n$", " ".join(command), flush=True)
            subprocess.run(
                command,
                cwd=repository,
                env=environment,
                check=True,
            )

        # Use this interpreter's pip to keep Maturin and the notebook kernel in
        # the same environment.
        run([
            sys.executable, "-m", "pip", "install", "--upgrade", "--quiet",
            "pip", "maturin>=1.8,<2", "jupyterlab", "ipykernel",
        ])

        # Windows and POSIX environments place console scripts differently.
        maturin_name = "maturin.exe" if os.name == "nt" else "maturin"
        maturin_candidates = (
            Path(sys.executable).parent / maturin_name,
            Path(sys.prefix) / "Scripts" / maturin_name,
            Path(sys.prefix) / "bin" / maturin_name,
        )
        maturin = next((path for path in maturin_candidates if path.is_file()), None)
        if maturin is None:
            raise RuntimeError(
                f"Maturin was installed but {maturin_name} was not found under {sys.prefix}."
            )

        # Remove stale venv/Conda markers before selecting this kernel's Python
        # explicitly for PyO3. This prevents builds from landing in another env.
        scripts = maturin.parent
        build_environment = os.environ.copy()
        if active_conda and not os.environ.get("VIRTUAL_ENV"):
            build_environment.pop("VIRTUAL_ENV", None)
            build_environment["CONDA_PREFIX"] = sys.prefix
        else:
            build_environment.pop("CONDA_PREFIX", None)
            build_environment.pop("CONDA_DEFAULT_ENV", None)
            build_environment["VIRTUAL_ENV"] = sys.prefix
        build_environment["PYO3_PYTHON"] = sys.executable
        build_environment["PATH"] = (
            str(scripts) + os.pathsep + build_environment.get("PATH", "")
        )

        # `develop` creates an editable install, so rebuilding updates this
        # checkout without copying Python sources into site-packages.
        run([
            str(maturin), "develop", "--release", "--manifest-path",
            "crates/tangle_python/Cargo.toml",
        ], environment=build_environment)
        # Register a stable display name that both Jupyter and VS Code can use.
        run([
            sys.executable, "-m", "ipykernel", "install", "--user",
            "--name", "tangle", "--display-name", "Python (TANGLE)",
        ])

        print("\\nInstallation complete. Restart this notebook kernel, then run verification.")
        """),
        md("""
        ## Verify the active kernel

        Restart the notebook kernel after the installation cell completes, then
        run this cell. Restarting is necessary because Python reads Maturin's
        editable-install path file when the kernel starts.
        """),
        code("""
        import importlib.util
        import os
        import platform
        import shutil
        import sys
        from pathlib import Path

        # Print both the interpreter and build tools because the most common
        # import failure is a notebook attached to the wrong Python kernel.
        print("Python:", sys.executable)
        print("Version:", sys.version.split()[0])
        print("Platform:", platform.platform())
        for executable in ("rustc", "cargo"):
            print(f"{executable:12s}", shutil.which(executable) or "NOT FOUND")
        maturin_name = "maturin.exe" if os.name == "nt" else "maturin"
        maturin_candidates = (
            Path(sys.executable).parent / maturin_name,
            Path(sys.prefix) / "Scripts" / maturin_name,
            Path(sys.prefix) / "bin" / maturin_name,
        )
        maturin = shutil.which("maturin") or next(
            (str(path) for path in maturin_candidates if path.is_file()),
            "NOT FOUND",
        )
        print(f"{'maturin':12s}", maturin)
        print(f"{'environment':12s}", os.environ.get("CONDA_DEFAULT_ENV") or sys.prefix)

        # `find_spec` checks visibility without importing the native module, so
        # the resulting error can still explain how to repair the environment.
        if importlib.util.find_spec("tangle") is None:
            raise RuntimeError(
                "TANGLE is not visible. Restart this kernel after installation "
                "and confirm that it uses the same Python environment."
            )
        import tangle
        print("Loaded:", tangle.__file__)
        print("Smoke test:", tangle.Cell([1e-3, 1e-3, 1e-3]).lengths)
        """),
        md("""
        ## What happened?

        - Cargo resolved and compiled the Rust workspace, CubeCL kernels, and
          the automatically downloaded CPU backend dependencies.
        - Maturin built the PyO3 extension and installed it as editable package
          `tangle` in the active Python environment.
        - IPykernel registered that interpreter for VS Code and JupyterLab.

        ## Rebuild after Rust binding changes

        Re-run the installation cell after changing Rust binding code, then
        restart the notebook kernel. Cargo reuses unchanged build artifacts.
        Pure notebook or Python-file changes do not require rebuilding.
        """),
    ],
    "01_high_level_overview.ipynb": [
        md("""
        # TANGLE at a glance

        TANGLE builds a fiber network by approximating manufacturing as an
        ordered, quasi-static recipe. Python describes the geometry and recipe;
        Rust validates and packs them; CubeCL keeps the active world resident on
        a CPU or GPU while contact, mechanics, and topology evolve.

        ```text
        Cell + materials + fiber collections
                         ↓
              ordered manufacturing recipe
                         ↓
          CubeCL relaxation + adaptive topology
                         ↓
             RunResult → OVITO / BPM
        ```

        This deliberately small two-ply example introduces the ideas developed
        individually in tutorials 02–14. All physical quantities use SI units.
        Building the objects is cheap; the guarded execution cell is the only
        part that launches the solver.
        """),
        code("""
        import inspect
        from pathlib import Path

        import tangle

        # Catch a stale compiled extension before a long recipe reaches a
        # keyword added by a newer notebook.
        run_parameters = inspect.signature(tangle.Recipe.run).parameters
        if "debug_ovito_path" not in run_parameters:
            raise RuntimeError(
                "This kernel has an older compiled TANGLE extension loaded. "
                "Run the installation cell in tutorial 00, restart the kernel, "
                "select Python (TANGLE), and then run this notebook from the top."
            )

        # Recipe construction is cheap; this switch controls the actual solve
        # and all filesystem output.
        RUN_OVERVIEW = False
        output = Path("output/overview")
        """),
        md("""
        ## 1. Describe the domain and fibers

        The cell is periodic in-plane and bounded through-thickness. A material
        supplies capsule diameter and an optional admissible bend radius. A
        collection holds placed centerlines, preferred rest centerlines, and
        metadata before anything enters the simulation.

        Here the top large fiber is placed slightly curved but has a straight
        rest centerline: relaxation treats that initial curvature as bending.
        Bulk and biased collections can instead be made with TANGLE's seeded
        fiber generators.
        """),
        code("""
        # Axis order is x, y, z. Periodic x/y represent a repeating sheet;
        # bounded z retains physical top and bottom surfaces.
        cell = tangle.Cell(
            [1.0e-3, 1.0e-3, 1.5e-3],
            periodic=[True, True, False],
        )
        # Diameter controls contact geometry. minimum_bend_radius is the hard
        # admissible-curvature scale, not the preferred rest shape.
        small = tangle.Material(
            "small", diameter=7.0e-6, minimum_bend_radius=35.0e-6
        )
        large = tangle.Material(
            "large", diameter=19.0e-6, minimum_bend_radius=75.0e-6
        )

        # Collections are detached local geometry. formation_layer metadata
        # lets later recipe operations address manufacturing plies.
        bottom = tangle.FiberCollection("bottom ply")
        bottom.add_fiber(
            [[-0.35e-3, -0.10e-3, 0.0], [0.35e-3, -0.10e-3, 0.0]],
            small,
            tags={"family": "x"},
            formation_layer=0,
        )
        bottom.add_fiber(
            [[0.12e-3, -0.35e-3, 0.0], [0.12e-3, 0.35e-3, 0.0]],
            small,
            tags={"family": "y"},
            formation_layer=0,
        )

        top = tangle.FiberCollection("top ply")
        # `placed` is the literal initial geometry. Giving it a straight rest
        # shape means the visible waviness initially stores bending strain.
        placed = [
            [-0.35e-3, 0.0, 0.0],
            [0.0, 12.0e-6, 0.0],
            [0.35e-3, 0.0, 0.0],
        ]
        straight_rest = [
            [-0.35e-3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.35e-3, 0.0, 0.0],
        ]
        top.add_fiber(
            placed,
            large,
            rest_centerline=straight_rest,
            tags={"family": "needle-eligible"},
            formation_layer=1,
        )
        top.add_fiber(
            [[-0.08e-3, -0.35e-3, 0.0], [-0.08e-3, 0.35e-3, 0.0]],
            small,
            tags={"family": "y"},
            formation_layer=1,
        )

        print(len(bottom), len(top), bottom.layers(), top.layers())
        """),
        md("""
        ## 2. Write the manufacturing recipe

        A recipe is an ordered operation list, not a time integrator. It can
        insert dormant collections, establish temporary kinematic targets,
        relax, release targets, compact the cell, and capture persistent
        junction topology. Explicit relaxation gates make the intended sequence
        auditable before the expensive run begins.

        This example settles the bottom ply, inserts and lowers the top ply,
        performs one visible needling displacement, releases the needle,
        compacts through-thickness, performs a strict final relaxation, and
        finally records selected contacts as junctions.
        """),
        code("""
        # Axis indices are 0=x, 1=y, 2=z. Here z is the stacking or
        # through-thickness direction used by all layer-aware operations.
        recipe = tangle.Recipe(cell, layer_axis=2)

        # Insert and settle the first ply before activating the second.
        recipe.insert(bottom, translation=[0.5e-3, 0.5e-3, 0.35e-3])
        recipe.relax(maximum_iterations=2_000)

        recipe.insert(top, translation=[0.5e-3, 0.5e-3, 0.80e-3])
        recipe.place_layer_above(
            layer=1, gap=5.0e-6, stiffness=0.5, max_translation=5.0e-6
        )

        # Penetration is hard during assembly, while bend cleanup is temporarily
        # soft so manufacturing motion can finish before the final strict pass.
        assembly_policy = tangle.SolvePolicy(
            "contact-first assembly",
            solver_penetration=0.2e-6,
            solver_curvature_ratio=2.0,
            acceptance_penetration=0.3e-6,
            penetration_enforcement="hard",
            acceptance_curvature_ratio=2.0,
            curvature_enforcement="soft",
            maximum_iterations=4_000,
            on_exhaustion="reject",
        )
        assembly_overrides = tangle.RelaxationOverrides()
        assembly_overrides.bend_stiffness = 0.2
        recipe.relax_with_policy(assembly_policy, assembly_overrides)
        recipe.release_layer_targets()

        # The needle pulls eligible large-fiber vertices; neighboring fibers
        # move only when contact transmits that displacement.
        recipe.needle_layer_circular(
            layer=1,
            center=[0.5e-3, 0.5e-3],
            diameter=120.0e-6,
            depth=0.25e-3,
            minimum_fiber_diameter=15.0e-6,
            stiffness=0.75,
            max_translation=5.0e-6,
            maximum_translation_over_fiber_diameter=0.5,
        )
        recipe.relax_until_targets_reached(0.2e-6, 3_000)
        recipe.release_needles()

        # Only z may shrink because the axis weights are [x=0, y=0, z=1].
        compaction = tangle.CompactionSettings.volume_fraction(
            0.002, axis_weights=[0.0, 0.0, 1.0]
        )
        compaction.maximum_steps = 20
        compaction.maximum_relax_windows = 4
        recipe.compact(compaction)
        recipe.relax(maximum_iterations=4_000)

        # Ordinary contacts remain transient until this explicit late capture.
        junctions = tangle.JunctionPolicy("late contact capture", "bond")
        junctions.maximum_surface_gap = 0.2e-6
        junctions.probability = 1.0
        junctions.material_pairs = [("large", "small")]
        recipe.capture_junctions(junctions)
        """),
        md("""
        ## 3. Configure one resident solve

        Global relaxation settings control contact resolution, fiber mechanics,
        batching, backend selection, and adaptive refinement/coarsening. Recipe
        policies above may temporarily override a few of them for one stage.
        A checkpoint can preserve the resident formation state and operation
        cursor for continuation.

        The CPU backend executes the same CubeCL kernels without a GPU. Change
        `backend` to `"wgpu"` for a supported accelerator.
        """),
        code("""
        # These are global defaults; a recipe policy may temporarily override
        # selected values for one manufacturing stage.
        settings = tangle.RelaxationSettings()
        settings.backend = "cpu"
        settings.motion_model = "flexible"
        settings.penetration_tolerance = 0.1e-6
        settings.curvature_ratio_tolerance = 1.05
        settings.max_step = 2.0e-6
        # Manufacturing targets are updated between batches. A short batch
        # keeps this small target-heavy recipe responsive without controlling
        # how many OVITO frames are retained.
        settings.iterations_per_batch = 6
        settings.adaptive_segmentation = (
            tangle.AdaptiveSegmentationSettings.profile("balanced")
        )

        # The restart stores the recipe cursor and resident solver state, not
        # merely a final list of downloaded centerlines.
        checkpoint = tangle.CheckpointSettings(
            "tutorial-overview",
            output / "overview.restart",
            interval_iterations=500,
        )

        print("Recipe operations:")
        for number, operation in enumerate(recipe.operations(), start=1):
            print(f"{number:2d}. {operation}")
        """),
        md("""
        ## 4. Run, inspect, and export

        `Recipe.run()` uploads the packed world, executes the ordered recipe,
        and returns final geometry plus convergence, topology, transfer, event,
        checkpoint, and junction reports. OVITO output visualizes the relaxed
        spherocylinders; BPM output converts them into a bonded-particle model
        that downstream solvers such as DIRT can consume. Passing
        `debug_ovito_path` with no `debug_snapshot_interval` records concise
        recipe keyframes. Set an integer interval only when detailed relaxation
        frames are needed.

        Change `RUN_OVERVIEW` to `True` only when you want to execute the solve.
        """),
        code("""
        if RUN_OVERVIEW:
            # With no snapshot interval, OVITO records recipe milestones only.
            output.mkdir(parents=True, exist_ok=True)
            debug_dump = output / "overview_debug.dump"
            debug_view = output / "overview_debug_view.py"
            debug_session = output / "overview_debug.ovito"
            result = recipe.run(
                settings,
                checkpoint=checkpoint,
                debug_ovito_path=debug_dump,
                debug_ovito_view_script_path=debug_view,
                debug_ovito_session_path=debug_session,
                debug_ovito_coloring="curvature_ratio",
            )

            summary = {
                "converged": result.converged,
                "iterations": result.iterations,
                "max_penetration_um": result.max_penetration * 1.0e6,
                "max_curvature_ratio": result.max_curvature_ratio,
                "active_segments": result.active_segments,
                "splits": result.segment_splits,
                "merges": result.segment_merges,
                "junctions": result.junction_count,
            }
            print(summary)
            print(f"Debug OVITO: {debug_dump} ({result.debug_ovito_frames} frames)")
            print(f"OVITO view recipe: {debug_view}")
            print(f"OVITO session target: {debug_session}")

            # Keep a separate one-frame file for inspecting only the final state.
            result.write_ovito(
                output / "overview_final.dump",
                view_script_path=output / "overview_final_view.py",
                session_path=output / "overview_final.ovito",
                coloring="curvature_ratio",
            )
            # Exact spherocylinders preserve the final active segmentation.
            result.export_bpm(
                output / "overview.data",
                mode="spherocylinders-exact",
                density=1_800.0,
            )
        """),
        md("""
        ## Where each idea goes next

        | Tutorial | Focus |
        | --- | --- |
        | 02 | Cells, periodic axes, and hard walls |
        | 03 | Materials, placed geometry, and rest geometry |
        | 04 | Detached fiber collections and metadata |
        | 05 | Seeded fiber population generation and directional bias |
        | 06 | Insertion, selections, and rigid transforms |
        | 07 | Relaxation, contact, mechanics, and CubeCL backends |
        | 08 | Adaptive refinement and coarsening |
        | 09 | Per-stage solve policies and temporary overrides |
        | 10 | Layer movement, target release, and needling |
        | 11 | Dynamic compaction and wall control |
        | 12 | Explicit junction capture |
        | 13 | Checkpoints, continuation, and branching |
        | 14 | Results, OVITO visualization, and BPM export |
        """),
    ],
    "02_cells_and_boundaries.ipynb": [
        md("""
        # Cells and boundary conditions

        A `Cell` defines the orthorhombic simulation domain. Lengths and origins
        are SI meters. Each periodic flag independently selects periodic
        minimum-image contact (`True`) or a bounded hard-wall axis (`False`).
        """),
        code("""
        import tangle

        # Omitting `periodic` gives three bounded axes with the origin at zero.
        bounded = tangle.Cell([1e-3, 2e-3, 3e-3])
        # Flags map to [x, y, z]. This sheet repeats in-plane but has bounded
        # through-thickness surfaces; the origin centers the x/y coordinates.
        periodic_sheet = tangle.Cell(
            [1e-3, 1e-3, 2e-3],
            periodic=[True, True, False],
            origin=[-0.5e-3, -0.5e-3, 0.0],
        )
        {
            "lengths": periodic_sheet.lengths,
            "periodic": periodic_sheet.periodic,
            "origin": periodic_sheet.origin,
        }
        """),
        md("""
        ## Boundary interpretation

        - Periodic axes wrap cell-list neighborhoods and use minimum-image
          capsule separation. They are useful for representative in-plane areas.
        - Bounded axes enforce planar walls. Use these for exposed surfaces or
          through-thickness compaction.
        - Cell coordinates describe placed geometry only; rest centerlines are
          intrinsic material data and need not lie inside the cell.
        """),
    ],
    "03_materials_and_centerlines.ipynb": [
        md("""
        # Materials, placed centerlines, and rest centerlines

        `diameter` controls capsule contact. `minimum_bend_radius` is an
        admissibility limit, not the preferred shape. A fiber's placed
        centerline is its initial simulation geometry; its rest centerline is
        used to derive rest lengths and rest turning angles.
        """),
        code("""
        import tangle

        # A missing minimum_bend_radius means there is no material curvature
        # admissibility limit, while diameter still defines contact radius.
        straight = tangle.Material("straight fiber", diameter=19e-6)
        bend_limited = tangle.Material(
            "bend-limited fiber",
            diameter=7e-6,
            minimum_bend_radius=35e-6,
        )
        print(straight.name, straight.diameter, straight.radius)
        print(bend_limited.minimum_bend_radius)
        """),
        md("""
        ## The three useful shape cases

        1. Straight placed + straight rest: initially unbent.
        2. Curved placed + straight rest: initially bent and driven toward a line.
        3. Curved placed + matching curved rest: the curvature is natural.

        The admissible bend radius is checked independently in every case.
        """),
        code("""
        # Placed coordinates are absolute simulation positions.
        placed = [[0.2e-3, 0.2e-3, 0.5e-3],
                  [0.5e-3, 0.3e-3, 0.5e-3],
                  [0.8e-3, 0.2e-3, 0.5e-3]]
        straight_rest = [[0.0, 0.0, 0.0],
                         [0.3e-3, 0.0, 0.0],
                         [0.6e-3, 0.0, 0.0]]

        fibers = tangle.FiberCollection("shape semantics")
        # Same placed geometry, different rest geometry: the first fiber wants
        # to straighten; the second considers its initial curve stress-free.
        fibers.add_fiber(placed, bend_limited, rest_centerline=straight_rest)
        fibers.add_fiber(placed, bend_limited, rest_centerline=placed)
        list(zip(fibers.centerlines(), fibers.rest_centerlines()))
        """),
    ],
    "04_fiber_collections.ipynb": [
        md("""
        # Fiber collections

        A `FiberCollection` is detached geometry that can be generated, loaded,
        transformed, combined, and later inserted into a recipe. It is allowed
        to contain overlaps; relaxation happens after insertion.
        """),
        code("""
        import tangle

        material = tangle.Material("fiber", diameter=10e-6)
        # Collection coordinates are local until Recipe.insert applies a rigid
        # transform. Overlap is allowed at this stage.
        ply = tangle.FiberCollection("ply 2")
        fiber_index = ply.add_fiber(
            [[0.0, 0.0, 0.0], [0.5e-3, 0.0, 0.0]],
            material,
            rest_centerline=[[0.0, 0.0, 0.0], [0.5e-3, 0.0, 0.0]],
            # Tags remain descriptive metadata; formation_layer participates in
            # later layer-aware recipe operations.
            tags={"family": "machine-direction", "source": "measured"},
            formation_layer=2,
        )
        print(fiber_index, len(ply), ply.layers())
        """),
        md("""
        ## Construction and selection operations

        - `add_fiber(...)` controls placed/rest centerlines, material, tags, and layer.
        - `from_centerlines(...)` assigns one material and optional layer in bulk.
        - `extend(...)` concatenates detached collections.
        - `select_layer(...)` creates a collection containing one formation layer.
        - `centerlines()` and `rest_centerlines()` return ordinary Python lists.
        """),
        code("""
        # Bulk construction is convenient when one material/layer applies to
        # many imported centerlines.
        transverse = tangle.FiberCollection.from_centerlines(
            [[[0.0, 0.0, 0.0], [0.0, 0.5e-3, 0.0]]],
            material,
            name="transverse",
            formation_layer=3,
        )
        all_fibers = tangle.FiberCollection("two plies")
        all_fibers.extend(ply)
        all_fibers.extend(transverse)
        # Selection returns another detached collection; it does not mutate the
        # combined source collection.
        selected = all_fibers.select_layer(2, name="only ply 2")
        print(len(all_fibers), len(selected), all_fibers.layers())
        """),
    ],
    "05_fiber_generation.ipynb": [
        md("""
        # Built-in fiber generators

        TANGLE provides small crossing generators for mechanics tests and a
        configurable population generator for material recipes. Generation is
        seeded and deterministic; relaxation is tolerance-deterministic rather
        than promised bitwise-identical across parallel backends.
        """),
        code("""
        import tangle

        cell = tangle.Cell([1e-3, 1e-3, 1e-3])
        # Point crossings are the minimal rigid-contact demonstration.
        point_crossing = tangle.generate_point_crossing(
            cell, count=8, length=0.8e-3, radius=9.5e-6,
            material_name="large", name="center crossing",
        )
        # Distinct placed/rest shape controls create initially bent fibers.
        curved_crossing = tangle.generate_multisegment_crossing(
            cell, count=4, segments_per_fiber=8, length=0.8e-3,
            placed_chord_fraction=0.8, radius=9.5e-6,
            rest_shape="straight", rest_amplitude=0.0,
            placed_shape="curved", placed_amplitude=0.1e-3,
            minimum_bend_radius=50e-6,
        )
        # A two-fiber pair is useful for isolated refinement/contact tests.
        pair = tangle.generate_fiber_pair_crossing(
            cell, segments_per_fiber=1, length=0.8e-3,
            radius=9.5e-6, axis_separation=10e-6,
            crossing_angle_degrees=90.0,
        )
        [len(point_crossing), len(curved_crossing), len(pair)]
        """),
        md("## Every `FiberPopulationSettings` field"),
        *settings_cells("FiberPopulationSettings", "population", POPULATION_FIELDS),
        code("""
        # Geometry ranges are sampled independently but reproducibly from seed.
        population.count = 100
        population.seed = 42
        population.length_minimum = 0.2e-3
        population.length_maximum = 0.3e-3
        population.radius_minimum = 3.5e-6
        population.radius_maximum = 9.5e-6
        population.curvature_amplitude_minimum = 0.0
        population.curvature_amplitude_maximum = 10e-6
        population.minimum_bend_radius = 35e-6
        # layered_biaxial places 40% near a primary in-plane direction, 40%
        # near its transverse direction, and leaves the remaining 20% random.
        population.orientation = "layered_biaxial"
        # For planar/biaxial distributions, orientation_axis is the plane normal.
        population.orientation_axis = [0.0, 0.0, 1.0]
        population.primary_fraction = 0.4
        population.cross_fraction = 0.4
        # Position axis 2 is z, so this creates four through-thickness plies.
        population.position = "layered"
        population.position_axis = 2
        population.layers = 4
        generated = tangle.generate_fiber_population(cell, population, name="four plies")
        print(len(generated), generated.layers())
        """),
    ],
    "06_insertion_and_transforms.ipynb": [
        md("""
        # Insertion, selections, and rigid transforms

        `Recipe.insert()` records a manufacturing activation operation. The
        collection is transformed once, packed into the eventual device world,
        and remains dormant until its operation is reached.
        """),
        code("""
        import tangle

        # This collection is authored around a local origin and is not yet in
        # the periodic simulation cell.
        cell = tangle.Cell([1e-3, 1e-3, 2e-3], periodic=[True, True, False])
        material = tangle.Material("fiber", diameter=19e-6)
        collection = tangle.FiberCollection.from_centerlines(
            [[[-0.3e-3, 0.0, 0.0], [0.3e-3, 0.0, 0.0]]],
            material,
            name="local-coordinate ply",
            formation_layer=0,
        )
        # 0=x, 1=y, 2=z; z is the layer stacking direction here.
        recipe = tangle.Recipe(cell, layer_axis=2)
        # Rotation is applied first, then translation places the rotated fiber.
        selection = recipe.insert(
            collection,
            name="placed ply",
            translation=[0.5e-3, 0.5e-3, 0.4e-3],
            rotation=[[0.0, -1.0, 0.0],
                      [1.0,  0.0, 0.0],
                      [0.0,  0.0, 1.0]],
        )
        print(selection.name, selection.fiber_ids, selection.formation_step)
        """),
        md("""
        `name` labels the returned `FiberSelection`; `translation` is in meters;
        and `rotation` is a 3×3 matrix applied before translation. Selections are
        stable handles for reporting and future selection-scoped APIs. Use
        `recipe.operations()` to audit ordering before a costly run and
        `recipe.centerlines()` to inspect the packed initial geometry.
        """),
        code("""
        # Methods append ordered operations; no solver launches until run().
        recipe.relax(maximum_iterations=2_000)
        print(*recipe.operations(), sep="\\n")
        recipe.centerlines()
        """),
    ],
    "07_relaxation_settings.ipynb": [
        md("""
        # Relaxation and contact settings

        Relaxation is a quasi-static sequence of contact, stretch, rest-bend,
        and admissible-curvature projections. Convergence requires both the
        penetration and curvature residuals to satisfy their configured limits.
        """),
        code("# Settings objects expose the same defaults used by Rust.\nimport tangle"),
        md("## Every `RelaxationSettings` field"),
        *settings_cells("RelaxationSettings", "settings", RELAXATION_FIELDS),
        md("""
        ## Cell-list broad phase

        `cell_size_scale` is the only exposed cell-list setting. A value of 1
        uses the minimum admissible cell size; larger cells reduce cell count
        but increase candidates per cell. It must be at least 1.
        """),
        code("""
        # Broad-phase cells must be at least one contact diameter wide. Larger
        # values trade fewer cells for more candidate capsule pairs per cell.
        settings.cell_list = tangle.CellListSettings(cell_size_scale=1.25)
        # The alias edits the same nested value.
        settings.cell_size_scale = 1.5
        settings.to_dict()
        """),
        md("""
        Use `motion_model="rigid_translation"` for straight rigid fibers and
        `"flexible"` for vertex-level deformation. The rigid mode also preserves
        curved centerlines; it applies translation only, not rotation.
        `backend="wgpu"` is the
        accelerator path; `"cpu"` runs the same CubeCL kernels through the
        native multithreaded CPU runtime.
        """),
    ],
    "08_adaptive_refinement.ipynb": [
        md("""
        # Adaptive segment refinement and coarsening

        Fibers may begin as one segment and activate preallocated dyadic child
        segments only where persistent contact demands resolution. Quiet,
        nearly straight sibling segments can later merge. Topology changes stay
        inside the resident CubeCL world.
        """),
        code("# Adaptive topology is configured through relaxation settings.\nimport tangle"),
        md("## Every `AdaptiveSegmentationSettings` field"),
        *settings_cells("AdaptiveSegmentationSettings", "adaptive", ADAPTIVE_FIELDS),
        code("""
        # Profiles are editable starting points rather than locked modes.
        profiles = {
            name: tangle.AdaptiveSegmentationSettings.profile(name).to_dict()
            for name in ("fast", "balanced", "strict")
        }
        profiles
        """),
        md("""
        Refinement cadence is independent of scheduler batch size. Persistence
        avoids splitting on a single transient overlap. Coarsening preserves
        pinned, targeted, substantially bent, and junction-bearing topology.
        """),
        code("""
        # Assigning None disables adaptation; the convenience methods install
        # or remove a default settings object explicitly.
        settings = tangle.RelaxationSettings()
        settings.adaptive_segmentation = adaptive
        settings.disable_adaptive_segmentation()
        assert settings.adaptive_segmentation is None
        settings.enable_adaptive_segmentation()
        settings.adaptive_segmentation.to_dict()
        """),
    ],
    "09_solve_policies_and_overrides.ipynb": [
        md("""
        # Recipe relaxation gates, solve policies, and overrides

        Global `RelaxationSettings` describe the solver. A `SolvePolicy`
        describes what one recipe gate must achieve, while
        `RelaxationOverrides` temporarily changes how that gate approaches its
        target. This supports contact-first assembly followed by strict final
        bending cleanup.
        """),
        code("# Policies control acceptance; overrides control the path there.\nimport tangle"),
        md(table([
            ("name", "Stage label used in reports.", "string"),
            ("solver_penetration", "Residual the stage solver works toward.", "m"),
            ("solver_curvature_ratio", "Curvature target used by the stage solver.", "ratio"),
            ("acceptance_penetration", "Residual required to accept the operation.", "m"),
            ("penetration_enforcement", "Whether failure rejects the stage.", "`hard` or `soft`"),
            ("acceptance_curvature_ratio", "Curvature ratio required for acceptance.", "ratio"),
            ("curvature_enforcement", "Whether curvature failure rejects the stage.", "`hard` or `soft`"),
            ("maximum_iterations", "Stage-specific iteration budget.", "count"),
            ("on_exhaustion", "Behavior when the budget is exhausted.", "`reject` or `continue`"),
        ])),
        code("""
        # This assembly stage must resolve contact but temporarily accepts a
        # curvature ratio up to 5 so bending cannot block deposition.
        contact_first = tangle.SolvePolicy(
            "contact-first settling",
            solver_penetration=0.1e-6,
            solver_curvature_ratio=5.0,
            acceptance_penetration=0.2e-6,
            penetration_enforcement="hard",
            acceptance_curvature_ratio=5.0,
            curvature_enforcement="soft",
            maximum_iterations=10_000,
            on_exhaustion="reject",
        )
        """),
        md(table([
            ("motion_model", "Temporary fiber motion model.", "string or `None`"),
            ("correction_fraction", "Temporary contact correction fraction.", "float or `None`"),
            ("contact_aggregation", "Temporary contact aggregation rule.", "string or `None`"),
            ("stretch_stiffness", "Temporary rest-length stiffness.", "float or `None`"),
            ("bend_stiffness", "Temporary rest-bend stiffness.", "float or `None`"),
            ("curvature_limit_stiffness", "Temporary hard-curvature stiffness.", "float or `None`"),
            ("constraint_iterations", "Temporary constraint sweep count.", "integer or `None`"),
            ("curvature_cleanup_sweeps", "Temporary curvature cleanup count.", "integer or `None`"),
        ])),
        code("""
        # Overrides apply only to relax_with_policy; global settings return for
        # later operations.
        overrides = tangle.RelaxationOverrides()
        overrides.bend_stiffness = 0.1
        overrides.curvature_limit_stiffness = 1.0
        overrides.contact_aggregation = "deepest_only"

        recipe = tangle.Recipe(tangle.Cell([1e-3, 1e-3, 1e-3]))
        # These calls represent distinct amounts or acceptance conditions.
        recipe.relax_for(100)  # fixed work; no acceptance gate
        recipe.relax(maximum_iterations=2_000)  # default hard gate
        recipe.relax_with_policy(contact_first, overrides)
        recipe.relax_until_targets_reached(0.1e-6, 5_000)
        recipe.set_material_bend_radius("fiber", 50e-6)
        print(*recipe.operations(), sep="\\n")
        """),
    ],
    "10_layer_motion_and_needling.ipynb": [
        md("""
        # Layer placement and needling operations

        These recipe commands approximate manufacturing by applying temporary
        device-resident targets. They must be separated by explicit relaxation
        operations. Releasing a target lets the displaced fibers and their
        contacts settle mechanically.
        """),
        code("""
        import tangle

        recipe = tangle.Recipe(
            tangle.Cell([1e-3, 1e-3, 3e-3], periodic=[True, True, False]),
            # 0=x, 1=y, 2=z; z is the layer and needle-motion direction.
            layer_axis=2,
        )
        """),
        md("""
        ## Layer commands

        - `move_layers(spacing_scale, stiffness, max_translation)` scales all
          current layer-center spacings.
        - `place_layer_above(layer, gap, ...)` brings one layer to a surface gap
          above the active stack.
        - `release_layer_targets()` removes those temporary clamps.
        - `fit_cell_to_active_fibers(axes, padding)` removes empty domain space.
        """),
        code("""
        # move_layers scales current layer-center spacing; it does not teleport
        # fibers or bypass contact relaxation.
        recipe.move_layers(0.8, stiffness=0.5, max_translation=5e-6)
        recipe.relax_for(500)
        recipe.place_layer_above(2, gap=2e-6, stiffness=0.5, max_translation=5e-6)
        recipe.relax_until_targets_reached(0.1e-6, 2_000)
        # Releasing temporary clamps lets later stages move the stack freely.
        recipe.release_layer_targets()
        recipe.fit_cell_to_active_fibers(
            axes=[False, False, True], padding=25e-6
        )
        """),
        md("""
        ## Needle commands

        `needle_layer_circular` selects one internal vertex from each eligible
        fiber intersecting a circular footprint. `needle_layer_random` samples
        eligible fibers by fraction. Both accept depth, optional minimum fiber
        diameter, target stiffness, absolute maximum motion, and a motion cap
        relative to the selected fiber diameter. Needle locations can be made
        random by sampling `center` in Python before adding the operation.
        """),
        code("""
        # Circular needling selects one internal vertex per eligible fiber in
        # the x/y footprint because z is the configured layer axis.
        recipe.needle_layer_circular(
            layer=2, center=[0.45e-3, 0.55e-3], diameter=100e-6,
            depth=0.6e-3, minimum_fiber_diameter=15e-6,
            stiffness=0.75, max_translation=5e-6,
            maximum_translation_over_fiber_diameter=0.5,
        )
        recipe.relax_until_targets_reached(0.1e-6, 5_000)
        # Hold the target through relaxation, then release it before continuing.
        recipe.release_needles()

        # Random needling samples fibers reproducibly from seed rather than by
        # spatial footprint.
        recipe.needle_layer_random(
            layer=3, fraction=0.15, depth=0.6e-3, seed=2026,
            minimum_fiber_diameter=15e-6, stiffness=0.75,
            max_translation=5e-6,
            maximum_translation_over_fiber_diameter=0.5,
        )
        recipe.release_needles()
        print(*recipe.operations(), sep="\\n")
        """),
    ],
    "11_compaction.ipynb": [
        md("""
        # Dynamic compaction

        `CompactionSettings` controls a closed-loop series of transactional cell
        changes. Each trial changes the cell, relaxes, checks guards, and is
        accepted or rolled back before the next increment.
        """),
        code("# Compaction is configured independently from the recipe using it.\nimport tangle"),
        md("## Every `CompactionSettings` field"),
        *settings_cells("CompactionSettings", "compaction", COMPACTION_FIELDS),
        md("""
        ## Common target/path combinations

        `volume_fraction()` is the concise constructor. Other stopping targets
        are selected by changing `target_type` and the scalar `target_value` or
        vector `target_values`. The path is independent of the target.
        """),
        code("""
        # axis_weights map to [x, y, z]; only the bounded z direction shortens.
        compaction = tangle.CompactionSettings.volume_fraction(
            0.40, axis_weights=[0.0, 0.0, 1.0]
        )
        compaction.kinematics = "moving_walls"
        compaction.cell_anchor = [0.5, 0.5, 0.5]  # both z faces move
        compaction.balance_opposing_faces = True
        compaction.maximum_penetration = 0.1e-6
        compaction.maximum_bend_ratio = 1.05

        # Copies make it easy to compare paths without rebuilding every guard.
        equal_pressure = compaction.copy()
        equal_pressure.path = "equal_pressure"
        equal_pressure.active_axes = [True, True, True]

        target_lengths = compaction.copy()
        target_lengths.target_type = "cell_lengths"
        target_lengths.target_values = [0.8e-3, 0.8e-3, 1.2e-3]
        """),
        code("""
        # Stage overrides affect only this compaction operation's relaxation.
        recipe = tangle.Recipe(tangle.Cell([1e-3, 1e-3, 2e-3]))
        stage_overrides = tangle.RelaxationOverrides()
        stage_overrides.contact_aggregation = "deepest_only"
        recipe.compact(compaction, stage_overrides)
        recipe.operations()
        """),
    ],
    "12_junction_capture.ipynb": [
        md("""
        # Persistent junction capture

        Ordinary contacts are transient. A junction exists only when a recipe
        explicitly promotes an eligible contact, typically after the geometry
        is substantially relaxed. Anchors use material coordinates so they
        survive adaptive remeshing.
        """),
        code("# Junction angles are expressed in radians.\nimport math\nimport tangle"),
        md("## Every `JunctionPolicy` field"),
        *settings_cells("JunctionPolicy", "policy", JUNCTION_FIELDS),
        code("""
        # The law name and parameter set are downstream labels; capture filters
        # decide which current contacts receive persistent material anchors.
        policy.name = "cured binder contacts"
        policy.law_name = "cohesive bond"
        policy.parameter_set = 2
        policy.maximum_surface_gap = 0.2e-6
        policy.minimum_crossing_angle = math.radians(20)
        policy.maximum_crossing_angle = math.pi / 2
        # Probability is deterministic for a fixed seed and candidate set.
        policy.probability = 0.25
        policy.seed = 9
        policy.material_pairs = [("large", "small"), ("large", "large")]
        policy.maximum_per_fiber_pair = 1
        policy.minimum_anchor_separation = 50e-6
        policy.candidate_capacity = 100_000
        """),
        md("""
        `capture_junctions(policy)` samples once at that recipe point.
        `relax_and_capture(iterations, every, policy)` samples repeatedly at an
        explicit cadence independent of scheduler batch size. Captured
        junctions are topology/export data; current relaxation does not enforce
        their mechanics, so late capture is normally the physically appropriate
        workflow.
        """),
        code("""
        recipe = tangle.Recipe(tangle.Cell([1e-3, 1e-3, 1e-3]))
        # Late capture records bonds after geometry has settled; contacts before
        # this explicit operation remain transient.
        recipe.relax(maximum_iterations=5_000)
        recipe.capture_junctions(policy)
        # Alternative repeated capture:
        # recipe.relax_and_capture(iterations=2_000, every=250, policy=policy)
        recipe.operations()
        """),
    ],
    "13_checkpoints_and_resume.ipynb": [
        md("""
        # Checkpoints and restart

        A rolling checkpoint atomically stores recipe position, partial
        operation state, device geometry, active topology, refinement history,
        cell bounds, and active targets. Scratch contact buffers are rebuilt on
        resume. Cross-revision compatibility is not guaranteed: archive the
        generating code revision, settings, backend, and an independent geometry
        export. Use a new output path when branching a cleanup study.
        """),
        code("# Path objects are accepted anywhere TANGLE expects a file path.\nfrom pathlib import Path\nimport tangle"),
        md("## Every `CheckpointSettings` field"),
        md(table(CHECKPOINT_FIELDS)),
        code("""
        # case_id prevents accidentally resuming an unrelated recipe that used
        # the same filesystem location.
        checkpoint = tangle.CheckpointSettings(
            "twenty-ply-needled-v1",
            Path("output/twenty_ply.restart"),
            interval_iterations=500,
            resume=False,
        )
        fields = [
            "case_id", "path", "interval_iterations", "resume",
            "resume_path", "resume_case_id", "fresh_formation_on_resume",
        ]
        {name: getattr(checkpoint, name) for name in fields}
        """),
        md("""
        Set `resume=True` to continue the same rolling file. To branch a saved
        state into a new experiment, use a new `path` and point `resume_path`
        at the old file. `resume_case_id` validates the source identity.
        `fresh_formation_on_resume=True` retains geometry/history but starts the
        supplied recipe from operation zero.
        """),
        code("""
        # Branching reads the old checkpoint but writes future progress to a new
        # rolling file, leaving the source restart untouched.
        branch = checkpoint.copy()
        branch.path = Path("output/alternate_cleanup.restart")
        branch.resume = True
        branch.resume_path = checkpoint.path
        branch.resume_case_id = checkpoint.case_id
        branch.fresh_formation_on_resume = True

        # Expensive execution is deliberately opt-in in this reference notebook.
        RUN_RECIPE = False
        if RUN_RECIPE:
            recipe = tangle.Recipe(tangle.Cell([1e-3, 1e-3, 1e-3]))
            result = recipe.run(tangle.RelaxationSettings(), checkpoint=branch)
        """),
    ],
    "14_results_and_exports.ipynb": [
        md("""
        # Run results, native analysis, OVITO, BPM, and PuMA export

        `Recipe.run()` returns geometry plus scalar solver, topology, transfer,
        event, checkpoint, and junction reports. Final geometry is downloaded
        once; ordinary batches download only compact status values unless debug
        snapshots were requested. Check `converged`, residuals, warnings, and
        recipe events before claiming a relaxed specimen. Export methods do
        not independently certify mechanical admissibility.
        """),
        code("# Export methods accept pathlib paths as well as strings.\nfrom pathlib import Path\nimport tangle"),
        md(table([
            ("fiber_count", "Fibers in the assembly.", "count"),
            ("iterations", "Lifetime solver iteration.", "count"),
            ("converged", "Whether the final acceptance criteria passed.", "boolean"),
            ("max_penetration", "Largest final capsule overlap.", "m"),
            ("max_curvature_ratio", "Largest final admissible-curvature utilization.", "ratio"),
            ("active_segments", "Segments active after adaptation.", "count"),
            ("active_vertices", "Vertices active after adaptation.", "count"),
            ("segment_splits", "Accepted adaptive splits.", "count"),
            ("segment_merges", "Accepted adaptive merges.", "count"),
            ("refinement_passes", "Refinement evaluations.", "count"),
            ("coarsening_passes", "Coarsening evaluations.", "count"),
            ("uploaded_bytes", "Host-to-device transfer volume.", "bytes"),
            ("downloaded_bytes", "Device-to-host transfer volume.", "bytes"),
            ("cell_count", "Broad-phase cells in the final grid.", "count"),
            ("events", "Human-readable recipe/solver events.", "list of strings"),
            ("warnings", "Nonfatal numerical warnings.", "list of strings"),
            ("junction_captures", "Reports from explicit capture operations.", "list of strings"),
            ("resumed", "Whether this run loaded a checkpoint.", "boolean"),
            ("resumed_iteration", "Iteration restored from a checkpoint.", "count or `None`"),
            ("checkpoint_saves", "Checkpoints written by this invocation.", "count"),
            ("last_checkpoint_iteration", "Iteration of the last save.", "count or `None`"),
            ("last_checkpoint_bytes", "Serialized size of the last save.", "bytes or `None`"),
            ("debug_ovito_frames", "Sparse debug trajectory frames written during the run.", "count"),
            ("junction_count", "Persistent junctions in the result.", "count"),
        ])),
        code("""
        # Build a minimal intersecting pair so the result contains meaningful
        # contact and curvature diagnostics.
        cell = tangle.Cell([1e-3, 1e-3, 1e-3])
        fibers = tangle.generate_fiber_pair_crossing(cell)
        recipe = tangle.Recipe(cell)
        recipe.insert(fibers)
        recipe.relax(maximum_iterations=2_000)

        # Keep reference notebooks safe to execute top-to-bottom by default.
        RUN_SOLVER = False
        if RUN_SOLVER:
            settings = tangle.RelaxationSettings()
            result = recipe.run(settings)
            print(result.events)
            print(result.warnings)
            final_centerlines = result.centerlines()
        """),
        md("""
        ## OVITO output

        `coloring` accepts `"fiber"`, `"curvature_ratio"`, or
        `"refinement_level"`. The optional view script and OVITO session make
        oriented spherocylinder rendering reproducible.
        """),
        code("""
        if RUN_SOLVER:
            # The generated view script restores oriented spherocylinder shape
            # and coloring when the raw dump is opened in OVITO.
            output = Path("output")
            output.mkdir(exist_ok=True)
            result.write_ovito(
                output / "result.dump",
                view_script_path=output / "result_view.py",
                session_path=output / "result.ovito",
                coloring="curvature_ratio",
            )
        """),
        md("""
        ## BPM export

        `export_bpm` writes bonded spheres or bonded spherocylinders without
        owning a downstream solver's runtime configuration or constitutive
        laws. The four modes separate exact centerline sampling from
        endpoint-preserving or constant-resolution sampling:

        - `spheres-exact` keeps the requested arc spacing exactly and shares
          unused length between the two fiber ends.
        - `spheres-dynamic` preserves every source segment endpoint and adjusts
          the spacing inside each segment.
        - `spherocylinders-exact` writes one capsule per active segment.
        - `spherocylinders-constant` uses the shortest active segment length
          throughout, with a shorter remainder only where required.

        Sphere center spacing is specified in radii and must lie from `1/3`
        through `1`. `density`, `atom_type`, and `bond_type` provide the
        material and topology metadata required by BPM consumers such as DIRT.
        """),
        code("""
        if RUN_SOLVER:
            # Dynamic spheres retain every TANGLE segment endpoint; actual bond
            # spacing may shift locally around the requested one-third radius.
            particles, bonds = result.export_bpm(
                output / "result.data",
                mode="spheres-dynamic",
                sphere_spacing_over_radius=1.0 / 3.0,
                density=1800.0,
                atom_type=1,
                bond_type=1,
            )
            print(particles, bonds)
        """),
        md("""
        ## Native characterization

        `characterize()` calls the Rust `tangle_characterize` crate on the
        exact centerlines and sections. It reports nominal swept volume rather
        than the geometric union of solids, so the field is deliberately named
        `nominal_swept_volume_fraction`. The complete versioned report is
        available as a dictionary, JSON string, or JSON file.
        """),
        code("""
        if RUN_SOLVER:
            # Length weighting treats centerline equally; volume weighting is
            # the appropriate voxel comparison when diameters differ.
            analysis = result.characterize()
            print(analysis.length_weighted_orientation_tensor)
            print(analysis.volume_weighted_orientation_tensor)
            analysis.write_json(output / "tangle_analysis.json")
        """),
        md("""
        ## PuMA-compatible voxel bundle

        `export_puma()` also dispatches to Rust. It writes cell-centered phase
        IDs and exact tangents to `domain.vti`, optional fiber-ID and smooth
        interface images, a manifest, and native analysis JSON. PuMA stays an
        independent optional application: import `pumapy` directly in the
        analysis environment rather than through a TANGLE adapter. The current
        voxelizer runs on the host and supports circular sections in diagonal
        orthorhombic cells. It validates assembly structure and grid settings,
        not residual penetration or final solver acceptance.
        """),
        code("""
        if RUN_SOLVER:
            # The cubic voxel size must tile all three orthorhombic cell edges.
            puma_bundle = result.export_puma(
                output / "result.puma",
                voxel_size=20e-6,
                include_fiber_ids=True,
                include_interface=True,
            )
            print(puma_bundle.domain_path)
            print(puma_bundle.voxel_volume_fraction)
            print(puma_bundle.ambiguous_voxels)
        """),
    ],
}


if __name__ == "__main__":
    for filename, cells in NOTEBOOKS.items():
        write_notebook(filename, cells)
    print(f"wrote {len(NOTEBOOKS)} notebooks to {HERE}")
