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


def table(
    rows: list[tuple[str, str, str]],
    header: tuple[str, str, str] = ("Setting", "Meaning", "Choices / units"),
) -> str:
    lines = ["| " + " | ".join(header) + " |", "| --- | --- | --- |"]
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
    ("curvature_ratio_tolerance", "Excess ratio allowed above one: the accepted curvature ratio is at most 1 + this tolerance.", "dimensionless excess"),
    ("constraint_iterations", "Constraint sweeps in each solver iteration.", "count"),
    ("curvature_cleanup_sweeps", "Extra hard-curvature projections per iteration.", "count"),
    ("max_step", "Maximum vertex displacement in one correction.", "length, m"),
    ("max_iterations", "Global relaxation iteration budget.", "count"),
    ("iterations_per_batch", "Iterations in one scheduler/device batch.", "count"),
    ("debug_snapshot_interval", "Periodic OVITO cadence; `None` keeps only keyframes when run() receives a debug path.", "iterations or `None`"),
    ("save_assembled_reference", "Stores the converged geometry as an assembled reference.", "boolean"),
    ("cell_size_scale", "Broad-phase cell size relative to the smallest admissible cell.", "at least 1"),
    ("neighbor_skin_scale", "Extra Verlet neighbor-list search distance, as a multiple of the largest fiber radius.", "finite, at least 0"),
    ("neighbor_capacity", "Neighbor-list length per segment; overflowing segments fall back to scanning cells.", "count, at least 1"),
    ("adaptive_segmentation", "Optional adaptive refinement configuration.", "`AdaptiveSegmentationSettings` or `None`"),
]

ADAPTIVE_FIELDS = [
    ("contact_length_over_diameter", "Refine contacted segments longer than this scale.", "L/D ratio"),
    ("min_length_over_diameter", "Hard lower segment-length scale.", "L/D ratio"),
    ("max_refinement_levels", "Depth of the preallocated dyadic split tree.", "1–30"),
    ("refinement_interval", "Cadence for evaluating possible splits.", "iterations"),
    ("refinement_persistence", "Repeated contact observations required before splitting.", "passes"),
    ("coarsening_persistence", "Quiet observations required before merging siblings.", "passes"),
    ("coarsening_error_over_diameter", "Maximum midpoint/chord error allowed for a merge.", "error/D"),
    ("coarsening_curvature_ratio", "Maximum curvature utilization allowed for a merge.", "ratio"),
]

POPULATION_FIELDS = [
    ("material", "Material assigned to every generated fiber (name, contact diameter, bend limit).", "`Material`"),
    ("count", "Number of fibers to generate.", "count"),
    ("segments_per_fiber", "Initial uniform centerline resolution.", "count"),
    ("seed", "Deterministic sampling seed.", "integer"),
    ("length", "Fiber length: one value, or a `(min, max)` uniform range.", "m or `(m, m)`"),
    ("diameter", "Sampled contact diameter; `None` uses the material's diameter.", "m, `(m, m)`, or `None`"),
    ("curvature_amplitude", "Generated waviness amplitude.", "m or `(m, m)`"),
    ("nominal_parent_length", "Optional physical parent length for periodic fragments.", "m or `None`"),
    ("orientation", "Orientation distribution object (see the next table).", "`IsotropicOrientation`, `PlanarOrientation`, `LayeredBiaxialOrientation`, `AlignedOrientation`"),
    ("position", "Center-position distribution object (see below).", "`UniformPosition`, `LayeredPosition`, `DensityGradientPosition`"),
    ("max_attempts_per_fiber", "Rejection-sampling budget per fiber.", "count"),
]

ORIENTATION_VARIANTS = [
    ("IsotropicOrientation()", "Uniform directions on the sphere.", "no options"),
    ("PlanarOrientation(normal=None, max_tilt=...)", "Directions near the plane with this normal; `normal=None` uses the cell's stack axis.", "`max_tilt` in rad"),
    ("LayeredBiaxialOrientation(primary_fraction=, cross_fraction=, max_in_plane_deviation=, max_tilt=, seed=)", "A primary in-plane direction, its transverse direction, and a random remainder, per layer.", "fractions 0–1, angles in rad"),
    ("AlignedOrientation(axis, max_angle=...)", "Directions within a cone around `axis`.", "axis letter/index or xyz vector; rad"),
]

POSITION_VARIANTS = [
    ("UniformPosition()", "Centers uniform in the cell.", "no options"),
    ("LayeredPosition(layer_count, axis=None, jitter_fraction=...)", "Centers on evenly spaced planes; each plane becomes a `formation_layer`.", "count; axis defaults to the stack axis; jitter relative to spacing"),
    ("DensityGradientPosition(axis=None, exponent=..., toward_high=...)", "Center density rises along one axis.", "positive exponent; boolean"),
]

COMPACTION_FIELDS = [
    ("target", "Stopping observable, as a target object (see the next table).", "`VolumeFractionTarget`, `CellVolumeTarget`, `CellLengthsTarget`, `MeanPressureTarget`, `DirectionalPressureTarget`, `PenaltyEnergyTarget`"),
    ("path", "Rule for distributing cell motion among axes (see below).", "`AxisWeightsPath`, `EqualPressurePath`, `StressRatioPath`, `MinimumWorkPath`"),
    ("kinematics", "How geometry follows cell changes.", "`rigid_fiber_centers`, `moving_walls`, `affine_vertices`"),
    ("cell_anchor", "Stationary fractional point while each cell axis changes.", "xyz in [0,1]"),
    ("balance_opposing_faces", "Balances work between low and high faces.", "boolean"),
    ("face_pressure_floor", "Floor used by opposing-face balancing.", "pressure"),
    ("face_balance_strength", "Strength of opposing-face feedback.", "0–1"),
    ("initial_log_strain", "First attempted logarithmic strain increment.", "positive strain"),
    ("min_log_strain", "Smallest retry increment.", "positive strain"),
    ("max_log_strain", "Largest grown increment.", "positive strain"),
    ("growth_factor", "Increment multiplier after easy accepted steps.", "greater than 1"),
    ("shrink_factor", "Increment multiplier after rejected steps.", "between 0 and 1"),
    ("relax_iterations", "Relaxation work allotted to each increment window.", "count"),
    ("max_shortening_over_min_diameter", "Caps an increment by the thinnest fiber size.", "ratio"),
    ("max_penetration", "Rejects a trial exceeding this overlap.", "m"),
    ("max_curvature_ratio", "Rejects a trial exceeding this curvature utilization.", "ratio"),
    ("max_pressure", "Pressure guard.", "pressure"),
    ("max_penalty_energy", "Formation-energy guard.", "energy"),
    ("max_steps", "Maximum accepted/retried compaction steps.", "count"),
    ("max_relax_windows", "Maximum windows spent settling one trial.", "count"),
    ("contact_energy_stiffness", "Contact contribution to the formation penalty.", "model stiffness"),
    ("stretch_energy_stiffness", "Stretch contribution to the formation penalty.", "model stiffness"),
    ("bending_energy_stiffness", "Bending contribution to the formation penalty.", "model stiffness"),
    ("target_tolerance", "Relative/absolute acceptance tolerance for the target.", "target-dependent"),
]

COMPACTION_TARGETS = [
    ("VolumeFractionTarget(value)", "Stop at a nominal fiber volume fraction.", "0–1"),
    ("CellVolumeTarget(value)", "Stop at a cell volume.", "m³"),
    ("CellLengthsTarget(lengths)", "Stop at three cell edge lengths.", "xyz, m"),
    ("MeanPressureTarget(value)", "Stop at a mean wall pressure.", "pressure"),
    ("DirectionalPressureTarget(pressures)", "Stop at per-axis wall pressures.", "xyz pressure"),
    ("PenaltyEnergyTarget(value)", "Stop at a formation-penalty energy.", "energy"),
]

COMPACTION_PATHS = [
    ("AxisWeightsPath(weights=None)", "Prescribed relative shortening; `None` shortens only the stack axis.", "`\"z\"`, `\"xy\"`, or nonnegative xyz weights"),
    ("EqualPressurePath(axes, pressure_floor=...)", "Feedback that equalizes pressure on the active axes.", "axis set such as `\"xyz\"`"),
    ("StressRatioPath(ratio, pressure_floor=...)", "Feedback toward a directional pressure ratio.", "nonnegative xyz"),
    ("MinimumWorkPath(axes)", "Moves whichever active axis needs the least incremental work.", "axis set such as `\"xy\"`"),
]

JUNCTION_FIELDS = [
    ("name", "Human-readable capture-policy name.", "nonempty string"),
    ("law_name", "Symbolic downstream junction law.", "nonempty string"),
    ("parameter_set", "Junction-law parameter-table identifier.", "integer"),
    ("max_surface_gap", "Largest surface gap eligible for capture.", "m"),
    ("min_crossing_angle", "Smallest accepted unsigned crossing angle.", "rad, 0 to pi/2"),
    ("max_crossing_angle", "Largest accepted unsigned crossing angle.", "rad, 0 to pi/2"),
    ("probability", "Deterministic seeded thinning probability.", "0–1"),
    ("seed", "Seed for probabilistic capture.", "integer"),
    ("material_pairs", "Optional allowed material-name pairs.", "list of pairs"),
    ("max_per_fiber_pair", "Maximum anchors captured between one fiber pair.", "positive count"),
    ("min_anchor_separation", "Required material-coordinate separation between anchors.", "m"),
    ("candidate_capacity", "Device candidate-buffer capacity.", "positive count"),
]

SOLVE_POLICY_FIELDS = [
    ("name", "Stage label used in reports and errors.", "string"),
    ("target_penetration", "Penetration residual the stage solver works toward.", "m"),
    ("target_curvature_ratio", "Curvature ratio the stage solver works toward.", "ratio"),
    ("max_penetration", "Penetration required to accept the stage; defaults to `target_penetration`.", "m"),
    ("max_curvature_ratio", "Curvature ratio required to accept the stage; defaults to `target_curvature_ratio`.", "ratio"),
    ("hard_penetration", "Whether a penetration miss fails the stage (`True`) or is only reported.", "boolean"),
    ("hard_curvature", "Whether a curvature miss fails the stage (`True`) or is only reported.", "boolean"),
    ("max_iterations", "Stage-specific iteration budget.", "count"),
    ("max_extra_iterations", "Extra iterations allowed after the budget while the stage would still fail; stops as soon as it can pass. Defaults to half of `max_iterations`.", "count"),
    ("on_budget_exhausted", "Behavior when the budget runs out before the targets are met.", "`\"fail\"` or `\"continue_if_hard_ok\"`"),
]

OVERRIDE_FIELDS = [
    ("motion_model", "Temporary fiber motion model.", "string or `None`"),
    ("correction_fraction", "Temporary contact correction fraction.", "float or `None`"),
    ("contact_aggregation", "Temporary contact aggregation rule.", "string or `None`"),
    ("stretch_stiffness", "Temporary rest-length stiffness.", "float or `None`"),
    ("bend_stiffness", "Temporary rest-bend stiffness.", "float or `None`"),
    ("curvature_limit_stiffness", "Temporary hard-curvature stiffness.", "float or `None`"),
    ("constraint_iterations", "Temporary constraint sweep count.", "integer or `None`"),
    ("curvature_cleanup_sweeps", "Temporary curvature cleanup count.", "integer or `None`"),
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
        individually in tutorials 02–14. All lengths are in meters;
        `tangle.units` provides `um`, `mm`, and `nm` multipliers so that
        `7 * um` reads as seven micrometers. Building the objects is cheap; the
        guarded execution cell is the only part that launches the solver.
        """),
        code("""
        import inspect
        from pathlib import Path

        import tangle
        from tangle.units import mm, um

        # Catch a stale compiled extension before a long recipe reaches a
        # name or keyword added by a newer notebook.
        run_parameters = inspect.signature(tangle.Recipe.run).parameters
        if not hasattr(tangle, "RecipeError") or "debug_ovito_path" not in run_parameters:
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

        The cell is periodic in-plane and bounded through-thickness. Because z
        is its only bounded axis, z becomes the cell's *stack axis*: the
        direction that layer placement, needling, and compaction act along. A
        material supplies capsule diameter and an optional admissible bend
        radius. A collection holds placed centerlines, preferred rest
        centerlines, and metadata before anything enters the simulation.

        Here the top large fiber is placed slightly curved but has a straight
        rest centerline: relaxation treats that initial curvature as bending.
        Bulk and biased collections can instead be made with TANGLE's seeded
        fiber generators.
        """),
        code("""
        # Axis order is x, y, z. periodic="xy" represents a repeating sheet;
        # bounded z retains physical top and bottom surfaces.
        cell = tangle.Cell([1.0 * mm, 1.0 * mm, 1.5 * mm], periodic="xy")
        # Diameter controls contact geometry. min_bend_radius is the hard
        # admissible-curvature scale, not the preferred rest shape.
        small = tangle.Material("small", diameter=7 * um, min_bend_radius=35 * um)
        large = tangle.Material("large", diameter=19 * um, min_bend_radius=75 * um)

        # Collections are detached local geometry. formation_layer metadata
        # lets later recipe operations address manufacturing plies.
        bottom = tangle.FiberCollection("bottom ply")
        bottom.add_fiber(
            [[-0.35 * mm, -0.10 * mm, 0.0], [0.35 * mm, -0.10 * mm, 0.0]],
            small,
            tags={"family": "x"},
            formation_layer=0,
        )
        bottom.add_fiber(
            [[0.12 * mm, -0.35 * mm, 0.0], [0.12 * mm, 0.35 * mm, 0.0]],
            small,
            tags={"family": "y"},
            formation_layer=0,
        )

        top = tangle.FiberCollection("top ply")
        # `placed` is the literal initial geometry. Giving it a straight rest
        # shape means the visible waviness initially stores bending strain.
        placed = [
            [-0.35 * mm, 0.0, 0.0],
            [0.0, 12 * um, 0.0],
            [0.35 * mm, 0.0, 0.0],
        ]
        straight_rest = [
            [-0.35 * mm, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.35 * mm, 0.0, 0.0],
        ]
        top.add_fiber(
            placed,
            large,
            rest_centerline=straight_rest,
            tags={"family": "needle-eligible"},
            formation_layer=1,
        )
        top.add_fiber(
            [[-0.08 * mm, -0.35 * mm, 0.0], [-0.08 * mm, 0.35 * mm, 0.0]],
            small,
            tags={"family": "y"},
            formation_layer=1,
        )

        print(len(bottom), len(top), bottom.layer_ids(), top.layer_ids())
        print("stack axis:", cell.stack_axis)
        """),
        md("""
        ## 2. Write the manufacturing recipe

        A recipe is an ordered operation list, not a time integrator. It can
        insert dormant collections, hold fibers on temporary kinematic
        targets, relax, compact the cell, and capture persistent junction
        topology. Explicit relaxation gates make the intended sequence
        auditable before the expensive run begins.

        Operations that hold fibers on targets (layer placement and needling)
        return a `HeldTargets` handle. Used in a `with` block, it releases the
        targets when the block ends, so every hold is visibly paired with its
        release.

        This example settles the bottom ply, inserts and lowers the top ply,
        performs one visible needling displacement, compacts through-thickness,
        performs a strict final `solve`, and finally records selected contacts
        as junctions.
        """),
        code("""
        # The recipe inherits stack_axis=2 (z) from the cell's bounded axis.
        recipe = tangle.Recipe(cell)

        # Insert and settle the first ply before activating the second.
        recipe.insert(bottom, translation=[0.5 * mm, 0.5 * mm, 0.35 * mm])
        recipe.relax_until_converged(max_iterations=2_000)

        recipe.insert(top, translation=[0.5 * mm, 0.5 * mm, 0.80 * mm])

        # A SolvePolicy is the acceptance rule for one stage. Penetration is
        # hard during assembly, while curvature is only reported so bending
        # cannot block deposition before the final strict pass.
        assembly_policy = tangle.SolvePolicy(
            "contact-first assembly",
            target_penetration=0.2 * um,
            max_penetration=0.3 * um,
            target_curvature_ratio=2.0,
            hard_curvature=False,
            max_iterations=4_000,
        )
        # The with block holds layer 1 just above the stack and releases the
        # placement targets when it ends. The overrides soften rest bending
        # for this one stage only.
        with recipe.place_layer_above(1, gap=5 * um, stiffness=0.5, max_translation=5 * um):
            recipe.solve(assembly_policy, tangle.RelaxationOverrides(bend_stiffness=0.2))

        # The needle pulls eligible large-fiber vertices; neighboring fibers
        # move only when contact transmits that displacement.
        with recipe.needle_layer(
            1,
            footprint=tangle.CircularFootprint([0.5 * mm, 0.5 * mm], diameter=120 * um),
            depth=0.25 * mm,
            min_fiber_diameter=15 * um,
            stiffness=0.75,
            max_translation=5 * um,
            max_translation_over_diameter=0.5,
        ):
            recipe.settle_targets(tolerance=0.2 * um, max_iterations=3_000)

        # volume_fraction() shortens only the stack axis (z) by default; any
        # other CompactionSettings field can follow as a keyword.
        compaction = tangle.CompactionSettings.volume_fraction(
            0.002, max_steps=20, max_relax_windows=4
        )
        recipe.compact(compaction)

        # Strict final gate: both limits are hard and default to the targets.
        # The curvature_cleanup preset temporarily favors pulling bends back
        # inside the admissible limit.
        final_policy = tangle.SolvePolicy(
            "final",
            target_penetration=0.1 * um,
            target_curvature_ratio=1.02,
            max_iterations=4_000,
        )
        recipe.solve(final_policy, tangle.RelaxationOverrides.preset("curvature_cleanup"))

        # Ordinary contacts remain transient until this explicit late capture.
        junctions = tangle.JunctionPolicy(
            "late contact capture",
            "bond",
            max_surface_gap=0.2 * um,
            probability=1.0,
            material_pairs=[("large", "small")],
        )
        recipe.capture_junctions(junctions)
        """),
        md("""
        ## 3. Configure one resident solve

        Configuration objects follow three naming conventions:

        - `*Settings` are run-wide: `RelaxationSettings` and
          `CheckpointSettings` apply to every step of the recipe.
        - A `*Policy` is the named rule one step follows: the `SolvePolicy`
          and `JunctionPolicy` above.
        - `*Overrides` are temporary deltas for the step they are passed to,
          such as `RelaxationOverrides.preset("curvature_cleanup")`.

        Every configuration class takes keyword arguments and has
        `replace(**changes)` for a modified copy. The global relaxation
        settings control contact resolution, fiber mechanics, batching,
        backend selection, and adaptive refinement/coarsening. A checkpoint can
        preserve the resident formation state and operation cursor for
        continuation.

        The default `"wgpu"` backend runs on a supported GPU (Metal, Vulkan,
        or DirectX). Set `backend="cpu"` on a machine without one; it executes
        the same CubeCL kernels, but compiles each kernel through LLVM the first
        time a process uses it, which can take minutes, and then runs one to two
        orders of magnitude slower per iteration.
        """),
        code("""
        # These are global defaults; a recipe policy or override may adjust
        # selected values for one manufacturing stage.
        settings = tangle.RelaxationSettings(
            motion_model="flexible",
            penetration_tolerance=0.1 * um,
            # An excess above one: accept curvature ratios up to 1.05.
            curvature_ratio_tolerance=0.05,
            max_step=2 * um,
            # Manufacturing targets are updated between batches. A short batch
            # keeps this small target-heavy recipe responsive without
            # controlling how many OVITO frames are retained.
            iterations_per_batch=6,
            adaptive_segmentation=tangle.AdaptiveSegmentationSettings.profile("balanced"),
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
        and returns a `RunResult` with the final geometry plus convergence,
        topology, transfer, event, checkpoint, and junction reports. It does
        not modify its inputs: the relaxed geometry is a new `Assembly` at
        `result.assembly`. If a step cannot meet its policy, `run()` raises
        `tangle.RecipeError`, which names the failing operation and the reason.

        OVITO output visualizes the relaxed spherocylinders; BPM output
        converts them into a bonded-particle model that downstream solvers
        such as DIRT can consume. Passing `debug_ovito_path` with no
        `debug_snapshot_interval` records concise recipe keyframes. Set an
        integer interval only when detailed relaxation frames are needed.

        Change `RUN_OVERVIEW` to `True` only when you want to execute the solve.
        """),
        code("""
        if RUN_OVERVIEW:
            # With no snapshot interval, OVITO records recipe milestones only.
            output.mkdir(parents=True, exist_ok=True)
            debug_dump = output / "overview_debug.dump"
            debug_view = output / "overview_debug_view.py"
            debug_session = output / "overview_debug.ovito"
            try:
                result = recipe.run(
                    settings,
                    checkpoint=checkpoint,
                    debug_ovito_path=debug_dump,
                    debug_ovito_view_script_path=debug_view,
                    debug_ovito_session_path=debug_session,
                    debug_ovito_coloring="curvature_ratio",
                )
            except tangle.RecipeError as error:
                # The error identifies the recipe step, not just the symptom.
                print(f"Step {error.operation_index} ({error.operation}) failed: {error.reason}")
                raise

            summary = {
                "converged": result.converged,
                "iterations": result.iterations,
                "max_penetration_um": result.max_penetration / um,
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

            # run() leaves its inputs unchanged; the relaxed state is a new
            # Assembly that can seed a follow-on Recipe.
            relaxed = result.assembly
            print(relaxed.fiber_count, relaxed.cell.lengths)

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
                mode="spherocylinders_exact",
                density=1_800.0,
            )
        """),
        md("""
        ## Where each idea goes next

        | Tutorial | Focus |
        | --- | --- |
        | 02 | Cells, periodic axes, the stack axis, and units |
        | 03 | Materials, placed geometry, and rest geometry |
        | 04 | Detached fiber collections and metadata |
        | 05 | Seeded fiber populations, orientation, and position models |
        | 06 | Insertion, selections, and rigid transforms |
        | 07 | Relaxation, contact, mechanics, and CubeCL backends |
        | 08 | Adaptive refinement and coarsening |
        | 09 | Per-stage solve policies, override presets, and `RecipeError` |
        | 10 | Layer movement, held targets, and needling |
        | 11 | Dynamic compaction targets, paths, and guards |
        | 12 | Explicit junction capture |
        | 13 | Checkpoints, continuation, and branching |
        | 14 | Results, native analysis, contact and neighbor metrics, OVITO, BPM, and PuMA export |
        """),
    ],
    "02_cells_and_boundaries.ipynb": [
        md("""
        # Cells and boundary conditions

        A `Cell` defines the orthorhombic simulation domain. Lengths and origins
        are in meters; `tangle.units` supplies `um`, `mm`, and `nm` as plain
        float multipliers. `periodic` names the axes that use periodic
        minimum-image contact; every other axis is bounded by hard walls. It
        accepts a string of axis letters such as `"xy"`, a single axis, or an
        `[x, y, z]` bool triple.
        """),
        code("""
        import tangle
        from tangle.units import mm, nm, um

        # Units are ordinary floats in meters, so they compose with arithmetic.
        print(7 * um, 1.5 * mm, 250 * nm)

        # Omitting `periodic` gives three bounded axes with the origin at zero.
        bounded = tangle.Cell([1 * mm, 2 * mm, 3 * mm])
        # This sheet repeats in x and y but has bounded through-thickness
        # surfaces; the origin centers the x/y coordinates.
        periodic_sheet = tangle.Cell(
            [1 * mm, 1 * mm, 2 * mm],
            periodic="xy",
            origin=[-0.5 * mm, -0.5 * mm, 0.0],
        )
        {
            "lengths": periodic_sheet.lengths,
            "periodic": periodic_sheet.periodic,
            "origin": periodic_sheet.origin,
            "stack_axis": periodic_sheet.stack_axis,
        }
        """),
        md("""
        ## The stack axis

        Layer-aware operations (layered generation, layer placement, needling,
        and default compaction) act along one *stack axis*. The cell infers
        it: z when z is bounded or every axis is periodic, otherwise the last
        bounded axis. A `periodic="xy"` sheet stacks along z and a
        `periodic="xz"` wall along y. Pass `stack_axis=` to the `Cell` to
        choose another; `Recipe(cell, stack_axis=)` exists too, but generators
        follow the cell, so the `Cell` argument keeps everything consistent. Axes may be written as `"x"`, `"y"`, `"z"` or `0`,
        `1`, `2`; the property always reports the index.
        """),
        code("""
        # A bool triple is equivalent to the axis-letter string.
        same_sheet = tangle.Cell([1 * mm, 1 * mm, 2 * mm], periodic=[True, True, False])
        # y is the only bounded axis here, so it is inferred as the stack axis.
        wall_in_y = tangle.Cell([1 * mm, 2 * mm, 1 * mm], periodic="xz")
        # A fully periodic box has no bounded axis, so name the stack axis.
        bulk = tangle.Cell([1 * mm, 1 * mm, 1 * mm], periodic="xyz", stack_axis="x")
        {
            "same flags": same_sheet.periodic == periodic_sheet.periodic,
            "wall_in_y": wall_in_y.stack_axis,
            "bulk": bulk.stack_axis,
            "bounded box": bounded.stack_axis,
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

        `diameter` controls capsule contact; TANGLE always takes diameters,
        never radii, and exposes `radius` only as a read-only convenience.
        `min_bend_radius` is an admissibility limit, not the preferred shape.
        A fiber's placed centerline is its initial simulation geometry; its
        rest centerline is used to derive rest lengths and rest turning angles.
        """),
        code("""
        import tangle
        from tangle.units import mm, um

        # A missing min_bend_radius means there is no material curvature
        # admissibility limit, while diameter still defines contact size.
        straight = tangle.Material("straight fiber", diameter=19 * um)
        bend_limited = tangle.Material(
            "bend-limited fiber",
            diameter=7 * um,
            min_bend_radius=35 * um,
        )
        print(straight.name, straight.diameter, straight.radius)
        print(bend_limited.min_bend_radius, straight.min_bend_radius)
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
        placed = [[0.2 * mm, 0.2 * mm, 0.5 * mm],
                  [0.5 * mm, 0.3 * mm, 0.5 * mm],
                  [0.8 * mm, 0.2 * mm, 0.5 * mm]]
        straight_rest = [[0.0, 0.0, 0.0],
                         [0.3 * mm, 0.0, 0.0],
                         [0.6 * mm, 0.0, 0.0]]

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
        from tangle.units import mm, um

        material = tangle.Material("fiber", diameter=10 * um)
        # Collection coordinates are local until Recipe.insert applies a rigid
        # transform. Overlap is allowed at this stage.
        ply = tangle.FiberCollection("ply 2")
        fiber_index = ply.add_fiber(
            [[0.0, 0.0, 0.0], [0.5 * mm, 0.0, 0.0]],
            material,
            rest_centerline=[[0.0, 0.0, 0.0], [0.5 * mm, 0.0, 0.0]],
            # Tags remain descriptive metadata; formation_layer participates in
            # later layer-aware recipe operations.
            tags={"family": "machine-direction", "source": "measured"},
            formation_layer=2,
        )
        print(fiber_index, len(ply), ply.layer_ids())
        """),
        md("""
        ## Construction and selection operations

        - `add_fiber(...)` controls placed/rest centerlines, material, tags, and layer.
        - `from_centerlines(...)` assigns one material and optional layer in bulk.
        - `a + b` returns a new collection containing both; `a.extend(b)`
          appends `b` to `a` in place.
        - `layer_ids()` lists the formation layers present.
        - `select_layer(...)` creates a collection containing one formation
          layer. A missing layer raises `ValueError` unless `allow_empty=True`.
        - `centerlines()` and `rest_centerlines()` return ordinary Python lists.
        """),
        code("""
        # Bulk construction is convenient when one material/layer applies to
        # many imported centerlines.
        transverse = tangle.FiberCollection.from_centerlines(
            [[[0.0, 0.0, 0.0], [0.0, 0.5 * mm, 0.0]]],
            material,
            name="transverse",
            formation_layer=3,
        )
        # `+` builds a new collection and leaves both operands unchanged.
        all_fibers = ply + transverse
        all_fibers.name = "two plies"
        # extend() appends in place when mutating a collection is intended.
        growing = tangle.FiberCollection("growing")
        growing.extend(ply)
        # Selection returns another detached collection; it does not mutate the
        # combined source collection.
        selected = all_fibers.select_layer(2, name="only ply 2")
        missing = all_fibers.select_layer(7, allow_empty=True)
        print(len(all_fibers), len(ply), len(growing), len(selected), len(missing))
        print(all_fibers.layer_ids())
        """),
    ],
    "05_fiber_generation.ipynb": [
        md("""
        # Built-in fiber generators

        TANGLE provides small crossing generators for mechanics tests and a
        configurable population generator for material recipes. Generation is
        seeded and deterministic; relaxation is tolerance-deterministic rather
        than promised bitwise-identical across parallel backends. Every
        generator takes a `material=` for fiber name, diameter, and bend limit.
        """),
        code("""
        import math

        import tangle
        from tangle.units import mm, um

        # periodic="xy" makes z the stack axis for layered populations below.
        cell = tangle.Cell([1 * mm, 1 * mm, 1 * mm], periodic="xy")
        large = tangle.Material("large", diameter=19 * um)
        bend_limited = tangle.Material("bend-limited", diameter=19 * um, min_bend_radius=50 * um)

        # Point crossings are the minimal rigid-contact demonstration.
        point_crossing = tangle.generate_point_crossing(
            cell, material=large, count=8, length=0.8 * mm, name="center crossing",
        )
        # Distinct placed/rest shape controls create initially bent fibers.
        curved_crossing = tangle.generate_multisegment_crossing(
            cell, material=bend_limited, count=4, segments_per_fiber=8,
            length=0.8 * mm, placed_chord_fraction=0.8,
            rest_shape="straight", rest_amplitude=0.0,
            placed_shape="curved", placed_amplitude=0.1 * mm,
        )
        # A two-fiber pair is useful for isolated refinement/contact tests.
        pair = tangle.generate_fiber_pair_crossing(
            cell, material=large, segments_per_fiber=1, length=0.8 * mm,
            axis_separation=10 * um, crossing_angle_degrees=90.0,
        )
        [len(point_crossing), len(curved_crossing), len(pair)]
        """),
        md("""
        ## Every `FiberPopulation` field

        `FiberPopulation` describes a seeded population with keyword
        arguments. Length-like fields accept a single value or a `(min, max)`
        tuple that is sampled uniformly. The defaults below are read from the
        compiled extension.
        """),
        *settings_cells("FiberPopulation", "population", POPULATION_FIELDS),
        md("""
        ## Orientation and position distributions

        Each distribution is its own class, so its options are keyword
        arguments of that class rather than loose fields that only apply in one
        mode. An axis or normal of `None` (the default) follows the cell's
        stack axis. Angles are in radians. `LayeredBiaxialOrientation` chooses
        its directions per layer, so it must be combined with
        `LayeredPosition`.
        """),
        md(table(ORIENTATION_VARIANTS, header=("Orientation", "Meaning", "Options"))),
        md(table(POSITION_VARIANTS, header=("Position", "Meaning", "Options"))),
        code("""
        # Keyword arguments describe the whole population in one expression.
        population = tangle.FiberPopulation(
            material=tangle.Material("fine", diameter=7 * um, min_bend_radius=35 * um),
            count=100,
            segments_per_fiber=8,
            seed=42,
            length=(0.2 * mm, 0.3 * mm),
            # Sample diameters between the fine and coarse sizes; None would
            # use the material's diameter for every fiber.
            diameter=(7 * um, 19 * um),
            curvature_amplitude=(0.0, 10 * um),
            # 40% near a primary in-plane direction, 40% near its transverse
            # direction, and the remaining 20% random. The plane normal
            # defaults to the stack axis (z).
            orientation=tangle.LayeredBiaxialOrientation(
                primary_fraction=0.4,
                cross_fraction=0.4,
                max_in_plane_deviation=math.radians(10),
                max_tilt=math.radians(5),
                seed=3,
            ),
            # Four evenly spaced planes along z become formation layers 0-3.
            position=tangle.LayeredPosition(4, jitter_fraction=0.2),
            max_attempts_per_fiber=256,
        )
        generated = tangle.generate_fiber_population(cell, population, name="four plies")
        print(len(generated), generated.layer_ids())
        """),
        code("""
        # replace() returns a modified copy, so variants share every other field.
        variants = {
            "isotropic, uniform": population.replace(
                orientation=tangle.IsotropicOrientation(),
                position=tangle.UniformPosition(),
            ),
            "planar felt": population.replace(
                orientation=tangle.PlanarOrientation(max_tilt=math.radians(5)),
            ),
            "aligned with x": population.replace(
                orientation=tangle.AlignedOrientation("x", max_angle=math.radians(15)),
                length=0.25 * mm,
            ),
            # Layered-biaxial orientation needs layers, so pair the density
            # gradient with a planar orientation instead.
            "denser toward high z": population.replace(
                orientation=tangle.PlanarOrientation(max_tilt=math.radians(5)),
                position=tangle.DensityGradientPosition(exponent=2.0, toward_high=True),
            ),
        }
        {name: len(tangle.generate_fiber_population(cell, variant)) for name, variant in variants.items()}
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
        from tangle.units import mm, um

        # This collection is authored around a local origin and is not yet in
        # the periodic simulation cell.
        cell = tangle.Cell([1 * mm, 1 * mm, 2 * mm], periodic="xy")
        material = tangle.Material("fiber", diameter=19 * um)
        collection = tangle.FiberCollection.from_centerlines(
            [[[-0.3 * mm, 0.0, 0.0], [0.3 * mm, 0.0, 0.0]]],
            material,
            name="local-coordinate ply",
            formation_layer=0,
        )
        # The recipe takes its stack axis (z) from the cell. Set it on the Cell
        # (stack_axis="x") so generators and the recipe agree.
        recipe = tangle.Recipe(cell)
        # Rotation is applied first, then translation places the rotated fiber.
        selection = recipe.insert(
            collection,
            name="placed ply",
            translation=[0.5 * mm, 0.5 * mm, 0.4 * mm],
            rotation=[[0.0, -1.0, 0.0],
                      [1.0,  0.0, 0.0],
                      [0.0,  0.0, 1.0]],
        )
        print(recipe.stack_axis, selection.name, selection.fiber_ids, selection.formation_step)
        """),
        md("""
        `name` labels the returned `FiberSelection`; `translation` is in meters;
        and `rotation` is a 3×3 matrix applied before translation. Selections are
        stable handles for reporting and future selection-scoped APIs. Use
        `recipe.operations()` to audit ordering before a costly run and
        `recipe.centerlines()` to inspect the packed initial geometry.

        A `Recipe` can also start from an existing `Assembly`, such as the
        `result.assembly` of an earlier run, to continue manufacturing from
        relaxed geometry.
        """),
        code("""
        # Methods append ordered operations; no solver launches until run().
        recipe.relax_until_converged(max_iterations=2_000)
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
        `RelaxationSettings` is run-wide: it applies to every step of a recipe
        unless a step receives temporary `RelaxationOverrides` (tutorial 09).
        """),
        code("# Settings objects expose the same defaults used by Rust.\nimport tangle\nfrom tangle.units import um"),
        md("## Every `RelaxationSettings` field"),
        *settings_cells("RelaxationSettings", "settings", RELAXATION_FIELDS),
        md("""
        ## Building and editing settings

        Pass only the values that differ from the defaults as keyword
        arguments. `replace(**changes)` returns a modified copy and leaves the
        original alone; attributes remain assignable for incremental edits.
        Misspelled keywords raise `TypeError` and unknown option strings raise
        `ValueError` on the line that introduced them.
        """),
        code("""
        # Keyword construction names only what differs from the defaults.
        settings = tangle.RelaxationSettings(
            backend="cpu", max_iterations=500, penetration_tolerance=0.1 * um
        )
        # replace() derives a stricter variant without touching `settings`.
        strict = settings.replace(max_iterations=5_000, curvature_ratio_tolerance=0.0)
        # Direct assignment still works for one-off edits.
        strict.max_step = 2 * um

        # Typos and invalid option strings fail immediately.
        for bad in ({"backnd": "cpu"}, {"backend": "cuda"}):
            try:
                tangle.RelaxationSettings(**bad)
            except (TypeError, ValueError) as error:
                print(type(error).__name__, error)
        settings.max_iterations, strict.max_iterations
        """),
        md("""
        ## Cell-list broad phase

        `cell_size_scale` sets the broad-phase cell size. A value of 1
        uses the minimum admissible cell size; larger cells reduce cell count
        but increase candidates per cell. It must be at least 1.

        Each segment also keeps a Verlet neighbor list built from the cells.
        `neighbor_skin_scale` is the extra search distance, as a multiple of
        the largest fiber radius. Lists are rebuilt only after some vertex
        moves more than half the skin, so a larger skin means fewer rebuilds
        but longer lists. `neighbor_capacity` is the list length per segment;
        a segment with more neighbors falls back to scanning its cells, so the
        result is the same, only slower. The skin must be finite and at least 0,
        and the capacity at least 1.
        """),
        code("""
        # Broad-phase cells must be at least one contact diameter wide. Larger
        # values trade fewer cells for more candidate capsule pairs per cell.
        coarse_cells = settings.replace(
            cell_size_scale=1.5, neighbor_skin_scale=2.0, neighbor_capacity=48
        )
        coarse_cells.to_dict()
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
        # Keywords after the profile name adjust individual fields.
        tuned = tangle.AdaptiveSegmentationSettings.profile(
            "balanced", refinement_interval=25, max_refinement_levels=6
        )
        profiles
        """),
        md("""
        Refinement cadence is independent of scheduler batch size. Persistence
        avoids splitting on a single transient overlap. Coarsening preserves
        pinned, targeted, substantially bent, and junction-bearing topology.
        """),
        code("""
        # Pass adaptive settings to the constructor; None disables adaptation.
        settings = tangle.RelaxationSettings(adaptive_segmentation=tuned)
        # The convenience methods install or remove a default settings object.
        settings.disable_adaptive_segmentation()
        assert settings.adaptive_segmentation is None
        settings.enable_adaptive_segmentation()
        settings.adaptive_segmentation.to_dict()
        """),
    ],
    "09_solve_policies_and_overrides.ipynb": [
        md("""
        # Recipe relaxation gates, solve policies, and overrides

        Global `RelaxationSettings` describe the solver for the whole run. A
        `SolvePolicy` is the rule one recipe gate must satisfy, while
        `RelaxationOverrides` temporarily changes how that gate approaches its
        targets. Both are passed to `Recipe.solve(policy, overrides)`. This
        supports contact-first assembly followed by strict final bending
        cleanup.
        """),
        code("# Policies control acceptance; overrides control the path there.\nimport tangle\nfrom tangle.units import mm, um"),
        md("## Every `SolvePolicy` field"),
        *settings_cells(
            "SolvePolicy",
            "default_policy",
            SOLVE_POLICY_FIELDS,
            extra="""
            A policy usually needs only its targets. `max_penetration` and
            `max_curvature_ratio` default to the targets, both limits are hard,
            and an exhausted budget fails the recipe. Loosen acceptance only
            where a stage should tolerate it.
            """,
        ),
        code("""
        # This assembly stage must resolve contact but only reports curvature,
        # accepting ratios up to 5 so bending cannot block deposition.
        contact_first = tangle.SolvePolicy(
            "contact-first settling",
            target_penetration=0.1 * um,
            max_penetration=0.2 * um,
            target_curvature_ratio=5.0,
            hard_curvature=False,
            max_iterations=10_000,
        )
        # A strict final gate needs only its targets; the acceptance limits
        # default to them and both are hard.
        final = tangle.SolvePolicy(
            "final",
            target_penetration=0.1 * um,
            target_curvature_ratio=1.02,
            max_iterations=5_000,
        )
        # Continue past an exhausted budget when every hard limit is met.
        lenient = final.replace(
            name="final (lenient)",
            hard_curvature=False,
            on_budget_exhausted="continue_if_hard_ok",
        )
        [final.max_penetration, final.max_curvature_ratio, final.hard_penetration, lenient.on_budget_exhausted]
        """),
        md("## Every `RelaxationOverrides` field"),
        md(table(OVERRIDE_FIELDS)),
        md("""
        A field left as `None` keeps the global setting. Three presets capture
        the common staging choices, and any field can follow the preset name
        as a keyword:

        - `contact_first`: full contact corrections with loose, cheap mechanics
          (no rest bending, weak curvature limit, one constraint sweep).
        - `contact_cleanup`: full contact corrections with strong stretch and
          curvature projection and many sweeps.
        - `curvature_cleanup`: small contact corrections with full curvature
          projection and many cleanup sweeps, for pulling bends back inside
          the admissible limit.
        """),
        code("""
        # Presets are ordinary RelaxationOverrides; print them to see exactly
        # which fields each one sets.
        for preset in ("contact_first", "contact_cleanup", "curvature_cleanup"):
            print(repr(tangle.RelaxationOverrides.preset(preset)))
        # Keywords after the preset name adjust individual fields.
        cleanup = tangle.RelaxationOverrides.preset("curvature_cleanup", curvature_cleanup_sweeps=32)
        # Hand-written overrides set only the fields they name.
        softer_bending = tangle.RelaxationOverrides(bend_stiffness=0.1, contact_aggregation="deepest_only")
        softer_bending
        """),
        code("""
        # Overrides apply only to the solve() they are passed to; the global
        # settings return for later operations.
        recipe = tangle.Recipe(tangle.Cell([1 * mm, 1 * mm, 1 * mm]))
        # These calls represent distinct amounts or acceptance conditions.
        recipe.relax_for(100)  # fixed work; no acceptance gate
        recipe.relax_until_converged(max_iterations=2_000)  # default hard gate
        recipe.solve(contact_first, tangle.RelaxationOverrides.preset("contact_first"))
        recipe.solve(final, cleanup)
        # Bend limits can change between stages, by material name or object.
        recipe.set_min_bend_radius("fiber", 50 * um)
        print(*recipe.operations(), sep="\\n")
        """),
        md("""
        `settle_targets(tolerance=, max_iterations=)` is the gate for fibers
        held on layer-placement or needle targets; tutorial 10 uses it.

        ## When a stage fails

        `Recipe.run()` raises `tangle.RecipeError` (a `RuntimeError`) when an
        operation cannot meet a hard limit within its budget plus
        `max_extra_iterations`, or fails validation. The exception says which step failed: `operation_index`
        (zero-based position in `recipe.operations()`), `operation` (its
        description), `iteration` (the solver iteration at failure), and
        `reason`.
        """),
        code("""
        # Deliberately give a stage one iteration to show the error fields.
        # It runs the solver, so it is opt-in. The default backend is "wgpu";
        # add backend="cpu" on a machine without a supported GPU.
        RUN_SOLVER = False
        if RUN_SOLVER:
            cell = tangle.Cell([1 * mm, 1 * mm, 1 * mm])
            fibers = tangle.generate_fiber_pair_crossing(
                cell,
                material=tangle.Material("fiber", diameter=19 * um),
                length=0.8 * mm,
                axis_separation=10 * um,
            )
            failing = tangle.Recipe(cell)
            failing.insert(fibers)
            failing.solve(final.replace(name="too short", max_iterations=1))
            try:
                failing.run(tangle.RelaxationSettings())
            except tangle.RecipeError as error:
                print(error.operation_index, error.operation, error.iteration)
                print(error.reason)
        """),
    ],
    "10_layer_motion_and_needling.ipynb": [
        md("""
        # Layer placement and needling operations

        These recipe commands approximate manufacturing by holding fibers on
        temporary device-resident targets. Each hold must be followed by an
        explicit relaxation. Releasing a target lets the displaced fibers and
        their contacts settle mechanically.

        Placement and needling return a `HeldTargets` handle. Use it as a
        context manager: the targets are held for the relaxation inside the
        `with` block and released when the block ends. Without `with`, call
        `release_layer_placement()` or `release_needles()` yourself.
        """),
        code("""
        import tangle
        from tangle.units import mm, um

        # periodic="xy" leaves z as the only bounded axis, so z is the stack
        # axis for layer and needle motion.
        recipe = tangle.Recipe(tangle.Cell([1 * mm, 1 * mm, 3 * mm], periodic="xy"))
        recipe.stack_axis
        """),
        md("""
        ## Layer commands

        - `scale_layer_spacing(factor, stiffness=, max_translation=)` scales
          all current layer-center spacings.
        - `place_layer_above(layer, gap=, ...)` brings one layer to a surface
          gap above the active stack.
        - `settle_targets(tolerance=, max_iterations=)` relaxes until held
          fibers reach their targets.
        - `release_layer_placement()` removes those temporary targets; a `with`
          block calls it for you.
        - `fit_cell_to_active_fibers(axes=, padding=)` removes empty domain
          space; `axes` accepts letters such as `"z"`.
        """),
        code("""
        # scale_layer_spacing scales current layer-center spacing; it does not
        # teleport fibers or bypass contact relaxation. The with block
        # releases the layer targets when it ends.
        with recipe.scale_layer_spacing(0.8, stiffness=0.5, max_translation=5 * um):
            recipe.relax_for(500)

        # Without `with`, the targets stay held until released explicitly.
        recipe.place_layer_above(2, gap=2 * um, stiffness=0.5, max_translation=5 * um)
        recipe.settle_targets(tolerance=0.1 * um, max_iterations=2_000)
        recipe.release_layer_placement()

        # Shrink the bounded z extent to the active fibers plus padding.
        recipe.fit_cell_to_active_fibers(axes="z", padding=25 * um)
        """),
        md("""
        ## Needle commands

        `needle_layer(layer, footprint=, depth=, ...)` pulls one internal
        vertex of each eligible fiber in `layer` along the stack axis by
        `depth`. The footprint chooses the fibers:

        - `CircularFootprint(center, diameter=)` selects fibers crossing an
          in-plane circle; `center` holds the two in-plane coordinates.
        - `CircularFootprint.random(diameter=, seed=)` places that circle at a
          seeded random center. The layer index is mixed into the seed, so one
          seed gives different punch locations on different layers.
        - `RandomFiberFraction(fraction, seed=)` samples a fraction of the
          layer's eligible fibers regardless of location.

        `min_fiber_diameter` restricts needling to thick fibers; `stiffness`,
        `max_translation`, and `max_translation_over_diameter` limit how fast
        the targets pull.
        """),
        code("""
        # A fixed circular punch in x/y (the axes other than the stack axis).
        with recipe.needle_layer(
            2,
            footprint=tangle.CircularFootprint([0.45 * mm, 0.55 * mm], diameter=100 * um),
            depth=0.6 * mm,
            min_fiber_diameter=15 * um,
            stiffness=0.75,
            max_translation=5 * um,
            max_translation_over_diameter=0.5,
        ):
            # Hold the target through relaxation; the block end releases it.
            recipe.settle_targets(tolerance=0.1 * um, max_iterations=5_000)

        # Seeded random punch locations: one stroke per seed.
        for stroke in range(3):
            with recipe.needle_layer(
                2,
                footprint=tangle.CircularFootprint.random(diameter=150 * um, seed=stroke),
                depth=350 * um,
            ):
                recipe.settle_targets(tolerance=0.1 * um, max_iterations=2_000)

        # Random needling samples fibers reproducibly by fraction rather than
        # by spatial footprint. Here the needle is released explicitly.
        recipe.needle_layer(
            3,
            footprint=tangle.RandomFiberFraction(0.15, seed=2026),
            depth=0.6 * mm,
            min_fiber_diameter=15 * um,
            stiffness=0.75,
            max_translation=5 * um,
            max_translation_over_diameter=0.5,
        )
        recipe.settle_targets(tolerance=0.1 * um, max_iterations=5_000)
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
        code("# Compaction is configured independently from the recipe using it.\nimport tangle\nfrom tangle.units import mm, um"),
        md("## Every `CompactionSettings` field"),
        *settings_cells("CompactionSettings", "compaction", COMPACTION_FIELDS),
        md("""
        ## Targets and paths

        The target says when to stop; the path says how the cell moves to get
        there. Both are small objects that carry only their own options, and
        they are chosen independently.
        """),
        md(table(COMPACTION_TARGETS, header=("Target", "Stops at", "Value"))),
        md(table(COMPACTION_PATHS, header=("Path", "Meaning", "Options"))),
        md("""
        ## Common target/path combinations

        `CompactionSettings.volume_fraction(v, **changes)` is the concise
        constructor: it sets `VolumeFractionTarget(v)` and the default
        `AxisWeightsPath()`, which shortens only the recipe's stack axis. Any
        other field follows as a keyword. `replace()` then swaps a target or
        path while keeping every guard.
        """),
        code("""
        # Only the stack axis (z for a periodic="xy" cell) shortens by default.
        compaction = tangle.CompactionSettings.volume_fraction(
            0.40,
            kinematics="moving_walls",
            cell_anchor=[0.5, 0.5, 0.5],  # both z faces move
            balance_opposing_faces=True,
            max_penetration=0.1 * um,
            max_curvature_ratio=1.05,
        )

        # replace() compares paths and targets without rebuilding every guard.
        explicit_z = compaction.replace(path=tangle.AxisWeightsPath("z"))
        equal_pressure = compaction.replace(path=tangle.EqualPressurePath("xyz"))
        stress_ratio = compaction.replace(
            path=tangle.StressRatioPath([1.0, 1.0, 2.0], pressure_floor=1.0)
        )
        least_work = compaction.replace(path=tangle.MinimumWorkPath("xy"))
        target_lengths = compaction.replace(
            target=tangle.CellLengthsTarget([0.8 * mm, 0.8 * mm, 1.2 * mm])
        )
        target_pressure = compaction.replace(target=tangle.MeanPressureTarget(1.0e3))

        # The constructor takes the target object directly, too.
        by_volume = tangle.CompactionSettings(
            tangle.CellVolumeTarget(0.8e-9), path=tangle.AxisWeightsPath([1.0, 1.0, 1.0])
        )
        by_direction = tangle.CompactionSettings(tangle.DirectionalPressureTarget([0.0, 0.0, 1.0e3]))
        by_energy = tangle.CompactionSettings(tangle.PenaltyEnergyTarget(1.0e-9))
        [compaction.target, compaction.path, equal_pressure.path, target_lengths.target]
        """),
        code("""
        # Overrides affect only the compaction operation they are passed to.
        recipe = tangle.Recipe(tangle.Cell([1 * mm, 1 * mm, 2 * mm], periodic="xy"))
        # A gentle first densification with a relaxation preset...
        recipe.compact(
            tangle.CompactionSettings.volume_fraction(0.13, kinematics="moving_walls"),
            tangle.RelaxationOverrides.preset("contact_first"),
        )
        # ...then the guarded final compaction with a hand-written override.
        recipe.compact(compaction, tangle.RelaxationOverrides(contact_aggregation="deepest_only"))
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
        code("# Junction angles are expressed in radians.\nimport math\nimport tangle\nfrom tangle.units import mm, um"),
        md("## Every `JunctionPolicy` field"),
        *settings_cells("JunctionPolicy", "policy", JUNCTION_FIELDS),
        code("""
        # The law name and parameter set are downstream labels; capture filters
        # decide which current contacts receive persistent material anchors.
        policy = tangle.JunctionPolicy(
            "cured binder contacts",
            "cohesive bond",
            parameter_set=2,
            max_surface_gap=0.2 * um,
            min_crossing_angle=math.radians(20),
            max_crossing_angle=math.pi / 2,
            # Probability is deterministic for a fixed seed and candidate set.
            probability=0.25,
            seed=9,
            material_pairs=[("large", "small"), ("large", "large")],
            max_per_fiber_pair=1,
            min_anchor_separation=50 * um,
            candidate_capacity=100_000,
        )
        # replace() derives a variant without repeating every filter.
        every_contact = policy.replace(name="all binder contacts", probability=1.0)
        """),
        md("""
        `capture_junctions(policy)` samples once at that recipe point.
        `relax_and_capture(iterations=, capture_every=, policy=)` samples
        repeatedly at an explicit cadence independent of scheduler batch size.
        Captured junctions are topology/export data; current relaxation does
        not enforce their mechanics, so late capture is normally the physically
        appropriate workflow.
        """),
        code("""
        recipe = tangle.Recipe(tangle.Cell([1 * mm, 1 * mm, 1 * mm]))
        # Late capture records bonds after geometry has settled; contacts before
        # this explicit operation remain transient.
        recipe.relax_until_converged(max_iterations=5_000)
        recipe.capture_junctions(policy)

        # Alternative: capture repeatedly while relaxing.
        repeated = tangle.Recipe(tangle.Cell([1 * mm, 1 * mm, 1 * mm]))
        repeated.relax_and_capture(iterations=2_000, capture_every=250, policy=every_contact)
        recipe.operations() + repeated.operations()
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
        branch = checkpoint.replace(
            path=Path("output/alternate_cleanup.restart"),
            resume=True,
            resume_path=checkpoint.path,
            resume_case_id=checkpoint.case_id,
            fresh_formation_on_resume=True,
        )
        {name: getattr(branch, name) for name in fields}
        """),
        md("""
        The complete round trip below first writes a checkpoint from a small
        two-fiber recipe, then branches a cleanup from it. A checkpoint is
        written only when a run crosses `interval_iterations`, so a short run
        with the 500-iteration cadence above would leave nothing to resume.
        """),
        code("""
        from tangle.units import mm, um

        # Execution is opt-in in this reference notebook. The default backend
        # is "wgpu"; add backend="cpu" on a machine without a supported GPU.
        RUN_RECIPE = False
        if RUN_RECIPE:
            fiber = tangle.Material("fiber", diameter=20 * um)
            crossing = tangle.FiberCollection("crossing")
            crossing.add_fiber([[0.2 * mm, 0.5 * mm, 0.5 * mm], [0.8 * mm, 0.5 * mm, 0.5 * mm]], fiber)
            crossing.add_fiber([[0.5 * mm, 0.2 * mm, 0.5 * mm], [0.5 * mm, 0.8 * mm, 0.5 * mm]], fiber)
            cell = tangle.Cell([1 * mm, 1 * mm, 1 * mm])
            # Default tolerances suit millimeter-scale fibers, so state micrometer ones.
            settings = tangle.RelaxationSettings(penetration_tolerance=0.1 * um, max_step=2 * um)

            # Save every 100 iterations so this 300-iteration run leaves a restart.
            source = tangle.CheckpointSettings(
                "two-fiber-demo", Path("output/two_fiber.restart"), interval_iterations=100
            )
            formation = tangle.Recipe(cell)
            formation.insert(crossing)
            formation.relax_for(300)
            formed = formation.run(settings, checkpoint=source)
            print(formed.checkpoint_saves, formed.last_checkpoint_iteration)

            # The branch carries the saved geometry into a new recipe; with
            # fresh_formation_on_resume it runs from that recipe's first operation.
            cleanup = tangle.Recipe(cell)
            cleanup.relax_until_converged()
            result = cleanup.run(
                settings,
                checkpoint=source.replace(
                    path=Path("output/two_fiber_cleanup.restart"),
                    resume=True,
                    resume_path=source.path,
                    resume_case_id=source.case_id,
                    fresh_formation_on_resume=True,
                ),
            )
            # The resumed, relaxed geometry is a new Assembly on the result.
            print(result.resumed, result.resumed_iteration, result.assembly.fiber_count)
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

        `run()` never modifies its inputs. The relaxed geometry is a new
        `Assembly` at `result.assembly`, which can seed a follow-on `Recipe`.
        A step that cannot meet its policy raises `tangle.RecipeError` instead
        of returning a result (tutorial 09).
        """),
        code("# Export methods accept pathlib paths as well as strings.\nfrom pathlib import Path\nimport tangle\nfrom tangle.units import mm, um"),
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
            ("assembly", "Relaxed geometry as a new assembly; the recipe's inputs are unchanged.", "`Assembly`"),
        ])),
        code("""
        # Build a minimal intersecting pair so the result contains meaningful
        # contact and curvature diagnostics.
        cell = tangle.Cell([1 * mm, 1 * mm, 1 * mm])
        fibers = tangle.generate_fiber_pair_crossing(
            cell,
            material=tangle.Material("fiber", diameter=19 * um),
            length=0.8 * mm,
            axis_separation=10 * um,
        )
        recipe = tangle.Recipe(cell)
        recipe.insert(fibers)
        recipe.relax_until_converged(max_iterations=2_000)

        # Keep reference notebooks safe to execute top-to-bottom by default.
        RUN_SOLVER = False
        if RUN_SOLVER:
            # Default tolerances suit millimeter-scale fibers, so state
            # micrometer ones here. The default backend is "wgpu"; add
            # backend="cpu" on a machine without a supported GPU.
            settings = tangle.RelaxationSettings(penetration_tolerance=0.1 * um, max_step=2 * um)
            try:
                result = recipe.run(settings)
            except tangle.RecipeError as error:
                # The error names the failing step and why it failed.
                print(error.operation_index, error.operation, error.iteration, error.reason)
                raise
            print(result.events)
            print(result.warnings)
            final_centerlines = result.centerlines()
            # The relaxed Assembly can seed a follow-on recipe.
            relaxed = result.assembly
            follow_on = tangle.Recipe(relaxed)
            print(relaxed.fiber_count, relaxed.cell.lengths)
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

        - `spheres_exact` keeps the requested arc spacing exactly and shares
          unused length between the two fiber ends.
        - `spheres_dynamic` preserves every source segment endpoint and adjusts
          the spacing inside each segment.
        - `spherocylinders_exact` writes one capsule per active segment.
        - `spherocylinders_constant` uses the shortest active segment length
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
                mode="spheres_dynamic",
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
        exact centerlines and sections; `result.assembly.characterize()` gives
        the same report. It reports nominal swept volume rather than the
        geometric union of solids, so the field is deliberately named
        `nominal_swept_volume_fraction`. Curvature is summarized by
        `max_curvature`, `max_curvature_ratio`, and
        `curvature_limit_violations`. The complete versioned report is
        available as a dictionary, JSON string, or JSON file.
        """),
        code("""
        if RUN_SOLVER:
            # Length weighting treats centerline equally; volume weighting is
            # the appropriate voxel comparison when diameters differ.
            analysis = result.characterize()
            print(analysis.length_weighted_orientation_tensor)
            print(analysis.volume_weighted_orientation_tensor)
            print(analysis.max_curvature_ratio, analysis.curvature_limit_violations)
            analysis.write_json(output / "tangle_analysis.json")
        """),
        md("""
        ## Contacts and neighbors

        `characterize_neighbors()` measures how fibers touch and travel
        together, which volume fraction and orientation tensors cannot show. It
        samples each centerline at a uniform spacing and works on an
        `Assembly` or a `RunResult`. The
        [analysis guide](../../../../docs/puma_interoperability.md#contacts-and-neighbors)
        defines each quantity.
        """),
        md(table([
            ("contact_gap", "Surface gap counted as touching (required).", "m"),
            ("neighbor_gap", "Surface gap counted as a neighbor.", "m, or `None` for twice the largest radius"),
            ("in_axis_angle_degrees", "Tangent angle below which a pair runs side by side.", "degrees"),
            ("sample_spacing", "Arc-length spacing between samples.", "m, or `None` for a quarter of the smallest radius"),
            ("max_lag", "Largest lag of the neighbor-turnover curve.", "m, or `None` for half the median fiber length (at most 200 samples)"),
            ("lag_count", "Logarithmically spaced turnover lags.", "count"),
        ])),
        md(table([
            ("contacts_per_length", "Contacts per unit centerline length.", "1/m"),
            ("contact_ratio_to_random", "Ratio to a random-placement baseline.", "ratio or `None`"),
            ("in_axis_contact_fraction", "Share of contacts between side-by-side fibers.", "fraction"),
            ("median_crossing_angle_degrees", "Median angle of crossing contacts.", "degrees or `None`"),
            ("median_excess_persistence", "Crossing contact length over a straight crossing's.", "ratio or `None`"),
            ("mean_free_length", "Mean centerline length between contacts.", "m or `None`"),
            ("mean_neighbors", "Mean neighbor count per sample.", "count"),
            ("neighbor_correlation_length", "Lag at which neighbor sets decorrelate to 1/e.", "m or `None`"),
            ("contact_count_dispersion", "Variance over mean of contacts per fiber.", "ratio or `None`"),
        ])),
        code("""
        # Any Assembly can be analyzed without a recipe or relaxation, for
        # example centerlines tracked from a CT scan. insert() adds them as-is.
        scan = tangle.Assembly(tangle.Cell([1 * mm, 1 * mm, 1 * mm]))
        scan.insert(
            tangle.FiberCollection.from_centerlines(
                [
                    [[0.1 * mm, 0.5 * mm, 0.5 * mm], [0.9 * mm, 0.5 * mm, 0.5 * mm]],
                    # 19.5 um apart: a 0.5 um surface gap between 19 um fibers.
                    [[0.5 * mm, 0.1 * mm, 0.5 * mm + 19.5 * um], [0.5 * mm, 0.9 * mm, 0.5 * mm + 19.5 * um]],
                ],
                tangle.Material("fiber", diameter=19 * um),
            )
        )
        # CT cannot resolve gaps below about one voxel, so use a contact gap of
        # that order on both sides of a comparison.
        reference = scan.characterize_neighbors(contact_gap=1 * um)
        print(reference.contact_count, reference.contacts_per_length)
        print(reference.in_axis_contact_fraction, reference.median_crossing_angle_degrees)
        """),
        code("""
        if RUN_SOLVER:
            # The same call on the relaxed result; keep the gaps and angle equal
            # when comparing two assemblies.
            neighbors = result.characterize_neighbors(
                contact_gap=1 * um,
                neighbor_gap=10 * um,
                in_axis_angle_degrees=20,
            )
            print(neighbors.contacts_per_length, neighbors.contact_ratio_to_random)
            print(neighbors.mean_neighbors, neighbors.neighbor_correlation_length)
            neighbors.write_json(output / "neighbors.json")
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
                voxel_size=20 * um,
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
