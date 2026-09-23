"""Twenty-ply control, needled, cleanup, and DEM-polish felt configurations.

This is the Python equivalent of the long-running Rust ``fake_felted``
example. The default configuration is intentionally full scale; use the
notebook's ``RUN_FULL_SCALE`` guard while editing the recipe interactively.
"""

import argparse
import math
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "fake_felted"
FOOTPRINT = 1.0e-3
STAGING_THICKNESS = 12.0e-3
LAYER_COUNT = 20
LAYER_SPACING = 50.0e-6
CONTACT_TOLERANCE = 0.30e-6
FINAL_BEND_RATIO = 1.001


def material_settings(
    count: int, name: str, seed: int, diameter: float
) -> tangle.FiberPopulationSettings:
    settings = tangle.FiberPopulationSettings()
    settings.count = count
    settings.segments_per_fiber = 32
    settings.seed = seed
    settings.nominal_parent_length = 0.0508
    settings.length_minimum, settings.length_maximum = 0.003, 0.004
    settings.radius_minimum = settings.radius_maximum = 0.5 * diameter
    settings.orientation = "layered_biaxial"
    settings.orientation_axis = [0.0, 0.0, 1.0]
    settings.primary_fraction = 0.40
    settings.cross_fraction = 0.40
    settings.maximum_in_plane_deviation = math.pi / 18.0
    settings.maximum_tilt = math.pi / 1_800.0
    settings.layer_orientation_seed = 20_260_920
    settings.position = "layered"
    settings.position_axis = 2
    settings.layers = LAYER_COUNT
    settings.jitter_fraction = 0.20
    settings.max_attempts_per_fiber = 2_048
    settings.material_name = name
    if diameter < 10.0e-6:
        settings.curvature_amplitude_minimum = 0.0
        settings.curvature_amplitude_maximum = 2.0e-6
        settings.minimum_bend_radius = 200.0e-6
    else:
        settings.curvature_amplitude_minimum = 0.0
        settings.curvature_amplitude_maximum = 8.0e-6
        settings.minimum_bend_radius = 60.0e-6
    return settings


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def needle_center(layer: int) -> list[float]:
    seed = 20_260_940
    unit = lambda bits: (bits >> 40) / float(1 << 24)
    return [
        FOOTPRINT * unit(splitmix64(seed ^ (2 * layer))),
        FOOTPRINT * unit(splitmix64(seed ^ (2 * layer + 1))),
    ]


def solve_policy(
    name: str,
    solver_penetration: float,
    solver_bend: float,
    acceptance_penetration: float,
    acceptance_bend: float,
    maximum_iterations: int,
    *,
    penetration_enforcement: str = "hard",
    curvature_enforcement: str = "hard",
    continue_soft: bool = False,
) -> tangle.SolvePolicy:
    return tangle.SolvePolicy(
        name,
        solver_penetration=solver_penetration,
        solver_curvature_ratio=solver_bend,
        acceptance_penetration=acceptance_penetration,
        penetration_enforcement=penetration_enforcement,
        acceptance_curvature_ratio=acceptance_bend,
        curvature_enforcement=curvature_enforcement,
        maximum_iterations=maximum_iterations,
        on_exhaustion=(
            "continue_if_hard_limits_satisfied" if continue_soft else "reject"
        ),
    )


def final_compaction() -> tangle.CompactionSettings:
    settings = tangle.CompactionSettings.volume_fraction(0.13, axis_weights=[0.0, 0.0, 1.0])
    settings.kinematics = "moving_walls"
    settings.cell_anchor = [0.0, 0.0, 0.5]
    settings.balance_opposing_faces = True
    settings.face_pressure_floor = 1.0e-9
    settings.face_balance_strength = 0.5
    settings.initial_log_strain = 0.02
    settings.minimum_log_strain = 0.001
    settings.maximum_log_strain = 0.04
    settings.growth_factor = 1.2
    settings.shrink_factor = 0.5
    settings.relax_iterations = 1_000
    settings.maximum_shortening_over_minimum_diameter = 0.5
    settings.maximum_penetration = 0.301e-6
    settings.maximum_bend_ratio = 1.05
    settings.maximum_steps = 512
    settings.maximum_relax_windows = 8
    return settings


def add_final_relaxation(recipe: tangle.Recipe) -> None:
    recipe.relax_with_policy(
        solve_policy(
            "final admissibility relaxation",
            CONTACT_TOLERANCE,
            1.00001,
            CONTACT_TOLERANCE,
            FINAL_BEND_RATIO,
            100_000,
        )
    )


def add_control_formation(
    recipe: tangle.Recipe,
    small: tangle.FiberCollection,
    large: tangle.FiberCollection,
    *,
    needled: bool,
) -> None:
    for layer in range(LAYER_COUNT):
        ply = small.select_layer(layer, name=f"mixed ply {layer}")
        ply.extend(large.select_layer(layer, allow_empty=True))
        recipe.insert(ply)
        recipe.relax_for(200 if layer == 0 else 100)
        if layer == 0:
            continue
        recipe.place_layer_above(
            layer, LAYER_SPACING, stiffness=1.0, max_translation=2.0e-6
        )
        recipe.relax_until_targets_reached(0.5e-6, 6_000)
        recipe.relax_for(80)
        recipe.release_layer_targets()
        recipe.relax_for(200)
        if needled and layer >= 2:
            recipe.needle_layer_circular(
                layer,
                needle_center(layer),
                150.0e-6,
                350.0e-6,
                minimum_fiber_diameter=0.99 * 19.0e-6,
                stiffness=0.5,
                max_translation=1.0e-6,
                maximum_translation_over_fiber_diameter=0.25,
            )
            recipe.relax_until_targets_reached(0.5e-6, 3_000)
            recipe.relax_for(150)
            recipe.release_needles()
            recipe.relax_for(200)
    recipe.release_needles()
    recipe.compact(final_compaction())
    add_final_relaxation(recipe)


def cleanup_overrides(kind: str) -> tangle.RelaxationOverrides:
    overrides = tangle.RelaxationOverrides()
    overrides.motion_model = "flexible"
    overrides.contact_aggregation = "deepest_only"
    if kind == "contact_first":
        overrides.correction_fraction = 0.8
        overrides.stretch_stiffness = 0.05
        overrides.bend_stiffness = 0.0
        overrides.curvature_limit_stiffness = 0.15
        overrides.constraint_iterations = 1
        overrides.curvature_cleanup_sweeps = 1
    elif kind == "curvature":
        overrides.correction_fraction = 0.05
        overrides.stretch_stiffness = 0.35
        overrides.bend_stiffness = 0.0
        overrides.curvature_limit_stiffness = 1.0
        overrides.constraint_iterations = 8
        overrides.curvature_cleanup_sweeps = 16
    elif kind == "contact":
        overrides.correction_fraction = 0.8
        overrides.stretch_stiffness = 0.35
        overrides.bend_stiffness = 0.0
        overrides.curvature_limit_stiffness = 0.75
        overrides.constraint_iterations = 8
        overrides.curvature_cleanup_sweeps = 16
    return overrides


def add_cleanup(recipe: tangle.Recipe, *, dem_polish: bool = False) -> None:
    if dem_polish:
        for name, penetration, budget in [
            ("DEM coarse contact polish", 0.05e-6, 75_000),
            ("DEM final contact polish", 0.01e-6, 150_000),
        ]:
            recipe.relax_with_policy(
                solve_policy(
                    name,
                    penetration,
                    FINAL_BEND_RATIO,
                    penetration,
                    FINAL_BEND_RATIO,
                    budget,
                ),
                cleanup_overrides("contact"),
            )
        return

    recipe.set_material_bend_radius("7 um stiff fiber", 200.0e-6)
    recipe.relax_with_policy(
        solve_policy(
            "contact-first settling",
            1.0e-6,
            5.0,
            2.0e-6,
            5.0,
            15_000,
            curvature_enforcement="soft",
            continue_soft=True,
        ),
        cleanup_overrides("contact_first"),
    )
    recipe.fit_cell_to_active_fibers(axes=[False, False, True], padding=25.0e-6)
    mechanics = tangle.RelaxationOverrides()
    mechanics.correction_fraction = 0.7
    mechanics.contact_aggregation = "penetration_weighted"
    mechanics.stretch_stiffness = 0.15
    mechanics.bend_stiffness = 0.01
    mechanics.curvature_limit_stiffness = 0.5
    mechanics.constraint_iterations = 4
    mechanics.curvature_cleanup_sweeps = 2
    recipe.relax_with_policy(
        solve_policy(
            "controlled mechanics recovery",
            1.0e-6,
            2.7,
            2.0e-6,
            2.7,
            20_000,
            curvature_enforcement="soft",
            continue_soft=True,
        ),
        mechanics,
    )
    compact = final_compaction()
    compact.initial_log_strain = 0.001
    compact.minimum_log_strain = 0.0001
    compact.maximum_log_strain = 0.005
    compact.growth_factor = 1.15
    compact.relax_iterations = 250
    compact.maximum_shortening_over_minimum_diameter = 0.25
    compact.maximum_penetration = 5.0e-6
    compact.maximum_bend_ratio = 100.0
    compact.maximum_relax_windows = 16
    compact_overrides = cleanup_overrides("contact_first")
    recipe.compact(compact, compact_overrides)

    stages = [
        ("curvature", "coarse curvature cleanup", 20.0e-6, 2.0, 20_000, "soft", "hard"),
        ("contact", "coarse contact settling", 2.0e-6, 1.25, 30_000, "hard", "hard"),
        ("curvature", "near-admissible curvature cleanup", 20.0e-6, 1.25, 40_000, "soft", "hard"),
        ("contact", "near-admissible contact settling", 0.75e-6, 1.05, 60_000, "hard", "hard"),
        ("curvature", "final hard curvature cleanup", 20.0e-6, FINAL_BEND_RATIO, 100_000, "soft", "hard"),
        ("contact", "final hard contact settling", CONTACT_TOLERANCE, FINAL_BEND_RATIO, 150_000, "hard", "hard"),
    ]
    for kind, name, penetration, bend, budget, p_enforce, b_enforce in stages:
        recipe.relax_with_policy(
            solve_policy(
                name,
                penetration,
                bend,
                penetration,
                bend,
                budget,
                penetration_enforcement=p_enforce,
                curvature_enforcement=b_enforce,
            ),
            cleanup_overrides(kind),
        )


def build(mode: str = "control") -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    valid_modes = {
        "control",
        "needled",
        "cleanup",
        "dem_polish",
        "needled_cleanup",
        "needled_dem_polish",
    }
    if mode not in valid_modes:
        raise ValueError(f"unknown fake-felted mode {mode!r}")
    cell = tangle.Cell(
        [FOOTPRINT, FOOTPRINT, STAGING_THICKNESS], periodic=[True, True, False]
    )
    small = tangle.generate_fiber_population(
        cell,
        material_settings(480, "7 um stiff fiber", 20_260_921, 7.0e-6),
        name="small",
    )
    large = tangle.generate_fiber_population(
        cell,
        material_settings(65, "19 um bendy fiber", 20_260_922, 19.0e-6),
        name="large",
    )
    recipe = tangle.Recipe(cell, layer_axis=2)
    if mode in ("control", "needled"):
        add_control_formation(recipe, small, large, needled=mode == "needled")
    else:
        # A resumed cleanup replaces the saved formation cursor with these
        # operations. Restored checkpoint geometry is authoritative.
        add_cleanup(recipe, dem_polish=mode in ("dem_polish", "needled_dem_polish"))

    settings = tangle.RelaxationSettings()
    dense = mode not in ("control", "needled")
    settings.constraint_iterations = 8
    settings.curvature_limit_safety_margin = 0.06
    settings.curvature_cleanup_sweeps = 16 if dense else 4
    settings.contact_aggregation = "deepest_only" if dense else "uniform_average"
    settings.max_iterations = 500_000
    settings.iterations_per_batch = 10
    settings.penetration_tolerance = 2.5e-6 if dense else CONTACT_TOLERANCE
    settings.correction_fraction = 0.8 if dense else 0.35
    settings.curvature_limit_stiffness = 0.75 if dense else 1.0
    settings.curvature_ratio_tolerance = 1.7 if dense else 1.0e-5
    settings.max_step = 2.0e-6
    adaptive = tangle.AdaptiveSegmentationSettings()
    adaptive.contact_length_over_diameter = 4.0
    adaptive.minimum_length_over_diameter = 2.0
    adaptive.maximum_refinement_levels = 6
    adaptive.refinement_interval = 32
    adaptive.refinement_persistence = 3
    adaptive.coarsening_persistence = 8
    settings.adaptive_segmentation = adaptive
    return recipe, settings


def run(
    mode: str = "control",
    *,
    resume: bool = False,
    resume_path: Path | None = None,
    resume_case_id: str | None = None,
) -> tangle.RunResult:
    recipe, settings = build(mode)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = tangle.CheckpointSettings(
        f"fake-felted-python-{mode}",
        OUTPUT / f"{mode}.restart",
        interval_iterations=500,
        resume=resume,
    )
    if resume_path is not None:
        checkpoint.resume = True
        checkpoint.resume_path = resume_path
        checkpoint.resume_case_id = resume_case_id or (
            "fake-felted-python-needled"
            if mode.startswith("needled_")
            else "fake-felted-python-control"
        )
        checkpoint.fresh_formation_on_resume = mode not in ("control", "needled")
    result = recipe.run(settings, checkpoint=checkpoint)
    result.write_ovito(
        OUTPUT / f"{mode}.dump",
        view_script_path=OUTPUT / f"{mode}_view.py",
        session_path=OUTPUT / f"{mode}.ovito",
        coloring="curvature_ratio",
    )
    result.export_bpm(
        OUTPUT / f"{mode}_capsules.data",
        mode="spherocylinders-exact",
        density=1_800.0,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "control",
            "needled",
            "cleanup",
            "dem_polish",
            "needled_cleanup",
            "needled_dem_polish",
        ),
        default="control",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=Path)
    parser.add_argument("--resume-case-id")
    args = parser.parse_args()
    print(
        run(
            args.mode,
            resume=args.resume,
            resume_path=args.resume_path,
            resume_case_id=args.resume_case_id,
        )
    )
