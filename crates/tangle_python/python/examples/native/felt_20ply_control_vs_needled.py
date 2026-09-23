"""Twenty-ply felt: unneedled control versus needled, with cleanup and DEM polish.

This is the Python equivalent of the long-running Rust
``examples/felt_20ply_control_vs_needled`` example. Each run is one
``--specimen`` (``control`` or ``needled``) at one ``--stage``:

``form``
    Deposit and relax the twenty plies (needling each ply from the third one
    on for the needled specimen), compact, and reach final admissibility.
``cleanup``
    Resume from the same specimen's ``form`` checkpoint and run the staged
    contact/curvature cleanup ladder.
``polish``
    Resume from the same specimen's ``cleanup`` checkpoint and polish contacts
    for a DEM handoff.

The default configuration is intentionally full scale; use the notebook's
``RUN_FULL_SCALE`` guard while editing the recipe interactively.
"""

import argparse
import math
from pathlib import Path

import tangle
from tangle.units import mm, um


OUTPUT = Path(__file__).parent / "output" / "felt_20ply_control_vs_needled"
SPECIMENS = ("control", "needled")
STAGES = ("form", "cleanup", "polish")
# cleanup and polish restart from the geometry that the previous stage saved.
SOURCE_STAGE = {"cleanup": "form", "polish": "cleanup"}

FOOTPRINT = 1 * mm
STAGING_THICKNESS = 12 * mm
LAYER_COUNT = 20
LAYER_SPACING = 50 * um
CONTACT_TOLERANCE = 0.30 * um
FINAL_CURVATURE_RATIO = 1.001

FINE = tangle.Material("fine_7um", diameter=7.0e-6, min_bend_radius=200.0e-6)
COARSE = tangle.Material("coarse_19um", diameter=19.0e-6, min_bend_radius=60.0e-6)


def ply_population(
    material: tangle.Material, count: int, seed: int, max_curvature_amplitude: float
) -> tangle.FiberPopulation:
    return tangle.FiberPopulation(
        material=material,
        count=count,
        segments_per_fiber=32,
        seed=seed,
        nominal_parent_length=0.0508,
        length=(3 * mm, 4 * mm),
        curvature_amplitude=(0.0, max_curvature_amplitude),
        orientation=tangle.LayeredBiaxialOrientation(
            primary_fraction=0.40,
            cross_fraction=0.40,
            max_in_plane_deviation=math.pi / 18.0,
            max_tilt=math.pi / 1_800.0,
            seed=20_260_920,
        ),
        position=tangle.LayeredPosition(LAYER_COUNT, jitter_fraction=0.20),
        max_attempts_per_fiber=2_048,
    )


def final_compaction() -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        0.13,
        kinematics="moving_walls",
        cell_anchor=[0.0, 0.0, 0.5],
        balance_opposing_faces=True,
        face_pressure_floor=1.0e-9,
        face_balance_strength=0.5,
        initial_log_strain=0.02,
        min_log_strain=0.001,
        max_log_strain=0.04,
        growth_factor=1.2,
        shrink_factor=0.5,
        relax_iterations=1_000,
        max_shortening_over_min_diameter=0.5,
        max_penetration=0.301 * um,
        max_curvature_ratio=1.05,
        max_steps=512,
        max_relax_windows=8,
    )


def add_final_relaxation(recipe: tangle.Recipe) -> None:
    recipe.solve(
        tangle.SolvePolicy(
            "final admissibility relaxation",
            target_penetration=CONTACT_TOLERANCE,
            target_curvature_ratio=1.00001,
            max_curvature_ratio=FINAL_CURVATURE_RATIO,
            max_iterations=100_000,
        )
    )


def add_ply_stack_formation(
    recipe: tangle.Recipe,
    fine: tangle.FiberCollection,
    coarse: tangle.FiberCollection,
    *,
    needle: bool = False,
) -> None:
    for layer in range(LAYER_COUNT):
        recipe.insert(
            fine.select_layer(layer, name=f"mixed ply {layer}")
            + coarse.select_layer(layer, allow_empty=True)
        )
        recipe.relax_for(200 if layer == 0 else 100)
        if layer == 0:
            continue
        with recipe.place_layer_above(
            layer, gap=LAYER_SPACING, stiffness=1.0, max_translation=2 * um
        ):
            recipe.settle_targets(tolerance=0.5 * um, max_iterations=6_000)
            recipe.relax_for(80)
        recipe.relax_for(200)
        if needle and layer >= 2:
            with recipe.needle_layer(
                layer,
                footprint=tangle.CircularFootprint.random(
                    diameter=150 * um, seed=20_260_940
                ),
                depth=350 * um,
                min_fiber_diameter=0.99 * COARSE.diameter,
                stiffness=0.5,
                max_translation=1 * um,
                max_translation_over_diameter=0.25,
            ):
                recipe.settle_targets(tolerance=0.5 * um, max_iterations=3_000)
                recipe.relax_for(150)
            recipe.relax_for(200)
    recipe.compact(final_compaction())
    add_final_relaxation(recipe)


# Cleanup ladder: alternate a curvature pass and a contact pass, tightening the
# targets from coarse to near-admissible to final.
CLEANUP_LADDER = [
    (
        tangle.SolvePolicy(
            "cleanup/1-curvature-coarse",
            target_penetration=20 * um,
            target_curvature_ratio=2.0,
            hard_penetration=False,
            max_iterations=20_000,
        ),
        tangle.RelaxationOverrides.preset("curvature_cleanup"),
    ),
    (
        tangle.SolvePolicy(
            "cleanup/2-contact-coarse",
            target_penetration=2 * um,
            target_curvature_ratio=1.25,
            max_iterations=30_000,
        ),
        tangle.RelaxationOverrides.preset("contact_cleanup"),
    ),
    (
        tangle.SolvePolicy(
            "cleanup/3-curvature-near",
            target_penetration=20 * um,
            target_curvature_ratio=1.25,
            hard_penetration=False,
            max_iterations=40_000,
        ),
        tangle.RelaxationOverrides.preset("curvature_cleanup"),
    ),
    (
        tangle.SolvePolicy(
            "cleanup/4-contact-near",
            target_penetration=0.75 * um,
            target_curvature_ratio=1.05,
            max_iterations=60_000,
        ),
        tangle.RelaxationOverrides.preset("contact_cleanup"),
    ),
    (
        tangle.SolvePolicy(
            "cleanup/5-curvature-final",
            target_penetration=20 * um,
            target_curvature_ratio=FINAL_CURVATURE_RATIO,
            hard_penetration=False,
            max_iterations=100_000,
        ),
        tangle.RelaxationOverrides.preset("curvature_cleanup"),
    ),
    (
        tangle.SolvePolicy(
            "cleanup/6-contact-final",
            target_penetration=CONTACT_TOLERANCE,
            target_curvature_ratio=FINAL_CURVATURE_RATIO,
            max_iterations=150_000,
        ),
        tangle.RelaxationOverrides.preset("contact_cleanup"),
    ),
]


def add_cleanup(recipe: tangle.Recipe) -> None:
    # A resumed recipe has no fibers yet, so this reasserts the fine fibers'
    # bend limit on the restored geometry.
    recipe.set_min_bend_radius(FINE, 200.0e-6)
    recipe.solve(
        tangle.SolvePolicy(
            "settle/1-contact-first",
            target_penetration=1 * um,
            max_penetration=2 * um,
            target_curvature_ratio=5.0,
            hard_curvature=False,
            max_iterations=15_000,
            on_budget_exhausted="continue_if_hard_ok",
        ),
        tangle.RelaxationOverrides.preset("contact_first"),
    )
    recipe.fit_cell_to_active_fibers(padding=25 * um)
    recipe.solve(
        tangle.SolvePolicy(
            "settle/2-mechanics-recovery",
            target_penetration=1 * um,
            max_penetration=2 * um,
            target_curvature_ratio=2.7,
            hard_curvature=False,
            max_iterations=20_000,
            on_budget_exhausted="continue_if_hard_ok",
        ),
        tangle.RelaxationOverrides(
            correction_fraction=0.7,
            contact_aggregation="penetration_weighted",
            stretch_stiffness=0.15,
            bend_stiffness=0.01,
            curvature_limit_stiffness=0.5,
            constraint_iterations=4,
            curvature_cleanup_sweeps=2,
        ),
    )
    recipe.compact(
        final_compaction().replace(
            initial_log_strain=0.001,
            min_log_strain=0.0001,
            max_log_strain=0.005,
            growth_factor=1.15,
            relax_iterations=250,
            max_shortening_over_min_diameter=0.25,
            max_penetration=5 * um,
            max_curvature_ratio=100.0,
            max_relax_windows=16,
        ),
        tangle.RelaxationOverrides.preset("contact_first"),
    )
    for policy, overrides in CLEANUP_LADDER:
        recipe.solve(policy, overrides)


def add_polish(recipe: tangle.Recipe) -> None:
    for name, penetration, budget in [
        ("polish/1-contact-coarse", 0.05 * um, 75_000),
        ("polish/2-contact-final", 0.01 * um, 150_000),
    ]:
        recipe.solve(
            tangle.SolvePolicy(
                name,
                target_penetration=penetration,
                target_curvature_ratio=FINAL_CURVATURE_RATIO,
                max_iterations=budget,
            ),
            tangle.RelaxationOverrides.preset("contact_cleanup"),
        )


def build(
    specimen: str = "control", stage: str = "form"
) -> tuple[tangle.Recipe, tangle.RelaxationSettings]:
    if specimen not in SPECIMENS:
        raise ValueError(f"unknown specimen {specimen!r}; expected one of {SPECIMENS}")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    cell = tangle.Cell([FOOTPRINT, FOOTPRINT, STAGING_THICKNESS], periodic="xy")
    recipe = tangle.Recipe(cell)
    if stage == "form":
        fine = tangle.generate_fiber_population(
            cell, ply_population(FINE, 480, 20_260_921, 2 * um), name="fine"
        )
        coarse = tangle.generate_fiber_population(
            cell, ply_population(COARSE, 65, 20_260_922, 8 * um), name="coarse"
        )
        add_ply_stack_formation(recipe, fine, coarse, needle=specimen == "needled")
    elif stage == "cleanup":
        # A resumed stage replaces the saved formation cursor with these
        # operations. Restored checkpoint geometry is authoritative.
        add_cleanup(recipe)
    else:
        add_polish(recipe)

    dense = stage != "form"
    settings = tangle.RelaxationSettings(
        constraint_iterations=8,
        curvature_limit_safety_margin=0.06,
        curvature_cleanup_sweeps=16 if dense else 4,
        contact_aggregation="deepest_only" if dense else "uniform_average",
        max_iterations=500_000,
        iterations_per_batch=10,
        penetration_tolerance=2.5e-6 if dense else CONTACT_TOLERANCE,
        correction_fraction=0.8 if dense else 0.35,
        curvature_limit_stiffness=0.75 if dense else 1.0,
        curvature_ratio_tolerance=1.7 if dense else 1.0e-5,
        max_step=2.0e-6,
        adaptive_segmentation=tangle.AdaptiveSegmentationSettings(
            contact_length_over_diameter=4.0,
            min_length_over_diameter=2.0,
            max_refinement_levels=6,
            refinement_interval=32,
            refinement_persistence=3,
            coarsening_persistence=8,
        ),
    )
    return recipe, settings


def case_id(specimen: str, stage: str) -> str:
    return f"felt-20ply-{specimen}-{stage}"


def checkpoint_path(specimen: str, stage: str) -> Path:
    return OUTPUT / f"{specimen}_{stage}.restart"


def checkpoint_settings(
    specimen: str, stage: str, *, resume: bool = False
) -> tangle.CheckpointSettings:
    """Checkpoint for one specimen and stage.

    ``resume=True`` continues this stage's own interrupted run. Otherwise
    ``cleanup`` and ``polish`` start from the checkpoint that the previous
    stage of the same specimen saved.
    """
    checkpoint = tangle.CheckpointSettings(
        case_id(specimen, stage),
        checkpoint_path(specimen, stage),
        interval_iterations=500,
        resume=resume,
    )
    if resume or stage == "form":
        return checkpoint
    source = SOURCE_STAGE[stage]
    return checkpoint.replace(
        resume=True,
        resume_path=checkpoint_path(specimen, source),
        resume_case_id=case_id(specimen, source),
        fresh_formation_on_resume=True,
    )


def run(
    specimen: str = "control", stage: str = "form", *, resume: bool = False
) -> tangle.RunResult:
    recipe, settings = build(specimen, stage)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_settings(specimen, stage, resume=resume)
    result = recipe.run(settings, checkpoint=checkpoint)
    stem = f"{specimen}_{stage}"
    result.write_ovito(
        OUTPUT / f"{stem}.dump",
        view_script_path=OUTPUT / f"{stem}_view.py",
        session_path=OUTPUT / f"{stem}.ovito",
        coloring="curvature_ratio",
    )
    result.export_bpm(
        OUTPUT / f"{stem}_capsules.data",
        mode="spherocylinders_exact",
        density=1_800.0,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--specimen", choices=SPECIMENS, default="control")
    parser.add_argument("--stage", choices=STAGES, default="form")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue this stage's own interrupted checkpoint",
    )
    args = parser.parse_args()
    print(run(args.specimen, args.stage, resume=args.resume))
