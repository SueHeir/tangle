"""TANGLE analysis -> VTI bundle -> direct PuMA analysis -> comparison.

PuMA is intentionally an optional, independent dependency. Install it in an
environment supported by PuMA, then run this file with that environment's
Python after installing the TANGLE extension.
"""

from __future__ import annotations

import json
from pathlib import Path

import tangle

CELL_LENGTH = 100.0e-6
FIBER_LENGTH = 80.0e-6
FIBER_DIAMETER = 10.0e-6
AXIS_SEPARATION = 12.0e-6
VOXEL_SIZE = 2.0e-6
OUTPUT = Path(__file__).with_name("output") / "puma_cross_validation.puma"


def make_orthogonal_pair() -> tangle.Assembly:
    """Build the same deterministic geometry as the native Rust example."""
    center = 0.5 * CELL_LENGTH
    half_length = 0.5 * FIBER_LENGTH
    half_gap = 0.5 * AXIS_SEPARATION
    material = tangle.Material("validation fiber", diameter=FIBER_DIAMETER)
    fibers = tangle.FiberCollection("orthogonal pair")
    fibers.add_fiber(
        [
            [center - half_length, center, center - half_gap],
            [center + half_length, center, center - half_gap],
        ],
        material,
    )
    fibers.add_fiber(
        [
            [center, center - half_length, center + half_gap],
            [center, center + half_length, center + half_gap],
        ],
        material,
    )
    assembly = tangle.Assembly(tangle.Cell([CELL_LENGTH] * 3))
    tangle.Recipe(assembly).insert(fibers, name="orthogonal pair")
    return assembly


def run_tangle() -> tuple[tangle.AnalysisReport, tangle.PumaExportReport]:
    assembly = make_orthogonal_pair()
    analysis = assembly.characterize()
    export = assembly.export_puma(OUTPUT, VOXEL_SIZE)
    print(
        f"TANGLE: nominal Vf={analysis.nominal_swept_volume_fraction:.6f}, "
        f"A_L={analysis.length_weighted_orientation_tensor}"
    )
    print(
        f"bundle: {export.voxel_counts} voxels, "
        f"voxel Vf={export.voxel_volume_fraction:.6f}, "
        f"ties={export.ambiguous_voxels}"
    )
    return analysis, export


def run_puma(
    analysis: tangle.AnalysisReport, export: tangle.PumaExportReport
) -> dict[str, object] | None:
    """Import PuMA directly; no TANGLE-side PuMA adapter is involved."""
    try:
        import numpy as np
        import pumapy as puma
    except ImportError:
        print(
            "PuMA comparison skipped: install PuMA from conda-forge "
            "(conda create -n puma conda-forge::puma) and build TANGLE into that environment."
        )
        return None

    workspace = puma.import_vti(str(export.domain_path), import_ws=True)
    highest_phase = int(workspace.matrix.max())
    solid_cutoff = (1, highest_phase)
    puma_volume_fraction = float(
        puma.compute_volume_fraction(workspace, solid_cutoff)
    )

    solid = (workspace.matrix >= 1) & (workspace.matrix <= highest_phase)
    tangents = workspace.orientation[solid]
    orientation = np.einsum("ni,nj->ij", tangents, tangents) / tangents.shape[0]
    reference = np.asarray(analysis.volume_weighted_orientation_tensor)
    comparison = {
        "voxel_size": VOXEL_SIZE,
        "tangle_nominal_swept_volume_fraction": (
            analysis.nominal_swept_volume_fraction
        ),
        "puma_voxel_volume_fraction": puma_volume_fraction,
        "volume_fraction_difference": (
            puma_volume_fraction - analysis.nominal_swept_volume_fraction
        ),
        "tangle_volume_weighted_orientation_tensor": reference.tolist(),
        "puma_exported_orientation_tensor": orientation.tolist(),
        "orientation_frobenius_error": float(np.linalg.norm(orientation - reference)),
    }
    comparison_path = OUTPUT / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"PuMA comparison: {comparison_path}")
    return comparison


if __name__ == "__main__":
    native_analysis, bundle = run_tangle()
    run_puma(native_analysis, bundle)
