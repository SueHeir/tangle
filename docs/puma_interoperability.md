# TANGLE–PuMA analysis and interoperability

Use **Python for the analysis workflow**: construct or run a TANGLE configuration,
call its native analysis, export a voxel bundle, and call `pumapy` directly.
PuMA is optional, not a TANGLE runtime dependency, and there is no
`tangle.interop.puma` adapter.

Start with the [Python script](../crates/tangle_python/python/examples/puma_cross_validation.py)
or [notebook](../crates/tangle_python/python/examples/puma_cross_validation.ipynb).
Install TANGLE using the [Python guide](../crates/tangle_python/README.md).
Install PuMA separately following its [official instructions](https://puma-nasa.readthedocs.io/en/latest/).
Choose a Python/NumPy environment supported by both packages; the saved-felt
comparison was tested with Python 3.12, NumPy 1.26.4, and PuMA 3.2.2. Those are
tested versions, not a claim that every newer combination works.

```console
python crates/tangle_python/python/examples/puma_cross_validation.py
```

Without PuMA, this script still writes TANGLE analysis and voxel output and
explicitly skips the external comparison. No GPU is needed for this static
fixture. It constructs two separated orthogonal fibers without relaxing them.

## Python entry points

`Assembly.characterize()` and `RunResult.characterize()` return an
`AnalysisReport`. `export_puma()` is available on both objects:

```python
# Given an existing assembly or a completed recipe result:
analysis = result.characterize()
analysis.write_json("output/native_analysis.json")
bundle = result.export_puma(
    "output/specimen.puma",
    voxel_size=2e-6,
    include_fiber_ids=True,
    include_interface=True,
)
print(analysis.nominal_swept_volume_fraction)
print(bundle.voxel_volume_fraction)
```

Both methods call public Rust APIs; Python does not duplicate the geometry or
voxelizer. Native equivalents are `characterize_assembly`,
`write_analysis_json`, and `write_puma_bundle`. The
[paired Rust fixture](../examples/puma_cross_validation) uses the same geometry
and can be compared byte-for-byte with the Python fixture's VTI/native JSON.

## Measurements available now

| Measurement | TANGLE | PuMA / direct Python analysis |
| --- | --- | --- |
| Total and per-material solid fraction | Nominal area × centerline length / cell volume | Occupied voxel fraction |
| Orientation | Length- and volume-weighted centerline tensors | Accumulation of exported tangents, or independent image-based orientation |
| Fiber statistics | Per-fiber length, equivalent diameter, stretch, maximum curvature, bend utilization; counts and material summaries | No individual-fiber reconstruction in this workflow |
| Surface area | Nominal lateral area derived from radius and length in the section report | Marching-cubes surface area |
| Void intercept lengths/connectivity | No native counterpart yet | PuMA mean intercept length and pore labeling |
| Transport/continuum mechanics | No native characterization solver | PuMA conductivity, diffusivity/tortuosity, permeability, radiation, elasticity |

PuMA's property methods are described in its
[analysis API](https://puma-nasa.readthedocs.io/en/latest/python_api/pumapy.material_properties.html).
Native TANGLE currently reports persistent junction **counts**, not a complete
contact graph, coordination distribution, or connectivity analysis.

### Match definitions before comparing

- **Nominal volume is not union volume.** TANGLE sums cross-sectional area ×
  placed length. It excludes caps and does not remove overlap at bends or
  between fibers. Voxelization measures the union of capped circular segments.
  A difference in VF alone is not an overlap diagnostic.
- **Match orientation weighting.** Length weighting treats each unit of
  centerline equally. Volume weighting uses area × segment length and is the
  appropriate comparison for mixed diameters. Tensors are sign-invariant and
  have trace one for nonempty input.
- **Exported tangents are not independent detection.** Accumulating the imported
  tangent field checks rasterization/weighting. PuMA structure-tensor estimation
  from the solid image provides an independent orientation comparison.
- **Surface definitions differ.** Nominal `2πrL` includes surfaces hidden at
  contacts; voxel marching cubes measures the union interface. PuMA 3.2.2's
  specific-area normalization uses `(Nx−1)(Ny−1)(Nz−1)h³`; the section report
  also divides its measured area by `Nx Ny Nz h³` for a matched crop volume.
- **Fiber waviness is not pore transport tortuosity.** The latter requires a
  transport calculation through the void.
- **Voxel connectivity is not bonding.** A connected solid voxel cluster does
  not establish TANGLE junctions, friction, or mechanically bonded contacts.
  PuMA continuum elasticity is not equivalent to a DIRT bonded-particle test.

## Bundle format (schema 1)

```text
specimen.puma/
├── domain.vti
├── fiber_ids.vti       # optional
├── interface.vti       # optional
├── manifest.json
└── tangle_analysis.json
```

- `domain.vti`: cubic, cell-centered VTK ImageData with UInt16 `phase_id`
  (0=void, positive IDs mapped to materials) and Float32 `orientation`
  (unit tangent, zero in void). These two arrays can be imported directly into
  one PuMA workspace.
- `fiber_ids.vti`: one-based owner export IDs, mapped back to stable source
  fiber IDs in the manifest.
- `interface.vti`: UInt8 smooth signed-distance-derived interface, solid
  threshold 128, useful for surface reconstruction. It is not a material map.
- `manifest.json`: cell, grid, array conventions, material/fiber maps,
  occupancy/ownership rules, ambiguity count, source provenance, and file
  sizes/FNV-1a hashes. Hashes are integrity fingerprints, not cryptographic
  signatures.
- `tangle_analysis.json`: schema-versioned native metrics, including material
  and per-fiber tables. Values use assembly length units (areas/volumes use
  their powers); orientation and curvature utilization are dimensionless.
  This is a structured report, not a per-metric ontology or histogram schema.

A voxel is occupied when its center lies inside a swept capsule. Ownership
uses minimum signed capsule distance, then a stable source fiber ID for ties.
Incident same-fiber segment ties use sign-aligned, length-weighted tangents.
The ambiguity count concerns ownership ties, not all intersecting volumes.

Only circular sections and diagonal orthorhombic cells are currently supported.
Cubic voxel size must tile every cell edge within the exporter's tolerance;
invalid grids return an error and nearest integer counts. No cell dimensions
are silently rescaled. Periodic images intersecting the fundamental cell are
included; bounded faces clip the raster window. PuMA boundary conditions must
still be chosen explicitly: an imported workspace does not carry all TANGLE
boundary semantics into a downstream solver.

The voxelizer currently runs on the **host**, not the resident relaxation
device. Fine full-domain grids can require substantial memory; start with
small windows and resolution studies.

## Direct PuMA comparison

After exporting a bundle, use PuMA normally:

```python
import numpy as np
import pumapy as puma

ws = puma.import_vti(str(bundle.domain_path), import_ws=True)
solid = (1, int(ws.matrix.max()))  # requires a nonempty solid phase
vf = puma.compute_volume_fraction(ws, solid)
mask = ws.matrix > 0
tangents = ws.orientation[mask].astype(np.float64)
voxel_tensor = tangents.T @ tangents / len(tangents)

# This second calculation infers directions from the image, not the import.
detected = ws.copy()
puma.compute_orientation_st(detected, solid, edt=True)
errors, mean_angle, std_angle = puma.compute_angular_differences(
    ws.matrix, ws.orientation, detected.orientation, solid
)
```

Structure-tensor smoothing parameters are in voxel units. For a resolution
study, keep their physical lengths fixed. Report sign-invariant angular errors,
including an interior subset to identify crop-edge effects.

## Admissibility and resolution limits

**Export does not certify mechanical acceptance.** The exporter validates
assembly invariants and grid settings, but does not independently recompute
penetration or reject all mechanically inadmissible configurations. Schema 1
does not store a solver acceptance certificate. Inspect `RunResult.converged`,
residuals, warnings, and recipe policies before claiming a relaxed result.
Static `Assembly` exports are also useful for deliberately unrelaxed fixtures.

Use at least three voxel sizes and report changes separately for VF,
orientation, surface area, connectivity, and each transport property. The
smallest fiber diameter controls required resolution. Do not assume nominal VF
is the exact fine-grid limit, or that convergence must be monotonic.

Full distributions, contact-state/graph summaries, analytic ray/capsule void
intercepts, exposed union surface estimation, elliptical voxelization, and an
export acceptance certificate remain **future work**.

## Saved 1 mm needled-felt section

The [section report workflow](../examples/puma_cross_validation/README.md#saved-needled-felt-section-report)
analyzes a central 240 µm cube of the saved twenty-ply needled specimen at
3, 2, and 1 µm voxel sizes. It includes phase VF, exact/exported and independently
detected orientation, surface area, void intercept lengths, and connectivity.

For the archived specimen, native VF was 9.828% versus PuMA 9.782% at 1 µm;
independent mean orientation error was 2.04°. This crop intersects 96 fibers
and is less dense than the whole specimen (~13% nominal VF). These are sample
results, not a universal accuracy guarantee or a representative-volume claim.

The legacy restart could not be decoded by current structs, so the report
reconstructs geometry from the final capsule export and verifies periodic
endpoint continuity. It cannot establish rest strain, stored bend limits, or
solver convergence. Source data and generated reports remain in ignored output
directories and are not distributed with a fresh clone.

A matched control/needled study across multiple windows, followed by transport
solves, is a next step—not a completed result of the current report.
