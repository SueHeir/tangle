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
| Contacts and neighbors | `characterize_neighbors()`: contact events, crossing angles, in-axis/out-of-axis split, excess persistence, free lengths, neighbor counts, neighbor turnover | No counterpart; voxel connectivity is not contact |
| Fiber shape | `characterize_shape()`: curvature and torsion distributions, tangent correlation and persistence length, curl index, Schladitz β orientation fit | No individual-fiber reconstruction in this workflow |

PuMA's property methods are described in its
[analysis API](https://puma-nasa.readthedocs.io/en/latest/python_api/pumapy.material_properties.html).
Persistent junctions are still reported only as counts; contacts are
characterized separately, as below.

## Contacts and neighbors

`characterize_neighbors()` samples every fiber at a uniform arc-length spacing
and, at each sample, finds the closest approach to every other fiber's
centerline. The surface gap is that axis distance minus both radii (elliptical
sections use the equal-area radius). Periodic axes use minimum-image
distances and must be orthorhombic.

- **Contact:** another fiber within `contact_gap`. A **contact event** is one
  maximal run of samples along a fiber over which the same other fiber stays a
  contact. Each contact is counted once from each participating fiber.
- **Neighbor:** another fiber within `neighbor_gap` (default: twice the largest
  radius).
- **In-axis / out-of-axis:** whether the acute angle between the two tangents
  is below `in_axis_angle_degrees` (default 20°).
- **Random baseline:** contacts per unit length expected if the same fibers,
  with the same length density and orientation distribution, were placed
  independently with overlap allowed, `2 λ_L E[(r_i + r_j + gap) sin γ]` (the
  Onsager excluded volume of two cylinders). It ignores fiber ends, which adds
  a few percent for fibers much longer than their diameter, and it uses the
  cell volume, so it is only meaningful when fibers fill the cell.
  `contact_ratio_to_random` above one means more contact than random
  placement, below one means less.
- **Excess persistence:** an out-of-axis event's length divided by the length
  a straight crossing at the same angle and closest approach would give,
  `2 √((r_i + r_j + gap)² − d_min²) / sin γ`. Random crossings give about one;
  fibers that wrap around each other give more. It is not defined for in-axis
  events; use `median_in_axis_contact_length` and turnover instead.
- **Neighbor turnover:** the mean Jaccard similarity between a fiber's
  neighbor sets at `s` and `s + lag`. `neighbor_correlation_length` is the lag
  at which it falls to `1/e`, or `None` if neighbors do not turn over within
  the largest lag (for example, a straight parallel bundle).
- **Contact-count dispersion:** variance over mean of contacts per fiber.
  Random placement of equal-length fibers gives about one; clustering gives
  more.

When comparing with a CT scan, run the same call on the tracked centerlines
(inserted with `Assembly.insert()`) using the same gaps and angle. CT cannot
resolve gaps smaller than about one voxel, so a contact tolerance of that order
is appropriate on both sides. Fiber tracking also truncates fibers at the scan
boundary; compare per-length quantities rather than per-fiber counts.

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

## Fiber shape

`characterize_shape()` resamples every fiber at a uniform arc-length
`sample_spacing` and treats the chords between successive samples as its
tangents. Everything is measured at that scale, so compare two structures only
at the same spacing. The default is the larger of the smallest fiber diameter
and the median polyline segment length: CT trackers resolve direction changes
over about one diameter, and a finer spacing on a coarse polyline would only
see its corners.

- **Curvature:** turning angle between the chords on either side of a sample,
  divided by the spacing. It is exact for a circular arc.
- **Torsion:** signed rotation of the binormal about the tangent per unit
  length, positive for a right-handed helix. It is only measured where the
  curvature on both sides is at least `min_torsion_curvature` (default a
  turning angle of 0.02 rad per spacing), because the binormal of a nearly
  straight fiber is noise. `torsion_defined_fraction` says how often that held.
- **Tangent correlation:** mean `t(u) · t(u + s)` over every fiber and position.
  `tangent_correlation_length` is the lag at which it falls to `1/e`.
  `persistence_length` is `L` from a least-squares fit of `exp(−s / L)` over the
  lags before the correlation drops below 0.05. That is the persistence length
  of a three-dimensional worm-like chain; a chain confined to a plane decays as
  `exp(−s / 2L_p)`, so its planar persistence length is half this value.
  Periodic waviness (a fixed sine) makes the correlation oscillate rather than
  decay; read the curve before trusting either length.
- **Curl index:** contour length over end-to-end distance, minus one, per fiber
  (zero for a straight fiber). CT tracking truncates fibers at the scan edge,
  which lowers it, so compare it on fibers of similar length.
- **Orientation:** `axis_cosine` is `|cos θ|` between each chord and
  `orientation_axis` (default z, through the thickness), and
  `mean_squared_axis_cosine` its mean square (one third when isotropic).
  `schladitz_beta` is the maximum-likelihood fit of the Schladitz et al. (2006)
  density `β / (4π (1 + (β² − 1) cos² θ)^{3/2})`: `β = 1` is isotropic, `β < 1`
  aligns fibers with the axis and `β > 1` lays them in the plane normal to it.
  `schladitz_fit_distance` is the Wasserstein distance between the observed and
  fitted `|cos θ|` distributions; a large value means the one-parameter model
  does not describe the structure (for example, a bimodal one).

Distributions are returned as `count`, `mean`, `standard_deviation` and evenly
spaced `quantiles` (minimum first, maximum last; `quantile_count`, default 101).
The quantile function is enough to plot a distribution and to compute its
Wasserstein distance to another, which is how a generated structure will be
scored against a scan.

## Scoring against a scan

`tangle.score_structure(candidate, reference, contact_gap)` compares a
generated structure with a reference, usually centerlines fitted to a CT scan
(`tangle.ct` `fit.to_assembly()`, or tracked centerlines added with
`Assembly.insert()`). Both must use the same length unit.

A raw difference cannot say whether a structure matches: a 10% difference in
median curvature may be inside the scan's own variation or far outside it. The
scorecard therefore cuts the reference region into `subdivisions` subvolumes
(default 2×2×2), tiles the candidate region with subvolumes of the same
physical size, and measures every metric in each. For each metric:

```text
score = median distance over (candidate, reference) subvolume pairs
        ───────────────────────────────────────────────────────────
        median distance over (reference, reference) subvolume pairs
```

The distance is the absolute difference for a scalar and the Wasserstein
distance for a distribution. A score near one means the candidate differs from
the scan about as much as the scan differs from itself at that scale; well
above one is a real difference. `Scorecard.table()` lists the metrics worst
first.

| Kind | Metrics |
| --- | --- |
| Scalars | `volume_fraction`, `length_density`, `mean_squared_axis_cosine`, `log_schladitz_beta`, `persistence_length`, `tangent_correlation_length`, `contacts_per_length`, `contact_ratio_to_random`, `in_axis_contact_fraction`, `mean_neighbors`, `neighbor_correlation_length` |
| Distributions | `curvature`, `absolute_torsion`, `curl_index`, `axis_cosine`, `fiber_length`, `crossing_angle`, `free_length`, `excess_persistence` |

- **Same settings on both sides.** Sample spacings, the neighbor gap, lag
  ranges and the torsion threshold are resolved once from the whole reference
  region and reused for every subvolume; the resolved values are in the
  report's `shape` and `neighbors` settings.
- **Cropping.** Each subvolume is analyzed as a non-periodic box. Fibers are
  clipped at its faces (periodic images included), and pieces shorter than
  `min_piece_length` (default the largest reference fiber diameter) are
  dropped, as a CT tracker drops fibers clipping a corner. Truncation shortens
  fibers and lowers the curl index equally on both sides because the
  subvolumes have the same size.
- **Regions.** `candidate_region` and `reference_region` take `(lower, upper)`
  corners. Restrict the candidate to the part its fibers actually fill, for
  example the thickness of a generated stack, or the volume fraction will be
  diluted by empty cell.
- **Undefined scores.** A score is `None` when a metric is undefined in the
  subvolumes (no contacts, no decay of the tangent correlation) or the
  reference spread is zero. At least two reference subvolumes are required.

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

Install PuMA with `conda create -n puma conda-forge::puma` and build TANGLE into
the same environment (see the [installation guide](../crates/tangle_python/README.md)).
NASA's PuMA ships `pumapy` through conda-forge, not PyPI: `pip install pumapy` installs an unrelated package with the same name.

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

Contact-graph connectivity (components, clustering), analytic ray/capsule void
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
