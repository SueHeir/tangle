# How `tangle.ct` fits fibers, step by step

This is the exact procedure in the code. Every step names its file and function
under `crates/tangle_python/python/tangle/ct/`. [ct_fitting.md](ct_fitting.md)
is the user guide.

## Conventions

- Internally, every length is in voxels and points are `(x, y, z)`. Voxel
  `[k, j, i]` of the `(z, y, x)` array is centered at `(i+½, j+½, k+½)`, the
  same grid `Assembly.export_puma` writes. Meters are used only at the edges:
  `FiberSpec`, `fit.json` and Tangle objects.
- `r` is the nominal radius, `FiberSpec.diameter / 2 / voxel_size`.
- A fiber is a polyline centerline with one radius, the same model as a Tangle
  fiber: a chain of capsules (sphero-cylinders).
- Nodes are kept about `r` apart (`FitSettings.node_spacing_radii = 1`), so
  each segment is roughly as long as it is thick.

## 1. Normalize the scan

`_image.normalize`

1. Blur with a Gaussian of σ = 0.7 voxels to take the edge off the noise.
2. Pick a threshold with Otsu's method on a subsample of slices. The void
   level and the fiber level are the medians of the voxels below and above it.
3. Rescale the image so void ≈ 0 and fiber ≈ 1. Everything after this works
   on this normalized image.

The fiber level from step 2 is refined later (step 5), because blurred edge
voxels pull the upper class median below the true fiber core.

## 2. Local tube direction

`_image.HessianField`

The Hessian (the second derivatives) of the normalized image is computed at
Gaussian scale σ = max(0.6 r, 1). It is sampled trilinearly at any point, and
its eigenvalues are sorted by magnitude:

- the fiber axis is the eigenvector of the smallest-magnitude eigenvalue;
- the tubularity is −(λ₂ + λ₃)/2, which is positive on a bright tube and
  about zero elsewhere.

## 3. Seeds

`_trace.ridge_seeds`

1. Take the foreground, the voxels with a normalized value above 0.5.
2. Compute the Euclidean distance transform of the foreground.
3. Seeds are ridge voxels: at least max(r/2, 1) deep, and a local maximum in
   their 3×3×3 neighborhood.
4. Seeds are processed deepest first, so tracing starts in fiber cores and
   crossings are met later.

## 4. Trace initial centerlines

`_trace.trace_fibers`, `_trace.Tracer`

From each seed not yet claimed by an earlier trace, the tracer walks both
ways along the Hessian axis. The step is max(0.75, r/2) voxels. The largest
turn per step is min(step / bend radius, 45°), so a trace can never bend
more sharply than `FiberSpec.min_bend_radius` allows.

Each step:

1. **Advance:** candidate = position + step × direction.
2. **Re-center** (`Tracer.recenter`): sample the cross-section, a disk of
   radius 1.4 r perpendicular to the direction, on a 0.5-voxel grid.
   - Each sample is weighted by its intensity (clipped at 0) and a Gaussian
     falloff of width 0.8 r.
   - Samples already claimed by another trace are weighted ×0.15, so a trace
     crosses another fiber instead of turning onto it.
   - The candidate moves to the weighted centroid, by at most 0.4 r.
3. **New direction:** normalize(0.5 × old direction + 0.15 × the step just
   taken + 0.35 × the Hessian axis). The Hessian term is used only where the
   tubularity exceeds 0.05. The turn is then clipped to the bend limit.
4. **Stop:**
   - when the trace leaves the volume, or
   - after two consecutive steps whose core intensity is below 0.5. The core
     intensity is the mean over a disk of radius r/2.
   Trailing points below 0.5 are dropped.

Then:

- **Duplicates** (`_trace._drop_claimed`): end runs that lie inside earlier
  traces are trimmed. A trace more than 30% inside earlier traces is
  rejected.
- **Kept traces:** a trace at least `min_length` long (default 3 diameters) is
  resampled to the node spacing. It claims the voxels within 1.1 r of it.
- **Rejected traces:** they are painted as "tried" within 0.75 r, so nearby
  seeds on the same blob are not traced again.

## 5. Re-level

`_fit._relevel`

Once the traces exist, the levels are measured again:

- the fiber level is the median intensity along all trace nodes (the true
  fiber cores);
- the void level is the median of the below-threshold voxels farther than
  2.5 r from any trace.

The image is rescaled to these levels. This fixed a +7% diameter bias.

## 6. The fit: rounds of continuous refinement and topology moves

`_fit.fit_fibers`

There are `rounds` rounds (default 3). Each round runs 12 iterations of the
continuous fit (6a), then the topology moves (6b). Every round except the
last then starts new fibers (6c).

### 6a. Continuous fit, one iteration

`_refine`

This is an EM-style loop. Ownership is the E-step; the moves are the M-step.

1. **Ownership and data force** (`data_step`).
   - Every voxel within 1.6 r of any centerline is assigned to the one
     segment whose capsule surface is nearest, that is, the smallest
     (distance to the centerline − that fiber's radius). This is the rule
     Tangle's voxel exporter uses. `_geometry.rasterize` does it per segment,
     in a window around the segment.
   - For each segment, take the centroid of its owned voxels, each weighted
     by intensity clipped to [0, 1.5].
   - Take the part of (centroid − segment midpoint) perpendicular to the
     segment.
   - Each node moves by 0.6 × the average of the shifts of its two
     segments.

   The force is lateral only; ends are handled in step 4.
2. **Bending** (`bend_step`): each interior node moves 0.3 of the way to the
   midpoint of its neighbors. This is a discrete Laplacian, which acts as
   the fiber's stiffness.
3. **Radius** (`radius_step`).
   - The owned intensity mass is the sum over owned voxels of intensities
     clipped to [−1.5, 1.5]. It is unclipped at 0, so zero-mean noise in
     the void cancels out.
   - Treating that mass as the volume of a capsule gives a measured radius,
     √(mass / (π (L + 4r/3))). The 4r/3 accounts for the two hemispherical
     caps.
   - The radius is blended 50/50 with the nominal radius and clamped to
     `diameter × (1 ± diameter_tolerance) / 2`.
4. **Ends** (`end_step`): each end takes up to 3 moves.
   - Probe the image at max(spacing, r/2) beyond the tip, along the tip's
     tangent.
   - If the probe reads above 0.55 and no other fiber owns that voxel, add a
     node there.
   - Otherwise, if the intensity at the tip is below 0.45, remove the tip
     node.
5. **Non-overlap** (`separate_step`): any two nodes of different fibers
   closer than the sum of their radii are pushed apart symmetrically by the
   overlap, in 2 passes.
6. **Respace** (`respace`): resample every centerline back to the node
   spacing.

There is no explicit forward model of the scanner's blur in this loop. An
intensity-weighted centroid is unbiased for a symmetric point-spread
function, so it doesn't need one.

### 6b. Topology moves, once per round

`_moves`, run in this order:

1. **Split** (`split_kinks`).
   - The turn at each node is measured over 3 node spacings on each side.
   - A kink is a turn larger than both 2 × the turn the bend radius allows
     over that window and 35°.
   - A fiber is cut at its worst kink, repeatedly. Such a kink usually means
     a trace ran from one fiber onto another, and the fit then pulled each
     part onto its own fiber.
   - Fits longer than `max_length`, if one is given, are cut where the
     smoothed image support along them is weakest.
   - Pieces shorter than `min_length` are dropped.
2. **One fiber or two** (`resolve_side_by_side`), for each fit i, weakest
   first:
   - Find the fit j that most of i's nodes lie within 2.3 r of. At least 3
     nodes must be close.
   - Render the neighborhood (the box around both, plus a 2.5 r margin,
     including every other fiber that reaches into it) twice, as a soft
     capsule occupancy with a 1.2-voxel edge (`render_occupancy`):
     - "both": i and j as they are;
     - "one": j moved to the midline of i and j over their overlap, and i
       reduced to its pieces outside the overlap.
   - Keep whichever has the smaller sum of squared differences from the
     image.

   This is needed because the non-overlap step pushes two fits on one fiber
   about a radius apart each, where they look like two touching fibers.
3. **Duplicates** (`trim_duplicates`), shortest fit first:
   - Nodes within 0.8 r of another fit's nodes are "covered". Two real
     fibers can't be that close.
   - A fit more than half covered is removed.
   - Otherwise its covered end runs are trimmed.
4. **Unsupported** (`remove_unsupported`): fits shorter than `min_length`, or
   with mean normalized intensity along the centerline below 0.5, are
   removed.
5. **Join** (`merge_fragments`). Two ends are joined when all of these hold:
   - they are within 4 r (`merge_gap_radii`);
   - their tangents face each other within 35°;
   - the gap direction agrees with both tangents;
   - the mean intensity along the straight bridge is at least 0.45;
   - the joined line has no kink by the rule in move 1.

   The best pair is joined first, by distance × (2 − cosine of the facing
   angle), and the search repeats until no pair qualifies.
6. **Respace** all fits.

### 6c. New fibers

At the end of each round except the last, the current fits claim the voxels
within 1.2 r of them. `trace_fibers` then runs again from ridge seeds in the
unclaimed foreground, with the same `_drop_claimed` rule, and the new traces
join the fit.

Each round logs:

- the fiber count, splits, duplicates and merges;
- the explained fraction, the share of foreground voxels inside a fitted
  capsule;
- the interior-end statistics below.

These go into `fit.json`'s `history`.

## 7. The fiber-length prior (`FiberSpec.length`)

`_ends`, used by the moves in 6b

Without a `length`, sections 1–6 are the whole method. With one:

- **Price of an end** (`_ends.end_cost`): if fiber ends are spread uniformly,
  a break occurs about once per `length` of fiber. A break's position can be
  resolved to about a diameter, so an interior end costs ln(length / diameter)
  nats. Ends within 1.5 r of the scan boundary are fibers leaving the scan
  and cost nothing (`_ends.interior_end_mask`).
- **Evidence scale** (`_ends.evidence_scale`), measured each round.
  - Render all fits (soft occupancy) and take σ², the mean squared residual
    per voxel within r + 2 of a fit. That is noise plus model misfit.
  - One nat of evidence is a change in squared residual of 2 σ² π r², one
    fiber cross-section's worth. Neighboring voxels move together on about
    that scale, so they aren't counted as independent.
- **Split** (`split_kinks`): before cutting at a kink, the nodes within 6 of
  it are smoothed (`_moves._smooth_kink`) until no kink is left.
  - If the smoothed fit's residual is not worse than the kinked one's by more
    than the two new ends would cost, the fit is kept whole, smoothed.
  - A trace that jumped fibers can't be smoothed without leaving both, so it
    is still cut.
- **Join** (`_moves._merge_with_prior`) replaces the fixed rules in move 5.
  - **Candidates:** any two ends within 16 r (`prior_merge_gap_radii`) that
    face each other within 45° (`prior_merge_angle_degrees`). Pieces whose
    ends overlap by up to 2 r (laterally within 1.5 r) also count.
  - **The joined line** (`_moves._join`) cuts both pieces back at the plane
    through the midpoint of their tips, then connects them.
  - **Score:** (residual of the two pieces − residual of the joined fiber) /
    scale + 2 × end cost. Candidates that would kink beyond the bend limit
    are skipped. Only candidates scoring above zero are kept.
  - **Pairing:** candidates are accepted best first, each end at most once,
    with no joins that close a loop (union-find), and chains are then
    stitched together. This pairs pieces at a crossing the way the scan
    supports best, rather than first come, first served.
- **One fiber or two** (`resolve_side_by_side`): the residual comparison
  also counts ends. The "one" rendering removes i's two ends but adds two per
  leftover piece.

The statistics (`_ends.end_statistics`) are computed whether or not a length
is given:

- interior ends N;
- total fitted length Λ inside the scan;
- the implied mean length 2Λ / N;
- with a `length`, the expected count 2Λ / length ± its square root.

They appear per round in `history`, in `population_summary()` and in the
score.

## 7b. Several fiber types

`_fit.fit_fibers`, `_fit._fit_type`, `_profile.CrossSection`

When `spec` is a list, or its profile isn't solid at brightness 1:

- **Levels:** `_fit._class_levels` splits the histogram into one class for
  void plus one per distinct brightness level of the types (rim and core
  count separately; 2 to 4 classes) by exhaustive multi-level Otsu. For 7 µm
  solid plus 19 µm rim/core fibers that is 4 classes: void, large-fiber core,
  large-fiber rim and small-fiber edge, small-fiber center. Void is the
  darkest class median and brightness 1 the brightest class median. The
  re-level step (5) is skipped.
- **Order:** brightest type first (by `CrossSection.brightness`), larger
  diameter first among equally bright types. Each type goes through sections
  3–7 in full, on its own detection image (`_fit._detection_image`):
  - A solid type brighter than every type fitted after it sees only the
    brightness above those types' brightest level `b`:
    `clip((image − b) / (brightness − b), 0)`. The 7 µm fibers are then the
    only thing in their image; the 19 µm rims (0.75) read 0.
  - A rimmed type sees the scan blurred with σ = r/2, divided by the blurred
    profile's value on the axis (`CrossSection.center_response`).
  - Any other solid type sees the scan divided by its brightness.
- **Fitted types are removed:** voxels within r + 2 of every fiber already
  fitted are set to void in the image the later types see (their detection
  image and their mass image). The earlier fibers also stay frozen:
  - they block tracing (their voxels within 1.2 r are pre-claimed);
  - they own voxels in the data force (6a.1) and in end growth (6a.4);
  - they push in the non-overlap step (6a.5) without moving.
- **Rim/core check** (`_moves.remove_off_profile`, each round after 7.4, for
  rimmed types): on the normalized scan, the median brightness on the fit's
  axis must be below the midpoint of the rim and core levels, and the median
  on a 12-spoke ring at r − rim/2 must exceed the axis by a quarter of the
  rim-core contrast. A cluster of solid bright fibers fails the first test;
  a fit running along one side of a rim fails the second.
- **Radii:** the owned mass is summed over the image this type sees, not the
  detection image, and converted to a radius by inverting the type's
  integrated profile (`CrossSection.radius_from_area`).

## 8. Where Tangle's own code comes in

- **Ownership rule:** the fitter uses the same nearest-capsule-surface rule
  as Tangle's voxel exporter, so a fit re-voxelized by Tangle gives the same
  labels.
- **Outputs:** `FitResult.to_collection` / `to_assembly` build a Tangle
  `FiberCollection` / `Assembly`. The cell is the scanned box, not periodic.
  There is one `Material` per fitted diameter, rounded to 10 nm, each with
  the spec's bend limit.
- **Relaxation after the fit:** the optional `FitResult.relax()` runs Tangle's
  contact relaxation, on the default GPU backend, to remove leftover overlaps.
  By default (`FitSettings.engine="numpy"`) Tangle's solver is not called
  inside the fitting loop; the Python `separate_step` and `bend_step` stand
  in for it. With `engine="tangle"` it is (section 8b).
- **Synthetic scans** (`_synthetic.synthetic_ct`): the ground truth is a
  Tangle structure.
  1. `export_puma(include_interface=True, include_fiber_ids=True)` gives
     partial-volume occupancy and per-voxel fiber IDs.
  2. The scan is attenuation = 0.05 + 0.95 × (occupancy blurred with a
     σ = 0.9 voxel Gaussian), plus Gaussian noise (σ = 0.12) and a weak
     cosine drift, scaled to uint16.
  3. `SyntheticScan.crop` cuts a window from a larger render, so fibers cross
     its boundary as in a real scan.

## 8b. The fit on Tangle's solver (`FitSettings(engine="tangle")`)

`_device.refine`, `tangle.ImageRelaxer` (kernels in
`crates/tangle_relax/src/device/image_force.rs`)

With `engine="tangle"`, each round's continuous fit (6a) is replaced by
`solver_batches` (3) batches on Tangle's own relaxation, the GPU by default
(`FitSettings.backend`). The topology moves (6b) and births stay as they
are. One batch:

1. **Upload.** Fibers are resampled to segments of 1.25 diameters (never
   shorter than one: Tangle's contact treats non-adjacent segments of one
   fiber as colliding) and placed in a closed cell padded by 3 r around the
   scan. Each fiber gets a material with its fitted diameter and the spec's
   bend limit, and a straight rest shape with its own segment lengths:
   bending then resists every curve, and a kinked fit does not keep its
   kinks as its natural shape. `neighbor_capacity` is 192, because
   overlapping starts overflow the default 48 slots into a slow fallback.
2. **Relax with the image force** for `solver_iterations` (300). Every
   iteration applies Tangle's contact, stretch, bending and bend-limit steps
   and one image step: each vertex samples the normalized scan on a polar
   grid across the fiber (4 rings × 12 spokes out to `solver_reach_radii` =
   1.4 r, Gaussian σ = 0.8 r, area-weighted), keeps the samples nearer its
   own capsule surface than any other fiber's, and moves sideways toward
   their brightness-weighted centroid at `solver_image_rate` (0.3), scaled
   by the brightness of its innermost ring and capped at the max step.
3. **Read the owned intensity** of every vertex (`vertex_image_stats`: area
   in voxels² of owned brightness). The mean over interior vertices is each
   fiber's cross-section area, turned into a radius as in 6a.3 (profile
   inversion for non-solid types), blended with the spec radius and clamped
   to the tolerance.
4. **Settle** for `solver_settle_iterations` (100) with the image force
   off. The image step re-adds a little curvature each iteration before the
   solver removes it; the settle ends on geometry the constraints alone
   accept (it converges in a few dozen iterations).
5. **Ends** grow or trim on the host (6a.4), and fibers are respaced to the
   fitter's node spacing.

After the last round, one more batch makes the last splits and joins
admissible. A type fitted after another (several fiber types) still uses
the NumPy loop, since its frozen predecessors would have to stay fixed in
the solver.

Checked with `examples/ct_gpu_geometry_check.py` on the single-type
synthetic scan (Mac GPU): from the NumPy fit with the image off, 46
overlapping pairs and 3 fibers over the bend limit go to 0 in 26
iterations; true fibers damaged with random kinks come back to within 0.25
voxels of the truth with no overlaps and the bend limit met; polishing the
NumPy fit keeps its score (35 of 40 recovered).

## 9. Outputs

`FitResult.write`

- `fit.json` (schema `tangle.ct.fit/1`) contains:
  - the spec and the levels;
  - each fiber's centerline in meters, its diameter and its support (the mean
    normalized intensity along it);
  - `population` (`population_summary()`): diameter and length statistics,
    fibers touching the boundary, waviness, the orientation tensor, volume
    fraction and the end statistics;
  - `history`.

  `load_fit` reloads it.
- `labels.tif`: the fiber ID of every voxel inside a fitted capsule. A voxel
  inside two capsules goes to the nearer surface.
- `overlay.tif` / `overlay.png`: the raw scan in grey with each fiber's voxels
  tinted in its own color, as an RGB stack and as three orthogonal slices.
- `suggested_population()`: a `tangle.FiberPopulation` built from the fitted
  statistics. The orientation is aligned, planar or isotropic, from the
  orientation tensor's eigenvalues.

## 10. Scoring against ground truth

`_evaluate.score`

- Each fitted fiber's owner is the true fiber whose voxels most of its
  centerline samples fall in. Purity is that fraction.
- A true fiber is:
  - **recovered** if one fit owned by it lies within the true radius along at
    least 80% of its in-volume length;
  - **split** if only several fits together reach 80%;
  - **missed** otherwise.
- A fit is **false** if it is unowned or its purity is under 0.5, and
  **merged** if its purity is between 0.5 and 0.8.
- **Centerline error:** the mean distance from a fit's samples to its true
  centerline, for fits with purity ≥ 0.8.
- **Diameter bias / RMS:** over the same fits.
- **Solid Dice:** the overlap of fitted solid and true solid.
- **Voxel label accuracy:** the fraction of true solid voxels whose fitted
  label maps to the right true fiber.
- **Interior ends and implied lengths:** reported for both the fit and the
  truth.
