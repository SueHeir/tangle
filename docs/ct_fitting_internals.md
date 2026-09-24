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

## 1. Input: grey ranges, a mask or a grey scan

`_fit._is_mask`, `_fit._mask_image`, `_ranges.range_image`, `_image.normalize`

With `FiberSpec.intensity = (low, high)` on every type, the raw scan is
read through the **grey ranges** (a mask input ignores them):

1. Blur with σ = 0.7 voxels. The void grey v is the median of the voxels
   darker than every range's low end.
2. For each type, a voxel's fiber fraction is 1 in [low, high];
   (g − v) / (low − v), clipped to [0, 1], below it (a partial-volume
   edge); and 1 − (g − high) / (high − low), clipped, above it, so noise
   just over the range still counts and a much brighter inclusion does
   not. The image is the largest fraction over the types.
3. Core-sized holes (as in the mask fill below) in the image > 0.5 are set
   to 1, and, per type, holes enclosed by that type's in-range voxels get
   that type's bit: a dim core inside a bright rim is fiber of the rim's
   type.
4. `exclude` voxels are set to 0 with no type bits. Levels: void v, fiber
   the middle of the first range, threshold halfway from v to the lowest
   range. No re-level (step 5).

Each voxel keeps one bit per type whose range it is in (`types`), used
for typing (7b).

With `FiberSpec.profile` on every type (`_grey`), the ranges come from the
profiles: on the denoised scan the void grey v is the median of voxels
darker than every profile, the noise σ_n 1.4826 × MAD of those below
halfway to it, and a type's range runs from its bright core (the dimmest
profile value within 1.5 voxels of the brightest one) to its brightest
value, widened by 2.5 σ_n (at least 5% of the contrast), with the low end
kept at least 60% of the way from v to the peak (an explicit `intensity`
wins). The core is what makes a thin, blurred fiber at least three voxels
wide; going further down the profile lets the dim contact between
touching fibers count as fiber, and they merge.
The denoised scan, profiles and v are kept for scoring: `render_grey`
draws fibers as the scan should show them (each voxel takes the profile
of the fiber whose surface is nearest, at d / R, and outside the surface
fades linearly from the profile's last value to v over 2 × 1.2 voxels;
`measure_profiles` measures on the blurred scan, so the blur inside the
fiber is in the profile).

A `bool` array, or one with only two values (the larger is fiber), is a
**mask**:

1. Holes in the mask no larger than a fiber core (area ≤ π (r_max + 1)²,
   r_max the largest type's radius) are filled, slice by slice along each
   axis (`_fit._core_holes`). A hollow fiber is a closed ring in the slices
   across it but a tube open at both ends in 3D, so a 3D fill would miss
   it; larger enclosed holes are void that crossing fibers happen to
   surround in a slice, and stay. `FitSettings.fill_mask_holes` turns this
   off.
2. `exclude` voxels are set to void.
3. The mask becomes a 0/1 image blurred with σ = 0.7 voxels, so the image
   force sees a smooth edge. Levels are void 0, fiber 1.

Anything else is a **grey scan**:

1. Blur with a Gaussian of σ = 0.7 voxels to take the edge off the noise.
2. Pick a threshold with Otsu's method on a subsample of slices. The void
   level and the fiber level are the medians of the voxels below and above it.
3. Rescale the image so void ≈ 0 and fiber ≈ 1, and set `exclude` voxels
   to 0. Everything after this works on this normalized image.

The fiber level from step 2 is refined later (step 5), because blurred edge
voxels pull the upper class median below the true fiber core. When fibers
are a small part of the scan, Otsu splits the noise histogram instead and
every noise blob is traced; a mask avoids estimating levels at all.

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

## 5. Re-level (grey scans, one type)

`_fit._relevel`

Once the traces exist, the levels are measured again:

- the fiber level is the median intensity along all trace nodes (the true
  fiber cores);
- the void level is the median of the below-threshold voxels farther than
  2.5 r from any trace.

The image is rescaled to these levels. This fixed a +7% diameter bias.

## 6. The fit: rounds of solver batches and topology moves

`_fit.fit_fibers`, `_fit._Fitter`

There are `rounds` rounds (default 3). Each round runs `solver_batches` (3)
batches of the continuous fit (6a), each followed by choosing every fiber's
type and radius (7b), then the topology moves (6b). Every round except the
last then starts new fibers (6c). After the last round one more batch makes
the last splits and joins admissible (end of 6a).

### 6a. Continuous fit on Tangle's solver, one batch

`_device.relax`, `tangle.ImageRelaxer` (kernels in
`crates/tangle_relax/src/device/image_force.rs`)

The fibers move only in Tangle's own relaxation, the GPU by default
(`FitSettings.backend`; `"cpu"` runs the same steps on the CPU).

1. **Ends** (`_refine.end_step`), on the host before the solve, so the
   solver also cleans up what end growth does. Each end takes up to 3
   moves. A capsule reaches r past its last node, so the foreground ends
   about r + δ (δ the mask margin, 7b) beyond the true end of the
   centerline:
   - if the image along the tip's tangent at r + δ + spacing/2 past the tip
     reads above 0.55 and no other fiber owns that voxel, add a node one
     spacing out;
   - otherwise, if the image at the tip, or at r + δ − spacing/2 past it,
     reads below 0.45, remove the tip node (the second test is skipped
     where it falls outside the scan, so fibers leaving the scan keep
     their ends);
   - this leaves the tip within half a spacing of the true end. (An earlier
     rule, grow while the image one spacing ahead is fiber and trim only
     where the tip itself is void, left traces that ran into the end cap
     about r too long.)
2. **Upload.** Fibers are resampled to segments of 1.25 of their own
   diameters (never shorter than one: Tangle's contact treats non-adjacent
   segments of one fiber as colliding) and placed in a closed cell padded by
   3 r_max around the scan. Each fiber gets a material with its radius and
   its type's bend limit, and a straight rest shape with its own segment
   lengths: bending then resists every curve, and a kinked fit does not
   keep its kinks as its natural shape. `neighbor_capacity` is 192, because
   overlapping starts overflow the default 48 slots into a slow fallback.
3. **Relax with the image force** for `solver_iterations` (300). Every
   iteration applies Tangle's contact, stretch, bending and bend-limit steps
   and one image step: each vertex samples the normalized scan on a polar
   grid across the fiber (4 rings × 12 spokes out to `solver_reach_radii` =
   1.4 r, Gaussian σ = 0.8 r, area-weighted), keeps the samples nearer its
   own capsule surface than any other fiber's (the rule Tangle's voxel
   exporter uses), and moves sideways toward their brightness-weighted
   centroid at `solver_image_rate` (0.3), scaled by the brightness of its
   innermost ring and capped at the max step. An intensity centroid is
   unbiased for a symmetric point-spread function, so no explicit blur
   model is needed.
4. **Settle** for `solver_settle_iterations` (100) with the image force
   off. The image step re-adds a little curvature each iteration before the
   solver removes it; the settle ends on geometry the constraints alone
   accept (it converges in a few dozen iterations).
5. **Cut void** (`_refine.cut_void`, `FitSettings.void_level` = 0.3).
   The image force only pulls toward fiber; void never pushes back, so a
   fit shoved off its fiber (by contact, or by a fit taking its place) can
   leave a tail in empty space that no later step notices (Liz, 2026-09-24:
   fibers "drift off into nothing, and another fiber will take over that
   location"). Every centerline node inside the scan whose fiber image
   reads below 0.3 is void. Void nodes at an end are trimmed back to the
   first supported node. An interior void stretch at least
   `void_gap_radii` (2) radii long is bridged by a straight line between
   its supported neighbors only when it is a real bow: it reaches at least
   `void_bridge_offset_radii` (1) radii off that line, and the line reads
   at least `void_bridge_level` (0.7) all the way (the fit bowed off its
   fiber and came back). Otherwise it splits the fit. (Audit against the
   truth on 7b99548: most stretches were one dim node about 0.3 r off a
   straight fit, where the fit hops from one fiber to another at a
   crossing. Their chord crosses the gap between touching fibers and reads
   0.4 to 0.6, and bridging them kept the merge: 82 of 111 wrong in the
   dense cube, 46 of 70 in two_types. The good bridges, which a test
   against other fits' cores had blocked, were bows of about 2 r with a
   chord of 0.9.) A stretch whose line reads at least `void_aligned_level`
   (0.5) and lies within `void_aligned_angle_degrees` (15°) of the scan's
   fiber axis (the type's Hessian) at both supported neighbors is bridged
   too: in the audit every such stretch was right (15 of 54 right ones, no
   hops). The next batch's end
   step regrows an end where the scan continues, and the topology moves
   rejoin pieces where they should be one fiber.
6. Types and radii are chosen again (7b) and fibers are respaced to the
   fitter's node spacing (r of the smallest type) for the topology moves.

The final batch's solver output is returned as the solver left it
(segments of 1.25 diameters, the radii and types it was solved with),
except that void is cut (step 5) and pieces then shorter than the minimum
length are dropped. (A join pass after this last cut was tried and
dropped: with the topology step's 16 radii and 45° it joined pieces of
different fibers at crossings, two_types 0.894 → 0.871, and limited to
touching, aligned ends it never fired.) Otherwise the fit is the
state the solver converged to. Measure it with `geometry_report(...,
spacing=1.25 * diameter)`: finer resampling puts nodes at the polyline's
corners and roughly doubles the discrete curvature there.

Checked with a since-removed `examples/ct_gpu_geometry_check.py` (see git history) on the single-type
synthetic scan (Mac GPU): true fibers damaged with random kinks come back
to within 0.25 voxels of the truth with no overlaps and the bend limit met.
Before the fitter ran only on the solver, the same check took the NumPy
fitting loop's fits from 46 overlapping pairs and 3 fibers over the bend
limit to 0 in 26 iterations.

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

   This is needed because contact pushes two fits on one fiber
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

## 7b. Fiber types and radii from thickness

`_fit._Fitter.classify`, `_fit._Fitter.trace`, `_fit._Fitter.topology`

`spec` may be a list of types that differ in diameter. With one type the
same steps size the fibers.

- **Thickness:** the distance transform of the foreground (image > 0.5),
  taken as the maximum over each 3×3×3 neighborhood and sampled along each
  fiber's interior nodes; the median is its measured radius m. On an axis
  the distance is about the radius, because the foreground edge sits at the
  half-maximum, which is the fiber surface; the neighborhood maximum keeps a
  centerline between voxel centers from reading an interpolated, lower
  value. (Subtracting half a voxel, as an earlier version did, made radii
  5–10% small.)
- **Margin:** a generous mask makes every fiber look thicker by some δ.
  With `FitSettings.thickness_margin_voxels` unset, δ is estimated: start
  at 0, give each fiber the type nearest m − δ, set δ to the median of
  m − r_type, repeat 3 times; δ is clamped to [−0.5, r_min]. It is logged
  per round as `thickness_margin`.
- **Type:** with profiles and several types, `_grey.profile_types`
  samples the grey across the fiber at up to 24 interior nodes, in four
  directions at 0 … 1.4 of each type's radius, and compares it with that
  type's drawn profile; the type with the lowest median squared misfit is
  taken when it is below 0.7 × the next type's. With grey ranges only, the
  type bits (step 1)
  are sampled at the fiber's interior nodes; a type whose bit is set at
  ≥ 60% of them, with no other type within 0.1 of it, is the fiber's type
  (`_Fitter._grey_types`). Otherwise the spec whose radius is nearest
  m − δ in log scale (by ratio). The type sets the fiber's radius prior, bend limit, minimum and maximum
  length and length prior.
- **Radius:** (m − δ + r_type) / 2, clamped to `diameter × (1 ±
  diameter_tolerance) / 2` of the type.
- Types are chosen after every solver batch and for new traces. Topology
  moves (6b) run per type, with that type's parameters: splits and joins
  never mix types.
- **Tracing** runs type by type, largest first. A larger type is seeded only
  where the foreground is at least 0.7 of its radius deep (the smallest type
  uses 0.5), so it starts only where the foreground is thicker than smaller
  fibers could make it. Each type traces with its own Hessian scale, and
  traces of earlier types claim their voxels.

## 8. Where Tangle's own code comes in

- **Ownership rule:** the fitter uses the same nearest-capsule-surface rule
  as Tangle's voxel exporter, so a fit re-voxelized by Tangle gives the same
  labels.
- **Outputs:** `FitResult.to_collection` / `to_assembly` build a Tangle
  `FiberCollection` / `Assembly`. The cell is the scanned box, not periodic.
  There is one `Material` per fitted diameter, rounded to 10 nm, each with
  the spec's bend limit.
- **The continuous fit** is Tangle's relaxation with the scan as an extra
  force (6a).
- **Relaxation after the fit:** the optional `FitResult.relax()` runs Tangle's
  plain contact relaxation (no image force) on the fitted assembly.
- **Synthetic scans** (`_synthetic.synthetic_ct`): the ground truth is a
  Tangle structure.
  1. `export_puma(include_interface=True, include_fiber_ids=True)` gives
     partial-volume occupancy and per-voxel fiber IDs.
  2. The scan is attenuation = 0.05 + 0.95 × (occupancy blurred with a
     σ = 0.9 voxel Gaussian), plus Gaussian noise (σ = 0.12) and a weak
     cosine drift, scaled to uint16.
  3. `SyntheticScan.crop` cuts a window from a larger render, so fibers cross
     its boundary as in a real scan.

## 9. Outputs

`FitResult.write`

- `fit.json` (schema `tangle.ct.fit/1`) contains:
  - the spec and the levels;
  - each fiber's centerline in meters, its diameter and its support (the mean
    normalized intensity along it);
  - each fiber's per-node confidence (see below);
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

### 9a. Confidence (`_confidence.node_confidence`)

After the final solve, each fiber is resampled every `spacing` (the node
spacing, one smallest radius) and read on a ring of 8 directions normal to
its tangent. With `r` the fiber's radius and `m` the thickness margin:

| Component | Read at | Score |
| --- | --- | --- |
| image | the axis and the ring at 0.5 r | mean of clip((I − 0.5)/0.4, 0, 1) |
| ownership | the same 9 points | 1 − fraction inside another fit's capsule |
| surround | the ring at r + m + 1 | 1 − fraction that is foreground (I > 0.5) and not within r + m + 0.5 of another fit |
| thickness | the axis | exp(−½((depth − m)/r − 1)² / 0.3²) |
| stability | the axis | exp(−½(d / 0.5 r)²), d = distance to the fiber before the final solve |

`depth` is the same local thickness the classifier uses (§7b). The product
of the five is median-filtered over three samples, and each stored node
takes the lowest sample within half a segment of it. Other fits are found
with a KD-tree over segment midpoints and exact point-to-segment
distances. The history gets a `confidence` entry with the mean, the number
of fibers whose lowest node is below 0.5, and the mean of each component.

### 9b. Redraw passes (`_regrow`, `_Fitter.redraw_loop`)

After the final solve and its confidence, up to `redraw_passes` passes:

1. `cut_unsure`: each fiber is resampled every `spacing` with its
   confidence interpolated along it. The first pass cuts on the full
   confidence, later passes on the confidence without stability. Nodes
   below `confidence_threshold` are unsure; inside a failed region's box,
   an unsure node also removes the nodes within `count · 2 r_max` of arc;
   inside a given-up box nothing is removed. Runs of the rest at least
   2 r long are kept as pieces; a piece end next to a removed stretch is a
   cut end. The anchors are the pieces less 2 r at each cut end, so the
   join can bend. Removed stretches within 2 r_max of each other form a
   region, a box around them padded by 2 r_max.
2. `grow_cut_ends`: the move depends on how often the region at the cut
   end failed before (`_Fitter._attempt`): 0, grow longest pieces first;
   1, do not grow (step 3 re-traces the region from fresh seeds); 2, grow
   shortest pieces first. Each growing cut end is extended with
   `Tracer.trace_one_way` from the tip along the end tangent (its own
   capsule is not avoided; the others are down-weighted as in tracing),
   for at most 20 r_max. Where the extension enters another piece, it is
   kept if it crosses at more than 30° and cut at the entry if it runs
   along it or ends inside it. Every piece is painted into a label volume
   as it grows, so a later end cannot grow over an earlier one.
   With `redraw_moves = "match"` (the default) step 2 is instead
   `_Fitter._match` (`_junctions`): every cut end is a port of the nearest
   region, with its outward tangent. Two ports may be paired when they are
   the same type, on different pieces, face each other, lie within 20 r_max,
   and a cubic Hermite bridge between them (tangent magnitude = their
   distance) has curvature ratio ≤ 1.2 against the bend limit. An unpaired
   port's fiber ends with its `trace_one_way` extension (cut at entry into
   another piece, as above). Every matching (up to 1024 per region) is
   scored in nats over the region box: squared residual of the scan against
   `render_occupancy` of the nearby pieces plus the chosen bridges and
   extensions (radius + margin), over `evidence_scale`; plus overlap voxels
   beyond one fiber over π r²; plus, per interior end,
   max(−ln(h(ℓ) D), 1), h the hazard of a gamma length distribution with
   mean L and shape `FitSettings.length_shape` (3) and ℓ the piece's
   length plus its extension (a lower bound when the piece runs on through
   another region or the scan boundary); plus, per join, the joined
   fiber's cumulative hazard −ln S(ℓ_a + ℓ_b + bridge) less the two
   pieces' (`_ends.length_end_cost`, `length_join_cost`). With profiles
   the residual is instead the denoised scan against the fibers drawn with
   their profiles (`render_grey`, the brighter fiber where two meet), over
   the grey evidence scale 2 σ² π r² (σ² the mean squared grey residual
   near the fibers). Every matching first gets a quick score, the sum of each element's
   own residual change and overlap with the fixed fibers (measured once
   per element); only the best 24 are drawn whole and scored exactly.
   A pass's region ranking is computed once and shared by its candidates. Shape 1 gives
   the old constant ln(L/D) and free joins. A piece's own (uncut) end
   inside a region box and away from the scan boundary is a port as well;
   unjoined it gets no extension and stays where it is. The
   plans are tried `redraw_plans` (3) at a time: `_Fitter.pick_plans`
   builds candidate c from every region's c-th best plan (after the
   attempt × 3 plans spent on earlier failures there), runs steps 3–4 on
   each, and gives every region the plan whose candidate scores best in
   the region box by `redraw_score` (fewest `residual_map` voxels, or most
   `coverage_map`); if regions disagree, that mix is built and solved once
   more. Only close calls get the extra solves: candidate c changes just
   the regions whose c-th plan scored within `redraw_plan_margin` (5) nats
   of their best, and is skipped when there are none. Each pass logs its
   `seconds`. `_junctions.assemble`
   chains the pieces through the bridges (a link that would close a loop
   is dropped). The history logs `plans_solved` and `plan_choices`.
3. `_Fitter.trace` seeds new fibers in the foreground still unclaimed;
   then the per-type topology step (§6b) joins the grown ends.
4. `solver_batches` batches and a final solve, with device nodes within
   0.3 r_min of an anchor pinned during the image run
   (`ImageRelaxer.set_pinned`; a build without it ignores the anchors and
   says so in the log). The image-off settle after it runs unpinned: two
   pinned stretches in contact could not otherwise be pushed apart, which
   tripled the overlapping pairs.
5. Keep or revert, per group: regions are grouped when one fiber the
   redraw changed spans both (`region_components`): an old fiber through
   the regions its cut stretches were in (`Cut.fiber_regions`), a new one
   through the regions of its nodes that are not on a pinned sure piece
   (`changed_regions`; a changed node outside every region joins the
   nearest, whose box grows to hold it). Tying regions by any fiber that
   merely passes through them chained every region of a dense scan into
   one group. With `redraw_score = "grey"` (the default with profiles:
   `"auto"`) a group is kept if its boxes' squared grey residual, scan
   against the fit drawn with its profiles, falls by more than one nat
   (the grey evidence scale). With `"mask"` (the default otherwise) a
   group is kept if
   its boxes' sum of `residual_map` (foreground farther than r + m from
   every fit, plus background inside a capsule) falls by more than 0.001
   per foreground voxel; with `"confidence"`, if the sum of `coverage_map`
   (per foreground voxel, the owning segment's confidence without
   stability) rises by that much; `"all"` keeps every group, for
   comparison. The merged fit takes the redraw's fibers except those
   in reverted groups, and the old fibers of reverted groups, plus any old
   fiber outside every region that no kept fiber follows any more (the
   redraw may have joined it into a reverted one). When old and new fibers
   are mixed, the merged fit gets one settle (every node pinned for the
   image run, so only the unpinned settle acts). The per-group decisions
   stand: a whole-pass check (first a total gain, then no loss above
   0.005) threw away passes with 10–17 accepted groups in two_types. The
   history logs the merged fit's `sure_coverage`, `sure_coverage_before`
   and `residual_change` (per foreground voxel). In the study the mask
   score correlated +0.81 with the truth's gain per group, confidence
   +0.67 (a since-removed `--redraw-study` in `ct_examples.py`, see git
   history).
6. After the passes, if any redraw was kept, the whole fit gets one
   unpinned solve (`FitSettings.redraw_polish`, logged as `polish`): the
   redrawn stretches were solved around pinned pieces, and on ca4cb83 they
   came out ~1 voxel off (single_type line error 0.23 → 0.93).
7. Failed regions are remembered as boxes with a failure count (a kept
   region clears overlapping ones); at `redraw_attempts` failures the box
   is given up. The passes end when nothing is left to cut.

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
