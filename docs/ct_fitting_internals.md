# How `tangle.ct` fits fibers, step by step

This is the exact procedure in the code. Every step names its file and function
under `crates/tangle_python/python/tangle/ct/`. [ct_fitting.md](ct_fitting.md)
is the user guide, and the [results page](../crates/tangle_python/python/examples/ct_results/README.md)
shows how well it works on synthetic scans, with pictures.

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
   leave a tail in empty space that no later step notices. Every
   centerline node inside the scan whose fiber image reads below 0.3 is
   void. Void nodes at an end are trimmed back to the
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

**Which births are kept** (`_Fitter.trace`). With `birth_ridge_min` (0.5 in
the packed-bundle settings, §9c), a new trace among existing fits is kept
only if at least that share of its nodes sit on a grey ridge: the brightest
point of the grey (the scan smoothed by `birth_ridge_sigma_voxels`, 1.6) on
a disc of radius r across the trace lies within r/2 of the node. On a
saturated range image a trace can run straight across a bundle, between its
fibers, and is then off the ridge most of its length. The same step serves
every birth after the first trace: a round's new fibers, a redraw's new
fibers and the final births (§9c).

A dim-cored coarse fiber is brighter on its rims than on its axis, so a
coarse trace on it fails this test along its whole length (along the true
axes of the benchmark's 109 coarse fibers the ridge share was at most 0.02,
so none passed 0.5). A coarse fiber lost in a round or a redraw was never
traced again; only its rims were, by fine fits. Two settings let coarse
fibers be born again:

- `birth_ridge_smallest_only` applies `birth_ridge_min` to the smallest type
  only;
- `coarse_birth_disc_max` (off by default; 0.8 in the packed-bundle settings)
  gates the larger types' births instead: a new trace of a larger type is
  dropped when its cross-section holds fine-bright grey, the disc test of
  §7b (`_holds_fine`) at this level. It runs after
  `grey_checked_traces` and `coarse_trace_axis_check`, never on the first
  trace, and needs `coarse_axis_grey_max`, whose denoised grey it reads.

On the benchmark (redraws off) the pair raised centerline F1 from 0.9065 to
0.9145 (7-fiber bundles +0.004, 19-fiber bundles +0.012; 10 structures up,
6 down) and cut the coarse fibers left with under 0.2 of their length
covered from 19 to 6. The cost is some coarse births over bundles:
coarse-fit centerline nearest a fine fiber rose from 144 to 394 voxels,
almost all of it on the 19-fiber bundles, and merged fibers from 86 to 93.

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

- **Thickness:** read two ways. The **depth**: the distance transform of
  the foreground (image > 0.5), taken as the maximum over each 3×3×3
  neighborhood and sampled along each fiber's interior nodes; the median
  is m. On an axis the distance is about the radius, because the
  foreground edge sits at the half-maximum, which is the fiber surface;
  the neighborhood maximum keeps a centerline between voxel centers from
  reading an interpolated, lower value. The **cross-section**
  (`_image.half_widths`): at up to 24 interior nodes the image is sampled
  along four directions normal to the fiber, every 0.25 voxel out to 1.6 ×
  the largest type radius; the median over nodes and directions at each
  distance is the fiber's cross-section, and w is where it falls below
  half its peak, walking outward from the peak (looked for within half
  that reach, so a bright rim around a dim core reads its outer edge).
  Fibers are typed and sized by m, except in a plain grey scan (no grey
  ranges, profiles or mask), where they are typed and sized by w. There
  the foreground is a threshold of the scan itself, and in a scan whose
  fibers are a few noise sigma above void (for example noise that is
  correlated over a voxel or two; the `noisy_two_types` example) it punches
  holes into dim fibers: m reads a fraction of their radius, every dim
  coarse fiber was typed fine and packed with fine fits (fitted plain,
  centerline F1 0.62 on `noisy_two_types` and 0.60 on `two_types`; with w,
  0.81 and 0.89). The median pools about a
  hundred samples per distance and survives the noise, but w reads a fiber
  packed among others too thick (its neighbours fill part of the
  cross-section; see §9a), so the hole-filled fiber
  image that grey ranges, profiles or a mask give is typed by m. A plain
  grey scan's core-sized foreground holes are filled too (as a mask's and
  the ranges'), with `fill_mask_holes`.
- **Margins:** a generous mask or range makes every fiber look thicker by
  some δ. With `FitSettings.thickness_margin_voxels` unset, one is
  estimated for each reading. For the reading that types: start at 0, give
  each fiber the type nearest m − δ, set δ to the median of m − r_type,
  repeat 3 times; for the other, the median excess over the resulting
  types. Both are clamped to [−0.5, r_min]. The depth's margin sets what
  reads the foreground (the solver's reach at fiber ends, the redraw's
  residual maps) and is logged per round as `thickness_margin`; the
  cross-section's is used by the confidence (§9a).
- **Type:** with profiles and several types, `_grey.profile_types`
  samples the grey across the fiber at up to 24 interior nodes, in four
  directions at 0 … 1.4 of each type's radius, and compares it with that
  type's drawn profile; the type with the lowest median squared misfit is
  taken when it is below 0.7 × the next type's. With grey ranges only, the
  type bits (step 1)
  are sampled at the fiber's interior nodes; a type whose bit is set at
  ≥ 60% of them, with no other type within 0.1 of it, is the fiber's type
  (`_Fitter._grey_types`). Otherwise the spec whose radius is nearest
  the thickness less its margin in log scale (by ratio). The type sets the fiber's radius prior, bend limit, minimum and maximum
  length and length prior.
- **Radius:** (thickness − δ + r_type) / 2, clamped to `diameter × (1 ±
  diameter_tolerance) / 2` of the type.
- **Bundle typed coarse:** in a packed bundle the foreground is one blob,
  so a fit between the bundle's fibers reads the bundle's depth and is
  typed coarse. `coarse_axis_grey_max` (0.15 in the packed-bundle settings)
  types a fit fine when its median axis grey lies more than that fraction
  up the smallest type's range, but a coarse fit laid over a bundle often
  has its axis on the dark gaps and escapes it. With grey ranges,
  `coarse_disc_max` (off by default; 0.8 in the packed-bundle settings) and
  `coarse_disc_classify`, every classify also reads such a fit's disc
  (`_Fitter._holds_fine`): at each interior node, the brightest grey of the
  denoised scan on a disc of the larger type's short radius across the fit,
  sampled every 0.75 voxel. If
  the median of these over the nodes lies above `coarse_disc_max` of the
  way up the smallest type's range, the fit holds fine-bright fibers and is
  typed as the smallest type. Only fits whose depth could make them that
  larger type at any margin are read. A dim-cored coarse fiber's disc holds
  only its own rim, which reads dimmer, and it stays coarse.

  `coarse_disc_trace` and `coarse_disc_end` (both on by default) use the same
  test to drop new coarse traces and to remove coarse fits before the final
  births (§9c). The packed-bundle settings turn both off and keep only the
  typing: the trace-time drop also removed a bright true coarse fiber (its
  disc read 0.86). On the benchmark the typing alone raised F1 from 0.890 to
  0.903 (19-fiber bundles 0.851 → 0.881, 7-fiber bundles 0.930 → 0.925).
  A variant that limited the test to packed regions and cut fine-bright
  stretches out of coarse traces and fits node by node (in every cleanup
  too), instead of judging whole fits, lost 0.022 on eight structures
  against the starting settings and was removed.
- **Dim-cored coarse fiber typed fine:** the opposite error. A dim core
  leaves holes in the foreground, so the depth of a coarse fit reads low
  and it is typed fine. With grey ranges and `coarse_axis_grey_min` (0.0 in
  the packed-bundle settings), a fit the depth types as the smallest type
  takes the type nearest its cross-section w (less its margin) when all of
  these hold:
  - its median axis grey (denoised scan, interior nodes) lies below
    `coarse_axis_grey_min` of the way up the smallest type's range (0: below
    the range's low end, where no fine fiber's axis reads);
  - the 90th percentile of the grey over discs of the largest type's short
    radius across it lies below `coarse_axis_disc_max` (0.3) of that range.
    A dark gap between a bundle's fibers has their bright axes within that
    disc; a dim core has only its rim;
  - w is nearer a larger type than the smallest;
  - its depth is at least `coarse_axis_depth_min` (4) voxels, it has at least
    10 nodes, and `grey_checked_traces` did not flag it.

  In full fits of four structures with the rule on, every one of its 24
  retypings (summed over all classify calls) fell inside a true coarse
  fiber. It shipped together with the ridge finish's rim drop (§9d): the two
  raised F1 from 0.890 to 0.894 (7-fiber bundles 0.930 → 0.935). Both
  typings with the rim drop score 0.9015 (7-fiber bundles 0.930, 19-fiber
  0.873): 0.0015 below the disc typing alone over all 18, but without its
  loss on the 7-fiber bundles. The packed-bundle settings keep both. Both
  typings read the denoised grey that `coarse_axis_grey_max` builds, and do
  nothing without it.

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
its tangent. With `r` the fiber's radius, `m` the foreground margin and
`δ` the width margin (§7b):

| Component | Read at | Score |
| --- | --- | --- |
| image | the axis and the ring at 0.5 r | mean of clip((I − 0.5)/0.4, 0, 1) |
| ownership | the same 9 points | 1 − fraction inside another fit's capsule |
| surround | the ring at r + m + 1 | 1 − fraction that is foreground (I > 0.5) and not within r + m + 0.5 of another fit |
| thickness | the axis, and the ring out to 1.6 r | exp(−½ e² / 0.3²), e = min(\|(depth − m)/r − 1\|, \|(w − δ)/r − 1\|) |
| stability | the axis | exp(−½(d / 0.5 r)²), d = distance to the fiber before the final solve |

`depth` is the foreground's local thickness (§7b) at the axis, and `w`
the radius of the local cross-section, as the classifier's thickness
(§7b) but per sample: the ring is read every half voxel out to 1.6 r, the
median at each distance is taken over the ring and two samples to either
side (40 values), and w is where it falls below half its peak
(`_image.half_radius`, the peak looked for within 0.8 r). The reading
closer to the radius counts. Depth is right where the foreground is clean
and wrong where a noisy scan's threshold punches holes into dim fibers;
the pooled cross-section survives noise but reads a fiber packed among
others too thick. On the true fibers of the examples, depth alone
misjudged (|e| > 0.3) 95% of the coarse nodes in `noisy_two_types` fitted
without grey information and the cross-section alone 24% of the nodes
in `scenario_dense_crossing`; the closer of the two at most 7%. The product
of the five is median-filtered over three samples, and each stored node
takes the lowest sample within half a segment of it. Other fits are found
with a KD-tree over segment midpoints and exact point-to-segment
distances. The history gets a `confidence` entry with the mean, the number
of fibers whose lowest node is below 0.5, and the mean of each component.

### 9b. Redraw passes (`_regrow`, `_Fitter.redraw_loop`)

The redraws are on by default (`redraw_passes = 5`). The packed-bundle
settings (§9c) turn them off (`redraw_passes = 0`): there the stages after
them (§9c–9g) repair the fit on the grey, and the regions the redraws kept
cost more than they gave. On the benchmark, with every change up to the
coarse typing of §7b on, F1 was 0.9015 with 5 passes, 0.9039 with 3 and
0.9065 with none (19-fiber bundles 0.873, 0.877, 0.883; 7-fiber bundles
0.930, 0.931, 0.930). The 5 passes kept 46 of 163 region groups over 44
passes, 8 of the 18 structures kept none, and the passes logged 1314 s in
all. A kept group can do lasting harm: in one run of `dense_hard_17` with the
coarse births of §6c on, a kept redraw left two coarse fits on one coarse
fiber; after the polish and the ridge finish they sat 5.5 and 7.4 voxels off
its axis, and the fiber was covered along 0.14 of its length, against 0.75
in the run without those births.

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

### 9c. After the redraws: the finish

`_fit.fit_fibers`

After the redraws (and their polish, when a redraw was kept) come the last
stages. None of them solves again. They run in this order, each only when
its switch is on:

| Stage | Switch | Default | Packed bundles | Section |
| --- | --- | --- | --- | --- |
| Recenter | `recenter_on_grey`, `recenter_last` | off | on | below |
| Coarse rescue | `coarse_rescue` | off | off | 9f |
| Coarse fits on bundles removed | `coarse_disc_max` with `coarse_disc_end` | off (needs `coarse_disc_max`) | off | 7b |
| Coarse join | `coarse_join` | off | off | 9f |
| Final births | `final_births` | off | on | below |
| Ridge finish | `ridge_finish` | off | on | 9d |
| Hole births | `hole_births` | off | on | 9e |
| Straight join | `straight_join` | off | on | 9g |
| Bend finish | `bend_finish` | off | on | 9g |
| Clean finish | `clean_finish` | off | off | 9g |
| Merge short | `merge_short` | off | off | 9g |
| Model pass | `model_recenter`, `residual_births` | off | on | 9h |
| Through block | `through_block` | off | off | 9g |

- **Recenter** (`_Fitter.recenter`): each fit of the smallest type moves
  across itself onto the brightest grey (the scan smoothed by
  `recenter_sigma_voxels`), by at most `recenter_max_radii` (0.5) r. With
  `recenter_last` it runs here, after the redraws, rather than before the
  confidence.
- **Final births**: one more round of new fibers (§6c, with its birth
  gates), recentered, with no solve after it: a solve drags some good births
  off their fibers, and the cleanup then removes them.

Each stage logs a `history` entry under its name ("ridge finish", "hole
births", "straight join", ...; the removal logs "coarse on bundles removed")
with its counts, and all but the removal save a snapshot with
`fit_fibers(snapshots=...)`. The coarse join and the stages from the ridge
finish on are skipped for a mask.

**The packed-bundle settings.** These are the settings the `dense_hard`
benchmark below recommends for scans of packed bundles, with grey ranges:

```python
ct.FitSettings(
    bright_seed_strength=0.05, grey_checked_traces=True,
    graded_image=True, graded_smallest_only=True,
    prior_merge_gap_radii=4, radius_cap_prior=True, births_solve_pinned=True,
    coarse_trace_axis_check=True, coarse_axis_grey_max=0.15,
    recenter_on_grey=True, recenter_each_solve=True, recenter_last=True,
    recenter_sigma_voxels=1.6,
    birth_ridge_min=0.5, birth_ridge_smallest_only=True,           # 6c
    coarse_birth_disc_max=0.8,                                     # 6c
    coarse_disc_max=0.8, coarse_disc_classify=True,                # 7b
    coarse_disc_trace=False, coarse_disc_end=False,                # 7b
    coarse_axis_grey_min=0.0,                                      # 7b
    redraw_passes=0,                                               # 9b
    final_births=True,
    ridge_finish=True, ridge_sigma_voxels=1.2,                     # 9d
    ridge_min_length_scale=0, ridge_refine=True, ridge_rim_share=0.7,
    ridge_rim_sigma=1.2, hole_births=True, hole_birth_ridge_min=0.3,  # 9e
    hole_birth_grey_min=0.65, hole_birth_passes=2, hole_birth_rim_drop=True,
    straight_join=True, bend_finish=True, bend_finish_grey_min=0.4,  # 9g
    model_recenter=True, residual_births=True,                     # 9h
)
```

**The benchmark.** Unless a number says otherwise, it is the mean
centerline F1 (§10) over `dense_hard_1` … `dense_hard_18` in
`examples/ct_examples.py`: packed bundles of fine fibers (6.5 µm, r = 3.25
voxels), 7 per bundle in 9 structures and 19 in the other 9, with flat oval
coarse fibers (17.3 µm wide, thickness 0.8 of the width) that have a bright
rim and a dim core, every fiber with its own brightness, scanned by the
simulated scanner at 1 µm voxels. The fits read one broad grey range for
every type (`--input broad`), so the smallest type's range that §6c, §7b
and §9f read is that shared range. Some numbers come from **exact replays**: the
stages from the ridge finish on rerun on the CPU from each structure's saved
state, which reproduces the GPU run's final fits exactly.

How the packed-bundle settings were reached, each row adding to the one
above:

| Change | All 18 | 7-fiber bundles | 19-fiber bundles |
| --- | --- | --- | --- |
| Starting settings (the block above without the settings the rows below add, and with `redraw_passes=5`) | 0.840 | 0.893 | 0.787 |
| Ridge finish: sharper grey, short pieces kept, short recenter (9d) | 0.871 | 0.916 | 0.827 |
| Hole births (9e) | 0.890 | 0.930 | 0.851 |
| Coarse typing (7b) and the ridge finish's rim drop (9d) | 0.9015 | 0.930 | 0.873 |
| Redraws off (9b) | 0.9065 | 0.930 | 0.883 |
| Coarse births (6c), rim drop on hole births (9e), straight join and bend finish (9g) | 0.915 | 0.935 | 0.895 |
| Model recenter and residual births (9h) | 0.942 | 0.956 | 0.929 |

On 18 structures no setting was tuned on (`dense_hard_19` … `dense_hard_36`,
with `DENSE_HARD_COUNT=36`; 8 with 7-fiber bundles, 10 with 19), the
starting settings score 0.836 and the packed-bundle settings 0.934, every
structure up (7-fiber bundles 0.860 → 0.942, 19-fiber bundles 0.816 →
0.929); without the model pass (9h) they score 0.911.

Several of these stages use fixed lengths in voxels (the 1.2-voxel grey and
Hessian of §9d and §9e, the 3-voxel cut runs, the join and walk distances of
§9g), chosen at a fine radius of 3.25 voxels. Hole births trust the foreground,
so any non-fiber foreground becomes holes to trace.


### 9d. Ridge finish (`_ridge.ridge_finish`)

Inside a packed bundle the range image is saturated, so nothing in the fit
image keeps a fine fit on its own fiber: where a neighbour comes close the
fit slides across and carries on along the neighbour, and fits sit beside
their axes. The grey, smoothed a little, still peaks on every axis. The
ridge finish is a last pass over the smallest type's fits on that grey;
larger types pass through unchanged.

A point is **on the ridge** when the brightest point of the grey on a disc
of radius r across the fit (the disc values averaged over 7 samples along
it) lies within r/2 of it. Each fit is sampled every half voxel (a fit of
fewer than 5 samples passes through untouched), and:

1. **Recenter** onto the ridge: each sample moves to the brightest point of
   a disc of radius r/2 (averaged over 7 samples), the shifts averaged over
   11 samples. This reads the recenter grey (`recenter_sigma_voxels`, 1.6),
   whose basin must reach fits half a radius off.
2. **Cut** every run of samples off the ridge at least 3 voxels long (a hop's
   crossing, a stretch between fibers, an overhanging end): an interior run
   splits the fit, an end run trims it. Samples inside a larger type's fit
   (its oval section at its fitted radius) are cut too: a coarse fiber's
   bright rim is a ridge a fine fit can follow. Pieces of at least 4
   samples are kept.
3. **Rim drop** (`ridge_rim_share`, below).
4. **Grow** each piece of at least 12 samples at both ends, one voxel at a
   time, to the brightest point of a disc of radius `ridge_extend_reach`
   (0.5) × r ahead of it, while the grey there is at least `ridge_finish_extend`
   (0.85) of the piece's own median grey, no other fit is within 1.02 r
   (about half the spacing of touching fibers), and the step stays in the
   scan and outside every larger type's fit; at most 80 steps.
5. **Join** piece ends, nearest first, when they are within 6 voxels and
   either both end directions (over 6 voxels) point along the gap within
   30°, or the ends are antiparallel within 30° and at most 0.6 r apart
   across the fiber, or they are within 4 voxels and run on past each other
   (the second piece's overlapping start is dropped). At least 80% of the
   straight bridge must be on the ridge (read without averaging along it).
6. **Refine** (`ridge_refine`, below).
7. **Drop** pieces shorter than `ridge_min_length_scale` × the type's minimum
   length.

The pieces go back to the node spacing. The log gives `cut_voxels`,
`grown_voxels`, `joins`, `short_dropped` and `rim_dropped`.

**Sharper grey** (`ridge_sigma_voxels`, 1.2 in the packed-bundle settings).
The cut (2), the growth (4) and the join's ridge test (5) read the scan
smoothed by this sigma; the recenter (1) and the refine (6) keep the
recenter grey. At 1.6 voxels a dim fiber packed beside a brighter one has no
ridge of its own: the neighbour's blur tilts the grey across it, the disc's
brightest point sits on the neighbour's side, and the cut removes a correct
fit along its whole length. About 1.2 keeps each fiber's own ridge. In
exact replays on seven structures with the starting settings, the sharper
grey alone raised F1 by 0.018 on average.

**Short pieces** (`ridge_min_length_scale`, 1 by default, 0 in the
packed-bundle settings). With 0 the finish keeps every piece of at least 4
samples (2 voxels) that the cut and the joins leave. The remnants are
mostly on their own fiber: in the same replays keeping them added 0.006 on
average, with 35 pieces per structure shorter than the minimum length
(against 1), at a centerline precision of 0.71. They fragment fibers, which
the straight join (§9g) partly repairs. At 0.25 the benchmark lost 0.002
against 0.

**Short recenter** (`ridge_refine`). After the join, every piece is
recentered once more on the recenter grey with a short reach: the
brightest point of a disc of radius 0.3 r, the disc values averaged over 21
samples (10 voxels) along the fit, the shifts over 11 samples. This settles
grown ends and bridges without letting a fit slide onto a neighbour. In the
replays it added 0.002 on average on top of the other two.

Together the three raised the benchmark from 0.840 to 0.871, every
structure up (by 0.009 to 0.059).

**Rim drop** (`ridge_rim_share`, 0.7 in the packed-bundle settings;
`ridge_rim_reach` 1.35, `ridge_rim_grey` 0.58). The solver keeps a coarse fit
one section away from a fine fit, so a fine fit on a dim-cored coarse
fiber's rim pushes the coarse fit off its axis, and the rim then lies just
outside the coarse fit's section, where step 2 does not cut it. After the
cut, a piece is dropped when at least `ridge_rim_share` of its samples lie
within `ridge_rim_reach` of a larger fit's oval section (the section at the
type's radius, 1 being its edge) and the median grey of those samples is at
most `ridge_rim_grey` of the way from the void grey to the fine pieces'
median grey: a coarse rim reads dimmer than a fine fiber. In an exact replay
on all 18 structures, with the hole births on, it raised F1 by 0.0019, 17
structures up and none down, through precision (+0.0045).

Other ridge-finish settings, off in the packed-bundle settings:

| Setting | Default | What it does | On the benchmark |
| --- | --- | --- | --- |
| `ridge_extend_reach` | 0.5 | the radius (in r) of the disc the growth steps to | 0.25 raised F1 by 0.0032 in an exact replay of the packed-bundle settings without the final passes (14 of 18 up) and cut hop length by 39%; on two earlier bases only 9 and 12 of 18 went up, so it is not adopted yet |

Tried and removed: a second growth-and-join pass after step 7 (+0.002 in an
exact replay with hole births, 12 of 18 structures up; the straight join of
§9g does the same job), cutting an off-ridge run only next to another fit,
and moving a coarse fit toward a dropped rim piece (+0.0005 offline).


### 9e. Hole births (`_Fitter.hole_births`, `_ridge.finish_holes`)

A dim fiber in a packed bundle is often never traced. On the graded image
its bright neighbours dim its axis below the tracer's 0.5, so it is not
seeded and a trace from it stops at once. Once its neighbours are fitted
and the ridge finish has put them on their axes, it is a fiber-wide stretch
of foreground that no fit claims. Hole births trace the smallest type there.

They run after the ridge finish and need `final_births` and
`ridge_rim_sigma`; without either they
are skipped. Their ridge test reads the birth grey, which only
`birth_ridge_min` builds: without it every hole trace passes. Each pass:

1. **Claim.** Every fit claims the foreground within `hole_birth_claim_radii`
   (1.2) × its radius.
2. **Trace.** Seeds lie on the distance-transform ridge of the unclaimed
   foreground, at least `hole_birth_depth_radii` (0.6) r deep. Traces run on
   the flat range image with the smallest type's radius, bend limit and
   minimum length; the claim down-weights the recentering, so a trace keeps
   to its hole.
3. **Ridge test.** A trace is kept when at least `hole_birth_ridge_min` (0.5;
   0.3 in the packed-bundle settings) of its nodes are on the ridge of the
   birth grey (§6c), with the rim exemption (below).
4. **Finish** (`finish_holes`), every half voxel along each trace:
   - cut runs off the ridge of at least 3 voxels, with the rim exemption;
   - cut samples inside a larger type's fit;
   - with `hole_birth_grey_min`, cut runs of at least 3 voxels whose grey is
     below the floor (below);
   - drop pieces shorter than the type's minimum length (not scaled by
     `ridge_min_length_scale`);
   - with `hole_birth_rim_drop`, drop pieces on a larger type's rim (below).

   The cut and the floor read the recenter grey (giving them the sharper
   ridge-finish grey instead cut correct hole length and lost 0.0014 in an
   exact replay).

The next pass traces around the fits the last one added, up to
`hole_birth_passes`; the passes stop early when one adds nothing. The log
gives `born`.

**Rim exemption** (`ridge_rim_sigma`, 1.2 in the packed-bundle settings). A
dim fiber packed beside a brighter one fails the ridge test although it is
traced right: the smoothed grey rises all the way across its axis onto the
bright neighbour's flank, so the disc's brightest point sits on the disc's
edge, toward the neighbour. With the exemption, a point whose brightest
disc point lies within half a voxel of the disc's edge still counts as on
the ridge where the grey curves down across the fit both ways: both
eigenvalues of the scan's Hessian (at scale `ridge_rim_sigma`) in the plane
across the fit are negative, so the point sits on a tube of its own
(`_Fitter.curves_down`). A trace between two fibers sits in a dip across
their contact and still fails; a fit half a radius or more off its own axis
has the brighter axis inside the disc, not on its edge, and still fails.
The exemption acts in the hole births' ridge test and cut only. Extending
it to the other births (rounds, redraws and the final births) lost 0.107 on
one structure and raised hop length from 0 to 94 voxels.

**Grey floor** (`hole_birth_grey_min`, 0.65 in the packed-bundle settings). A
hole trace can run out of the bundle into the dim halo between fibers or
along the void. The floor is `hole_birth_grey_min` × the median grey along
the smallest type's fits. With `hole_birth_grey_void` it is instead that
share of the way from the void grey to that median, for a scan whose void
grey is far above zero, where a plain share of the median can fall below
the void and cut nothing. The packed-bundle settings use the plain share.

**Rim drop** (`hole_birth_rim_drop`, on in the packed-bundle settings). The
cut above removes only samples inside a larger fit's own section, and the
floor keeps a coarse rim, which is bright. So beside a coarse fit pushed off
its axis the rim lobe, just outside the section, is traced as a hole. The
rim drop applies the ridge finish's rim test (§9d: `ridge_rim_reach`,
`ridge_rim_grey`) with `hole_birth_rim_share` (0.7) as its share, so it
works with `ridge_rim_share` off. In an exact replay of the packed-bundle
settings without the final passes it raised F1 by 0.0013 (11 structures up,
none down), with recall unchanged on every structure and the hole length on
coarse fibers down from 1089 to 470 voxels.

**What hole births give.** In an offline check on the final fits of eight
structures after the sharper ridge finish, hole births with the default
values (ridge share 0.5, no floor, one pass) raised F1 by 0.011 and the
packed-bundle values (0.3, 0.65, two passes) by 0.016. Fine fibers missed
along their whole length fell from 19 (no hole births) to 4 (defaults) and
to 2 (packed-bundle values). Hole length farther than r from every true
axis was 286 voxels with the defaults and 26 with the packed-bundle values,
and with the latter 90% of the hole length lay on its own fiber's axis. In
the full fit they raised the benchmark from 0.871 to 0.890, every structure
up (by 0.003 to 0.042).

Joining each pass's hole fits to the fits whose ends they continue, with
the ridge finish's join gates, changed F1 by −0.00001 with the coarse
births of §6c on and made one wrong join; the straight join (§9g) covers
the same case.


### 9f. Coarse passes

**Coarse join** (`_join.coarse_join`, `coarse_join`, off). A coarse fiber
often ends the fit in pieces the cleanup no longer joins: two pieces facing
each other across a short gap, or two pieces running past each other side
by side, each on one rim of the same dim-cored fiber, with their axes about
two short radii apart, which the duplicate tests read as two touching
fibers. Before the final births, pairs of a larger type's piece ends are
joined best first, r being that type's short radius:

- **across a gap**: the ends antiparallel within 35°, at most 4 r apart,
  each pointing along the bridge within 35° (or the two lines at most 0.6 r
  apart across the fiber), and the fit image along the bridge at least 0.5
  on average;
- **side by side**: the ends antiparallel within 35°, running past each
  other by at most 4 r (or stopping at most r short of each other), at most
  2.2 r apart across the fiber, and no bright band between them: across the
  overlap, the grey between the two axes stays below the brighter axis plus
  0.1 of the smallest type's range. Two touching fibers show their rims
  between their axes; two fits on one fiber show its dim core. The overlap
  is replaced by the midline of the two fits.

The log gives `gap_joins` and `side_joins`. On eight structures of the
benchmark with the starting settings it found no pair to join, so it is off.

**Coarse rescue** (`_rescue.coarse_rescue`, `coarse_rescue`, off) traces the
largest type again at the end, ignoring the fit, and puts the dim-cored,
unshared candidates in place of the coarse fits they cover. It is meant for
`trace_smallest_first`, where fine fits claim the coarse fibers' rims.
An add-only form (keeping every solved coarse fit and adding only the
candidates' uncovered stretches) lost 0.016 on `dense_hard_4`, and across
four structures 3 of the 6 pieces it added were stray coarse fits over fine
fibers, so it was not kept.


### 9g. Final passes (`_join`)

These run last, in this order, on the smallest type unless noted. Nothing
after the last solve keeps to the bend limit, so the bend finish is where
it is enforced again. In the packed-bundle settings the straight join and
the bend finish are on; the other three are off.

1. **Straight join** (`straight_join`). Joins pieces that continue each
   other in a straight line: fibers broken by crossings, by dim stretches,
   or into a hole birth beside an existing fit. A pair of ends qualifies
   when their directions (over the last 8 voxels) are antiparallel within
   `straight_join_angle` (15°), each piece's extended axis passes within
   `straight_join_offset` (1) radius of the other's end, and either
   - the ends face each other across a gap of at most `straight_join_gap`
     (30) voxels whose straight bridge reads at least
     `straight_join_support` (0.5) in the fit image on average (None: no
     check), or
   - the pieces run past each other along the same axis by at most
     `straight_join_overlap` (20) voxels, and the second piece's
     overlapping start is dropped.

   Candidates are taken shortest first, each end joins once, and no join
   closes a loop; a chain takes its pieces' median radius.
   `straight_join_types = "all"` joins within every type. The log gives
   `joins`.

   It is for fiber counts and lengths, not F1. In an exact replay of the
   hole-birth row of the §9c table it changed F1 by −0.0003, cut split
   fibers from 546 to 342 and fits per true fiber from 1.725 to 1.493, and
   made 505 joins, at least 22 of them between pieces of different true
   fibers. Looser gates (20°, 1.5 r) joined more (342 → 250 split fibers)
   for 0.0005 less F1; a 60-voxel gap joined almost nothing more (340) and
   lost 0.0003.

2. **Bend finish** (`bend_finish`; `bend_finish_grey_min` 0.6 by default, 0.4
   in the packed-bundle settings). Each fit is resampled at the node
   spacing s. Every node more than the bend limit's sagitta s²/(2R) off the
   midpoint of its neighbours is pulled toward that midpoint by half the
   excess, sweep after sweep (up to 400), R being the type's
   `min_bend_radius`, or `bend_finish_diameters` of its diameters. Then,
   with `bend_finish_grey_min`, runs of at least 3 voxels inside the scan
   where the smoothed fit reads below that share of the way from the void
   grey to the fits' median grey (recenter grey) are cut: a trace that
   wandered along noise has no fiber under its smoothed path. Pieces a cut
   leaves shorter than the type's minimum length are dropped.
   `bend_finish_types = "all"` smooths every type. The log gives
   `moved_voxels_mean`, `cut_voxels` and `pieces_dropped`.

   The gain is in the cut. In an exact replay of the same row the bend
   finish at 0.4 raised F1 by 0.0036 (precision 0.902 → 0.916, recall
   0.880 → 0.874) and cut hop length from 896 to 384 voxels. With the
   straight join, it gave +0.0039 at 0.4 and −0.0011 with no cut (0.0). The
   default 0.6 was not measured. A limit much stiffer than the fibers'
   wrecks the fit: at `bend_finish_diameters=20`, four times the benchmark
   fibers' limit (with the straight join on), two structures fell from
   0.940 and 0.901 (the same passes at the fibers' own limit) to 0.711 and
   0.669.

   The two passes together, measured on the packed-bundle settings (the
   GPU run against an exact replay of the same run without them): F1 0.9159
   → 0.9150, split fibers 570 → 383, recovered fibers 1398 → 1575, fits
   3740 → 3231.

3. **Clean finish** (`clean_finish`, off). Shortest fit first, every sample
   (1 voxel apart) within `clean_finish_reach_radii` (1.25) radii of another
   fit of the smallest type is doubled when the denoised grey at the
   midpoint between the two is at least the dimmer axis's grey (less one
   grey unit): no darker gap lies between them, so they sit on one fiber,
   while two touching fibers show a dip at their contact. Doubled stretches
   are cut out of the shorter fit. With `clean_finish_min_length` (voxels;
   None keeps all), the smallest type's pieces shorter than that are then
   dropped. The log gives `doubled_voxels` and `short_dropped`. Use it where
   two fits often run along one fiber. On two structures of the benchmark
   (exact replays of the hole-birth row of the §9c table, after the straight
   join and the bend finish) the doubled cut
   alone changed F1 by +0.0004 and +0.0015, and dropping pieces under the
   minimum length cost 0.0035 and 0.0045. It was not run on the full set.

4. **Merge short** (`merge_short`, off). Adds each of the smallest type's
   pieces shorter than `merge_short_length` (None: the type's minimum
   length) to the end of a longer fit it continues. A short piece's own
   direction says little, so only the long fit's end direction is used:
   the piece qualifies when, along that direction, its closest point lies
   at most `merge_short_gap` (30) voxels ahead of the end and no more than
   r behind it, and all its samples lie within 1.5 r of the end's extended
   axis. Each piece
   goes to the nearest qualifying end, and an end takes its pieces in order
   of distance. Then the straight join runs with `merge_short_join_gap` (60)
   voxels as its gap, the bend finish without its cut smooths the
   junctions, and every piece of the smallest type still shorter than
   `merge_short_length` is dropped. The log gives `merged`, `joins` and
   `short_dropped`. It suits scans where fiber counts and lengths matter
   more than traced length. On the benchmark short remnants are worth
   keeping (§9d, and the clean finish above), and merge short was not run
   on it.

5. **Through block** (`through_block`, off). Only for scans whose fibers all
   run through the block, none ending inside it. Over three rounds of
   (grey support, dim run, join gap) = (0.3, 10, 120), (0.3, 25, 200) and
   (0.25, 40, 240), lengths in voxels, it joins ends that continue each
   other (the straight join with 12°, 1.5 r, a 20-voxel overlap and no
   bridge check), walks every end still more than 4 voxels inside the
   block toward the faces (`_join.extend_through_ridge`), and joins again.
   - A walk steps one voxel at a time. The brightest point of the sharp
     grey (`ridge_sigma_voxels`, else 1.2) on a disc of radius 0.3 r across
     the step sets the next direction, which turns by at most 1/R radians per
     step (R the bend limit in voxels: `bend_finish_diameters`, else the
     type's).
   - It stops 4 voxels from a face; at a fit running within 10° of it
     closer than 0.45 of the two radii's sum (a fit crossing at a steeper
     angle is passed: fibers cross over and under each other; at a shallower
     pass angle a walk slides along a nearly parallel neighbour and doubles
     it); or after dim-run
     voxels in a row that read below the support share of the way from the
     void grey to the fits' median grey. The end moves to the last voxel
     that read bright enough, so a walk keeps only the dim stretches that
     come back onto fiber.

   Finally, runs of at least 5 voxels reading below 0.3 of the way from
   void to the fits' median grey (a walk or join over empty space) are cut,
   and pieces the cut leaves shorter than the minimum length are dropped.
   The log gives `joins`, `grown_voxels`, `ghost_voxels_cut` and
   `through_share`, the share of fits with both ends within two diameters
   of a face (a fiber leaving the block still ends a little inside it). Most scans have fibers that end inside the block; there, joins up
   to 240 voxels long with no bridge check, and walks to the faces, would
   join fibers end to end and lengthen them. The benchmark's fibers are
   0.55 to 0.9 of the block side long (`dense_hard_settings`), so it was not
   run there.

### 9h. Model pass (`_model`)

`model_recenter` and `residual_births` (both off) run after the other final
passes, before the through block, on the smallest type's fits. The fits
are drawn as the grey they should show, **summed** over fibers: fit `f`
adds `a_f · p_k(d / r_f)` at distance `d` from its axis, with `p_k` its
type's radial profile (`p_k(0) = 1`, knots every 0.125 r out to 2.5 r) and
`a_f` its brightness over the void. The profiles come from a least-squares
fit over up to 200 000 voxels near the fits (a light curvature penalty
fills knots few voxels reach), then each fit's brightness by projection
with the other fits held; the void is the median grey beyond every fit's
reach. The profile takes up the scanner's blur and dark phase-contrast
halo, which on the benchmark is about −0.25 of the fiber contrast at 1.5 r.
The sum is drawn from nearest-fiber rasters (`_geometry.rasterize`) of
groups of fits that never come within 2.5 radii of each other.

1. **Model recenter** (`model_recenter`; `model_recenter_sweeps` = 2,
   `model_recenter_max_radii` = 0.5). Per sweep the model is remeasured.
   Each fit samples, on a grid 2.2 r wide across each node (averaged over
   the node and a voxel either side along the fit), the scan less the
   void and every other fit's drawn grey, and moves the node to the offset
   within `model_recenter_max_radii` r where its own profile correlates
   best with it; moves are smoothed over five nodes. A fit in a packed
   bundle is no longer pulled toward its neighbours' bodies and away from
   their halos.
2. **Residual births** (`residual_births`; `residual_birth_level` = 0.4,
   `residual_birth_min` = 0.45, `residual_birth_near_share` = 0.5). The
   residual (scan less void less the drawn fits, over the fits' median
   brightness, smoothed by 1 voxel) is traced with the §4 tracer where it
   is above the level. A trace is kept when its median residual is at least
   `residual_birth_min`, at most `residual_birth_near_share` of it lies
   within a radius of a fit and at most a third inside a larger type's
   fit. The recenter then runs again with the new fits.

The log entry "model pass" gives `moved` (the mean node move of the last
sweep, voxels), `born` and `void`. On the benchmark the pass took the
packed-bundle settings from 0.915 to 0.942 on structures 1–18 and from
0.911 to 0.934 on 19–36, every structure up, and roughly doubled the fit
time of a 160³ scan.

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
- **Centerline recall, precision and F1**
  (`_evaluate.centerline_agreement`). These have an owner rule of their
  own, on centerlines rather than voxels: a fit belongs to the true fiber
  whose centerline most of its samples lie within 0.5 r of, r being that
  true fiber's radius (for an oval fiber, the equivalent radius √(ab) of its
  semi-axes a and b). **Recall** is the share of true centerline length in
  the scan that the fits belonging to that fiber pass within 0.5 r of;
  **precision** is the share of fitted centerline length that lies within
  0.5 r of the centerline of the fiber its fit belongs to; **F1** is their
  harmonic mean. True pieces that only clip a corner of the scan (shorter in
  it than the fit's minimum length) are left out of recall. Pieces of one
  true fiber all count toward it, so a split fiber loses nothing where each
  piece follows it; a fit that follows one fiber and then another loses the
  length along the second. Unlike the voxel label accuracy, it ignores how
  the capsule edges fill voxels.

- **Interior ends and implied lengths:** reported for both the fit and the
  truth.
