# Fitting Tangle fibers to CT scans

`tangle.ct` segments individual fibers in a CT scan using what you already
know about the fibers. You provide the diameter, and optionally a bend limit
and a length range. The fitter places Tangle fibers and moves them until they
match the scan. It returns:

- a Tangle configuration: an `Assembly` to continue working with in Tangle,
  a `FiberPopulation` built from the fitted statistics, and a `fit.json` file
  you can reload later;
- a label volume, where each voxel holds the ID of the fiber that fills it;
- an overlay of the fibers on the raw scan, one color per fiber, as an RGB
  TIFF stack and a PNG with three orthogonal slices.

The fibers move in Tangle's own solver (`tangle.ImageRelaxer`), on the GPU
by default, with the scan as an extra force. Contact, segment lengths and
the bend limit hold throughout, so every fit comes out as round,
non-overlapping tubes within the bend limit. The rest is Python and needs
NumPy and SciPy. `tifffile` and `matplotlib` are optional and add the TIFF
and PNG outputs.

The input is the **raw grey scan with a grey profile per fiber type**
(`FiberSpec(profile=(...))`: the type's grey at evenly spaced radii from
the axis to the surface, measured on a few fibers of the scan), the raw
scan with just a grey range per type (`FiberSpec(intensity=(low, high))`),
a **binary fiber mask** (a `bool` array, or any array with two values), or a
grey scan without ranges. The ranges decide what is fiber: a voxel inside
a type's range is fiber of that type, a voxel between the void grey and a
range is part fiber (a fiber edge), and anything else is void, so the fit
sees sub-voxel edges and each fiber's grey as well as its size. A mask may
over-reach: some background kept as fiber and fibers a little thicker than
they are. Pass `exclude=` with a mask of voxels known not to be fiber to
hide them from the fit.

```python
import tangle.ct as ct
from tangle.units import um

spec = ct.FiberSpec(diameter=10 * um, min_bend_radius=40 * um)
fit = ct.fit_fibers(mask, voxel_size=1.3 * um, spec=spec)   # mask: (z, y, x) array
profiled = spec.replace(profile=(21e3, 21e3, 22e3, 30e3, 31e3))  # grey from axis to surface
fit = ct.fit_fibers(volume, 1.3 * um, profiled)              # raw scan, fits judged on grey
fit = ct.fit_fibers(mask, 1.3 * um, spec, exclude=not_fiber)  # optional: voxels known not to be fiber
fit.write("fit_output", volume=volume)   # fit.json, labels.tif, overlay.tif, overlay.png
assembly = fit.to_assembly()             # cell = scanned volume, not periodic
population = fit.suggested_population()  # generate statistically similar structures
relaxed, run = fit.relax()               # optional: clean up remaining overlaps with Tangle's solver
```

## What you specify

| `FiberSpec` field | Meaning |
| --- | --- |
| `diameter` | Typical fiber diameter. The scan needs at least 2 voxels across a fiber; 6 or more works best. |
| `diameter_tolerance` | Fitted diameters are kept within `diameter × (1 ± tolerance)`. Default 0.25. |
| `min_bend_radius` | The tightest bend a fiber can make. It limits how sharply a trace can turn, decides where a fit is split at a kink, and becomes the Tangle material's bend limit. Default 5 diameters. |
| `min_length` | Shorter fragments are dropped. Default 3 diameters. |
| `max_length` | Optional. Fits longer than this are split where the scan gives them the least support. |
| `length` | Optional typical fiber length. Turns on the fiber-length prior (below). A rough value is enough. |
| `profile` | Optional grey profile of this type in the raw scan (after a σ = 0.7 voxel denoise): grey values at evenly spaced radii from the axis to the surface, e.g. a dim core in a bright rim. Give it for every type or none. It sets the grey ranges and noise, fits are judged by drawing them with their profiles and comparing with the scan, and a fiber's type is the profile that matches the grey across it. `tangle.ct._grey.measure_profiles` measures profiles around known fibers. |
| `intensity` | Optional `(low, high)` grey range of this type's voxels in the raw scan (after a light σ = 0.7 voxel denoise). Give it for every type or none. Take it from the bright peak of the histogram: blurred edges and dim cores are handled without it. |

`FitSettings` holds the numerical settings (rounds, solver iterations,
merge gap and so on). The defaults are meant to work without changes.
`backend="cpu"` runs the solver where there is no GPU.
`thickness_margin_voxels` says how much a generous mask over-reaches
(voxels, in radius); by default it is estimated from the fits.

## How the fit works

The method follows the literature review in the project files. In short
(every step, threshold and function is in [ct_fitting_internals.md](ct_fitting_internals.md)):

1. **Input.** With grey ranges, the denoised scan becomes each voxel's
   fiber fraction: 1 inside a range, rising linearly from the void grey
   (the median of voxels darker than every range) up to a range, fading
   out over one range width above it; core-sized holes are filled. A mask
   is used as 0 (void) and 1 (fiber), with holes the size
   of a fiber core filled (a threshold can miss a dim core) and a light
   blur. A grey scan without ranges is denoised and mapped to 0…1 with Otsu's threshold,
   then re-leveled on the traced fiber cores. Estimating grey levels fails
   when fibers are a small part of the scan (the threshold then splits the
   noise), which is why a mask is the better input.
2. **Trace.** Seeds are points on the ridge of the foreground distance
   transform, deepest first. From each seed the trace:
   - steps along the local tube axis, taken from the Hessian at a scale of
     about 0.6 radius and blended with the previous direction;
   - re-centers on the intensity centroid in the cross-section at each step;
   - never turns faster than the bend limit allows;
   - stops where the core intensity falls below 0.5.

   Voxels claimed by earlier traces count for less when re-centering, so a
   trace crosses another fiber instead of turning onto it. A trace that mostly
   follows an existing fit is discarded.
3. **Fit.** A few solver batches per round. In each batch:
   - fiber ends grow or shrink to follow the scan;
   - Tangle relaxes the fibers with the image force on: every vertex moves
     sideways toward the centroid of the scan it owns (nearest capsule
     surface, the rule Tangle's voxel exporter uses), while contact,
     stretch, bending and the bend limit act as in any Tangle run;
   - a short relaxation with the image off leaves the fibers admissible;
   - each fiber's type and radius are read off its thickness (below).
4. **Topology moves.** After each round of the fit:
   - fits are split at kinks the bend limit does not allow, and at
     `max_length`;
   - side-by-side fits are checked against the scan to decide whether they
     are one fiber or two (see below);
   - duplicate fits and fits with little image support are removed;
   - fragments that continue each other are merged, unless the join would
     create a kink;
   - new traces are started in foreground not yet explained by any fit.

**One fiber or two?** If two fits land on the same fiber, contact pushes
them apart until each sits about a radius off the true axis. At
that point they look like two touching fibers. To tell the cases apart, the
fitter renders the neighborhood twice: once with both fibers, and once with a
single fiber along their midline. It keeps whichever rendering leaves the
smaller squared residual against the scan.

## The fiber-length prior

If you know roughly how long the fibers are, set `FiberSpec(length=...)`.
The fitter then treats fiber ends as rare, which stops it from breaking long
fibers into pieces wherever the scan is unclear.

The reasoning: if fibers of mean length L have their ends spread uniformly
through the material, a scan holds 2Λ/L fiber ends, where Λ is the total
fiber length inside it. That holds however the scan boundary cuts the
fibers, and ends on the boundary are fibers leaving the scan, so they don't
count. Put another way, a break is expected about once per L of fiber, and
a position along a fiber can be resolved to about a diameter D, so every end
inside the scan costs ln(L/D) nats. For 1 mm fibers of 10 µm that is 4.6.

The scan's evidence is measured in the same units. It is the change in
squared residual between two renderings of the neighborhood, divided by
2σ²·πr². Here σ² is the fit's own mean squared residual per voxel near the
fibers (noise plus misfit such as lobed cross-sections), and πr² is one fiber
cross-section, the scale on which neighboring voxels move together. With the
prior on:

- **Joins** are considered for any two ends within 16 radii whose directions
  agree within 45°, including pieces that overlap a little. A join is made
  when the joined fiber explains the scan better than the two pieces, with
  2·ln(L/D) credited for the two ends it removes. The joins are chosen
  together, best first, with each end used once, so pieces meeting at a
  crossing are paired the way the scan supports best.
- **Kinks** sharper than the bend limit are first smoothed out. A fiber is cut
  only when the smoothed fiber explains the scan worse by more than the two
  new ends cost. A trace that jumped onto another fiber can't be smoothed
  without leaving both fibers, so it is still cut.
- **One fiber or two** also charges or credits the ends each rendering has.

Every fit also reports its interior ends and the fiber length they imply,
2Λ divided by the number of interior ends (`interior_ends` and
`implied_mean_length` in the population summary, per round in the history).
With `length` set it adds the count expected, and its Poisson spread. An
implied length far below the true one means the fit is over-split.

Ends are not perfectly uniform in real materials, for example in layered
felts, so this is a soft cost rather than a hard count.
The `long_fibers` example (below) tests it on a scan cropped from a larger
cell, so fibers run through the scan boundary as in a real scan.

## Several fiber types

A scan can hold fiber types that differ in size. Pass a list of specs, and
the fitter decides each fiber's type by its size:

```python
fine = ct.FiberSpec(diameter=7 * um, name="fine_7um")
coarse = ct.FiberSpec(diameter=19 * um, name="coarse_19um")
fit = ct.fit_fibers(mask, voxel_size=1.25 * um, spec=[fine, coarse])
```

- A fiber's thickness is the foreground's depth (distance to the nearest
  void voxel) along its centerline, less the margin by which the mask
  over-reaches. With grey ranges, a fiber whose centerline is at least
  60% in one type's range (a dim core ringed by that range counts) takes
  that type; otherwise, and without ranges, its type is the spec whose
  diameter is nearest in ratio. The type sets its radius prior, bend limit, minimum length and
  length prior.
- Types are re-chosen after every solver batch, so a fiber can change type
  as its fit improves. Splits and joins happen within a type.
- Tracing starts with the largest type, seeded only where the foreground is
  thicker than the smaller types could make it.
- The margin is estimated as the median excess of the fibers over their
  nearest type, or set with `FitSettings(thickness_margin_voxels=...)`.
- `fit.json` stores every spec and each fiber's type. `score()` reports
  recovery for each type. `suggested_population()` returns one population
  per type.

The `two_types` example (below) renders a scan of 7 µm solid fibers and
19 µm fibers with a bright rim and a dim core and fits both types from
the raw scan with a grey profile per type (dim core, bright rim), which
also decides each fiber's type, or from a generous mask with
`--input mask`.

## Checking a fit against known answers

`ct.synthetic_ct(assembly_or_result, voxel_size)` renders any Tangle structure
as a CT-like scan with known ground truth. It builds on `export_puma`: the
smooth interface image gives partial-volume occupancy, and the exporter's
fiber IDs give the true labels. On top of that it applies a Gaussian blur,
void/fiber contrast, noise and a weak low-frequency drift.

`scan.fiber_mask(level=0.5)` thresholds the scan into a binary mask,
`level` of the way from the void to the fiber grey level; below 0.5 it
over-reaches, as a generous threshold does.

`ct.score(fit, scan)` reports:

- fibers recovered whole, split into pieces, or missed;
- false and merged fits;
- mean centerline error;
- diameter bias;
- solid Dice;
- the fraction of fiber voxels assigned to the correct fiber.

The unit tests use a small three-fiber case.

`ct.geometry_report(centerlines, radii, min_bend_radius)` measures how far
fibers are from valid Tangle fibers: the curvature ratio against the bend
limit, the deepest overlap between two fibers and the shortest segment.

## How sure the fit is

Every fitted node gets a confidence in [0, 1], computed without any ground
truth, in `FitResult.confidence` (one array per fiber) and in `fit.json`.
`FitResult.confidence_volume()` spreads it over the fitted voxels. A node
is trusted when, where it sits:

- the scan is fiber across the core of its capsule (**image**);
- just outside its capsule the scan is void, or belongs to another fit
  (**surround**: unexplained fiber there means a fiber is missing, or this
  fit is off-center);
- its core is not also inside another fit (**ownership**);
- the foreground's thickness there matches its radius (**thickness**: a fit
  along the contact of two touching fibers reads the wrong thickness);
- it barely moved in the final solve (**stability**).

The confidence is the product of the five.

The fit then uses it to **redraw the unsure parts**
(`FitSettings.redraw_passes`, 5 by default). Each pass:

1. cuts every stretch whose confidence is below
   `FitSettings.confidence_threshold` (0.5) out of its fiber, keeping the
   sure pieces, and groups the cut stretches into regions;
2. treats every sure piece that ends in a region as a loose end, and tries
   **every combination**: each loose end either connects to another loose
   end of the same fiber type, through a smooth bridge that keeps within
   the bend limit, or the fiber ends in the region (growing along its own
   direction until the scan stops being fiber). Each combination is scored
   by how well it explains the scan in the region, the overlaps it makes,
   and the fiber ends it leaves, priced by the fiber-length prior for the
   length the fiber would have: ending a fiber much shorter than the
   typical length is dear, ending one near or past it cheap, and a join
   that makes a fiber far longer than typical costs extra
   (`FitSettings.length_shape`, a gamma length distribution; 1 prices
   every end the same). A sure piece that already ends inside the region
   is a loose end too, so it can be joined. The best three are each solved on the GPU with
   the sure pieces pinned, and every region keeps the one that best
   explains the scan there (by the same score as step 5); a retry tries
   the next three
   (`FitSettings.redraw_plans`, `FitSettings.redraw_moves = "match"`; `"grow"` instead only grows the
   ends along their own direction);
3. joins ends that meet and traces new fibers in whatever is still
   unexplained, with the usual steps;
4. re-solves with the sure pieces pinned, so they stay where they are and
   everything else fits around them; a short settle without the scan then
   runs unpinned, so fibers that touch can still be pushed apart;
5. keeps or reverts the redraw **region by region**: a region's redraw is
   kept only if the scan's grey there is matched better by the fit drawn
   with its fiber profiles (by more than a nat; the default with
   profiles), or, without profiles, if it leaves fewer voxels wrong there,
   foreground the fit misses plus fit over void
   (`FitSettings.redraw_score = "mask"`), or with
   `"confidence"` if it raises that region's **sure coverage** (its
   foreground explained by the fit, each voxel weighted by the confidence
   of the fit that owns it). Two regions are decided together only when
   one redrawn stretch spans both. Each region's decision stands: there is
   no second check on the whole pass.

A region whose redraw failed gets a different move the next time, with a
wider cut: the next-best combination (with `"grow"`: the second try
re-traces it from fresh seeds instead of growing into it, and the third
grows again with the shortest pieces first). After
`FitSettings.redraw_attempts` (3) failures it is left alone, so every
pass can only improve the fit and the passes end. Keeping or reverting uses the
confidence without the stability check, since a redrawn stretch moves
because it was redrawn. The history records every pass: regions, how many
were kept, widened or given up, and the sure coverage.

## Examples

[`ct_examples.py`](../crates/tangle_python/python/examples/ct_examples.py)
holds every example, and every example is run the same way. A synthetic
scan with known true fibers is fitted from the raw scan, with each type's
grey profile measured around its true fibers (as one would on a few
fibers of a real scan), with Tangle's solver on the GPU.
`--input mask` fits a generous thresholded mask instead. Each writes exactly
these files to `<output>/<example>/`. The folder is emptied first, so there
is only ever one result per example:

| File | Contents |
| --- | --- |
| `raw.tif` | the rendered scan |
| `input.tif` | what the fit sees, 0 (void) to 255 (fiber): the fiber fraction from the grey ranges, or the mask |
| `true.tif` | the true fibers, one color per fiber, over the scan (RGB) |
| `segment.tif` | the fitted fibers, one color per fiber, over the scan (RGB) |
| `diff.tif` | where fit and truth disagree, over the dimmed scan (RGB): red = true fiber left empty (missed), blue = fit over void (extra), orange = fiber voxel given to the wrong fiber |
| `confidence.tif` | the fit's own confidence in each fitted voxel (no ground truth used), over the dimmed scan (RGB): green = sure, yellow, red = unsure. `score.json` records how well it picks out the errors in `diff.tif` |
| `fit.json` | the fit (`ct.load_fit`) |
| `score.json` | score against the truth, geometry report, run time |

`<output>/summary.md` has one row per example. The examples are
`single_type` (40 wavy 12 µm fibers), `long_fibers` (long fibers cropped
by the scan), `two_types` (7 µm and 19 µm fibers), and one `scenario_*` per
fitting step: a straight fiber, interior ends, gaps of 2, 6 and 12 radii,
crossings at 90° and 30°, touching parallel fibers, a piece below the
minimum length and a bend near the limit, plus `scenario_missed_fiber`:
`single_type`'s densest bundle (a fiber touching 11 others at 4–88°,
which the fit split in two) cut out into its own 150 µm scan.

`varied_1` … `varied_8` are fresh structures for checking that the fitter
holds up on structures it was not tuned on. Each draws its settings from
its own seed, so a name is always the same structure: 8–16 µm fibers at
2.5–4.5 voxels radius, planar, aligned, biaxial and isotropic orientations
(two each), solid fraction 0.04–0.14, and varied waviness, bend limit, scan
noise and blur, all in a scan 160 voxels a side. `score.json` records the
settings. They run only when named, or with `--varied`.

```
python ct_examples.py --list
python ct_examples.py                       # all but varied_*, into $TANGLE_CT_OUTPUT or examples/output/ct
python ct_examples.py --varied              # all, including varied_*
python ct_examples.py two_types scenario_gap_6r --output ~/ct-results
```

Results of the last full run: [`examples/ct_results/`](../crates/tangle_python/python/examples/ct_results/) (summary.md, per-example score.json and fit.json; TIFFs for two_types and varied_3; pictures in `images/`, made by `make_images.py`).

A new example is a function returning an `Example` (scan, specs), added to
`EXAMPLES`; the runner writes its files, so every example keeps the same
layout.

## Limits and next steps

- **Cross-sections are circular.** Tangle's capsules have a single radius, so
  ribbon-like or lobed fibers get an equal-area diameter.
- **No explicit PSF.** The data force is an intensity centroid rather than a
  full blurred forward model. It is unbiased for a symmetric point-spread
  function but ignores other artifacts, such as streaks or rings.
- **Speed.** The continuous fit costs about 1.3 ms per solver iteration on
  an M-series GPU even for 475 fibers; tracing and the one-fiber-or-two
  check, single-threaded Python, are the slow steps. The CPU backend runs
  the same steps but is far too slow for real scans; the tests run on the
  GPU too and are skipped on CI, which has none.
- **Validation.** Next are benchmarks with real ground truth: the
  Math2Market FiberFind validation set and the DTU multimodal glass-fiber
  scans. See the project's dataset notes.
