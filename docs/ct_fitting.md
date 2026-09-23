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

The input is a grey-level scan or, better, a **binary fiber mask** (a
`bool` array, or any array with two values). A mask may over-reach: some
background kept as fiber and fibers a little thicker than they are. Pass
`exclude=` with a mask of voxels known not to be fiber to hide them from
the fit.

```python
import tangle.ct as ct
from tangle.units import um

spec = ct.FiberSpec(diameter=10 * um, min_bend_radius=40 * um)
fit = ct.fit_fibers(mask, voxel_size=1.3 * um, spec=spec)   # mask: (z, y, x) array
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

`FitSettings` holds the numerical settings (rounds, solver iterations,
merge gap and so on). The defaults are meant to work without changes.
`backend="cpu"` runs the solver where there is no GPU.
`thickness_margin_voxels` says how much a generous mask over-reaches
(voxels, in radius); by default it is estimated from the fits.

## How the fit works

The method follows the literature review in the project files. In short
(every step, threshold and function is in [ct_fitting_internals.md](ct_fitting_internals.md)):

1. **Input.** A mask is used as 0 (void) and 1 (fiber), with holes the size
   of a fiber core filled (a threshold can miss a dim core) and a light
   blur. A grey scan is denoised and mapped to 0…1 with Otsu's threshold,
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
  over-reaches. Its type is the spec whose diameter is nearest in ratio,
  and that type sets its radius prior, bend limit, minimum length and
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
19 µm fibers with a bright rim and a dim core, thresholds a generous mask
(the large fibers' cores come out as holes, which are filled) and fits both
types from it.

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

## Examples

[`ct_examples.py`](../crates/tangle_python/python/examples/ct_examples.py)
holds every example, and every example is run the same way. A synthetic
scan with known true fibers is thresholded into a generous mask, then
fitted from the mask with Tangle's solver on the GPU. Each writes exactly
these files to `<output>/<example>/`. The folder is emptied first, so there
is only ever one result per example:

| File | Contents |
| --- | --- |
| `raw.tif` | the rendered scan |
| `mask.tif` | the fiber mask the fit starts from |
| `true.tif` | the true fibers, one color per fiber, over the scan (RGB) |
| `segment.tif` | the fitted fibers, one color per fiber, over the scan (RGB) |
| `fit.json` | the fit (`ct.load_fit`) |
| `score.json` | score against the truth, geometry report, run time |

`<output>/summary.md` has one row per example. The examples are
`single_type` (40 wavy 12 µm fibers), `long_fibers` (long fibers cropped
by the scan), `two_types` (7 µm and 19 µm fibers), and one `scenario_*` per
fitting step: a straight fiber, interior ends, gaps of 2, 6 and 12 radii,
crossings at 90° and 30°, touching parallel fibers, a piece below the
minimum length and a bend near the limit.

```
python ct_examples.py --list
python ct_examples.py                       # all, into $TANGLE_CT_OUTPUT or examples/output/ct
python ct_examples.py two_types scenario_gap_6r --output ~/ct-results
```

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
