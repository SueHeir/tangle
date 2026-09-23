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

The module is pure Python. It needs NumPy and SciPy. `tifffile` and
`matplotlib` are optional and add the TIFF and PNG outputs.

```python
import tangle.ct as ct
from tangle.units import um

spec = ct.FiberSpec(diameter=10 * um, min_bend_radius=40 * um)
fit = ct.fit_fibers(volume, voxel_size=1.3 * um, spec=spec)   # volume: (z, y, x) array
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
| `profile` | Brightness across the fiber, a `CrossSection`. Solid at brightness 1 by default (see Several fiber types). |

`FitSettings` holds the numerical settings (rounds, rates, merge gap and so
on). The defaults are meant to work without changes.

## How the fit works

The method follows the literature review in the project files. In short
(every step, threshold and function is in [ct_fitting_internals.md](ct_fitting_internals.md)):

1. **Normalize.** A light Gaussian denoise, then Otsu's threshold, maps void
   to 0 and fiber to 1. Once the first traces exist, the fiber level is reset
   to the median intensity at the fiber cores. The Otsu class median sits
   below the core, because blurred edge voxels are counted in the fiber class.
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
3. **Fit.** This is an EM-style loop. In each iteration:
   - every voxel near a fiber is assigned to the nearest capsule surface,
     the same rule Tangle's voxel exporter uses;
   - each centerline segment moves sideways toward the intensity centroid of
     the voxels it owns;
   - a bending step smooths each centerline;
   - each radius moves toward the equivalent radius of the intensity it owns,
     pulled toward the diameter you specified;
   - fiber ends grow or shrink to follow the scan;
   - overlapping fibers are pushed apart.
4. **Topology moves.** After each round of the fit:
   - fits are split at kinks the bend limit does not allow, and at
     `max_length`;
   - side-by-side fits are checked against the scan to decide whether they
     are one fiber or two (see below);
   - duplicate fits and fits with little image support are removed;
   - fragments that continue each other are merged, unless the join would
     create a kink;
   - new traces are started in foreground not yet explained by any fit.

**One fiber or two?** If two fits land on the same fiber, the non-overlap
step pushes them apart until each sits about a radius off the true axis. At
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
implied length far below the true one means the fit is over-split. For
example, the FiberForm fit without the prior has 603 interior ends and
implies 134 µm fibers; 1 mm fibers would give about 81 ± 9 ends.

Ends are not perfectly uniform in real materials, for example in layered
felts, so this is a soft cost rather than a hard count.
[`ct_fit_synthetic_long.py`](../crates/tangle_python/python/examples/ct_fit_synthetic_long.py)
tests it on a scan cropped from a larger cell, so fibers run through the
scan boundary as in a real scan. It fits with and without the prior and
scores both.

## Several fiber types

A scan can hold fiber types that differ in size and in brightness, and the
brightest fibers are not necessarily the ones of interest. Describe each
type's cross-section with a `CrossSection` and pass a list of specs:

```python
small = ct.FiberSpec(diameter=7 * um, name="fine_7um")  # solid, brightest (1)
large = ct.FiberSpec(
    diameter=19 * um,
    profile=ct.CrossSection(brightness=0.75, rim=2 * um, core=1 / 3),  # rim 0.75, core 0.25
    name="coarse_19um",
)
fit = ct.fit_fibers(volume, voxel_size=1.25 * um, spec=[small, large])
```

Brightness uses a scale where void is 0 and the brightest type is 1. How
it works:

- Void and the reference fiber level come from a multi-level threshold with
  one class per brightness level (void, large-fiber core, rim, small-fiber
  center), because a two-class one would split a dim type from the bright
  one.
- Types are fitted one after another, brightest first. A solid type that is
  brighter than the rest sees only the brightness above them, so the 7 µm
  fibers are fitted from the top grey level alone and the 19 µm rims don't
  look like them. A rimmed type's image is smoothed at half its radius,
  which fills the dim core, and rescaled so the type reads about 1 on its
  axis.
- Fibers already found are taken out of the image the later types see, and
  they keep their voxels. A cluster of small bright fibers therefore can't
  be traced as one large fiber.
- A fit of a rimmed type is kept only if the scan shows its dim core inside
  a brighter rim along it.
- Radii are sized from the scan itself by inverting the type's profile. For
  a rimmed type, the owned brightness isn't just π r².
- `fit.json` stores every spec and each fiber's type. `score()` reports
  recovery for each type. `suggested_population()` returns one population
  per type.

[`ct_fit_two_types.py`](../crates/tangle_python/python/examples/ct_fit_two_types.py)
renders and fits such a scan: 7 µm solid fibers and 19 µm fibers with a
rim at 0.75 and a core at 0.25.

## Checking a fit against known answers

`ct.synthetic_ct(assembly_or_result, voxel_size)` renders any Tangle structure
as a CT-like scan with known ground truth. It builds on `export_puma`: the
smooth interface image gives partial-volume occupancy, and the exporter's
fiber IDs give the true labels. On top of that it applies a Gaussian blur,
void/fiber contrast, noise and a weak low-frequency drift.

`ct.score(fit, scan)` reports:

- fibers recovered whole, split into pieces, or missed;
- false and merged fits;
- mean centerline error;
- diameter bias;
- solid Dice;
- the fraction of fiber voxels assigned to the correct fiber.

[`ct_fit_synthetic.py`](../crates/tangle_python/python/examples/ct_fit_synthetic.py)
runs a full check: 40 relaxed wavy planar fibers of 12 µm diameter, imaged at
1.5 µm voxels (8 voxels across a fiber, noise σ = 12% of contrast, 160³
voxels). Current results on that scan:

| Measure | Result |
| --- | --- |
| Fibers recovered whole / split / missed | 34 / 4 / 2 of 40 |
| False fits | 0 |
| Mean centerline error | 0.41 voxel (0.6 µm) |
| Diameter bias / RMS error | −0.05 µm / 0.16 µm |
| Fiber voxels labeled with the right fiber | 95% |
| Run time (one CPU core) | about 2.5 minutes |

The unit tests use a small three-fiber case. It is recovered exactly: 0.2 voxel
centerline error, and diameter within 1%.

`ct.geometry_report(centerlines, radii, min_bend_radius)` measures how far
fibers are from valid Tangle fibers: the curvature ratio against the bend
limit, the deepest overlap between two fibers and the shortest segment.
[`ct_gpu_geometry_check.py`](../crates/tangle_python/python/examples/ct_gpu_geometry_check.py)
uses it to check that Tangle's solver, run with the scan as an extra force,
repairs fiber geometry: on the CPU fit with the image off, on true fibers
damaged with sharp kinks, and on the CPU fit with the image on.

## Real data: PuMA FiberForm

[`ct_fit_puma_fiberform.py`](../crates/tangle_python/python/examples/ct_fit_puma_fiberform.py)
downloads PuMA's `200_fiberform.tif` and fits it. This is a real 200³
micro-CT crop of FiberForm carbon felt at 1.3 µm voxels, distributed with
PuMA under the NASA Open Source Agreement.

There is no fiber-level ground truth for this scan, so judge the result from
the overlay. FiberForm fibers have lobed, non-circular cross-sections and
carbonized binder at many crossings. A circular fit is therefore an
equal-area approximation, and binder blobs are left unexplained or become
short fits.

## Limits and next steps

- **Cross-sections are circular.** Tangle's capsules have a single radius, so
  ribbon-like or lobed fibers get an equal-area diameter.
- **No explicit PSF.** The data force is an intensity centroid rather than a
  full blurred forward model. It is unbiased for a symmetric point-spread
  function but ignores other artifacts, such as streaks or rings.
- **Speed.** Everything runs single-threaded in Python/NumPy. A 160³ scan
  with 40 fibers takes minutes. Rasterization and the per-segment data force
  are the hot spots and would be natural to move into Rust.
- **Validation.** Next are benchmarks with real ground truth: the
  Math2Market FiberFind validation set and the DTU multimodal glass-fiber
  scans. See the project's dataset notes.
