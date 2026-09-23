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

`FitSettings` holds the numerical settings (rounds, rates, merge gap and so
on). The defaults are meant to work without changes.

## How the fit works

The method follows the literature review in the project files. In short:

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
