# The dense_hard benchmark

`dense_hard_<n>` is a family of synthetic scans in
[`ct_examples.py`](../crates/tangle_python/python/examples/ct_examples.py)
built to be hard for `tangle.ct`: packed bundles of fine fibers with flat
oval coarse fibers, scanned by the simulated CT scanner (`ct.Scanner`, in
`tangle/ct/_scanner.py`) at 1 µm voxels. Every fiber has its own
brightness, and the coarse fibers are dim with a dimmer core, so no grey
threshold separates the two types. The fit gets one broad grey range for
both.

This page describes what a structure is, how to run and score one, the
recommended `FitSettings`, and how each change to them moved the score.
The fitting method itself is in [ct_fitting.md](ct_fitting.md) and
[ct_fitting_internals.md](ct_fitting_internals.md).

## What a structure is

`dense_hard_settings(n)` draws structure `n` from the seed `7000 + n`, so a
name is always the same structure. Five fields are drawn; the rest are
fixed:

| Field | Value | Meaning |
| --- | --- | --- |
| `voxel_um` | 1.0 | voxel size (µm) |
| `side_um` | 160 | the scan is a cube 160 voxels a side |
| `orientation`, `tilt` | planar, **0.15–0.30** | fibers lie near the xy plane, tilted out of it by at most `tilt` radians (`PlanarOrientation(max_tilt=...)`) |
| `fine_um` | 6.5 | fine fiber diameter: round, 3.25 voxels in radius |
| `fine_fraction` | **0.10–0.13** | target volume fraction of fine fibers (the fibers in the scan fill 0.70–0.97 of it) |
| `per_bundle` | **7 or 19**, equal odds | fine fibers per bundle |
| `coarse_um`, `coarse_thickness_ratio` | 17.3, 0.8 | coarse fibers are ovals 17.3 µm wide and 13.84 µm thick |
| `coarse_fraction` | **0.03–0.05** | target volume fraction of coarse fibers (0.72–1.10 of it in the scan); they lie loose, not in bundles |
| `coarse_brightness` | 0.6 | a coarse fiber's rim brightness, where a fine fiber is 1 and void 0 |
| `coarse_rim_um`, `coarse_core` | 3.0, 0.6 | a rim 3 µm thick on the equivalent radius (a round core 4.7 µm in radius) at 0.6 of the rim's brightness (0.36 of a fine fiber) |
| `brightness_spread` | 0.25 | each fiber's brightness is scaled by its own log-normal factor (σ = 0.25, median 1) |
| `length_fraction` | 0.55–0.90 | drawn fiber lengths as fractions of the side: 88–144 µm, mean 116 µm (trimmed bundle members make the true fine fibers 71–145 µm, median 106 µm) |
| `photons` | **1040–1560** | photons per detector pixel: `SCANNER`'s 1300 times 0.8–1.2 |
| `noise_blur` | 0 | no blur after the photons are counted, so the detector noise is pixel to pixel, not blotchy |
| `resolution_um` | 2.0 | the scanner's resolution (full width at half maximum of its blur) |
| `fiber_motion_um` | 1.0 | each fiber moves 1 µm (RMS) on its own path during the scan |
| `delta_beta` | 12, 9 | phase-to-absorption ratio of the fine and the coarse material, which sets the edge fringes |

(Drawn fields in bold. `SCANNER` also sets `fiber_attenuation` 0.005 per
voxel and `propagation` 4 (pixels²); the rest are `ct.Scanner` defaults:
`void_attenuation` 0.0005 per voxel, no ring artefacts, and no drift of the
whole structure during the scan.)

`scanned()` turns these into a structure:

- **Fiber counts.** Each type gets `fraction × volume / (cross-section ×
  mean length)` fibers, at least 4, with the round cross-section π(d/2)² of
  the fiber's width d. The fine fibers are split into `count // per_bundle`
  bundles.
- **Bundles.** Each bundle follows one leader fiber drawn from the fine
  population, with a stiffer bend limit so its outer members stay within
  theirs. The members are hexagonally packed around the leader (a center, a
  ring of 6, then a ring of 12) at a pitch of 1.02 diameters, so
  neighbours start 0.13 µm apart. Each member's ends are trimmed by up to a
  tenth of its length, so the ends are staggered. Members that would stick
  out of the cell's top or bottom, or bend past the limit, are left out,
  so a bundle can hold fewer than 7 or 19 fibers.
- **Shape.** Every fiber has a waviness of 0.15–0.6 diameters and a bend
  limit of 5 diameters (32.5 µm fine, 86.5 µm coarse). The cell is not
  periodic: the scan is the whole cell.
- **Truth.** The fibers are relaxed once with Tangle's solver and cached in
  `<output>/.cache`, then scanned by `ct.Scanner` with the profiles above.
- **Specs.** The fit gets one `FiberSpec` per type: the diameter, a bend
  limit of 5 diameters, `length=116 µm` (the mean of the drawn range)
  and, for the coarse type, `thickness=13.84 µm`.

In structures 1–36 this gives 82–133 fine and 5–8 coarse fibers per scan.
None of them is a stub (a piece shorter than the fit's minimum length).
Structures 1–18 hold 9 of each bundle size; structures 19–36 hold 8 with
7-fiber bundles and 10 with 19-fiber bundles.

What makes the set hard:

- the fine fibers are 3.25 voxels in radius, packed with almost no gap,
  and blurred by a 2 µm resolution and 1 µm of motion, so touching fibers
  run together;
- the brightness spread puts some fine fibers well below their neighbours,
  and a dim fiber inside a bundle is easy to miss;
- a coarse fiber is dimmer than the fine fibers and dimmer still in its
  core, so its rim reads as a bright lobe that a fine fit can follow, and
  its depth in the foreground reads thin through the dim core;
- a 7-fiber bundle (about 20 µm across) is about as wide as a coarse fiber
  (17.3 µm);
- one grey range serves both types, so the grey alone cannot type a fiber.

## Running it

The structures run only when named. `DENSE_HARD_COUNT` (default 18) sets
how many are registered, so structures past 18 need it:

```bash
cd crates/tangle_python/python/examples
SETTINGS=(
  --set bright_seed_strength=0.05 --set graded_image=true --set graded_smallest_only=true
  --set grey_checked_traces=true --set coarse_trace_axis_check=true
  --set coarse_axis_grey_max=0.15 --set coarse_axis_grey_min=0.0
  --set coarse_disc_max=0.8 --set coarse_disc_classify=true
  --set coarse_disc_trace=false --set coarse_disc_end=false
  --set radius_cap_prior=true --set births_solve_pinned=true
  --set birth_ridge_min=0.5 --set birth_ridge_smallest_only=true --set coarse_birth_disc_max=0.8
  --set prior_merge_gap_radii=4
  --set recenter_on_grey=true --set recenter_sigma_voxels=1.6
  --set recenter_each_solve=true --set recenter_last=true
  --set final_births=true --set redraw_passes=0
  --set ridge_finish=true --set ridge_sigma_voxels=1.2 --set ridge_min_length_scale=0
  --set ridge_refine=true --set ridge_rim_share=0.7
  --set ridge_rim_sigma=1.2 --set hole_births=true --set hole_birth_ridge_min=0.3
  --set hole_birth_grey_min=0.65 --set hole_birth_passes=2 --set hole_birth_rim_drop=true
  --set straight_join=true --set bend_finish=true --set bend_finish_grey_min=0.4
)
python ct_examples.py dense_hard_5 --input broad "${SETTINGS[@]}"
python ct_examples.py $(seq -f 'dense_hard_%g' 1 18) --input broad "${SETTINGS[@]}" --output ~/ct-dense-hard
DENSE_HARD_COUNT=36 python ct_examples.py $(seq -f 'dense_hard_%g' 19 36) --input broad "${SETTINGS[@]}"
```

(An array keeps the flags as separate words in both bash and zsh.)

`--input broad` gives both types the same grey range
(`FiberSpec(intensity=...)`): from a third of the way to the brightest
type's axis grey, above the void grey, to a third past it
(`BROAD_RANGE = (0.33, 1.32)`, on the scan denoised at σ = 0.7 voxels).
It stands for a range picked by eye from a histogram: only the brightest
axis grey, and the cut-off below which voxels count as void, are read off
the true fibers, and there is no range per type.

Each structure writes the usual files to `<output>/dense_hard_<n>/` and a
row to `<output>/summary.md` (see [ct_fitting.md](ct_fitting.md#examples)).
The first run of a structure also relaxes its true fibers, which takes
longer; later runs read them from the cache. With these settings, a fit
took about 25–80 s per structure on the GPU, three fits at a time.

The same fit from Python, on a scan `volume` with 1 µm voxels:

```python
import tangle.ct as ct
from tangle.units import um

settings = ct.FitSettings(
    backend="wgpu",
    # tracing and typing
    bright_seed_strength=0.05,
    graded_image=True,
    graded_smallest_only=True,
    grey_checked_traces=True,
    coarse_trace_axis_check=True,
    coarse_axis_grey_max=0.15,
    coarse_axis_grey_min=0.0,        # step 4
    coarse_disc_max=0.8,             # step 3
    coarse_disc_classify=True,       # step 3
    coarse_disc_trace=False,         # step 3
    coarse_disc_end=False,           # step 3
    radius_cap_prior=True,
    # solves, new fibers, recentering
    births_solve_pinned=True,
    birth_ridge_min=0.5,
    birth_ridge_smallest_only=True,  # step 6
    coarse_birth_disc_max=0.8,       # step 6
    prior_merge_gap_radii=4,
    recenter_on_grey=True,
    recenter_sigma_voxels=1.6,
    recenter_each_solve=True,
    recenter_last=True,
    final_births=True,
    redraw_passes=0,                 # step 5
    # ridge finish
    ridge_finish=True,
    ridge_sigma_voxels=1.2,          # step 1
    ridge_min_length_scale=0,        # step 1
    ridge_refine=True,               # step 1
    ridge_rim_share=0.7,             # step 4
    # hole births
    ridge_rim_sigma=1.2,             # step 2
    hole_births=True,                # step 2
    hole_birth_ridge_min=0.3,        # step 2
    hole_birth_grey_min=0.65,        # step 2
    hole_birth_passes=2,             # step 2
    hole_birth_rim_drop=True,        # step 6
    # last passes
    straight_join=True,              # step 6
    bend_finish=True,                # step 6
    bend_finish_grey_min=0.4,        # step 6
)
grey = (low, high)  # one grey range for both types, as --input broad picks it
fine = ct.FiberSpec(diameter=6.5 * um, min_bend_radius=32.5 * um, length=116 * um,
                    intensity=grey, name="fine_6.5um")
coarse = ct.FiberSpec(diameter=17.3 * um, thickness=13.84 * um, min_bend_radius=86.5 * um,
                      length=116 * um, intensity=grey, name="coarse_17.3um")
fit = ct.fit_fibers(volume, 1 * um, [fine, coarse], settings)
```

The steps named in the comments are described under
[What each step changed](#what-each-step-changed). The fields without a
step are the starting settings.

## How a fit is scored

The benchmark number is the **centerline F1** that `ct.score` reports
(`_evaluate.centerline_agreement`), averaged over the structures with equal
weight. `summary.md` has it in the column "centerline recall / precision /
F1"; the runner does not print the mean.

- Every true and fitted centerline is resampled every half voxel, and only
  the points inside the scan count.
- A fitted point is **on** a true fiber when that fiber's centerline is the
  nearest true centerline and lies within half the fiber's radius: 1.63
  voxels for a fine fiber. An oval coarse fiber counts with its equivalent
  radius, √(ab) = 7.74 voxels, so 3.87 voxels.
- Each fit **belongs** to the true fiber it is most often on.
- **Recall** is the share of true centerline length that lies within that
  tolerance of a fit belonging to the same fiber. The pieces of a split
  fiber all count.
- **Precision** is the share of fitted centerline length that is on the
  fiber the fit belongs to. A fit that follows one fiber and then hops to a
  neighbour loses the length on the neighbour.
- **F1** is their harmonic mean.
- True pieces shorter in the scan than the fit's minimum length (stubs)
  are left out. This set has none.

F1 counts length, not fibers. A fiber traced in three pieces scores as well
as one traced whole, so read the split, merged and fitted counts in
`summary.md` alongside it.

## Results

Structures 1–18 were used to choose the settings. Each step below adds
settings to the one before. F1 is the mean over the structures; the bundle
columns average the 9 structures with 7-fiber and the 9 with 19-fiber
bundles.

| Step | F1, all 18 | Recall / precision | 7-fiber bundles | 19-fiber bundles |
| --- | --- | --- | --- | --- |
| Starting settings | 0.8400 | 0.787 / 0.904 | 0.8926 | 0.7875 |
| 1. Sharper ridge-finish grey | 0.8713 | 0.842 / 0.904 | 0.9158 | 0.8268 |
| 2. Hole births | 0.8903 | 0.880 / 0.902 | 0.9296 | 0.8510 |
| 3. Disc typing of coarse fits | 0.9030 | 0.896 / 0.911 | 0.9245 | 0.8814 |
| 4. Dim-axis typing and rim drop | 0.9015 | 0.894 / 0.910 | 0.9301 | 0.8729 |
| 5. No redraw passes | 0.9065 | 0.897 / 0.917 | 0.9297 | 0.8833 |
| 6. Coarse births, hole rim test, straight join, bend finish (**current**) | **0.9150** | 0.895 / 0.936 | **0.9348** | **0.8952** |

Structures 19–36 were not used for tuning. They check that the gains
carry over to structures the settings have not seen:

| Settings | F1, 19–36 | 7-fiber bundles (8) | 19-fiber bundles (10) |
| --- | --- | --- | --- |
| Starting settings | 0.8357 | 0.8598 | 0.8165 |
| Through step 5 | 0.9049 | 0.9148 | 0.8969 |
| Current | **0.9114** | **0.9227** | **0.9024** |

The gain on the unseen structures (+0.076) is close to the gain on the
tuning structures (+0.075).

The fits are deterministic: a second run of the current settings on
structures 1–18 gave the same F1 on every structure.

Per structure, with the starting and the current settings:

| Structure | Bundle | Photons | Fibers (fine + coarse) | Starting F1 | Current F1 |
| --- | --- | --- | --- | --- | --- |
| `dense_hard_1` | 19 | 1137 | 111 + 6 | 0.796 | 0.900 |
| `dense_hard_2` | 19 | 1079 | 107 + 7 | 0.843 | 0.882 |
| `dense_hard_3` | 7 | 1544 | 106 + 5 | 0.885 | 0.895 |
| `dense_hard_4` | 7 | 1254 | 108 + 5 | 0.874 | 0.926 |
| `dense_hard_5` | 19 | 1205 | 128 + 6 | 0.657 | 0.834 |
| `dense_hard_6` | 7 | 1118 | 118 + 7 | 0.906 | 0.954 |
| `dense_hard_7` | 7 | 1193 | 131 + 7 | 0.893 | 0.931 |
| `dense_hard_8` | 7 | 1469 | 131 + 6 | 0.869 | 0.933 |
| `dense_hard_9` | 7 | 1229 | 119 + 6 | 0.905 | 0.944 |
| `dense_hard_10` | 19 | 1487 | 114 + 6 | 0.815 | 0.889 |
| `dense_hard_11` | 19 | 1540 | 107 + 6 | 0.713 | 0.880 |
| `dense_hard_12` | 7 | 1213 | 124 + 5 | 0.892 | 0.931 |
| `dense_hard_13` | 19 | 1126 | 114 + 5 | 0.810 | 0.906 |
| `dense_hard_14` | 19 | 1135 | 103 + 6 | 0.817 | 0.924 |
| `dense_hard_15` | 7 | 1375 | 120 + 5 | 0.905 | 0.949 |
| `dense_hard_16` | 7 | 1043 | 108 + 5 | 0.904 | 0.951 |
| `dense_hard_17` | 19 | 1419 | 109 + 8 | 0.785 | 0.923 |
| `dense_hard_18` | 19 | 1224 | 114 + 8 | 0.849 | 0.919 |
| `dense_hard_19` | 19 | 1123 | 100 + 7 | 0.826 | 0.914 |
| `dense_hard_20` | 19 | 1546 | 107 + 7 | 0.830 | 0.912 |
| `dense_hard_21` | 19 | 1143 | 92 + 5 | 0.806 | 0.879 |
| `dense_hard_22` | 19 | 1529 | 82 + 5 | 0.838 | 0.942 |
| `dense_hard_23` | 19 | 1305 | 133 + 7 | 0.834 | 0.934 |
| `dense_hard_24` | 7 | 1193 | 110 + 6 | 0.869 | 0.935 |
| `dense_hard_25` | 19 | 1512 | 127 + 7 | 0.841 | 0.912 |
| `dense_hard_26` | 7 | 1504 | 131 + 7 | 0.853 | 0.935 |
| `dense_hard_27` | 7 | 1416 | 109 + 6 | 0.828 | 0.909 |
| `dense_hard_28` | 19 | 1338 | 95 + 7 | 0.768 | 0.869 |
| `dense_hard_29` | 19 | 1378 | 114 + 7 | 0.879 | 0.923 |
| `dense_hard_30` | 19 | 1209 | 95 + 5 | 0.756 | 0.839 |
| `dense_hard_31` | 19 | 1352 | 114 + 5 | 0.786 | 0.900 |
| `dense_hard_32` | 7 | 1349 | 124 + 6 | 0.824 | 0.896 |
| `dense_hard_33` | 7 | 1166 | 101 + 6 | 0.848 | 0.917 |
| `dense_hard_34` | 7 | 1383 | 112 + 7 | 0.904 | 0.928 |
| `dense_hard_35` | 7 | 1230 | 124 + 7 | 0.843 | 0.925 |
| `dense_hard_36` | 7 | 1217 | 119 + 6 | 0.910 | 0.936 |

Every structure went up. The 19-fiber bundles gained the most: +0.108 on
1–18 against +0.042 for the 7-fiber bundles.

## What each step changed

**Starting settings.** The fine type traces and relaxes on the graded
grey, which keeps the dip between touching fibers that the flat range
image loses (`graded_image`, `graded_smallest_only`), with extra seeds on
strong tubes in the grey (`bright_seed_strength`). A coarse trace whose
axis reads as bright as fine fibers is dropped as a bundle, and a coarse
fit with a bright axis is typed fine (`coarse_trace_axis_check`,
`coarse_axis_grey_max`). `grey_checked_traces` was on in every run but has
no effect here: its tests compare one grey range per type, and `--input
broad` gives both types the same range. No fit's radius exceeds its type's
(`radius_cap_prior`). New fibers solve with the old ones pinned, and at
least half their nodes must lie on a grey ridge (`births_solve_pinned`,
`birth_ridge_min`). The length prior's joins reach 4 radii instead of 16
(`prior_merge_gap_radii`). Fits are recentered on the grey after every
solve outside the redraw passes and once more at the end. A last round of
new fibers follows, and the ridge finish cleans up the fine fits. The
redraw passes are on (5, the default).

**1. Sharper ridge-finish grey** (`ridge_sigma_voxels=1.2`,
`ridge_min_length_scale=0`, `ridge_refine`). The ridge finish read the
grey at σ = 1.6 voxels. At that blur a dim fiber beside a brighter one has
no ridge of its own, so the off-ridge cut removed correct fits along their
whole length. At 1.2 each fiber keeps its own ridge. Pieces shorter than
the minimum length are now kept, and a short-reach recenter settles grown
ends and bridges. All 18 structures went up.

**2. Hole births** (`hole_births`, `ridge_rim_sigma=1.2`,
`hole_birth_ridge_min=0.3`, `hole_birth_grey_min=0.65`,
`hole_birth_passes=2`). A dim fiber inside a bundle is never traced: on the
graded grey, divided by its bright neighbours', its axis reads below the
tracer's threshold. Once its neighbours are fitted, it is a fiber-wide
stretch of foreground that no fit claims. After the ridge finish, fibers
are traced in these holes, in up to two passes. A hole trace is kept when
at least 0.3 of its nodes lie on the grey ridge. A node whose brightest
point sits on the rim of its disc, toward a brighter neighbour, still
counts when the grey curves down across the fit. Stretches where the grey
falls below 0.65 of the fine fits' median are cut. All 18 structures went
up.

**3. Disc typing of coarse fits** (`coarse_disc_max=0.8`,
`coarse_disc_classify`; `coarse_disc_trace` and `coarse_disc_end` off). In
a packed bundle a fine fit's foreground depth is the whole bundle's, so
the typing after each solver batch turned fine fits between a bundle's
fibers coarse. Their axes lie on the dark gaps, so the axis grey test
misses them. Now a fit whose disc, 6.9 voxels in radius (half the coarse
thickness), holds fine-bright grey is typed fine: the median over its nodes
of the disc's brightest grey lies above 0.8 of the fine range. The same
test on new coarse traces and at the end of the fit stays off: it also
removed a bright coarse fiber, and the extra fine traces it made room for
crowded the coarse fibers' rims, which cost the 7-fiber bundles (0.9215
with the test in all three places, 0.9245 with the typing alone). Against
step 2, the 19-fiber bundles gained 0.030 and the 7-fiber bundles lost
0.005.

**4. Dim-axis typing and rim drop** (`coarse_axis_grey_min=0.0`,
`ridge_rim_share=0.7`).

- A coarse fiber's dim core leaves holes in the foreground, so its depth
  reads thin and a fit on it can be typed fine. Such a fit is now typed by
  its width when all of these hold: its median axis grey is below the fine
  range, no fine-bright grey lies within 6.9 voxels (half the coarse
  thickness) of its axis (the disc's 90th percentile under 0.3 of the
  range), its depth is at least 4 voxels and it has at least 10 nodes.
- The ridge finish drops a fine piece that runs along a coarse fiber's
  bright rim, just outside the coarse fit. The test: at least 0.7 of the
  piece lies within 1.35 of the coarse fit's oval section, and its median
  grey is at most 0.58 of the way from void to the fine fits' median grey.

Added to step 2 alone, this raised the mean to 0.8936 (+0.0033). On top of
step 3 the mean fell by 0.0015: the 7-fiber bundles gained 0.0056 and the
19-fiber bundles lost 0.0086, over half of it on `dense_hard_1` (0.882 to
0.834).

**5. No redraw passes** (`redraw_passes=0`). With the steps above, the
redraw passes lowered the mean: 0.9015 with 5 passes, 0.9038 with 3 and
0.9065 with none. They are off, which also shortens each fit.

**6. Coarse births, hole rim test, straight join, bend finish.** Coarse
births and the hole rim test took the mean to 0.9159 (from an exact replay
of the last stages, as for the loss table under Caveats), and the last two
passes to 0.9150.

- **Coarse births** (`birth_ridge_smallest_only`,
  `coarse_birth_disc_max=0.8`). A dim-cored coarse fiber is brighter on its
  rims than on its axis, so a coarse trace on it is off the grey ridge
  along its whole length, and the ridge test on new fibers dropped every
  coarse birth. A coarse fiber lost in a round was never traced again;
  fine fits filled its rims instead. The ridge test now applies to the
  fine type only. A new coarse trace is dropped instead when its disc holds
  fine-bright grey (the step 3 test, at 0.8), which marks a coarse trace
  laid over a bundle. On the step 5 settings this alone added 0.0080, and
  the coarse fibers left almost untraced as coarse (under a fifth of their
  length) fell from 19 to 6. It costs some coarse fits laid over fine
  bundles, and merged fits rose from 86 to 93. The largest losses were on
  three structures with 19-fiber bundles (`dense_hard_2`, `5` and `10`).
- **Hole rim test** (`hole_birth_rim_drop`). Beside a coarse fit that the
  solver has pushed off its axis, the hole births can trace the rim lobe
  just outside the fit. The rim test of step 4 now drops those hole pieces
  too. On top of the coarse births it added 0.0013 and lowered no
  structure.
- **Straight join** (`straight_join`). A last pass joins fine pieces that
  continue each other in a straight line: ends antiparallel within 15°,
  each piece's extended axis within one radius of the other's end, and
  either a gap of at most 30 voxels whose bridge reads at least 0.5 in the
  fit image, or an overlap of at most 20 voxels.
- **Bend finish** (`bend_finish`, `bend_finish_grey_min=0.4`). The steps
  after the last solve (ridge finish, hole births, joins) do not enforce
  the bend limit. The bend finish smooths every fine fit until it keeps to
  its type's limit, then cuts runs where the smoothed fit reads below 0.4
  of the way from void to the fits' median grey. A trace that wandered
  along noise has no fiber under its smoothed path.

The straight join and the bend finish were first measured together by
re-running only the last stages on the step 2 fits. There the two passes
added 0.0039 and cut the fibers split into several fits from 546 to 343
over the 18 structures. On the current settings they leave the mean about
where it was: 0.9159 without them (the same exact replay as the loss table
below) against 0.9150 with them, 8 structures up and 10 down. They are
kept because they cut split fibers from 570 to 383 and fits from 3740 to
3231.

## Caveats

**The fibers end inside the scan.** Each structure is a whole cell, not a
crop of a larger one. The true fine fibers are 71–145 µm long (median
106 µm) in a 160 µm scan, so almost every fiber has both ends inside it:
1.88 interior ends per true fiber over structures 1–18. Several settings
work on ends: the hole births, the ridge finish's end growth and the
joins. In a scan whose fibers run through the whole block, ends are rare
and these settings weigh differently. `FitSettings.through_block` is meant
for that case; it is not part of the recommended settings and was not
tuned here.

**The fibers bend a lot.** With a waviness of 0.15–0.6 diameters and a
bend limit of 5 diameters, the true fine fibers have a median radius of
curvature of 10.5 diameters, measured at their nodes. Each fiber's
tightest bend has a median of 6.7 diameters, close to the limit.
Straighter fibers were not part of the tuning set.

**Where the remaining F1 is lost.** The loss can be split exactly into
classes, per structure. These are the largest, as mean F1 points lost over
structures 1–18. They were measured with the current settings less the
straight join and the bend finish (mean F1 0.9159):

| Class | What it is | F1 lost |
| --- | --- | --- |
| Off axis | a fine fit on its own fiber, but 0.5–1 radius off the axis | 0.026 |
| End runs | a fiber's ends that its fit stops short of | 0.014 |
| Coarse rim | fine fits lying on a coarse fiber, mostly along its bright rim | 0.010 |
| Hops | a fine fit that leaves its fiber and runs 6 voxels or more along a neighbour | 0.004 |
| Coarse untraced | coarse fiber length that no fit of its own traces | 0.004 |
| Whole misses | fine fibers that no fit traces at all | 0.002 |
| Coarse off axis | a coarse fit next to its fiber but off the axis | 0.002 |
| Coarse on fine | a coarse fit laid over fine fibers | 0.001 |

Together these make 0.063 of the 0.084 lost; the rest is spread over
smaller classes. The straight join and the bend finish together cut the
hop length by about a third on the step 2 fits (the straight join alone
raised it; the bend finish cut it), so the hop row probably overstates the
current loss; it was not measured on the current settings.

Much of the off-axis loss comes from the simulated scan, not from the fit.
The truth is each fiber's rest position, but the scanner moves every fiber
by 1 µm (RMS) during the scan, so the grey peaks where the fiber spent the
scan. In an audit of earlier settings, 61% of the off-axis length sat on
grey maxima made by the motion. No fit that follows the grey can recover
that part.

**Spread between structures.** With the current settings, F1 on 1–18 runs
from 0.834 (`dense_hard_5`) to 0.954 (`dense_hard_6`), a standard
deviation of 0.031. The 19-fiber bundles score lower. In audits of
earlier settings, `dense_hard_5` missed more fine fiber than its
composition explains, and the cause is still open.

**Cascades.** A fit is deterministic, but a change anywhere shifts which
fibers are traced first, and the difference cascades through the rest of
the fit. Changes that should not touch a structure move single structures
by 0.01–0.02, and `dense_hard_1` by 0.02–0.05. Its F1 through the steps
above was 0.796, 0.833, 0.852, 0.882, 0.834, 0.862 and 0.900. Judge a
change on the mean over all 18 structures, with the two bundle sizes
reported apart, and never on one structure. The paired standard error of
a change in the mean is about 0.004 (step 6's coarse births and hole rim
test: +0.0093 ± 0.0037).

**Settings in voxels.** Three blurs in the recommended settings are in
voxels, not radii: `ridge_sigma_voxels`, `ridge_rim_sigma` and
`recenter_sigma_voxels`, tuned to a fine radius of 3.25 voxels. So are
several defaults the recommended settings rely on: the straight join's
longest gap and overlap (30 and 20 voxels), `coarse_axis_depth_min`
(4 voxels), `birth_ridge_sigma_voxels` (1.6) and `denoise_sigma_voxels`
(0.7). A scan with fibers of another size in voxels may need them
scaled. The hole births trust the foreground, so any foreground that is
not fiber becomes a hole to trace.
Keeping short pieces (`ridge_min_length_scale=0`) and the hole births both
add fragments: the current settings fit 3231 fibers for 2181 true ones on
structures 1–18 (1.48 per true fiber, against 1.30 with the starting
settings).
