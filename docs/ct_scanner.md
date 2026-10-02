# Simulated CT scanner

`ct.Scanner` renders a Tangle structure as a CT scan by simulating the
acquisition: x-ray projections of the sample, free-space propagation (which
gives phase-contrast fringes at every surface), detector blur, photon
counting and filtered back-projection. The blur, the noise texture and the
edge fringes then come out of the same steps a parallel-beam CT scanner
goes through, instead of being added to the image one by one. Every setting
is a physical one; none is fitted.

For the whole process in pictures (geometry, the object, each acquisition
stage, the settings side by side) see [ct_synthetic.md](ct_synthetic.md).

The true fibers are known, so a scan made this way tests the fitter
([ct_fitting.md](ct_fitting.md)) against ground truth. The scanner is used
through `ct.synthetic_ct`, which also draws the fibers: several fiber
types, each with its own brightness, a dim core inside a bright rim, oval
cross-sections ([oval_fibers.md](oval_fibers.md)) and a brightness that
differs from fiber to fiber. The code is in
[`_scanner.py`](../crates/tangle_python/python/tangle/ct/_scanner.py) and
[`_synthetic.py`](../crates/tangle_python/python/tangle/ct/_synthetic.py).

```python
import tangle
import tangle.ct as ct
from tangle.units import um

# A 64 µm box: three straight fine fibers and one flat oval coarse fiber.
cell = tangle.Cell([64 * um] * 3)
fine = tangle.Material("fine_6.5um", diameter=6.5 * um, min_bend_radius=32 * um)
coarse = tangle.Material("coarse_17um", diameter=17 * um, thickness=13.6 * um, min_bend_radius=85 * um)
assembly = tangle.Assembly(cell)
assembly.insert(tangle.FiberCollection.from_centerlines([
    [[4 * um, 20 * um, 20 * um], [60 * um, 22 * um, 20 * um]],
    [[4 * um, 27 * um, 20 * um], [60 * um, 29 * um, 20 * um]],
    [[20 * um, 4 * um, 44 * um], [22 * um, 60 * um, 44 * um]],
], fine), name="fine")
flat = tangle.FiberCollection("coarse")
flat.add_fiber([[4 * um, 44 * um, 30 * um], [60 * um, 44 * um, 30 * um]], coarse)  # long axis lies flat
assembly.insert(flat, name="coarse")

scanner = ct.Scanner(
    photons=1300,             # per detector pixel and projection, unattenuated
    fiber_attenuation=0.005,  # per voxel, for a fiber of brightness 1
    resolution=2 * um,        # full width at half maximum of the blur
    delta_beta=(12.0, 9.0),   # one per profile below
    propagation=4.0,          # Fresnel distance, pixels²
    fiber_motion=1 * um,      # RMS, each fiber on its own path
)
profiles = [
    (6.5 * um, ct.CrossSection()),                                     # solid, brightness 1
    (17 * um, ct.CrossSection(brightness=0.6, rim=3 * um, core=0.6)),  # dim, with a dimmer core
]
scan = ct.synthetic_ct(assembly, 1 * um, seed=1, profiles=profiles, scanner=scanner, brightness_spread=0.25)

scan.volume       # (z, y, x) uint16
scan.labels       # (z, y, x) true fiber id of every voxel, 0 = void
scan.types        # each true fiber's profile: [0 0 0 1]
scan.semi_axes    # long and short semi-axes of every true fiber (voxels); None when all are round
ct.save_overlay_figure("scan.png", scan.volume, scan.labels)  # three slices: scan above, labels tinted below (needs matplotlib)
```

`scan.centerlines` are the true centerlines in voxel units, `(x, y, z)`,
at the fibers' rest positions. `Scanner` is a frozen dataclass: vary one
setting with `dataclasses.replace(scanner, photons=5000)`.

## What it simulates

Arrays are `(z, y, x)`. The sample turns about the z axis, and one voxel is
one detector pixel (no magnification). `synthetic_ct(source, voxel_size,
scanner=...)` goes through these steps:

1. **Occupancy.** Tangle's PuMA export draws every fiber, round or oval,
   as a partial-volume occupancy from 0 to 1 per voxel, and gives each
   voxel's fiber id. The ids are the true labels.
2. **Brightness per fiber type** (`profiles`). Each fiber takes the profile
   whose diameter is nearest its own (for an oval, its long width), and its
   occupancy is multiplied by the profile's brightness. With a rim, voxels
   closer to the axis than the radius less the rim get `core` times that
   brightness. Without profiles every fiber has brightness 1.
3. **Brightness per fiber** (`brightness_spread`). Each fiber's brightness
   is multiplied by its own log-normal factor with median 1. At 0.25, 90%
   of the factors lie between 0.66 and 1.51. The factors come from their
   own random stream, so the same seed gives the same factors whatever the
   scanner settings. The noise pattern still changes when the spread is
   turned on, because the photon counts change.
4. **Attenuation and phase.** The sample's attenuation per voxel is
   `void_attenuation + (fiber_attenuation − void_attenuation) × level`,
   where `level` is the occupancy times the brightness. With a single
   `delta_beta`, the phase follows the whole attenuation. With one ratio
   per profile, each fiber's phase follows its own attenuation times its
   type's ratio, and the void shifts no phase.
5. **Motion** (`fiber_motion`, `drift`; off by default). The angles are
   split into `motion_steps` equal stretches, and each stretch sees the
   sample where it stood then. Each fiber follows its own smooth random
   path in 3D (a random walk over the stretches, its mean removed), scaled
   so the RMS displacement over all fibers and stretches is `fiber_motion`.
   `drift` moves the whole sample along one such path. Void voxels move
   with their nearest fiber, so partial-volume edges move along. A moving
   fiber comes out blurred, a still one sharp. The truth is the rest
   position, the mean of each path.
6. **Projection.** Parallel-beam line integrals at `angles` angles evenly
   spaced over 180°, one detector row per z slice (Joseph's method). The
   row spans the slice's diagonal plus 4 pixels, so every ray through the
   sample is recorded.
   With `beam_hardening` (off by default), the line integral `p` is read
   as `p / (1 + beam_hardening × p)`: a polychromatic beam hardens on its
   way through the sample, so long and dense paths absorb less than their
   attenuation says. It reconstructs with a darker middle (cupping) and
   dark streaks between dense objects.
7. **Propagation** (`delta_beta`, `propagation`). Each projection becomes
   the exit wave of the sample: amplitude `exp(−p/2)` and phase
   `−delta_beta × p/2`, where `p` is the projected attenuation. The wave
   travels the Fresnel distance `propagation` in free space (a Fresnel
   propagator over the detector's z and u axes), and the detector records
   its intensity. This makes the fringe of propagation-based phase
   contrast: bright just inside every surface, dark just outside, strongest
   where two surfaces face each other. At `propagation=0`, or with a
   single `delta_beta` of 0, the projection is plain absorption, `exp(−p)`.
8. **Detector blur** (`detector_blur` or `resolution`). A Gaussian blur of
   the intensity along both detector axes, before the photons are counted.
9. **Counting** (`photons`, `ring_strength`). Each detector pixel has a
   gain `1 + ring_strength × N(0, 1)`, the same at every angle. The counts
   are Poisson with mean `intensity × gain × photons`.
10. **Noise blur** (`noise_blur`). A Gaussian blur of the counts, as a
    scintillator spreads each absorbed photon's light over its neighbours.
    It blurs the noise too, so the noise comes out blotchy.
11. **Flat field.** `−ln(counts / photons)`, with an ideal, noise-free flat
    and zero counts read as half a photon. The gain error is not corrected,
    so it reconstructs as rings.
12. **Reconstruction.** Filtered back-projection, slice by slice, with a
    Shepp–Logan filter. With `slice_drift` (off by default), each slice's
    gain and offset then wander along z, a smooth random walk with, half the
    time, one step (as where a scan is stitched from two heights).
13. **Grey levels.** The reconstruction is divided by `fiber_attenuation`,
    so a solid fiber of brightness 1 reads 1 and the void
    `void_attenuation / fiber_attenuation`. `synthetic_ct`'s `drift` (a
    grey level, not the scanner's motion) then adds a low dome: 0 at the
    side walls, `drift` in the middle of each slice, 0.05 by default.
    Last, the 0.5th and 99.5th percentiles are mapped to 0 and 65535 and
    the volume is stored as `uint16`.

## Settings

### `Scanner`

| Field | Unit | Default | Effect |
| --- | --- | --- | --- |
| `photons` | photons per detector pixel and projection, unattenuated | 5000 | Photon noise. Four times the photons halve the noise. |
| `angles` | projections over 180° | the detector row's width: the slice diagonal, rounded up, + 4 pixels (231 for a 160-voxel slice) | More angles, less noise per voxel (see below). |
| `fiber_attenuation` | per voxel, for a fiber of brightness 1 | 0.02 | The fibers' absorption. Contrast to noise grows with it; the phase is proportional to it. |
| `void_attenuation` | per voxel | 0.0005 | The background's absorption. The void reads `void_attenuation / fiber_attenuation` of a fiber. |
| `delta_beta` | ratio of refractive index decrement to absorption index | 0 | The phase is `delta_beta` times half the projected attenuation. One ratio, or one per profile (`profiles` needed, same length). A denser, more absorbing material has a lower ratio and shows less fringe. A single 0: no phase. |
| `propagation` | pixels²: wavelength × sample-detector distance / pixel² | 0 | Fresnel distance. Longer makes the fringes deeper and wider. 0: absorption only. |
| `detector_blur` | pixels, Gaussian σ | 0.6 | Blur before counting (source size, scintillator). Blurs the image, not the noise. |
| `resolution` | meters, full width at half maximum | None | Replaces `detector_blur` with σ = resolution / (2.355 × voxel size), so the blur keeps its physical size at any voxel size. Needs the voxel size (given by `synthetic_ct`). |
| `ring_strength` | relative, σ of each detector pixel's gain | 0 | Rings about the rotation axis. |
| `noise_blur` | pixels, Gaussian σ | 0 | Blur after counting: blurs the noise as well, so it comes out correlated and lower. |
| `fiber_motion` | meters, RMS | 0 | Each fiber moves on its own path during the scan. |
| `drift` | meters, RMS | 0 | The whole sample moves on one path during the scan. |
| `motion_steps` | stretches of angles | 8 | How finely the motion is resolved. Used only with motion. |
| `beam_hardening` | per unit of projected attenuation | 0 | Cupping and streaks between dense objects (step 6). |
| `slice_drift` | relative gain, RMS (the offset in fiber attenuations) | 0 | Slices brighter or darker than their neighbours (step 12). |

A fiber's projected phase should stay within a few radians (`delta_beta`
times half its attenuation across it). Far beyond that the fringes ring
instead of edging each surface.

### `synthetic_ct` with a scanner

| Argument | Unit | Default | With a scanner |
| --- | --- | --- | --- |
| `voxel_size` | meters | required | The voxel and detector pixel size. |
| `seed` | | 0 | Seeds the noise, the motion and (in a separate stream) the brightness factors. The same seed and structure give the same volume. |
| `profiles` | list of `(diameter in m, CrossSection)` | None | One brightness profile per fiber type (step 2). `types` in the result holds each fiber's profile index. |
| `brightness_spread` | σ of the log-normal factor | 0 | Each fiber's own brightness (step 3). Needs `profiles`. The phase follows the same factor. |
| `drift` | grey, where a fiber of brightness 1 reads 1 | 0.05 | The grey dome of step 13. Not the same as `Scanner.drift`. |
| `psf_sigma_voxels`, `noise`, `noise_correlation`, `phase_contrast`, `phase_sigma_voxels`, `void_level` | | | Ignored: the scanner makes the blur, the noise and the fringes. |

### `CrossSection`

| Field | Unit | Default | Meaning |
| --- | --- | --- | --- |
| `brightness` | 0 (void) to 1 (the brightest type) | 1 | The fiber's brightness, or its rim's if it has one. |
| `rim` | meters | None | Rim thickness; None for a solid fiber. |
| `core` | fraction of the rim brightness | 1 | Brightness inside the rim. |

## What the settings do

The numbers below show the size of the effects. They are illustrations
from small test volumes, not guarantees. Grey is in **fiber-contrast
units**: the void is 0 and a solid fiber of brightness 1, seen by
absorption alone, is 1 (before the drift dome and the `uint16` scaling).

**Fringes and blur.** One straight 6.5 µm fiber (radius 3.25 voxels) along
z at 1 µm voxels, `fiber_attenuation` 0.005, without noise (10¹² photons),
averaged around the axis:

| Settings | On the axis | Brightest inside | Darkest outside | Where (voxels from the axis) |
| --- | --- | --- | --- | --- |
| absorption, `detector_blur=0` | 0.99 | 0.99 | none | |
| absorption, `resolution` 2 µm | 0.97 | 0.97 | none | |
| `delta_beta` 12, `propagation` 4, `detector_blur=0` | 0.83 | 2.52 | −0.88 | 4.25–4.75 |
| `delta_beta` 12, `propagation` 4, `resolution` 2 µm | 1.54 | 1.85 | −0.38 | 4.75–5.25 |
| `delta_beta` 6, `propagation` 4, `resolution` 2 µm | 1.27 | 1.37 | −0.17 | 4.75–5.25 |
| `delta_beta` 24, `propagation` 4, `resolution` 2 µm | 2.03 | 2.77 | −0.84 | 4.25–4.75 |
| `delta_beta` 12, `propagation` 1, `resolution` 2 µm | 1.10 | 1.13 | −0.07 | 4.75–5.25 |
| `delta_beta` 12, `propagation` 16, `resolution` 2 µm | 5.66 | 5.66 | −1.10 | 5.25–5.75 |

With the settings the examples use (the fourth row), a thin fiber reads
well above 1 inside and has a dark band down to −0.38 one to two voxels
outside its surface, gone about 7 voxels from the axis. A 17.3 µm solid
fiber (radius 8.65 voxels) with the same settings reads 0.99 on its axis,
up to 1.58 just inside its surface and −0.30 just outside: the fringe
edges a thick fiber but fills a thin one. A fiber's grey therefore
depends on its size and its neighbours as well as its material, and the
brightest 0.5% of voxels are clipped in the `uint16` volume.

**Noise.** The void alone, read in the middle of each slice, with
`fiber_attenuation` 0.005, `resolution` 2 µm, `delta_beta` 12 and
`propagation` 4, on a 160-voxel slice (231 angles) unless noted:

| Settings | Noise (standard deviation) | Correlation with the next voxel, in x / in z |
| --- | --- | --- |
| `photons` 1300 | 0.213 | 0.21 / 0.00 |
| `photons` 5200 | 0.106 | 0.20 / 0.01 |
| `fiber_attenuation` 0.01 | 0.101 | 0.21 / 0.00 |
| `photons` 1300, no phase, `detector_blur` 0.6 | 0.213 | 0.21 / 0.01 |
| `photons` 1300, no phase, `detector_blur=0` | 0.211 | 0.20 / 0.00 |
| `photons` 1300, 64-voxel slice (95 angles) | 0.321 | 0.21 / 0.00 |
| `photons` 1300, 64-voxel slice, `angles=231` | 0.209 | 0.19 / 0.03 |
| `photons` 226, `noise_blur` 0.5 | 0.295 | 0.33 / 0.27 |
| `photons` 1300, `noise_blur` 1 | 0.036 | 0.73 / 0.78 |
| `photons` 56, `noise_blur` 1 | 0.173 | 0.73 / 0.78 |
| `photons` 37, `noise_blur` 1 | 0.212 | 0.73 / 0.78 |
| `Scanner()`, every field default (5000 photons, `fiber_attenuation` 0.02, `detector_blur` 0.6, no phase) | 0.025 | 0.20 / 0.01 |

- Noise falls as the square root of the photons, and in these units as
  the fiber attenuation grows.
- Blur before counting (`detector_blur`, `resolution`) leaves the noise
  as it is. Blur after counting (`noise_blur`) lowers it and spreads it
  over neighbouring voxels, within a slice and between slices. With it,
  far fewer photons give the same noise: 37 photons at a noise blur of 1
  give 0.212, as 1300 without it do (0.213). The 56 photons that
  `bundled_two_types` uses give 0.173.
- The default number of angles grows with the slice, so a smaller box is
  noisier per voxel at the same `photons`. To keep the noise of a larger
  scan, render the larger box and crop it (`SyntheticScan.crop`), or fix
  `angles`.

## What the examples add

[`ct_examples.py`](../crates/tangle_python/python/examples/ct_examples.py)
defines one scanner, `SCANNER`, which every scanner example starts from, and
`SCANNER_PROFILES`, the profiles the three `*_two_types` scanner examples
use with it:

| `SCANNER` field | Value |
| --- | --- |
| `photons` | 1300 |
| `fiber_attenuation` | 0.005 |
| `resolution` | 2 µm |
| `delta_beta` | (12, 9): fine, coarse |
| `propagation` | 4 |
| `fiber_motion` | 1 µm |
| the rest | defaults |

`SCANNER_PROFILES` are a solid 7 µm type of brightness 1 and a solid 19 µm
type of brightness 0.55: the coarse fibers are a lighter material, so the
fine fibers are the brightest thing in the scan, each with a dark band
just outside it, and the coarse ones dim with a faint rim.

- `scanner_two_types`: `two_types`' fibers (7 and 19 µm) at 1.25 µm
  voxels, scanned by `SCANNER`. The 320 µm cell, periodic in x and y, is
  rendered whole and cropped to its central 200 µm, so fibers run through
  the scan's boundary.
- `bundled_two_types`: the fine fibers in bundles of nineteen, scanned
  with `replace(SCANNER, noise_blur=1.0, photons=56)`, so the noise is
  blotchy.
- `oval_two_types`: flat oval coarse fibers, 19 by 12 µm, scanned by
  `SCANNER`.

### Scanned structures

`scanned(index, settings)` builds a structure and a scanner from one
settings dictionary, drawn from the example's seed:

- a closed box 160 voxels a side, rendered whole;
- a fine fiber type, packed in hexagonal bundles when `per_bundle` is 7 or
  19 (a center fiber, a ring of six, then a ring of twelve, their ends
  staggered; members that would leave the box through its top or bottom,
  or bend past the limit, are left out), and optionally a coarse type,
  loose, and oval when its
  thickness ratio is below 1;
- the number of fibers from each type's solid fraction, the bend limit at
  5 diameters and a curvature amplitude of 0.15–0.6 diameters;
- a solid fine profile of brightness 1, and a coarse profile of the
  drawn brightness, with a rim and core where the settings give them;
- `replace(SCANNER, ...)` with the drawn photons, noise blur, resolution,
  fiber motion and `delta_beta`.

There are two families of settings. `scanned_*` spans a wide range of
structures and scanners. `dense_hard_*` holds one kind of structure,
packed bundles of fine fibers with flat oval coarse fibers at 1 µm voxels,
each fiber with its own brightness and the coarse fibers dim with a
dimmer core, so no grey threshold separates the types:

| Setting | `scanned_*` | `dense_hard_*` |
| --- | --- | --- |
| seed | 5000 + n | 7000 + n |
| voxel | 1.0–1.5 µm | 1.0 µm |
| orientation | planar (1 in 2), aligned or biaxial (1 in 4 each) | planar |
| tilt, or cone half-angle when aligned (radians) | 0.15–0.45 | 0.15–0.3 |
| fine diameter | 5–9 µm | 6.5 µm |
| fine solid fraction | 0.04–0.12 | 0.10–0.13 |
| fibers per bundle | 1, 7 or 19 | 7 or 19 |
| coarse type | 14–22 µm, with probability 0.7 | 17.3 µm wide, 0.8 as thick |
| coarse oval | `scanned_201` … `208`: 0.55–0.8 as thick as wide | always |
| coarse solid fraction | 0.02–0.06 | 0.03–0.05 |
| coarse profile | solid, brightness 0.4–0.7 | brightness 0.6, 3 µm rim, core 0.6 |
| `brightness_spread` | none | 0.25 |
| fiber length | from 0.5–0.7 up to 0.75–0.95 of the box side | from 0.55 up to 0.9 of the box side |
| `photons` | 1300 × 0.6–1.6, divided by 23 × `noise_blur`² when that is above 1 | 1300 × 0.8–1.2 |
| `noise_blur` | 0, 0.5 or 1 pixel | 0 |
| `resolution` | 1.5–3.0 µm | 2 µm |
| `fiber_motion` | 0–1.5 µm | 1 µm |
| `delta_beta` (fine, coarse) | 6–16, 5–12 | 12, 9 |

Each example's `score.json` records its settings. The photons fall with
the noise blur, as in `bundled_two_types`. This keeps the void noise of
the same order but does not match it: with the random factor of 0.6–1.6
at 1, the
noise is 0.213 without noise blur (1300 photons), 0.295 at 0.5 (226
photons) and 0.167 at 1 (57 photons).

From `crates/tangle_python/python/examples/`:

```
python ct_examples.py --scanned tune     # scanned_1 … scanned_16
python ct_examples.py --scanned check    # scanned_101 … scanned_108
python ct_examples.py --scanned oval     # scanned_201 … scanned_208
python ct_examples.py dense_hard_1 dense_hard_7
DENSE_HARD_COUNT=36 python ct_examples.py dense_hard_19 dense_hard_20
```

`dense_hard_*` run only when named, never by default or with `--scanned`.
`dense_hard_1` … `dense_hard_18` exist by default; `DENSE_HARD_COUNT=N`
registers `dense_hard_1` … `dense_hard_N`, so structures beyond 18 are
drawn the same way from their own seeds, for checking on structures the
fitter was not tuned on.

### Caching

The slow step is relaxing the structure (Tangle's solver, on the GPU by
default), not scanning it. The examples relax each structure once and keep
its centerlines (and long axes, for ovals) in `<output>/.cache/`. The
scanned examples name that file by a hash of the whole settings
dictionary, so any change, a scanner setting such as `photons` included,
gets a new file and a new relaxation. The scan itself is not cached: each
run renders it again from the cached centerlines, and the same centerlines
and seed give the same volume. To reproduce a scan exactly, keep the
output folder's `.cache`; a fresh folder relaxes the structures again.

Building a `dense_hard_*` scan from its cached structure (a 160-voxel box)
took about 6–20 s on a laptop CPU, depending on its load, with a peak
resident memory of about 1–1.5 GB; `scanner_two_types` (a 256-voxel
render, then cropped) took roughly 40–100 s and 2–3 GB.

`ct_examples.py --blur` sets `psf_sigma_voxels`, which a scanner ignores,
so it does not change any scanner example.

## Limits

- **Geometry.** Parallel beam, rotation about z, unit magnification. No
  cone beam and no field-of-view truncation: the sample ends at the
  volume's box, which every projection sees whole. Because the void is a
  block the size of the volume, the outermost two or three voxels of each
  slice carry a faint edge of their own (up to 0.06 in fiber-contrast
  units).
- **Physics.** One energy: no beam hardening and no scatter. The phase
  follows the projection approximation and Fresnel propagation of the exit
  wave. There is no phase-retrieval step: the reconstruction is filtered
  back-projection of the log intensity, so the fringes stay in the volume
  as edge contrast. The noise is Poisson counting alone, with an ideal flat
  field; there is no dark current or read noise.
- **Materials.** Two levels: void, and fiber times its brightness. The
  brightness factor of `brightness_spread` is constant along a fiber. The
  dim core of a rim profile is a round disk of the equivalent radius less
  the rim, also in an oval fiber: for a 17.3 × 13.84 µm oval with a 3 µm
  rim, the rim is about 3.9 µm wide at the long ends and 2.2 µm on the
  flat sides.
- **Motion.** Each fiber moves as a whole, without bending or turning, and
  holds still within each stretch of angles. Each voxel moves with the
  fiber nearest it at rest, so fibers never overlap in the scan. A fiber
  that moves into a neighbour is cut off where their regions meet. When
  touching fibers move apart, the gap fills with copies of their edges;
  void stretches only where void separated them at rest. The RMS
  is taken over all fibers together, so with only a few fibers one fiber
  can move much more or less than `fiber_motion`. The motion draws come
  before the noise from the same seed, so turning motion on also changes
  the noise.
- **Grey levels are relative.** Each scan is scaled to `uint16` by its own
  percentiles, with the brightest and darkest 0.5% clipped, and carries the
  `drift` dome unless `drift=0`.
- **Noise depends on the box.** The default number of angles grows with
  the slice (see above).
- **Periodic cells.** Fiber brightness is drawn across periodic walls, but
  the scanner sees the cell as a box, so a fiber that crosses a periodic
  wall is cut at one face and goes on at the opposite one. The examples
  render a larger cell and crop its middle.
- **Cost.** It runs on the CPU. The projections are held in memory as
  float32 `(angles, z, width)` arrays, several at once, and the time grows
  steeply with the box: projection and back-projection each take about
  `angles × voxels` steps.
