# CT map network: fibers from a 3D U-Net

`tangle.ct` fits fibers to the grey scan directly. This is a second route: a
3D U-Net, trained on Tangle's own simulated scans, turns the scan into maps that
make fibers easy to follow, and a short tracer reads the fibers off the maps.
The traced centerlines can then go through one Tangle solve (contacts, bend
limits, known fiber sizes) like any fit.

The code is in
[`crates/tangle_python/python/examples/ct_unet/`](../crates/tangle_python/python/examples/ct_unet/).
It needs PyTorch (Apple GPU, CUDA or CPU), NumPy, SciPy and, for the figure,
matplotlib.

![One simulated scan, the network's maps and the traced fibers](media/ct-unet-maps.png)

*A slice of `dense_hard_7` (never seen in training). From left: the simulated
scan; the axis heatmap; each fiber voxel's distance to its own axis; the fiber
type; the traced centerlines (magenta fine, red coarse) over the true ones.*

## Results on the simulated sets

Centerline F1 (`ct.score`'s centerline agreement: the share of the true axes
traced within half a radius, and of the traced axes lying on their own fiber):

| set | grey fitter (best settings) | map network + tracer |
|---|---|---|
| `dense_hard_1`-`18` (the tuning set) | 0.915 | **0.995** |
| `dense_hard_19`-`36` (fresh structures) | 0.911 | **0.995** |
| `scanned_101`-`108` (other voxel sizes, blur, orientations) | | **0.974** |

The network scans a 160³ volume in about a second and the tracer takes a few
seconds on the CPU. On `scanned_101`-`108` the weakest case (0.88) is a scan
whose fine fibers are under 4 voxels across.

## What the network predicts

For every voxel, 13 numbers:

1. **Axis heatmap:** a peak on every fiber centerline, exp(-d²/2s²) with d the
   distance to the nearest axis and s = max(1, 0.3 r). It is kept sharp on
   purpose: the score only counts a point within half a radius of the axis,
   1.6 voxels for a 6.5-voxel fiber.
2. **Offset to the axis** (3): the vector from the voxel to its own fiber's
   centerline. Two touching fibers push their voxels in opposite directions,
   so the boundary between them is explicit even where the grey shows no gap.
3. **Direction** (6): the fiber's tangent as the sign-free tensor t tᵀ, so a
   fiber pointing +x and one pointing -x read the same.
4. **Type** (3): void, fine, coarse.

The network is a plain 3D U-Net: four 2x poolings (128³ down to 8³, 16 to 256
channels), two 3x3x3 convolutions with group norm per level, skip connections,
and a 1x1 head. It has 5.6 M weights. Training crops are 128³; any volume whose
sides are divisible by 16 can be predicted in one go, and bigger ones in tiles.

## The tracer

[`trace_maps.py`](../crates/tangle_python/python/examples/ct_unet/trace_maps.py):

1. **Votes:** every voxel the network calls fiber votes for its axis at voxel +
   offset. Votes are pooled in 1-voxel cells; a cell with enough votes is an
   axis point, carrying the mean position, the principal direction and the
   majority type.
2. **Tracking:** from the best-supported unused axis point, a track steps 1.5
   voxels at a time both ways along the direction, moving to the mean of the
   axis points ahead whose direction agrees (so a crossing fiber's points are
   ignored), coasting over short gaps and stopping where it runs onto a fiber
   already traced. A track uses up the points around it (1.2 voxels for fine,
   4 for coarse fibers).
3. **Tidy** (optional, `tidy_up`): traces lying on top of a longer one for over
   half their length are dropped, and traces whose ends face each other across
   a short straight gap are joined. This barely changes F1 but brings the
   number of traces close to the number of fibers, which matters for a solve
   afterwards.

Whole scans are traced from overlapping tiles (`tiled_axis_points`: 128³ tiles
every 64 voxels). Each voxel votes from the tile whose center it is nearest, so
it always has at least 32 voxels of context around it, the votes of all tiles
pool into one set of axis points, and the fibers are tracked once over the
whole volume, with no seams to stitch.

## Training data

[`make_data.py`](../crates/tangle_python/python/examples/ct_unet/make_data.py)
renders `ct_examples` structures with the `dense_hard` settings (packed bundles,
flat oval coarse fibers, phase-contrast halos, per-fiber brightness) and, for
variety, the `scanned` settings (voxel size, blur, orientation, fiber sizes), at
indices well past the test sets, so `dense_hard_1`-`36` and `scanned_101`-`108`
are never seen. About 600 volumes of 160³, ~8 s each. Each file stores the scan,
the labels and, per voxel, the nearest true axis point, so the targets are
cheap gathers at load time.

[`train.py`](../crates/tangle_python/python/examples/ct_unet/train.py): 128³
random crops, xy swaps and flips in x, y and z (the fibers lie in planes, so no
general rotations), grey-level jitter, AdamW with a one-cycle schedule, 12,000
steps at batch 1 (about 5.5 hours on an Apple M5 Pro GPU).

**Correlated noise.** CT reconstructions often have noise correlated over about
a voxel, so the void looks blotchy; the simulated scans mostly have near-white
noise. A network trained only on those calls blotches the size of a coarse
fiber "coarse fiber". `--noise 0.25` adds a noise field blurred by 0.6-1.3
voxels to 70% of the crops; fine-tuning the trained network 4,000 steps with it
(`--init`) removes these false fibers, at no cost on the simulated sets.

## Fiber diameters along each fiber (optional)

[`diameters.py`](../crates/tangle_python/python/examples/ct_unet/diameters.py)
measures the diameter at every 1-voxel node of a traced (or solved) fiber: each
fiber voxel is counted for the node its axis vote lands on, so touching fibers
keep their own voxels. Area per unit length gives the equal-area diameter; the
spread of those voxels across the axis gives the long and short widths of an
oval and its long-axis direction. On the simulated sets
([`check_diameters.py`](../crates/tangle_python/python/examples/ct_unet/check_diameters.py))
it reads about 0.4 voxel high for fibers from 5 to 20 voxels across, and the
along-fiber spread of a truly constant fiber reads about 0.3-0.6 voxel (fine)
and 1.2-1.9 voxels (coarse): the floor below which a change along a fiber is
noise. The output adds a diameter per node, so it is off by default and meant
for measurement, not for simulation input.

## Running it

From `crates/tangle_python/python/examples/ct_unet/`, with `tangle` importable:

```sh
python make_data.py DATA 1001 400                      # training scans (indices past the test sets)
python make_data.py VAL 5000 16                        # validation scans
python train.py DATA VAL RUN --steps 12000             # train (Apple GPU if present)
python train.py DATA VAL RUN2 --steps 4000 --lr 1e-3 --noise 0.25 --init RUN/best.pt
python score_trace.py RUN2/best.pt CACHE dense_hard_1 dense_hard_2 --tidy   # centerline F1 against the truth
python check_diameters.py RUN2/best.pt CACHE dense_hard_7                    # diameters against the truth
python make_figure.py RUN2/best.pt CACHE ct-unet-maps.png
```

`CACHE` is a folder of `ct_examples` truth caches (as `ct_examples.py --output`
writes them); use the same caches as the grey-fitter runs you compare with, so
the scans are identical. In Python, for your own scan:

```python
import torch
from maps import UNet3D, levels, predict
from trace_maps import tiled_axis_points, track, tidy_up

model = UNet3D(16).to("mps"); model.load_state_dict(torch.load("best.pt", map_location="mps")["model"])
grey = levels(volume[::4, ::4, ::4])            # one grey scaling for the whole scan
points = tiled_axis_points(lambda z, y, x: predict(model, volume[z:z+128, y:y+128, x:x+128], "mps", grey),
                           volume.shape)
lines, types = track(*points, shape=volume.shape, min_length=3 * fine_diameter_voxels)
lines, types = tidy_up(lines, types, fine_radius_voxels, coarse_radius_voxels)
```

## Limits

- Trained and tested on one simulator. On a scan whose noise, contrast or fiber
  sizes lie outside the training range, check the traces by eye before
  trusting them, and add training data that covers it.
- The tracer's output is centerlines and a type, not a finished Tangle
  assembly: run a Tangle solve on them for contacts and bend limits.
- The grey-level check some pipelines apply after tracing (drop traces no
  brighter than the void) is a safety net, not part of the method.
