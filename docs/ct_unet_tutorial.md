# Tutorial: finding the fibers in your own CT scan

This walks through running Tangle's CT map network on any CT scan of fibers,
from opening the file to fiber tables, pictures to check by eye, and files for
Fiji, ParaView, OVITO, Excel and Tangle. You need no machine learning
background: the network is used as a ready-made tool, and every step says what
to look at and what to change when something looks wrong.

How the network works, and how well it does on simulated scans, is in the
[CT map network guide](ct_unet.md). In one paragraph: a 3D U-Net reads the scan
in 128-voxel cubes and marks, for every voxel, whether it is fiber, where the
middle of its fiber is and which way the fiber runs; a short tracer then
follows the fibers through those marks. You get each fiber's centerline,
diameter and type, and the bonds between fibers where the sample is bonded.

All the scripts are in
[`crates/tangle_python/python/examples/ct_unet/`](../crates/tangle_python/python/examples/ct_unet/):

| script | what it does |
|---|---|
| `make_demo_scan.py` | makes a small scan with known fibers, to try everything on first |
| `check_scan.py` | a first look at your scan: size, voxel size, grey levels and the fiber sizes |
| `find_fibers.py` | finds the fibers and writes all the results |
| `compare_fibers.py` | scores found fibers against known ones |
| `scan_io.py`, `fiber_outputs.py` | the file reading and writing the scripts share |

Every script explains its options with `--help`.

## What you need

- **Python 3.11 or newer** (what Tangle itself needs) with NumPy, SciPy,
  tifffile, matplotlib and PyTorch, best in a virtual environment of their
  own. On a Mac or Linux:

  ```sh
  python3 -m venv ct-env
  source ct-env/bin/activate
  python -m pip install numpy scipy tifffile matplotlib torch
  ```

  (On Windows, `py -m venv ct-env` and `ct-env\Scripts\activate`.) The
  `python` in the commands below is the environment's, so activate it again in
  each new terminal (`source ct-env/bin/activate`, from the folder you made it
  in; a Mac has no plain `python` outside one). If you have built Tangle
  already ([Python guide](../crates/tangle_python/README.md)), activate its
  environment instead and install only `torch` there.
  On a PC with an NVIDIA graphics card, install the CUDA build of PyTorch
  instead (the command for your system is on [pytorch.org](https://pytorch.org)).
  A few file formats need one more package, named in the error message if you
  need it: `h5py` (HDF5), `zarr` (Zarr), `nibabel` (NIfTI), `pillow` (PNG,
  JPEG or BMP slices).
- **A GPU helps a lot.** The network runs on an Apple silicon GPU, an NVIDIA GPU
  or the CPU; on the CPU it is much slower, fine for a small piece but not for
  a whole scan. On an Apple M5 Pro, the demo scan of step 1 takes 4 s with the
  GPU and 70 s on the CPU alone.
- **The trained network**, a `.pt` file. It is not published yet: train one
  with `train.py` (see [Running it](ct_unet.md#running-it) in the guide) or ask
  the Tangle authors for the current one. Below it is called `best.pt`; use
  your file's name and path.
- **The scripts**: clone the repository, or copy the `ct_unet` folder. Finding
  fibers does not need Tangle itself to be built; only the last step, carrying
  the fibers into Tangle, does.

Run the scripts from the `ct_unet` folder (or give their full path; they find
each other either way):

```sh
cd crates/tangle_python/python/examples/ct_unet
```

## 1. Try it on a demo scan

Before your own scan, check that everything runs on a scan whose fibers are
known:

```sh
python make_demo_scan.py demo
python find_fibers.py demo/demo_scan.tif --weights best.pt --diameters 10.5um,21um --out demo/found
python compare_fibers.py demo/found demo/truth
```

`make_demo_scan.py` writes `demo/demo_scan.tif`, a 160-voxel cube of 1.5 µm
voxels with two kinds of gently curved fibers (10.5 and 21 µm across), and the
true fibers in `demo/truth/`. `find_fibers.py` finds the fibers in it and
writes everything to `demo/found/`. `compare_fibers.py` prints how much of the
true fibers' length was found (recall), how much of what was found lies on a
true fiber (precision), and how well the diameters and types match. With the
current network on an Apple M5 Pro, the three commands take about 1 s, 4 s
and under a second, and the comparison reads:

```text
found 66 fibers, 12.5 mm; true 67 fibers, 12.1 mm
centerlines within 0.5 true radii: recall 0.996, precision 0.966, F1 0.981
true type 1: 53 fibers, median diameter 10.5 um; 51 found fibers lie on them, median diameter 10.4 um
true type 2: 14 fibers, median diameter 21 um; 14 found fibers lie on them, median diameter 20.1 um
typed like the true fiber they lie on: 100.0% of the matched found length
```

Your numbers should come out close to these (another network file gives
somewhat different ones); much lower ones mean something is wrong with the
install or the network file. Open `demo/found/overlay.png` to see what a
result looks like.

The demo scan is much simpler than a real scan; it shows that the install and
the network file work, not how well the method does on your sample.

## 2. Look at your scan

```sh
python check_scan.py my_scan.tif
```

It saves `my_scan_check.png` in the current folder (or the one given with
`--out`), the middle slice in each direction next to the histogram of grey
values, and prints what the file holds:

```text
my_scan.tif: TIFF stack (ImageJ), 1024 x 1024 x 800 voxels (x, y, z), uint16, 1.7 GB
Voxel size 1.3 um (from the TIFF's ImageJ metadata); the scan is 1.33 mm x 1.33 mm x 1.04 mm
Grey values: 0.5th percentile 912, median 1043 (the network's empty space), 99.5th percentile 4521 (its bright fiber)
Solid/void split (Otsu): 2210; about 14% of the scan is above it
```

Check three things:

- **The voxel size.** It is read from the file when the file says it (ImageJ
  and OME TIFF, MetaImage, NRRD, VTK image data, NIfTI, OME-Zarr, HDF5
  `element_size_um`, a SkyScan `.log` next to the slices). Otherwise, or if it
  is wrong, give it with `--voxel-size 1.3um` (units `um`, `nm` or `mm`). If you
  really don't know it, `--voxel-size 1um` makes every length come out in
  voxels.
- **The fibers are brighter than the space around them.** The network expects
  bright fibers. If yours are dark, add `--invert` to every command.
- **Most of the scan is empty space.** The network takes the median grey as
  "empty space". For a mostly solid sample `check_scan.py` warns you and prints
  the `--levels` to use instead.

It also warns when the voxels are not cubes (resample the scan first, e.g. in
Fiji with Image > Scale), and when part of the scan is one exact value, as
outside the round field of view of many reconstructions: crop to the inside of
the sample there, so the network never sees that artificial edge.

**File formats.** A TIFF stack (one multi-page file), a folder of 2D slices
(TIFF, PNG, JPEG or BMP; give the folder, or a pattern in quotes such as
`"rec/*.bmp"`), NumPy `.npy` and `.npz`, MetaImage (`.mhd`, `.mha`), NRRD
(`.nrrd`, `.nhdr`), VTK image data (`.vti`), HDF5 (`.h5`, `.hdf5`, `.nxs`;
pick the dataset with `--dataset` if there are several), Zarr folders, NIfTI
(`.nii`, `.nii.gz`), and raw binary data of any name with its size and type:

```sh
python check_scan.py scan.raw --raw-shape 1024x1024x800 --raw-dtype uint16 --voxel-size 2um
```

(`--raw-shape` is width x height x number of slices; add `--raw-endian big` or
`--raw-header-bytes N` if your scanner's notes say so.) Anything else, most CT
software and Fiji can save as a TIFF stack. Big files are read piece by piece,
never whole, except compressed ones.

## 3. Measure the fiber sizes

The network finds fibers more accurately when it is told their diameters, and
the diameters also decide each fiber's type. If you know them (from a data
sheet, or by measuring a few fibers in Fiji), use those. If not, let the
network measure them:

```sh
python check_scan.py my_scan.tif --weights best.pt
```

This runs the network, told nothing, on a 192-voxel cube from the middle of
the scan (`--size` to change it), and groups the diameters it finds into fiber
types:

```text
Found 96 fibers. Their diameters (um), grouped into types:
  type 1: 71 fibers, 58% of the length, median 12.4 um (9.5 voxels as analysed), middle half 11.2-13.6
  type 2: 25 fibers, 42% of the length, median 30.1 um (23.2 voxels as analysed), middle half 27.9-32.5

Next, find the fibers in a small piece first (then drop --center-crop for the whole scan):
  python find_fibers.py my_scan.tif --weights best.pt --diameters 12.4um,30.1um --center-crop 256
```

(The numbers here are made up for the example.) It saves
`my_scan_check_fibers.png` with the fibers it found. If the sample has a
different number of fiber types than it guessed, rerun with `--types 2` (or
however many), or give the diameters you know.

**The sizes the network handles.** It was trained on fibers 3.5 to 28 voxels
across. If your thickest fibers are wider, average the scan into bigger voxels
with `--bin 2` (or 3, 4: every 2 x 2 x 2 voxels become one); `check_scan.py`
and `find_fibers.py` both suggest it. Fibers thinner than about 4 voxels can't
be helped by any setting: they need a scan with smaller voxels.

## 4. A first run on a small piece

```sh
python find_fibers.py my_scan.tif --weights best.pt --diameters 12um,30um --center-crop 256
```

`--center-crop 256` analyses a 256-voxel cube from the middle (27 tiles for
the network), which is quick on a GPU: about 10 s in all on an M5 Pro. The
results go to `my_scan_fibers/` (or the folder you give with `--out`).

Open `overlay.png` for a quick look, then **open `overlay.tif` in Fiji**
(File > Open; it opens as a stack in micrometers) and page through the slices.
Each traced fiber is tinted and outlined in its own color, the same color all
along it; bonds are yellow dots. (With `--invert` the overlay shows the scan
inverted, as the network saw it.) Things to look for:

| what you see | what to change |
|---|---|
| fibers left grey (missed) | check the diameters (in voxels, 3.5 to 28), `--invert`, the grey levels in step 2 |
| tint over empty space (false fibers) | grainy or blotchy empty space can look like fiber to the network; check the grey levels, try `--levels`, and treat the result with care |
| one fiber in several colors | the trace broke where the fiber was hard to follow (crossings, faint stretches); some of this is normal |
| fibers drawn too thick or too thin | the diameters given are off; rerun `check_scan.py --weights` or measure a few fibers |
| many short pieces | raise `--min-length` (e.g. `--min-length 60um`) |
| the wrong types | give `--diameters` (or `--types`) |
| missed or extra bonds | bonds are the least reliable output (see the guide); say whether the sample is bonded with `--bonded yes` or `--bonded no` |

`--crop` picks any other piece: `--crop x=100:612,y=0:512,z=200:456` (voxel
ranges as Fiji shows them, ends excluded; an axis left out is kept whole; z
counts slices from 0, while Fiji's slice numbers start at 1).

## 5. The whole scan

Leave out the crop:

```sh
python find_fibers.py my_scan.tif --weights best.pt --diameters 12um,30um
```

It prints how many 128-voxel tiles the network has to look at and, as it goes,
the time left. On an M5 Pro the network takes about 0.3 s a tile on the GPU
(a 384-voxel cube, 125 tiles, took 42 s in all) and about 8 s a tile on the
CPU. On a big scan this takes a while; meanwhile:

- **Stopping is safe.** Press Ctrl-C, or let the computer sleep. Running the
  same command again carries on from the saved progress in
  `my_scan_fibers/work/`; `--restart` starts over.
- **Reruns are quick.** A rerun that changes only how the fibers are traced or
  written (`--min-length`, `--types`, `--stacks`, `--binder`) reuses the
  network's work. Changing the region, the network, `--diameters`,
  `--bonded` or `--levels` runs the network again. Each run replaces the
  results of the one before in the same folder (here, the small piece from
  step 4); give `--out` a new folder to keep both.
- **Disk space.** A scan that can't be read in place (slices, compressed or
  binned files) is copied once to `work/volume.npy`, as big as the region in
  its own data type. Delete `work/` when you are done.
- **Very big scans.** The overlay and label stacks are skipped above 600
  million voxels (about an 840-voxel cube); `--stacks yes` writes them anyway,
  at 5 bytes per voxel. A side longer than 7,000 voxels must be split with `--crop`
  or binned.

## 6. The results

| file | what it holds | open it with |
|---|---|---|
| `summary.txt` | fiber counts and diameters per type, lengths, orientation, bonds, volume fractions, and how the run was made | any text editor |
| `overlay.png` | the middle slice each way, the scan above and the traced fibers below | any image viewer |
| `overlay.tif` | the scan with every traced fiber tinted and outlined, bonds in yellow (RGB stack) | Fiji |
| `labels.tif` | each voxel's fiber number (0 = no fiber), as drawn from the traced centerlines and diameters | Fiji (Image > Lookup Tables > glasbey shows each fiber in its own color) |
| `binder.tif` | with `--binder`: solid voxels outside every traced fiber (255), see below | Fiji |
| `fibers.csv` | one row per fiber (columns below) | Excel, Python, R |
| `centerlines.csv` | every centerline point: fiber, point number, x, y, z | Excel, Python |
| `bonds.csv` | every bond: its number, the two fibers it joins, where it is, how strong the network's signal was | Excel, Python |
| `diameters.csv` | with `--diameter-profile`: the diameter at every voxel along every fiber, and an oval's long and short widths | Excel, Python |
| `fibers.vtk`, `bonds.vtk` | the centerlines and bonds | ParaView |
| `fibers.dump`, `view_in_ovito.py` | the fibers as chains of capsules, and bonds | OVITO |
| `fit.json` | the fibers in Tangle's format, smoothed to its bend limit | Tangle (step 7) |
| `run.json` | every setting of the run | a text editor |

**Coordinates and units.** Everything is in micrometers, measured from the
corner of the analysed region: x across the image (columns), y down it
(rows), z through the slices. With a crop, `summary.txt` and `run.json` say
where that corner sits in the scan. Fiber, type and bond numbers start at 1;
type 1 is the thinnest.

**`fibers.csv` columns:**

| column | meaning |
|---|---|
| `fiber`, `type` | the fiber's number (as in `labels.tif`) and type |
| `diameter_um` | its diameter, from the network's measurement along it |
| `length_um` | its length inside the region |
| `support` | the share of its trace backed by the network (below 0.5: check it by eye) |
| `straightness` | end-to-end distance over length (1 = straight) |
| `out_of_plane_deg` | the angle between the fiber and the slice (xy) plane, 0 to 90 |
| `in_plane_deg` | its direction within the xy plane, 0 to 180 degrees from x |
| `touches_edge` | 1 when an end is at the region's faces, so its length is a lower bound |
| `start_*_um`, `end_*_um` | its two ends |
| `diameter_median_um` and after | with `--diameter-profile`: the median and spread of its diameter along it, and an oval's long and short widths |

**In ParaView:** open `fibers.vtk`, add a Tube filter, set its Scalars to
`radius_um` and Vary Radius to "By Absolute Scalar", and color by `fiber_id`
or `type`.

**In OVITO:** run `view_in_ovito.py` with OVITO's Python (`ovitos
view_in_ovito.py`, or `python view_in_ovito.py` where the `ovito` package is
installed) and open the `fibers.ovito` it saves; or load `fibers.dump` in
OVITO directly and set the particle shape to Spherocylinder.

**Binder.** `--binder` writes `binder.tif`: voxels brighter than the solid/void
split that are not inside a traced fiber (fibers always win, with a one-voxel
margin so fiber edges don't count as binder), without pieces smaller than 20
voxels. It depends on how well the fibers were traced: missed fibers turn up
as binder.

**Diameters along each fiber.** `--diameter-profile` runs the network a second
time to measure the diameter at every voxel along every fiber (see
[the guide](ct_unet.md#fiber-diameters-along-each-fiber-optional)). On
simulated scans it reads about 0.4 voxel high, and a fiber of constant size
still shows a spread of about 0.3 to 0.6 voxel (fine fibers) to 1.2 to 1.9
voxels (coarse) along it: smaller changes are noise.

## 7. Into Tangle

`fit.json` is in the same format as `tangle.ct`'s own fits, so with Tangle
built (see the [Python guide](../crates/tangle_python/README.md); install with
the `ct` extras) the fibers carry straight on:

```python
import tangle.ct as ct

fit = ct.load_fit("my_scan_fibers/fit.json")
assembly = fit.to_assembly()                 # a Tangle assembly; its cell is the analysed region
relaxed, run = fit.relax()                   # Tangle's contact solve: no overlaps, bend limits kept
print(run.max_penetration)
populations = fit.suggested_population()     # FiberPopulation(s) with the measured statistics, one per type
```

Tangle takes a fiber only if it bends no tighter than its type's bend limit
(5 diameters unless set otherwise), and traced centerlines wiggle by a fraction
of a voxel. So in `fit.json` each centerline is resampled one radius of its
type apart and smoothed where it bends tighter than that; `summary.txt` says
how far this moved the fibers (on the demo scan, 2.6 µm at most), and the
tables and stacks keep the traced lines. The traced centerlines can also
overlap a little where fibers touch; `relax` pushes them apart with Tangle's
own solver. On the demo scan it takes under 2 s on an M5 Pro's GPU and leaves
overlaps of a few nanometers. From an assembly, everything else in Tangle
applies: analysis, OVITO and PuMA export (see the
[PuMA guide](puma_interoperability.md)).

## Options at a glance

`find_fibers.py SCAN --weights NETWORK.pt [options]`:

| option | what it does |
|---|---|
| `--voxel-size 1.3um` | the voxel size, when the file doesn't say or says it wrong |
| `--diameters 12um,30um` | the fiber diameters, one per type (also `16vox`: voxels of the scan) |
| `--types 2` | how many fiber types, when their diameters aren't known |
| `--bonded yes` / `no` | whether the fibers are bonded |
| `--center-crop 256` | analyse a cube from the middle |
| `--crop x=0:512,y=0:512,z=100:356` | analyse any box |
| `--bin 2` | average 2 x 2 x 2 voxels into one (for thick fibers) |
| `--invert` | the fibers are darker than the space around them |
| `--levels 1040,4500` | the grey of empty space and of fiber, in the scan's own grey values (for mostly solid scans; write `--levels=-1000,85` when a value is negative) |
| `--min-length 50um` | drop traced fibers shorter than this (default 3 diameters of the thinnest type) |
| `--out FOLDER` | where the results go |
| `--stacks yes` / `no` | write (or skip) `overlay.tif` and `labels.tif` |
| `--binder` | also write `binder.tif` |
| `--diameter-profile` | the diameter along every fiber |
| `--device cpu` | run the network on the CPU (or `mps`, `cuda`; default: a GPU if there is one) |
| `--restart` | throw away saved progress |
| `--dataset`, `--raw-shape`, `--raw-dtype`, `--raw-endian`, `--raw-header-bytes` | for HDF5, Zarr, `.npz`, `.vti` and raw files |

## When something goes wrong

| message or symptom | what to do |
|---|---|
| "The file doesn't say its voxel size" | add `--voxel-size` |
| "Don't know how to read ... files" | save the scan as a TIFF stack (Fiji: File > Save As > Tiff...), or give `--raw-shape` and `--raw-dtype` for raw data |
| "... holds one 2D image" | give the folder of slices (or a pattern), not one slice |
| "... holds more than one volume" | the file has several channels or time points: save the one you want as its own stack |
| "needs the Python package ..." | install the package it names |
| "Couldn't load the network" | check the `.pt` file's path; a network saved by a much newer PyTorch may need that PyTorch |
| "no GPU found" | it runs on the CPU, slowly: try a `--center-crop 128` first |
| warning about dark fibers | add `--invert` if the fibers really are darker than the space around them |
| warning that most of the scan is solid | use the `--levels` it prints |
| warning about voxels that are exactly one value, or not numbers (NaN) | crop to the inside of the field of view |
| "almost no contrast" | the region may lie outside the sample: check the crop, or give `--levels` |
| fibers wider than 28 voxels | `--bin 2` (or the value it suggests) |
| fibers narrower than about 4 voxels | they need a scan with smaller voxels |
| the run stopped | run the same command again: it carries on |
| "Not enough disk space" | free some space or write elsewhere with `--out`; delete old `work/` folders; a crop or `--bin` makes the copy smaller, and `--stacks no` skips the stacks |
| a side longer than 7,000 voxels | split the scan with `--crop`, or `--bin 2` |

## Limits

- The network learned from Tangle's simulated scans. Real scans have noise,
  rings, streaks and contrast it hasn't seen, so always check the result by
  eye in `overlay.tif` before using the numbers.
- Fibers must be 3.5 to 28 voxels across (bin for thicker ones), brighter than
  their surroundings (or `--invert`), in cubic voxels, with mostly empty space
  around them (or `--levels`).
- Bonds come from the network's bond-point peaks and are found less reliably
  than fibers (see the guide's numbers).
- The result is centerlines with diameters and types, not a solved
  structure: `fit.relax()` in Tangle makes it one.
