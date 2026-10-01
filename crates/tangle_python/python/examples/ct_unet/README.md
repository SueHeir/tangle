# ct_unet: fibers from a CT map network

A 3D U-Net, trained on Tangle's simulated scans, turns a CT scan into maps (axis heatmap, offset to each
voxel's own axis, sign-free direction, fiber type), and a short tracer reads the fibers off the maps.
Centerline F1 on the simulated `dense_hard` sets: 0.995, against 0.915 for the grey fitter.
See [docs/ct_unet.md](../../../../../docs/ct_unet.md) for the method, results and commands, and the
[tutorial](../../../../../docs/ct_unet_tutorial.md) to run it on your own scan.

| file | what it does |
|---|---|
| `check_scan.py` | a first look at any scan: size, voxel size, grey levels, and the fiber sizes the network finds |
| `find_fibers.py` | the fibers of any scan, from tiles of the network to tables, Fiji stacks, ParaView, OVITO and `fit.json` |
| `scan_io.py`, `fiber_outputs.py` | reading scans in the common formats; writing the results |
| `make_demo_scan.py`, `compare_fibers.py` | a small made-up scan with known fibers, and scoring found fibers against known ones |
| `make_data.py` | simulated training scans and their truth |
| `maps.py` | the network, its targets, augmentation, losses, `predict` |
| `train.py` | training (and fine-tuning with `--init`, correlated noise with `--noise`) |
| `trace_maps.py` | axis votes, tracking, tiled volumes, `tidy_up` (doubles, gap joins) |
| `score_trace.py` | centerline F1 of the traced fibers against the truth |
| `diameters.py`, `check_diameters.py` | optional per-node diameters and oval widths, and their check |
| `fit_maps.py` | feeds a clean image made from the maps to `ct.fit_fibers` (a first attempt; it scores poorly) |
| `make_figure.py` | the figure in the docs |
