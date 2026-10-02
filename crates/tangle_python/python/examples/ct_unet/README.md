# ct_unet: fibers from a CT map network

A 3D U-Net, trained on Tangle's simulated scans, turns a CT scan into maps (axis heatmap, offset to each
voxel's own axis, sign-free direction, fiber type), and a short tracer reads the fibers off the maps.
Centerline F1 on the simulated `dense_hard` sets: 0.995, against 0.915 for the grey fitter.
See [docs/ct_unet.md](../../../../../docs/ct_unet.md) for the method, results and commands.

| file | what it does |
|---|---|
| `make_data.py` | simulated training scans and their truth |
| `mixed.py` | the `--mixed` scans: every kind of variety at random inside each scan |
| `overlaps.py` | the check that no true fibers pass through each other |
| `maps.py` | the network, its targets, its optional hints, augmentation, losses, `predict` |
| `train.py` | training (and fine-tuning with `--init`, correlated noise with `--noise`) |
| `trace_maps.py` | axis votes, tracking, tiled volumes, `tidy_up` (doubles, gap joins) |
| `score_trace.py` | centerline F1 of the traced fibers against the truth |
| `diameters.py`, `check_diameters.py` | optional per-node diameters and oval widths, and their check |
| `fit_maps.py` | feeds a clean image made from the maps to `ct.fit_fibers` (a first attempt; it scores poorly) |
| `make_figure.py` | the figure in the docs |
| `runpod/pod.py` | training on a rented GPU pod in one command (see [runpod/README.md](runpod/README.md)) |
