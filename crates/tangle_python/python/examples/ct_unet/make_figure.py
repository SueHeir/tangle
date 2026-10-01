"""The README figure: one simulated scan (dense_hard_7), the network's maps, and the traced fibers against the truth.

usage: python make_figure.py CHECKPOINT CACHE_DIR OUT.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import ct_examples as ex  # noqa: E402
from maps import FIBER, HEAT, OFFSET, RADIUS, load, predict  # noqa: E402
from fiber_types import assign_types, features  # noqa: E402
from maps import levels  # noqa: E402
from trace_maps import trace  # noqa: E402

checkpoint, cache, out = sys.argv[1:4]
name, y = "dense_hard_7", 80
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = load(checkpoint, device)
scan = ex.EXAMPLES[name](Path(cache) / f"{name}.json").scan
maps = predict(model, scan.volume, device)
lines, radii = trace(maps, min_length=19.5)
types, _ = assign_types(features(lines, radii, scan.volume, levels(scan.volume)), 2)

fig, ax = plt.subplots(1, 5, figsize=(20, 4.6))
panels = [
    (scan.volume[:, y, :], "gray", "simulated scan (xz slice)"),
    (maps[HEAT][0][:, y, :], "magma", "axis heatmap"),
    (np.linalg.norm(maps[OFFSET][:, :, y, :], axis=0) * (maps[FIBER][0][:, y, :] > 0.5), "viridis",
     "distance to own axis (voxels)"),
    (np.where(maps[FIBER][0][:, y, :] > 0.5, 2 * maps[RADIUS][0][:, y, :], np.nan), "plasma",
     "fiber diameter map (voxels)"),
]
for a, (img, cmap, title) in zip(ax, panels):
    a.imshow(img, cmap=cmap, origin="lower", interpolation="nearest")
    a.set_title(title)
a = ax[4]
a.imshow(scan.volume[:, y, :], cmap="gray", origin="lower", alpha=0.5)
for line in scan.centerlines:
    l = np.asarray(line)
    near = np.abs(l[:, 1] - y) < 6
    if near.any():
        a.plot(l[near, 0], l[near, 2], ".", color="lime", ms=2)
for line, t in zip(lines, types):
    l = np.asarray(line)
    near = np.abs(l[:, 1] - y) < 6
    if near.any():
        a.plot(l[near, 0], l[near, 2], ".", color="red" if t == 1 else "magenta", ms=1)
a.set_title("traced and typed (magenta/red) on truth (green)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
