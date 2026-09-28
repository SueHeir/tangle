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
from maps import HEAT, OFFSET, TYPE, UNet3D, predict  # noqa: E402
from trace_maps import trace  # noqa: E402

checkpoint, cache, out = sys.argv[1:4]
name, y = "dense_hard_7", 80
device = "mps" if torch.backends.mps.is_available() else "cpu"
state = torch.load(checkpoint, map_location=device)
model = UNet3D(base=state.get("base", 16)).to(device)
model.load_state_dict(state["model"])
scan = ex.EXAMPLES[name](Path(cache) / f"{name}.json").scan
maps = predict(model, scan.volume, device)
lines, types = trace(maps, min_length=19.5, tidy=(3.25, 7.75))

fig, ax = plt.subplots(1, 5, figsize=(20, 4.6))
panels = [
    (scan.volume[:, y, :], "gray", "simulated scan (xz slice)"),
    (maps[HEAT][0][:, y, :], "magma", "axis heatmap"),
    (np.linalg.norm(maps[OFFSET][:, :, y, :], axis=0) * (np.argmax(maps[TYPE], 0)[:, y, :] > 0), "viridis",
     "distance to own axis (voxels)"),
    (np.argmax(maps[TYPE], 0)[:, y, :], "Set1_r", "type: void / fine / coarse"),
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
        a.plot(l[near, 0], l[near, 2], ".", color="red" if t == 2 else "magenta", ms=1)
a.set_title("traced (magenta/red) on truth (green), ±6 voxels")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
