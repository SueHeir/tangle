"""How good the network's maps are, voxel by voxel, on make_data volumes.

usage: python map_quality.py CHECKPOINT NPZ...
"""
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

from maps import HEAT, OFFSET, TYPE, UNet3D, predict, targets

device = "mps" if torch.backends.mps.is_available() else "cpu"
state = torch.load(sys.argv[1], map_location=device)
model = UNet3D(base=state.get("base", 16)).to(device)
model.load_state_dict(state["model"])
print(f"checkpoint step {state['step']}")
for path in sys.argv[2:]:
    data = np.load(path)
    maps = predict(model, data["volume"], device)
    t = targets(data)
    kind = np.argmax(maps[TYPE], axis=0)
    fg_true, fg_pred = t["type"] > 0, kind > 0
    type_acc = (kind[fg_true] == t["type"][fg_true]).mean()
    iou = (fg_true & fg_pred).sum() / (fg_true | fg_pred).sum()
    own = t["own"][0] > 0
    off_err = np.linalg.norm(maps[OFFSET] - t["offset"], axis=0)[own]
    # Axis points: every fiber voxel voting at voxel + offset; how close the votes land to a true axis.
    z, y, x = np.nonzero(fg_pred)
    votes = np.stack([x, y, z], 1) + 0.5 + maps[OFFSET][:, z, y, x].T
    d, i = cKDTree(data["p_pos"]).query(votes)
    tol = 0.5 * data["p_rad"][i]
    heat_hi = maps[HEAT][0] > 0.5
    zz, yy, xx = np.nonzero(heat_hi)
    dh, ih = cKDTree(data["p_pos"]).query(np.stack([xx, yy, zz], 1) + 0.5)
    print(f"{path.split('/')[-1]}: fg IoU {iou:.3f}, type acc {type_acc:.3f}, offset err median {np.median(off_err):.2f} "
          f"p90 {np.percentile(off_err, 90):.2f} vox, votes within tol {np.mean(d <= tol):.3f}, "
          f"heat>0.5 voxels within tol {np.mean(dh <= 0.5 * data['p_rad'][ih]):.3f} "
          f"(true heat>0.5 {int((t['heat'] > 0.5).sum())}, pred {int(heat_hi.sum())})")
