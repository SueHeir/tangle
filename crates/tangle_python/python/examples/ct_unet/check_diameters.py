"""How well diameters.py measures fiber diameters, on ct_examples structures with known truth.

usage: python check_diameters.py CHECKPOINT CACHE_DIR EXAMPLE...

Each traced fiber is matched to the true fiber most of its nodes lie on; its median measured diameter is compared
with the true one (equal-area; for ovals also the long and short widths).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parents[1]))
import ct_examples as ex  # noqa: E402
from diameters import measure  # noqa: E402
from maps import load, predict  # noqa: E402
from trace_maps import trace  # noqa: E402

checkpoint, cache, *names = sys.argv[1:]
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = load(checkpoint, device)
for name in names:
    scan = ex.EXAMPLES[name](Path(cache) / f"{name}.json").scan
    um = scan.voxel_size * 1e6
    maps = predict(model, scan.volume, device)
    lines, _ = trace(maps, min_length=15.0)
    fibers = measure(maps, lines, spacing_um=um)
    truth = np.concatenate([np.asarray(l) for l in scan.centerlines])
    truth_id = np.concatenate([np.full(len(l), g) for g, l in enumerate(scan.centerlines)])
    tree = cKDTree(truth)
    rows = []
    for f in fibers:
        ok = np.isfinite(f["diameter"])
        if ok.sum() < 5:
            continue
        _, i = tree.query(f["nodes"][ok])
        g = int(np.bincount(truth_id[i]).argmax())
        true_d = 2 * scan.radii[g] * um
        if scan.semi_axes is not None:
            true_long, true_short = 2 * scan.semi_axes[g] * um
        else:
            true_long = true_short = true_d
        rows.append((true_d, np.median(f["diameter"][ok]), true_long, np.median(f["long"][ok]),
                     true_short, np.median(f["short"][ok]), np.std(f["diameter"][ok])))
    rows = np.array(rows)
    print(f"{name} ({um:.2f} um voxels, {len(rows)} fibers):")
    for d in np.unique(np.round(rows[:, 0], 1)):
        r = rows[np.abs(rows[:, 0] - d) < 0.3]
        err = r[:, 1] - r[:, 0]
        print(f"  true {d:5.1f} um (long {r[0, 2]:.1f} short {r[0, 4]:.1f}), n {len(r)}: measured median {np.median(r[:, 1]):.2f}"
              f" (error median {np.median(err):+.2f}, IQR {np.percentile(err, 25):+.2f}..{np.percentile(err, 75):+.2f}),"
              f" long {np.median(r[:, 3]):.1f} short {np.median(r[:, 5]):.1f}, along-fiber sd {np.median(r[:, 6]):.2f}",
              flush=True)
