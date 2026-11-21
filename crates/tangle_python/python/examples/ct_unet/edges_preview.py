"""A look at ``edges_<index>`` scans (``edges.py``): per scan, the middle slice as the network sees it (scaled by
the stored greys), the fiber truth on it, and the grey along a line across the field of view's edge.

    python edges_preview.py FOLDER [OUT.png]

Also prints, as a check of ``edges.truncation_profile``, a round sample reconstructed without truncation (it
should come back at its own attenuation inside) and with each padding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def profile_check() -> None:
    import edges

    radius = 600.0
    r = np.array([0, 300, 500, 560, 590, 598, 600, 610, 650, 800], float)
    print("r (voxels from the axis):", r.astype(int).tolist())
    print("no truncation, sample 0.8 R, density 0.2:", np.round(edges.truncation_profile(radius, 0.8 * radius, 0.2,
                                                                                        "zero", 0.2, r), 3).tolist())
    for pad in ("zero", "edge", "taper"):
        print(f"sample 2 R, density 0.2, {pad} padding:",
              np.round(edges.truncation_profile(radius, 2 * radius, 0.2, pad, 0.2, r), 3).tolist())


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from maps import normalize

    folder = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "edges_preview.png"
    files = sorted(folder.glob("edges_*.npz"))
    if not files:
        raise SystemExit(f"no edges_*.npz in {folder}")
    fig, axes = plt.subplots(len(files), 3, figsize=(11, 3.4 * len(files)), squeeze=False)
    for row, path in zip(axes, files):
        data = np.load(path)
        s = json.loads(path.with_suffix(".json").read_text())["settings"]
        f = s["field"]
        image = normalize(data["volume"], tuple(float(g) for g in data["grey"]))
        z = image.shape[0] // 2
        row[0].imshow(image[z], cmap="gray", vmin=-1.0, vmax=1.5)
        row[0].set_title(f"{path.stem}: {f['kind']}, fill {f['fill']}, {f['padding']} padding\n"
                         f"R {f['radius']:.0f}, sample {f['sample']} R, rings {f['rings']}, "
                         f"rim {'yes' if f['rim'] else 'no'}", fontsize=8)
        row[1].imshow(image[z], cmap="gray", vmin=-1.0, vmax=1.5)
        row[1].imshow(np.ma.masked_equal(data["labels"][z] > 0, False), cmap="autumn", alpha=0.5)
        row[1].set_title("truth: fibers (orange)", fontsize=8)
        # a line through the middle, across the edge (towards the axis)
        n = image.shape[1]
        t = np.arange(n)
        ux, uy = np.cos(f["angle"]), np.sin(f["angle"])
        x = np.clip(np.round(n / 2 + (t - n / 2) * ux).astype(int), 0, n - 1)
        y = np.clip(np.round(n / 2 + (t - n / 2) * uy).astype(int), 0, n - 1)
        line = image[z - 2:z + 3][:, y, x].mean(0)
        row[2].plot(t, line, lw=1)
        row[2].axhline(0, color="k", lw=0.5)
        row[2].axhline(1, color="tab:orange", lw=0.5)
        row[2].set_ylim(-3, 3)
        row[2].set_title("grey across the edge (0 empty space, 1 bright fiber)", fontsize=8)
        for a in row[:2]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"saved {out}")


if __name__ == "__main__":
    profile_check()
    if len(sys.argv) > 1:
        main()
