"""The guide's validation figures: what the network's maps should be, and what it draws, on held-out simulated scans.

One figure per case group: two scans, a truth row and a prediction row each, a column per output. Each scan's slice
is the one with the most fibers, binder and bonds in its truth, never picked by how well the network did. The
network gets no hints. Scans come from make_data; a fiber-free scan is scaled by its stored greys, as in training.

usage: python make_validation_figures.py CHECKPOINT SCANS_DIR OUT_DIR

SCANS_DIR holds one folder per case group (GROUPS), each with make_data volumes; the first two by name are used.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from maps import BINDER, BONDPT, CHANNELS, DIRECTION, FIBER, HEAT, OFFSET, RADIUS, load, predict
from score_trace import _samples_inside, centerline_agreement, from_npz, true_maps
from trace_maps import principal, trace

GROUPS = {  # folder: title on the figure
    "dense_hard": "dense_hard: two fiber types, dense and bent",
    "scanned": "scanned: other voxel sizes, blur and orientations",
    "pairs": "pairs: fine fibers in touching side-by-side pairs",
    "hardpairs": "hard pairs: thin touching pairs, blurred and noisy",
    "stiff": "stiff: nearly straight fibers bonded at tight crossings",
    "web_easy": "easy webs: binder webs and fillets, clean scans",
    "web2": "webs: binder webs as bright as the fibers (blur capped at 0.7 of the thinnest fiber)",
    "mixed": "mixed: every kind of variety at random inside each scan",
    "seams": "seams: mixed, plus binder seams along neighbouring fibers",
    "bends": "bends: one straight and one bendable type, same size and material",
    "empty": "empty: no fibers at all, only noise and black regions",
}
FIBER_CUT = 0.5  # the offset, direction and radius maps only mean something on fiber voxels
EMPTY = "#26262b"  # voxels with nothing to show in those maps


def infer(model, path, device):
    """(scan, true maps, predicted maps, one-line score) for a make_data volume."""
    volume, lines, radii, _, data = from_npz(path)
    grey = tuple(float(g) for g in data["grey"]) if "grey" in data else None
    maps = pad_predict(model, volume, device, grey)
    truth = true_maps(data) if lines else np.zeros((CHANNELS, *volume.shape), np.float32)  # no fibers: all zero
    shortest = 6.0 * float(radii.min()) if len(radii) else 10.0
    traced, _ = trace(maps, min_length=shortest)
    if not lines:
        return volume, truth, maps, f"{len(traced)} false traces"
    shape = volume.shape[::-1]
    stubs = {g for g, line in enumerate(lines) if 0.5 * len(_samples_inside(line, shape)) < shortest}
    return volume, truth, maps, f"F1 {centerline_agreement(traced, lines, radii, shape, skip=stubs)['f1']:.2f}"


def pad_predict(model, volume, device, grey):
    """``predict`` for any shape: edge-padded up to sides divisible by 16 and cropped back."""
    pad = [(0, -n % 16) for n in volume.shape]
    maps = predict(model, np.pad(volume, pad, mode="edge"), device, grey)
    return maps[(slice(None), *(slice(0, n) for n in volume.shape))]


def best_slice(truth):
    """The z with the most structure in the truth (fibers, then binder and bonds), or the middle of an empty scan."""
    nz = truth.shape[1]
    margin = min(8, nz // 4)
    per_z = ((truth[FIBER][0] > 0.5).mean((1, 2)) + 5.0 * (truth[BINDER][0] > 0.5).mean((1, 2))
             + 50.0 * (truth[BONDPT][0] > 0.3).mean((1, 2)))
    if per_z.max() == 0.0:
        return nz // 2
    return margin + int(per_z[margin:nz - margin].argmax())


def direction_rgb(maps, z, on_fiber):
    """The fiber direction as colour: |x| red, |y| green, |z| blue (t and -t look alike, as in the tensor)."""
    ny, nx = on_fiber.shape
    tensor = maps[DIRECTION][:, z].reshape(6, -1).T
    rgb = np.abs(principal(tensor)).reshape(ny, nx, 3)
    rgb[~on_fiber] = matplotlib.colors.to_rgb(EMPTY)
    return rgb


def panels(volume, maps, z, is_truth):
    """The figure's columns for one row: (image, colormap, low, high)."""
    on_fiber = maps[FIBER][0, z] > FIBER_CUT
    if is_truth:
        on_fiber = on_fiber & (np.abs(maps[OFFSET][:, z]).sum(0) > 0)  # truth: voxels with their own axis
    distance = np.ma.masked_where(~on_fiber, np.linalg.norm(maps[OFFSET][:, z], axis=0))
    diameter = np.ma.masked_where(~on_fiber, 2.0 * maps[RADIUS][0, z])
    return [
        (maps[HEAT][0, z], "magma", 0.0, 1.0),
        (maps[FIBER][0, z], "gray", 0.0, 1.0),
        (distance, "viridis", 0.0, None),
        (direction_rgb(maps, z, on_fiber), None, None, None),
        (diameter, "plasma", 0.0, None),
        (maps[BINDER][0, z], "cividis", 0.0, 1.0),
        (maps[BONDPT][0, z], "inferno", 0.0, 1.0),
    ]


COLUMNS = ["CT slice", "axis heatmap", "fiber", "distance to own axis\n(offset, voxels)",
           "direction\n(|x| red, |y| green, |z| blue)", "fiber diameter\n(radius map, voxels)", "binder", "bond points"]


def group_figure(model, files, title, out, device):
    fig = plt.figure(figsize=(2.05 * len(COLUMNS) + 0.5, 4 * 2.15 + 0.7), facecolor="white", layout="constrained")
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    grid = fig.add_gridspec(4, len(COLUMNS))
    for s, path in enumerate(files):
        volume, truth, maps, score = infer(model, path, device)
        z = best_slice(truth)
        print(f"  {path.stem}: slice z={z}, {score}", flush=True)
        low, high = np.percentile(volume[z], [0.5, 99.8])
        rows = [panels(volume, truth, z, True), panels(volume, maps, z, False)]
        limits = {}  # the same scale for truth and prediction, column by column
        for c in (2, 4):
            limits[c] = max(float(np.ma.max(rows[r][c][0])) if np.ma.count(rows[r][c][0]) else 1.0 for r in (0, 1))
        for r, row in enumerate(rows):
            ax = fig.add_subplot(grid[2 * s + r, 0])
            ax.imshow(volume[z], cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            ax.set_ylabel(f"{path.stem}\nz = {z}, truth" if r == 0 else f"network, {score}", fontsize=9)
            if s == 0 and r == 0:
                ax.set_title(COLUMNS[0], fontsize=9)
            for c, (image, cmap, vmin, vmax) in enumerate(row):
                ax_c = fig.add_subplot(grid[2 * s + r, c + 1])
                colormap = plt.get_cmap(cmap).copy() if cmap else None
                if colormap:
                    colormap.set_bad(EMPTY)
                ax_c.imshow(image, cmap=colormap, vmin=vmin, vmax=limits.get(c, vmax), interpolation="nearest")
                if s == 0 and r == 0:
                    ax_c.set_title(COLUMNS[c + 1], fontsize=9)
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out, flush=True)


CHANNEL_NAMES = ["axis heatmap", "offset x", "offset y", "offset z", "direction xx", "direction yy", "direction zz",
                 "direction xy", "direction xz", "direction yz", "fiber", "radius", "binder", "bond points"]


def channel_figure(model, path, out, device):
    """Every one of the 14 numbers per voxel, plain, for one scan (truth above prediction)."""
    volume, truth, maps, score = infer(model, path, device)
    z = best_slice(truth)
    per_row = 7
    fig, axes = plt.subplots(4, per_row, figsize=(2.0 * per_row, 4 * 2.2), facecolor="white", layout="constrained")
    fig.suptitle(f"All 14 output channels, {path.stem}, z = {z} (network: {score})", fontsize=13, x=0.01, ha="left")
    for i, name in enumerate(CHANNEL_NAMES):
        pair = truth[i, z], maps[i, z]
        signed = 1 <= i <= 3 or 7 <= i <= 9
        reach = max(float(np.abs(pair[0]).max()), float(np.percentile(np.abs(pair[1]), 99.8)), 1e-6)
        for r in range(2):
            ax = axes[2 * (i // per_row) + r, i % per_row]
            ax.imshow(pair[r], cmap="RdBu_r" if signed else "magma", vmin=-reach if signed else 0.0, vmax=reach,
                      interpolation="nearest")
            ax.set_title(f"{name}, {'truth' if r == 0 else 'network'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    checkpoint, scans, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load(checkpoint, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    for group, title in GROUPS.items():
        files = sorted((scans / group).glob("*.npz"))[:2]
        print(group, flush=True)
        if files:
            group_figure(model, files, title, out_dir / f"ct-unet-val-{group.replace('_', '-')}.png", device)
    channel_figure(model, sorted((scans / "mixed").glob("*.npz"))[0], out_dir / "ct-unet-val-all-channels.png", device)
