"""The guide's diagram of the network, with its sizes read from a saved checkpoint.

usage: python make_architecture_figure.py CHECKPOINT OUT.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

CROP = 128  # voxels: a training crop's side, for the sizes on the diagram
INK, MUTED, LINE = "#1f2328", "#5b6470", "#7a8390"
ENCODER, DECODER, HINTS, HEAD = "#dbe7f5", "#dcefe3", "#f6e6cf", "#ece4f4"
OUTPUTS = [("axis heatmap", 1, "a peak on every centerline"), ("offset", 3, "voxel to its own fiber's axis"),
           ("direction", 6, "sign-free tangent tensor t tᵀ"), ("fiber", 1, "fiber or void"),
           ("radius", 1, "log radius of the voxel's fiber"), ("binder", 1, "binder or not"),
           ("bond points", 1, "a peak at each bond's center")]


def box(ax, x, y, w, h, text, color, size=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=color, edgecolor=LINE, linewidth=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)


def arrow(ax, start, end, label=None, color=LINE, style="-|>", at=(0.0, 0.0), **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, color=color, linewidth=1.1,
                                 shrinkA=0, shrinkB=0, **kwargs))
    if label:
        ax.text((start[0] + end[0]) / 2 + at[0], (start[1] + end[1]) / 2 + at[1], label, fontsize=8.5, color=MUTED,
                ha="center", va="center")


checkpoint, out = sys.argv[1:3]
state = torch.load(checkpoint, map_location="cpu")
weights = state["model"]
levels = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("down."))
widths = [weights[f"down.{k}.0.weight"].shape[0] for k in range(levels)]
outputs = weights["head.weight"].shape[0]
assert outputs == sum(n for _, n, _ in OUTPUTS)
hints = weights["film.0.weight"].shape[1]
film_hidden = weights["film.0.weight"].shape[0]
film_out = weights[next(k for k in weights if k.startswith("film.") and k.endswith(".weight") and k != "film.0.weight")].shape[0]
total = sum(v.numel() for v in weights.values() if v.dtype.is_floating_point)
groups = state.get("group_sizes")

fig, ax = plt.subplots(figsize=(14.5, 8.6), facecolor="white", layout="constrained")
ax.set_xlim(-1.0, 16.5)
ax.set_ylim(-1.75, 8.1)
ax.axis("off")
ax.text(-0.95, 7.85, f"The CT map network: a 3D U-Net with hint conditioning ({total / 1e6:.1f} million weights)",
        fontsize=14, color=INK, ha="left", va="center")

step_y, left, right, w, h = 1.42, 1.8, 9.2, 2.75, 0.92
y_of = [6.6 - k * step_y for k in range(levels)]
x_enc = [left + 0.35 * k for k in range(levels)]
x_dec = [right - 0.35 * k for k in range(levels)]
size = [CROP // 2**k for k in range(levels)]

# Input
box(ax, -0.3, y_of[0], 1.15, h, f"CT scan\n1 × {CROP}³\n(grey, scaled)", "#eeeeee", 8.8)
arrow(ax, (0.275, y_of[0]), (x_enc[0] - w / 2, y_of[0]))

# Encoder, bottleneck, decoder
for k in range(levels):
    label = "bottleneck" if k == levels - 1 else f"encoder {k + 1}"
    x = (x_enc[k] + x_dec[k]) / 2 if k == levels - 1 else x_enc[k]
    box(ax, x, y_of[k], w, h, f"{label}\n{widths[k]} channels × {size[k]}³", ENCODER)
    if k < levels - 1:
        box(ax, x_dec[k], y_of[k], w, h, f"decoder {k + 1}\n{2 * widths[k]} → {widths[k]} channels × {size[k]}³", DECODER)
        arrow(ax, (x_enc[k] + w / 2, y_of[k]), (x_dec[k] - w / 2, y_of[k]), "skip: concatenated",
              at=(0.0, 0.13), linestyle=(0, (4, 3)))
        x_next = x_enc[k + 1] if k < levels - 2 else (x_enc[k + 1] + x_dec[k + 1]) / 2 - 0.75
        arrow(ax, (x_enc[k], y_of[k] - h / 2), (x_next, y_of[k + 1] + h / 2), "max pool ÷2",
              at=(-0.62, 0.0) if k < levels - 2 else (-1.05, -0.12))
        x_prev = x_dec[k + 1] if k < levels - 2 else (x_enc[k + 1] + x_dec[k + 1]) / 2 + 0.75
        arrow(ax, (x_prev, y_of[k + 1] + h / 2), (x_dec[k], y_of[k] - h / 2), "transposed\nconv ×2",
              at=(0.62, 0.0) if k < levels - 2 else (1.05, -0.12))

# Head and outputs
x_head = 11.5
box(ax, x_head, y_of[0], 0.95, h, f"1×1×1\nconv\n{widths[0]} → {outputs}", HEAD, 8.8)
arrow(ax, (x_dec[0] + w / 2, y_of[0]), (x_head - 0.475, y_of[0]))
y_out = y_of[0] + 0.6
channel = 0
rows = []
for name, count, meaning in OUTPUTS:
    span = f"{channel}" if count == 1 else f"{channel}-{channel + count - 1}"
    rows.append((name, count, meaning, span))
    channel += count
for i, (name, count, meaning, span) in enumerate(rows):
    y = y_out - i * 0.62
    box(ax, 13.25, y, 1.42, 0.5, f"{name} ({count})", HEAD, 8.8)
    ax.text(14.05, y + 0.09, f"channel{'' if count == 1 else 's'} {span}", fontsize=8, color=MUTED, va="center")
    ax.text(14.05, y - 0.12, meaning, fontsize=8, color=MUTED, va="center")
    arrow(ax, (x_head + 0.475, y_of[0]), (12.54, y), connectionstyle="arc3,rad=0.0")

# Hints (FiLM)
box(ax, 14.2, 0.15, 2.55, 1.0, f"optional hints: {hints} numbers\nsizes, section shapes, hollow,\nbinder, broken, dust, voids", HINTS, 8.8)
box(ax, 14.2, -1.05, 2.55, 0.8, f"small network: {hints} → {film_hidden} → {film_out}\n(linear, SiLU, linear)", HINTS, 8.8)
arrow(ax, (14.2, -0.35), (14.2, -0.65))
arrow(ax, (12.92, -1.05), (9.4, -1.05), color="#b9863c")
ax.text(5.5, -1.05, "FiLM: after every encoder, bottleneck and decoder block, each channel is scaled and shifted:\n"
        "y → y (1 + scale) + shift, by that block's own numbers. All zeros (nothing said) is trained on too.",
        fontsize=9, color=INK, ha="center", va="center", linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=HINTS, edgecolor=LINE, linewidth=0.9))

# What a block is
group_note = f"; {'/'.join(map(str, groups))} channels per norm group by level" if groups else ""
ax.text(-0.95, 0.25, "Every encoder and decoder block: two rounds of 3×3×3 convolution, group norm, SiLU" + group_note + ".\n"
        f"Sizes shown are for a {CROP}³ training crop; any volume with sides divisible by 16 goes through in one go.",
        fontsize=9, color=MUTED, ha="left", va="center", linespacing=1.3)

fig.savefig(out, dpi=150)
print("wrote", out, f"levels {levels}, widths {widths}, outputs {outputs}, hints {hints}, weights {total:,}")
