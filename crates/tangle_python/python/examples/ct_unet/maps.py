"""The CT map network: a 3D U-Net from a scan to maps a fiber fitter reads easily.

Output channels, per voxel (vectors in voxel units, components in (x, y, z)
order; arrays in (z, y, x) order like the scans):

* 0: axis heatmap logit. The target is exp(-d^2 / 2 s^2) with d the distance
  to the nearest true axis and s = max(1, 0.3 r): a sharp ridge on every
  centerline, so touching fibers stay separate ridges.
* 1-3: offset from the voxel center to its own fiber's axis.
* 4-9: fiber direction as the sign-free tensor t t^T (xx, yy, zz, xy, xz, yz).
* 10: fiber (vs void) logit.
* 11: log of the local fiber radius (voxels; the equal-area radius of the fiber the voxel belongs to).

No fiber types: the network only finds fibers, whatever their material, and the radius map lets the tracer and
the per-fiber typing (``fiber_types.py``) work for any mix of sizes. The first networks (13 channels: void /
fine / coarse logits in place of 10-11) still load; ``predict`` turns their output into this layout.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HEAT, OFFSET, DIRECTION, FIBER, RADIUS = slice(0, 1), slice(1, 4), slice(4, 10), slice(10, 11), slice(11, 12)
CHANNELS = 12
_OLD_TYPE = slice(10, 13)  # the first networks' void / fine / coarse logits
_OLD_RADIUS = (2.75, 6.75)  # voxels: what "fine" and "coarse" meant for them


def levels(volume: np.ndarray) -> tuple[float, float]:
    """The grey levels normalize maps to 0 and 1: the median (void) and the 99.5th percentile."""
    low, high = np.percentile(volume, [50.0, 99.5])
    return float(low), float(high)


def normalize(volume: np.ndarray, grey: tuple[float, float] | None = None) -> np.ndarray:
    """The scan as float32 with the void near 0 and bright fibers near 1 (robust percentiles).
    Tiles of one scan pass the whole scan's ``grey`` levels, so they all scale alike."""
    v = volume.astype(np.float32)
    low, high = grey or levels(v)
    return (v - np.float32(low)) / np.float32(max(high - low, 1e-6))


def targets(data, window=None) -> dict:
    """Training targets for ``data`` (a make_data npz), optionally in a (z, y, x) slice ``window``."""
    near = data["near"] if window is None else data["near"][window]
    labels = data["labels"] if window is None else data["labels"][window]
    pos, tan, fid, rad, typ = (data[f"p_{k}"] for k in ("pos", "tan", "fid", "rad", "typ"))
    nz, ny, nx = near.shape
    z0, y0, x0 = (0, 0, 0) if window is None else (window[0].start, window[1].start, window[2].start)
    grid = np.stack(np.meshgrid(np.arange(nx) + x0 + 0.5, np.arange(ny) + y0 + 0.5, np.arange(nz) + z0 + 0.5,
                                indexing="xy"), axis=-1)  # (y, x, z, 3) from meshgrid order
    grid = np.transpose(grid, (2, 0, 1, 3))  # (z, y, x, 3) holding (x, y, z)
    has = near >= 0
    i = np.where(has, near, 0)
    offset = pos[i] - grid
    distance = np.linalg.norm(offset, axis=-1)
    sigma = np.maximum(1.0, 0.3 * rad[i])
    heat = np.where(has, np.exp(-0.5 * (distance / sigma) ** 2), 0.0).astype(np.float32)
    # Offsets and directions only where the voxel's nearest axis is its own fiber's.
    own = has & (labels > 0) & (fid[i] == labels)
    t = tan[i]
    direction = np.stack([t[..., 0] ** 2, t[..., 1] ** 2, t[..., 2] ** 2, t[..., 0] * t[..., 1],
                          t[..., 0] * t[..., 2], t[..., 1] * t[..., 2]], axis=0)
    return {
        "heat": heat[None],
        "offset": np.moveaxis(offset, -1, 0).astype(np.float32) * own[None],
        "direction": direction.astype(np.float32) * own[None],
        "own": own[None].astype(np.float32),
        "fiber": (labels > 0)[None].astype(np.float32),
        "radius": (np.log(np.maximum(rad[i], 0.5)) * own)[None].astype(np.float32),
    }


# Vector-component fixes for the 16 xy symmetries and the z flip (arrays (C, z, y, x)).
def flip(sample: dict, axis: int) -> dict:
    """Mirror along x (axis 0), y (1) or z (2) of the (x, y, z) components."""
    array_axis = {0: -1, 1: -2, 2: -3}[axis]
    out = {k: np.flip(v, array_axis).copy() for k, v in sample.items()}
    out["offset"][axis] *= -1
    for c, (a, b) in zip(range(3, 6), [(0, 1), (0, 2), (1, 2)]):
        if axis in (a, b):
            out["direction"][c] *= -1
    return out


def swap_xy(sample: dict) -> dict:
    out = {k: np.swapaxes(v, -1, -2).copy() for k, v in sample.items()}
    out["offset"] = out["offset"][[1, 0, 2]]
    out["direction"] = out["direction"][[1, 0, 2, 3, 5, 4]]
    return out


def augment(sample: dict, rng: np.random.Generator) -> dict:
    if rng.random() < 0.5:
        sample = swap_xy(sample)
    for axis in range(3):
        if rng.random() < 0.5:
            sample = flip(sample, axis)
    return sample


def block(cin, cout, groups=None):
    groups = groups or min(8, cout // 4)
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, base: int = 16, levels: int = 4, channels: int = CHANNELS, group_sizes=None):
        """``group_sizes``: channels per norm group at each level (default: 8 groups, at least 4 channels each);
        a widened network keeps its source's group sizes, so the old channels stay in groups of their own."""
        super().__init__()
        widths = [base * 2**k for k in range(levels + 1)]
        self.group_sizes = list(group_sizes) if group_sizes else None
        g = [w // group_sizes[k] for k, w in enumerate(widths)] if group_sizes else [None] * len(widths)
        self.down = nn.ModuleList([block(1, widths[0], g[0])] + [block(widths[k], widths[k + 1], g[k + 1])
                                                                 for k in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(widths[k + 1], widths[k], 2, stride=2) for k in range(levels)])
        self.merge = nn.ModuleList([block(2 * widths[k], widths[k], g[k]) for k in range(levels)])
        self.head = nn.Conv3d(widths[0], channels, 1)

    def forward(self, x):
        skips = []
        for k, layer in enumerate(self.down):
            x = layer(x if k == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        x = skips.pop()
        for k in reversed(range(len(self.up))):
            x = self.merge[k](torch.cat([self.up[k](x), skips.pop()], dim=1))
        return self.head(x)


def loss_terms(out: torch.Tensor, batch: dict) -> dict:
    heat = batch["heat"]
    weight = 1.0 + 9.0 * (heat > 0.1)
    own = batch["own"]
    n_own = own.sum().clamp(min=1.0)
    fiber = batch["fiber"]
    return {
        "heat": (F.binary_cross_entropy_with_logits(out[:, HEAT], heat, reduction="none") * weight).mean(),
        "offset": (F.smooth_l1_loss(out[:, OFFSET], batch["offset"], reduction="none") * own).sum() / (3 * n_own),
        "direction": ((out[:, DIRECTION] - batch["direction"]) ** 2 * own).sum() / (6 * n_own) * 10.0,
        "fiber": (F.binary_cross_entropy_with_logits(out[:, FIBER], fiber, reduction="none")
                  * (1.0 + 2.0 * fiber)).mean(),
        "radius": (F.smooth_l1_loss(out[:, RADIUS], batch["radius"], reduction="none", beta=0.1) * own).sum() / n_own,
    }


def load(path, device: str) -> nn.Module:
    """A saved network (either layout) on ``device``."""
    state = torch.load(path, map_location=device)
    channels = state["model"]["head.weight"].shape[0]
    model = UNet3D(base=state.get("base", 16), channels=channels, group_sizes=state.get("group_sizes")).to(device)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def predict(model: nn.Module, volume: np.ndarray, device: str, grey: tuple[float, float] | None = None) -> np.ndarray:
    """The network's maps for a whole scan (sides divisible by 16), float32 (CHANNELS, z, y, x): heatmap and
    fiber as probabilities, offsets in voxels, the direction tensor as predicted, the radius in voxels."""
    model.eval()
    x = torch.from_numpy(normalize(volume, grey))[None, None].to(device)
    out = model(x)[0]
    maps = torch.empty((CHANNELS, *out.shape[1:]), device=out.device)
    maps[HEAT] = torch.sigmoid(out[HEAT])
    maps[OFFSET], maps[DIRECTION] = out[OFFSET], out[DIRECTION]
    if out.shape[0] == CHANNELS:
        maps[FIBER] = torch.sigmoid(out[FIBER])
        maps[RADIUS] = torch.exp(out[RADIUS])
    else:  # a first-layout network: fiber = not void, radius from its fine / coarse call
        p = torch.softmax(out[_OLD_TYPE], dim=0)
        maps[FIBER] = 1.0 - p[0:1]
        maps[RADIUS] = torch.where(p[2:3] > p[1:2], _OLD_RADIUS[1], _OLD_RADIUS[0])
    return maps.cpu().numpy()


def widen(small: nn.Module, base: int) -> nn.Module:
    """A wider copy of ``small`` (Net2Net style) that starts out computing nearly the same maps.

    Every layer's trained channels are copied into the first channels of the wider layer; the new channels get
    fresh weights for their own outputs but zero weight wherever they feed an old channel, so on the first step
    they change nothing downstream. Group norm regroups some layers' channels, so the start is close to, not
    exactly, the small network.
    """
    old_w = [small.down[0][0].out_channels * 2**k for k in range(len(small.down))]
    sizes = [w // small.down[k][1].num_groups for k, w in enumerate(old_w)]
    wide = UNet3D(base=base, channels=small.head.out_channels, group_sizes=sizes)
    new_w = [base * 2**k for k in range(len(small.down))]
    small_state, wide_state = small.state_dict(), wide.state_dict()
    for name, target in wide_state.items():
        source = small_state[name]
        if target.shape == source.shape:
            wide_state[name] = source.clone()
            continue
        target = target.clone()
        if target.ndim > 1:
            target[:, source.shape[1]:] = 0.0  # new inputs feed nothing yet
            if name.startswith("merge.") and name.endswith(".0.weight"):
                # the first conv of a merge block reads [upsampled, skip]: both halves move
                k = int(name.split(".")[1])
                o, n = old_w[k], new_w[k]
                target[:, :n * 2] = 0.0
                target[: source.shape[0], :o] = source[:, :o]
                target[: source.shape[0], n:n + o] = source[:, o:]
            elif name.startswith("up.") and name.endswith(".weight"):
                # transposed conv: (in, out, ...)
                target[source.shape[0]:] = 0.0
                target[: source.shape[0], : source.shape[1]] = source
            else:
                target[: source.shape[0], : source.shape[1]] = source
        else:
            target[: source.shape[0]] = source
        wide_state[name] = target
    wide.load_state_dict(wide_state)
    return wide
