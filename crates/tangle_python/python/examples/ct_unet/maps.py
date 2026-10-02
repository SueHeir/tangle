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

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HEAT, OFFSET, DIRECTION, FIBER, RADIUS = slice(0, 1), slice(1, 4), slice(4, 10), slice(10, 11), slice(11, 12)
BINDER = slice(12, 13)  # binder (bond) voxel logit
BONDPT = slice(13, 14)  # bond-point heatmap logit: a peak at each bond's center (as HEAT is for fiber axes)
BOND_SIGMA = 1.5  # voxels: width of the bond-point peaks
BINDER_POSITIVE = 9.0  # extra loss weight on true binder voxels (binder is ~1% of the voxels)
SIZE_BINS = np.linspace(np.log(3.0), np.log(32.0), 16)  # log-diameter bins (voxels) of the size code
SIZE_CODE = len(SIZE_BINS) + 1 + 3  # the first hints: the bins, 1 = sizes known, then binder (known, present, size)
TYPE_SLOTS = 4  # fiber types described one by one, thinnest first
SLOT = 6  # per type: given, log diameter, shape known, flatness, hollow known, hollow
EXTRAS = ("broken", "dust", "voids")  # things besides fibers and binder, each yes / no / not said
HINT_CODE = SIZE_CODE + TYPE_SLOTS * SLOT + 2 * len(EXTRAS)  # every hint (networks from before take the first 20)


def size_code(diameters=None, bonds: bool | None = None, bond_ratio: float | None = None, *, ratios=None,
              hollow=None, broken: bool | None = None, dust: bool | None = None,
              voids: bool | None = None) -> np.ndarray:
    """What is known about the scan, as the network's conditioning vector. Every part is optional: None (or an
    empty list) means "not said", and all zeros (nothing said) is a case the network trains on too.

    ``diameters``: the fiber types' diameters (voxels; equal-area for ovals), as a soft histogram over log
    diameter (each a Gaussian bump 0.1 wide in log) plus a "known" flag, and (the thinnest four) one by one.
    ``ratios``: per type, the thickness over the width of its section (1 = round, 0.5 = twice as wide as thick),
    and ``hollow``: per type, True for fibers with an empty core; both in the order of ``diameters``, with None
    for a type nobody says. ``bonds``: True (there is binder: bonds, webs or blobs), False (no binder) or None;
    ``bond_ratio``: the rough bond radius over the thinner fiber's radius, if known. ``broken``, ``dust``,
    ``voids``: True (the sample has short broken fiber pieces / dust particles / voids), False or None.

    The first SIZE_CODE entries are the hints of the first conditioned networks, unchanged, so those networks
    (which read only that much) see exactly what they were trained with."""
    code = np.zeros(HINT_CODE, np.float32)
    n = len(SIZE_BINS)
    if diameters is not None and len(diameters):
        for d in diameters:
            code[:n] = np.maximum(code[:n], np.exp(-0.5 * ((SIZE_BINS - np.log(d)) / 0.1) ** 2))
        code[n] = 1.0
        order = np.argsort(diameters, kind="stable")[:TYPE_SLOTS]
        for slot, t in enumerate(order):
            k = SIZE_CODE + slot * SLOT
            code[k] = 1.0
            code[k + 1] = (np.log(diameters[t]) - SIZE_BINS[0]) / (SIZE_BINS[-1] - SIZE_BINS[0])
            if ratios is not None and ratios[t] is not None:
                code[k + 2], code[k + 3] = 1.0, 1.0 - float(ratios[t])
            if hollow is not None and hollow[t] is not None:
                code[k + 4], code[k + 5] = 1.0, float(bool(hollow[t]))
    if bonds is not None:
        code[n + 1] = 1.0
        code[n + 2] = 1.0 if bonds else 0.0
        if bonds and bond_ratio:
            code[n + 3] = float(bond_ratio)
    for e, said in enumerate((broken, dust, voids)):
        if said is not None:
            k = SIZE_CODE + TYPE_SLOTS * SLOT + 2 * e
            code[k], code[k + 1] = 1.0, float(bool(said))
    return code


def scan_hints(data) -> dict:
    """Everything a user could truthfully say about a make_data scan (``size_code``'s arguments, with the
    diameters as each type's median equal-area diameter). Scans from before the hints were stored carry no
    hollow fibers, broken pieces, dust or voids, which is what they say."""
    files = data.files
    typ, rad, fid = data["p_typ"], data["p_rad"], data["p_fid"]
    if "type_info" in files:  # per type: diameter, thickness / width, hollow
        info = np.asarray(data["type_info"], np.float64).reshape(-1, 3)
        diameters, ratios, hollow = info[:, 0].tolist(), info[:, 1].tolist(), (info[:, 2] > 0.5).tolist()
    else:
        kinds = np.unique(typ)
        diameters = [2.0 * float(np.median(rad[typ == t])) for t in kinds]
        ratios, hollow = [1.0] * len(kinds), [False] * len(kinds)
        if "semi_axes" in files:  # one row per fiber (one-based ids, as p_fid)
            semi = np.asarray(data["semi_axes"], np.float64).reshape(-1, 2)
            ids, first = np.unique(fid, return_index=True)
            if len(ids) and ids.max() <= len(semi):
                ratio = semi[ids - 1].min(1) / np.maximum(semi[ids - 1].max(1), 1e-9)
                ratios = [round(float(np.median(ratio[typ[first] == t])), 3) for t in kinds]
    bonded = "bond_labels" in files and bool(data["bond_labels"].any())
    binder = bool(data["binder_mask"].any()) if "binder_mask" in files else bonded
    flags = np.asarray(data["hint_flags"]).astype(bool).tolist() if "hint_flags" in files else [False] * 3
    return {"diameters": diameters, "ratios": ratios, "hollow": hollow, "bonds": binder,
            "bond_ratio": float(data["bond_ratio"]) if bonded and "bond_ratio" in files else None,
            **dict(zip(EXTRAS, flags))}


def draw_hints(truth: dict, rng: np.random.Generator, blank: float = 0.35, sizes: float = 0.6,
               binder: float = 0.5, extra: float = 0.5) -> dict:
    """``size_code`` arguments for one training crop: what a user might say about the scan, never wrong but
    often incomplete. With chance ``blank`` nothing at all; otherwise each part on its own: the diameters
    (``sizes``; each off by ~10%, as a user's estimate is), and with them the section shapes and hollowness
    (``extra`` each), the binder (``binder``; with the bond size half the time when bonded), and each of
    broken pieces, dust and voids (``extra``)."""
    if rng.random() < blank:
        return {}
    said = {}
    if rng.random() < sizes:
        said["diameters"] = [d * float(np.exp(rng.normal(0.0, 0.1))) for d in truth["diameters"]]
        if rng.random() < extra:
            said["ratios"] = [float(np.clip(r + rng.normal(0.0, 0.05), 0.1, 1.0)) if r < 0.97 else 1.0
                              for r in truth["ratios"]]
        if rng.random() < extra:
            said["hollow"] = list(truth["hollow"])
    if rng.random() < binder:
        said["bonds"] = truth["bonds"]
        if truth["bonds"] and truth["bond_ratio"] and rng.random() < 0.5:
            said["bond_ratio"] = truth["bond_ratio"] * float(np.exp(rng.normal(0.0, 0.15)))
    for key in EXTRAS:
        if rng.random() < extra:
            said[key] = truth[key]
    return said
CHANNELS = 14
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


def _binder(data, window, shape):
    """The binder target: all binder (bonds and coatings) when stored, else the bond voxels, else none."""
    for key in ("binder_mask", "bond_labels"):
        if key in data:
            a = data[key] if window is None else data[key][window]
            return (a > 0)[None].astype(np.float32)
    return np.zeros((1, *shape), np.float32)


def _bond_points(data, window, shape):
    """Gaussian peaks (BOND_SIGMA) at the true bond centers inside ``window``."""
    out = np.zeros(shape, np.float32)
    if "bond_centers" not in data or not len(data["bond_centers"]):
        return out[None]
    z0, y0, x0 = (0, 0, 0) if window is None else (window[0].start, window[1].start, window[2].start)
    reach = int(np.ceil(3 * BOND_SIGMA))
    for c in np.asarray(data["bond_centers"], np.float64):
        p = c - np.array([x0, y0, z0]) - 0.5  # voxel index (x, y, z) in the window
        lo = np.maximum(np.floor(p).astype(int) - reach, 0)
        hi = np.minimum(np.floor(p).astype(int) + reach + 2, np.array(shape[::-1]))
        if np.any(hi <= lo):
            continue
        z, y, x = np.mgrid[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
        g = np.exp(-0.5 * ((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) / BOND_SIGMA**2)
        box = out[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
        np.maximum(box, g, out=box)
    return out[None]


def targets(data, window=None) -> dict:
    """Training targets for ``data`` (a make_data npz), optionally in a (z, y, x) slice ``window``.

    In float32, with the voxel centres broadcast along each axis rather than built as a (z, y, x, 3) grid: this
    is most of the CPU time of a training crop."""
    near = data["near"] if window is None else data["near"][window]
    labels = data["labels"] if window is None else data["labels"][window]
    pos, tan, fid, rad = (data[f"p_{k}"] for k in ("pos", "tan", "fid", "rad"))
    nz, ny, nx = near.shape
    z0, y0, x0 = (0, 0, 0) if window is None else (window[0].start, window[1].start, window[2].start)
    has = near >= 0
    i = np.where(has, near, 0)
    p = pos.astype(np.float32, copy=False)[i]
    offset = np.empty((3, nz, ny, nx), np.float32)  # (x, y, z) components
    np.subtract(p[..., 0], np.arange(nx, dtype=np.float32) + np.float32(x0 + 0.5), out=offset[0])
    np.subtract(p[..., 1], (np.arange(ny, dtype=np.float32) + np.float32(y0 + 0.5))[:, None], out=offset[1])
    np.subtract(p[..., 2], (np.arange(nz, dtype=np.float32) + np.float32(z0 + 0.5))[:, None, None], out=offset[2])
    r = rad.astype(np.float32, copy=False)[i]
    sigma = np.maximum(np.float32(1.0), np.float32(0.3) * r)
    heat = np.exp(np.float32(-0.5) * (offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2) / (sigma * sigma))
    heat[~has] = 0.0
    # Offsets and directions only where the voxel's nearest axis is its own fiber's.
    own = (has & (labels > 0) & (fid[i] == labels)).astype(np.float32)
    t = tan.astype(np.float32, copy=False)[i]
    direction = np.stack([t[..., 0] ** 2, t[..., 1] ** 2, t[..., 2] ** 2, t[..., 0] * t[..., 1],
                          t[..., 0] * t[..., 2], t[..., 1] * t[..., 2]], axis=0)
    offset *= own
    direction *= own
    return {
        "heat": heat[None],
        "offset": offset,
        "direction": direction,
        "own": own[None],
        "fiber": (labels > 0)[None].astype(np.float32),
        "binder": _binder(data, window, labels.shape),
        "bondpt": _bond_points(data, window, labels.shape),
        "radius": (np.log(np.maximum(r, np.float32(0.5))) * own)[None],
    }


# Vector-component fixes for the 16 xy symmetries and the z flip (arrays (C, z, y, x)).
def flip(sample: dict, axis: int) -> dict:
    """Mirror along x (axis 0), y (1) or z (2) of the (x, y, z) components."""
    array_axis = {0: -1, 1: -2, 2: -3}[axis]
    out = {k: np.flip(v, array_axis) for k, v in sample.items()}  # views: the caller makes them contiguous once
    out["offset"][axis] *= -1
    for c, (a, b) in zip(range(3, 6), [(0, 1), (0, 2), (1, 2)]):
        if axis in (a, b):
            out["direction"][c] *= -1
    return out


def swap_xy(sample: dict) -> dict:
    out = {k: np.swapaxes(v, -1, -2) for k, v in sample.items()}  # views, as in flip
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
    def __init__(self, base: int = 16, levels: int = 4, channels: int = CHANNELS, group_sizes=None,
                 condition: bool = False, code_size: int = HINT_CODE):
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
        # Optional conditioning on what the user says about the scan (``size_code``): a small network turns the
        # code into a per-channel scale and shift after every block (FiLM). Its last layer starts at zero, so a
        # conditioned network starts out computing exactly what the network it was made from computes.
        self.condition = condition
        if condition:
            self.film_widths = [widths[k] for k in range(levels + 1)] + [widths[k] for k in range(levels)]
            self.film = nn.Sequential(nn.Linear(code_size, 64), nn.SiLU(), nn.Linear(64, 2 * sum(self.film_widths)))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(self, x, code=None):
        films = None
        if self.condition:
            if code is None:
                code = torch.zeros((x.shape[0], self.film[0].in_features), device=x.device)
            films = list(torch.split(self.film(code), [2 * w for w in self.film_widths], dim=1))

        def apply(y, i):
            if films is None:
                return y
            scale, shift = films[i].chunk(2, dim=1)
            return y * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]

        skips = []
        for k, layer in enumerate(self.down):
            x = apply(layer(x if k == 0 else F.max_pool3d(x, 2)), k)
            skips.append(x)
        x = skips.pop()
        for k in reversed(range(len(self.up))):
            x = apply(self.merge[k](torch.cat([self.up[k](x), skips.pop()], dim=1)), len(self.down) + k)
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
        "binder": (F.binary_cross_entropy_with_logits(out[:, BINDER], batch["binder"], reduction="none")
                   * (1.0 + BINDER_POSITIVE * batch["binder"])).mean(),
        "bondpt": (F.binary_cross_entropy_with_logits(out[:, BONDPT], batch["bondpt"], reduction="none")
                   * (1.0 + 19.0 * (batch["bondpt"] > 0.1))).mean(),
    }


def load(path, device: str) -> nn.Module:
    """A saved network (any layout) on ``device``; ``model.layout`` says which (see ``predict``)."""
    state = torch.load(path, map_location=device)
    channels = state["model"]["head.weight"].shape[0]
    film = [k for k in state["model"] if k.startswith("film.")]
    model = UNet3D(base=state.get("base", 16), channels=channels, group_sizes=state.get("group_sizes"),
                   condition=bool(film), code_size=state["model"]["film.0.weight"].shape[1] if film else SIZE_CODE
                   ).to(device)
    model.load_state_dict(state["model"])
    model.layout = {14: "bondpoints"}.get(channels) or state.get("layout") or {13: "types", 12: "fibers"}[channels]
    return model


@torch.no_grad()
def predict(model: nn.Module, volume: np.ndarray, device: str, grey: tuple[float, float] | None = None,
            diameters=None, bonds: bool | None = None, bond_ratio: float | None = None, **hints) -> np.ndarray:
    """The network's maps for a whole scan (sides divisible by 16), float32 (CHANNELS, z, y, x): heatmap, fiber
    and binder as probabilities (binder 0 for networks without it), offsets in voxels, the direction tensor as
    predicted, the radius in voxels. ``diameters``, ``bonds``, ``bond_ratio`` and ``hints`` (``ratios``,
    ``hollow``, ``broken``, ``dust``, ``voids``) are the optional hints of ``size_code``; a network trained
    before a hint existed does not read it."""
    model.eval()
    x = torch.from_numpy(normalize(volume, grey))[None, None].to(device)
    if getattr(model, "condition", False):
        code = size_code(diameters, bonds, bond_ratio, **hints)[: model.film[0].in_features]
        out = model(x, torch.from_numpy(code)[None].to(device))[0]
    else:
        out = model(x)[0]
    maps = torch.zeros((CHANNELS, *out.shape[1:]), device=out.device)
    maps[HEAT] = torch.sigmoid(out[HEAT])
    maps[OFFSET], maps[DIRECTION] = out[OFFSET], out[DIRECTION]
    layout = getattr(model, "layout", "bondpoints")
    if layout in ("fibers", "bonds", "bondpoints"):
        maps[FIBER] = torch.sigmoid(out[FIBER])
        maps[RADIUS] = torch.exp(out[RADIUS])
        if layout in ("bonds", "bondpoints"):
            maps[BINDER] = torch.sigmoid(out[BINDER])
        if layout == "bondpoints":
            maps[BONDPT] = torch.sigmoid(out[BONDPT])
    else:  # a first-layout network: fiber = not void, radius from its fine / coarse call
        p = torch.softmax(out[_OLD_TYPE], dim=0)
        maps[FIBER] = 1.0 - p[0:1]
        maps[RADIUS] = torch.where(p[2:3] > p[1:2], _OLD_RADIUS[1], _OLD_RADIUS[0])
    return maps.cpu().numpy()


def widen(small: nn.Module, base: int) -> nn.Module:
    """A wider copy of ``small`` (Net2Net style) that starts out computing the same maps.

    Every layer's trained channels are copied into the first channels of the wider layer; the new channels get
    fresh weights for their own outputs but zero weight wherever they feed an old channel, so on the first step
    they change nothing downstream. Each level keeps its norm groups' size, so the old channels stay in groups of
    their own; ``base`` must keep every level a whole number of groups (a ValueError names widths that do). A
    hint-conditioned network stays conditioned: its hint network is copied and every old channel keeps its scale
    and shift, while the new channels start with none. It keeps ``small``'s outputs and hint inputs: grow those
    first (train.py --init does) to widen a network from before the newer outputs or hints.
    """
    levels = len(small.down)
    old_w = [small.down[0][0].out_channels * 2**k for k in range(levels)]
    sizes = [w // small.down[k][1].num_groups for k, w in enumerate(old_w)]
    new_w = [base * 2**k for k in range(levels)]
    if base <= old_w[0] or any(n % size for n, size in zip(new_w, sizes)):
        step = math.lcm(*(size // math.gcd(size, 2**k) for k, size in enumerate(sizes)))
        fits = [(old_w[0] // step + 1 + i) * step for i in range(3)]
        raise ValueError(f"can't widen base {old_w[0]} to {base}: its norm groups ({', '.join(map(str, sizes))} "
                         f"channels, level by level) need a wider base that is a multiple of {step}, e.g. "
                         f"{', '.join(map(str, fits))}")
    condition = getattr(small, "condition", False)
    wide = UNet3D(base=base, channels=small.head.out_channels, group_sizes=sizes, condition=condition,
                  code_size=small.film[0].in_features if condition else HINT_CODE)
    small_state, wide_state = small.state_dict(), wide.state_dict()
    for name, target in wide_state.items():
        source = small_state[name]
        if target.shape == source.shape:
            wide_state[name] = source.clone()
            continue
        target = target.clone()
        if name.startswith("film."):
            # the hint network's last layer: per block, a scale row for each channel, then a shift row for each;
            # the old channels keep theirs, the new ones start at zero (left as they are)
            target.zero_()
            old_at = new_at = 0
            for o, n in zip(small.film_widths, wide.film_widths):
                target[new_at:new_at + o] = source[old_at:old_at + o]
                target[new_at + n:new_at + n + o] = source[old_at + o:old_at + 2 * o]
                old_at, new_at = old_at + 2 * o, new_at + 2 * n
        elif target.ndim > 1:
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
