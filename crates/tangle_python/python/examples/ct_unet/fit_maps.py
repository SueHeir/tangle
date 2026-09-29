"""Fit ct_examples structures from the network's maps instead of the grey scan.

The network turns the scan into a clean image, which goes to ``ct.fit_fibers``
in place of the scan; the fit is scored against the truth as usual. Every
ct_examples option still applies (``--set``, ``--output``, ...).

usage: python fit_maps.py CHECKPOINT IMAGE EXAMPLE... [ct_examples options]

IMAGE is how the maps become the fitted image:

* ``heat``: the axis heatmap, fine fibers at grey 1000 and coarse at 600.
* ``cone``: each fiber voxel at its type's grey times 1 - (d / r)^2, with d
  the predicted distance to its axis and r its type's radius, so touching
  fibers meet at a dark seam.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import ct_examples as ex  # noqa: E402
from maps import HEAT, OFFSET, TYPE, UNet3D, predict  # noqa: E402

GREY = {1: 1000.0, 2: 600.0}  # fine, coarse
RANGES = {1: (800.0, 1400.0), 2: (400.0, 799.0)}
VOID = 100.0


def clean_image(maps: np.ndarray, kind: str, radii: dict) -> np.ndarray:
    types = np.argmax(maps[TYPE], axis=0)
    image = np.full(types.shape, VOID, dtype=np.float32)
    for t, grey in GREY.items():
        where = types == t
        if kind == "heat":
            value = maps[HEAT][0]
        else:
            distance = np.linalg.norm(maps[OFFSET], axis=0)
            value = np.clip(1.0 - (distance / radii[t]) ** 2, 0.0, 1.0)
        image[where] = VOID + (grey - VOID) * value[where]
    return image


def true_maps(scan) -> np.ndarray:
    from make_data import nearest_points, point_table
    from maps import CHANNELS, DIRECTION, targets

    table = point_table(scan)
    data = {"near": nearest_points(scan.volume.shape, table["pos"]), "labels": scan.labels,
            **{f"p_{k}": v for k, v in table.items()}}
    t = targets(data)
    maps = np.zeros((CHANNELS, *scan.volume.shape), dtype=np.float32)
    maps[HEAT], maps[OFFSET], maps[DIRECTION] = t["heat"], t["offset"], t["direction"]
    for c in range(3):
        maps[TYPE.start + c] = t["type"] == c
    return maps


def main() -> None:
    checkpoint, kind, *rest = sys.argv[1:]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if checkpoint != "truth":  # "truth": the true maps, to test the fitting step alone
        state = torch.load(checkpoint, map_location=device)
        model = UNet3D(base=state.get("base", 16)).to(device)
        model.load_state_dict(state["model"])

    def fit_input(scan, spec, input="grey"):
        specs = spec if isinstance(spec, list) else [spec]
        maps = true_maps(scan) if checkpoint == "truth" else predict(model, scan.volume, device)
        if output:  # the maps, to look at next to the fit
            np.save(output / f"{name}_maps.npy", maps.astype(np.float16))
        h = scan.voxel_size
        # Types by diameter: the thinnest spec is fine (grey 1000), the next coarse.
        order = sorted(range(len(specs)), key=lambda k: specs[k].diameter)
        radii = {1 + rank: 0.5 * specs[k].diameter / h for rank, k in enumerate(order)}
        image = clean_image(maps, kind, radii)
        out = [None] * len(specs)
        for rank, k in enumerate(order):
            out[k] = specs[k].replace(intensity=RANGES[1 + rank])
        seen = np.clip((image - VOID) / (GREY[1] - VOID), 0.0, 1.0)
        return image, out if isinstance(spec, list) else out[0], seen

    name = rest[0]
    output = Path(rest[rest.index("--output") + 1]) if "--output" in rest else None
    ex.fit_input = fit_input
    sys.argv = [sys.argv[0], *rest]
    ex.main()


if __name__ == "__main__":
    main()
