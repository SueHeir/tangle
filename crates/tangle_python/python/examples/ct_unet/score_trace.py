"""Centerline F1 of trace_maps fibers on ct_examples structures.

usage: python score_trace.py MAPS CACHE_DIR EXAMPLE... [--min-votes N]

MAPS is ``truth`` (the true maps), a checkpoint (.pt), or a folder of saved
``<example>_maps.npy``. CACHE_DIR holds the truth caches (``.cache``), as the
grey fitter's runs use, so the scans are the same ones.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import ct_examples as ex  # noqa: E402
from tangle.ct._evaluate import _samples_inside, centerline_agreement  # noqa: E402
from trace_maps import trace  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps")
    parser.add_argument("cache", type=Path)
    parser.add_argument("examples", nargs="+")
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--tidy", action="store_true", help="drop doubled traces and join gaps")
    args = parser.parse_args()
    model = None
    if args.maps.endswith(".pt"):
        import torch
        from maps import UNet3D, predict

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        state = torch.load(args.maps, map_location=device)
        model = UNet3D(base=state.get("base", 16)).to(device)
        model.load_state_dict(state["model"])
    scores = []
    for name in args.examples:
        example = ex.EXAMPLES[name](args.cache / f"{name}.json")
        scan = example.scan
        if args.maps.startswith("fit:"):
            maps = None
        elif args.maps == "truth":
            from fit_maps import true_maps
            maps = true_maps(scan)
        elif model is not None:
            maps = predict(model, scan.volume, device)
        else:
            maps = np.load(Path(args.maps) / f"{name}_maps.npy").astype(np.float32)
        specs = example.spec if isinstance(example.spec, list) else [example.spec]
        fine = min(s.diameter for s in specs) / scan.voxel_size
        started = time.perf_counter()
        if args.maps.startswith("fit:"):  # a grey fit's fit.json, to check this scorer against ct.score
            import json
            saved = json.loads((Path(args.maps[4:]) / name / "fit.json").read_text())
            lines = [np.asarray(f["centerline"]) / saved["voxel_size"] for f in saved["fibers"]]
        else:
            coarse = max(s.diameter for s in specs) / scan.voxel_size
            lines, types = trace(maps, min_length=3.0 * fine, min_votes=args.min_votes,
                                 tidy=(0.5 * fine, 0.5 * coarse) if args.tidy else None)
        seconds = time.perf_counter() - started
        shape = scan.volume.shape[::-1]
        # True pieces shorter in the volume than the minimum fit length are stubs, left out as ct.score does.
        stubs = {g for g, line in enumerate(scan.centerlines) if 0.5 * len(_samples_inside(line, shape)) < 3.0 * fine}
        agree = centerline_agreement(lines, scan.centerlines, np.asarray(scan.radii), shape, skip=stubs)
        scores.append(agree["f1"])
        print(f"{name}: true {len(scan.centerlines) - len(stubs)} traced {len(lines)} recall {agree['recall']:.3f} "
              f"precision {agree['precision']:.3f} F1 {agree['f1']:.3f} ({seconds:.1f} s)", flush=True)
    print(f"mean F1 {np.mean(scores):.3f} over {len(scores)}")


if __name__ == "__main__":
    main()
