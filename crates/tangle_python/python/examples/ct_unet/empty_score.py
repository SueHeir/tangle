"""False fibers on fiber-free scans (empty.py): what a network finds where there is nothing.

usage: python empty_score.py CHECKPOINT FILE.npz... [--sizes]

Each scan is scaled by its stored greys, as training scales it. Per scan: traced fibers and their total length
(voxels), and the share of voxels the network calls fiber (p > 0.5) and binder, apart for the black regions and
the rest. ``--sizes`` tells the network the fiber types' diameters (the scan's type_info), as a user would;
without it, no hints. Traces shorter than 3 x the thinnest type's diameter are dropped, as score_trace does.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from maps import BINDER, FIBER, load, predict  # noqa: E402
from trace_maps import trace  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--sizes", action="store_true", help="give the network the fiber types' diameters")
    parser.add_argument("--min-votes", type=int, default=3)
    args = parser.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load(args.checkpoint, device)
    rows = []
    for name in args.files:
        data = np.load(name)
        volume, grey = data["volume"], tuple(float(g) for g in data["grey"])
        diameters = [float(d) for d in np.asarray(data["type_info"]).reshape(-1, 3)[:, 0]]
        started = time.perf_counter()
        maps = predict(model, volume, device, grey, diameters=diameters if args.sizes else None)
        lines, _ = trace(maps, min_length=3.0 * min(diameters), min_votes=args.min_votes)
        length = float(sum(np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for line in lines))
        black = volume == 0
        fiber, binder = maps[FIBER][0] > 0.5, maps[BINDER][0] > 0.5
        row = {"traces": len(lines), "length": length,
               "fiber": float(fiber[~black].mean()) if (~black).any() else 0.0,
               "fiber_black": float(fiber[black].mean()) if black.any() else 0.0,
               "binder": float(binder[~black].mean()) if (~black).any() else 0.0,
               "binder_black": float(binder[black].mean()) if black.any() else 0.0,
               "black": float(black.mean())}
        rows.append(row)
        print(f"{Path(name).stem}: {row['traces']} traces, {length:.0f} vox long; fiber {row['fiber']:.4f} "
              f"(black {row['fiber_black']:.4f}), binder {row['binder']:.4f} (black {row['binder_black']:.4f}); "
              f"black share {row['black']:.2f} ({time.perf_counter() - started:.1f} s)", flush=True)
    clean = sum(r["traces"] == 0 for r in rows)
    print(f"mean over {len(rows)}: {np.mean([r['traces'] for r in rows]):.2f} traces, "
          f"{np.mean([r['length'] for r in rows]):.0f} vox; fiber {np.mean([r['fiber'] for r in rows]):.4f}, "
          f"in black {np.mean([r['fiber_black'] for r in rows if r['black'] > 0] or [0]):.4f}; "
          f"binder {np.mean([r['binder'] for r in rows]):.4f}; {clean}/{len(rows)} scans with no traces")


if __name__ == "__main__":
    main()
