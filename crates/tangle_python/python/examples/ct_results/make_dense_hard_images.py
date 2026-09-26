"""Before/after pictures of ``dense_hard`` structures for the README, written to ``images/``.

For each structure, one z slice (the one with the most fiber) of the scan, the truth, the fit and the difference,
for two ``ct_examples.py --output`` folders run with different settings (a row each)::

    python make_dense_hard_images.py --before ~/runs/start --after ~/runs/recommended 7 11

The scores in the row titles are the centerline recall, precision and F1 from each run's ``score.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tifffile  # noqa: E402

from make_images import DIFF_LEGEND, DPI, HERE, busiest_slice, legend  # noqa: E402


def score(folder: Path) -> str:
    s = json.loads((folder / "score.json").read_text())["score"]
    return f"F1 {s['centerline_f1']:.3f} (recall {s['centerline_recall']:.3f}, precision {s['centerline_precision']:.3f})"


def before_after(before: Path, after: Path, name: str, out: Path) -> None:
    z = busiest_slice(tifffile.imread(after / name / "true.tif"))
    columns = [("raw", "Scan"), ("true", "Truth"), ("segment", "Fit"), ("diff", "Difference")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.4), dpi=DPI, facecolor="white")
    for row, (folder, label) in enumerate([(before, "starting settings"), (after, "recommended settings")]):
        for col, (key, title) in enumerate(columns):
            image = tifffile.imread(folder / name / f"{key}.tif")[z]
            ax = axes[row, col]
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{label}\n{score(folder / name)}", fontsize=10)
    legend(axes[1, 3], DIFF_LEGEND)
    fig.suptitle(f"{name} (slice z = {z})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_before_after.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True, nargs="+", help="one or more folders; the first holding a structure is used")
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("structures", nargs="+", type=int)
    args = p.parse_args()
    out = HERE / "images"
    out.mkdir(exist_ok=True)
    for n in args.structures:
        name = f"dense_hard_{n}"
        before = next(folder for folder in args.before if (folder / name / "raw.tif").exists())
        before_after(before, args.after, name, out)
        print(out / f"dense_hard_{n}_before_after.png")


if __name__ == "__main__":
    main()
