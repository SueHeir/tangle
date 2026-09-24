"""Pictures of the CT example results for the README, written to ``images/``.

For a few examples, from the files ``ct_examples.py`` writes:

* ``<example>_slices.png``: one z slice (the one with the most fiber) of
  raw.tif, true.tif, segment.tif and diff.tif side by side;
* ``<example>_confidence.png``: the same slice of confidence.tif;
* ``<example>_fibers_3d.png``: the true and the fitted centerlines in 3D,
  colored as in true.tif and segment.tif;
* ``<example>_blur.png`` (with ``--blur SIGMA=FOLDER ...``): the scan and the
  difference on one slice for runs rendered with different blur
  (``ct_examples.py --blur``).

The TIFFs of most examples are not kept in git, so point ``--source`` at a
full ``ct_examples.py --output`` folder. The 3D pictures rebuild each
example's truth from that folder's ``.cache`` (fast once the truth is
cached)::

    python make_images.py --source ~/ct-results
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
SLICES = ["two_types", "varied_3", "varied_4", "single_type", "scenario_dense_crossing"]
CONFIDENCE = ["two_types", "varied_3"]
FIBERS_3D = ["two_types", "varied_3"]
DPI = 110

# As ct_examples.py colors diff.tif and confidence.tif.
DIFF_LEGEND = [("missed", (230, 50, 50)), ("extra", (60, 120, 255)), ("wrong fiber", (255, 200, 0))]
CONFIDENCE_LEGEND = [("sure", (0, 200, 40)), ("unsure", (230, 0, 40))]


def busiest_slice(true_rgb: np.ndarray) -> int:
    """The z slice with the most true fiber (colored voxels in true.tif)."""
    tint = np.abs(true_rgb.astype(np.int16) - true_rgb.mean(axis=-1, keepdims=True).astype(np.int16)).sum(axis=-1)
    return int(np.argmax((tint > 30).reshape(len(tint), -1).sum(axis=1)))


def legend(ax, entries) -> None:
    ax.legend(
        handles=[Patch(facecolor=np.array(color) / 255, label=label) for label, color in entries],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(entries), frameon=False, fontsize=10,
    )


def slices(source: Path, name: str, out: Path) -> int:
    stacks = {key: tifffile.imread(source / name / f"{key}.tif") for key in ("raw", "true", "segment", "diff")}
    z = busiest_slice(stacks["true"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), dpi=DPI, facecolor="white")
    for ax, (key, title) in zip(axes, [("raw", "Scan"), ("true", "Truth"), ("segment", "Fit"), ("diff", "Difference")]):
        image = stacks[key][z]
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    legend(axes[3], DIFF_LEGEND)
    fig.suptitle(f"{name} (slice z = {z})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_slices.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return z


def confidence(source: Path, name: str, out: Path) -> None:
    z = busiest_slice(tifffile.imread(source / name / "true.tif"))
    image = tifffile.imread(source / name / "confidence.tif")[z]
    fig, ax = plt.subplots(figsize=(4.8, 5.0), dpi=DPI, facecolor="white")
    ax.imshow(image, interpolation="nearest")
    ax.set_title(f"{name}: confidence (z = {z})", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    legend(ax, CONFIDENCE_LEGEND)
    fig.tight_layout()
    fig.savefig(out / f"{name}_confidence.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def true_pieces(scan) -> list[tuple[int, np.ndarray]]:
    """Every true centerline's runs inside the scan (with its periodic images), in voxels."""
    from tangle.ct._geometry import resample

    upper = np.array(scan.labels.shape[::-1], dtype=np.float64)
    period = np.asarray(scan.period)
    shifts = [np.array(c) * period for c in itertools.product(*[(-2, -1, 0, 1, 2) if p > 0 else (0,) for p in period])]
    pieces = []
    for label, line in enumerate(scan.centerlines, start=1):
        dense = resample(np.asarray(line, dtype=np.float64), 0.5)
        for shift in shifts:
            moved = dense + shift
            inside = np.all((moved >= 0) & (moved <= upper), axis=1)
            runs = np.split(np.arange(len(moved)), np.flatnonzero(np.diff(inside.astype(int))) + 1)
            pieces += [(label, moved[run]) for run in runs if inside[run[0]] and len(run) >= 2]
    return pieces


def fibers_3d(source: Path, name: str, out: Path) -> None:
    sys.path.insert(0, str(HERE.parent))
    import ct_examples
    from tangle import ct
    from tangle.ct._overlay import fiber_palette

    scan = ct_examples.EXAMPLES[name](source / ".cache" / f"{name}.json").scan
    fit = ct.load_fit(source / name / "fit.json")
    side = np.array(scan.labels.shape[::-1])
    truth_colors = fiber_palette(len(scan.centerlines))
    fit_colors = fiber_palette(len(fit.centerlines))
    fig = plt.figure(figsize=(12, 5.6), dpi=DPI, facecolor="white")
    panels = [
        ("Truth", [(label, line) for label, line in true_pieces(scan)], truth_colors),
        ("Fit", [(k + 1, np.asarray(line)) for k, line in enumerate(fit.centerlines)], fit_colors),
    ]
    for index, (title, lines, colors) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        for label, line in lines:
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=colors[label], linewidth=1.2)
        ax.set_xlim(0, side[0])
        ax.set_ylim(0, side[1])
        ax.set_zlim(0, side[2])
        ax.set_box_aspect(side)
        ax.view_init(elev=22, azim=-58)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_title(f"{title} ({len({label for label, _ in lines})} fibers)", fontsize=13)
    fig.suptitle(f"{name}: centerlines", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_fibers_3d.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def blur_comparison(runs: list[tuple[str, Path]], name: str, out: Path) -> None:
    """Scan and difference on the same slice, one column per blur level."""
    import json

    z = busiest_slice(tifffile.imread(runs[0][1] / name / "true.tif"))
    fig, axes = plt.subplots(2, len(runs), figsize=(3.9 * len(runs), 8.2), dpi=DPI, facecolor="white")
    for column, (sigma, folder) in enumerate(runs):
        score = json.loads((folder / name / "score.json").read_text())["score"]
        for row, key in enumerate(("raw", "diff")):
            ax = axes[row, column]
            image = tifffile.imread(folder / name / f"{key}.tif")[z]
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, column].set_title(f"blur {sigma} voxels", fontsize=13)
        axes[1, column].set_title(
            f"{score['recovered']}/{score['true_fibers_in_volume']} recovered, F1 {score['centerline_f1']:.3f}", fontsize=11
        )
    axes[0, 0].set_ylabel("Scan", fontsize=13)
    axes[1, 0].set_ylabel("Difference", fontsize=13)
    legend(axes[1, len(runs) // 2], DIFF_LEGEND)
    fig.suptitle(f"{name}: the same fibers scanned with less blur (slice z = {z})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_blur.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=HERE, help="a ct_examples.py --output folder")
    parser.add_argument("--out", type=Path, default=HERE / "images")
    parser.add_argument(
        "--blur", nargs="+", metavar="SIGMA=FOLDER", default=None,
        help="output folders of runs with different --blur, for <example>_blur.png (only these are drawn)",
    )
    parser.add_argument("--blur-examples", nargs="+", default=["varied_3", "two_types"])
    args = parser.parse_args()
    if args.blur:
        runs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).expanduser()) for item in args.blur]
        args.out.mkdir(parents=True, exist_ok=True)
        for name in args.blur_examples:
            blur_comparison(runs, name, args.out)
            print(f"{name}_blur.png")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SLICES:
        print(f"{name}_slices.png (z = {slices(args.source, name, args.out)})")
    for name in CONFIDENCE:
        confidence(args.source, name, args.out)
        print(f"{name}_confidence.png")
    for name in FIBERS_3D:
        fibers_3d(args.source, name, args.out)
        print(f"{name}_fibers_3d.png")


if __name__ == "__main__":
    main()
