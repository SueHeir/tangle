"""A first look at a CT scan before finding its fibers: what the file holds, its grey levels and, given the
network, the fiber sizes in its middle.

    python check_scan.py SCAN [--voxel-size 1.3um] [--weights NETWORK.pt]

It prints the scan's size, type and voxel size (and where that came from), its grey levels and anything
that breaks the network's assumptions (fibers darker than the space around them, a mostly solid scan, an
empty border outside the reconstructed field of view), and saves <scan>_check.png: the middle slice each way
and the grey histogram.

With ``--weights`` it also runs the network, told nothing, on a cube from the middle of the scan (``--size``,
192 voxels), traces the fibers there and groups their diameters into types: the numbers to give
find_fibers.py as ``--diameters``. It ends with the find_fibers.py command to try next.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np

import find_fibers
import fiber_outputs
import scan_io
from scan_io import ScanError

say = find_fibers.say


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_scan.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="A first look at a CT scan: size, voxel size, grey levels and, with --weights, the fiber sizes.",
        epilog="examples:\n  python check_scan.py scan.tif\n  python check_scan.py slices/ --voxel-size 2.5um "
               "--weights best.pt\n\nreads " + scan_io.FORMATS)
    scan_io.add_input_arguments(parser)
    parser.add_argument("--weights", metavar="NETWORK.pt", help="also measure the fibers in the middle of the scan")
    parser.add_argument("--size", type=int, default=192, metavar="N", help="side of the cube the network looks at "
                        "(default 192 voxels)")
    parser.add_argument("--bin", type=int, default=1, metavar="K", help="average K x K x K voxels into one first, "
                        "as find_fibers.py --bin")
    parser.add_argument("--invert", action="store_true", help="the fibers are darker than the space around them")
    parser.add_argument("--types", type=int, metavar="K", help="how many fiber types to group the fibers into "
                        "(default: guessed, 1 to 4)")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="where the network runs (auto: a GPU when there is one)")
    parser.add_argument("--out", metavar="FOLDER", help="where the pictures go (default: the current folder)")
    return parser


def main(argv=None) -> int:
    args = arguments().parse_args(argv)
    try:
        run(args)
    except ScanError as error:
        print(f"\ncheck_scan: {error}", file=sys.stderr)
        return 1
    return 0


def run(args) -> None:
    scan = scan_io.scan_from_args(args)
    for line in scan.describe():
        say(line)
    nz, ny, nx = scan.shape
    if min(scan.shape) < 32:
        say(f"Warning: the scan is only {min(scan.shape)} voxels thick along one side")
    sample = scan_io.grey_sample(scan.array)
    if args.invert:
        sample = -sample
    check = scan_io.check_grey(sample, inverted=args.invert, original=(lambda v: -v) if args.invert else None)
    sign = -1.0 if args.invert else 1.0
    if args.invert:  # quote the scan's own grey values, where the fibers are the dark end
        say(f"Grey values: 0.5th percentile {-check.bright:.6g} (the network's fiber, with --invert), median "
            f"{-check.void:.6g} (its empty space), 99.5th percentile {-check.low:.6g}")
        say(f"Solid/void split (Otsu): {-check.threshold:.6g}; about {100 * check.solid:.0f}% of the scan is below it")
    else:
        say(f"Grey values: 0.5th percentile {check.low:.6g}, median {check.void:.6g} (the network's empty space), "
            f"99.5th percentile {check.bright:.6g} (its bright fiber)")
        say(f"Solid/void split (Otsu): {check.threshold:.6g}; about {100 * check.solid:.0f}% of the scan is above it")
    for warning in check.warnings:
        say(f"  Warning: {warning}")
    out = Path(args.out).expanduser() if args.out else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    name = find_fibers.stem(scan.path)
    picture = scan_picture(scan, sample * sign, check, sign, out / f"{name}_check.png")
    if picture:
        say(f"Saved {picture}: the middle slice each way and the grey histogram")
    if not scan.voxel_um:
        say("The voxel size is unknown: find it in the scanner's log or reconstruction settings, and give it as "
            "--voxel-size (e.g. 1.3um)")
    command = ["python", "find_fibers.py", args.scan, "--weights", args.weights or "NETWORK.pt"]
    if args.voxel_size or not scan.voxel_um:
        command += ["--voxel-size", args.voxel_size or "VOXEL_SIZE"]
    for option in ("dataset", "raw_shape", "raw_dtype"):
        if getattr(args, option):
            command += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    if args.raw_header_bytes:
        command += ["--raw-header-bytes", str(args.raw_header_bytes)]
    if args.raw_endian != "little":
        command += ["--raw-endian", args.raw_endian]
    if args.invert:
        command.append("--invert")
    if check.solid > 0.45:
        command.append(f"--levels={sign * check.dark_void:.6g},{sign * check.bright:.6g}")  # "=": a value may be negative
    if args.weights:
        command += measure(scan, args, out, name)
    elif args.bin > 1:
        command += ["--bin", str(args.bin)]
    say("")
    if max(scan.shape) > 256:
        say("Next, find the fibers in a small piece first (then drop --center-crop for the whole scan):")
        command += ["--center-crop", "256"]
    else:
        say("Next, find the fibers:")
    say("  " + " ".join(shlex.quote(part) for part in command))


def measure(scan, args, out: Path, name: str) -> list[str]:
    """Run the network on the middle cube, report the fiber sizes it finds, and return find_fibers options."""
    k = args.bin
    box = scan_io.center_box(args.size * k, scan.shape)
    box = scan_io.Box(box.z0, box.z0 + box.shape[0] // k * k, box.y0, box.y0 + box.shape[1] // k * k,
                      box.x0, box.x0 + box.shape[2] // k * k)
    if min(box.shape) // k < 32:
        raise ScanError(f"The middle cube is too small to measure fibers in ({' x '.join(map(str, box.shape[::-1]))})")
    block = np.asarray(scan.array[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1], dtype=np.float32)
    if k > 1:
        a, b, c = (n // k for n in block.shape)
        block = block.reshape(a, k, b, k, c, k).mean(axis=(1, 3, 5), dtype=np.float32)
    if args.invert:
        block = -block
    known = bool(scan.voxel_um)
    voxel = scan.voxel_um * k if known else float(k)  # without a voxel size, "um" below are voxels
    unit = "um" if known else "voxels of the scan"
    say("")
    say(f"Measuring the fibers in the middle {block.shape[2]} x {block.shape[1]} x {block.shape[0]} voxels"
        + (f" (binned {k} x {k} x {k})" if k > 1 else "") + ", telling the network nothing")
    device = find_fibers.pick_device(args.device)
    model = find_fibers.load_network(find_fibers.network_file(args.weights), device)
    # the cube's own grey levels, as find_fibers.py --center-crop would take them
    cube = scan_io.check_grey(block.ravel()[::max(1, block.size // 8_000_000)], inverted=args.invert)
    grey = [cube.void, cube.bright]
    fibers, _ = find_fibers.find_in_region(block, voxel, model, device, grey, types=args.types)
    if not fibers.count:
        say("No fibers found in the middle of the scan. Check the grey levels and polarity above, the crop, and "
            "that the fibers are at least ~4 voxels across")
        return ["--bin", str(k)] if k > 1 else []
    d = fibers.diameters_um()
    length = fibers.lengths_um()
    say(f"Found {fibers.count} fibers. Their diameters ({'um' if known else 'voxels'}), grouped into types:")
    medians = []
    for t in range(len(fibers.type_names)):
        mine = fibers.types == t
        if not mine.any():
            continue
        q1, median, q3 = np.percentile(d[mine], [25, 50, 75])
        medians.append(median)
        say(f"  type {t + 1}: {int(mine.sum())} fibers, {100 * length[mine].sum() / length.sum():.0f}% of the length,"
            f" median {median:.3g} {unit} ({median / voxel:.1f} voxels as analysed), middle half {q1:.3g}-{q3:.3g}")
    if fibers.bonds:
        say(f"The network sees {len(fibers.bonds)} bonds between them (binder where fibers cross): if the sample is "
            "bonded, add --bonded yes")
    vox = [m / voxel for m in medians]
    options = []
    if max(vox) > find_fibers.TRAINED[1]:
        b = int(np.ceil(max(vox) / 24.0))
        say(f"The thickest fibers are {max(vox):.0f} voxels across, more than the network was trained on: bin the "
            f"scan, --bin {b * k}")
        k = b * k
    elif min(vox) < find_fibers.TRAINED[0]:
        say(f"The thinnest fibers are {min(vox):.1f} voxels across, at the edge of what the network sees"
            + (": try a smaller --bin" if k > 1 else ""))
    if k > 1:
        options += ["--bin", str(k)]
    options += ["--diameters", ",".join(f"{m:.3g}{'um' if known else 'vox'}" for m in medians)]
    if len(medians) > 1:
        say("If the sample really has a different number of fiber types, rerun with --types K, or give find_fibers.py "
            "the diameters you know")
    picture = fiber_outputs.write_preview(fibers, out / f"{name}_check_fibers.png", block,
                                          (cube.low, cube.bright), f"{fibers.count} fibers in the middle of {name}")
    if picture:
        say(f"Saved {picture}: the fibers found in the middle cube")
    return options


def scan_picture(scan, sample: np.ndarray, check, sign: float, path: Path) -> Path | None:
    """The middle slice each way (side views only where reading across the slices is quick) and the histogram."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    nz, ny, nx = scan.shape
    views = [(f"slice z = {nz // 2} (x across, y down)", lambda: scan.array[nz // 2])]
    if not isinstance(scan.array, scan_io._SliceStack) or nz <= 300:
        views += [(f"row y = {ny // 2} (x across, z down)", lambda: scan.array[:, ny // 2, :]),
                  (f"column x = {nx // 2} (y across, z down)", lambda: scan.array[:, :, nx // 2])]
    low, high = sorted((sign * check.low, sign * check.bright))
    figure, axes = plt.subplots(1, len(views) + 1, figsize=(5 * (len(views) + 1), 5), squeeze=False)
    for ax, (title, read) in zip(axes[0], views):
        image = np.asarray(read(), np.float32)
        step = max(1, max(image.shape) // 1200)
        ax.imshow(image[::step, ::step], cmap="gray", vmin=low, vmax=high, interpolation="nearest",
                  extent=(0, image.shape[1], image.shape[0], 0))
        ax.set_title(title, fontsize=10)
    ax = axes[0, -1]
    values = sample[np.isfinite(sample)]
    if check.fill is not None:
        values = values[values != sign * check.fill]
    ax.hist(values, bins=256, color="0.4", log=True)
    for value, label, color in ((check.void, "median (empty space)", "tab:blue"),
                                (check.bright, "99.5th percentile (bright fiber)", "tab:red"),
                                (check.threshold, "solid/void split", "tab:green")):
        ax.axvline(sign * value, color=color, label=label)
    ax.legend(fontsize=8)
    ax.set_title("grey values" + (" (fill value left out)" if check.fill is not None else ""), fontsize=10)
    voxel = f", {scan.voxel_um:.4g} um voxels" if scan.voxel_um else ""
    figure.suptitle(f"{scan.path.name}: {scan_io.scan_dims(scan)} voxels{voxel}", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


if __name__ == "__main__":
    sys.exit(main())
