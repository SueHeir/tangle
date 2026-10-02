# Copied from examples/ct_unet/find_fibers.py by its sync_package.py: change that file, then run it again.
"""Find the fibers in a CT scan with the map network: centerlines, diameters, types and bonds.

    python find_fibers.py SCAN --weights NETWORK.pt --diameters 12um,30um

The tutorial, docs/ct_unet_tutorial.md, walks through a first run; ``--help`` lists every option. In short:

1. The scan is read in any common format (``scan_io.py``) and cut to the region asked for (``--crop``,
   ``--center-crop``, ``--bin``, ``--invert``); a region that can't be read in place is copied once to
   OUT/work/volume.npy.
2. The network runs on 128-voxel tiles every 64 voxels, and every voxel it calls fiber votes for its fiber's
   axis from the tile it is most central in, as ``trace_maps.tiled_axis_points`` does. The votes are saved
   in OUT/work as the tiles go, so a stopped run carries on where it stopped, and a rerun that changes only
   the tracing (``--min-length``, ``--types``, outputs) doesn't run the network again.
3. The fibers are tracked through the votes once over the whole region (``trace_maps``), typed by their
   diameter (``fiber_types``) and joined by the bonds the network finds (``bonds``).
4. The results go to OUT (default: <scan name>_fibers): see ``fiber_outputs.py`` and the tutorial.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import fiber_outputs
from . import scan_io
from .fiber_outputs import Fibers, polyline_length
from .scan_io import ScanError

TILE, STRIDE = 128, 64  # the network's training crop, and a tile every half tile
TRAINED = (3.5, 28.0)  # fiber diameters (voxels) the network was trained on
MAX_AXIS = 7000  # trace_maps packs each vote's cell into one number: every axis must stay under ~7,000 voxels
CELLS = ("key", "count", "position", "tensor", "radius")  # what trace_maps.vote_cells returns
EXAMPLES = """examples:
  python find_fibers.py scan.tif --weights best.pt --center-crop 256       a first try in the middle
  python find_fibers.py scan.tif --weights best.pt --diameters 12um,30um   two fiber types of known size
  python find_fibers.py slices/ --weights best.pt --voxel-size 2.5um --bonded yes --out run1
"""


def say(text: str = "") -> None:
    print(text, flush=True)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find_fibers.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Find the fibers in a CT scan with the map network (docs/ct_unet_tutorial.md walks through it).",
        epilog=EXAMPLES + "\nreads " + scan_io.FORMATS)
    scan_io.add_input_arguments(parser)
    parser.add_argument("--weights", required=True, metavar="NETWORK.pt", help="the trained network (a .pt file)")
    parser.add_argument("--out", metavar="FOLDER", help="where the results go (default: <scan name>_fibers in the "
                        "current folder)")
    known = parser.add_argument_group("what you know about the sample (all optional; the network is more accurate "
                                      "when told)")
    known.add_argument("--diameters", metavar="D1,D2", help="the fiber diameters, one per fiber type, e.g. 12um,30um")
    known.add_argument("--types", type=int, metavar="K", help="how many fiber types there are, when their diameters "
                       "aren't known (default: guessed, 1 to 4)")
    known.add_argument("--bonded", choices=("yes", "no", "unknown"), default="unknown",
                       help="whether the fibers are bonded (binder where they cross)")
    scan_io.add_region_arguments(parser)
    more = parser.add_argument_group("tracing and outputs")
    more.add_argument("--min-length", metavar="L", help="drop traced fibers shorter than this, e.g. 40um (default: 3 "
                      "diameters of the thinnest type)")
    more.add_argument("--stacks", choices=("auto", "yes", "no"), default="auto", help="write overlay.tif and "
                      "labels.tif (auto: for regions under 600 million voxels)")
    more.add_argument("--binder", action="store_true", help="also write binder.tif: solid voxels outside the traced "
                      "fibers")
    more.add_argument("--diameter-profile", action="store_true", help="measure the diameter all along every fiber "
                      "(runs the network a second time)")
    more.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                      help="where the network runs (auto: a GPU when there is one)")
    more.add_argument("--restart", action="store_true", help="throw away the saved progress in FOLDER/work")
    return parser


def main(argv=None) -> int:
    args = arguments().parse_args(argv)
    try:
        run(args)
    except ScanError as error:
        print(f"\nfind_fibers: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to carry on from the saved progress.", file=sys.stderr)
        return 130
    return 0


def run(args) -> Path:
    started = time.time()
    scan = scan_io.scan_from_args(args)
    for line in scan.describe():
        say(line)
    if not scan.voxel_um:
        raise ScanError("The file doesn't say its voxel size: give it with --voxel-size, e.g. --voxel-size 1.3um. "
                        "(If you don't know it, --voxel-size 1um gives every length in voxels.)")
    weights = network_file(args.weights)
    box = scan_io.region_box(args, scan.shape)
    k = args.bin
    voxel = scan.voxel_um * k
    shape = tuple(n // k for n in box.shape)
    if max(shape) > MAX_AXIS:
        raise ScanError(f"The region is {max(shape):,} voxels long on one side; the tracer handles up to "
                        f"{MAX_AXIS:,}. Analyse it in pieces with --crop, or bin it with --bin 2")
    # lengths given in voxels ("16vox") count the scan's own voxels, before any binning
    sizes = sorted(scan_io.parse_lengths(args.diameters, scan.voxel_um, "diameter")) if args.diameters else None
    if args.types is not None and (args.types < 1 or (sizes and args.types != len(sizes))):
        raise ScanError("--types is the number of fiber types, 1 or more (and with --diameters, one diameter per type)")
    min_length = scan_io.parse_length(args.min_length, scan.voxel_um, "length") if args.min_length else None
    say(f"Region: {box.describe(scan.shape)}" + (f", binned {k} x {k} x {k}" if k > 1 else "")
        + (", inverted" if args.invert else "") + f": {shape[2]} x {shape[1]} x {shape[0]} voxels of {voxel:.4g} um")
    if sizes:
        size_advice(sizes, voxel, k)
    out = Path(args.out).expanduser() if args.out else Path.cwd() / f"{stem(scan.path)}_fibers"
    work = out / "work"
    if args.restart and work.exists():
        shutil.rmtree(work)
    out.mkdir(parents=True, exist_ok=True)
    say(f"Results go to {out}")

    region_key = plain({"scan": scan.fingerprint, "box": [box.z0, box.z1, box.y0, box.y1, box.x0, box.x1],
                        "bin": k, "invert": args.invert})
    region = load_region(scan, box, args, work, region_key)
    grey, check = grey_levels(region, args)
    device = pick_device(args.device)
    model = load_network(weights, device)
    bonded = {"yes": True, "no": False, "unknown": None}[args.bonded]
    describe_hints(model, sizes, bonded)
    stat = weights.stat()
    vote_key = plain({"region": region_key, "levels": grey, "bonded": bonded,
                      "diameters": [round(d / voxel, 3) for d in sizes] if sizes else None,
                      "network": {"path": str(weights.resolve()), "bytes": stat.st_size, "modified": int(stat.st_mtime)},
                      "tile": TILE, "stride": STRIDE, "version": 1})
    fibers, info = find_in_region(region, voxel, model, device, grey, sizes, args.types, bonded, min_length,
                                  work=work, key=vote_key, profile=args.diameter_profile)

    say("Writing the results")
    fiber_outputs.clear_outputs(out)  # an earlier run's bonds or stacks would no longer match these fibers
    history = [{"stage": "ct_unet find_fibers", "network": weights.name, "bonded": args.bonded,
                "diameters_um": sizes, "voxel_size_um": voxel, "region": box.describe(scan.shape), "bin": k,
                "invert": args.invert}]
    levels = {"void": grey[0], "fiber": grey[1], "threshold": check.threshold}
    fit_path, fit_notes = fiber_outputs.write_fit_json(fibers, out / "fit.json", levels, history, sizes)
    paths = [fit_path]
    paths += fiber_outputs.write_tables(fibers, out)
    paths += fiber_outputs.write_vtk(fibers, out)
    paths += fiber_outputs.write_ovito(fibers, out)
    voxels = int(np.prod(shape))
    stacks = args.stacks == "yes" or (args.stacks == "auto" and voxels <= fiber_outputs.STACK_LIMIT)
    display = (check.low, check.bright)
    fractions = None
    if stacks or args.binder:
        scan_io.check_space(out, voxels * ((5 if stacks else 0) + (2 if args.binder else 0)), "the TIFF stacks",
                            "or rerun with --stacks no (the network's work is saved, so that is quick)")
        say("Drawing the fibers into TIFF stacks")
        written, fractions = fiber_outputs.write_stacks(fibers, out, region, display, overlay=stacks, labels=stacks,
                                                        binder_level=check.threshold if args.binder else None, say=say)
        paths += written
    if args.stacks == "auto" and not stacks:
        say(f"  The region is big ({voxels / 1e6:,.0f} million voxels), so overlay.tif and labels.tif were skipped "
            f"(the overlay alone would take {scan_io.human_bytes(3 * voxels)}). Add --stacks yes to write them.")
    preview = fiber_outputs.write_preview(fibers, out / "overlay.png", region, display,
                                          f"{fibers.count:,} fibers found in {scan.path.name}")
    if preview:
        paths.append(preview)

    seconds = time.time() - started
    report = fiber_outputs.summary(fibers, sizes, fractions) + (fit_notes if fibers.count else [])
    origin = [box.x0, box.y0, box.z0]
    header = [f"find_fibers.py results for {scan.path}", "",
              f"Region: {box.describe(scan.shape)}" + (f", binned {k} x {k} x {k}" if k > 1 else "")
              + (", inverted" if args.invert else "") + f": {shape[2]} x {shape[1]} x {shape[0]} voxels of "
              f"{voxel:.4g} um. All coordinates are from its corner, which is voxel x {origin[0]}, y {origin[1]}, "
              f"z {origin[2]} of the scan.",
              f"Network: {weights.name} on {device}; {info['hints']}",
              f"Grey levels: empty space {grey[0]:.6g}, bright fiber {grey[1]:.6g}", ""]
    files = ["", "Files:"] + [f"  {p.name}" for p in paths] + ["", f"Took {duration(seconds)}."]
    (out / "summary.txt").write_text("\n".join(header + report + files) + "\n")
    run_info = {"scan": str(scan.path), "format": scan.kind, "scan_voxel_size_um": scan.voxel_um,
                "voxel_size_from": scan.voxel_from, "voxel_size_um": voxel, "bin": k, "invert": args.invert,
                "region_voxels_xyz": [[box.x0, box.x1], [box.y0, box.y1], [box.z0, box.z1]],
                "region_origin_um": [v * scan.voxel_um for v in origin], "network": str(weights), "device": device,
                "diameters_um": sizes, "types": args.types, "bonded": args.bonded, "grey_levels": grey,
                "min_length_um": info["min_length_um"], "fibers": fibers.count, "bonds": len(fibers.bonds),
                "seconds": round(seconds, 1), "command": " ".join(sys.argv)}
    (out / "run.json").write_text(json.dumps(run_info, indent=1) + "\n")
    paths += [out / "summary.txt", out / "run.json"]

    say("")
    for line in report:
        say(line)
    say("")
    say(f"Done in {duration(seconds)}. The results are in {out}:")
    say("  summary.txt (this summary), overlay.png (a quick look), fibers.csv (one row per fiber),")
    say("  fit.json (for Tangle), fibers.vtk (ParaView), fibers.dump (OVITO)"
        + (", overlay.tif and labels.tif (Fiji)" if stacks else ""))
    if stacks:
        say("Check the fibers by eye: open overlay.tif in Fiji and page through the slices.")
    if work.exists():
        size = sum(p.stat().st_size for p in work.iterdir() if p.is_file())
        say(f"{work} ({scan_io.human_bytes(size)}) makes reruns fast; delete it when you are done.")
    return out


# -- the pieces --------------------------------------------------------------------------------------------------


def find_in_region(region, voxel_um: float, model, device: str, grey, sizes_um=None, types=None, bonded=None,
                   min_length_um=None, work: Path | None = None, key=None, profile: bool = False) -> tuple[Fibers, dict]:
    """The fibers in a (z, y, x) region (array-like): network votes, tracking, types, bonds (see the module notes).

    ``sizes_um``: known fiber diameters (one per type), ``types``: the number of types when the sizes aren't
    known, ``bonded``: True, False or None (unknown). With ``work`` and ``key``, the votes are saved there and
    reused while ``key`` (a description of the region, network and hints) stays the same."""
    from .trace_maps import cells_to_points, tidy_up, track

    shape = tuple(int(n) for n in region.shape)
    sizes = [d / voxel_um for d in sizes_um] if sizes_um else None
    want_bonds = getattr(model, "layout", "bondpoints") == "bondpoints" and bonded is not False
    hints = {"diameters": sizes, "bonds": bonded}
    cells, centers, strengths = network_votes(region, model, device, grey, hints, want_bonds, work, key)

    points = cells_to_points(*cells, min_votes=3)
    if min_length_um:
        shortest = min_length_um / voxel_um
    elif sizes:
        shortest = 3.0 * min(sizes)
    else:
        shortest = 3.0 * TRAINED[0]
    say(f"Tracing the fibers through {len(points[0]):,} axis points")
    lines, radii = track(*points, shape=shape, min_length=shortest, log=lambda text: say("  " + text))
    lines, radii = tidy_up(lines, radii)
    lines, radii = [np.asarray(line, float) for line in lines], np.asarray(radii, float)
    kinds, names = fiber_kinds(lines, radii, sizes, types, region, grey, voxel_um)
    if not min_length_um and not sizes and lines:
        # sizes not known beforehand: as with known sizes, drop traces shorter than 3 diameters of the thinnest type
        thinnest = min(float(np.median(2.0 * radii[kinds == t])) for t in np.unique(kinds))
        keep = np.array([polyline_length(line) >= 3.0 * thinnest for line in lines])
        lines, radii, kinds = [l for l, k in zip(lines, keep) if k], radii[keep], kinds[keep]
        shortest = max(shortest, 3.0 * thinnest)
        kinds, names = settle_types(lines, radii, kinds, voxel_um, fold=not types)  # a type may have lost its fibers
    say(f"  {len(lines):,} fibers")
    bonds = []
    if want_bonds and len(centers) and lines:
        from .bonds import bonds_at

        bonds = bonds_at(centers, strengths, lines, radii)
        say(f"  {len(bonds):,} bonds")
    support = backing(lines, points[0])
    profiles = None
    if profile and lines:
        profiles = diameter_profiles(region, lines, model, device, grey, hints, voxel_um)
    fibers = Fibers(lines, radii, kinds, support, bonds, voxel_um, shape, names, profiles)
    given = ("diameters " + ", ".join(f"{d:.4g} um" for d in sizes_um)) if sizes_um else ""
    if getattr(model, "condition", False):
        told = [given] if given else []
        if bonded is not None:
            told.append("bonded" if bonded else "not bonded")
        said = "told: " + "; ".join(told) if told else "no hints"
    else:
        said = "this network takes no hints" + (f"; {given} used to type the fibers" if given else "")
    return fibers, {"hints": said, "min_length_um": shortest * voxel_um}


def network_votes(region, model, device: str, grey, hints: dict, want_bonds: bool, work: Path | None = None,
                  key=None):
    """Every fiber voxel's axis vote, pooled in 1-voxel cells over the whole region (``trace_maps.vote_cells``),
    and the bond-point peaks (centers, strengths). Saved to ``work`` as the tiles go, when it is given."""
    from .bonds import bond_peaks
    from .trace_maps import merge_cells, vote_cells

    plan = tile_plan(region.shape)
    layers, per_layer = plan[0], len(plan[1]) * len(plan[2])
    total = len(layers) * per_layer
    cells, centers, strengths, first = None, [], [], 0
    saved = work / "votes.npz" if work is not None else None
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
        if read_json(work / "votes.json") == key and saved.exists():
            with np.load(saved) as data:
                first = int(data["layers_done"])
                cells = tuple(data[name] for name in CELLS)
                centers, strengths = [data["bond_centers"]], [data["bond_strengths"]]
            if first >= len(layers):
                say("Network: reusing the saved votes (same region, network and hints)")
            else:
                say(f"Network: carrying on from the saved progress ({first} of {len(layers)} layers of tiles done)")
        else:
            for stale in (saved, work / "votes.json"):
                if stale.exists():
                    stale.unlink()
            write_json(work / "votes.json", key)
    left = (len(layers) - first) * per_layer
    if left:
        say(f"Running the network on {left:,} tile{'s' if left > 1 else ''} of {TILE} x {TILE} x {TILE} voxels"
            + (" (this is slow on the CPU)" if device == "cpu" and left > 50 else ""))
    done, started, shown, stored = 0, time.time(), time.time(), time.time()
    for layer, (oz, z0, z1) in enumerate(layers):
        if layer < first:
            continue
        parts = [] if cells is None else [cells]
        for oy, y0, y1 in plan[1]:
            for ox, x0, x1 in plan[2]:
                maps = tile_maps(region, (oz, oy, ox), model, device, grey, hints)
                window = (slice(z0 - oz, z1 - oz), slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
                parts.append(vote_cells(maps, (oz, oy, ox), window))
                if want_bonds:
                    c, s = bond_peaks(maps, 0.5, window, (oz, oy, ox))
                    centers.append(c)
                    strengths.append(s)
                done += 1
                if time.time() - shown > 30:
                    shown = time.time()
                    say(f"  tiles {total - left + done:,} of {total:,}, about "
                        f"{duration((shown - started) / done * (left - done))} to go")
        cells = merge_cells(*[np.concatenate(c) for c in zip(*parts)])
        if centers:
            centers, strengths = [np.concatenate(centers)], [np.concatenate(strengths)]
        if saved is not None and (layer == len(layers) - 1 or time.time() - stored > 120):
            partial = saved.with_name("votes.partial.npz")
            np.savez(partial, **dict(zip(CELLS, cells)),
                     bond_centers=centers[0] if centers else np.zeros((0, 3)),
                     bond_strengths=strengths[0] if strengths else np.zeros(0), layers_done=layer + 1)
            os.replace(partial, saved)
            stored = time.time()
    centers = np.concatenate(centers) if centers else np.zeros((0, 3))
    strengths = np.concatenate(strengths) if strengths else np.zeros(0)
    return cells, centers, strengths


def tile_plan(shape, tile: int = TILE, stride: int = STRIDE) -> list[list[tuple[int, int, int]]]:
    """Per axis (z, y, x): (tile start, core start, core end). Tiles every ``stride`` voxels, the last flush with
    the end; each voxel belongs to the core of the tile whose center is nearest (``trace_maps.tiled_axis_points``)."""
    plan = []
    for n in shape:
        starts = list(range(0, max(n - tile, 0) + 1, stride))
        if starts[-1] + tile < n:
            starts.append(n - tile)
        cuts = [0] + [(a + tile + b) // 2 for a, b in zip(starts[:-1], starts[1:])] + [n]
        plan.append(list(zip(starts, cuts[:-1], cuts[1:])))
    return plan


def tile_maps(region, corner, model, device: str, grey, hints: dict) -> np.ndarray:
    """The network's maps of the tile at ``corner`` (z, y, x). A region thinner than a tile is padded with
    empty space (the void grey), as the network sees beyond the edges of its training crops."""
    from .maps import predict

    z, y, x = corner
    block = np.asarray(region[z:z + TILE, y:y + TILE, x:x + TILE], dtype=np.float32)
    if not np.isfinite(block).all():  # NaN outside a reconstruction's field of view: empty space
        block = np.nan_to_num(block, nan=grey[0], posinf=grey[1], neginf=grey[0])
    size = block.shape
    if size != (TILE, TILE, TILE):
        padded = np.full((TILE, TILE, TILE), grey[0], dtype=np.float32)
        padded[:size[0], :size[1], :size[2]] = block
        block = padded
    maps = predict(model, block, device, grey, diameters=hints["diameters"], bonds=hints["bonds"])
    return maps[:, :size[0], :size[1], :size[2]]


def fiber_kinds(lines, radii, sizes, k, region, grey, voxel_um: float):
    """Each fiber's type (0 = the thinnest) and a name per type: by the nearest known diameter, or else by
    clustering the fibers' diameters and axis grey (``fiber_types``; ``k`` types, or 1-4 by BIC)."""
    from .fiber_types import assign_by_size, assign_types, features

    if sizes:
        kinds = assign_by_size(radii, sizes) if len(sizes) > 1 and lines else np.zeros(len(lines), int)
        return np.asarray(kinds, int), [f"{d * voxel_um:.4g} um" for d in sizes]
    if not lines:
        return np.zeros(0, int), []
    if k == 1 or len(lines) < 2:
        kinds = np.zeros(len(lines), int)
    else:
        feats = features(lines, radii, region, grey)
        for column in feats.T:  # a fiber through NaN voxels: give it the typical grey
            bad = ~np.isfinite(column)
            column[bad] = np.median(column[~bad]) if (~bad).any() else 0.0
        kinds = np.asarray(assign_types(feats, min(k, len(lines)) if k else None)[0], int)
    return settle_types(lines, radii, kinds, voxel_um, fold=not k)


def settle_types(lines, radii, kinds, voxel_um: float, fold: bool) -> tuple[np.ndarray, list[str]]:
    """Clustered types numbered from 0 by median diameter (0 = the thinnest) with none left empty, and a name
    for each. With ``fold`` (the number of types was guessed), a type of fewer than 3 fibers or under 2% of
    the traced length joins the type nearest to it in diameter: a few odd traces are not a fiber type."""
    kinds = np.array(kinds, dtype=int)
    radii = np.asarray(radii, float)
    if fold and len(lines):
        length = np.array([polyline_length(line) for line in lines])
        size = np.log(2.0 * np.maximum(radii, 1e-6))
        while True:
            used = np.unique(kinds)
            if len(used) < 2:
                break
            count = np.array([np.sum(kinds == t) for t in used])
            share = np.array([length[kinds == t].sum() for t in used]) / max(float(length.sum()), 1e-12)
            small = (count < 3) | (share < 0.02)
            if not small.any():
                break
            t = used[np.argmin(np.where(small, share, np.inf))]
            rest = used[used != t]
            centers = np.array([np.median(size[kinds == r]) for r in rest])
            kinds[kinds == t] = rest[np.argmin(np.abs(centers - np.median(size[kinds == t])))]
    used = np.unique(kinds)
    rank = np.argsort(np.argsort([np.median(radii[kinds == t]) for t in used]))
    kinds = rank[np.searchsorted(used, kinds)] if len(used) else kinds
    names = [f"about {2.0 * float(np.median(radii[kinds == t])) * voxel_um:.3g} um" for t in range(len(used))]
    return kinds, names


def backing(lines, positions) -> np.ndarray:
    """The share of each trace that runs within a voxel of an axis point; the rest coasted over gaps."""
    if not lines:
        return np.zeros(0)
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    shares = []
    for line in lines:
        distance, _ = tree.query(fiber_outputs.resample(line, 1.0), distance_upper_bound=1.0)
        shares.append(float(np.isfinite(distance).mean()))
    return np.array(shares)


def diameter_profiles(region, lines, model, device: str, grey, hints: dict, voxel_um: float) -> list[dict]:
    """The diameter at every voxel along every fiber (``diameters.Accumulator``): the network's second pass."""
    from .diameters import Accumulator

    plan = tile_plan(region.shape)
    total = len(plan[0]) * len(plan[1]) * len(plan[2])
    say(f"Measuring the diameter along every fiber: the network again, {total:,} tile{'s' if total > 1 else ''}")
    accumulator = Accumulator(lines, tuple(int(n) for n in region.shape))
    done, started, shown = 0, time.time(), time.time()
    for oz, z0, z1 in plan[0]:
        for oy, y0, y1 in plan[1]:
            for ox, x0, x1 in plan[2]:
                maps = tile_maps(region, (oz, oy, ox), model, device, grey, hints)
                accumulator.add(maps, (oz, oy, ox), (slice(z0 - oz, z1 - oz), slice(y0 - oy, y1 - oy),
                                                     slice(x0 - ox, x1 - ox)))
                done += 1
                if time.time() - shown > 30:
                    shown = time.time()
                    say(f"  tiles {done:,} of {total:,}, about {duration((shown - started) / done * (total - done))} to go")
    return accumulator.result(spacing_um=voxel_um)


def load_region(scan, box, args, work: Path, key: dict):
    """The region as an array: the scan itself where it can be read in place, else a copy in ``work`` (reused
    while ``key`` matches)."""
    target = work / "volume.npy"
    if read_json(work / "volume.json") == key and target.exists():
        say(f"  using the copy of the region made before ({target})")
        return np.load(target, mmap_mode="r")
    for stale in (target, work / "volume.json"):  # a copy of another region: free its disk space first
        if stale.exists():
            stale.unlink()
    region, copied = scan_io.prepare_region(scan, box, args.bin, args.invert, work, say)
    if copied:
        write_json(work / "volume.json", key)
    return region


def grey_levels(region, args):
    """The grey levels the network scales the scan by: (empty space, bright fiber), and the grey check."""
    flip = scan_io.flip(region.dtype) if args.invert else None  # the region is inverted; the user's greys aren't
    check = scan_io.check_grey(scan_io.grey_sample(region), inverted=args.invert, original=flip)
    if args.levels:
        values = [v for v in args.levels.replace(";", ",").split(",") if v.strip()]
        try:
            void, bright = (float(v) for v in values)
        except ValueError:
            raise ScanError(f"--levels '{args.levels}': give two grey values, empty space then fiber, "
                            "e.g. --levels=1200,5400") from None
        if flip:  # given in the scan's own grey values: invert them as the scan was
            void, bright = flip(void), flip(bright)
        if bright <= void:
            raise ScanError("--levels: the fiber grey must be " + ("below the empty-space grey (with --invert the "
                            "fibers are the dark ones)" if args.invert else "above the empty-space grey"))
        say(f"Grey levels (--levels): empty space {void:.6g}, bright fiber {bright:.6g}"
            + (" after inverting" if args.invert else ""))
        return [float(void), float(bright)], check
    say(f"Grey levels: empty space {check.void:.6g}, bright fiber {check.bright:.6g} (the median and the 99.5th "
        "percentile" + (", after inverting)" if args.invert else ")"))
    for warning in check.warnings:
        say(f"  Warning: {warning}")
    if check.bright <= check.void:
        raise ScanError("The region has almost no contrast: its median grey equals its 99.5th percentile. Check the "
                        "crop, or give the grey levels with --levels=EMPTY,FIBER")
    return [check.void, check.bright], check


def size_advice(sizes_um, voxel_um: float, k: int) -> None:
    """Say how the given fiber sizes compare with what the network was trained on, and what to do about it."""
    vox = [d / voxel_um for d in sizes_um]
    say("Fiber diameters: " + ", ".join(f"{d:.4g} um = {v:.1f} voxels" for d, v in zip(sizes_um, vox))
        + f" (the network was trained on {TRAINED[0]:g}-{TRAINED[1]:g} voxels)")
    if max(vox) > TRAINED[1]:
        b = int(np.ceil(max(vox) / 24.0))
        fits = min(vox) / b >= TRAINED[0]
        say("  Warning: the thickest fibers are bigger than the network knows. "
            + (f"Bin the scan: --bin {b * k} makes them {max(vox) / b:.1f} voxels across" if fits
               else "The sizes span more than the network covers in one pass; bin for the thick ones or not at all "
                    "for the thin ones (two runs)"))
    if min(vox) < TRAINED[0]:
        say(f"  Warning: the thinnest fibers are under {TRAINED[0]:g} voxels across; the network will miss many of "
            "them" + (" (try a smaller --bin)" if k > 1 else ". A scan with smaller voxels is the fix"))


def describe_hints(model, sizes, bonded) -> None:
    told = []
    if sizes:
        told.append("the diameters")
    if bonded is not None:
        told.append("bonded" if bonded else "not bonded")
    if not getattr(model, "condition", False):
        if told:
            say("This network takes no hints: the diameters are used only to type the fibers")
    elif told:
        say("Telling the network: " + " and ".join(told))
    else:
        say("No hints given (--diameters, --bonded): the network works without them, a little less accurately")
    if bonded and getattr(model, "layout", "") != "bondpoints":
        say("This network has no bond-point output, so no bonds are looked for")


def network_file(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_file():
        raise ScanError(f"Can't find the network file {path}. It is the .pt file train.py writes (best.pt); the "
                        "tutorial says where to get one")
    return path


def pick_device(name: str) -> str:
    torch = scan_io.need("torch", "Running the network", "torch")
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    chosen = name != "auto"
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    elif name == "cuda" and not torch.cuda.is_available():
        raise ScanError("--device cuda: PyTorch sees no CUDA GPU here (a CUDA build of PyTorch is needed)")
    elif name == "mps" and not has_mps:
        raise ScanError("--device mps: PyTorch sees no Apple GPU here")
    where = {"cuda": "the GPU", "mps": "the Apple GPU", "cpu": "the CPU"}[name]
    if name == "cpu":
        where += (" (slow: try a small --center-crop first)" if chosen
                  else " (no GPU found: it runs, slowly; try a small --center-crop first)")
    say(f"Network on {where}")
    return name


def load_network(path: Path, device: str):
    from .maps import load

    try:
        return load(path, device)
    except Exception as error:
        raise ScanError(f"Couldn't load the network from {path}: {error}") from None


def stem(path: Path) -> str:
    name = Path(path).name
    for suffix in (".nii.gz", ".ome.tiff", ".ome.tif"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem if Path(name).suffix else name


def plain(value):
    """``value`` as JSON would give it back, so saved and new descriptions compare equal."""
    return json.loads(json.dumps(value))


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds // 3600} h {round(seconds % 3600 / 60):02d} min"


if __name__ == "__main__":
    sys.exit(main())
