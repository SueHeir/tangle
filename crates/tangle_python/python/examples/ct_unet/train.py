"""Train the CT map network on make_data volumes (PyTorch, Apple GPU by default).

usage: python train.py DATA VAL OUT [--steps N] [--crop 128] [--base 16] [--resume]

DATA and VAL may each be several folders joined by commas (e.g. dense_hard-style and varied data together).

Writes OUT/last.pt every checkpoint and OUT/best.pt at the lowest validation
loss, and one line per log interval to stdout.

Hints (--condition): each training crop is told a random part of what is true about its scan, and about a third
of them nothing at all, so one network works with any hints or none (maps.draw_hints; --blank-rate, --size-rate,
--hint-rate, --extra-rate). A network from before the newer hints grows their inputs, unread at first (--init).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from maps import (BINDER, CHANNELS, HINT_CODE, UNet3D, augment, draw_hints, load, loss_terms, normalize, scan_hints,
                  size_code, targets, widen)

AMP = None  # autocast dtype on CUDA (set from --amp)
WEIGHTS = {"heat": 1.0, "offset": 1.0, "direction": 1.0, "fiber": 0.5, "radius": 1.0, "binder": 1.0, "bondpt": 1.0}


class Crops(Dataset):
    def __init__(self, files, crop: int, per_volume: int, seed: int, train: bool = True, noise: float = 0.0,
                 hint_rates: dict | None = None):
        """``hint_rates``: ``draw_hints``' chances (blank, sizes, binder, extra) for training crops."""
        self.files, self.crop, self.per_volume, self.seed, self.train = files, crop, per_volume, seed, train
        self.noise, self.hint_rates = noise, hint_rates or {}

    def __len__(self):
        return len(self.files) * self.per_volume

    def __getitem__(self, item):
        rng = np.random.default_rng((self.seed, item, int(time.time() * 1e3) if self.train else 0))
        data = np.load(self.files[item // self.per_volume])
        shape = data["volume"].shape
        low = [int(rng.integers(0, s - self.crop + 1)) if self.train else (s - self.crop) // 2 for s in shape]
        window = tuple(slice(l, l + self.crop) for l in low)
        sample = targets(data, window)
        sample["image"] = normalize(data["volume"])[window][None]
        # The hints, as a user would give them: in training, a random part of what is true about the scan (often
        # nothing; see draw_hints); in validation, everything for every other scan and nothing for the rest, so
        # the best checkpoint is the best one both ways.
        truth = scan_hints(data)
        if self.train:
            said = draw_hints(truth, rng, **self.hint_rates)
        else:
            said = truth if (item // self.per_volume) % 2 == 0 else {}
        if self.train:
            sample = augment(sample, rng)
            # Grey level jitter: real scans differ in contrast and offset.
            sample["image"] = (sample["image"] * rng.uniform(0.8, 1.25) + rng.uniform(-0.1, 0.1)).astype(np.float32)
            if self.noise > 0 and rng.random() < 0.7:
                # Correlated noise: CT reconstructions often have noise correlated over about a
                # voxel, which the simulated scans mostly lack; without it the network can call
                # blotchy void "dim coarse fiber". Sd up to --noise (1 = the fine axis).
                from scipy.ndimage import gaussian_filter

                field = gaussian_filter(rng.standard_normal(sample["image"].shape[1:]).astype(np.float32),
                                        rng.uniform(0.6, 1.3, size=3))
                field *= rng.uniform(0.3, 1.0) * self.noise / max(float(field.std()), 1e-6)
                sample["image"] = sample["image"] + field[None]
        sample["code"] = size_code(**said)  # after the flips, which act on every map
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sample.items()}


def save(state, path: Path):
    """Write then rename, so a reader never sees a half-written checkpoint."""
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def run_batch(model, batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
        out = model(batch["image"], batch["code"]) if getattr(model, "condition", False) else model(batch["image"])
    terms = loss_terms(out.float(), batch)
    return sum(WEIGHTS[k] * v for k, v in terms.items()), terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("val")
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1,
                        help="training crops per step (a big GPU sits partly idle on one 128^3 crop); validation "
                             "stays one crop at a time, so its loss compares across batch sizes")
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init", type=Path, help="start from this checkpoint's weights (fine-tuning)")
    parser.add_argument("--blank-rate", type=float, default=0.35,
                        help="share of training crops given no hints at all (the network must work without them)")
    parser.add_argument("--size-rate", type=float, default=0.6,
                        help="of the other crops, the share given the fiber types' diameters")
    parser.add_argument("--hint-rate", type=float, default=0.5,
                        help="of the other crops, the share told whether there is binder")
    parser.add_argument("--extra-rate", type=float, default=0.5,
                        help="of the other crops, the share given each further hint: the types' section shapes and "
                             "hollowness (with the diameters), and whether there are broken pieces, dust, voids")
    parser.add_argument("--binder-weight", type=float, default=1.0, help="weight of the binder loss")
    parser.add_argument("--binder-positive", type=float, default=9.0, help="extra weight on true binder voxels")
    parser.add_argument("--condition", action="store_true", help="add the fiber-size conditioning (FiLM)")
    parser.add_argument("--widen", type=int, help="with --init: widen that network to this base width (Net2Net)")
    parser.add_argument("--noise", type=float, default=0.0, help="correlated-noise augmentation sd (0 = off)")
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off", help="mixed precision (CUDA only)")
    parser.add_argument("--refresh-every", type=int, default=0,
                        help="re-list the data folders every N steps (0 = once per pass), so scans still arriving join")
    parser.add_argument("--pace", action="append", default=[], metavar="DIR:TOTAL",
                        help="at each refresh, wait until DIR holds its share of TOTAL scans (TOTAL x the share of "
                             "steps done by the next refresh)")
    parser.add_argument("--exclude", type=Path,
                        help="leave out the volumes listed in this file, one FOLDER/NAME per line (e.g. those "
                             "overlaps.py finds with fibers through each other)")
    args = parser.parse_args()
    skip = set(args.exclude.read_text().split()) if args.exclude else set()

    def volumes(folders: str) -> list[Path]:
        return sorted(f for d in folders.split(",") for f in Path(d).glob("*.npz")
                      if f"{f.parent.name}/{f.stem}" not in skip)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    global AMP
    AMP = None if args.amp == "off" or device != "cuda" else {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    scaler = torch.amp.GradScaler("cuda", enabled=AMP is torch.float16)
    args.out.mkdir(parents=True, exist_ok=True)
    import maps as maps_module

    WEIGHTS["binder"] = args.binder_weight
    maps_module.BINDER_POSITIVE = args.binder_positive

    if args.init:
        source = load(args.init, "cpu")
        args.base = source.down[0][0].out_channels
        hints_in = source.film[0].in_features if getattr(source, "condition", False) else HINT_CODE
        if source.head.out_channels == CHANNELS and hints_in == HINT_CODE:
            model = source.to(device)
        elif source.layout in ("fibers", "bonds", "bondpoints"):
            # a network with fewer outputs or a shorter hint: grow it. Every weight it has is copied; new
            # outputs start at "nothing here" and new hint inputs start unread.
            model = UNet3D(base=args.base, group_sizes=source.group_sizes, condition=source.condition).to(device)
            grown = model.state_dict()
            for name, value in source.state_dict().items():
                target = grown[name].clone()
                if target.shape == value.shape:
                    target = value.clone()
                else:
                    target.zero_()
                    target[tuple(slice(0, n) for n in value.shape)] = value
                grown[name] = target
            for new_channel in range(source.head.out_channels, CHANNELS):
                grown["head.bias"][new_channel] = -4.0
            model.load_state_dict(grown)
        else:
            # a first-layout network: its body, and the head channels both layouts share (heatmap, offset,
            # direction: the first 10); its void / fine / coarse logits do not fit the fiber and radius outputs
            model = UNet3D(base=args.base, group_sizes=source.group_sizes).to(device)
            weights = source.state_dict()
            head_w, head_b = weights.pop("head.weight"), weights.pop("head.bias")
            model.load_state_dict(weights, strict=False)
            with torch.no_grad():
                model.head.weight[:10] = head_w[:10]
                model.head.bias[:10] = head_b[:10]
    else:
        model = UNet3D(base=args.base).to(device)
    if args.widen:  # after --init has grown the outputs and hint inputs, so the wide network has them all
        if not args.init:
            raise SystemExit("--widen widens the --init network: give --init too")
        try:
            model = widen(model.cpu(), args.widen).to(device)
        except ValueError as error:
            raise SystemExit(f"--widen: {error}") from None
        args.base = args.widen
    step, best = 0, float("inf")
    if args.condition and not getattr(model, "condition", False):
        conditioned = UNet3D(base=args.base, channels=model.head.out_channels, group_sizes=model.group_sizes,
                             condition=True).to(device)
        conditioned.load_state_dict(model.state_dict(), strict=False)
        model = conditioned
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    if args.resume and (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best = state["step"], state["best"]
    (args.out / "config.json").write_text(json.dumps({**vars(args), "data": args.data, "val": args.val,
                                                       "out": str(args.out), "init": str(args.init), "weights": WEIGHTS}, indent=1, default=str) + "\n")

    val_files = volumes(args.val)
    # Validation crops are the same every time, so any number of workers gives the same loss; more of them
    # keep a fast GPU from idling through each check.
    val_loader = DataLoader(Crops(val_files, args.crop, 1, 0, train=False), batch_size=1,
                            num_workers=max(2, args.workers))
    started, window, waited = time.perf_counter(), [], 0.0  # waited: time the loop spent waiting for crops
    while step < args.steps:
        # Pacing: hold training until each paced folder has its share of the scans for the coming steps.
        for spec in args.pace:
            folder, total = spec.rsplit(":", 1)
            span = args.refresh_every or args.steps
            need = min(int(total), int(np.ceil(int(total) * min(1.0, (step + span) / args.steps))))
            while (have := len(list(Path(folder).glob("*.npz")))) < need:
                print(f"PAUSED step {step}: {folder} has {have}/{need} scans, waiting", flush=True)
                time.sleep(120)
        # Re-list each pass (or every --refresh-every steps) so volumes still being generated join as they land.
        files = volumes(args.data)
        rates = {"blank": args.blank_rate, "sizes": args.size_rate, "binder": args.hint_rate, "extra": args.extra_rate}
        loader = DataLoader(Crops(files, args.crop, 2, step, noise=args.noise, hint_rates=rates),
                            batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                            persistent_workers=False, prefetch_factor=2)
        model.train()
        tick = time.perf_counter()
        for batch in loader:
            waited += time.perf_counter() - tick
            loss, terms = run_batch(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            refresh = args.refresh_every and step % args.refresh_every == 0
            window.append({k: float(v.detach()) for k, v in terms.items()})
            if step % args.log_every == 0:
                mean = {k: np.mean([w[k] for w in window]) for k in window[0]}
                window = []
                # data: seconds per step spent waiting for crops (near the step time = the CPU is the limit)
                print(f"step {step} files {len(files)} " + " ".join(f"{k} {v:.4f}" for k, v in mean.items())
                      + f" lr {scheduler.get_last_lr()[0]:.2e} {time.perf_counter() - started:.0f} s"
                      + f" data {waited / args.log_every:.3f} s/step", flush=True)
                waited = 0.0
            if step % args.val_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    vals = [run_batch(model, b, device) for b in val_loader]
                val = float(np.mean([float(v[0]) for v in vals]))
                parts = {k: np.mean([float(v[1][k]) for v in vals]) for k in vals[0][1]}
                print(f"VAL step {step} loss {val:.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()), flush=True)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "step": step, "best": min(best, val), "base": args.base, "condition": bool(getattr(model, "condition", False)), "layout": "bondpoints",
                         "group_sizes": model.group_sizes}
                save(state, args.out / "last.pt")
                if val < best:
                    best = val
                    save(state, args.out / "best.pt")
                model.train()
            if step >= args.steps or refresh:
                break
            tick = time.perf_counter()


if __name__ == "__main__":
    main()
