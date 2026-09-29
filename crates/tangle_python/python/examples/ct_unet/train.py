"""Train the CT map network on make_data volumes (PyTorch, Apple GPU by default).

usage: python train.py DATA VAL OUT [--steps N] [--crop 128] [--base 16] [--resume]

DATA and VAL may each be several folders joined by commas (e.g. dense_hard-style and varied data together).

Writes OUT/last.pt every checkpoint and OUT/best.pt at the lowest validation
loss, and one line per log interval to stdout.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from maps import CHANNELS, UNet3D, augment, load, loss_terms, normalize, size_code, targets, widen

WEIGHTS = {"heat": 1.0, "offset": 1.0, "direction": 1.0, "fiber": 0.5, "radius": 1.0}


class Crops(Dataset):
    def __init__(self, files, crop: int, per_volume: int, seed: int, train: bool = True, noise: float = 0.0):
        self.files, self.crop, self.per_volume, self.seed, self.train = files, crop, per_volume, seed, train
        self.noise = noise

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
        # the scan's fiber sizes (each type's median equal-area diameter), as a user would give them: roughly
        # (x/÷ ~10%) in training, and 20% of the time not at all
        typ, rad = data["p_typ"], data["p_rad"]
        sizes = [2.0 * float(np.median(rad[typ == t])) for t in np.unique(typ)]
        if self.train:
            sizes = [] if rng.random() < 0.2 else [d * float(np.exp(rng.normal(0.0, 0.1))) for d in sizes]
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
        sample["code"] = size_code(sizes)
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sample.items()}


def save(state, path: Path):
    """Write then rename, so a reader never sees a half-written checkpoint."""
    torch.save(state, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def run_batch(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch["image"], batch["code"]) if getattr(model, "condition", False) else model(batch["image"])
    terms = loss_terms(out, batch)
    return sum(WEIGHTS[k] * v for k, v in terms.items()), terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("val")
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init", type=Path, help="start from this checkpoint's weights (fine-tuning)")
    parser.add_argument("--condition", action="store_true", help="add the fiber-size conditioning (FiLM)")
    parser.add_argument("--widen", type=int, help="with --init: widen that network to this base width (Net2Net)")
    parser.add_argument("--noise", type=float, default=0.0, help="correlated-noise augmentation sd (0 = off)")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.widen:
        model = widen(load(args.init, "cpu"), args.widen).to(device)
        args.base = args.widen
    elif args.init:
        source = load(args.init, "cpu")
        args.base = source.down[0][0].out_channels
        if source.head.out_channels == CHANNELS:
            model = source.to(device)
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
                                                       "out": str(args.out), "init": str(args.init), "weights": WEIGHTS}, indent=1) + "\n")

    val_files = sorted(f for d in args.val.split(",") for f in Path(d).glob("*.npz"))
    val_loader = DataLoader(Crops(val_files, args.crop, 1, 0, train=False), batch_size=1, num_workers=2)
    started, window = time.perf_counter(), []
    while step < args.steps:
        # Re-list each pass so volumes still being generated join as they land.
        files = sorted(f for d in args.data.split(",") for f in Path(d).glob("*.npz"))
        loader = DataLoader(Crops(files, args.crop, 2, step, noise=args.noise), batch_size=1, shuffle=True, num_workers=args.workers,
                            persistent_workers=False, prefetch_factor=2)
        model.train()
        for batch in loader:
            loss, terms = run_batch(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            window.append({k: float(v.detach()) for k, v in terms.items()})
            if step % args.log_every == 0:
                mean = {k: np.mean([w[k] for w in window]) for k in window[0]}
                window = []
                print(f"step {step} files {len(files)} " + " ".join(f"{k} {v:.4f}" for k, v in mean.items())
                      + f" lr {scheduler.get_last_lr()[0]:.2e} {time.perf_counter() - started:.0f} s", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    vals = [run_batch(model, b, device) for b in val_loader]
                val = float(np.mean([float(v[0]) for v in vals]))
                parts = {k: np.mean([float(v[1][k]) for v in vals]) for k in vals[0][1]}
                print(f"VAL step {step} loss {val:.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()), flush=True)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "step": step, "best": min(best, val), "base": args.base, "condition": bool(getattr(model, "condition", False)),
                         "group_sizes": model.group_sizes}
                save(state, args.out / "last.pt")
                if val < best:
                    best = val
                    save(state, args.out / "best.pt")
                model.train()
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
