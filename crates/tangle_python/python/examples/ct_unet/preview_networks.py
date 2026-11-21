"""A look at made scans' fiber orientations: per scan, three middle slices of the volume and the true centerlines
in 3D, colored by direction (red x, green y, blue z), titled with the orientation setting and the measured
orientation tensor's eigenvalues (1/3, 1/3, 1/3: fully random in 3D; 1/2, 1/2, 0: random in a plane; 1, 0, 0: all
along one axis).

usage: python preview_networks.py DATA_DIR OUT.png [PATTERN (default networks_*)]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def orientation_tensor(tan: np.ndarray) -> np.ndarray:
    """Eigenvalues (largest first) of the mean of t t^T over the centerline points (evenly spaced along fibers)."""
    t = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    return np.sort(np.linalg.eigvalsh(t.T @ t / max(len(t), 1)))[::-1]


def describe(settings: dict) -> str:
    o = settings.get("orientation", "?")
    if o == "spread":
        return f"spread {np.degrees(settings['cone']):.0f} deg"
    if o == "planar":
        return f"planar, tilt {np.degrees(settings['tilt']):.0f} deg"
    if o == "domains":
        return f"{len(settings['domains'])} domains"
    if o == "pertype":
        return "per type: " + ", ".join("random" if t["orient"]["orientation"] == "isotropic"
                                        else f"{np.degrees(t['orient']['cone']):.0f} deg" for t in settings["types"])
    return o


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    pattern = sys.argv[3] if len(sys.argv) > 3 else "networks_*"
    files = sorted(data.glob(f"{pattern}.npz"))
    if not files:
        sys.exit(f"no {pattern}.npz in {data}")
    fig = plt.figure(figsize=(16, 4 * len(files)))
    rows = []
    for row, path in enumerate(files):
        scan = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        s = meta.get("settings") or {}
        vol = scan["volume"].astype(np.float32)
        pos, tan = scan["p_pos"], scan["p_tan"]
        eig = orientation_tensor(tan)
        shapes = sorted({t.get("shape", "?") + (" ribbon" if t.get("convolution") else "") for t in s.get("types", [])})
        rows.append({"scan": path.stem, "orientation": describe(s), "eigenvalues": eig.round(3).tolist(),
                     "shapes": shapes, "fibers": meta.get("fibers")})
        title = f"{path.stem}: {describe(s)}; {', '.join(shapes)}\neigenvalues {eig[0]:.2f} {eig[1]:.2f} {eig[2]:.2f}"
        mid = [n // 2 for n in vol.shape]
        lo, hi = np.percentile(vol, [1, 99.5])
        for k, (img, name) in enumerate([(vol[mid[0]], "x middle"), (vol[:, mid[1]], "y middle"),
                                         (vol[:, :, mid[2]], "z middle")]):
            a = fig.add_subplot(len(files), 4, 4 * row + k + 1)
            a.imshow(img.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
            a.set_title(name if k else title, fontsize=9, loc="left" if not k else "center")
            a.set_xticks([]); a.set_yticks([])
        a = fig.add_subplot(len(files), 4, 4 * row + 4, projection="3d")
        keep = np.arange(0, len(pos), max(1, len(pos) // 40_000))
        color = np.abs(tan[keep]) / np.maximum(np.linalg.norm(tan[keep], axis=1, keepdims=True), 1e-12)
        a.scatter(*pos[keep].T, c=np.clip(color, 0, 1), s=0.3, depthshade=False)
        a.set_title("centerlines (red x, green y, blue z)", fontsize=9)
        a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
    print("wrote", out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
