"""Pictures of how a simulated CT scan is made, for docs/ct_synthetic.md.

usage: python make_figures.py CACHE_DIR OUT_DIR [--ovito PYTHON_WITH_OVITO]

CACHE_DIR holds ct_examples truth caches (``dense_hard_7-*.json``); a missing one is relaxed and written.
Writes into OUT_DIR:

* ``ct-synth-geometry.png``   the Tangle structure (dense_hard_7), rendered in OVITO when --ovito is given
* ``ct-synth-object.png``     one slice of what the scanner sees: labels, occupancy, attenuation, phase
* ``ct-synth-acquisition.png`` the acquisition, stage by stage, for that slice
* ``ct-synth-settings.png``   the same slice under different scanner settings
* ``ct-synth-sets.png``       one slice of several dense_hard and scanned structures
"""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import ct_examples as ex  # noqa: E402
from tangle import ct  # noqa: E402
from tangle.ct import _native, _scanner  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("cache", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--ovito")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
um = 1e-6
Z = 80  # the slice shown (z, voxels)


def capture(index, settings=ex.dense_hard_settings):
    """The truth assembly and render arguments of a ct_examples structure, plus its scan."""
    seen = {}
    original = ex.render_scan

    def spy(*a, **k):
        seen["args"], seen["kwargs"] = a, k
        return original(*a, **k)

    ex.render_scan = spy
    try:
        example = ex.scanned(index, settings)(args.cache / f"{'dense_hard' if settings is ex.dense_hard_settings else 'scanned'}_{index}.json")
    finally:
        ex.render_scan = original
    return example, seen["args"], seen["kwargs"]


example, (truth, voxel), kwargs = capture(7)
scan = example.scan
print("dense_hard_7:", len(scan.centerlines), "fibers")

# ---- 1. geometry --------------------------------------------------------------------------------------------
dump = args.out / "geometry.dump"
rows = []
lengths = np.array(truth.cell.lengths) / um
for f, (line, t) in enumerate(zip(scan.centerlines, scan.types)):
    l = np.asarray(line) * voxel / um
    r = scan.radii[f] * voxel / um
    for k in range(len(l) - 1):
        a, b = l[k], l[k + 1]
        d = b - a
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d /= n
        ax = np.cross([0.0, 0.0, 1.0], d)
        s = np.linalg.norm(ax)
        ang = np.arctan2(s, d[2])
        q = np.r_[ax / s * np.sin(ang / 2), np.cos(ang / 2)] if s > 1e-9 else np.array([0, 0, 0, 1.0])
        rows.append((f + 1, r, n, q, 0.5 * (a + b), int(t)))
with open(dump, "w") as fh:
    fh.write(f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(rows)}\nITEM: BOX BOUNDS ff ff ff\n")
    for L in lengths:
        fh.write(f"0 {L:.6g}\n")
    fh.write("ITEM: ATOMS id mol type AsphericalShape.X AsphericalShape.Y AsphericalShape.Z quati quatj quatk quatw "
             "x y z fiber_type\n")
    for i, (m, r, length, q, c, t) in enumerate(rows, 1):
        fh.write(f"{i} {m} 1 {r:.3f} {r:.3f} {length:.3f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                 f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {t}\n")
if args.ovito:
    render = args.out / "render_geometry.py"
    render.write_text(f'''
from ovito.io import import_file
from ovito.modifiers import ColorCodingModifier
from ovito.vis import ParticlesVis, Viewport
p = import_file({str(dump)!r}); p.add_to_scene()
p.compute().particles.vis.shape = ParticlesVis.Shape.Spherocylinder
p.modifiers.append(ColorCodingModifier(property="fiber_type", start_value=0, end_value=1.4,
                                       gradient=ColorCodingModifier.Viridis()))
v = Viewport(type=Viewport.Type.Perspective, camera_dir=(-1, -0.6, -0.7)); v.zoom_all()
v.render_image(filename={str(args.out / "ct-synth-geometry.png")!r}, size=(1400, 1050), background=(1, 1, 1))
''')
    subprocess.run([args.ovito, str(render)], check=True)
    render.unlink()
    # trim the white margin around the render
    img = plt.imread(args.out / "ct-synth-geometry.png")
    rows, cols = np.nonzero(img[..., :3].min(axis=2) < 0.97)
    pad = 20
    img = img[max(rows.min() - pad, 0):rows.max() + pad, max(cols.min() - pad, 0):cols.max() + pad]
    plt.imsave(args.out / "ct-synth-geometry.png", img)
dump.unlink()

# ---- 2. the object: what the scanner is given ----------------------------------------------------------------
captured = {}
original_acquire = _scanner.acquire


def spy_acquire(attenuation, scanner, rng, voxel_size=None, phase=None, warp=None):
    captured.update(attenuation=attenuation, phase=phase, scanner=scanner, voxel_size=voxel_size)
    return original_acquire(attenuation, scanner, rng, voxel_size, phase, warp)


_scanner.acquire = spy_acquire
still = ct.synthetic_ct(truth, voxel, **{**kwargs, "scanner": replace(kwargs["scanner"], fiber_motion=0.0)})
_scanner.acquire = original_acquire
att, phase, scanner = captured["attenuation"], captured["phase"], captured["scanner"]

fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
cmap = plt.get_cmap("tab20")
lab = scan.labels[Z].astype(float)
ax[0].imshow(np.where(lab > 0, lab % 20, np.nan), cmap=cmap, interpolation="nearest")
ax[0].set_title("1. fibers, one colour each (labels)")
ax[1].imshow(att[Z], cmap="gray")
ax[1].set_title("2. attenuation per voxel\n(occupancy x brightness: dim coarse, dimmer core)")
ax[2].imshow(phase[Z] if phase is not None else att[Z], cmap="magma")
ax[2].set_title("3. phase shift per voxel\n(attenuation x delta/beta, per material)")
ax[3].imshow(scan.volume[Z], cmap="gray")
ax[3].set_title("4. the finished scan (same slice)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.subplots_adjust(top=0.8)
fig.suptitle("dense_hard_7, slice z = 80 (160 x 160 um, 1 um voxels)")
fig.savefig(args.out / "ct-synth-object.png", dpi=100)
plt.close(fig)

# ---- 3. the acquisition, stage by stage (a slab around the slice, the same code as _scanner.acquire) ----------
slab = slice(Z - 12, Z + 12)
a3, p3 = att[slab], phase[slab]
nz, ny, nx = a3.shape
width = int(np.ceil(np.hypot(ny, nx))) + 4
count = width
angles = np.arange(count) * (np.pi / count)
line = _native.project(a3, angles, width)          # (angle, z, u)
pline = _native.project(p3, angles, width)
absorption = np.exp(-line)
propagated = _scanner._propagate(line, pline, scanner.propagation)
blur = scanner.resolution / (2.0 * np.sqrt(2.0 * np.log(2.0)) * voxel)
from scipy.ndimage import gaussian_filter  # noqa: E402

blurred = gaussian_filter(propagated, (0, blur, blur))
rng = np.random.default_rng(1)
counts = rng.poisson(np.clip(blurred, 0, None) * scanner.photons).astype(np.float32)
measured = -np.log(np.maximum(counts, 0.5) / scanner.photons)
clean = -np.log(np.maximum(blurred, 1e-6))
recon_clean = _native.back_project(_scanner._filter(clean), angles, a3.shape) * (np.pi / count)
recon_noisy = _native.back_project(_scanner._filter(measured), angles, a3.shape) * (np.pi / count)
k = 12  # the shown slice inside the slab
fig, ax = plt.subplots(2, 4, figsize=(18, 9))
ax[0, 0].imshow(line[:, k, :], cmap="gray_r", aspect="auto"); ax[0, 0].set_title("a. projections (sinogram):\nattenuation summed along each ray, 180 deg")
u0 = width // 2 - 40
ax[0, 1].plot(absorption[0, k, u0:u0 + 80], label="absorption only")
ax[0, 1].plot(propagated[0, k, u0:u0 + 80], label="after free-space propagation")
ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("b. one projection row: propagation\nadds bright/dark fringes at every edge")
ax[0, 2].plot(blurred[0, k, u0:u0 + 80], label="detector blur")
ax[0, 2].plot(counts[0, k, u0:u0 + 80] / scanner.photons, lw=0.8, label="photon counts")
ax[0, 2].legend(fontsize=8); ax[0, 2].set_title(f"c. detector: blur (resolution {scanner.resolution / um:g} um),\nPoisson noise ({scanner.photons:g} photons/pixel)")
ax[0, 3].imshow(measured[:, k, :], cmap="gray_r", aspect="auto"); ax[0, 3].set_title("d. measured sinogram: -log(counts / flat)")
ax[1, 0].imshow(a3[k], cmap="gray"); ax[1, 0].set_title("e. the object (attenuation)")
ax[1, 1].imshow(recon_clean[k], cmap="gray"); ax[1, 1].set_title("f. filtered back-projection, no noise:\nblur and phase halos")
ax[1, 2].imshow(recon_noisy[k], cmap="gray"); ax[1, 2].set_title("g. with photon noise")
ax[1, 3].imshow(scan.volume[Z], cmap="gray"); ax[1, 3].set_title("h. the example as rendered\n(plus 1 um fiber motion)")
for a in ax[1]:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(args.out / "ct-synth-acquisition.png", dpi=100)
plt.close(fig)

# ---- 4. scanner settings -------------------------------------------------------------------------------------
base = kwargs["scanner"]
variants = [
    ("as dense_hard", base, {}),
    ("resolution 3 um", replace(base, resolution=3 * um), {}),
    ("resolution 1.2 um", replace(base, resolution=1.2 * um), {}),
    ("4x fewer photons", replace(base, photons=base.photons / 4), {}),
    ("no propagation (no halo)", replace(base, propagation=0.0), {}),
    ("2x propagation", replace(base, propagation=2 * base.propagation), {}),
    ("noise blur 1 px (blotchy)", replace(base, noise_blur=1.0, photons=base.photons / 23), {}),
    ("fiber motion 2.5 um", replace(base, fiber_motion=2.5 * um), {}),
    ("no brightness spread", base, {"brightness_spread": 0.0}),
    ("brightness spread 0.5", base, {"brightness_spread": 0.5}),
    ("ring artifacts", replace(base, ring_strength=0.01), {}),
    ("sample drift 3 um", replace(base, drift=3 * um), {}),
]
fig, ax = plt.subplots(3, 4, figsize=(16, 12.5))
for a, (title, sc, extra) in zip(ax.flat, variants):
    s = ct.synthetic_ct(truth, voxel, **{**kwargs, "scanner": sc, **extra})
    a.imshow(s.volume[Z, 20:120, 20:120], cmap="gray")
    a.set_title(title)
    a.set_xticks([]); a.set_yticks([])
    print("  ", title)
fig.suptitle("the same 100 x 100 um of dense_hard_7 under different scanner settings", fontsize=14)
fig.tight_layout()
fig.savefig(args.out / "ct-synth-settings.png", dpi=90)
plt.close(fig)

# ---- 5. the sets ---------------------------------------------------------------------------------------------
picks = [("dense_hard", i) for i in (1, 7, 12)] + [("scanned", i) for i in (101, 103, 106)]
fig, ax = plt.subplots(2, 3, figsize=(13, 9))
for a, (kind, i) in zip(ax.flat, picks):
    settings = ex.dense_hard_settings if kind == "dense_hard" else ex.scanned_settings
    e = ex.scanned(i, settings)(args.cache / f"{kind}_{i}.json")
    v = settings(i)
    a.imshow(e.scan.volume[:, e.scan.volume.shape[1] // 2, :], cmap="gray", origin="lower")
    a.set_title(f"{kind}_{i}: {v['voxel_um']:g} um voxels, fine {v['fine_um']:g} um,\n"
                f"bundles of {v['per_bundle']}, coarse {v['coarse_um'] or 'none'}", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.subplots_adjust(top=0.9, hspace=0.25)
fig.suptitle("xz slices of test structures (z up): dense_hard (top) and scanned (bottom)")
fig.savefig(args.out / "ct-synth-sets.png", dpi=95)
plt.close(fig)
print("wrote", sorted(p.name for p in args.out.glob("ct-synth-*.png")))
