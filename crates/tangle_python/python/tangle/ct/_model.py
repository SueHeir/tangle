"""The scan less what the fits already explain: a model recenter and births on the residual.

Inside a packed bundle a fit's own grey is not all there is across it: each neighbour adds its bright body
and, on a phase-contrast scan, a dark halo just outside its surface. A recenter onto the brightest grey
(``recenter_on_grey``, the ridge finish) is pulled by both, so fits in a bundle sit a radius or so off their
axes, and an untraced fiber between fitted ones is a bright stripe no tracer seeds on.

Here the fits are drawn as the grey they should show, **summed** over fibers: fit ``f`` adds
``a_f · p_k(d / r_f)`` at distance ``d`` from its axis, where ``p_k`` is its type's radial profile
(``p_k(0) = 1``) and ``a_f`` its own brightness over the void. Both are measured on the scan
(:func:`measure_profiles`): the profiles by least squares over voxels near the fits, then each fit's
brightness by projection with its neighbours held. The profile carries whatever the scanner does (blur,
dark halos), so nothing about it is assumed.

* :func:`model_recenter` (``FitSettings.model_recenter``) moves each fit of the smallest type across
  itself to the offset where its own profile best matches the grey left once every *other* fit's drawn
  grey is taken off (a matched filter on the residual), smoothed along the fit, in a few sweeps.
* :func:`residual_births` (``FitSettings.residual_births``) traces fibers of the smallest type on that
  residual, divided by the fits' median brightness: a fiber no fit explains stands out at about its own
  brightness over a residual that is flat elsewhere.

Distances and radii are in voxels; an oval type's are in units of its short semi-axis (its radius), as
``_geometry.rasterize`` gives them with sections.
"""

from __future__ import annotations

import numpy as np

from ._geometry import frame, rasterize, resample, sample_image, tangents

_BINS = np.linspace(0.0, 2.5, 21)  # the profile's knots, in radii from the axis
_SAMPLES = 200_000  # voxels the profiles are measured on
_SMOOTH = 0.05  # the profile's curvature penalty, against the data term


def _groups(lines: list[np.ndarray], reach: np.ndarray) -> np.ndarray:
    """A group per fit such that no two fits in a group come within their reaches of each other, so a
    nearest-fiber raster of one group gives every voxel its distance to each fit there."""
    from scipy.spatial import cKDTree

    samples = [resample(np.asarray(line, dtype=np.float64), 1.0) for line in lines]
    points = np.vstack(samples)
    ids = np.concatenate([np.full(len(s), i) for i, s in enumerate(samples)])
    pairs = cKDTree(points).query_pairs(2.0 * float(np.max(reach)) + 1.0, output_type="ndarray")
    a, b = ids[pairs[:, 0]], ids[pairs[:, 1]]
    near = [set() for _ in lines]
    for x, y in zip(a[a != b], b[a != b]):
        near[x].add(int(y))
        near[y].add(int(x))
    group = np.full(len(lines), -1)
    for i in np.argsort([-len(n) for n in near], kind="stable"):
        used = {group[j] for j in near[i]}
        g = 0
        while g in used:
            g += 1
        group[i] = g
    return group


def _rasters(shape, lines, radii, sections):
    """Per group of mutually distant fits: ``(ids, labels, distance / radius)``, the fits of the group and a
    nearest-fiber raster of them within ``_BINS[-1]`` radii."""
    keep = [i for i, line in enumerate(lines) if len(line) >= 2]
    if not keep:
        return []
    radii = np.asarray(radii, dtype=np.float64)
    reach = _BINS[-1] * radii[keep]
    group = _groups([lines[i] for i in keep], reach)
    out = []
    for g in range(int(group.max()) + 1):
        ids = np.asarray([keep[j] for j in np.flatnonzero(group == g)])
        labels, distance, _ = rasterize(
            shape, [lines[i] for i in ids], radii[ids], reach=_BINS[-1] * radii[ids],
            sections=None if sections is None else [sections[i] for i in ids],
        )
        on = labels > 0
        owner = ids[labels[on] - 1]
        out.append((np.flatnonzero(on.ravel()), owner, distance[on] / radii[owner]))
    return out


def _hat(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolation knots and weights of ``x`` (radii) on ``_BINS``."""
    step = _BINS[1] - _BINS[0]
    t = np.clip(x / step, 0.0, len(_BINS) - 1 - 1e-9)
    k = np.floor(t).astype(int)
    return k, t - k


def draw(shape, rasters, types, amplitudes, profiles) -> np.ndarray:
    """The fits' summed grey over the void, ``(z, y, x)`` float32."""
    drawn = np.zeros(int(np.prod(shape)), dtype=np.float64)
    for flat, owner, x in rasters:
        kinds = np.asarray(types, dtype=int)[owner]
        k, w = _hat(x)
        table = np.asarray(profiles)[kinds]
        value = table[np.arange(len(k)), k] * (1 - w) + table[np.arange(len(k)), np.minimum(k + 1, len(_BINS) - 1)] * w
        np.add.at(drawn, flat, np.asarray(amplitudes)[owner] * value)
    return drawn.reshape(shape).astype(np.float32)


def measure_profiles(grey: np.ndarray, void: float, rasters, types, lines, kinds: int, *, iterations: int = 2,
                     seed: int = 0):
    """Each type's radial profile (``p(0) = 1``, on ``_BINS``) and each fit's brightness over the void,
    by least squares over up to ``_SAMPLES`` voxels near the fits. A type with no fits keeps a flat disc."""
    types = np.asarray(types, dtype=int)
    amplitudes = np.array([
        max(float(np.median(sample_image(grey, np.asarray(line)))) - void, 1e-6) if len(line) else 1e-6
        for line in lines
    ])
    nb = len(_BINS)
    profiles = np.tile(np.where(_BINS <= 1.0, 1.0, 0.0), (kinds, 1))
    if not rasters:
        return profiles, amplitudes
    near = np.unique(np.concatenate([flat for flat, _, _ in rasters]))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(near, size=min(_SAMPLES, len(near)), replace=False))
    target = grey.ravel()[chosen].astype(np.float64) - void
    # per raster: rows (into chosen), owner, knot, weight
    parts = []
    for flat, owner, x in rasters:
        at = np.searchsorted(chosen, flat)
        hit = (at < len(chosen)) & (chosen[np.minimum(at, len(chosen) - 1)] == flat)
        k, w = _hat(x[hit])
        parts.append((at[hit], owner[hit], k, w))
    present = sorted({int(types[i]) for i in range(len(lines)) if len(lines[i]) >= 2})
    for _ in range(iterations):
        design = np.zeros((len(chosen), kinds * nb))
        for rows, owner, k, w in parts:
            a = amplitudes[owner]
            col = types[owner] * nb + k
            np.add.at(design, (rows, col), a * (1 - w))
            np.add.at(design, (rows, np.minimum(col + 1, types[owner] * nb + nb - 1)), a * w)
        used = np.concatenate([np.arange(kind * nb, kind * nb + nb) for kind in present])
        # a light curvature penalty fills knots few voxels reach (the axis knot, on a fiber between voxel
        # centres) from their neighbours
        weight = _SMOOTH * float(np.median(amplitudes)) * np.sqrt(len(chosen) / nb)
        smooth = np.zeros((len(present) * (nb - 2), len(used)))
        for j in range(len(present)):
            for k in range(nb - 2):
                smooth[j * (nb - 2) + k, j * nb + k : j * nb + k + 3] = (weight, -2.0 * weight, weight)
        solution, *_ = np.linalg.lstsq(
            np.vstack([design[:, used], smooth]), np.concatenate([target, np.zeros(len(smooth))]), rcond=None
        )
        for j, kind in enumerate(present):
            p = solution[j * nb : (j + 1) * nb]
            core = float(p[_BINS <= 0.3].mean())  # the axis bin alone holds few voxels
            if core > 0:
                profiles[kind] = p / core
        # each fit's brightness, with the others held
        value = np.zeros(len(chosen))
        own = [[] for _ in lines]
        for rows, owner, k, w in parts:
            table = profiles[types[owner]]
            v = table[np.arange(len(k)), k] * (1 - w) + table[np.arange(len(k)), np.minimum(k + 1, nb - 1)] * w
            np.add.at(value, rows, amplitudes[owner] * v)
            order = np.argsort(owner, kind="stable")
            bounds = np.searchsorted(owner[order], np.arange(len(lines) + 1))
            for f in np.flatnonzero(np.diff(bounds)):
                sel = order[bounds[f] : bounds[f + 1]]
                own[f].append((rows[sel], v[sel]))
        rest = target - value
        for f, pieces in enumerate(own):
            if not pieces:
                continue
            rows = np.concatenate([p[0] for p in pieces])
            q = np.concatenate([p[1] for p in pieces])
            left = rest[rows] + amplitudes[f] * q
            amplitudes[f] = max(float((left * q).sum() / max((q * q).sum(), 1e-12)), 1e-6)
    return profiles, amplitudes


class GreyModel:
    """The fits' summed grey on ``grey`` (the denoised scan) over ``void``, remeasured by :meth:`update`."""

    def __init__(self, grey: np.ndarray, kinds: int, sections=None, void: float | None = None) -> None:
        self.grey = np.asarray(grey, dtype=np.float32)
        self.void = void  # None: the median grey beyond every fit's reach, read on the first update
        self.kinds = kinds
        self.sections = sections  # lines, radii, types -> per-line sections, or None

    def update(self, lines, radii, types) -> None:
        shape = self.grey.shape
        sections = None if self.sections is None else self.sections(lines, radii, types)
        rasters = _rasters(shape, lines, radii, sections)
        if self.void is None:
            away = np.ones(self.grey.size, dtype=bool)
            for flat, _, _ in rasters:
                away[flat] = False
            self.void = float(np.median(self.grey.ravel()[away])) if away.any() else float(np.median(self.grey))
        self.profiles, self.amplitudes = measure_profiles(self.grey, self.void, rasters, types, lines, self.kinds)
        self.drawn = draw(shape, rasters, types, self.amplitudes, self.profiles)

    def profile(self, kind: int, x: np.ndarray) -> np.ndarray:
        return np.interp(x, _BINS, self.profiles[kind], right=0.0)


def model_recenter(model: GreyModel, lines, radii, types, kind: int, *, sweeps: int, max_radii: float,
                   step: float = 0.5):
    """Fits of type ``kind`` moved across themselves onto their own grey, the other fits' drawn grey taken
    off (see the module notes); returns the lines and the mean move (voxels) of the last sweep."""
    lines = [np.array(line, dtype=np.float64) for line in lines]
    types = np.asarray(types, dtype=int)
    radii = np.asarray(radii, dtype=np.float64)
    moved = 0.0
    for _ in range(max(int(sweeps), 1)):
        model.update(lines, radii, types)
        rest = model.grey - model.void - model.drawn
        moves = []
        new = list(lines)
        for i, (line, r, k) in enumerate(zip(lines, radii, types)):
            if int(k) != kind or len(line) < 2:
                continue
            t, e1, e2 = frame(line)
            grid = np.arange(-2.2 * r, 2.2 * r + 1e-9, step)
            u, v = (a.ravel() for a in np.meshgrid(grid, grid))
            shifts = np.arange(-max_radii * r, max_radii * r + 1e-9, step)
            su, sv = (a.ravel() for a in np.meshgrid(shifts, shifts))
            inside = su**2 + sv**2 <= (max_radii * r) ** 2 + 1e-9
            su, sv = su[inside], sv[inside]
            template = model.profile(kind, np.hypot(u[None] - su[:, None], v[None] - sv[:, None]) / r)
            left = np.zeros((len(line), len(u)))
            for along in (-1.0, 0.0, 1.0):
                points = (line[:, None, :] + along * t[:, None, :] + u[None, :, None] * e1[:, None, :]
                          + v[None, :, None] * e2[:, None, :])
                left += sample_image(rest, points.reshape(-1, 3)).reshape(len(line), len(u))
            left = left / 3.0 + model.amplitudes[i] * model.profile(kind, np.hypot(u, v) / r)[None]
            best = np.argmax(left @ template.T, axis=1)
            shift = su[best][:, None] * e1 + sv[best][:, None] * e2
            if len(shift) >= 3:
                padded = np.concatenate([shift[:1], shift[:1], shift, shift[-1:], shift[-1:]])
                shift = sum(padded[j : j + len(shift)] for j in range(5)) / 5.0
            new[i] = line + shift
            moves.append(float(np.linalg.norm(shift, axis=1).mean()))
        lines = new
        moved = float(np.mean(moves)) if moves else 0.0
    return lines, moved


def residual_births(fitter, model: GreyModel, lines, radii, types, kind: int, *, level: float, median_min: float,
                    near_share: float, smooth: float = 1.0, depth_radii: float = 0.5) -> list[np.ndarray]:
    """Fibers of type ``kind`` traced on the residual (the scan less the fits' drawn grey, over the fits'
    median brightness, smoothed by ``smooth`` voxels): seeded and traced where it is above ``level``, kept
    when its median along the trace is at least ``median_min``, at most ``near_share`` of the trace lies
    within a radius of a fit already there and at most a third inside a larger type's fit."""
    from scipy.spatial import cKDTree

    from . import _native
    from ._evaluate import _samples_inside
    from ._image import HessianField
    from ._ridge import _coarse_sections

    types = np.asarray(types, dtype=int)
    own = [i for i, k in enumerate(types) if int(k) == kind and len(lines[i]) >= 2]
    if not own:  # no fit of the type to read its brightness from
        return []
    model.update(lines, radii, types)
    scale = float(np.median(model.amplitudes[own]))
    rest = _native.gaussian((model.grey - model.void - model.drawn) / max(scale, 1e-6), smooth)
    evidence = np.clip(rest, 0.0, 1.0).astype(np.float32)
    foreground = rest > level
    edt, peak = _native.foreground_depth(foreground)
    r = float(fitter.radius[kind])
    found = _native.trace_fibers(
        evidence, HessianField(evidence, sigma=max(0.6 * r, 1.0)).native, np.zeros(rest.shape, dtype=np.int32),
        edt, peak, radius=r, min_bend_radius=float(fitter.bend[kind]), step=max(0.75, 0.5 * r),
        min_length=float(fitter.min_length[kind]), node_spacing=fitter.spacing, label_offset=len(lines),
        max_fibers=None, seed_depth_radii=depth_radii, bright_seed_strength=None, peak_floor=0.0,
        claim_radii=fitter.settings.trace_claim_radii,
    )
    shape = rest.shape
    samples = [_samples_inside(np.asarray(line, dtype=np.float64), shape, 0.5) for line in lines]
    samples = [s for s in samples if len(s)]
    tree = cKDTree(np.vstack(samples)) if samples else None
    larger = _coarse_sections(fitter, lines, radii, types, kind)
    born = []
    for line in found:
        s = _samples_inside(line, shape, 0.5)
        if len(s) < 5 or np.median(sample_image(rest, s)) < median_min:
            continue
        if tree is not None and np.mean(tree.query(s)[0] < r) > near_share:
            continue
        if np.mean(larger(s)) > 1.0 / 3.0:
            continue
        born.append(line)
    return born
