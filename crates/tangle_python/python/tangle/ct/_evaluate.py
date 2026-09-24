"""Score a fit against known ground truth (for example a :func:`synthetic_ct`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._ends import end_statistics
from ._geometry import resample


def _samples_inside(line: np.ndarray, shape: tuple[int, int, int], spacing: float = 0.5) -> np.ndarray:
    dense = resample(line, spacing)
    upper = np.array(shape[::-1], dtype=np.float64)
    inside = np.all((dense >= 0) & (dense < upper), axis=1)
    return dense[inside]


def _voxel(points: np.ndarray, shape) -> tuple[np.ndarray, ...]:
    index = np.clip(np.floor(points).astype(int), 0, np.array(shape[::-1]) - 1)
    return index[:, 2], index[:, 1], index[:, 0]


def centerline_agreement(
    fit_lines: list[np.ndarray],
    truth_lines: list[np.ndarray],
    truth_radii: np.ndarray,
    shape: tuple[int, int, int],
    *,
    tolerance_radii: float = 0.5,
    skip: set[int] | frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """How much of the fibers' centerlines the fit traces, fiber by fiber.

    Only the centerlines count, not how the capsule edges fill voxels. Each
    fitted fiber belongs to the true fiber whose centerline most of its own
    lies on (within ``tolerance_radii`` of that fiber's radius).

    * **recall**: the fraction of true centerline length (in the volume;
      true fibers in ``skip``, zero-based, are left out) that the fitted
      fiber belonging to that true fiber passes within tolerance of;
    * **precision**: the fraction of fitted centerline length that lies
      within tolerance of the centerline of the true fiber it belongs to;
    * **f1**: their harmonic mean.

    A fiber split into pieces still counts where each piece follows it
    (pieces of one true fiber all belong to it); a fit that follows one
    fiber and then another loses the length along the second.
    """
    from scipy.spatial import cKDTree

    radii = np.asarray(truth_radii, dtype=np.float64)
    truth_samples = [_samples_inside(line, shape, 0.5) for line in truth_lines]
    reference = [resample(np.asarray(line, dtype=np.float64), 0.25) for line in truth_lines]
    all_truth = np.vstack([r for r in reference if len(r)]) if any(len(r) for r in reference) else np.zeros((0, 3))
    truth_id = np.concatenate([np.full(len(r), g) for g, r in enumerate(reference)]) if len(all_truth) else np.zeros(0, int)
    fit_samples = [_samples_inside(line, shape, 0.5) for line in fit_lines]
    owner = np.full(len(fit_lines), -1)
    on_own = 0
    fit_total = sum(len(samples) for samples in fit_samples)
    if len(all_truth) and fit_total:
        tree = cKDTree(all_truth)
        for f, samples in enumerate(fit_samples):
            if not len(samples):
                continue
            distance, index = tree.query(samples)
            nearest = truth_id[index]
            on = distance <= tolerance_radii * radii[nearest]
            if not on.any():
                continue
            owner[f] = int(np.bincount(nearest[on]).argmax())
            on_own += int((on & (nearest == owner[f])).sum())
    traced = total = 0
    for g, samples in enumerate(truth_samples):
        if g in skip or not len(samples):
            continue
        total += len(samples)
        mine = [fit_samples[f] for f in np.flatnonzero(owner == g) if len(fit_samples[f])]
        if not mine:
            continue
        distance, _ = cKDTree(np.vstack(mine)).query(samples)
        traced += int((distance <= tolerance_radii * radii[g]).sum())
    recall = traced / total if total else None  # None: nothing to trace / nothing fitted
    precision = on_own / fit_total if fit_total else None
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12) if recall is not None and precision is not None else None,
        "tolerance_radii": tolerance_radii,
    }


def score(fit, truth, *, coverage_threshold: float = 0.8, min_length: float | None = None) -> dict[str, Any]:
    """Fiber-level and voxel-level agreement between ``fit`` and ``truth``.

    ``truth`` needs ``labels`` (one-based ids, ``(z, y, x)``), ``centerlines``
    and ``radii`` in voxel units, as :class:`SyntheticScan` provides.

    * A true fiber is **recovered** when one fitted fiber lies within its radius
      along at least ``coverage_threshold`` of its in-volume length, **split**
      when that needs several fitted fibers, **missed** otherwise.
    * A fitted fiber is **false** when most of it lies in void, and **merged**
      when less than 80% of it follows a single true fiber.
    * True pieces shorter in the volume than ``min_length`` (meters; default
      the fit's own minimum length, 3 diameters unless set) are stubs where a
      fiber clips a corner of the scan. The fitter drops fits that short, so
      stubs are counted separately and left out of recall.
    * ``centerline_*``: :func:`centerline_agreement` (stubs left out), the
      share of the true centerlines the right fitted fiber traces within
      half a radius, and of the fitted centerlines that lie on their own
      fiber. Unlike ``voxel_label_accuracy`` it ignores how the capsule
      edges fill voxels.
    """
    from scipy.spatial import cKDTree

    shape = fit.shape
    truth_labels = np.asarray(truth.labels)
    count = len(truth.centerlines)
    fit_samples = [_samples_inside(line, shape) for line in fit.centerlines]
    owner = []
    purity = []
    for samples in fit_samples:
        if len(samples) == 0:
            owner.append(0)
            purity.append(0.0)
            continue
        hits = truth_labels[_voxel(samples, shape)]
        values, counts = np.unique(hits, return_counts=True)
        best = int(np.argmax(counts))
        owner.append(int(values[best]))
        purity.append(float(counts[best]) / len(hits))
    owner = np.array(owner)
    purity = np.array(purity)
    trees = [cKDTree(s) if len(s) else None for s in fit_samples]

    if min_length is None:
        min_length = fit.spec.min_length or 3.0 * fit.spec.diameter
    shortest = min_length / fit.voxel_size
    recovered = split = missed = stubs = 0
    per_truth = []
    for g in range(1, count + 1):
        samples = _samples_inside(truth.centerlines[g - 1], shape)
        if len(samples) == 0:
            continue
        if 0.5 * len(samples) < shortest:  # samples are 0.5 voxel apart
            stubs += 1
            per_truth.append({"id": g, "state": "stub", "coverage": None, "union_coverage": None, "fitted": 0})
            continue
        radius = float(truth.radii[g - 1])
        mapped = [f for f in np.flatnonzero(owner == g) if trees[f] is not None]
        near = np.zeros((len(mapped), len(samples)), dtype=bool)
        for row, f in enumerate(mapped):
            distance, _ = trees[f].query(samples)
            near[row] = distance <= radius
        single = float(near.mean(axis=1).max()) if mapped else 0.0
        union = float(near.any(axis=0).mean()) if mapped else 0.0
        if single >= coverage_threshold:
            recovered += 1
            state = "recovered"
        elif union >= coverage_threshold and len(mapped) > 1:
            split += 1
            state = "split"
        else:
            missed += 1
            state = "missed"
        per_truth.append({"id": g, "state": state, "coverage": single, "union_coverage": union, "fitted": len(mapped)})
    present = recovered + split + missed

    false_positive = int(((owner == 0) | (purity < 0.5)).sum())
    merged = int(((owner > 0) & (purity >= 0.5) & (purity < 0.8)).sum())

    errors, radius_errors = [], []
    truth_trees = {}
    for f, samples in enumerate(fit_samples):
        g = owner[f]
        if g == 0 or purity[f] < 0.8 or len(samples) == 0:
            continue
        if g not in truth_trees:
            truth_trees[g] = cKDTree(resample(truth.centerlines[g - 1], 0.25))
        distance, _ = truth_trees[g].query(samples)
        errors.append(float(distance.mean()))
        radius_errors.append(float(fit.radii[f] - truth.radii[g - 1]))

    fit_labels = fit.label_volume()
    mapping = np.concatenate([[0], owner]).astype(np.int32)
    mapped_labels = mapping[fit_labels]
    truth_solid = truth_labels > 0
    fit_solid = fit_labels > 0
    dice = 2.0 * float((truth_solid & fit_solid).sum()) / max(float(truth_solid.sum() + fit_solid.sum()), 1.0)
    label_accuracy = float((mapped_labels[truth_solid] == truth_labels[truth_solid]).mean()) if truth_solid.any() else 0.0

    h = fit.voxel_size
    per_type = None
    truth_types = getattr(truth, "types", None)
    if truth_types is not None and getattr(fit, "types", None) is not None:
        per_type = {}
        states = {entry["id"]: entry["state"] for entry in per_truth}
        for kind in np.unique(truth_types):
            ids = [g for g in range(1, count + 1) if truth_types[g - 1] == kind and g in states]
            mine = [f for f in range(len(owner)) if owner[f] > 0 and truth_types[owner[f] - 1] == kind]
            per_type[int(kind)] = {
                "true_fibers": sum(states[g] != "stub" for g in ids),
                "recovered": sum(states[g] == "recovered" for g in ids),
                "split": sum(states[g] == "split" for g in ids),
                "missed": sum(states[g] == "missed" for g in ids),
                "fitted": len(mine),
                "fitted_as_this_type": sum(int(fit.types[f]) == int(kind) for f in mine),
            }
    stub_ids = {entry["id"] - 1 for entry in per_truth if entry["state"] == "stub"}
    centerline = centerline_agreement(
        fit.centerlines, truth.centerlines, np.asarray(truth.radii), shape, skip=stub_ids
    )
    fit_ends = end_statistics(fit.centerlines, fit.radii, shape)
    truth_ends = end_statistics(truth.centerlines, np.asarray(truth.radii), shape)
    precision = (len(owner) - false_positive) / max(len(owner), 1)
    recall = recovered / max(present, 1)
    return {
        "true_fibers_in_volume": present,
        "fitted_fibers": int(len(owner)),
        "recovered": recovered,
        "split": split,
        "missed": missed,
        "stubs": stubs,
        "false_fibers": false_positive,
        "merged_fibers": merged,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "centerline_error_voxels": float(np.mean(errors)) if errors else None,
        "centerline_error_m": float(np.mean(errors)) * h if errors else None,
        "diameter_bias_m": 2 * float(np.mean(radius_errors)) * h if radius_errors else None,
        "diameter_rms_error_m": 2 * float(np.sqrt(np.mean(np.square(radius_errors)))) * h if radius_errors else None,
        "solid_dice": dice,
        "voxel_label_accuracy": label_accuracy,
        "centerline_recall": centerline["recall"],
        "centerline_precision": centerline["precision"],
        "centerline_f1": centerline["f1"],
        "interior_ends_fit": fit_ends["interior_ends"],
        "interior_ends_truth": truth_ends["interior_ends"],
        "implied_mean_length_fit_m": fit_ends["implied_length"] * h if fit_ends["implied_length"] else None,
        "implied_mean_length_truth_m": truth_ends["implied_length"] * h if truth_ends["implied_length"] else None,
        "per_type": per_type,
        "per_true_fiber": per_truth,
        "per_fitted_fiber": [
            {
                "id": f + 1,
                "true_id": int(owner[f]),
                "purity": float(purity[f]),
                "state": "false" if owner[f] == 0 or purity[f] < 0.5 else "merged" if purity[f] < 0.8 else "matched",
            }
            for f in range(len(owner))
        ],
    }


def geometry_report(
    centerlines: list[np.ndarray],
    radii: np.ndarray,
    min_bend_radius: float,
    *,
    penetration_tolerance: float = 0.05,
    spacing: float | None = None,
) -> dict[str, Any]:
    """How far fibers are from being valid Tangle fibers (voxel units).

    ``spacing`` resamples every fiber to that node spacing first. Discrete
    curvature depends on the spacing: resampling a polyline more finely than
    its own segments puts nodes at its corners, where the turn per unit
    length roughly doubles. Compare fits at the spacing Tangle uses for them
    (1.25 diameters in ``engine="tangle"``).

    * ``curvature_ratio_max`` / ``_p95``: largest discrete curvature per fiber
      times ``min_bend_radius`` (1 is the bend limit);
      ``fibers_over_bend_limit`` counts fibers above 1.01.
    * ``max_penetration_radii``: deepest overlap between two different fibers'
      capsules, in radii; ``overlapping_pairs`` counts fiber pairs that
      overlap by more than ``penetration_tolerance`` radii.
    * ``min_segment_diameters``: shortest segment in diameters, and
      ``segment_length_cv`` the spread of segment lengths.
    """
    from scipy.spatial import cKDTree

    from ._refine import curvature_ratio

    radii = np.asarray(radii, dtype=np.float64)
    lines = [np.asarray(line, dtype=np.float64) for line in centerlines if len(line) >= 2]
    if spacing:
        lines = [resample(line, spacing) for line in lines]
    keep = [i for i, line in enumerate(centerlines) if len(line) >= 2]
    radii = radii[keep]
    report: dict[str, Any] = {"fibers": len(lines)}
    if not lines:
        return report
    ratios = curvature_ratio(lines, min_bend_radius)
    report["curvature_ratio_max"] = float(ratios.max())
    report["curvature_ratio_p95"] = float(np.percentile(ratios, 95))
    report["fibers_over_bend_limit"] = int((ratios > 1.01).sum())

    spacing = 0.25 * float(radii.min())
    dense = [resample(line, spacing) for line in lines]
    points = np.concatenate(dense)
    owner = np.concatenate([np.full(len(d), i) for i, d in enumerate(dense)])
    pairs = cKDTree(points).query_pairs(2.0 * float(radii.max()), output_type="ndarray")
    worst, overlapping = 0.0, 0
    if len(pairs):
        pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
    if len(pairs):
        a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
        distance = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        scale = 0.5 * (radii[a] + radii[b])
        depth = (radii[a] + radii[b] - distance) / scale
        worst = float(max(depth.max(), 0.0))
        deep = depth > penetration_tolerance
        overlapping = len({(min(i, j), max(i, j)) for i, j in zip(a[deep], b[deep])})
    report["max_penetration_radii"] = worst
    report["overlapping_pairs"] = overlapping

    segments = np.concatenate([np.linalg.norm(np.diff(line, axis=0), axis=1) for line in lines])
    diameters = np.concatenate([np.full(len(line) - 1, 2.0 * r) for line, r in zip(lines, radii)])
    report["min_segment_diameters"] = float((segments / diameters).min())
    report["segment_length_cv"] = float(segments.std() / max(segments.mean(), 1e-12))
    report["total_length"] = float(segments.sum())
    return report
