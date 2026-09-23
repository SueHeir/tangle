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


def score(fit, truth, *, coverage_threshold: float = 0.8) -> dict[str, Any]:
    """Fiber-level and voxel-level agreement between ``fit`` and ``truth``.

    ``truth`` needs ``labels`` (one-based ids, ``(z, y, x)``), ``centerlines``
    and ``radii`` in voxel units, as :class:`SyntheticScan` provides.

    * A true fiber is **recovered** when one fitted fiber lies within its radius
      along at least ``coverage_threshold`` of its in-volume length, **split**
      when that needs several fitted fibers, **missed** otherwise.
    * A fitted fiber is **false** when most of it lies in void, and **merged**
      when less than 80% of it follows a single true fiber.
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

    recovered = split = missed = 0
    per_truth = []
    for g in range(1, count + 1):
        samples = _samples_inside(truth.centerlines[g - 1], shape)
        if len(samples) == 0:
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
        "interior_ends_fit": fit_ends["interior_ends"],
        "interior_ends_truth": truth_ends["interior_ends"],
        "implied_mean_length_fit_m": fit_ends["implied_length"] * h if fit_ends["implied_length"] else None,
        "implied_mean_length_truth_m": truth_ends["implied_length"] * h if truth_ends["implied_length"] else None,
        "per_true_fiber": per_truth,
    }
