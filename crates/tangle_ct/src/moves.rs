//! Discrete topology moves between solver batches, as `tangle/ct/_moves.py`:
//! remove duplicates and unsupported fibers, merge fragments that continue
//! each other, split kinks, and decide whether side-by-side fits are one
//! fiber. Births are re-traces (see `trace`).

use std::collections::{HashMap, HashSet};

use crate::line::{add, dot, norm, polyline_length, scale, sub, tangents, Point};
use crate::render::{
    crop, local_box, local_residual, near_box, render_occupancy, squared_residual, Corner,
};
use crate::sample::trilinear;
use crate::Shape;

/// Soft edge half-width of rendered capsules, voxels.
pub const EDGE: f64 = 1.2;

/// Points bucketed on a cubic grid, for neighbor searches within a radius.
struct PointGrid {
    points: Vec<Point>,
    cell: f64,
    buckets: HashMap<[i64; 3], Vec<usize>>,
}

impl PointGrid {
    fn new(points: Vec<Point>, cell: f64) -> Self {
        let cell = cell.max(1e-6);
        let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (index, p) in points.iter().enumerate() {
            buckets.entry(Self::key(p, cell)).or_default().push(index);
        }
        Self {
            points,
            cell,
            buckets,
        }
    }

    fn key(p: &Point, cell: f64) -> [i64; 3] {
        [
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        ]
    }

    /// `(distance, index)` of every point within `radius` of `p` (unsorted).
    fn within(&self, p: Point, radius: f64) -> Vec<(f64, usize)> {
        let reach = (radius / self.cell).ceil() as i64;
        let center = Self::key(&p, self.cell);
        let mut out = Vec::new();
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    let key = [center[0] + dx, center[1] + dy, center[2] + dz];
                    if let Some(bucket) = self.buckets.get(&key) {
                        for &index in bucket {
                            let d = norm(sub(self.points[index], p));
                            if d <= radius {
                                out.push((d, index));
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn mean_sample(image: &[f32], shape: Shape, points: &[Point]) -> f64 {
    points
        .iter()
        .map(|&p| trilinear(image, shape, p, 0.0))
        .sum::<f64>()
        / points.len() as f64
}

/// Sorted by `key`, ties keeping their order (NumPy's stable argsort).
fn argsort_by(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    order
}

/// Fibers mostly lying inside another fiber removed, and end runs that do
/// trimmed (`_moves.trim_duplicates`). Two distinct fibers never have
/// centerlines closer than the sum of their radii, so nodes within
/// `closeness` times the larger of the two radii of another fiber's nodes
/// are re-traces of it (the larger: a thin fit inside a thick fiber sits
/// off its axis by up to the difference of the radii).
pub fn trim_duplicates(
    lines: &[Vec<Point>],
    radii: &[f64],
    min_length: f64,
    closeness: f64,
) -> (Vec<Vec<Point>>, Vec<f64>) {
    let mut lines = lines.to_vec();
    let lengths: Vec<f64> = lines.iter().map(|l| polyline_length(l)).collect();
    let mut removed = vec![false; lines.len()];
    // One grid over every node; trimming only drops nodes, so a node that
    // is gone is skipped through `alive`.
    let mut owner = Vec::new();
    let mut slot = Vec::new();
    let mut points = Vec::new();
    for (k, line) in lines.iter().enumerate() {
        for (m, &p) in line.iter().enumerate() {
            points.push(p);
            owner.push(k);
            slot.push(m);
        }
    }
    let largest = radii.iter().cloned().fold(0.0, f64::max);
    let grid = PointGrid::new(points, closeness * largest);
    // Node range of each line still in it (trimming keeps a contiguous run).
    let mut alive: Vec<(usize, usize)> = lines.iter().map(|l| (0, l.len())).collect();
    for index in argsort_by(&lengths) {
        if (0..lines.len()).all(|j| j == index || removed[j]) {
            continue;
        }
        let line = &lines[index];
        let covered: Vec<bool> = line
            .iter()
            .map(|&p| {
                grid.within(p, closeness * largest).iter().any(|&(d, q)| {
                    let k = owner[q];
                    d < closeness * radii[index].max(radii[k])
                        && k != index
                        && !removed[k]
                        && (alive[k].0..alive[k].1).contains(&slot[q])
                })
            })
            .collect();
        if !covered.is_empty() && covered.iter().filter(|&&c| c).count() * 2 > covered.len() {
            removed[index] = true;
            continue;
        }
        let (mut start, mut stop) = (0, line.len());
        while start < stop && covered[start] {
            start += 1;
        }
        while stop > start && covered[stop - 1] {
            stop -= 1;
        }
        let trimmed = line[start..stop].to_vec();
        if trimmed.len() < 2 || polyline_length(&trimmed) < min_length {
            removed[index] = true;
        } else {
            lines[index] = trimmed;
            let offset = alive[index].0;
            alive[index] = (offset + start, offset + stop);
        }
    }
    keep_unremoved(lines, radii, &removed)
}

fn keep_unremoved(
    lines: Vec<Vec<Point>>,
    radii: &[f64],
    removed: &[bool],
) -> (Vec<Vec<Point>>, Vec<f64>) {
    let mut out_lines = Vec::new();
    let mut out_radii = Vec::new();
    for (k, line) in lines.into_iter().enumerate() {
        if !removed[k] {
            out_lines.push(line);
            out_radii.push(radii[k]);
        }
    }
    (out_lines, out_radii)
}

/// Fibers shorter than `min_length` or whose mean image value is below
/// `min_support` removed (`_moves.remove_unsupported`).
pub fn remove_unsupported(
    image: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    min_length: f64,
    min_support: f64,
) -> (Vec<Vec<Point>>, Vec<f64>) {
    let removed: Vec<bool> = lines
        .iter()
        .map(|line| {
            !(line.len() >= 2
                && polyline_length(line) >= min_length
                && mean_sample(image, shape, line) >= min_support)
        })
        .collect();
    keep_unremoved(lines.to_vec(), radii, &removed)
}

/// End point and outward unit tangent (over up to three node spacings).
fn end_of(line: &[Point], which: usize) -> (Point, Point) {
    let count = (line.len() - 1).min(3);
    let (tip, inner) = if which == 0 {
        (line[0], line[count])
    } else {
        (line[line.len() - 1], line[line.len() - 1 - count])
    };
    let tangent = sub(tip, inner);
    (tip, scale(tangent, 1.0 / norm(tangent).max(1e-12)))
}

/// `line` reversed.
fn reversed(line: &[Point]) -> Vec<Point> {
    line.iter().rev().copied().collect()
}

/// `n` evenly spaced points from `a` to `b` (`np.linspace`).
fn linspace_points(a: Point, gap: Point, n: usize) -> Vec<Point> {
    let step = 1.0 / (n - 1) as f64;
    (0..n)
        .map(|i| {
            let t = if i + 1 == n { 1.0 } else { i as f64 * step };
            add(a, scale(gap, t))
        })
        .collect()
}

/// Settings shared by the join moves.
#[derive(Clone, Copy, Debug)]
pub struct MergeSettings {
    pub max_gap: f64,
    pub max_angle_degrees: f64,
    pub min_bridge_support: f64,
    pub min_bend_radius: Option<f64>,
    pub kink_threshold: f64,
    pub end_cost: f64,
    pub scale: f64,
    pub max_prior_gap: Option<f64>,
    pub max_prior_angle_degrees: f64,
}

/// Pairs of ends that continue each other across a short gap joined
/// (`_moves.merge_fragments`). With an `end_cost`, the fixed tests give way
/// to [`merge_with_prior`]. Returns the lines, radii and number of joins.
pub fn merge_fragments(
    image: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    settings: MergeSettings,
) -> (Vec<Vec<Point>>, Vec<f64>, usize) {
    if settings.end_cost > 0.0 {
        return merge_with_prior(
            image,
            shape,
            lines,
            radii,
            settings.max_prior_gap.unwrap_or(4.0 * settings.max_gap),
            settings.max_prior_angle_degrees,
            settings.min_bend_radius,
            settings.kink_threshold,
            settings.end_cost,
            settings.scale,
        );
    }
    // Each line carries an identity, so a rejected pair stays rejected
    // while other joins rebuild the list.
    let mut lines: Vec<(usize, Vec<Point>)> = lines.iter().cloned().enumerate().collect();
    let mut next_id = lines.len();
    let mut radii = radii.to_vec();
    let cos_limit = settings.max_angle_degrees.to_radians().cos();
    let mut merges = 0;
    let mut rejected: HashSet<(usize, usize)> = HashSet::new();
    loop {
        let ends: Vec<(usize, usize, Point, Point)> = lines
            .iter()
            .enumerate()
            .filter(|(_, (_, l))| l.len() >= 2)
            .flat_map(|(i, (_, l))| {
                (0..2).map(move |e| {
                    let (tip, tangent) = end_of(l, e);
                    (i, e, tip, tangent)
                })
            })
            .collect();
        let mut best: Option<(f64, usize, usize, usize, usize)> = None;
        for a in 0..ends.len() {
            let (i, ei, pi, ti) = ends[a];
            for &(j, ej, pj, tj) in &ends[a + 1..] {
                if i == j || rejected.contains(&(lines[i].0, lines[j].0)) {
                    continue;
                }
                let gap = sub(pj, pi);
                let distance = norm(gap);
                if distance > settings.max_gap {
                    continue;
                }
                let facing = -dot(ti, tj);
                if facing < cos_limit {
                    continue;
                }
                if distance > 0.5 * radii[i].min(radii[j]) {
                    let unit = scale(gap, 1.0 / distance);
                    if dot(unit, ti) < cos_limit || -dot(unit, tj) < cos_limit {
                        continue;
                    }
                    let bridge = linspace_points(pi, gap, (distance as usize).max(2));
                    if mean_sample(image, shape, &bridge) < settings.min_bridge_support {
                        continue;
                    }
                }
                let score = distance * (2.0 - facing);
                if best.is_none_or(|b| score < b.0) {
                    best = Some((score, i, ei, j, ej));
                }
            }
        }
        let Some((_, i, ei, j, ej)) = best else { break };
        let first = if ei == 1 {
            lines[i].1.clone()
        } else {
            reversed(&lines[i].1)
        };
        let second = if ej == 0 {
            lines[j].1.clone()
        } else {
            reversed(&lines[j].1)
        };
        let mut joined = first;
        joined.extend(second);
        if let Some(bend) = settings.min_bend_radius {
            if kink_index(&joined, bend, settings.kink_threshold, 35.0, 3).is_some() {
                rejected.insert((lines[i].0, lines[j].0));
                continue;
            }
        }
        let (length_i, length_j) = (polyline_length(&lines[i].1), polyline_length(&lines[j].1));
        let radius = (radii[i] * length_i + radii[j] * length_j) / (length_i + length_j).max(1e-9);
        let keep: Vec<usize> = (0..lines.len()).filter(|&k| k != i && k != j).collect();
        radii = keep.iter().map(|&k| radii[k]).chain([radius]).collect();
        let mut rebuilt: Vec<(usize, Vec<Point>)> =
            keep.iter().map(|&k| lines[k].clone()).collect();
        rebuilt.push((next_id, joined));
        next_id += 1;
        lines = rebuilt;
        merges += 1;
    }
    (lines.into_iter().map(|(_, l)| l).collect(), radii, merges)
}

/// `first` (ending at the junction) joined to `second` (starting there),
/// both cut back at the plane through the midpoint of their tips.
fn join(first: &[Point], second: &[Point]) -> Vec<Point> {
    let (_, out) = end_of(first, 1);
    let (_, back) = end_of(second, 0);
    let axis = sub(out, back);
    let axis = scale(axis, 1.0 / norm(axis).max(1e-12));
    let middle = scale(add(first[first.len() - 1], second[0]), 0.5);
    let mut stop = first.len();
    while stop > 2 && dot(sub(first[stop - 1], middle), axis) > 0.0 {
        stop -= 1;
    }
    let mut start = 0;
    while start + 2 < second.len() && dot(sub(second[start], middle), axis) < 0.0 {
        start += 1;
    }
    first[..stop]
        .iter()
        .chain(&second[start..])
        .copied()
        .collect()
}

/// `line` ordered so that its end `end` comes first (`at_start`) or last.
fn oriented(line: &[Point], end: usize, at_start: bool) -> Vec<Point> {
    let forward = if at_start { end == 0 } else { end == 1 };
    if forward {
        line.to_vec()
    } else {
        reversed(line)
    }
}

/// Aligned end pairs joined where the scan and the length prior favor it
/// (`_moves._merge_with_prior`): candidates within `max_gap` scored by the
/// drop in squared residual of the neighborhood (over `scale`) plus two
/// `end_cost`s, taken best first, each end once, no loops.
#[allow(clippy::too_many_arguments)]
pub fn merge_with_prior(
    image: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    max_gap: f64,
    max_angle_degrees: f64,
    min_bend_radius: Option<f64>,
    kink_threshold: f64,
    end_cost: f64,
    scale_: f64,
) -> (Vec<Vec<Point>>, Vec<f64>, usize) {
    let ends: Vec<(usize, usize, Point, Point)> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.len() >= 3)
        .flat_map(|(i, l)| {
            (0..2).map(move |e| {
                let (tip, tangent) = end_of(l, e);
                (i, e, tip, tangent)
            })
        })
        .collect();
    if ends.len() < 2 {
        return (lines.to_vec(), radii.to_vec(), 0);
    }
    let cos_limit = max_angle_degrees.to_radians().cos();
    let mut geometric: Vec<(f64, usize, usize)> = Vec::new();
    for a in 0..ends.len() {
        for b in a + 1..ends.len() {
            let (i, _, pi, ti) = ends[a];
            let (j, _, pj, tj) = ends[b];
            let gap = sub(pj, pi);
            let distance = norm(gap);
            if distance > max_gap || i == j || -dot(ti, tj) < cos_limit {
                continue;
            }
            let r = 0.5 * (radii[i] + radii[j]);
            let along = dot(gap, ti);
            if along < -2.0 * r || -dot(gap, tj) < -2.0 * r {
                continue;
            }
            if distance > r {
                let lateral = norm(sub(gap, scale(ti, along)));
                if along <= 0.0 && lateral > 1.5 * r {
                    continue;
                }
                if along > 0.0
                    && (along / distance < cos_limit || -dot(gap, tj) / distance < cos_limit)
                {
                    continue;
                }
                // Quick bound from the straight bridge: filling a gap of length g
                // with mean intensity m changes the residual by about
                // π r² g (1 - 2m); skip joins that can't come close to paying.
                let bridge = linspace_points(pi, gap, (distance as usize).max(2));
                let mean = mean_sample(image, shape, &bridge);
                if std::f64::consts::PI * r * r * along.max(0.0) * (2.0 * mean - 1.0) / scale_
                    + 4.0 * end_cost
                    < 0.0
                {
                    continue;
                }
            }
            geometric.push((distance, a, b));
        }
    }
    geometric.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut per_end: HashMap<usize, usize> = HashMap::new();
    let mut shortlist = Vec::new();
    for &(_, a, b) in &geometric {
        if *per_end.get(&a).unwrap_or(&0) >= 4 || *per_end.get(&b).unwrap_or(&0) >= 4 {
            continue;
        }
        *per_end.entry(a).or_default() += 1;
        *per_end.entry(b).or_default() += 1;
        shortlist.push((a, b));
    }

    let mut candidates: Vec<(f64, usize, usize, usize, usize)> = Vec::new();
    for (a, b) in shortlist {
        let (i, ei, _, _) = ends[a];
        let (j, ej, _, _) = ends[b];
        let r = 0.5 * (radii[i] + radii[j]);
        let head = oriented(&lines[i], ei, false);
        let joined = join(&head, &oriented(&lines[j], ej, true));
        if let Some(bend) = min_bend_radius {
            let junction = head.len();
            let excess = kink_excess(&joined, bend, kink_threshold, 35.0, 3);
            let low = junction.saturating_sub(6).min(excess.len());
            let high = (junction + 3).min(excess.len()).max(low);
            if excess[low..high].iter().any(|&v| v > 1.0) {
                continue;
            }
        }
        let ends_of = |line: &[Point], e: usize| -> Vec<Point> {
            if e == 1 {
                line[line.len() - 2..].to_vec()
            } else {
                line[..2].to_vec()
            }
        };
        let mut region = ends_of(&lines[i], ei);
        region.extend(ends_of(&lines[j], ej));
        let (low, high) = local_box(&region, 2.0 * r, shape);
        let others = near_box(lines, radii, low, high, &[i, j]);
        let base = context(&others, lines, radii, low, high);
        let (length_i, length_j) = (polyline_length(&lines[i]), polyline_length(&lines[j]));
        let radius = (radii[i] * length_i + radii[j] * length_j) / (length_i + length_j).max(1e-9);
        let apart = local_residual(
            image,
            shape,
            low,
            high,
            &[&lines[i], &lines[j]],
            &[radii[i], radii[j]],
            base.as_deref(),
            EDGE,
        );
        let together = local_residual(
            image,
            shape,
            low,
            high,
            &[&joined],
            &[radius],
            base.as_deref(),
            EDGE,
        );
        let odds = (apart - together) / scale_ + 2.0 * end_cost;
        if odds > 0.0 {
            candidates.push((odds, i, ei, j, ej));
        }
    }

    // Best first (ties as Python's reverse tuple sort); each end once; no loops.
    candidates.sort_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then(y.1.cmp(&x.1))
            .then(y.2.cmp(&x.2))
            .then(y.3.cmp(&x.3))
            .then(y.4.cmp(&x.4))
    });
    let mut parent: Vec<usize> = (0..lines.len()).collect();
    fn root(parent: &mut [usize], mut k: usize) -> usize {
        while parent[k] != k {
            parent[k] = parent[parent[k]];
            k = parent[k];
        }
        k
    }
    let mut link: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for &(_, i, ei, j, ej) in &candidates {
        if link.contains_key(&(i, ei))
            || link.contains_key(&(j, ej))
            || root(&mut parent, i) == root(&mut parent, j)
        {
            continue;
        }
        link.insert((i, ei), (j, ej));
        link.insert((j, ej), (i, ei));
        let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
        parent[ri] = rj;
    }
    if link.is_empty() {
        return (lines.to_vec(), radii.to_vec(), 0);
    }

    // Walk each chain from a fiber with a free end.
    let mut used = vec![false; lines.len()];
    let mut merged_lines = Vec::new();
    let mut merged_radii = Vec::new();
    for start in 0..lines.len() {
        if used[start] {
            continue;
        }
        let Some(entry) = (0..2).find(|&e| !link.contains_key(&(start, e))) else {
            continue; // interior of a chain; reached from its free end
        };
        let (mut i, mut entry) = (start, entry);
        let mut path = oriented(&lines[i], entry, true);
        let mut total = polyline_length(&lines[i]);
        let mut weight = radii[i] * total;
        used[i] = true;
        while let Some(&(j, ej)) = link.get(&(i, 1 - entry)) {
            path = join(&path, &oriented(&lines[j], ej, true));
            let length = polyline_length(&lines[j]);
            weight += radii[j] * length;
            total += length;
            used[j] = true;
            i = j;
            entry = ej;
        }
        merged_lines.push(path);
        merged_radii.push(weight / total.max(1e-9));
    }
    (merged_lines, merged_radii, link.len() / 2)
}

/// The rendering of `others` over the box, or `None` when there are none.
fn context(
    others: &[usize],
    lines: &[Vec<Point>],
    radii: &[f64],
    low: Corner,
    high: Corner,
) -> Option<Vec<f64>> {
    if others.is_empty() {
        return None;
    }
    let refs: Vec<&[Point]> = others.iter().map(|&k| lines[k].as_slice()).collect();
    let r: Vec<f64> = others.iter().map(|&k| radii[k]).collect();
    Some(render_occupancy(low, high, &refs, &r, EDGE))
}

/// Turn over `k` node spacings on each side of nodes `k..n-k`, as a
/// multiple of the allowed turn (`_moves._kink_excess`).
pub fn kink_excess(
    line: &[Point],
    min_bend_radius: f64,
    threshold: f64,
    min_angle_degrees: f64,
    k: usize,
) -> Vec<f64> {
    if line.len() < 2 * k + 1 {
        return Vec::new();
    }
    let floor = min_angle_degrees.to_radians();
    (k..line.len() - k)
        .map(|m| {
            let before = sub(line[m], line[m - k]);
            let after = sub(line[m + k], line[m]);
            let (nb, na) = (norm(before), norm(after));
            let cosine = (dot(before, after) / (nb * na).max(1e-12)).clamp(-1.0, 1.0);
            let window = 0.5 * (nb + na);
            let limit = (threshold * window / min_bend_radius).max(floor);
            cosine.acos() / limit
        })
        .collect()
}

/// The node of the worst kink beyond the limit, if any (`_moves._kink_index`).
pub fn kink_index(
    line: &[Point],
    min_bend_radius: f64,
    threshold: f64,
    min_angle_degrees: f64,
    k: usize,
) -> Option<usize> {
    let excess = kink_excess(line, min_bend_radius, threshold, min_angle_degrees, k);
    let (worst, &value) =
        excess
            .iter()
            .enumerate()
            .fold(None, |best: Option<(usize, &f64)>, (i, v)| match best {
                Some((_, b)) if *b >= *v => best,
                _ => Some((i, v)),
            })?;
    (value > 1.0).then_some(worst + k)
}

/// The nodes around `cut` relaxed until no kink is left there, or `None`.
fn smooth_kink(
    line: &[Point],
    cut: usize,
    min_bend_radius: f64,
    threshold: f64,
    min_angle_degrees: f64,
) -> Option<Vec<Point>> {
    let k = 3;
    let low = cut.saturating_sub(2 * k).max(1);
    let high = (cut + 2 * k).min(line.len().checked_sub(2)?);
    if high <= low {
        return None;
    }
    let mut smoothed = line.to_vec();
    for _ in 0..40 {
        let before = smoothed.clone();
        for m in low..=high {
            let target = scale(add(before[m - 1], before[m + 1]), 0.5);
            smoothed[m] = add(before[m], scale(sub(target, before[m]), 0.5));
        }
        let excess = kink_excess(&smoothed, min_bend_radius, threshold, min_angle_degrees, k);
        let from = low.saturating_sub(k).min(excess.len());
        let to = (high + 1).saturating_sub(k).min(excess.len()).max(from);
        let near = &excess[from..to];
        if near.is_empty() {
            return None;
        }
        if near.iter().all(|&v| v <= 1.0) {
            return Some(smoothed);
        }
    }
    None
}

/// Where to cut an overlong fiber: the weakest (3-node mean) image value
/// at least `min_length` from both ends, or the middle of that range.
fn weakest_index(line: &[Point], image: Option<(&[f32], Shape)>, min_length: f64) -> Option<usize> {
    let mut arc = vec![0.0];
    for pair in line.windows(2) {
        arc.push(arc[arc.len() - 1] + norm(sub(pair[1], pair[0])));
    }
    let total = arc[arc.len() - 1];
    let allowed: Vec<usize> = (0..line.len())
        .filter(|&m| arc[m] >= min_length && arc[m] <= total - min_length)
        .collect();
    if allowed.is_empty() {
        return None;
    }
    let Some((image, shape)) = image else {
        return Some(allowed[allowed.len() / 2]);
    };
    let values: Vec<f64> = line
        .iter()
        .map(|&p| trilinear(image, shape, p, 0.0))
        .collect();
    let smooth = |m: usize| -> f64 {
        let at = |i: isize| {
            if i >= 0 && (i as usize) < values.len() {
                values[i as usize]
            } else {
                0.0
            }
        };
        let m = m as isize;
        let third = 1.0 / 3.0;
        at(m - 1) * third + at(m) * third + at(m + 1) * third
    };
    let mut best = allowed[0];
    for &m in &allowed[1..] {
        if smooth(m) < smooth(best) {
            best = m;
        }
    }
    Some(best)
}

/// Settings of [`split_kinks`].
#[derive(Clone, Copy, Debug)]
pub struct SplitSettings {
    pub min_bend_radius: f64,
    pub min_length: f64,
    pub max_length: Option<f64>,
    pub threshold: f64,
    pub min_angle_degrees: f64,
    pub end_cost: f64,
    pub scale: f64,
}

/// Fibers split at kinks sharper than the bend limit, and overlong ones cut
/// at their weakest point (`_moves.split_kinks`). With an `end_cost` and
/// the image, a kink is first smoothed out, and only cut when the smoothed
/// fiber matches the scan worse than the kinked one by more than the two
/// new ends cost.
pub fn split_kinks(
    image: Option<(&[f32], Shape)>,
    lines: &[Vec<Point>],
    radii: &[f64],
    settings: SplitSettings,
) -> (Vec<Vec<Point>>, Vec<f64>, usize) {
    let s = settings;
    let mut queue: Vec<(Vec<Point>, f64, usize)> = lines
        .iter()
        .zip(radii)
        .enumerate()
        .map(|(source, (l, &r))| (l.clone(), r, source))
        .collect();
    let mut done: Vec<(Vec<Point>, f64)> = Vec::new();
    let mut splits = 0;
    let mut kept = 0;
    while let Some((line, radius, source)) = queue.pop() {
        let mut cut = kink_index(
            &line,
            s.min_bend_radius,
            s.threshold,
            s.min_angle_degrees,
            3,
        );
        if let (Some(at), Some((img, shape))) = (cut, image) {
            if s.end_cost > 0.0 && kept < 10 * lines.len() {
                if let Some(smoothed) = smooth_kink(
                    &line,
                    at,
                    s.min_bend_radius,
                    s.threshold,
                    s.min_angle_degrees,
                ) {
                    let from = at.saturating_sub(6);
                    let mut window: Vec<Point> = line[from..(at + 7).min(line.len())].to_vec();
                    window.extend_from_slice(&smoothed[from..(at + 7).min(smoothed.len())]);
                    let (low, high) = local_box(&window, 2.5 * radius, shape);
                    let others = near_box(lines, radii, low, high, &[source]);
                    let base = context(&others, lines, radii, low, high);
                    let kinked = local_residual(
                        img,
                        shape,
                        low,
                        high,
                        &[&line],
                        &[radius],
                        base.as_deref(),
                        EDGE,
                    );
                    let smooth = local_residual(
                        img,
                        shape,
                        low,
                        high,
                        &[&smoothed],
                        &[radius],
                        base.as_deref(),
                        EDGE,
                    );
                    if (kinked - smooth) / s.scale + 2.0 * s.end_cost > 0.0 {
                        kept += 1;
                        queue.push((smoothed, radius, source));
                        continue;
                    }
                }
            }
        }
        if cut.is_none() && s.max_length.is_some_and(|m| polyline_length(&line) > m) {
            cut = weakest_index(&line, image, s.min_length);
        }
        let Some(at) = cut else {
            done.push((line, radius));
            continue;
        };
        splits += 1;
        for part in [&line[..at + 1], &line[at..]] {
            if part.len() >= 2 && polyline_length(part) >= s.min_length {
                queue.push((part.to_vec(), radius, source));
            }
        }
    }
    let (lines, radii) = done.into_iter().unzip();
    (lines, radii, splits)
}

/// The runs of `keep` nodes as pieces of at least 2 nodes and `min_length`.
fn runs(line: &[Point], keep: &[bool], min_length: f64) -> Vec<Vec<Point>> {
    let mut pieces = Vec::new();
    let mut start = None;
    for k in 0..=line.len() {
        let flag = k < line.len() && keep[k];
        match (flag, start) {
            (true, None) => start = Some(k),
            (false, Some(s)) => {
                let piece = &line[s..k];
                if piece.len() >= 2 && polyline_length(piece) >= min_length {
                    pieces.push(piece.to_vec());
                }
                start = None;
            }
            _ => {}
        }
    }
    pieces
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Settings of [`resolve_side_by_side`].
#[derive(Clone, Copy, Debug)]
pub struct SideBySide {
    pub min_length: f64,
    pub reach: f64,
    pub end_cost: f64,
    pub scale: f64,
    pub max_angle_degrees: f64,
}

/// Decides, from the image, whether two adjacent parallel fits are one
/// fiber (`_moves.resolve_side_by_side`): for every pair running side by
/// side, the image near them is compared with both fibers as they are and
/// with one fiber along their midline; the smaller squared residual (with
/// `end_cost` nats per end added or removed, in units of `scale`) wins.
/// Returns the lines, radii and number of pairs merged.
pub fn resolve_side_by_side(
    image: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    s: SideBySide,
) -> (Vec<Vec<Point>>, Vec<f64>, usize) {
    let mut lines = lines.to_vec();
    let mut radii = radii.to_vec();
    let cos_limit = s.max_angle_degrees.to_radians().cos();
    let mut changed = 0;

    struct Index {
        grid: PointGrid,
        owner: Vec<usize>,
        starts: HashMap<usize, usize>,
        boxes: Vec<Option<(Point, Point)>>,
    }
    let largest = radii.iter().cloned().fold(0.0, f64::max);
    let index = |lines: &[Vec<Point>], radii: &[f64]| -> Index {
        let mut points = Vec::new();
        let mut owner = Vec::new();
        let mut starts = HashMap::new();
        let mut boxes = Vec::new();
        for (k, line) in lines.iter().enumerate() {
            if line.len() < 2 {
                boxes.push(None);
                continue;
            }
            starts.insert(k, points.len());
            points.extend_from_slice(line);
            owner.extend(std::iter::repeat_n(k, line.len()));
            let mut low = [f64::INFINITY; 3];
            let mut high = [f64::NEG_INFINITY; 3];
            for p in line {
                for a in 0..3 {
                    low[a] = low[a].min(p[a]);
                    high[a] = high[a].max(p[a]);
                }
            }
            let pad = 2.0 * radii[k];
            boxes.push(Some((
                [low[0] - pad, low[1] - pad, low[2] - pad],
                [high[0] + pad, high[1] + pad, high[2] + pad],
            )));
        }
        Index {
            grid: PointGrid::new(points, s.reach * largest),
            owner,
            starts,
            boxes,
        }
    };

    let mut current = index(&lines, &radii);
    let supports: Vec<f64> = lines.iter().map(|l| mean_sample(image, shape, l)).collect();
    let mut order: std::collections::VecDeque<usize> = argsort_by(&supports).into();
    let mut done: HashSet<usize> = HashSet::new();
    while !current.grid.points.is_empty() {
        let Some(i) = order.pop_front() else { break };
        if done.contains(&i) || lines[i].len() < 2 {
            continue;
        }
        let bound = s.reach * radii[i];
        let mut spacings: Vec<f64> = lines[i].windows(2).map(|p| norm(sub(p[1], p[0]))).collect();
        let segment = median(&mut spacings);
        let own = (2.0 * s.reach * radii[i] / segment.max(1e-6)) as usize + 1;
        let k_near = (own + 8).min(current.grid.points.len());
        if k_near < 2 {
            break;
        }
        // Per node of fit i: the nearest node of another fit among its
        // k_near nearest within the bound.
        let mut partner: Vec<Option<(usize, usize)>> = Vec::with_capacity(lines[i].len());
        for &p in &lines[i] {
            let mut near = current.grid.within(p, bound);
            near.retain(|&(d, _)| d < bound);
            near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            near.truncate(k_near);
            partner.push(
                near.iter()
                    .find(|&&(_, q)| current.owner[q] != i)
                    .map(|&(_, q)| (current.owner[q], q)),
            );
        }
        if partner.iter().filter(|p| p.is_some()).count() < 3 {
            continue;
        }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &(j, _) in partner.iter().flatten() {
            *counts.entry(j).or_default() += 1;
        }
        let top = *counts.values().max().unwrap_or(&0);
        if top < 3 {
            continue;
        }
        let j = *counts
            .iter()
            .filter(|&(_, &c)| c == top)
            .map(|(j, _)| j)
            .min()
            .unwrap();
        let mine: Vec<usize> = (0..lines[i].len())
            .filter(|&m| matches!(partner[m], Some((o, _)) if o == j))
            .collect();
        let theirs: Vec<usize> = mine
            .iter()
            .map(|&m| partner[m].unwrap().1 - current.starts[&j])
            .collect();
        // Crossing fits come close at a few nodes but are not parallel.
        let (ti, tj) = (tangents(&lines[i]), tangents(&lines[j]));
        let mut cosine: Vec<f64> = mine
            .iter()
            .zip(&theirs)
            .map(|(&m, &t)| dot(ti[m], tj[t]).abs())
            .collect();
        if median(&mut cosine) < cos_limit {
            continue;
        }
        let mut region: Vec<Point> = mine.iter().map(|&m| lines[i][m]).collect();
        region.extend(theirs.iter().map(|&t| lines[j][t]));
        let (low, high) = local_box(&region, 2.5 * radii[i].max(radii[j]), shape);
        let observed = crop(image, shape, low, high);
        let neighbors: Vec<usize> = (0..lines.len())
            .filter(|&k| k != i && k != j)
            .filter(|&k| {
                current
                    .boxes
                    .get(k)
                    .copied()
                    .flatten()
                    .is_some_and(|(lo, hi)| {
                        (0..3).all(|a| lo[a] < high[a] as f64 && hi[a] > low[a] as f64)
                    })
            })
            .collect();
        // Occupancy is a max-union, so the neighbors are drawn once for both renderings.
        let base = context(&neighbors, &lines, &radii, low, high)
            .unwrap_or_else(|| vec![0.0; observed.len()]);
        let union = |mut drawn: Vec<f64>| -> Vec<f64> {
            for (d, &b) in drawn.iter_mut().zip(&base) {
                *d = d.max(b);
            }
            drawn
        };
        let both = union(render_occupancy(
            low,
            high,
            &[&lines[i], &lines[j]],
            &[radii[i], radii[j]],
            EDGE,
        ));
        let mut merged_j = lines[j].clone();
        for (&m, &t) in mine.iter().zip(&theirs) {
            merged_j[t] = scale(add(lines[j][t], lines[i][m]), 0.5);
        }
        let mut keep_i = vec![true; lines[i].len()];
        for &m in &mine {
            keep_i[m] = false;
        }
        let pieces = runs(&lines[i], &keep_i, s.min_length);
        let mut drawn: Vec<&[Point]> = vec![&merged_j];
        drawn.extend(pieces.iter().map(|p| p.as_slice()));
        let drawn_radii: Vec<f64> = std::iter::once(radii[j])
            .chain(std::iter::repeat_n(radii[i], pieces.len()))
            .collect();
        let one = union(render_occupancy(low, high, &drawn, &drawn_radii, EDGE));
        let added_ends = 2.0 * pieces.len() as f64 - 2.0;
        let gain =
            (squared_residual(&observed, &both) - squared_residual(&observed, &one)) / s.scale;
        if gain - added_ends * s.end_cost > 0.0 {
            let radius_i = radii[i];
            lines[j] = merged_j;
            let mut pieces = pieces.into_iter();
            lines[i] = pieces.next().unwrap_or_default();
            for piece in pieces {
                lines.push(piece);
                radii.push(radius_i);
            }
            done.insert(j);
            changed += 1;
            current = index(&lines, &radii);
        }
        done.insert(i);
    }
    let keep: Vec<usize> = (0..lines.len()).filter(|&k| lines[k].len() >= 2).collect();
    (
        keep.iter().map(|&k| lines[k].clone()).collect(),
        keep.iter().map(|&k| radii[k]).collect(),
        changed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(x0: f64, x1: f64, y: f64, step: f64) -> Vec<Point> {
        let n = ((x1 - x0) / step).round() as usize;
        (0..=n).map(|i| [x0 + i as f64 * step, y, 10.0]).collect()
    }

    #[test]
    fn a_retrace_inside_another_fiber_is_removed() {
        let long = straight(0.0, 40.0, 10.0, 2.0);
        let copy = straight(10.0, 20.0, 10.3, 2.0);
        let (lines, radii) = trim_duplicates(&[long.clone(), copy], &[3.0, 3.0], 4.0, 0.8);
        assert_eq!(lines, vec![long]);
        assert_eq!(radii, vec![3.0]);
    }

    #[test]
    fn a_thin_retrace_inside_a_thick_fiber_is_removed() {
        let thick = straight(0.0, 40.0, 10.0, 2.0);
        let thin = straight(10.0, 20.0, 14.0, 2.0);
        let (lines, radii) = trim_duplicates(&[thick.clone(), thin], &[8.0, 2.5], 4.0, 0.8);
        assert_eq!(lines, vec![thick.clone()]);
        assert_eq!(radii, vec![8.0]);
        // A thin fiber touching the thick one is kept.
        let beside = straight(10.0, 20.0, 20.5, 2.0);
        let (lines, _) = trim_duplicates(&[thick.clone(), beside], &[8.0, 2.5], 4.0, 0.8);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn a_kink_is_found_where_the_line_turns() {
        let mut line = straight(0.0, 20.0, 10.0, 2.0);
        line.extend((1..=10).map(|i| [20.0, 10.0 + 2.0 * i as f64, 10.0]));
        assert_eq!(kink_index(&line, 40.0, 2.0, 35.0, 3), Some(10));
        assert_eq!(
            kink_index(&straight(0.0, 40.0, 10.0, 2.0), 40.0, 2.0, 35.0, 3),
            None
        );
        let (pieces, _, splits) = split_kinks(
            None,
            &[line],
            &[2.0],
            SplitSettings {
                min_bend_radius: 40.0,
                min_length: 6.0,
                max_length: None,
                threshold: 2.0,
                min_angle_degrees: 35.0,
                end_cost: 0.0,
                scale: 1.0,
            },
        );
        assert_eq!((pieces.len(), splits), (2, 1));
    }

    #[test]
    fn two_aligned_fragments_are_joined() {
        let shape = [20, 20, 60];
        let mut image = vec![0.0f32; 20 * 20 * 60];
        for k in 8..12 {
            for j in 8..12 {
                for i in 5..55 {
                    image[(k * 20 + j) * 60 + i] = 1.0;
                }
            }
        }
        let a = straight(6.0, 24.0, 10.0, 2.0);
        let b = straight(28.0, 50.0, 10.0, 2.0);
        let settings = MergeSettings {
            max_gap: 6.0,
            max_angle_degrees: 35.0,
            min_bridge_support: 0.45,
            min_bend_radius: Some(40.0),
            kink_threshold: 2.0,
            end_cost: 0.0,
            scale: 1.0,
            max_prior_gap: None,
            max_prior_angle_degrees: 45.0,
        };
        let (lines, _, merges) = merge_fragments(&image, shape, &[a, b], &[2.0, 2.0], settings);
        assert_eq!((lines.len(), merges), (1, 1));
        assert_eq!(lines[0].first().unwrap()[0], 6.0);
        assert_eq!(lines[0].last().unwrap()[0], 50.0);
    }
}
