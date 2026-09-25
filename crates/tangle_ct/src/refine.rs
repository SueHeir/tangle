//! Host-side steps of the continuous fit, between solver batches: fiber
//! ends, void cuts, node spacing and support, as `tangle/ct/_refine.py`.

use crate::line::{add, dot, norm, scale, sub, tangents, Point};
use crate::raster::segment_distance;
use crate::sample::trilinear;
use crate::Shape;

/// Which fiber owns single voxels, as `rasterize(..., signed=True)` decides
/// it: the fiber whose capsule surface is nearest the voxel center among
/// the segments within that fiber's radius of it (0 for none), ties going
/// to the lower segment. For callers that read a few voxels only.
pub struct OwnerLookup {
    segments: Vec<(Point, Point, i32, f64)>,
}

impl OwnerLookup {
    pub fn new(lines: &[Vec<Point>], radii: &[f64]) -> Self {
        let mut segments = Vec::new();
        for (f, line) in lines.iter().enumerate() {
            for pair in line.windows(2) {
                segments.push((pair[0], pair[1], f as i32 + 1, radii[f]));
            }
        }
        Self { segments }
    }

    /// The owner of voxel `[k, j, i]` (one-based, 0 for none).
    pub fn owner(&self, voxel: [usize; 3]) -> i32 {
        let center = [
            voxel[2] as f64 + 0.5,
            voxel[1] as f64 + 0.5,
            voxel[0] as f64 + 0.5,
        ];
        let mut best = (f64::INFINITY, 0);
        for &(a, b, label, radius) in &self.segments {
            let distance = segment_distance(center, a, b);
            if distance <= radius && distance - radius < best.0 {
                best = (distance - radius, label);
            }
        }
        best.1
    }
}

fn inside(shape: Shape, point: Point) -> bool {
    (0..3).all(|a| point[a] >= 0.5 && point[a] <= shape[2 - a] as f64 - 0.5)
}

/// Each line with its ends grown or trimmed by up to `max_moves` steps of
/// length `step`.
///
/// A fiber's capsule reaches a radius past its last node, so the scan's
/// foreground ends about `reach` (the radius, plus any margin by which the
/// foreground over-reaches) beyond the true end of the centerline. An end
/// grows while the scan is still fiber half a step past that (and the voxel
/// there is not another fiber's, by `owners`), and is trimmed while it is
/// void half a step short of it, which leaves the tip within half a step of
/// the true end. A fiber that leaves the scan is not trimmed at the boundary.
pub fn end_step(
    image: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    reach: &[f64],
    step: f64,
    owners: &OwnerLookup,
    max_moves: usize,
) -> Vec<Vec<Point>> {
    let value = |p: Point| trilinear(image, shape, p, 0.0);
    lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let mut line = line.clone();
            let cap = reach[index];
            for front in [true, false] {
                for _ in 0..max_moves {
                    let n = line.len();
                    if n < 3 {
                        break;
                    }
                    let (tip, inner) = if front {
                        (line[0], line[1])
                    } else {
                        (line[n - 1], line[n - 2])
                    };
                    let direction = sub(tip, inner);
                    let direction = scale(direction, 1.0 / norm(direction).max(1e-12));
                    let ahead = add(tip, scale(direction, cap + 0.5 * step));
                    let short = add(tip, scale(direction, (cap - 0.5 * step).max(0.0)));
                    let ahead_inside = inside(shape, ahead);
                    let mut owner = 0;
                    if ahead_inside {
                        let mut voxel = [0usize; 3];
                        for axis in 0..3 {
                            let c = ahead[2 - axis].floor().clamp(0.0, (shape[axis] - 1) as f64);
                            voxel[axis] = c as usize;
                        }
                        owner = owners.owner(voxel);
                    }
                    if ahead_inside
                        && value(ahead) > 0.55
                        && (owner == 0 || owner == index as i32 + 1)
                    {
                        let extended = add(tip, scale(direction, step));
                        if front {
                            line.insert(0, extended);
                        } else {
                            line.push(extended);
                        }
                    } else if value(tip) < 0.45 || (inside(shape, short) && value(short) < 0.45) {
                        if front {
                            line.remove(0);
                        } else {
                            line.pop();
                        }
                    } else {
                        break;
                    }
                }
            }
            line
        })
        .collect()
}

/// The thresholds of [`cut_void`] (see `_refine.cut_void`).
#[derive(Clone, Copy, Debug)]
pub struct VoidRules {
    pub level: f64,
    pub min_gap_radii: f64,
    pub bridge_level: f64,
    pub bridge_offset_radii: f64,
    pub aligned_level: f64,
    pub aligned_angle_degrees: f64,
}

/// What [`cut_void`] did: nodes dropped, splits and stretches bridged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoidCounts {
    pub trimmed: usize,
    pub splits: usize,
    pub bridged: usize,
}

/// Every fit cut where its centerline sits in void (image below `level`).
///
/// Void nodes at an end are trimmed back to the first supported node. An
/// interior void stretch shorter than `min_gap_radii` radii is kept (a dip,
/// not a drift); a longer one is bridged when it is a bow, reaching at least
/// `bridge_offset_radii` radii off the straight line between its supported
/// neighbors, and that line reads at least `bridge_level` all the way;
/// otherwise it splits the fit. With `directions` (`directions(fit, [a, b])`:
/// the scan's local fiber axis at the two supported neighbors), a stretch
/// whose line reads at least `aligned_level` and lies within
/// `aligned_angle_degrees` of the axis at both is bridged too. Nodes outside
/// the scan are left alone.
///
/// Returns the pieces, the index of the fit each came from, and the counts.
pub fn cut_void(
    image: &[f32],
    shape: Shape,
    lines: &[Vec<Point>],
    radii: &[f64],
    rules: VoidRules,
    mut directions: Option<&mut dyn FnMut(usize, [Point; 2]) -> [Point; 2]>,
) -> (Vec<Vec<Point>>, Vec<usize>, VoidCounts) {
    let value = |p: Point| trilinear(image, shape, p, 0.0);
    let aligned_cosine = rules.aligned_angle_degrees.to_radians().cos();
    let mut pieces = Vec::new();
    let mut source = Vec::new();
    let mut counts = VoidCounts::default();
    for (index, line) in lines.iter().enumerate() {
        let n = line.len();
        if n < 2 {
            pieces.push(line.clone());
            source.push(index);
            continue;
        }
        let radius = radii[index];
        let mut keep: Vec<bool> = line
            .iter()
            .map(|&p| !(value(p) < rules.level && inside(shape, p)))
            .collect();
        let mut arc = vec![0.0];
        for pair in line.windows(2) {
            arc.push(arc[arc.len() - 1] + norm(sub(pair[1], pair[0])));
        }
        let spacing = (arc[n - 1] / (n - 1) as f64).max(1e-6);
        // (first void node, first node after, new nodes)
        let mut bridges: Vec<(usize, usize, Vec<Point>)> = Vec::new();
        let mut start: Option<usize> = None;
        for k in 0..=n {
            let weak = k < n && !keep[k];
            if weak && start.is_none() {
                start = Some(k);
            } else if !weak {
                let Some(first) = start.take() else { continue };
                if first == 0 || k == n {
                    continue;
                }
                if arc[k] - arc[first - 1] < rules.min_gap_radii * radius {
                    keep[first..k].iter_mut().for_each(|v| *v = true);
                    continue;
                }
                let (a, b) = (line[first - 1], line[k]);
                let chord = sub(b, a);
                let count = ((norm(chord) / spacing).ceil() as usize).max(2);
                let across: Vec<Point> = (1..count)
                    .map(|i| add(a, scale(chord, i as f64 * (1.0 / count as f64))))
                    .collect();
                let chord2 = dot(chord, chord).max(1e-12);
                let offset = line[first..k]
                    .iter()
                    .map(|&p| {
                        let away = sub(p, a);
                        let along = (dot(away, chord) / chord2).clamp(0.0, 1.0);
                        norm(sub(away, scale(chord, along)))
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                let lowest = across
                    .iter()
                    .map(|&p| value(p))
                    .fold(f64::INFINITY, f64::min);
                let bow =
                    offset >= rules.bridge_offset_radii * radius && lowest >= rules.bridge_level;
                let mut aligned = false;
                if !bow && lowest >= rules.aligned_level {
                    if let Some(directions) = directions.as_mut() {
                        let unit = scale(chord, 1.0 / norm(chord).max(1e-12));
                        let axes = directions(index, [a, b]);
                        let cosine = dot(axes[0], unit).abs().min(dot(axes[1], unit).abs());
                        aligned = cosine >= aligned_cosine;
                    }
                }
                if bow || aligned {
                    bridges.push((first, k, across));
                    keep[first..k].iter_mut().for_each(|v| *v = true);
                }
            }
        }
        counts.trimmed += keep.iter().filter(|&&v| !v).count();
        let (line, keep) = if bridges.is_empty() {
            (line.clone(), keep)
        } else {
            let mut new_line = Vec::new();
            let mut new_keep = Vec::new();
            let mut previous = 0;
            counts.bridged += bridges.len();
            for (first, after, across) in bridges {
                new_line.extend_from_slice(&line[previous..first]);
                new_keep.extend_from_slice(&keep[previous..first]);
                new_keep.extend(std::iter::repeat_n(true, across.len()));
                new_line.extend(across);
                previous = after;
            }
            new_line.extend_from_slice(&line[previous..]);
            new_keep.extend_from_slice(&keep[previous..]);
            (new_line, new_keep)
        };
        let mut kept = 0;
        let mut run_start = 0;
        for k in 1..=line.len() {
            if k == line.len() || keep[k] != keep[run_start] {
                if keep[run_start] && k - run_start >= 2 {
                    pieces.push(line[run_start..k].to_vec());
                    source.push(index);
                    kept += 1;
                }
                run_start = k;
            }
        }
        counts.splits += kept.max(1) - 1;
    }
    (pieces, source, counts)
}

/// Mean image value along each centerline (about 1 on a real fiber; 0 for
/// an empty line).
pub fn support(image: &[f32], shape: Shape, lines: &[Vec<Point>]) -> Vec<f64> {
    lines
        .iter()
        .map(|line| {
            if line.is_empty() {
                return 0.0;
            }
            line.iter()
                .map(|&p| trilinear(image, shape, p, 0.0))
                .sum::<f64>()
                / line.len() as f64
        })
        .collect()
}

/// Largest discrete curvature times the admissible bend radius, per line.
pub fn curvature_ratio(lines: &[Vec<Point>], min_bend_radius: f64) -> Vec<f64> {
    lines
        .iter()
        .map(|line| {
            if line.len() < 3 {
                return 0.0;
            }
            let t = tangents(line);
            let mut largest = f64::NEG_INFINITY;
            for i in 0..line.len() - 1 {
                let turn = dot(t[i + 1], t[i]).clamp(-1.0, 1.0).acos();
                let spacing = norm(sub(line[i + 1], line[i])).max(1e-9);
                largest = largest.max(turn / spacing);
            }
            largest * min_bend_radius
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rod_image() -> (Vec<f32>, Shape) {
        // A fiber along x over voxels 5-39 and another from 45 to the edge.
        let shape = [20, 20, 60];
        let mut image = vec![0.0f32; 20 * 20 * 60];
        for k in 8..12 {
            for j in 8..12 {
                for i in (5..40).chain(45..60) {
                    image[(k * 20 + j) * 60 + i] = 1.0;
                }
            }
        }
        (image, shape)
    }

    const RULES: VoidRules = VoidRules {
        level: 0.3,
        min_gap_radii: 2.0,
        bridge_level: 0.7,
        bridge_offset_radii: 1.0,
        aligned_level: 0.5,
        aligned_angle_degrees: 15.0,
    };

    #[test]
    fn a_line_over_a_gap_is_split_and_a_short_dip_kept() {
        let (image, shape) = rod_image();
        let across: Vec<Point> = (6..57).map(|x| [x as f64, 10.0, 10.0]).collect();
        let (pieces, source, counts) = cut_void(&image, shape, &[across], &[2.0], RULES, None);
        assert_eq!(pieces.len(), 2);
        assert_eq!(source, vec![0, 0]);
        assert_eq!(pieces[0].last().unwrap()[0], 40.0);
        assert_eq!(pieces[1][0][0], 45.0);
        assert_eq!(
            counts,
            VoidCounts {
                trimmed: 4,
                splits: 1,
                bridged: 0
            }
        );
        let dip: Vec<Point> = (30..50).map(|x| [x as f64, 10.0, 10.0]).collect();
        let (pieces, _, _) = cut_void(&image, shape, &[dip.clone()], &[4.0], RULES, None);
        assert_eq!(pieces, vec![dip]);
    }

    #[test]
    fn a_hop_is_bridged_only_along_the_scan_axis() {
        let (image, shape) = rod_image();
        let hop = vec![
            [10.0, 10.0, 10.0],
            [15.0, 10.0, 10.0],
            [20.0, 13.5, 10.0],
            [25.0, 10.0, 10.0],
            [30.0, 10.0, 10.0],
        ];
        let lines = [hop];
        let (_, source, counts) = cut_void(&image, shape, &lines, &[4.0], RULES, None);
        assert_eq!((source.len(), counts.bridged), (2, 0));
        let mut along_x = |_: usize, _: [Point; 2]| [[1.0, 0.0, 0.0]; 2];
        let (_, source, counts) =
            cut_void(&image, shape, &lines, &[4.0], RULES, Some(&mut along_x));
        assert_eq!((source.len(), counts.bridged), (1, 1));
    }

    #[test]
    fn ends_grow_to_the_fiber_end() {
        let (image, shape) = rod_image();
        let line: Vec<Point> = (15..25).map(|x| [x as f64 + 0.5, 10.0, 10.0]).collect();
        let lines = vec![line];
        let owners = OwnerLookup::new(&lines, &[2.0]);
        let out = end_step(&image, shape, &lines, &[2.0], 1.0, &owners, 3);
        assert_eq!(out[0].len(), 16);
        assert_eq!(out[0][0][0], 12.5);
        assert_eq!(out[0].last().unwrap()[0], 27.5);
    }

    #[test]
    fn the_nearer_surface_owns_a_voxel() {
        let lines = vec![
            vec![[1.5, 0.5, 0.5], [1.6, 0.5, 0.5]],
            vec![[7.5, 0.5, 0.5], [7.6, 0.5, 0.5]],
        ];
        let owners = OwnerLookup::new(&lines, &[1.0, 4.0]);
        assert_eq!(owners.owner([0, 0, 3]), 2);
        assert_eq!(owners.owner([0, 0, 1]), 1);
        assert_eq!(owners.owner([0, 0, 20]), 0);
    }
}
