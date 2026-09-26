//! Drawing fitted fibers (capsules around centerline polylines) into voxels.

use crate::{strides, voxel_count, Shape};

/// Distance from `p` to segment `ab` (to `a` when the segment is a point).
#[inline]
pub fn segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let denominator = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if denominator <= 1e-12 {
        0.0
    } else {
        ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / denominator).clamp(0.0, 1.0)
    };
    let d = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// A fit's cross-section, for drawing it: round, or an oval whose long
/// semi-axis is `ratio` times the fit's radius (its short one), lying along
/// `axes` (one unit vector per node, across the centerline, consecutive
/// ones pointing the same way).
#[derive(Clone, Copy, Debug)]
pub enum Section<'a> {
    Round,
    Oval { ratio: f64, axes: &'a [[f64; 3]] },
}

impl Section<'_> {
    /// How much farther than a round section's the section reaches.
    pub fn stretch(&self) -> f64 {
        match self {
            Section::Round => 1.0,
            Section::Oval { ratio, .. } => ratio.max(1.0),
        }
    }

    /// The distance from `p` to segment `s` (nodes `a`, `b`), with the offset
    /// along an oval's long axis divided by its ratio: an oval of short
    /// semi-axis r then reads as a round fiber of radius r (the long axis at
    /// the nearest point is interpolated between the two nodes', less its
    /// part along the segment).
    #[inline]
    pub fn distance(&self, p: [f64; 3], a: [f64; 3], b: [f64; 3], s: usize) -> f64 {
        let Section::Oval { ratio, axes } = *self else {
            return segment_distance(p, a, b);
        };
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let length2 = dot(ab, ab);
        let t = if length2 <= 1e-12 { 0.0 } else { (dot(ap, ab) / length2).clamp(0.0, 1.0) };
        let d = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
        let (la, lb) = (axes[s], axes[(s + 1).min(axes.len() - 1)]);
        let mut long = [
            la[0] + t * (lb[0] - la[0]),
            la[1] + t * (lb[1] - la[1]),
            la[2] + t * (lb[2] - la[2]),
        ];
        if length2 > 1e-12 {
            let along = dot(long, ab) / length2;
            long = [long[0] - along * ab[0], long[1] - along * ab[1], long[2] - along * ab[2]];
        }
        let norm = dot(long, long).sqrt();
        let squared = dot(d, d);
        if norm < 1e-9 {
            return squared.sqrt();
        }
        let u = dot(d, long) / norm;
        (squared - u * u + (u / ratio) * (u / ratio)).max(0.0).sqrt()
    }
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Voxel index box `[low, high)` (`x, y, z`) around segment `ab` grown by `pad`, clipped to the volume.
fn segment_box(
    shape: Shape,
    a: [f64; 3],
    b: [f64; 3],
    pad: f64,
) -> Option<([usize; 3], [usize; 3])> {
    let mut low = [0usize; 3];
    let mut high = [0usize; 3];
    for axis in 0..3 {
        let upper = shape[2 - axis] as f64;
        let lo = (a[axis].min(b[axis]) - pad).floor().max(0.0);
        let hi = (a[axis].max(b[axis]) + pad).ceil().min(upper);
        if !(hi > lo) {
            return None;
        }
        low[axis] = lo as usize;
        high[axis] = hi as usize;
    }
    Some((low, high))
}

/// Nearest-fiber ownership of every voxel within `reach[f]` of fiber `f`'s
/// centerline, as `tangle.ct._geometry.rasterize`.
pub struct Raster {
    /// One-based fiber of each voxel, 0 where none reaches.
    pub labels: Vec<i32>,
    /// The owning distance (minus the owner's radius when signed); +∞ where none.
    pub distance: Vec<f32>,
    /// Global segment index of the owner (segment `s` of fiber `f` is
    /// `offsets[f] + s`, offsets from node counts minus one); −1 where none.
    pub segment: Vec<i32>,
}

/// `lines` are `(x, y, z)` polylines. A voxel is owned by the segment whose
/// distance to the voxel center (minus the fiber's radius when `signed`, so
/// the nearest capsule surface wins) is smallest among segments within their
/// fiber's `reach`; ties go to the lower segment.
pub fn rasterize(
    shape: Shape,
    lines: &[Vec<[f64; 3]>],
    radii: &[f64],
    reach: &[f64],
    signed: bool,
) -> Raster {
    rasterize_sections(shape, lines, radii, reach, signed, None)
}

/// `rasterize` with each line's cross-section (`None`: all round). An oval
/// fit's distances are `Section::distance`, so its radius and reach are in
/// units of its short semi-axis.
pub fn rasterize_sections(
    shape: Shape,
    lines: &[Vec<[f64; 3]>],
    radii: &[f64],
    reach: &[f64],
    signed: bool,
    sections: Option<&[Section]>,
) -> Raster {
    let n = voxel_count(shape);
    let s = strides(shape);
    let mut best = vec![f64::INFINITY; n];
    let mut owner = vec![-1i32; n];
    let mut labels = vec![0i32; n];
    let mut segment = 0i32;
    for (f, line) in lines.iter().enumerate() {
        let limit = reach[f];
        let key = if signed { radii[f] } else { 0.0 };
        let section = sections.map_or(Section::Round, |all| all[f]);
        for (s, pair) in line.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            if let Some((low, high)) = segment_box(shape, a, b, limit * section.stretch() + 0.5) {
                for k in low[2]..high[2] {
                    for j in low[1]..high[1] {
                        for i in low[0]..high[0] {
                            let p = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                            let distance = section.distance(p, a, b, s);
                            if distance > limit {
                                continue;
                            }
                            let value = distance - key;
                            let index = k * s[0] + j * s[1] + i;
                            if value < best[index] {
                                best[index] = value;
                                owner[index] = segment;
                                labels[index] = f as i32 + 1;
                            }
                        }
                    }
                }
            }
            segment += 1;
        }
    }
    Raster {
        labels,
        distance: best.iter().map(|&v| v as f32).collect(),
        segment: owner,
    }
}

/// Sets the voxels of `target` within `reach` of `line` to `value` (only
/// where `target` is 0 when `only_empty`), as `tangle.ct._geometry.paint`.
pub fn paint(
    target: &mut [i32],
    shape: Shape,
    line: &[[f64; 3]],
    reach: f64,
    value: i32,
    only_empty: bool,
) {
    paint_section(target, shape, line, reach, value, only_empty, Section::Round);
}

/// `paint` of a line with the given cross-section (`reach` in units of an
/// oval's short semi-axis).
pub fn paint_section(
    target: &mut [i32],
    shape: Shape,
    line: &[[f64; 3]],
    reach: f64,
    value: i32,
    only_empty: bool,
    section: Section,
) {
    let s = strides(shape);
    let doubled;
    let line = if line.len() == 1 {
        doubled = [line[0], line[0]];
        &doubled[..]
    } else {
        line
    };
    for (segment, pair) in line.windows(2).enumerate() {
        let (a, b) = (pair[0], pair[1]);
        if let Some((low, high)) = segment_box(shape, a, b, reach * section.stretch() + 0.5) {
            for k in low[2]..high[2] {
                for j in low[1]..high[1] {
                    for i in low[0]..high[0] {
                        let p = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                        let index = k * s[0] + j * s[1] + i;
                        if (only_empty && target[index] != 0) || section.distance(p, a, b, segment) > reach {
                            continue;
                        }
                        target[index] = value;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearer_surface_owns_the_voxel() {
        let shape = [1, 1, 10];
        let lines = vec![
            vec![[1.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
            vec![[7.5, 0.5, 0.5], [7.5, 0.5, 0.5]],
        ];
        let raster = rasterize(shape, &lines, &[1.0, 4.0], &[6.0, 6.0], true);
        // voxel 3 (center 3.5): 2 − 1 = 1 from fiber 1, 4 − 4 = 0 from fiber 2
        assert_eq!(raster.labels[3], 2);
        assert_eq!(raster.labels[1], 1);
        assert_eq!(raster.segment[3], 1);
        let plain = rasterize(shape, &lines, &[1.0, 4.0], &[6.0, 6.0], false);
        assert_eq!(plain.labels[3], 1);
    }

    #[test]
    fn an_oval_section_reaches_along_its_long_axis() {
        let shape = [21, 21, 21];
        // A line along x through the middle, long axis along y, 2:1.
        let line = vec![[2.5, 10.5, 10.5], [18.5, 10.5, 10.5]];
        let axes = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let oval = Section::Oval { ratio: 2.0, axes: &axes };
        let raster = rasterize_sections(shape, &[line.clone()], &[3.0], &[3.0], false, Some(&[oval]));
        let at = |x: usize, y: usize, z: usize| raster.labels[z * 21 * 21 + y * 21 + x];
        assert_eq!(at(10, 15, 10), 1); // 5 along the long axis: inside 6
        assert_eq!(at(10, 17, 10), 0); // 7 along it: outside
        assert_eq!(at(10, 10, 14), 0); // 4 along the short one: outside 3
        assert_eq!(at(10, 10, 12), 1);
        let mut target = vec![0i32; 21 * 21 * 21];
        paint_section(&mut target, shape, &line, 3.0, 1, false, oval);
        assert_eq!(target, raster.labels);
    }

    #[test]
    fn paint_fills_a_capsule() {
        let shape = [5, 5, 5];
        let mut target = vec![0i32; 125];
        paint(&mut target, shape, &[[2.5, 2.5, 2.5]], 1.0, 7, false);
        assert_eq!(target.iter().filter(|&&v| v == 7).count(), 7);
    }
}
