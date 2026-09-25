//! Polyline helpers in voxel coordinates `(x, y, z)`, as `tangle/ct/_geometry.py`.

/// A point `(x, y, z)` in voxel units.
pub type Point = [f64; 3];

#[inline]
pub fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: Point, s: f64) -> Point {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Arc length of a polyline.
pub fn polyline_length(points: &[Point]) -> f64 {
    points.windows(2).map(|p| norm(sub(p[1], p[0]))).sum()
}

/// A polyline resampled to (nearly) uniform arc-length `spacing`, as
/// `_geometry.resample` (`np.linspace` targets, `np.interp` per axis).
pub fn resample(points: &[Point], spacing: f64) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut kept = vec![points[0]];
    for pair in points.windows(2) {
        if norm(sub(pair[1], pair[0])) > 1e-9 {
            kept.push(pair[1]);
        }
    }
    if kept.len() < 2 {
        return kept;
    }
    let mut arc = vec![0.0];
    for pair in kept.windows(2) {
        let last = *arc.last().unwrap();
        arc.push(last + norm(sub(pair[1], pair[0])));
    }
    let total = *arc.last().unwrap();
    let count = ((total / spacing).round_ties_even() as usize).max(1) + 1;
    let step = total / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    let mut j = 0;
    for t in 0..count {
        let target = if t + 1 == count {
            total
        } else {
            t as f64 * step
        };
        if target >= total {
            out.push(*kept.last().unwrap());
            continue;
        }
        while j + 2 < arc.len() && arc[j + 1] <= target {
            j += 1;
        }
        let mut point = [0.0; 3];
        for (axis, value) in point.iter_mut().enumerate() {
            let slope = (kept[j + 1][axis] - kept[j][axis]) / (arc[j + 1] - arc[j]);
            *value = slope * (target - arc[j]) + kept[j][axis];
        }
        out.push(point);
    }
    out
}


/// Unit tangents at the nodes: central differences, one-sided at the ends
/// (`np.gradient`), normalized.
pub fn tangents(points: &[Point]) -> Vec<Point> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let t = if n < 2 {
                [0.0; 3]
            } else if i == 0 {
                sub(points[1], points[0])
            } else if i == n - 1 {
                sub(points[n - 1], points[n - 2])
            } else {
                scale(sub(points[i + 1], points[i - 1]), 0.5)
            };
            scale(t, 1.0 / norm(t).max(1e-12))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_spaces_nodes_evenly() {
        let line = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let out = resample(&line, 2.5);
        assert_eq!(out.len(), 5);
        assert_eq!(out[1], [2.5, 0.0, 0.0]);
        assert_eq!(out[4], [10.0, 0.0, 0.0]);
        // round half to even, as numpy: 5 / 2 = 2.5 → 2 segments
        assert_eq!(resample(&[[0.0; 3], [5.0, 0.0, 0.0]], 2.0).len(), 3);
    }

    #[test]
    fn tangents_are_unit_central_differences() {
        let t = tangents(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]);
        assert_eq!(t[0], [1.0, 0.0, 0.0]);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((t[1][0] - h).abs() < 1e-12 && (t[1][1] - h).abs() < 1e-12);
        assert_eq!(t[2], [0.0, 1.0, 0.0]);
    }
}
