//! Long-axis directions ("directors") for non-circular cross-sections.
//!
//! An elliptical fiber stores one unit director per vertex. The director lies
//! in the plane perpendicular to the local centerline tangent and points along
//! the section's first semi-axis (see [`crate::Section::Elliptical`]). Circular
//! fibers ignore their directors.

use crate::math::{cross, dot, norm, sub};
use crate::Vec3;

/// Unit tangent at each vertex of a polyline: the normalized chord through the
/// neighbouring vertices, or the end segment at the two ends.
pub fn polyline_tangents(points: &[Vec3]) -> Vec<Vec3> {
    let count = points.len();
    (0..count)
        .map(|index| {
            if count < 2 {
                return [1.0, 0.0, 0.0];
            }
            let before = points[index.saturating_sub(1)];
            let after = points[(index + 1).min(count - 1)];
            normalize_or(sub(after, before), [1.0, 0.0, 0.0])
        })
        .collect()
}

/// Default director for a tangent: the direction perpendicular to the tangent
/// that lies in the xy plane, so the section lies flat with its short axis as
/// close to z as possible. A tangent along z falls back to x.
pub fn default_director(tangent: Vec3) -> Vec3 {
    let flat = cross([0.0, 0.0, 1.0], tangent);
    if norm(flat) > 1.0e-9 * norm(tangent).max(f64::MIN_POSITIVE) {
        normalize_or(flat, [1.0, 0.0, 0.0])
    } else {
        project_director([1.0, 0.0, 0.0], tangent)
    }
}

/// Default directors for every vertex of a polyline, with signs chosen so
/// consecutive directors never flip (an ellipse is unchanged by the sign).
pub fn default_directors(points: &[Vec3]) -> Vec<Vec3> {
    let mut directors: Vec<Vec3> = polyline_tangents(points)
        .into_iter()
        .map(default_director)
        .collect();
    align_director_signs(&mut directors);
    directors
}

/// Projects a director onto the plane perpendicular to `tangent` and
/// normalizes it. A director parallel to the tangent is replaced by the
/// default director.
pub fn project_director(director: Vec3, tangent: Vec3) -> Vec3 {
    let tangent_length_squared = dot(tangent, tangent);
    if tangent_length_squared <= f64::MIN_POSITIVE {
        return normalize_or(director, [1.0, 0.0, 0.0]);
    }
    let along = dot(director, tangent) / tangent_length_squared;
    let projected = [
        director[0] - along * tangent[0],
        director[1] - along * tangent[1],
        director[2] - along * tangent[2],
    ];
    if norm(projected) <= 1.0e-9 * norm(director).max(f64::MIN_POSITIVE) {
        default_director(tangent)
    } else {
        normalize_or(projected, [1.0, 0.0, 0.0])
    }
}

/// Projects each director perpendicular to its vertex tangent and aligns
/// consecutive signs.
pub fn orthonormalize_directors(points: &[Vec3], directors: &mut [Vec3]) {
    for (director, tangent) in directors.iter_mut().zip(polyline_tangents(points)) {
        *director = project_director(*director, tangent);
    }
    align_director_signs(directors);
}

fn align_director_signs(directors: &mut [Vec3]) {
    for index in 1..directors.len() {
        if dot(directors[index - 1], directors[index]) < 0.0 {
            directors[index] = directors[index].map(|value| -value);
        }
    }
}

fn normalize_or(value: Vec3, fallback: Vec3) -> Vec3 {
    let length = norm(value);
    if length.is_finite() && length > f64::MIN_POSITIVE {
        value.map(|component| component / length)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        norm(sub(a, b)) < 1.0e-12
    }

    #[test]
    fn default_director_lies_flat_and_perpendicular() {
        let tangent = [1.0, 1.0, 0.5];
        let director = default_director(tangent);
        assert!(dot(director, tangent).abs() < 1.0e-12);
        assert!(director[2].abs() < 1.0e-12);
        assert!((norm(director) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn vertical_tangent_falls_back_to_x() {
        assert!(close(default_director([0.0, 0.0, 2.0]), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn default_directors_do_not_flip_along_a_bend() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let directors = default_directors(&points);
        for pair in directors.windows(2) {
            assert!(dot(pair[0], pair[1]) >= 0.0);
        }
    }

    #[test]
    fn projection_removes_the_tangent_component() {
        let director = project_director([1.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(close(director, [0.0, 1.0, 0.0]));
    }
}
