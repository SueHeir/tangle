use crate::math::{dot, norm, sub};
use crate::Vec3;

/// Mesh-aware curvature measured at one internal polyline vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexCurvature {
    /// Angle between the incoming and outgoing segment tangents in radians.
    pub turning_angle: f64,
    /// Discrete curvature based on turning angle and local dual length.
    pub curvature: f64,
    /// Reciprocal curvature, or infinity for a straight vertex.
    pub bend_radius: f64,
}

/// Measures discrete curvature at three consecutive centerline vertices.
///
/// The definition `2 sin(theta / 2) / dual_length` exactly recovers the
/// curvature of three samples from a circle with equal adjacent chord lengths
/// and remains stable under refinement. Returns `None` when either adjacent
/// segment is degenerate.
pub fn measure_vertex_curvature(
    previous: Vec3,
    vertex: Vec3,
    next: Vec3,
) -> Option<VertexCurvature> {
    let incoming = sub(vertex, previous);
    let outgoing = sub(next, vertex);
    let incoming_length = norm(incoming);
    let outgoing_length = norm(outgoing);
    if incoming_length <= f64::EPSILON || outgoing_length <= f64::EPSILON {
        return None;
    }
    let cosine = (dot(incoming, outgoing) / (incoming_length * outgoing_length)).clamp(-1.0, 1.0);
    let turning_angle = cosine.acos();
    let dual_length = 0.5 * (incoming_length + outgoing_length);
    let curvature = 2.0 * (0.5 * turning_angle).sin() / dual_length;
    let bend_radius = if curvature <= f64::EPSILON {
        f64::INFINITY
    } else {
        curvature.recip()
    };
    Some(VertexCurvature {
        turning_angle,
        curvature,
        bend_radius,
    })
}

/// Returns the largest valid internal-vertex curvature in a polyline.
pub fn maximum_polyline_curvature(points: &[Vec3]) -> f64 {
    points
        .windows(3)
        .filter_map(|triple| measure_vertex_curvature(triple[0], triple[1], triple[2]))
        .map(|measurement| measurement.curvature)
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_vertex_has_zero_curvature_and_infinite_radius() {
        let measurement =
            measure_vertex_curvature([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert_eq!(measurement.turning_angle, 0.0);
        assert_eq!(measurement.curvature, 0.0);
        assert!(measurement.bend_radius.is_infinite());
    }

    #[test]
    fn circular_arc_curvature_is_stable_under_refinement() {
        let coarse = sampled_circle(2.0, 9);
        let fine = sampled_circle(2.0, 33);
        let coarse_curvature = maximum_polyline_curvature(&coarse);
        let fine_curvature = maximum_polyline_curvature(&fine);
        assert!((coarse_curvature - 0.5).abs() < 1.0e-12);
        assert!((fine_curvature - 0.5).abs() < 1.0e-11);
        assert!((coarse_curvature - fine_curvature).abs() < 1.0e-11);
    }

    fn sampled_circle(radius: f64, points: usize) -> Vec<Vec3> {
        (0..points)
            .map(|index| {
                let angle =
                    index as f64 * std::f64::consts::FRAC_PI_2 / (points.saturating_sub(1)) as f64;
                [radius * angle.cos(), radius * angle.sin(), 0.0]
            })
            .collect()
    }
}
