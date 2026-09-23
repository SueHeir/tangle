//! Shape cases shared by the flexible-fiber tutorial and its regression test.

use tangle_generate::CenterlineShape;

/// Minimum physically admissible bend radius used by the tutorial.
pub const MINIMUM_BEND_RADIUS: f64 = 0.12;

/// One intrinsic-versus-placed fiber-shape experiment.
#[derive(Clone, Copy, Debug)]
pub struct ShapeCase {
    /// Stable output name.
    pub slug: &'static str,
    /// Short explanation of the mechanics represented by this case.
    pub description: &'static str,
    /// Unloaded centerline shape.
    pub intrinsic_shape: CenterlineShape,
    /// Centerline shape at generation time.
    pub placed_shape: CenterlineShape,
    /// Initial end-to-end distance divided by fiber length.
    pub placed_chord_fraction: f64,
}

/// Returns the four shape-semantics cases demonstrated by the example.
pub fn shape_cases() -> [ShapeCase; 4] {
    [
        ShapeCase {
            slug: "straight_intrinsic_straight_placed",
            description: "straight now; wants to be straight",
            intrinsic_shape: CenterlineShape::Straight,
            placed_shape: CenterlineShape::Straight,
            placed_chord_fraction: 1.0,
        },
        ShapeCase {
            slug: "straight_intrinsic_curved_placed",
            description: "curved now; wants to become straight",
            intrinsic_shape: CenterlineShape::Straight,
            placed_shape: CenterlineShape::Curved { amplitude: 0.04 },
            placed_chord_fraction: 0.9,
        },
        ShapeCase {
            slug: "curved_intrinsic_curved_placed",
            description: "curved now; wants to retain its natural curve",
            intrinsic_shape: CenterlineShape::Curved { amplitude: 0.04 },
            placed_shape: CenterlineShape::Curved { amplitude: 0.04 },
            placed_chord_fraction: 1.0,
        },
        ShapeCase {
            slug: "straight_intrinsic_overbent_placed",
            description: "starts beyond its bend limit; relaxation must smooth it",
            intrinsic_shape: CenterlineShape::Straight,
            placed_shape: CenterlineShape::Curved { amplitude: 0.12 },
            placed_chord_fraction: 0.7,
        },
    ]
}
