//! Entanglement: fiber writhe and the linking of contacting fibers.
//!
//! Contact counts cannot tell two fibers that merely cross from two that wrap
//! around each other, and curvature cannot tell a fiber that bends in a plane
//! from one that coils. Both are captured by the Gauss linking integral
//!
//! ```text
//! Lk(A, B) = (1 / 4π) ∮_A ∮_B (r_A − r_B) · (dr_A × dr_B) / |r_A − r_B|³
//! ```
//!
//! evaluated exactly for polylines with the segment-pair solid-angle formula
//! of Klenin and Langowski (2000). For open fibers it is not an integer; it
//! measures how much of the sphere of directions one fiber sweeps around the
//! other.
//!
//! * **Writhe** is the integral of a fiber with itself: zero for a planar
//!   fiber, growing as it coils. It is reported per fiber and per unit length,
//!   and its sign gives handedness.
//! * **Contact linking** is the integral between two contacting fibers,
//!   restricted to a window of arc length centered on their contact on each
//!   fiber. A single straight crossing approaches one half in magnitude as
//!   the window grows, at any angle, though shallow crossings need a longer
//!   window to get there; each full wrap of one fiber around the other adds
//!   about one.
//!
//! Both are measured on fibers resampled at a uniform spacing, so compare
//! structures only at the same spacing and window.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

use tangle_core::{FiberAssembly, FiberId, Vec3};

use crate::distribution::Distribution;
use crate::neighbors::{collect_fibers, dot, norm, sub, NeighborMetrics};
use crate::shape::resample;

/// Schema version of [`EntanglementMetrics`].
pub const ENTANGLEMENT_SCHEMA_VERSION: u32 = 1;

/// Settings for [`analyze_entanglement`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntanglementConfig {
    /// Arc-length spacing at which fibers are resampled. `None` uses the
    /// smallest fiber diameter.
    pub sample_spacing: Option<f64>,
    /// Arc length of the window on each fiber, centered on a contact, over
    /// which contact linking is integrated. `None` uses 20 sample spacings.
    pub window: Option<f64>,
    /// Number of evenly spaced quantiles stored for each distribution.
    pub quantile_count: usize,
}

impl Default for EntanglementConfig {
    fn default() -> Self {
        Self {
            sample_spacing: None,
            window: None,
            quantile_count: 101,
        }
    }
}

/// Writhe of one fiber.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FiberEntanglementMetrics {
    /// Stable source fiber identifier.
    pub fiber_id: FiberId,
    /// Placed centerline length.
    pub length: f64,
    /// Signed writhe of the resampled centerline.
    pub writhe: f64,
}

/// Entanglement statistics of one assembly state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntanglementMetrics {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Arc-length sample spacing actually used.
    pub sample_spacing: f64,
    /// Contact-linking window actually used.
    pub window: f64,
    /// Signed writhe, one value per fiber.
    pub writhe: Option<Distribution>,
    /// Absolute writhe divided by fiber length, one value per fiber.
    pub absolute_writhe_per_length: Option<Distribution>,
    /// Total absolute writhe divided by total fiber length.
    pub mean_absolute_writhe_per_length: Option<f64>,
    /// Signed linking of each contacting pair, counted once per pair at its
    /// first contact event.
    pub contact_linking: Option<Distribution>,
    /// Absolute linking of each contacting pair.
    pub absolute_contact_linking: Option<Distribution>,
    /// Per-fiber writhe in topology order.
    pub fibers: Vec<FiberEntanglementMetrics>,
}

/// Invalid entanglement settings.
#[derive(Clone, Debug, PartialEq)]
pub enum EntanglementError {
    /// A length setting was not positive and finite.
    InvalidLength {
        /// Setting name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// Fewer than two quantiles were requested.
    InvalidQuantileCount(usize),
}

impl fmt::Display for EntanglementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { name, value } => {
                write!(formatter, "{name} must be positive and finite, got {value}")
            }
            Self::InvalidQuantileCount(count) => {
                write!(formatter, "quantile_count must be at least 2, got {count}")
            }
        }
    }
}

impl std::error::Error for EntanglementError {}

/// Measures fiber writhe, and the linking of the contacting pairs listed in
/// `neighbors`, which must come from [`crate::analyze_neighbors`] on the same
/// assembly.
pub fn analyze_entanglement(
    assembly: &FiberAssembly,
    neighbors: &NeighborMetrics,
    config: &EntanglementConfig,
) -> Result<EntanglementMetrics, EntanglementError> {
    if config.quantile_count < 2 {
        return Err(EntanglementError::InvalidQuantileCount(
            config.quantile_count,
        ));
    }
    let fibers = collect_fibers(assembly);
    let minimum_diameter = fibers
        .iter()
        .map(|f| 2.0 * f.radius)
        .filter(|d| *d > 0.0)
        .fold(f64::INFINITY, f64::min);
    let spacing = positive(
        "sample_spacing",
        config
            .sample_spacing
            .unwrap_or(if minimum_diameter.is_finite() {
                minimum_diameter
            } else {
                1.0
            }),
    )?;
    let window = positive("window", config.window.unwrap_or(20.0 * spacing))?;
    let quantiles = config.quantile_count;

    let samples: Vec<Vec<Vec3>> = fibers.iter().map(|f| resample(f, spacing)).collect();
    let per_fiber: Vec<FiberEntanglementMetrics> = fibers
        .iter()
        .zip(&samples)
        .map(|(fiber, points)| FiberEntanglementMetrics {
            fiber_id: fiber.id,
            length: fiber.length,
            writhe: writhe(points),
        })
        .collect();

    let index: HashMap<FiberId, usize> =
        fibers.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
    let lattice = Lattice::new(assembly);
    let half = (0.5 * window / spacing).round().max(1.0) as usize;
    let mut linked = std::collections::HashSet::new();
    let mut linking = Vec::new();
    for event in &neighbors.events {
        let (Some(&a), Some(&b)) = (index.get(&event.fiber_id), index.get(&event.other_fiber_id))
        else {
            continue;
        };
        if a == b || !linked.insert((a.min(b), a.max(b))) {
            continue;
        }
        let first = &samples[a];
        if first.len() < 2 || samples[b].len() < 2 {
            continue;
        }
        let center_arc = event.start + 0.5 * event.length;
        let center = ((center_arc / spacing).round() as usize).min(first.len() - 1);
        let first_window =
            &first[center.saturating_sub(half)..(center + half + 1).min(first.len())];

        // The other fiber's nearest image, and its sample closest to the
        // contact.
        let target = first[center];
        let (closest, shift) = samples[b]
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let delta = lattice.minimum_image(sub(*q, target));
                let shift = sub(add(target, delta), *q);
                (i, shift, norm(delta))
            })
            .min_by(|x, y| x.2.total_cmp(&y.2))
            .map(|(i, shift, _)| (i, shift))
            .expect("fiber has samples");
        let other = &samples[b];
        let second_window: Vec<Vec3> = other
            [closest.saturating_sub(half)..(closest + half + 1).min(other.len())]
            .iter()
            .map(|q| add(*q, shift))
            .collect();
        linking.push(linking_number(first_window, &second_window));
    }

    let total_length: f64 = per_fiber.iter().map(|f| f.length).sum();
    let total_absolute_writhe: f64 = per_fiber.iter().map(|f| f.writhe.abs()).sum();
    Ok(EntanglementMetrics {
        schema_version: ENTANGLEMENT_SCHEMA_VERSION,
        sample_spacing: spacing,
        window,
        writhe: Distribution::from_values(per_fiber.iter().map(|f| f.writhe), quantiles),
        absolute_writhe_per_length: Distribution::from_values(
            per_fiber
                .iter()
                .filter(|f| f.length > 0.0)
                .map(|f| f.writhe.abs() / f.length),
            quantiles,
        ),
        mean_absolute_writhe_per_length: (total_length > 0.0)
            .then(|| total_absolute_writhe / total_length),
        absolute_contact_linking: Distribution::from_values(
            linking.iter().map(|l| l.abs()),
            quantiles,
        ),
        contact_linking: Distribution::from_values(linking, quantiles),
        fibers: per_fiber,
    })
}

fn positive(name: &'static str, value: f64) -> Result<f64, EntanglementError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(EntanglementError::InvalidLength { name, value })
    }
}

/// Minimum-image translations on periodic axes of an orthorhombic cell.
struct Lattice {
    lengths: Vec3,
    periodic: [bool; 3],
}

impl Lattice {
    fn new(assembly: &FiberAssembly) -> Self {
        let cell = &assembly.cell;
        Self {
            lengths: std::array::from_fn(|axis| cell.basis[axis][axis]),
            periodic: cell.periodic,
        }
    }

    fn minimum_image(&self, mut delta: Vec3) -> Vec3 {
        for axis in 0..3 {
            if self.periodic[axis] && self.lengths[axis] > 0.0 {
                let length = self.lengths[axis];
                delta[axis] -= length * (delta[axis] / length).round();
            }
        }
        delta
    }
}

// ---------------------------------------------------------------------------
// Gauss integrals

/// Gauss linking integral between two open polylines.
pub(crate) fn linking_number(first: &[Vec3], second: &[Vec3]) -> f64 {
    let mut total = 0.0;
    for a in first.windows(2) {
        for b in second.windows(2) {
            total += segment_solid_angle(a[0], a[1], b[0], b[1]);
        }
    }
    total / (4.0 * PI)
}

/// Writhe of an open polyline: the Gauss integral over every pair of
/// non-adjacent segments. Adjacent segments share a vertex and contribute
/// nothing.
pub(crate) fn writhe(points: &[Vec3]) -> f64 {
    let segments = points.len().saturating_sub(1);
    let mut total = 0.0;
    for i in 0..segments {
        for j in i + 2..segments {
            total += segment_solid_angle(points[i], points[i + 1], points[j], points[j + 1]);
        }
    }
    // Each unordered pair counts twice in the double integral.
    2.0 * total / (4.0 * PI)
}

/// Signed solid angle `Ω` of the Gauss map of segments `p1 → p2` and
/// `p3 → p4` (Klenin & Langowski 2000). Degenerate or coplanar pairs give 0.
fn segment_solid_angle(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> f64 {
    let r13 = sub(p3, p1);
    let r14 = sub(p4, p1);
    let r23 = sub(p3, p2);
    let r24 = sub(p4, p2);
    let normals = [
        cross(r13, r14),
        cross(r14, r24),
        cross(r24, r23),
        cross(r23, r13),
    ];
    let mut units = [[0.0; 3]; 4];
    for (unit, normal) in units.iter_mut().zip(normals) {
        let length = norm(normal);
        if length <= 1e-300 || !length.is_finite() {
            return 0.0;
        }
        *unit = [normal[0] / length, normal[1] / length, normal[2] / length];
    }
    let angle: f64 = (0..4)
        .map(|k| dot(units[k], units[(k + 1) % 4]).clamp(-1.0, 1.0).asin())
        .sum();
    let orientation = dot(cross(sub(p4, p3), sub(p2, p1)), r13);
    if orientation > 0.0 {
        angle
    } else if orientation < 0.0 {
        -angle
    } else {
        0.0
    }
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{analyze_neighbors, NeighborAnalysisConfig};
    use tangle_core::{PeriodicCell, Section};

    fn circle(center: Vec3, radius: f64, normal_axis: usize, segments: usize) -> Vec<Vec3> {
        let (u, v) = match normal_axis {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        (0..=segments)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / segments as f64;
                let mut p = center;
                p[u] += radius * t.cos();
                p[v] += radius * t.sin();
                p
            })
            .collect()
    }

    fn mirrored(points: &[Vec3]) -> Vec<Vec3> {
        points.iter().map(|p| [p[0], p[1], -p[2]]).collect()
    }

    #[test]
    fn a_hopf_link_has_linking_number_one_and_its_mirror_minus_one() {
        // Two unit circles, each through the other's center.
        let first = circle([0.0; 3], 1.0, 2, 400);
        let second = circle([1.0, 0.0, 0.0], 1.0, 1, 400);
        let linking = linking_number(&first, &second);
        assert!((linking.abs() - 1.0).abs() < 1e-6, "{linking}");
        let mirror = linking_number(&mirrored(&first), &mirrored(&second));
        assert!((mirror + linking).abs() < 1e-9);
        // Unlinked circles far apart.
        let far = circle([5.0, 0.0, 0.0], 1.0, 1, 400);
        assert!(linking_number(&first, &far).abs() < 1e-6);
    }

    #[test]
    fn planar_fibers_have_no_writhe_and_coils_have_handed_writhe() {
        let planar = circle([0.0; 3], 1.0, 2, 200);
        assert!(writhe(&planar[..150]).abs() < 1e-12);
        let helix: Vec<Vec3> = (0..=600)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / 100.0;
                [t.cos(), t.sin(), 0.2 * t]
            })
            .collect();
        let right = writhe(&helix);
        assert!(right.abs() > 0.5, "{right}");
        assert!((writhe(&mirrored(&helix)) + right).abs() < 1e-9);
    }

    #[test]
    fn wrapped_fibers_link_more_than_crossing_fibers() {
        let radius = 0.05;
        let cell = PeriodicCell::orthorhombic([4.0; 3], [false; 3]);
        let build = |second: Vec<Vec3>| {
            let mut assembly = FiberAssembly::new(cell);
            let material = assembly.materials.add("fiber");
            let section = assembly.sections.add(Section::Circular { radius });
            let first: Vec<Vec3> = (0..=200)
                .map(|i| [1.0 + 2.0 * i as f64 / 200.0, 2.0, 2.0])
                .collect();
            assembly
                .add_fiber(FiberId(1), material, section, &first, &first)
                .unwrap();
            assembly
                .add_fiber(FiberId(2), material, section, &second, &second)
                .unwrap();
            let mut neighbor_config = NeighborAnalysisConfig::new(0.02);
            neighbor_config.sample_spacing = Some(0.01);
            let neighbors = analyze_neighbors(&assembly, &neighbor_config).unwrap();
            let config = EntanglementConfig {
                sample_spacing: Some(0.02),
                window: Some(1.0),
                ..EntanglementConfig::default()
            };
            analyze_entanglement(&assembly, &neighbors, &config).unwrap()
        };
        // A straight fiber crossing on top.
        let crossing: Vec<Vec3> = (0..=100)
            .map(|i| [2.0, 1.5 + i as f64 / 100.0, 2.0 + 2.0 * radius])
            .collect();
        // A fiber coiling once around the first, in contact all the way.
        let wrapped: Vec<Vec3> = (0..=200)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / 200.0;
                let r = 2.0 * radius;
                [
                    1.8 + 0.4 * i as f64 / 200.0,
                    2.0 + r * t.cos(),
                    2.0 + r * t.sin(),
                ]
            })
            .collect();
        let crossing = build(crossing);
        let wrapped = build(wrapped);
        let crossing_linking = crossing.absolute_contact_linking.unwrap().quantile(1.0);
        let wrapped_linking = wrapped.absolute_contact_linking.unwrap().quantile(1.0);
        // A perpendicular crossing tends to 1/2 as the window grows; one full
        // wrap adds about one.
        assert!(
            crossing_linking > 0.3 && crossing_linking < 0.5,
            "{crossing_linking}"
        );
        assert!(wrapped_linking > 0.8, "{wrapped_linking}");
        // One pair, counted once even though each fiber reports the contact.
        assert_eq!(crossing.contact_linking.unwrap().count, 1);
        // Straight fibers have no writhe.
        assert!(crossing.mean_absolute_writhe_per_length.unwrap() < 1e-12);
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
        let neighbors = analyze_neighbors(&assembly, &NeighborAnalysisConfig::new(0.01)).unwrap();
        let check = |config: EntanglementConfig| {
            analyze_entanglement(&assembly, &neighbors, &config).unwrap_err()
        };
        assert_eq!(
            check(EntanglementConfig {
                quantile_count: 1,
                ..EntanglementConfig::default()
            }),
            EntanglementError::InvalidQuantileCount(1)
        );
        assert!(matches!(
            check(EntanglementConfig {
                window: Some(0.0),
                ..EntanglementConfig::default()
            }),
            EntanglementError::InvalidLength { name: "window", .. }
        ));
    }
}
