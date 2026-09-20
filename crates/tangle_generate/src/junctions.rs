use std::cmp::Ordering;

use tangle_core::{
    FiberAnchor, FiberAssembly, JunctionId, JunctionLawId, JunctionParameterId, MaterialId,
};
use tangle_relax::{ContactCapture, PackedAssembly, SegmentContactCandidate};

/// An unordered pair of material names eligible for junction formation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JunctionMaterialPair {
    /// One material name.
    pub first: String,
    /// The other material name.
    pub second: String,
}

impl JunctionMaterialPair {
    /// Creates an unordered material-pair filter.
    pub fn new(first: impl Into<String>, second: impl Into<String>) -> Self {
        Self {
            first: first.into(),
            second: second.into(),
        }
    }

    fn matches(&self, first: &str, second: &str) -> bool {
        (self.first == first && self.second == second)
            || (self.first == second && self.second == first)
    }
}

/// Deterministic rules for promoting current contacts to persistent junctions.
#[derive(Clone, Debug, PartialEq)]
pub struct JunctionCapturePolicy {
    /// Human-readable policy name recorded in formation reports.
    pub name: String,
    /// Symbolic junction-law name installed in the assembly law table.
    pub law_name: String,
    /// Plugin-owned parameter-set identifier stored on created junctions.
    pub parameters: JunctionParameterId,
    /// Largest allowed signed separation between fiber surfaces.
    pub maximum_surface_gap: f32,
    /// Smallest allowed acute angle between contacting segments, in radians.
    pub minimum_crossing_angle: f32,
    /// Largest allowed acute angle between contacting segments, in radians.
    pub maximum_crossing_angle: f32,
    /// Deterministic acceptance probability in `[0, 1]`.
    pub probability: f32,
    /// Seed used by deterministic candidate sampling.
    pub seed: u64,
    /// Optional eligible material-name pairs; empty accepts every pair.
    pub material_pairs: Vec<JunctionMaterialPair>,
    /// Maximum junction count accepted between the same two fibers.
    pub maximum_per_fiber_pair: usize,
    /// Minimum material-coordinate separation from an existing junction on
    /// both participating fibers.
    pub minimum_anchor_separation: f64,
    /// Maximum number of GPU candidates captured at one checkpoint.
    pub candidate_capacity: usize,
}

impl JunctionCapturePolicy {
    /// Creates a permissive touching-contact policy for one junction law.
    pub fn touching(name: impl Into<String>, law_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            law_name: law_name.into(),
            parameters: JunctionParameterId(0),
            maximum_surface_gap: 1.0e-4,
            minimum_crossing_angle: 0.0,
            maximum_crossing_angle: std::f32::consts::FRAC_PI_2,
            probability: 1.0,
            seed: 0,
            material_pairs: Vec::new(),
            maximum_per_fiber_pair: 1,
            minimum_anchor_separation: 1.0e-6,
            candidate_capacity: 65_536,
        }
    }
}

/// Counts reported by one explicit junction-capture checkpoint.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JunctionCaptureReport {
    /// Policy that selected this set of junctions.
    pub policy_name: String,
    /// GPU contact candidates examined by the host policy.
    pub candidates: usize,
    /// Persistent junctions appended to the assembly.
    pub created: usize,
    /// Candidates rejected by material, angle, probability, or spacing rules.
    pub rejected: usize,
}

pub(crate) fn validate_junction_policy(policy: &JunctionCapturePolicy) {
    assert!(!policy.name.is_empty());
    assert!(!policy.law_name.is_empty());
    assert!(policy.maximum_surface_gap.is_finite() && policy.maximum_surface_gap >= 0.0);
    assert!(policy.minimum_crossing_angle.is_finite());
    assert!(policy.maximum_crossing_angle.is_finite());
    assert!(policy.minimum_crossing_angle >= 0.0);
    assert!(policy.maximum_crossing_angle <= std::f32::consts::FRAC_PI_2);
    assert!(policy.minimum_crossing_angle <= policy.maximum_crossing_angle);
    assert!((0.0..=1.0).contains(&policy.probability));
    assert!(policy.maximum_per_fiber_pair > 0);
    assert!(policy.minimum_anchor_separation >= 0.0);
    assert!(policy.candidate_capacity > 0);
}

pub(crate) fn form_junctions_from_capture(
    assembly: &mut FiberAssembly,
    packed: &PackedAssembly,
    mut capture: ContactCapture,
    policy: &JunctionCapturePolicy,
) -> JunctionCaptureReport {
    validate_junction_policy(policy);
    assert!(
        !capture.overflow,
        "junction policy '{}' exceeded its candidate or broad-phase capacity",
        policy.name
    );
    capture.candidates.sort_by(compare_candidates);
    let candidates = capture.candidates.len();
    let law = find_or_add_law(assembly, &policy.law_name);
    let mut created = 0;

    for candidate in capture.candidates {
        if candidate.crossing_angle < policy.minimum_crossing_angle
            || candidate.crossing_angle > policy.maximum_crossing_angle
        {
            continue;
        }
        let first_fiber = packed.segment_fibers[candidate.first_segment as usize] as usize;
        let second_fiber = packed.segment_fibers[candidate.second_segment as usize] as usize;
        if !materials_allowed(assembly, first_fiber, second_fiber, policy) {
            continue;
        }
        let mut anchors = [
            anchor_for_segment(
                assembly,
                packed,
                candidate.first_segment as usize,
                candidate.first_coordinate,
            ),
            anchor_for_segment(
                assembly,
                packed,
                candidate.second_segment as usize,
                candidate.second_coordinate,
            ),
        ];
        if anchors[1].fiber < anchors[0].fiber {
            anchors.swap(0, 1);
        }
        if deterministic_unit(policy.seed, anchors) > policy.probability as f64 {
            continue;
        }
        if pair_junction_count(assembly, anchors[0].fiber, anchors[1].fiber)
            >= policy.maximum_per_fiber_pair
            || has_nearby_junction(assembly, anchors, policy.minimum_anchor_separation)
        {
            continue;
        }
        let id = JunctionId(
            assembly
                .junctions
                .junctions
                .iter()
                .map(|junction| junction.id.0)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        assembly
            .add_junction(id, law, policy.parameters, &anchors)
            .expect("captured junction anchors must reference generated fibers");
        created += 1;
    }

    JunctionCaptureReport {
        policy_name: policy.name.clone(),
        candidates,
        created,
        rejected: candidates - created,
    }
}

fn compare_candidates(
    first: &SegmentContactCandidate,
    second: &SegmentContactCandidate,
) -> Ordering {
    first
        .surface_gap
        .total_cmp(&second.surface_gap)
        .then(first.first_segment.cmp(&second.first_segment))
        .then(first.second_segment.cmp(&second.second_segment))
        .then(first.first_coordinate.total_cmp(&second.first_coordinate))
        .then(first.second_coordinate.total_cmp(&second.second_coordinate))
}

fn find_or_add_law(assembly: &mut FiberAssembly, name: &str) -> JunctionLawId {
    assembly
        .junction_laws
        .entries
        .iter()
        .position(|law| law.name == name)
        .map(|index| JunctionLawId(index as u32))
        .unwrap_or_else(|| assembly.junction_laws.add(name))
}

fn materials_allowed(
    assembly: &FiberAssembly,
    first_fiber: usize,
    second_fiber: usize,
    policy: &JunctionCapturePolicy,
) -> bool {
    if policy.material_pairs.is_empty() {
        return true;
    }
    let material_name = |material: MaterialId| {
        assembly.materials.entries[material.0 as usize]
            .name
            .as_str()
    };
    let first = material_name(assembly.topology.fibers[first_fiber].material);
    let second = material_name(assembly.topology.fibers[second_fiber].material);
    policy
        .material_pairs
        .iter()
        .any(|pair| pair.matches(first, second))
}

fn anchor_for_segment(
    assembly: &FiberAssembly,
    packed: &PackedAssembly,
    segment: usize,
    coordinate: f32,
) -> FiberAnchor {
    let fiber_index = packed.segment_fibers[segment] as usize;
    let fiber = &assembly.topology.fibers[fiber_index];
    let fiber_vertex_start = packed.fiber_vertex_spans[2 * fiber_index] as usize;
    let first_vertex = packed.segment_vertices[2 * segment] as usize;
    let second_vertex = packed.segment_vertices[2 * segment + 1] as usize;
    let mut rest_arc_length = 0.0_f64;
    for vertex in fiber_vertex_start..first_vertex {
        rest_arc_length += packed_intrinsic_distance(packed, vertex, vertex + 1);
    }
    rest_arc_length +=
        coordinate as f64 * packed_intrinsic_distance(packed, first_vertex, second_vertex);
    FiberAnchor {
        fiber: fiber.id,
        rest_arc_length,
        section_offset: None,
    }
}

fn packed_intrinsic_distance(packed: &PackedAssembly, first: usize, second: usize) -> f64 {
    let dx = packed.intrinsic_positions[3 * second] - packed.intrinsic_positions[3 * first];
    let dy = packed.intrinsic_positions[3 * second + 1] - packed.intrinsic_positions[3 * first + 1];
    let dz = packed.intrinsic_positions[3 * second + 2] - packed.intrinsic_positions[3 * first + 2];
    (dx * dx + dy * dy + dz * dz).sqrt() as f64
}

fn pair_junction_count(
    assembly: &FiberAssembly,
    first: tangle_core::FiberId,
    second: tangle_core::FiberId,
) -> usize {
    assembly
        .junctions
        .junctions
        .iter()
        .filter(|junction| {
            let start = junction.anchors.start as usize;
            let end = start + junction.anchors.len as usize;
            let anchors = &assembly.junctions.anchors[start..end];
            anchors.len() == 2
                && ((anchors[0].fiber == first && anchors[1].fiber == second)
                    || (anchors[0].fiber == second && anchors[1].fiber == first))
        })
        .count()
}

fn has_nearby_junction(
    assembly: &FiberAssembly,
    candidate: [FiberAnchor; 2],
    minimum_separation: f64,
) -> bool {
    let tolerance = minimum_separation.max(1.0e-9);
    assembly.junctions.junctions.iter().any(|junction| {
        let start = junction.anchors.start as usize;
        let end = start + junction.anchors.len as usize;
        let anchors = &assembly.junctions.anchors[start..end];
        if anchors.len() != 2 {
            return false;
        }
        let ordered = if anchors[0].fiber <= anchors[1].fiber {
            [anchors[0], anchors[1]]
        } else {
            [anchors[1], anchors[0]]
        };
        ordered[0].fiber == candidate[0].fiber
            && ordered[1].fiber == candidate[1].fiber
            && (ordered[0].rest_arc_length - candidate[0].rest_arc_length).abs() <= tolerance
            && (ordered[1].rest_arc_length - candidate[1].rest_arc_length).abs() <= tolerance
    })
}

fn deterministic_unit(seed: u64, anchors: [FiberAnchor; 2]) -> f64 {
    let mut value = seed
        ^ (anchors[0].fiber.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (anchors[1].fiber.0 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ anchors[0].rest_arc_length.to_bits().rotate_left(17)
        ^ anchors[1].rest_arc_length.to_bits().rotate_left(41);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell, Section};

    #[test]
    fn promotes_a_contact_once_with_material_anchors() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.1 });
        for (id, points) in [
            (1, [[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]]),
            (2, [[1.0, 0.5, 1.0], [1.0, 1.5, 1.0]]),
        ] {
            assembly
                .add_fiber(FiberId(id), material, section, &points, &points)
                .unwrap();
        }
        let packed = PackedAssembly::from_assembly(&assembly).unwrap();
        let contact = SegmentContactCandidate {
            first_segment: 0,
            second_segment: 1,
            first_coordinate: 0.5,
            second_coordinate: 0.5,
            surface_gap: 0.0,
            crossing_angle: std::f32::consts::FRAC_PI_2,
        };
        let policy = JunctionCapturePolicy::touching("crossing", "welded");

        let first = form_junctions_from_capture(
            &mut assembly,
            &packed,
            ContactCapture {
                candidates: vec![contact],
                overflow: false,
            },
            &policy,
        );
        assert_eq!(first.created, 1);
        assert_eq!(assembly.junctions.junctions.len(), 1);
        assert_eq!(assembly.junctions.anchors[0].rest_arc_length, 0.5);
        assert_eq!(assembly.junctions.anchors[1].rest_arc_length, 0.5);

        let second = form_junctions_from_capture(
            &mut assembly,
            &packed,
            ContactCapture {
                candidates: vec![contact],
                overflow: false,
            },
            &policy,
        );
        assert_eq!(second.created, 0);
        assert_eq!(assembly.junctions.junctions.len(), 1);
    }
}
