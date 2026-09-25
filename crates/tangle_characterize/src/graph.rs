//! Statistics of the fiber contact network.
//!
//! Contact events say where fibers touch; the contact graph says how the
//! touching fibers are connected. Each fiber is a node, and two fibers are
//! joined by an edge when they have at least one contact event. The graph is
//! what carries load in a bonded or entangled network, so its connectivity
//! (coordination number, clustering, percolation) is a direct target for a
//! generator to match.

use std::collections::{BTreeMap, HashMap};

use tangle_core::FiberId;

use crate::neighbors::NeighborMetrics;

/// Schema version of [`ContactGraphMetrics`].
pub const CONTACT_GRAPH_SCHEMA_VERSION: u32 = 1;

/// Contact-network statistics derived from a [`NeighborMetrics`] report.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContactGraphMetrics {
    /// Version of the serialized schema.
    pub schema_version: u32,
    /// Number of fibers (nodes).
    pub fibers: usize,
    /// Number of fiber pairs in contact (edges).
    pub edges: usize,
    /// Mean number of distinct fibers each fiber touches (coordination
    /// number).
    pub mean_degree: f64,
    /// Twice the edge count divided by the total fiber length: contacting
    /// partners per unit length, which does not depend on how fibers were
    /// truncated at a scan or subvolume boundary.
    pub degree_per_length: Option<f64>,
    /// Fibers with each degree; entry `n` counts fibers touching `n` others.
    pub degree_histogram: Vec<usize>,
    /// Fraction of fibers that touch no other fiber.
    pub isolated_fraction: f64,
    /// Mean number of separate contact events per contacting pair. Straight
    /// fibers cross at most once; values above one mean fibers wrap, twist
    /// or wander back to each other.
    pub mean_contacts_per_pair: Option<f64>,
    /// Pairs with each number of separate contact events; entry `n` counts
    /// pairs that touch `n` times.
    pub pair_contact_histogram: Vec<usize>,
    /// Fraction of contacting pairs that touch more than once.
    pub repeated_contact_fraction: Option<f64>,
    /// Mean local clustering coefficient over fibers with at least two
    /// partners: the fraction of a fiber's partners that also touch each
    /// other.
    pub average_clustering: Option<f64>,
    /// Global clustering: closed triples over connected triples.
    pub transitivity: Option<f64>,
    /// Number of connected components, isolated fibers included.
    pub components: usize,
    /// Fraction of fibers in the largest connected component.
    pub largest_component_fraction: f64,
    /// Fraction of fiber length in the largest connected component.
    pub largest_component_length_fraction: f64,
    /// Degree of every fiber in the neighbor report's fiber order.
    pub degrees: Vec<usize>,
}

/// Builds the contact graph of a neighbor report and measures it.
///
/// A pair's number of contact events is the larger of the counts seen from
/// its two fibers; they differ only when one fiber's contact run is split by
/// sampling and the other's is not.
pub fn analyze_contact_graph(metrics: &NeighborMetrics) -> ContactGraphMetrics {
    let index: HashMap<FiberId, usize> = metrics
        .fibers
        .iter()
        .enumerate()
        .map(|(i, fiber)| (fiber.fiber_id, i))
        .collect();
    let count = metrics.fibers.len();

    // Events per ordered (fiber, other) pair, then per unordered pair.
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for event in &metrics.events {
        if let (Some(&a), Some(&b)) = (index.get(&event.fiber_id), index.get(&event.other_fiber_id))
        {
            if a != b {
                *directed.entry((a, b)).or_default() += 1;
            }
        }
    }
    let mut pairs: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (&(a, b), &events) in &directed {
        let entry = pairs.entry((a.min(b), a.max(b))).or_default();
        *entry = (*entry).max(events);
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut pair_contact_histogram = Vec::new();
    for (&(a, b), &events) in &pairs {
        adjacency[a].push(b);
        adjacency[b].push(a);
        bump(&mut pair_contact_histogram, events);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    let degrees: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut degree_histogram = Vec::new();
    for &degree in &degrees {
        bump(&mut degree_histogram, degree);
    }

    // Triangles at each fiber, from sorted adjacency lists.
    let mut local_clustering = Vec::new();
    let (mut closed, mut triples) = (0usize, 0usize);
    for neighbors in &adjacency {
        let degree = neighbors.len();
        if degree < 2 {
            continue;
        }
        let mut triangles = 0usize;
        for (position, &u) in neighbors.iter().enumerate() {
            for &w in &neighbors[position + 1..] {
                if adjacency[u].binary_search(&w).is_ok() {
                    triangles += 1;
                }
            }
        }
        let possible = degree * (degree - 1) / 2;
        closed += triangles;
        triples += possible;
        local_clustering.push(triangles as f64 / possible as f64);
    }

    // Connected components by union-find.
    let mut parent: Vec<usize> = (0..count).collect();
    for &(a, b) in pairs.keys() {
        let (root_a, root_b) = (find(&mut parent, a), find(&mut parent, b));
        if root_a != root_b {
            parent[root_a] = root_b;
        }
    }
    let mut component_fibers: HashMap<usize, (usize, f64)> = HashMap::new();
    for (fiber, fiber_metrics) in metrics.fibers.iter().enumerate() {
        let root = find(&mut parent, fiber);
        let entry = component_fibers.entry(root).or_default();
        entry.0 += 1;
        entry.1 += fiber_metrics.length;
    }
    let largest = component_fibers
        .values()
        .copied()
        .max_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)))
        .unwrap_or((0, 0.0));

    let edges = pairs.len();
    let total_events: usize = pairs.values().sum();
    let repeated = pairs.values().filter(|events| **events > 1).count();
    ContactGraphMetrics {
        schema_version: CONTACT_GRAPH_SCHEMA_VERSION,
        fibers: count,
        edges,
        mean_degree: if count > 0 {
            2.0 * edges as f64 / count as f64
        } else {
            0.0
        },
        degree_per_length: (metrics.total_length > 0.0)
            .then(|| 2.0 * edges as f64 / metrics.total_length),
        degree_histogram,
        isolated_fraction: if count > 0 {
            degrees.iter().filter(|d| **d == 0).count() as f64 / count as f64
        } else {
            0.0
        },
        mean_contacts_per_pair: (edges > 0).then(|| total_events as f64 / edges as f64),
        pair_contact_histogram,
        repeated_contact_fraction: (edges > 0).then(|| repeated as f64 / edges as f64),
        average_clustering: (!local_clustering.is_empty())
            .then(|| local_clustering.iter().sum::<f64>() / local_clustering.len() as f64),
        transitivity: (triples > 0).then(|| closed as f64 / triples as f64),
        components: component_fibers.len(),
        largest_component_fraction: if count > 0 {
            largest.0 as f64 / count as f64
        } else {
            0.0
        },
        largest_component_length_fraction: if metrics.total_length > 0.0 {
            largest.1 / metrics.total_length
        } else {
            0.0
        },
        degrees,
    }
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn bump(histogram: &mut Vec<usize>, index: usize) {
    if histogram.len() <= index {
        histogram.resize(index + 1, 0);
    }
    histogram[index] += 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbors::{ContactEvent, FiberNeighborMetrics};
    use crate::{analyze_neighbors, NeighborAnalysisConfig};
    use tangle_core::{FiberAssembly, PeriodicCell, Section, Vec3};

    fn report(lengths: &[f64], contacts: &[(u32, u32, usize)]) -> NeighborMetrics {
        let event = |fiber: u32, other: u32| ContactEvent {
            fiber_id: FiberId(fiber),
            other_fiber_id: FiberId(other),
            start: 0.0,
            length: 0.1,
            crossing_angle: 1.0,
            closest_axis_distance: 0.1,
            straight_crossing_length: 0.1,
            in_axis: false,
        };
        let mut events = Vec::new();
        for &(a, b, times) in contacts {
            for _ in 0..times {
                events.push(event(a, b));
                events.push(event(b, a));
            }
        }
        NeighborMetrics {
            schema_version: 1,
            contact_gap: 0.0,
            neighbor_gap: 0.0,
            in_axis_angle: 0.3,
            sample_spacing: 0.01,
            samples: 0,
            total_length: lengths.iter().sum(),
            contacts: events.len(),
            contacts_per_length: 0.0,
            in_axis_contact_fraction: 0.0,
            random_baseline_contacts_per_length: None,
            contact_ratio_to_random: None,
            contact_count_dispersion: None,
            median_crossing_angle: None,
            median_excess_persistence: None,
            median_in_axis_contact_length: None,
            mean_free_length: None,
            mean_neighbors: 0.0,
            mean_in_axis_neighbors: 0.0,
            neighbor_count_histogram: Vec::new(),
            in_axis_neighbor_count_histogram: Vec::new(),
            turnover_lags: Vec::new(),
            neighbor_turnover: Vec::new(),
            in_axis_neighbor_turnover: Vec::new(),
            neighbor_correlation_length: None,
            in_axis_correlation_length: None,
            free_lengths: Vec::new(),
            events,
            fibers: lengths
                .iter()
                .enumerate()
                .map(|(i, length)| FiberNeighborMetrics {
                    fiber_id: FiberId(i as u32 + 1),
                    length: *length,
                    contacts: 0,
                    in_axis_contacts: 0,
                    mean_neighbors: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn a_triangle_with_a_tail_and_an_isolated_fiber() {
        // 1-2-3 triangle, 3-4 tail touching twice, 5 isolated.
        let graph = analyze_contact_graph(&report(
            &[1.0, 1.0, 1.0, 1.0, 2.0],
            &[(1, 2, 1), (2, 3, 1), (1, 3, 1), (3, 4, 2)],
        ));
        assert_eq!(graph.fibers, 5);
        assert_eq!(graph.edges, 4);
        assert_eq!(graph.degrees, vec![2, 2, 3, 1, 0]);
        assert_eq!(graph.degree_histogram, vec![1, 1, 2, 1]);
        assert!((graph.mean_degree - 1.6).abs() < 1e-12);
        assert!((graph.degree_per_length.unwrap() - 8.0 / 6.0).abs() < 1e-12);
        assert!((graph.isolated_fraction - 0.2).abs() < 1e-12);
        assert_eq!(graph.pair_contact_histogram, vec![0, 3, 1]);
        assert!((graph.mean_contacts_per_pair.unwrap() - 1.25).abs() < 1e-12);
        assert!((graph.repeated_contact_fraction.unwrap() - 0.25).abs() < 1e-12);
        // Fibers 1 and 2 are fully clustered; fiber 3 closes 1 of 3 pairs.
        let expected = (1.0 + 1.0 + 1.0 / 3.0) / 3.0;
        assert!((graph.average_clustering.unwrap() - expected).abs() < 1e-12);
        // 3 closed triples out of 1 + 1 + 3.
        assert!((graph.transitivity.unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(graph.components, 2);
        assert!((graph.largest_component_fraction - 0.8).abs() < 1e-12);
        assert!((graph.largest_component_length_fraction - 4.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn empty_and_contact_free_reports_are_handled() {
        let graph = analyze_contact_graph(&report(&[], &[]));
        assert_eq!(graph.fibers, 0);
        assert_eq!(graph.components, 0);
        assert_eq!(graph.largest_component_fraction, 0.0);
        assert!(graph.degree_per_length.is_none());

        let graph = analyze_contact_graph(&report(&[1.0, 1.0], &[]));
        assert_eq!(graph.edges, 0);
        assert_eq!(graph.components, 2);
        assert_eq!(graph.isolated_fraction, 1.0);
        assert!(graph.mean_contacts_per_pair.is_none());
        assert!(graph.average_clustering.is_none());
        assert!(graph.transitivity.is_none());
    }

    #[test]
    fn crossing_fibers_measured_end_to_end_form_one_edge() {
        let radius = 0.05;
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius });
        let lines: [[Vec3; 2]; 3] = [
            [[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]],
            [
                [1.0, 0.5, 1.0 + 2.0 * radius],
                [1.0, 1.5, 1.0 + 2.0 * radius],
            ],
            [[0.2, 0.2, 0.2], [0.4, 0.2, 0.2]],
        ];
        for (i, line) in lines.iter().enumerate() {
            assembly
                .add_fiber(FiberId(i as u32 + 1), material, section, line, line)
                .unwrap();
        }
        let mut config = NeighborAnalysisConfig::new(0.01);
        config.sample_spacing = Some(0.002);
        let neighbors = analyze_neighbors(&assembly, &config).unwrap();
        let graph = analyze_contact_graph(&neighbors);
        assert_eq!(graph.edges, 1);
        assert_eq!(graph.degrees, vec![1, 1, 0]);
        assert_eq!(graph.pair_contact_histogram, vec![0, 1]);
        assert_eq!(graph.components, 2);
    }
}
