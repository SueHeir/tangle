use super::*;
use tangle_core::{FiberId, PeriodicCell, Section};

#[test]
fn active_capsule_bounds_include_radius_and_padding() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    let points = [[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]];
    assembly
        .add_fiber(FiberId(1), material, section, &points, &points)
        .unwrap();
    let packed = tangle_relax::PackedAssembly::from_assembly(&assembly).unwrap();

    let (lower, upper) =
        active_capsule_bounds(&packed, &packed.positions, &packed.segment_active, 0.02);

    for (actual, expected) in lower.into_iter().zip([0.17, 0.27, 0.37]) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
    for (actual, expected) in upper.into_iter().zip([0.83, 0.73, 0.63]) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
}

#[test]
fn soft_bend_limit_can_be_deferred_only_after_hard_contact_passes() {
    let acceptance = RelaxationAcceptance {
        penetration: AcceptanceLimit::hard(0.3e-6),
        curvature_ratio: AcceptanceLimit::soft(1.05),
    };
    let mut relaxation = RelaxationState {
        max_penetration: 0.2e-6,
        max_curvature_ratio: 1.08,
        ..RelaxationState::default()
    };

    assert!(!acceptance_satisfied(acceptance, &relaxation));
    assert!(hard_acceptance_satisfied(acceptance, &relaxation));
    assert!(unmet_acceptance_limits(acceptance, &relaxation).contains("Soft"));

    relaxation.max_penetration = 0.4e-6;
    assert!(!hard_acceptance_satisfied(acceptance, &relaxation));
    assert!(unmet_acceptance_limits(acceptance, &relaxation).contains("Hard"));
}

#[test]
fn final_policy_requires_both_contact_and_bending() {
    let acceptance = RelaxationAcceptance {
        penetration: AcceptanceLimit::hard(0.3e-6),
        curvature_ratio: AcceptanceLimit::hard(1.001),
    };
    let mut relaxation = RelaxationState {
        max_penetration: 0.2e-6,
        max_curvature_ratio: 1.0009,
        ..RelaxationState::default()
    };
    assert!(acceptance_satisfied(acceptance, &relaxation));
    assert!(hard_acceptance_satisfied(acceptance, &relaxation));

    relaxation.max_curvature_ratio = 1.0011;
    assert!(!acceptance_satisfied(acceptance, &relaxation));
    assert!(!hard_acceptance_satisfied(acceptance, &relaxation));
}

fn formation_policy(on_exhaustion: SolveExhaustion) -> SolvePolicy {
    SolvePolicy {
        name: "formation".to_string(),
        solver_targets: RelaxationTargets {
            penetration: 0.3e-6,
            curvature_ratio: 1.000_01,
        },
        acceptance: RelaxationAcceptance {
            penetration: AcceptanceLimit::hard(0.3e-6),
            curvature_ratio: AcceptanceLimit::soft(1.05),
        },
        maximum_iterations: 1_000,
        extra_iterations: SolvePolicy::default_extra_iterations(1_000),
        on_exhaustion,
    }
}

#[test]
fn exhausted_solve_keeps_going_until_hard_limits_hold() {
    let policy = formation_policy(SolveExhaustion::ContinueIfHardLimitsSatisfied);
    assert_eq!(policy.extra_iterations, 500);
    let mut relaxation = RelaxationState {
        max_penetration: 0.38e-6,
        max_curvature_ratio: 1.07,
        ..RelaxationState::default()
    };

    // Within the budget the solver simply continues.
    assert_eq!(
        policy_gate(&policy, 400, true, &relaxation),
        PolicyGate::Solve { remaining: 600 }
    );
    // A hard penetration miss at the end of the budget no longer rejects:
    // the gate grants the extension instead.
    assert_eq!(
        policy_gate(&policy, 1_000, true, &relaxation),
        PolicyGate::Solve { remaining: 500 }
    );
    // The first batch whose hard limits hold ends the extension, deferring
    // the soft bend limit as before.
    relaxation.max_penetration = 0.25e-6;
    assert_eq!(
        policy_gate(&policy, 1_130, true, &relaxation),
        PolicyGate::ContinueWithWarning
    );
    // Only when the extension is spent too does the gate reject.
    relaxation.max_penetration = 0.38e-6;
    assert_eq!(
        policy_gate(&policy, 1_500, true, &relaxation),
        PolicyGate::Reject
    );
    // Meeting every limit is still accepted at any point.
    relaxation.max_penetration = 0.25e-6;
    relaxation.max_curvature_ratio = 1.04;
    assert_eq!(
        policy_gate(&policy, 1_200, true, &relaxation),
        PolicyGate::Accept
    );
    assert!(extension_note(&policy, 1_200).contains("200 iterations past"));
    assert!(extension_note(&policy, 900).is_empty());
}

#[test]
fn rejecting_policy_extends_until_every_limit_holds() {
    let policy = formation_policy(SolveExhaustion::Reject);
    // Hard limits alone do not pass a rejecting gate, so it keeps solving.
    let relaxation = RelaxationState {
        max_penetration: 0.25e-6,
        max_curvature_ratio: 1.07,
        ..RelaxationState::default()
    };
    assert_eq!(
        policy_gate(&policy, 1_000, true, &relaxation),
        PolicyGate::Solve { remaining: 500 }
    );
    assert_eq!(
        policy_gate(&policy, 1_500, true, &relaxation),
        PolicyGate::Reject
    );
    // Without an extension the old behavior is unchanged.
    let strict = SolvePolicy {
        extra_iterations: 0,
        ..policy
    };
    assert_eq!(
        policy_gate(&strict, 1_000, true, &relaxation),
        PolicyGate::Reject
    );
    // Nothing is decided before the geometry has been measured once.
    assert_eq!(
        policy_gate(&strict, 0, false, &relaxation),
        PolicyGate::Solve { remaining: 1_000 }
    );
}

#[test]
fn needling_selection_is_reproducible_internal_and_layer_scoped() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    for (id, layer) in [(1, 0), (2, 1), (3, 1)] {
        assembly
            .add_fiber(
                FiberId(id),
                material,
                section,
                &[
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.3, 0.0, 0.0],
                ],
                &[
                    [0.2, 0.2, 0.2],
                    [0.3, 0.2, 0.2],
                    [0.4, 0.2, 0.2],
                    [0.5, 0.2, 0.2],
                ],
            )
            .unwrap();
        assembly
            .set_fiber_formation_layer(FiberId(id), Some(layer))
            .unwrap();
        assembly.topology.fibers[(id - 1) as usize].formation_step = layer;
    }
    let config = NeedlingConfig {
        layer: 1,
        selection: NeedlingSelection::RandomFiberFraction {
            fraction: 1.0,
            seed: 42,
        },
        minimum_fiber_diameter: None,
        depth: 0.1,
        stiffness: 1.0,
        max_translation: 0.1,
        maximum_translation_over_fiber_diameter: 0.25,
    };

    let packed = tangle_relax::PackedAssembly::from_assembly(&assembly).unwrap();
    let first = selected_needling_vertices_from_mask(
        &assembly,
        &packed,
        &packed.positions,
        &packed.vertex_active,
        &packed.segment_active,
        Some(1),
        2,
        config,
    );
    let second = selected_needling_vertices_from_mask(
        &assembly,
        &packed,
        &packed.positions,
        &packed.vertex_active,
        &packed.segment_active,
        Some(1),
        2,
        config,
    );
    assert_eq!(first, second);
    assert_eq!(first.len(), 2);
    for vertex in first {
        let fiber = assembly
            .topology
            .fibers
            .iter()
            .find(|fiber| {
                vertex >= fiber.vertices.start && vertex < fiber.vertices.start + fiber.vertices.len
            })
            .unwrap();
        assert_eq!(fiber.formation_layer, Some(1));
        assert!(vertex > fiber.vertices.start);
        assert!(vertex + 1 < fiber.vertices.start + fiber.vertices.len);
    }
}

#[test]
fn circular_needling_selects_one_internal_vertex_per_fiber_in_footprint() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let thin = assembly.sections.add(Section::Circular { radius: 0.01 });
    let thick = assembly.sections.add(Section::Circular { radius: 0.02 });
    for (id, layer, y, section) in [
        (1, 1, 0.5, thin),
        (2, 1, 0.5, thick),
        (3, 0, 0.5, thick),
        (4, 1, 0.8, thick),
    ] {
        let placed = [[0.3, y, 0.5], [0.4, y, 0.5], [0.5, y, 0.5], [0.6, y, 0.5]];
        assembly
            .add_fiber(FiberId(id), material, section, &placed, &placed)
            .unwrap();
        assembly
            .set_fiber_formation_layer(FiberId(id), Some(layer))
            .unwrap();
        assembly.topology.fibers[(id - 1) as usize].formation_step = layer;
    }
    let config = NeedlingConfig {
        layer: 1,
        selection: NeedlingSelection::CircularFootprint {
            center: [0.5, 0.5],
            diameter: 0.1,
        },
        minimum_fiber_diameter: Some(0.03),
        depth: 0.1,
        stiffness: 1.0,
        max_translation: 0.1,
        maximum_translation_over_fiber_diameter: 0.25,
    };

    let packed = tangle_relax::PackedAssembly::from_assembly(&assembly).unwrap();
    let selected = selected_needling_vertices_from_mask(
        &assembly,
        &packed,
        &packed.positions,
        &packed.vertex_active,
        &packed.segment_active,
        Some(1),
        2,
        config,
    );

    assert_eq!(selected, vec![6]);
}

#[test]
fn random_footprint_center_is_deterministic_and_inside_the_footprint() {
    // Pinned against the felt example's original local implementation.
    assert_eq!(
        random_footprint_center(20_260_940, 2, [0.0, 0.0], [1.0e-3, 1.0e-3]),
        [0.000_797_881_1, 0.000_692_911_7]
    );
    for layer in [0, 1, 7, u32::MAX] {
        let center = random_footprint_center(11, layer, [-1.0, 2.0], [0.5, 0.25]);
        assert_eq!(
            center,
            random_footprint_center(11, layer, [-1.0, 2.0], [0.5, 0.25])
        );
        assert!((-1.0..-0.5).contains(&center[0]));
        assert!((2.0..2.25).contains(&center[1]));
    }
    assert_ne!(
        random_footprint_center(11, 0, [0.0, 0.0], [1.0, 1.0]),
        random_footprint_center(11, 1, [0.0, 0.0], [1.0, 1.0])
    );
}
