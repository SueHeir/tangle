//! Formation recipes and population specifications for the fake felt.

use super::*;

pub(super) fn needled_felt_recipe() -> Vec<FormationOperation> {
    let mut operations = vec![
        FormationOperation::ActivateFibersThrough(0),
        FormationOperation::RelaxFor(200),
    ];
    for layer in 1..LAYER_COUNT {
        operations.extend([
            FormationOperation::ActivateFibersThrough(layer),
            FormationOperation::RelaxFor(100),
            FormationOperation::PlaceLayerAbove {
                layer,
                gap: LAYER_SPACING,
                stiffness: 1.0,
                max_translation: 2.0e-6,
            },
            FormationOperation::RelaxUntilTargetsReached {
                tolerance: 0.5e-6,
                maximum_iterations: 6_000,
            },
            FormationOperation::RelaxFor(80),
            FormationOperation::ReleaseLayerTargets,
            FormationOperation::RelaxFor(200),
        ]);
        if layer >= FIRST_NEEDLED_LAYER {
            operations.extend([
                FormationOperation::NeedleLayer(NeedlingConfig {
                    layer,
                    selection: NeedlingSelection::CircularFootprint {
                        center: random_needle_center(layer),
                        diameter: NEEDLE_DIAMETER,
                    },
                    // The large, bend-tolerant fibers are the carried fibers;
                    // contact remains free to displace every other fiber.
                    minimum_fiber_diameter: Some(0.99 * 19.0e-6),
                    depth: NEEDLE_DEPTH,
                    stiffness: 0.5,
                    max_translation: 1.0e-6,
                    maximum_translation_over_fiber_diameter: 0.25,
                }),
                FormationOperation::RelaxUntilTargetsReached {
                    tolerance: 0.5e-6,
                    maximum_iterations: 3_000,
                },
                FormationOperation::RelaxFor(NEEDLE_TARGET_HOLD_ITERATIONS),
                FormationOperation::ReleaseNeedles,
                FormationOperation::RelaxFor(200),
            ]);
        }
    }
    operations.extend([
        FormationOperation::ReleaseNeedles,
        FormationOperation::Compact(final_compaction()),
        final_relaxation(),
    ]);
    operations
}

pub(super) fn random_needle_center(layer: u32) -> [f32; 2] {
    let x = splitmix64(NEEDLE_POSITION_SEED ^ (2 * layer) as u64);
    let y = splitmix64(NEEDLE_POSITION_SEED ^ (2 * layer + 1) as u64);
    [
        FOOTPRINT as f32 * unit_interval(x),
        FOOTPRINT as f32 * unit_interval(y),
    ]
}

pub(super) fn unit_interval(bits: u64) -> f32 {
    (((bits >> 40) as f64) * (1.0 / ((1_u64 << 24) as f64))) as f32
}

pub(super) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(super) fn dem_polish_recipe() -> Vec<FormationOperation> {
    vec![
        flexible_contact_cleanup(
            "DEM coarse contact polish",
            0.05e-6,
            FINAL_BEND_RATIO,
            75_000,
        ),
        flexible_contact_cleanup(
            "DEM final contact polish",
            DEM_CONTACT_TOLERANCE,
            FINAL_BEND_RATIO,
            150_000,
        ),
    ]
}

pub(super) fn cleanup_recipe() -> Vec<FormationOperation> {
    vec![
        // The original hypothetical 500 um limit is 71 diameters and proved
        // incompatible with a contact-free 13% felt. A 200 um limit remains a
        // tight 29-diameter constraint while admitting the generated weave.
        FormationOperation::SetMaterialBendRadius {
            material_name: "7 um stiff fiber".to_string(),
            minimum_bend_radius: 200.0e-6,
        },
        // First remove the worst contacts without making bend convergence the
        // gate that holds up the stage.
        contact_first_cleanup(),
        // The 12 mm cell is a construction workspace. Remove only its empty
        // z headspace before applying any mechanical compaction.
        FormationOperation::FitCellToActiveFibers {
            axes: [false, false, true],
            padding: 25.0e-6,
        },
        mechanics_recovery(),
        FormationOperation::CompactWithOverrides {
            config: cleanup_compaction(),
            overrides: RelaxationOverrides {
                motion_model: None,
                correction_fraction: Some(0.8),
                contact_aggregation: Some(ContactAggregation::DeepestOnly),
                stretch_stiffness: Some(0.05),
                bend_stiffness: Some(0.0),
                curvature_limit_stiffness: Some(0.15),
                constraint_iterations: Some(1),
                curvature_cleanup_sweeps: Some(1),
            },
        },
        curvature_cleanup("coarse curvature cleanup", 2.0, 20_000),
        flexible_contact_cleanup("coarse contact settling", 2.0e-6, 1.25, 30_000),
        curvature_cleanup("near-admissible curvature cleanup", 1.25, 40_000),
        flexible_contact_cleanup("near-admissible contact settling", 0.75e-6, 1.05, 60_000),
        curvature_cleanup("final hard curvature cleanup", FINAL_BEND_RATIO, 100_000),
        flexible_contact_cleanup(
            "final hard contact settling",
            CONTACT_TOLERANCE,
            FINAL_BEND_RATIO,
            150_000,
        ),
    ]
}

pub(super) fn mechanics_recovery() -> FormationOperation {
    FormationOperation::RelaxWithOverrides {
        policy: SolvePolicy {
            name: "controlled mechanics recovery".to_string(),
            solver_targets: RelaxationTargets {
                penetration: 1.0e-6,
                curvature_ratio: 2.7,
            },
            acceptance: RelaxationAcceptance {
                penetration: AcceptanceLimit::hard(2.0e-6),
                curvature_ratio: AcceptanceLimit::soft(2.7),
            },
            maximum_iterations: 20_000,
            on_exhaustion: SolveExhaustion::ContinueIfHardLimitsSatisfied,
        },
        overrides: RelaxationOverrides {
            motion_model: None,
            correction_fraction: Some(0.7),
            contact_aggregation: Some(ContactAggregation::PenetrationWeighted),
            stretch_stiffness: Some(0.15),
            bend_stiffness: Some(0.01),
            curvature_limit_stiffness: Some(0.5),
            constraint_iterations: Some(4),
            curvature_cleanup_sweeps: Some(2),
        },
    }
}

pub(super) fn contact_first_cleanup() -> FormationOperation {
    FormationOperation::RelaxWithOverrides {
        policy: SolvePolicy {
            name: "contact-first settling".to_string(),
            solver_targets: RelaxationTargets {
                penetration: 1.0e-6,
                curvature_ratio: 5.0,
            },
            acceptance: RelaxationAcceptance {
                penetration: AcceptanceLimit::hard(2.0e-6),
                curvature_ratio: AcceptanceLimit::soft(5.0),
            },
            maximum_iterations: 15_000,
            on_exhaustion: SolveExhaustion::ContinueIfHardLimitsSatisfied,
        },
        overrides: RelaxationOverrides {
            motion_model: None,
            correction_fraction: Some(0.8),
            contact_aggregation: Some(ContactAggregation::DeepestOnly),
            stretch_stiffness: Some(0.05),
            bend_stiffness: Some(0.0),
            curvature_limit_stiffness: Some(0.15),
            constraint_iterations: Some(1),
            curvature_cleanup_sweeps: Some(1),
        },
    }
}

pub(super) fn curvature_cleanup(
    name: &str,
    bend_ratio: f32,
    maximum_iterations: usize,
) -> FormationOperation {
    FormationOperation::RelaxWithOverrides {
        policy: SolvePolicy {
            name: name.to_string(),
            // Curvature projection is allowed to open contacts temporarily.
            // The following rigid stage removes them without changing shape.
            solver_targets: RelaxationTargets {
                penetration: 20.0e-6,
                curvature_ratio: bend_ratio,
            },
            acceptance: RelaxationAcceptance {
                penetration: AcceptanceLimit::soft(20.0e-6),
                curvature_ratio: AcceptanceLimit::hard(bend_ratio),
            },
            maximum_iterations,
            on_exhaustion: SolveExhaustion::Reject,
        },
        overrides: RelaxationOverrides {
            motion_model: Some(FiberMotion::Flexible),
            correction_fraction: Some(0.05),
            contact_aggregation: Some(ContactAggregation::DeepestOnly),
            stretch_stiffness: Some(0.35),
            bend_stiffness: Some(0.0),
            curvature_limit_stiffness: Some(1.0),
            constraint_iterations: Some(8),
            curvature_cleanup_sweeps: Some(16),
        },
    }
}

pub(super) fn flexible_contact_cleanup(
    name: &str,
    penetration: f32,
    bend_ratio: f32,
    maximum_iterations: usize,
) -> FormationOperation {
    FormationOperation::RelaxWithOverrides {
        policy: SolvePolicy {
            name: name.to_string(),
            solver_targets: RelaxationTargets {
                penetration,
                curvature_ratio: bend_ratio,
            },
            acceptance: RelaxationAcceptance {
                penetration: AcceptanceLimit::hard(penetration),
                curvature_ratio: AcceptanceLimit::hard(bend_ratio),
            },
            maximum_iterations,
            on_exhaustion: SolveExhaustion::Reject,
        },
        overrides: RelaxationOverrides {
            motion_model: Some(FiberMotion::Flexible),
            correction_fraction: Some(0.8),
            contact_aggregation: Some(ContactAggregation::DeepestOnly),
            stretch_stiffness: Some(0.35),
            bend_stiffness: Some(0.0),
            curvature_limit_stiffness: Some(0.75),
            constraint_iterations: Some(8),
            curvature_cleanup_sweeps: Some(16),
        },
    }
}

pub(super) fn cleanup_compaction() -> CompactionConfig {
    CompactionConfig {
        guards: CompactionGuards {
            // Construction is intentionally allowed to remain soft. The
            // staged cleanup operations following compaction own the hard
            // physical acceptance decision.
            maximum_penetration: 5.0e-6,
            maximum_bend_ratio: 100.0,
            maximum_steps: 512,
            maximum_relax_windows: 16,
            ..CompactionGuards::default()
        },
        increment: AdaptiveCompactionIncrement {
            initial_log_strain: 0.001,
            minimum_log_strain: 0.000_1,
            maximum_log_strain: 0.005,
            growth_factor: 1.15,
            shrink_factor: 0.5,
            relax_iterations: 250,
            maximum_shortening_over_minimum_diameter: 0.25,
        },
        ..final_compaction()
    }
}

pub(super) fn felt_recipe() -> Vec<FormationOperation> {
    let mut operations = vec![
        FormationOperation::ActivateFibersThrough(0),
        FormationOperation::RelaxFor(200),
    ];
    for layer in 1..LAYER_COUNT {
        operations.extend([
            FormationOperation::ActivateFibersThrough(layer),
            FormationOperation::RelaxFor(100),
            FormationOperation::PlaceLayerAbove {
                layer,
                gap: LAYER_SPACING,
                stiffness: 1.0,
                max_translation: 2.0e-6,
            },
            FormationOperation::RelaxUntilTargetsReached {
                tolerance: 0.5e-6,
                maximum_iterations: 6_000,
            },
            FormationOperation::RelaxFor(80),
            FormationOperation::ReleaseLayerTargets,
            FormationOperation::RelaxFor(200),
        ]);
    }
    operations.extend([
        FormationOperation::Compact(final_compaction()),
        final_relaxation(),
    ]);
    operations
}

pub(super) fn final_relaxation() -> FormationOperation {
    FormationOperation::RelaxWithPolicy(SolvePolicy {
        name: "final admissibility relaxation".to_string(),
        solver_targets: final_solver_targets(),
        acceptance: RelaxationAcceptance {
            penetration: AcceptanceLimit::hard(CONTACT_TOLERANCE),
            curvature_ratio: AcceptanceLimit::hard(FINAL_BEND_RATIO),
        },
        maximum_iterations: 100_000,
        on_exhaustion: SolveExhaustion::Reject,
    })
}

pub(super) fn final_solver_targets() -> RelaxationTargets {
    RelaxationTargets {
        penetration: CONTACT_TOLERANCE,
        curvature_ratio: 1.000_01,
    }
}

pub(super) fn final_compaction() -> CompactionConfig {
    CompactionConfig {
        target: CompactionTarget::NominalVolumeFraction(TARGET_VOLUME_FRACTION),
        path: CompactionPath::AxisWeights([0.0, 0.0, 1.0]),
        kinematics: CompactionKinematics::MovingWalls,
        // Close the nonperiodic z cell symmetrically so the batting is
        // compacted by both platens instead of only by the top platen.
        cell_anchor: [0.0, 0.0, 0.5],
        balance_opposing_faces: true,
        face_pressure_floor: 1.0e-9,
        face_balance_strength: 0.5,
        increment: AdaptiveCompactionIncrement {
            initial_log_strain: 0.02,
            minimum_log_strain: 0.001,
            maximum_log_strain: 0.04,
            growth_factor: 1.2,
            shrink_factor: 0.5,
            relax_iterations: 1_000,
            maximum_shortening_over_minimum_diameter: 0.5,
        },
        guards: CompactionGuards {
            maximum_penetration: 0.301e-6,
            maximum_bend_ratio: 1.05,
            maximum_steps: 512,
            maximum_relax_windows: 8,
            ..CompactionGuards::default()
        },
        energy_model: CompactionEnergyModel::default(),
        target_tolerance: 1.0e-3,
    }
}

pub(super) fn common_population(
    count: usize,
    material_name: &str,
    seed: u64,
) -> FiberPopulationSpec {
    FiberPopulationSpec {
        count,
        segments_per_fiber: SEGMENTS_PER_FIBER,
        seed,
        nominal_parent_length: Some(0.050_8),
        length: ScalarDistribution::Uniform {
            minimum: 0.003,
            maximum: 0.004,
        },
        orientation: OrientationDistribution::LayeredBiaxial {
            normal: [0.0, 0.0, 1.0],
            primary_fraction: 0.40,
            cross_fraction: 0.40,
            maximum_in_plane_deviation: PI / 18.0,
            maximum_tilt: PI / 1_800.0,
            layer_seed: 20_260_920,
        },
        position: PositionDistribution::Layered {
            axis: 2,
            layers: LAYER_COUNT as usize,
            jitter_fraction: 0.20,
        },
        max_attempts_per_fiber: 2_048,
        material_name: material_name.to_string(),
        ..FiberPopulationSpec::default()
    }
}

pub(super) fn small_fibers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        radius: ScalarDistribution::Constant(3.5e-6),
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 2.0e-6,
        },
        // Still a tight constraint at about 29 fiber diameters, but compatible
        // with a contact-free 13% felt without requiring brittle breakage.
        minimum_bend_radius: Some(0.000_200),
        ..common_population(SMALL_FIBERS, "7 um stiff fiber", 20_260_921)
    }
}

pub(super) fn large_fibers() -> FiberPopulationSpec {
    FiberPopulationSpec {
        radius: ScalarDistribution::Constant(9.5e-6),
        intrinsic_curvature_amplitude: ScalarDistribution::Uniform {
            minimum: 0.0,
            maximum: 8.0e-6,
        },
        minimum_bend_radius: Some(60.0e-6),
        ..common_population(LARGE_FIBERS, "19 um bendy fiber", 20_260_922)
    }
}

pub(super) fn print_population_design() {
    let mean_length = 3.5e-3;
    let small_volume = SMALL_FIBERS as f64 * PI * (3.5e-6_f64).powi(2) * mean_length;
    let large_volume = LARGE_FIBERS as f64 * PI * (9.5e-6_f64).powi(2) * mean_length;
    println!("fake felted population design:");
    println!("  20 plies in a 1.0 mm x 1.0 mm periodic footprint");
    println!("  {SMALL_FIBERS} x 7 um fibers, {LARGE_FIBERS} x 19 um fibers");
    println!(
        "  expected volume split: {:.2}% small / {:.2}% large (nominal Vf {:.3} at 1 mm thick)",
        100.0 * small_volume / (small_volume + large_volume),
        100.0 * large_volume / (small_volume + large_volume),
        (small_volume + large_volume) / (FOOTPRINT * FOOTPRINT * 1.0e-3)
    );
}
