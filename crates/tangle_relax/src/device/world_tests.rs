use super::*;
#[cfg(all(not(feature = "wgpu"), feature = "cpu"))]
use cubecl::cpu::{CpuDevice as WgpuDevice, CpuRuntime as WgpuRuntime};
#[cfg(feature = "wgpu")]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use tangle_core::{FiberAssembly, FiberBendLimit, FiberId, PeriodicCell, Section};

fn crossed_assembly() -> FiberAssembly {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    for (id, placed) in [
        (1, [[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]]),
        (2, [[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]]),
    ] {
        assembly
            .add_fiber(FiberId(id), material, section, &placed, &placed)
            .unwrap();
    }
    assembly
}

#[test]
fn flexible_crossed_fibers_relax_on_device() {
    let assembly = crossed_assembly();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        max_step: 0.2,
        max_iterations: 20,
        iterations_per_batch: 20,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 20);
    assert!(status.converged, "{status:?}");
    assert!(status.total_iterations > 0);
    assert!(status.max_penetration <= config.penetration_tolerance);
}

#[test]
fn formation_steps_activate_prepacked_fibers_without_reupload() {
    let mut assembly = crossed_assembly();
    assembly.set_fiber_formation_step(FiberId(2), 1).unwrap();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );

    assert_eq!(world.activate_formation_step(0), (1, 2));
    let positions = world.download_positions();
    let mut first_stage = assembly.clone();
    world
        .unpack_positions(&positions, &mut first_stage)
        .unwrap();
    assert_eq!(first_stage.topology.fibers.len(), 1);
    assert_eq!(first_stage.topology.fibers[0].id, FiberId(1));

    assert_eq!(world.activate_formation_step(1), (2, 4));
    let positions = world.download_positions();
    let mut second_stage = assembly.clone();
    world
        .unpack_positions(&positions, &mut second_stage)
        .unwrap();
    assert_eq!(second_stage.topology.fibers.len(), 2);
}

#[test]
fn staged_activation_preserves_adaptive_splits() {
    let mut assembly = crossed_assembly();
    assembly.set_fiber_formation_step(FiberId(2), 1).unwrap();
    let adaptive = AdaptiveSegmentationConfig {
        contact_length_over_diameter: 2.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 4,
        refinement_interval: 1,
        refinement_persistence: 1,
        coarsening_persistence: 0,
        ..AdaptiveSegmentationConfig::default()
    };
    let packed =
        PackedAssembly::from_assembly_with_options(&assembly, Some(adaptive), false).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );

    assert_eq!(world.activate_formation_step(0), (1, 2));
    assert_eq!(world.activate_formation_step(1), (2, 4));
    let config = RelaxationConfig {
        adaptive_segmentation: Some(adaptive),
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        constraint_iterations: 4,
        max_step: 0.2,
        max_iterations: 20,
        iterations_per_batch: 20,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 20);
    assert!(status.segment_splits > 0, "{status:?}");
    let active_after_refinement = (status.segment_splits + 2, status.segment_splits + 4);
    assert_eq!(
        world.activate_formation_step(1),
        active_after_refinement,
        "reactivating a formation step reset adaptive topology"
    );
}

#[test]
fn adaptive_refinement_requires_persistent_contact() {
    let assembly = crossed_assembly();
    let adaptive = AdaptiveSegmentationConfig {
        contact_length_over_diameter: 2.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 4,
        refinement_interval: 1,
        refinement_persistence: 3,
        coarsening_persistence: 0,
        ..AdaptiveSegmentationConfig::default()
    };
    let packed =
        PackedAssembly::from_assembly_with_options(&assembly, Some(adaptive), true).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    let config = RelaxationConfig {
        adaptive_segmentation: Some(adaptive),
        pin_fiber_ends: true,
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        max_step: 0.2,
        max_iterations: 10,
        iterations_per_batch: 10,
        ..RelaxationConfig::default()
    };

    let early = world.run_batch(&config, 2);
    assert_eq!(early.segment_splits, 0, "{early:?}");
    let persistent = world.run_batch(&config, 1);
    assert!(persistent.segment_splits > 0, "{persistent:?}");
}

fn manually_refined_straight_fiber(
    pin_midpoint: bool,
    midpoint_offset: f32,
) -> (PackedAssembly, AdaptiveSegmentationConfig, u32) {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let points = [[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]];
    assembly
        .add_fiber(FiberId(1), material, section, &points, &points)
        .unwrap();
    let adaptive = AdaptiveSegmentationConfig {
        // Disable further splitting in this coarsening fixture.
        contact_length_over_diameter: 100.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 4,
        refinement_interval: 1,
        refinement_persistence: 1,
        coarsening_persistence: 2,
        coarsening_error_over_diameter: 0.1,
        coarsening_curvature_ratio: 0.25,
    };
    let mut packed =
        PackedAssembly::from_assembly_with_options(&assembly, Some(adaptive), false).unwrap();
    let root = 0_usize;
    let left = packed.segment_children[2 * root] as usize;
    let right = packed.segment_children[2 * root + 1] as usize;
    let first = packed.segment_vertices[2 * root] as usize;
    let second = packed.segment_vertices[2 * root + 1] as usize;
    let midpoint = (first + second) / 2;
    packed.segment_active[root] = 0;
    packed.segment_active[left] = 1;
    packed.segment_active[right] = 1;
    packed.segment_birth_epochs[left] = 0;
    packed.segment_birth_epochs[right] = 0;
    packed.vertex_active[midpoint] = 1;
    packed.vertex_segments[2 * first + 1] = left as u32;
    packed.vertex_segments[2 * midpoint] = left as u32;
    packed.vertex_segments[2 * midpoint + 1] = right as u32;
    packed.vertex_segments[2 * second] = right as u32;
    packed.vertex_pinned[midpoint] = u32::from(pin_midpoint);
    packed.positions[3 * midpoint + 1] += midpoint_offset;
    (packed, adaptive, midpoint as u32)
}

fn forced_adaptation_config(adaptive: AdaptiveSegmentationConfig) -> RelaxationConfig {
    RelaxationConfig {
        adaptive_segmentation: Some(adaptive),
        // Keep the device loop active while the deliberately contact-free
        // fixture accumulates its coarsening cooldown.
        force_full_iterations: true,
        max_step: 0.2,
        max_iterations: 10,
        iterations_per_batch: 10,
        ..RelaxationConfig::default()
    }
}

#[test]
fn adaptive_coarsening_merges_quiet_collinear_siblings() {
    let (packed, adaptive, midpoint) = manually_refined_straight_fiber(false, 0.0);
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    let status = world.run_batch(&forced_adaptation_config(adaptive), 2);
    assert_eq!(status.segment_merges, 1, "{status:?}");
    assert_eq!(world.download_segment_active().iter().sum::<u32>(), 1);
    assert_eq!(world.download_vertex_active()[midpoint as usize], 0);
}

#[test]
fn adaptive_coarsening_preserves_pinned_or_deformed_midpoints() {
    for (pinned, offset) in [(true, 0.0), (false, 0.05)] {
        let (packed, mut adaptive, _) = manually_refined_straight_fiber(pinned, offset);
        adaptive.coarsening_persistence = 1;
        let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
            &WgpuDevice::default(),
            packed,
            CellListConfig::default(),
            0.2,
        );
        let status = world.run_batch(&forced_adaptation_config(adaptive), 3);
        assert_eq!(status.segment_merges, 0, "{status:?}");
        assert_eq!(world.download_segment_active().iter().sum::<u32>(), 2);
    }
}

#[test]
fn adaptive_coarsening_waits_for_active_vertex_target_release() {
    let (packed, mut adaptive, midpoint) = manually_refined_straight_fiber(false, 0.0);
    adaptive.coarsening_persistence = 1;
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    world.apply_vertex_displacement_targets(0, &[midpoint], 1.0e-6, 1.0, 0.1);
    let held = world.run_batch(&forced_adaptation_config(adaptive), 2);
    assert_eq!(held.segment_merges, 0, "{held:?}");
    world.clear_vertex_targets();
    let released = world.run_batch(&forced_adaptation_config(adaptive), 1);
    assert_eq!(released.segment_merges, 1, "{released:?}");
}

#[test]
fn fully_pinned_overlap_is_not_accepted() {
    let assembly = crossed_assembly();
    let packed = PackedAssembly::from_assembly_with_options(&assembly, None, true).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        max_step: 0.2,
        max_iterations: 5,
        iterations_per_batch: 5,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 5);
    assert!(!status.converged, "{status:?}");
    assert!(status.max_penetration > config.penetration_tolerance);
}

#[test]
fn adaptive_refinement_is_independent_of_batch_boundaries() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let angle = 12.0_f64.to_radians();
    for (id, direction, z) in [
        (1, [1.0, 0.0, 0.0], 1.99),
        (2, [angle.cos(), angle.sin(), 0.0], 2.01),
    ] {
        let placed = [
            [2.0 - direction[0], 2.0 - direction[1], z],
            [2.0 + direction[0], 2.0 + direction[1], z],
        ];
        assembly
            .add_fiber(FiberId(id), material, section, &placed, &placed)
            .unwrap();
    }
    let adaptive = AdaptiveSegmentationConfig {
        contact_length_over_diameter: 2.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 6,
        refinement_interval: 3,
        refinement_persistence: 1,
        coarsening_persistence: 0,
        ..AdaptiveSegmentationConfig::default()
    };
    let config = RelaxationConfig {
        adaptive_segmentation: Some(adaptive),
        pin_fiber_ends: true,
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        stretch_stiffness: 0.5,
        bend_stiffness: 0.15,
        constraint_iterations: 4,
        max_step: 0.02,
        max_iterations: 100,
        ..RelaxationConfig::default()
    };

    let run = |batch_size| {
        let packed = PackedAssembly::from_assembly_with_options(
            &assembly,
            Some(adaptive),
            config.pin_fiber_ends,
        )
        .unwrap();
        let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
            &WgpuDevice::default(),
            packed,
            CellListConfig::default(),
            config.max_step,
        );
        let mut status = BatchStatus::default();
        while !status.converged && status.total_iterations < config.max_iterations {
            status = world.run_batch(&config, batch_size);
        }
        (status, world.download_vertex_active())
    };

    let (short_batches, short_active) = run(2);
    let (long_batches, long_active) = run(7);
    assert!(short_batches.converged, "{short_batches:?}");
    assert!(long_batches.converged, "{long_batches:?}");
    assert_eq!(short_batches.segment_splits, long_batches.segment_splits);
    assert_eq!(
        short_batches.refinement_passes,
        long_batches.refinement_passes
    );
    assert_eq!(short_active, long_active);
}

#[test]
fn compact_cell_list_accepts_dense_cells_without_capacity_tuning() {
    // The long, thin box creates more than one 256-cell scan block, while
    // all three segments deliberately occupy the same high-index cell.
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([40.0, 0.2, 0.2], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.02 });
    let placed = [[39.0, 0.1, 0.1], [39.01, 0.1, 0.1]];
    for id in 1..=3 {
        assembly
            .add_fiber(FiberId(id), material, section, &placed, &placed)
            .unwrap();
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.01,
    );
    let config = RelaxationConfig {
        max_step: 0.01,
        max_iterations: 5,
        iterations_per_batch: 5,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 5);
    assert!(!status.cell_list_overflow, "{status:?}");
    assert!(status.max_penetration.is_finite());
}

#[test]
fn curvature_limit_moves_a_free_middle_between_pinned_ends() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    assembly
        .add_fiber(
            FiberId(1),
            material,
            section,
            &[[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
            &[[0.8, 1.0, 1.0], [1.0, 1.15, 1.0], [1.2, 1.0, 1.0]],
        )
        .unwrap();
    assembly
        .set_fiber_bend_limit(
            FiberId(1),
            Some(FiberBendLimit {
                minimum_bend_radius: 0.25,
            }),
        )
        .unwrap();
    let packed = PackedAssembly::from_assembly_with_options(&assembly, None, true).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.02,
    );
    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-5,
        correction_fraction: 0.8,
        stretch_stiffness: 0.0,
        bend_stiffness: 0.0,
        curvature_limit_stiffness: 1.0,
        curvature_ratio_tolerance: 1.0e-5,
        constraint_iterations: 1,
        max_step: 0.02,
        max_iterations: 40,
        iterations_per_batch: 40,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 40);
    assert!(status.converged, "{status:?}");
    assert!(status.max_curvature_ratio <= 1.0 + 1.0e-5);
    let positions = world.download_positions();
    assert_eq!(&positions[0..3], &[0.8, 1.0, 1.0]);
    assert_eq!(&positions[6..9], &[1.2, 1.0, 1.0]);
    assert!(positions[4] < 1.15);
}

#[test]
fn fully_pinned_bend_violation_is_not_accepted() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    assembly
        .add_fiber(
            FiberId(1),
            material,
            section,
            &[[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
            &[[0.8, 1.0, 1.0], [1.0, 1.15, 1.0], [1.2, 1.0, 1.0]],
        )
        .unwrap();
    assembly
        .set_fiber_bend_limit(
            FiberId(1),
            Some(FiberBendLimit {
                minimum_bend_radius: 0.25,
            }),
        )
        .unwrap();
    let mut packed = PackedAssembly::from_assembly(&assembly).unwrap();
    packed.vertex_pinned.fill(1);
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.02,
    );
    let config = RelaxationConfig {
        stretch_stiffness: 0.0,
        bend_stiffness: 0.0,
        curvature_limit_stiffness: 1.0,
        curvature_ratio_tolerance: 1.0e-5,
        constraint_iterations: 1,
        max_iterations: 5,
        iterations_per_batch: 5,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 5);
    assert!(!status.converged, "{status:?}");
    assert!(status.max_curvature_ratio > 1.0 + config.curvature_ratio_tolerance);
}

#[test]
fn layer_target_command_updates_resident_geometry_without_reupload() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    for (id, layer, z) in [(1, 0, 0.2), (2, 1, 0.8)] {
        assembly
            .add_fiber(
                FiberId(id),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                &[[0.4, 0.5, z], [0.6, 0.5, z]],
            )
            .unwrap();
        assembly
            .set_fiber_formation_layer(FiberId(id), Some(layer))
            .unwrap();
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.01,
    );
    world.apply_layer_targets(2, &[0.3, 0.7], 1.0, 1.0);
    let positions = world.download_positions();
    let z: Vec<_> = positions.chunks_exact(3).map(|point| point[2]).collect();
    assert!(z[..2].iter().all(|value| (*value - 0.3).abs() < 1.0e-6));
    assert!(z[2..].iter().all(|value| (*value - 0.7).abs() < 1.0e-6));
}

#[test]
fn single_layer_target_leaves_the_relaxed_stack_free() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    for (id, layer, z) in [(1, 0, 0.24), (2, 1, 0.8)] {
        assembly
            .add_fiber(
                FiberId(id),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                &[[0.4, 0.5, z], [0.6, 0.5, z]],
            )
            .unwrap();
        assembly
            .set_fiber_formation_layer(FiberId(id), Some(layer))
            .unwrap();
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.01,
    );
    world.apply_single_layer_target(2, &[0.2, 0.7], 1, 1.0, 1.0);
    let target_error = world.formation_target_error();
    assert_eq!(target_error.vertex, None);
    assert!(target_error.layer.unwrap() < 1.0e-6);
    let positions = world.download_positions();
    assert!((positions[2] - 0.24).abs() < 1.0e-6);
    assert!((positions[5] - 0.24).abs() < 1.0e-6);
    assert!((positions[8] - 0.7).abs() < 1.0e-6);
    assert!((positions[11] - 0.7).abs() < 1.0e-6);
}

#[test]
fn layer_targets_do_not_move_inactive_future_layers() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    for (id, layer, z) in [(1, 0, 0.2), (2, 1, 0.8)] {
        assembly
            .add_fiber(
                FiberId(id),
                material,
                section,
                &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                &[[0.4, 0.5, z], [0.6, 0.5, z]],
            )
            .unwrap();
        assembly
            .set_fiber_formation_layer(FiberId(id), Some(layer))
            .unwrap();
        assembly.topology.fibers[(id - 1) as usize].formation_step = layer;
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.01,
    );
    world.activate_formation_step(0);
    world.apply_layer_targets(2, &[0.3, 0.7], 1.0, 1.0);
    let positions = world.download_positions();
    assert!((positions[2] - 0.3).abs() < 1.0e-6);
    assert!((positions[8] - 0.8).abs() < 1.0e-6);
}

#[test]
fn vertex_displacement_target_is_initialized_from_resident_geometry() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    assembly
        .add_fiber(
            FiberId(1),
            material,
            section,
            &[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            &[[0.4, 0.5, 0.5], [0.5, 0.5, 0.5], [0.6, 0.5, 0.5]],
        )
        .unwrap();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.1,
    );
    world.apply_vertex_displacement_targets(2, &[1], -0.1, 1.0, 0.2);
    let initial_error = world.formation_target_error();
    assert_eq!(initial_error.layer, None);
    assert!((initial_error.vertex.unwrap() - 0.1).abs() < 1.0e-6);
    let config = RelaxationConfig {
        motion_model: FiberMotion::RigidTranslation,
        force_full_iterations: true,
        ..RelaxationConfig::default()
    };
    world.run_batch(&config, 1);
    assert!(world.formation_target_error().vertex.unwrap().abs() < 1.0e-6);
    let positions = world.download_positions();
    assert!((positions[5] - 0.4).abs() < 1.0e-6);
    assert!((positions[2] - 0.5).abs() < 1.0e-6);
    assert!((positions[8] - 0.5).abs() < 1.0e-6);
}

#[test]
fn vertex_displacement_command_advances_independently_of_mechanics() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    assembly
        .add_fiber(
            FiberId(1),
            material,
            section,
            &[[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
            &[[0.8, 1.0, 1.0], [1.0, 1.0, 1.0], [1.2, 1.0, 1.0]],
        )
        .unwrap();
    assembly
        .set_fiber_bend_limit(
            FiberId(1),
            Some(FiberBendLimit {
                minimum_bend_radius: 0.25,
            }),
        )
        .unwrap();
    let packed = PackedAssembly::from_assembly_with_options(&assembly, None, true).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.02,
    );
    world.apply_vertex_displacement_targets(1, &[1], -0.05, 1.0, 0.01);
    let config = RelaxationConfig {
        force_full_iterations: true,
        stretch_stiffness: 1.0,
        bend_stiffness: 1.0,
        curvature_limit_stiffness: 1.0,
        constraint_iterations: 4,
        ..RelaxationConfig::default()
    };

    world.run_batch(&config, 3);

    let checkpoint = world.checkpoint();
    let target = checkpoint.vertex_target.unwrap();
    assert!((target.coordinates[0] - 0.95).abs() < 1.0e-6);
    assert!((target.command_coordinates[0] - 0.97).abs() < 1.0e-6);
    assert!((checkpoint.packed.positions[4] - 0.97).abs() < 1.0e-6);
}

#[test]
fn persistent_layer_targets_are_independent_of_batch_boundaries() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    assembly
        .add_fiber(
            FiberId(1),
            material,
            section,
            &[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            &[[0.4, 0.5, 0.2], [0.6, 0.5, 0.2]],
        )
        .unwrap();
    assembly
        .set_fiber_formation_layer(FiberId(1), Some(0))
        .unwrap();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let config = RelaxationConfig {
        force_full_iterations: true,
        ..RelaxationConfig::default()
    };
    let run = |batches: &[usize]| {
        let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
            &WgpuDevice::default(),
            packed.clone(),
            CellListConfig::default(),
            0.01,
        );
        world.apply_layer_targets(2, &[0.8], 0.2, 1.0);
        for &iterations in batches {
            world.run_batch(&config, iterations);
        }
        world.download_positions()
    };

    let split = run(&[2, 3]);
    let single = run(&[5]);
    assert_eq!(split, single);
}

#[test]
fn rigid_translation_preserves_centerline_vectors() {
    let assembly = crossed_assembly();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let initial_vectors: Vec<_> = packed
        .positions
        .chunks_exact(6)
        .map(|fiber| {
            [
                fiber[3] - fiber[0],
                fiber[4] - fiber[1],
                fiber[5] - fiber[2],
            ]
        })
        .collect();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    let config = RelaxationConfig {
        motion_model: FiberMotion::RigidTranslation,
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        max_step: 0.2,
        max_iterations: 20,
        iterations_per_batch: 20,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 20);
    assert!(status.converged, "{status:?}");
    let positions = world.download_positions();
    for (fiber, expected) in positions.chunks_exact(6).zip(initial_vectors) {
        let actual = [
            fiber[3] - fiber[0],
            fiber[4] - fiber[1],
            fiber[5] - fiber[2],
        ];
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-6),
            "actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn rigid_translation_preserves_adaptive_discretization() {
    let assembly = crossed_assembly();
    let adaptive = AdaptiveSegmentationConfig {
        contact_length_over_diameter: 2.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 4,
        refinement_interval: 1,
        refinement_persistence: 1,
        coarsening_persistence: 0,
        ..AdaptiveSegmentationConfig::default()
    };
    let packed =
        PackedAssembly::from_assembly_with_options(&assembly, Some(adaptive), false).unwrap();
    let initial_active = packed
        .segment_active
        .iter()
        .filter(|active| **active != 0)
        .count();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.2,
    );
    let config = RelaxationConfig {
        motion_model: FiberMotion::RigidTranslation,
        adaptive_segmentation: Some(adaptive),
        force_full_iterations: true,
        penetration_tolerance: 1.0e-5,
        correction_fraction: 1.0,
        max_step: 0.2,
        max_iterations: 5,
        iterations_per_batch: 5,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 5);
    assert_eq!(status.segment_splits, 0, "{status:?}");
    assert_eq!(world.active_segment_count, initial_active);
}

#[test]
fn rigid_center_compaction_updates_cell_without_straining_fibers() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let points = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
    assembly
        .add_fiber(FiberId(1), material, section, &points, &points)
        .unwrap();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.1,
    );

    world.compact_cell(
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 4.0],
        CompactionKinematics::RigidFiberCenters,
    );

    assert_eq!(world.cell_bounds(), ([0.0; 3], [2.0, 4.0, 4.0]));
    let positions = world.download_positions();
    assert!((positions[3] - positions[0] - 1.0).abs() < 1.0e-6);
    assert!((0.5 * (positions[0] + positions[3]) - 0.75).abs() < 1.0e-6);
}

#[test]
fn affine_compaction_scales_centerline_with_cell() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let points = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
    assembly
        .add_fiber(FiberId(1), material, section, &points, &points)
        .unwrap();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.1,
    );

    world.compact_cell(
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 4.0],
        CompactionKinematics::AffineVertices,
    );

    let positions = world.download_positions();
    assert!((positions[3] - positions[0] - 0.5).abs() < 1.0e-6);
}

#[test]
fn periodic_broad_phase_captures_minimum_image_contact() {
    let mut assembly =
        FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [true, false, false]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    for (id, x) in [(1, 0.05), (2, 3.95)] {
        let points = [[x, 1.5, 2.0], [x, 2.5, 2.0]];
        assembly
            .add_fiber(FiberId(id), material, section, &points, &points)
            .unwrap();
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.1,
    );

    let capture = world.capture_contacts(0.0, 16);
    assert!(!capture.overflow);
    assert_eq!(capture.candidates.len(), 1, "{capture:?}");
    assert!((capture.candidates[0].surface_gap + 0.1).abs() < 1.0e-5);

    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-5,
        correction_fraction: 0.8,
        max_step: 0.05,
        max_iterations: 40,
        iterations_per_batch: 40,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 40);
    assert!(status.converged, "{status:?}");
    assert!(status.max_penetration <= config.penetration_tolerance);
}

#[test]
fn si_scale_interior_crossing_is_not_misclassified_as_parallel() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([0.002; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 10.0e-6 });
    for (id, points) in [
        (1, [[0.000_6, 0.001, 0.001], [0.001_4, 0.001, 0.001]]),
        (2, [[0.001, 0.000_6, 0.001], [0.001, 0.001_4, 0.001]]),
    ] {
        assembly
            .add_fiber(FiberId(id), material, section, &points, &points)
            .unwrap();
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        1.0e-6,
    );

    let capture = world.capture_contacts(0.0, 16);
    assert!(!capture.overflow);
    assert_eq!(capture.candidates.len(), 1, "{capture:?}");
    assert!((capture.candidates[0].surface_gap + 20.0e-6).abs() < 0.01e-6);
}

#[test]
fn periodic_contact_reduces_multiple_unwrapped_box_images() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic(
        [0.002_5, 0.002_5, 0.002],
        [true, true, false],
    ));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 10.0e-6 });
    for (id, points) in [
        (1, [[0.000_5, 0.000_6, 0.001], [0.000_5, 0.001_4, 0.001]]),
        (2, [[0.005_1, 0.001, 0.001], [0.005_9, 0.001, 0.001]]),
    ] {
        assembly
            .add_fiber(FiberId(id), material, section, &points, &points)
            .unwrap();
    }
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        1.0e-6,
    );

    let capture = world.capture_contacts(0.0, 16);
    assert!(!capture.overflow);
    assert_eq!(capture.candidates.len(), 1, "{capture:?}");
    assert!((capture.candidates[0].surface_gap + 20.0e-6).abs() < 0.01e-6);
}

#[test]
fn periodic_broad_phase_resolves_nonadjacent_self_image_contact() {
    let mut assembly =
        FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [true, false, false]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let points = [
        [0.05, 1.0, 2.0],
        [0.05, 1.5, 2.0],
        [2.0, 1.5, 2.0],
        [3.95, 1.5, 2.0],
        [3.95, 1.0, 2.0],
    ];
    assembly
        .add_fiber(FiberId(1), material, section, &points, &points)
        .unwrap();
    let packed = PackedAssembly::from_assembly(&assembly).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.1,
    );
    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-4,
        correction_fraction: 0.8,
        max_step: 0.05,
        max_iterations: 1_000,
        iterations_per_batch: 1_000,
        ..RelaxationConfig::default()
    };

    let initial = world.run_batch(&config, 1);
    assert!(initial.max_penetration > 0.05, "{initial:?}");
    let status = world.run_batch(&config, 999);
    assert!(status.max_penetration < 0.005, "{status:?}");
}

#[test]
fn curvature_limit_reaches_vertices_packed_after_dormant_fibers() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.01 });
    // A dormant fiber packed first places every active vertex past the
    // active-vertex count, so kernels sized by that count must not index
    // vertices directly.
    let dormant = (0..200)
        .map(|index| [0.1 + 0.004 * index as f64, 0.2, 0.2])
        .collect::<Vec<_>>();
    assembly
        .add_fiber(FiberId(1), material, section, &dormant, &dormant)
        .unwrap();
    assembly.set_fiber_formation_step(FiberId(1), 1).unwrap();
    assembly
        .add_fiber(
            FiberId(2),
            material,
            section,
            &[[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
            &[[0.8, 1.0, 1.0], [1.0, 1.15, 1.0], [1.2, 1.0, 1.0]],
        )
        .unwrap();
    assembly
        .set_fiber_bend_limit(
            FiberId(2),
            Some(FiberBendLimit {
                minimum_bend_radius: 0.25,
            }),
        )
        .unwrap();
    let packed = PackedAssembly::from_assembly_with_options(&assembly, None, true).unwrap();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.02,
    );
    assert_eq!(world.activate_formation_step(0), (2, 3));
    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-5,
        correction_fraction: 0.8,
        stretch_stiffness: 0.0,
        bend_stiffness: 0.0,
        curvature_limit_stiffness: 1.0,
        curvature_ratio_tolerance: 1.0e-5,
        constraint_iterations: 1,
        max_step: 0.02,
        max_iterations: 40,
        iterations_per_batch: 40,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 40);
    assert!(status.converged, "{status:?}");
    assert!(status.max_curvature_ratio <= 1.0 + 1.0e-5);
}
