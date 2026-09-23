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

/// A pinned 81-vertex zig-zag whose every interior vertex violates the bend
/// limit, relaxed by curvature projection alone.
fn pinned_zigzag_world() -> DeviceFiberWorld<WgpuRuntime> {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.005 });
    let intrinsic = (0..81)
        .map(|index| [0.02 * index as f64, 0.0, 0.0])
        .collect::<Vec<_>>();
    let placed = (0..81)
        .map(|index| {
            let wiggle = if index % 2 == 0 { 0.0 } else { 0.01 };
            [0.2 + 0.02 * index as f64, 1.0 + wiggle, 1.0]
        })
        .collect::<Vec<_>>();
    assembly
        .add_fiber(FiberId(1), material, section, &intrinsic, &placed)
        .unwrap();
    assembly
        .set_fiber_bend_limit(
            FiberId(1),
            Some(FiberBendLimit {
                minimum_bend_radius: 0.3,
            }),
        )
        .unwrap();
    let packed = PackedAssembly::from_assembly_with_options(&assembly, None, true).unwrap();
    DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.02,
    )
}

#[test]
fn curvature_cleanup_straightens_a_long_zigzag_in_few_iterations() {
    let mut world = pinned_zigzag_world();
    let config = RelaxationConfig {
        penetration_tolerance: 1.0e-5,
        stretch_stiffness: 0.0,
        bend_stiffness: 0.0,
        curvature_limit_stiffness: 1.0,
        curvature_ratio_tolerance: 1.0e-5,
        constraint_iterations: 1,
        curvature_cleanup_sweeps: 1,
        max_step: 0.02,
        max_iterations: ZIGZAG_ITERATION_BUDGET,
        iterations_per_batch: ZIGZAG_ITERATION_BUDGET,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, ZIGZAG_ITERATION_BUDGET);
    // Gauss–Seidel sweeps let each projection build on its neighbors'
    // corrections along the chain; averaged (Jacobi) projections of the same
    // triplets need more than twice as many iterations here.
    assert!(status.converged, "{status:?}");
    assert!(status.max_curvature_ratio <= 1.0 + 1.0e-5, "{status:?}");
    let positions = world.download_positions();
    assert_eq!(&positions[0..3], &[0.2, 1.0, 1.0]);
    assert_eq!(&positions[240..243], &[1.8, 1.0, 1.0]);
}

// Gauss–Seidel converges this fixture in 22 iterations on Metal; the averaged
// Jacobi sweeps it replaced needed 50.
const ZIGZAG_ITERATION_BUDGET: usize = 30;

#[test]
fn curvature_cleanup_skips_unrefined_midpoint_slots() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([4.0; 3], [false; 3]));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let points = [[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]];
    assembly
        .add_fiber(FiberId(1), material, section, &points, &points)
        .unwrap();
    assembly
        .set_fiber_bend_limit(
            FiberId(1),
            Some(FiberBendLimit {
                minimum_bend_radius: 4.0,
            }),
        )
        .unwrap();
    let adaptive = AdaptiveSegmentationConfig {
        contact_length_over_diameter: 100.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 4,
        refinement_interval: 1_000,
        ..AdaptiveSegmentationConfig::default()
    };
    let mut packed =
        PackedAssembly::from_assembly_with_options(&assembly, Some(adaptive), true).unwrap();
    // Activate only the middle of the nine reserved vertex slots, so the
    // active chain 0 - 4 - 8 is separated by inactive slots.
    let root = 0_usize;
    let left = packed.segment_children[2 * root] as usize;
    let right = packed.segment_children[2 * root + 1] as usize;
    let first = packed.segment_vertices[2 * root] as usize;
    let second = packed.segment_vertices[2 * root + 1] as usize;
    let midpoint = (first + second) / 2;
    assert_eq!((first, midpoint, second), (0, 4, 8));
    packed.segment_active[root] = 0;
    packed.segment_active[left] = 1;
    packed.segment_active[right] = 1;
    packed.vertex_active[midpoint] = 1;
    packed.vertex_segments[2 * first + 1] = left as u32;
    packed.vertex_segments[2 * midpoint] = left as u32;
    packed.vertex_segments[2 * midpoint + 1] = right as u32;
    packed.vertex_segments[2 * second] = right as u32;
    packed.positions[3 * midpoint + 1] += 0.5;
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        CellListConfig::default(),
        0.5,
    );
    let config = RelaxationConfig {
        adaptive_segmentation: Some(adaptive),
        penetration_tolerance: 1.0e-5,
        stretch_stiffness: 0.0,
        bend_stiffness: 0.0,
        curvature_limit_stiffness: 1.0,
        curvature_ratio_tolerance: 1.0e-5,
        constraint_iterations: 1,
        max_step: 0.5,
        max_iterations: 40,
        iterations_per_batch: 40,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 40);
    assert!(status.converged, "{status:?}");
    assert!(status.max_curvature_ratio <= 1.0 + 1.0e-5, "{status:?}");
    let positions = world.download_positions();
    assert!(positions[3 * midpoint + 1] < 2.5, "{positions:?}");
    assert_eq!(&positions[0..3], &[1.0, 2.0, 2.0]);
    assert_eq!(&positions[24..27], &[3.0, 2.0, 2.0]);
}

fn dense_crossed_mat() -> PackedAssembly {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic(
        [2.4, 2.4, 1.0],
        [true, true, false],
    ));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.1 });
    let mut id = 1;
    for index in 0..12 {
        let offset = 0.05 + 0.18 * index as f64;
        let wobble = 0.03 * (index % 3) as f64;
        let along_x = (0..=8)
            .map(|point| [0.2 + 0.25 * point as f64, offset, 0.45 + wobble])
            .collect::<Vec<_>>();
        let along_y = (0..=8)
            .map(|point| [offset + 0.07, 0.2 + 0.25 * point as f64, 0.55 - wobble])
            .collect::<Vec<_>>();
        for placed in [along_x, along_y] {
            assembly
                .add_fiber(FiberId(id), material, section, &placed, &placed)
                .unwrap();
            id += 1;
        }
    }
    PackedAssembly::from_assembly_with_options(&assembly, None, false).unwrap()
}

fn relax_dense_mat(cell_list: CellListConfig) -> (Vec<f32>, u32) {
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        dense_crossed_mat(),
        cell_list,
        0.01,
    );
    let config = RelaxationConfig {
        penetration_tolerance: 0.0,
        force_full_iterations: true,
        max_step: 0.01,
        max_iterations: 40,
        iterations_per_batch: 40,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 40);
    assert!(status.max_penetration.is_finite(), "{status:?}");
    (world.download_positions(), world.neighbor_list_rebuilds())
}

#[test]
fn neighbor_lists_match_rebuilding_every_iteration() {
    let every_iteration = CellListConfig {
        neighbor_skin_scale: 0.0,
        ..CellListConfig::default()
    };
    let (reference, reference_rebuilds) = relax_dense_mat(every_iteration);
    let (listed, listed_rebuilds) = relax_dense_mat(CellListConfig::default());
    // One slot per segment forces the overflow path for every crowded segment.
    let (overflowed, _) = relax_dense_mat(CellListConfig {
        neighbor_capacity: 1,
        ..CellListConfig::default()
    });
    // Lists built at different iterations visit the same contacts in a
    // different order, so the last float bits of each correction sum differ;
    // the contacts themselves are identical.
    let largest_difference = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max)
    };
    assert!(largest_difference(&listed, &reference) < 1.0e-4);
    assert!(largest_difference(&overflowed, &reference) < 1.0e-4);
    assert!(
        listed_rebuilds < reference_rebuilds,
        "skin did not reuse lists: {listed_rebuilds} vs {reference_rebuilds}"
    );
}

/// Distance between the axes of two segments, in f64 on the host.
fn host_segment_axis_distance(p1: [f64; 3], q1: [f64; 3], p2: [f64; 3], q2: [f64; 3]) -> f64 {
    let d1 = [q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2]];
    let d2 = [q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2]];
    let r = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (a, e, f, c, b) = (
        dot(d1, d1),
        dot(d2, d2),
        dot(d2, r),
        dot(d1, r),
        dot(d1, d2),
    );
    let denominator = a * e - b * b;
    let mut s = if denominator > 1.0e-12 * a * e {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    let delta = [
        p2[0] + d2[0] * t - p1[0] - d1[0] * s,
        p2[1] + d2[1] * t - p1[1] - d1[1] * s,
        p2[2] + d2[2] * t - p1[2] - d1[2] * s,
    ];
    dot(delta, delta).sqrt()
}

#[test]
fn neighbor_lists_hold_every_pair_within_the_skin() {
    let packed = dense_crossed_mat();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed.clone(),
        CellListConfig::default(),
        0.01,
    );
    let config = RelaxationConfig {
        max_step: 0.01,
        max_iterations: 1,
        iterations_per_batch: 1,
        ..RelaxationConfig::default()
    };
    // The first contact pass builds the lists from the uploaded positions.
    world.run_batch(&config, 1);
    let read_u32 =
        |handle: &Handle| u32::from_bytes(&world.client.read_one(handle.clone()).unwrap()).to_vec();
    let counts = read_u32(&world.neighbor_counts);
    let lists = read_u32(&world.neighbor_segments);
    let capacity = world.neighbor_capacity as usize;
    let skin = f64::from(world.neighbor_skin);

    let vertex = |index: u32| {
        let index = 3 * index as usize;
        [
            f64::from(packed.positions[index]),
            f64::from(packed.positions[index + 1]),
            f64::from(packed.positions[index + 2]),
        ]
    };
    let lengths = [2.4, 2.4, 1.0];
    let periodic = [true, true, false];
    let segments = packed.segment_count();
    let mut checked = 0;
    for first in 0..segments {
        let count = counts[first] as usize;
        assert!(count <= capacity, "segment {first} overflowed: {count}");
        let listed = &lists[first * capacity..first * capacity + count];
        let mut unique = listed.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            count,
            "segment {first} lists a neighbor twice"
        );
        let (a, b) = (
            packed.segment_vertices[2 * first],
            packed.segment_vertices[2 * first + 1],
        );
        let (p1, q1) = (vertex(a), vertex(b));
        for second in 0..segments {
            let (c, d) = (
                packed.segment_vertices[2 * second],
                packed.segment_vertices[2 * second + 1],
            );
            let adjacent = packed.segment_fibers[first] == packed.segment_fibers[second]
                && (a == c || a == d || b == c || b == d);
            if second == first || adjacent {
                assert!(!listed.contains(&(second as u32)));
                continue;
            }
            let (mut p2, mut q2) = (vertex(c), vertex(d));
            for axis in 0..3 {
                if periodic[axis] {
                    let shift = (0.5 * (p2[axis] + q2[axis] - p1[axis] - q1[axis]) / lengths[axis])
                        .round()
                        * lengths[axis];
                    p2[axis] -= shift;
                    q2[axis] -= shift;
                }
            }
            let interaction = f64::from(packed.segment_radii[first])
                + f64::from(packed.segment_radii[second])
                + skin;
            let distance = host_segment_axis_distance(p1, q1, p2, q2);
            let is_listed = listed.contains(&(second as u32));
            // Leave a thin band around the cutoff for f32 rounding.
            if distance < 0.999 * interaction {
                assert!(is_listed, "{first}-{second} at {distance} missing");
                checked += 1;
            } else if distance > 1.001 * interaction {
                assert!(!is_listed, "{first}-{second} at {distance} listed");
            }
        }
    }
    assert!(checked > segments, "fixture has too few neighbor pairs");
}

#[test]
fn relaxation_is_bitwise_repeatable() {
    // Same input, code and backend give identical positions: the cell list
    // is ranked into segment order after its atomic scatter, so every
    // correction sum adds the same terms in the same order.
    for cell_list in [
        CellListConfig::default(),
        // One slot per segment sends crowded segments through the cell scan.
        CellListConfig {
            neighbor_capacity: 1,
            ..CellListConfig::default()
        },
    ] {
        let (first, first_rebuilds) = relax_dense_mat(cell_list);
        let (second, second_rebuilds) = relax_dense_mat(cell_list);
        assert_eq!(first_rebuilds, second_rebuilds);
        let first_bits: Vec<u32> = first.iter().map(|value| value.to_bits()).collect();
        let second_bits: Vec<u32> = second.iter().map(|value| value.to_bits()).collect();
        assert!(
            first_bits == second_bits,
            "{cell_list:?} relaxed differently"
        );
    }
}

/// Twenty 1.6-long single-segment fibers crossing in two layers 0.03 apart,
/// packed adaptively: the unrefined roots are far longer than the finest
/// leaves, so the neighbor grid bins each root as several proxy pieces.
fn long_crossing_fibers() -> (PackedAssembly, f32) {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic(
        [2.0, 2.0, 1.0],
        [true, true, false],
    ));
    let material = assembly.materials.add("fiber");
    let section = assembly.sections.add(Section::Circular { radius: 0.02 });
    let mut id = 1;
    for index in 0..10 {
        let offset = 0.1 + 0.19 * index as f64;
        for placed in [
            [[0.2, offset, 0.45], [1.8, offset + 0.05, 0.45]],
            [[offset + 0.07, 0.2, 0.48], [offset + 0.02, 1.8, 0.48]],
        ] {
            assembly
                .add_fiber(FiberId(id), material, section, &placed, &placed)
                .unwrap();
            id += 1;
        }
    }
    let adaptive = AdaptiveSegmentationConfig {
        contact_length_over_diameter: 100.0,
        minimum_length_over_diameter: 1.0,
        maximum_refinement_levels: 4,
        ..AdaptiveSegmentationConfig::default()
    };
    let packed =
        PackedAssembly::from_assembly_with_options(&assembly, Some(adaptive), false).unwrap();
    (packed, 0.01)
}

#[test]
fn long_segments_are_binned_as_proxy_pieces() {
    let (packed, max_step) = long_crossing_fibers();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed.clone(),
        CellListConfig::default(),
        max_step,
    );
    // The grid follows the 0.1-long leaves, not the 1.6-long roots.
    assert!(world.cell_size() < 0.5, "cell size {}", world.cell_size());
    assert!(world.segment_cell_size > 1.6);
    let proxies = u32::from_bytes(
        &world
            .client
            .read_one(world.segment_proxies.clone())
            .unwrap(),
    )
    .to_vec();
    let active: Vec<usize> = (0..packed.segment_count())
        .filter(|&segment| packed.segment_active[segment] != 0)
        .collect();
    assert_eq!(active.len(), 20);
    assert!(active.iter().all(|&segment| proxies[segment] >= 5));

    let config = RelaxationConfig {
        max_step,
        max_iterations: 1,
        iterations_per_batch: 1,
        ..RelaxationConfig::default()
    };
    // The first contact pass builds the lists from the uploaded positions.
    world.run_batch(&config, 1);
    let read_u32 =
        |handle: &Handle| u32::from_bytes(&world.client.read_one(handle.clone()).unwrap()).to_vec();
    let counts = read_u32(&world.neighbor_counts);
    let lists = read_u32(&world.neighbor_segments);
    let capacity = world.neighbor_capacity as usize;
    let skin = f64::from(world.neighbor_skin);
    let vertex = |index: u32| {
        let index = 3 * index as usize;
        [
            f64::from(packed.positions[index]),
            f64::from(packed.positions[index + 1]),
            f64::from(packed.positions[index + 2]),
        ]
    };
    let lengths = [2.0, 2.0, 1.0];
    let periodic = [true, true, false];
    let mut listed_pairs = 0;
    for &first in &active {
        let count = counts[first] as usize;
        assert!(count <= capacity, "segment {first} overflowed: {count}");
        let listed = &lists[first * capacity..first * capacity + count];
        let mut unique = listed.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            count,
            "segment {first} lists a neighbor twice"
        );
        let (p1, q1) = (
            vertex(packed.segment_vertices[2 * first]),
            vertex(packed.segment_vertices[2 * first + 1]),
        );
        for &second in &active {
            if second == first {
                continue;
            }
            let (mut p2, mut q2) = (
                vertex(packed.segment_vertices[2 * second]),
                vertex(packed.segment_vertices[2 * second + 1]),
            );
            for axis in 0..3 {
                if periodic[axis] {
                    let shift = (0.5 * (p2[axis] + q2[axis] - p1[axis] - q1[axis]) / lengths[axis])
                        .round()
                        * lengths[axis];
                    p2[axis] -= shift;
                    q2[axis] -= shift;
                }
            }
            let interaction = 0.04 + skin;
            let distance = host_segment_axis_distance(p1, q1, p2, q2);
            let is_listed = listed.contains(&(second as u32));
            if distance < 0.999 * interaction {
                assert!(is_listed, "{first}-{second} at {distance} missing");
                listed_pairs += 1;
            } else if distance > 1.001 * interaction {
                assert!(!is_listed, "{first}-{second} at {distance} listed");
            }
        }
    }
    // Most of the 100 x/y fiber pairs cross (each crossing counts from both
    // sides); the fibers' ends leave a few apart.
    assert!(listed_pairs > 150, "{listed_pairs} neighbor pairs");
}

fn relax_long_crossing_fibers(cell_list: CellListConfig) -> Vec<f32> {
    let (packed, max_step) = long_crossing_fibers();
    let mut world = DeviceFiberWorld::<WgpuRuntime>::upload(
        &WgpuDevice::default(),
        packed,
        cell_list,
        max_step,
    );
    let config = RelaxationConfig {
        penetration_tolerance: 0.0,
        force_full_iterations: true,
        max_step,
        max_iterations: 30,
        iterations_per_batch: 30,
        ..RelaxationConfig::default()
    };
    let status = world.run_batch(&config, 30);
    assert!(status.max_penetration.is_finite(), "{status:?}");
    world.download_positions()
}

#[test]
fn proxy_overflow_scan_matches_neighbor_lists() {
    let reference = relax_long_crossing_fibers(CellListConfig {
        neighbor_skin_scale: 0.0,
        ..CellListConfig::default()
    });
    let listed = relax_long_crossing_fibers(CellListConfig::default());
    // One slot per segment sends every crossing through the per-proxy cell
    // scan, which must find each contact exactly once.
    let overflowed = relax_long_crossing_fibers(CellListConfig {
        neighbor_capacity: 1,
        ..CellListConfig::default()
    });
    let largest_difference = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max)
    };
    // The crossings overlap (0.03 apart, radii 0.02), so relaxation moves them.
    let (packed, _) = long_crossing_fibers();
    assert!(largest_difference(&reference, &packed.positions) > 1.0e-3);
    assert!(largest_difference(&listed, &reference) < 1.0e-4);
    assert!(largest_difference(&overflowed, &reference) < 1.0e-4);
}

#[test]
fn proxy_relaxation_is_bitwise_repeatable() {
    // Several proxy pieces append to one neighbor list concurrently; the
    // lists are sorted after the build, so repeated runs stay identical.
    for cell_list in [
        CellListConfig::default(),
        CellListConfig {
            neighbor_capacity: 1,
            ..CellListConfig::default()
        },
    ] {
        let first = relax_long_crossing_fibers(cell_list);
        let second = relax_long_crossing_fibers(cell_list);
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert!(
            bits(&first) == bits(&second),
            "{cell_list:?} relaxed differently"
        );
    }
}
