//! Closed-loop compaction operation controller.

use super::*;

pub(super) fn advance_compaction(
    config: &CompactionConfig,
    relaxation_config: &RelaxationConfig,
    assembly: &mut FiberAssembly,
    device: &mut DeviceState,
    relaxation: &mut RelaxationState,
    workflow: &mut WorkflowControl,
    state: &mut FormationRecipeState,
) -> CompactionProgress {
    let operation = state.next_operation;
    let world = device
        .world
        .as_mut()
        .expect("compaction requires an uploaded CubeCL device world");
    let is_new = state.active_compaction.is_none();
    let mut active = state.active_compaction.take().unwrap_or_else(|| {
        let metrics =
            world.compaction_metrics(relaxation_config.correction_fraction, config.energy_model);
        relaxation.downloaded_bytes += 13 * std::mem::size_of::<f32>();
        ActiveCompaction {
            operation,
            solid_volume: nominal_fiber_volume(assembly, state.active_formation_step),
            next_log_strain: config.increment.initial_log_strain,
            pending_started_at: Some(relaxation.iterations),
            relaxation_windows: 0,
            steps: 0,
            last_log_strain: 0.0,
            previous_lengths: cell_lengths(assembly),
            cumulative_work: [0.0; 3],
            metrics,
            baseline_pending: true,
            accepted_device: None,
            accepted_relaxation: None,
        }
    });
    assert_eq!(active.operation, operation);

    // Prescribed formation intervals deliberately disable early convergence.
    // Establish an unconstrained admissible baseline before applying the first
    // cell deformation or evaluating post-step safety guards.
    if is_new {
        relaxation.converged = false;
        hold_for_relaxation(workflow, config.increment.relax_iterations);
        state.active_compaction = Some(active);
        return CompactionProgress::Pending;
    }

    loop {
        if let Some(started) = active.pending_started_at {
            let elapsed = relaxation.iterations.saturating_sub(started);
            if !relaxation.converged && elapsed < config.increment.relax_iterations {
                hold_for_relaxation(
                    workflow,
                    config.increment.relax_iterations.saturating_sub(elapsed),
                );
                state.active_compaction = Some(active);
                return CompactionProgress::Pending;
            }
            if !relaxation.converged {
                active.relaxation_windows += 1;
                if active.baseline_pending
                    && active.relaxation_windows < config.guards.maximum_relax_windows
                {
                    active.pending_started_at = Some(relaxation.iterations);
                    relaxation.converged = false;
                    hold_for_relaxation(workflow, config.increment.relax_iterations);
                    state.active_compaction = Some(active);
                    return CompactionProgress::Pending;
                }
                if active.baseline_pending {
                    active.metrics = world.compaction_metrics(
                        relaxation_config.correction_fraction,
                        config.energy_model,
                    );
                    relaxation.downloaded_bytes += 13 * std::mem::size_of::<f32>();
                    let metrics = active.metrics;
                    return finish_compaction(
                        state,
                        assembly,
                        active,
                        CompactionStopReason::GuardReached(format!(
                            "initial formation state did not relax (penalty energy {:.3e})",
                            metrics.total_energy
                        )),
                    );
                }

                rollback_compaction_trial(relaxation_config, world, assembly, relaxation, &active);
                let attempted = active.next_log_strain;
                let reduced = (attempted * config.increment.shrink_factor)
                    .max(config.increment.minimum_log_strain);
                let at_minimum =
                    attempted <= config.increment.minimum_log_strain * (1.0 + 16.0 * f32::EPSILON);
                if at_minimum || active.relaxation_windows >= config.guards.maximum_relax_windows {
                    record_debug_snapshot(relaxation_config, world, assembly, relaxation);
                    return finish_compaction(
                        state,
                        assembly,
                        active,
                        CompactionStopReason::Jammed,
                    );
                }
                active.next_log_strain = reduced;
                active.pending_started_at = None;
                continue;
            }

            active.metrics = world
                .compaction_metrics(relaxation_config.correction_fraction, config.energy_model);
            let face_pressures = world.wall_face_pressures();
            relaxation.downloaded_bytes += 13 * std::mem::size_of::<f32>();
            record_debug_snapshot(relaxation_config, world, assembly, relaxation);
            if active.baseline_pending {
                active.baseline_pending = false;
                active.pending_started_at = None;
                active.relaxation_windows = 0;
                active.accepted_device = Some(world.checkpoint());
                active.accepted_relaxation = Some(AcceptedRelaxation::capture(relaxation));
                continue;
            }
            let metrics = active.metrics;
            accept_compaction_step(
                state,
                assembly,
                relaxation,
                &mut active,
                metrics,
                face_pressures,
            );
            active.accepted_device = Some(world.checkpoint());
            active.accepted_relaxation = Some(AcceptedRelaxation::capture(relaxation));
            active.pending_started_at = None;
            active.relaxation_windows = 0;
            active.next_log_strain = if elapsed <= config.increment.relax_iterations / 2 {
                (active.next_log_strain * config.increment.growth_factor)
                    .min(config.increment.maximum_log_strain)
            } else {
                (active.next_log_strain * config.increment.shrink_factor)
                    .max(config.increment.minimum_log_strain)
            };
        }

        let lengths = cell_lengths(assembly);
        let volume_fraction = active.solid_volume / cell_volume(lengths);
        if let Some(reason) = guard_reached(config, relaxation, active.metrics, active.steps) {
            return finish_compaction(
                state,
                assembly,
                active,
                CompactionStopReason::GuardReached(reason),
            );
        }
        if target_reached(
            config.target,
            config.target_tolerance,
            lengths,
            volume_fraction,
            active.metrics,
        ) {
            return finish_compaction(state, assembly, active, CompactionStopReason::TargetReached);
        }

        let (weights, remaining) = strain_direction_and_remaining(
            config.target,
            config.path,
            lengths,
            volume_fraction,
            active.metrics,
        );
        let mut requested = remaining
            .map_or(active.next_log_strain, |value| {
                active.next_log_strain.min(value)
            })
            .max(f32::EPSILON);
        let minimum_diameter = assembly
            .topology
            .fibers
            .iter()
            .filter(|fiber| {
                state
                    .active_formation_step
                    .is_none_or(|step| fiber.formation_step <= step)
            })
            .map(
                |fiber| match assembly.sections.entries[fiber.section.0 as usize] {
                    tangle_core::Section::Circular { radius } => 2.0 * radius,
                    tangle_core::Section::Elliptical { semi_axes } => {
                        2.0 * semi_axes[0].min(semi_axes[1])
                    }
                },
            )
            .fold(f64::INFINITY, f64::min);
        assert!(minimum_diameter.is_finite() && minimum_diameter > 0.0);
        requested = diameter_limited_log_strain(
            requested,
            weights,
            lengths,
            minimum_diameter,
            config.increment.maximum_shortening_over_minimum_diameter,
        );
        let maximum_radius = assembly
            .topology
            .fibers
            .iter()
            .filter(|fiber| {
                state
                    .active_formation_step
                    .is_none_or(|step| fiber.formation_step <= step)
            })
            .map(
                |fiber| match assembly.sections.entries[fiber.section.0 as usize] {
                    tangle_core::Section::Circular { radius } => radius,
                    tangle_core::Section::Elliptical { semi_axes } => {
                        semi_axes[0].max(semi_axes[1])
                    }
                },
            )
            .fold(0.0_f64, f64::max);
        let minimum_length = 2.001 * maximum_radius;
        let mut new_lengths = lengths;
        for axis in 0..3 {
            new_lengths[axis] =
                (lengths[axis] * (-(requested * weights[axis]) as f64).exp()).max(minimum_length);
        }
        let applied = (0..3)
            .map(|axis| (lengths[axis] / new_lengths[axis]).ln())
            .sum::<f64>() as f32;
        if applied <= f32::EPSILON {
            return finish_compaction(
                state,
                assembly,
                active,
                CompactionStopReason::GuardReached(
                    "cell reached the minimum fiber-diameter extent".to_string(),
                ),
            );
        }
        let (old_lower, _) = world.cell_bounds();
        let mut cell_anchor = config.cell_anchor;
        if config.balance_opposing_faces {
            active.metrics = world
                .compaction_metrics(relaxation_config.correction_fraction, config.energy_model);
            let face_pressures = world.wall_face_pressures();
            relaxation.downloaded_bytes += 13 * std::mem::size_of::<f32>();
            for axis in 0..3 {
                if weights[axis] > 0.0 && !assembly.cell.periodic[axis] {
                    let lower_ease = 1.0 / (face_pressures[axis][0] + config.face_pressure_floor);
                    let upper_ease = 1.0 / (face_pressures[axis][1] + config.face_pressure_floor);
                    let adaptive_anchor = lower_ease / (lower_ease + upper_ease);
                    cell_anchor[axis] = config.cell_anchor[axis]
                        + config.face_balance_strength
                            * (adaptive_anchor - config.cell_anchor[axis]);
                }
            }
        }
        let new_lower: [f32; 3] = std::array::from_fn(|axis| {
            let shortening = lengths[axis] as f32 - new_lengths[axis] as f32;
            old_lower[axis] + cell_anchor[axis] * shortening
        });
        let new_upper: [f32; 3] =
            std::array::from_fn(|axis| new_lower[axis] + new_lengths[axis] as f32);
        world.compact_cell(new_lower, new_upper, config.kinematics);
        update_host_cell(assembly, new_lower, new_upper);
        active.previous_lengths = lengths;
        active.last_log_strain = applied;
        active.pending_started_at = Some(relaxation.iterations);
        relaxation.converged = false;
        hold_for_relaxation(workflow, config.increment.relax_iterations);
        state.active_compaction = Some(active);
        return CompactionProgress::Pending;
    }
}

pub(super) fn accept_compaction_step(
    state: &mut FormationRecipeState,
    assembly: &FiberAssembly,
    relaxation: &RelaxationState,
    active: &mut ActiveCompaction,
    metrics: CompactionMetrics,
    face_pressures: [[f32; 2]; 3],
) {
    let lengths = cell_lengths(assembly);
    let accepted = active
        .accepted_device
        .as_ref()
        .expect("accepted compaction step has no previous device state");
    let current_lower = assembly.cell.origin.map(|value| value as f32);
    let current_upper: [f32; 3] = std::array::from_fn(|axis| {
        (assembly.cell.origin[axis] + assembly.cell.basis[axis][axis]) as f32
    });
    let mut face_motion = [[0.0_f32; 2]; 3];
    let mut face_work = [[0.0_f32; 2]; 3];
    for axis in 0..3 {
        let face_area = lengths[(axis + 1) % 3] * lengths[(axis + 2) % 3];
        let shortening = (active.previous_lengths[axis] - lengths[axis]).max(0.0);
        active.cumulative_work[axis] += metrics.pressure[axis] * (face_area * shortening) as f32;
        let lower_motion = (current_lower[axis] - accepted.packed.cell_lower[axis]).max(0.0);
        let upper_motion = (accepted.packed.cell_upper[axis] - current_upper[axis]).max(0.0);
        face_motion[axis] = [lower_motion, upper_motion];
        face_work[axis][0] = face_pressures[axis][0] * face_area as f32 * lower_motion;
        face_work[axis][1] = face_pressures[axis][1] * face_area as f32 * upper_motion;
    }
    active.steps += 1;
    active.metrics = metrics;
    state.compaction_steps.push(CompactionStepReport {
        operation: active.operation,
        step: active.steps,
        iteration: relaxation.iterations,
        cell_lengths: lengths,
        nominal_volume_fraction: active.solid_volume / cell_volume(lengths),
        log_volume_strain: active.last_log_strain,
        max_penetration: relaxation.max_penetration,
        max_bend_ratio: relaxation.max_curvature_ratio,
        metrics,
        cumulative_work: active.cumulative_work,
    });
    let mut readings = Vec::new();
    for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
        if face_pressures[axis].iter().any(|value| *value > 0.0)
            || face_work[axis].iter().any(|value| *value > 0.0)
        {
            readings.push(format!(
                "{name}- p={:.3e} dx={:.3e} dW={:.3e}, {name}+ p={:.3e} dx={:.3e} dW={:.3e}",
                face_pressures[axis][0],
                face_motion[axis][0],
                face_work[axis][0],
                face_pressures[axis][1],
                face_motion[axis][1],
                face_work[axis][1],
            ));
        }
    }
    if !readings.is_empty() {
        state.events.push(FormationEvent {
            operation: active.operation,
            iteration: relaxation.iterations,
            description: format!(
                "compaction step {} wall pressure/motion/work: {}",
                active.steps,
                readings.join("; ")
            ),
        });
    }
}

pub(super) fn rollback_compaction_trial(
    config: &RelaxationConfig,
    world: &mut DeviceWorld,
    assembly: &mut FiberAssembly,
    relaxation: &mut RelaxationState,
    active: &ActiveCompaction,
) {
    let mut checkpoint = active
        .accepted_device
        .clone()
        .expect("compaction trial started without an accepted rollback state");
    let accepted = active
        .accepted_relaxation
        .expect("compaction trial started without accepted relaxation metrics");
    let current_iteration = relaxation.iterations;
    checkpoint.total_iterations = current_iteration;
    let lower = checkpoint.packed.cell_lower;
    let upper = checkpoint.packed.cell_upper;
    *world = DeviceWorld::restore(config, checkpoint);
    update_host_cell(assembly, lower, upper);
    accepted.restore(relaxation);
    relaxation.iterations = current_iteration;
    relaxation.cell_size = world.cell_size();
    relaxation.cell_count = world.cell_count();
}

pub(super) fn record_debug_snapshot(
    config: &RelaxationConfig,
    world: &DeviceWorld,
    assembly: &FiberAssembly,
    relaxation: &mut RelaxationState,
) {
    if config.debug_snapshot_interval.is_none() {
        return;
    }
    let positions = world.download_positions();
    relaxation.downloaded_bytes += positions.len() * std::mem::size_of::<f32>();
    let mut snapshot_assembly = assembly.clone();
    world
        .unpack_positions(&positions, &mut snapshot_assembly)
        .expect("CubeCL compaction snapshot topology was invalid");
    relaxation.snapshots.push(RelaxationSnapshot {
        iteration: relaxation.iterations,
        positions,
        assembly: Some(snapshot_assembly),
    });
}

pub(super) fn finish_compaction(
    state: &mut FormationRecipeState,
    assembly: &FiberAssembly,
    active: ActiveCompaction,
    reason: CompactionStopReason,
) -> CompactionProgress {
    let lengths = cell_lengths(assembly);
    let volume_fraction = active.solid_volume / cell_volume(lengths);
    let description = format!(
        "compact in {} steps to nominal volume fraction {:.4} ({reason:?})",
        active.steps, volume_fraction
    );
    state.compactions.push(CompactionReport {
        operation: active.operation,
        reason,
        steps: active.steps,
        cell_lengths: lengths,
        nominal_volume_fraction: volume_fraction,
        metrics: active.metrics,
        cumulative_work: active.cumulative_work,
    });
    CompactionProgress::Finished(description)
}

pub(super) fn hold_for_relaxation(workflow: &mut WorkflowControl, remaining: usize) {
    workflow.hold_relax_stage = true;
    workflow.force_full_batch = false;
    workflow.batch_iteration_limit = Some(remaining.max(1));
}

pub(super) fn cell_lengths(assembly: &FiberAssembly) -> [f64; 3] {
    [
        assembly.cell.basis[0][0],
        assembly.cell.basis[1][1],
        assembly.cell.basis[2][2],
    ]
}

pub(super) fn cell_volume(lengths: [f64; 3]) -> f64 {
    lengths[0] * lengths[1] * lengths[2]
}

pub(super) fn active_capsule_bounds(
    packed: &tangle_relax::PackedAssembly,
    positions: &[f32],
    segment_active: &[u32],
    padding: f32,
) -> ([f32; 3], [f32; 3]) {
    assert_eq!(positions.len(), 3 * packed.vertex_count());
    assert_eq!(segment_active.len(), packed.segment_count());
    let mut lower = [f32::INFINITY; 3];
    let mut upper = [f32::NEG_INFINITY; 3];
    let mut found = false;
    for (segment, active) in segment_active.iter().enumerate() {
        if *active == 0 {
            continue;
        }
        found = true;
        let radius = packed.segment_radii[segment] + padding;
        for endpoint in 0..2 {
            let vertex = packed.segment_vertices[2 * segment + endpoint] as usize;
            for axis in 0..3 {
                let coordinate = positions[3 * vertex + axis];
                lower[axis] = lower[axis].min(coordinate - radius);
                upper[axis] = upper[axis].max(coordinate + radius);
            }
        }
    }
    assert!(found, "cannot fit a cell without active fiber segments");
    (lower, upper)
}

pub(super) fn update_host_cell(assembly: &mut FiberAssembly, lower: [f32; 3], upper: [f32; 3]) {
    assembly.cell.origin = lower.map(f64::from);
    assembly.cell.basis = [
        [(upper[0] - lower[0]) as f64, 0.0, 0.0],
        [0.0, (upper[1] - lower[1]) as f64, 0.0],
        [0.0, 0.0, (upper[2] - lower[2]) as f64],
    ];
}

pub(super) fn target_reached(
    target: CompactionTarget,
    tolerance: f32,
    lengths: [f64; 3],
    volume_fraction: f64,
    metrics: CompactionMetrics,
) -> bool {
    let tolerance = tolerance as f64;
    match target {
        CompactionTarget::NominalVolumeFraction(value) => {
            volume_fraction >= value * (1.0 - tolerance)
        }
        CompactionTarget::CellVolume(value) => cell_volume(lengths) <= value * (1.0 + tolerance),
        CompactionTarget::CellLengths(targets) => {
            (0..3).all(|axis| lengths[axis] <= targets[axis] * (1.0 + tolerance))
        }
        CompactionTarget::MeanPressure(value) => {
            metrics.pressure.iter().sum::<f32>() / 3.0 >= value * (1.0 - tolerance as f32)
        }
        CompactionTarget::DirectionalPressure(targets) => (0..3).all(|axis| {
            targets[axis] == 0.0
                || metrics.pressure[axis] >= targets[axis] * (1.0 - tolerance as f32)
        }),
        CompactionTarget::PenaltyEnergy(value) => {
            metrics.total_energy >= value * (1.0 - tolerance as f32)
        }
    }
}

pub(super) fn strain_direction_and_remaining(
    target: CompactionTarget,
    path: CompactionPath,
    lengths: [f64; 3],
    volume_fraction: f64,
    metrics: CompactionMetrics,
) -> ([f32; 3], Option<f32>) {
    match target {
        CompactionTarget::CellLengths(targets) => {
            let remaining: [f32; 3] =
                std::array::from_fn(|axis| (lengths[axis] / targets[axis]).ln().max(0.0) as f32);
            let total = remaining.iter().sum::<f32>();
            (remaining.map(|value| value / total), Some(total))
        }
        CompactionTarget::NominalVolumeFraction(value) => (
            axis_weights(path, metrics),
            Some((value / volume_fraction).ln().max(0.0) as f32),
        ),
        CompactionTarget::CellVolume(value) => (
            axis_weights(path, metrics),
            Some((cell_volume(lengths) / value).ln().max(0.0) as f32),
        ),
        CompactionTarget::MeanPressure(_)
        | CompactionTarget::DirectionalPressure(_)
        | CompactionTarget::PenaltyEnergy(_) => (axis_weights(path, metrics), None),
    }
}

pub(super) fn guard_reached(
    config: &CompactionConfig,
    relaxation: &RelaxationState,
    metrics: CompactionMetrics,
    steps: usize,
) -> Option<String> {
    if steps >= config.guards.maximum_steps {
        return Some("maximum compaction steps".to_string());
    }
    if relaxation.max_penetration > config.guards.maximum_penetration {
        return Some("maximum penetration".to_string());
    }
    if relaxation.max_curvature_ratio > config.guards.maximum_bend_ratio {
        return Some("maximum bend ratio".to_string());
    }
    if metrics
        .pressure
        .iter()
        .any(|pressure| *pressure > config.guards.maximum_pressure)
    {
        return Some("maximum pressure".to_string());
    }
    if metrics.total_energy > config.guards.maximum_penalty_energy {
        return Some("maximum penalty energy".to_string());
    }
    None
}
