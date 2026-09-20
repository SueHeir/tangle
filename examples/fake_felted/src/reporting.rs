//! Console progress and final diagnostics for the fake felt.

use super::*;

pub(super) fn print_progress(
    config: Res<FormationRecipeConfig>,
    recipe: Res<FormationRecipeState>,
    relaxation: Res<RelaxationState>,
    checkpoint: Res<CheckpointReport>,
    mut progress: ResMut<Progress>,
) {
    if progress.next_iteration == 0 {
        println!(
            "{} {} formation operations; progress every {} iterations",
            if checkpoint.resumed {
                "resuming"
            } else {
                "starting"
            },
            config.operations.len(),
            PROGRESS_INTERVAL
        );
        progress.next_iteration = PROGRESS_INTERVAL;
        if checkpoint.resumed {
            progress.events = recipe.events.len();
        }
    }
    for event in &recipe.events[progress.events..] {
        println!(
            "  operation {:>3}/{} at iteration {:>7}: {}",
            event.operation + 1,
            config.operations.len(),
            event.iteration,
            event.description
        );
    }
    progress.events = recipe.events.len();
    if relaxation.iterations >= progress.next_iteration {
        println!(
            "  relax {:>7}: penetration {:.3} um, bend {:.5}, active segments {}",
            relaxation.iterations,
            1.0e6 * relaxation.max_penetration,
            relaxation.max_curvature_ratio,
            relaxation.active_segments
        );
        progress.next_iteration =
            (relaxation.iterations / PROGRESS_INTERVAL + 1) * PROGRESS_INTERVAL;
    }
    std::io::stdout().flush().expect("progress output failed");
}

pub(super) fn report(app: &App) {
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .expect("assembly resource");
    let recipe = app
        .get_resource_ref::<FormationRecipeState>()
        .expect("recipe resource");
    let relaxation = app
        .get_resource_ref::<RelaxationState>()
        .expect("relaxation resource");
    println!("fake felted result:");
    println!(
        "  {} fibers, {} active segments, final cell {:.1} x {:.1} x {:.1} um",
        assembly.topology.fibers.len(),
        relaxation.active_segments,
        1.0e6 * assembly.cell.basis[0][0],
        1.0e6 * assembly.cell.basis[1][1],
        1.0e6 * assembly.cell.basis[2][2]
    );
    if relaxation.cell_list_overflow {
        println!("  REJECTED: GPU cell-list capacity overflowed");
    }
    let junctions = recipe
        .junction_captures
        .last()
        .map_or(0, |capture| capture.created);
    println!("  persistent inter-fiber bonds: {junctions} (contact/friction only)");
    if let Some(failure) = &recipe.failure {
        println!("  REJECTED: {}", failure.reason);
    }
    drop(recipe);
    // Always audit the exported handoff state. This distinguishes a genuine
    // inter-fiber residual from a nonadjacent self-contact and makes the DEM
    // startup tolerance independently visible in successful runs.
    print_contact_diagnostic(app, relaxation.max_penetration);
    if let Some(export) = app.get_resource_ref::<DemCapsuleBpmExportReport>() {
        if !export.data_path.as_os_str().is_empty() {
            println!(
                "  exported {} capsules and {} bonds to {}",
                export.capsules,
                export.bonds,
                export.data_path.display()
            );
            for mapping in &export.atom_types {
                println!(
                    "    atom type {}: {} ({} capsules)",
                    mapping.atom_type, mapping.material_name, mapping.particles
                );
            }
            if let Some(path) = &export.dirt_config_path {
                println!("  DIRT loading config: {}", path.display());
            }
        }
    }
}

pub(super) fn print_contact_diagnostic(app: &App, reported_maximum: f32) {
    const CAPACITY: usize = 500_000;
    let cell = app
        .resource_cell(TypeId::of::<DeviceState>())
        .expect("device-state resource");
    let (capture, packed) = {
        let mut resource = cell.borrow_mut();
        let device = resource
            .downcast_mut::<DeviceState>()
            .expect("device-state resource type");
        let world = device.world.as_mut().expect("resident device world");
        let packed = world.packed().clone();
        (world.capture_contacts(0.0, CAPACITY), packed)
    };
    let Some(worst) = capture
        .candidates
        .iter()
        .min_by(|first, second| first.surface_gap.total_cmp(&second.surface_gap))
    else {
        println!(
            "  contact audit: no inter-fiber penetration captured; the {:.3} um residual is nonadjacent same-fiber contact",
            1.0e6 * reported_maximum
        );
        return;
    };
    let inter_fiber_penetration = (-worst.surface_gap).max(0.0);
    let first_owner = packed.segment_fibers[worst.first_segment as usize] as usize;
    let second_owner = packed.segment_fibers[worst.second_segment as usize] as usize;
    let first_center_z = packed_fiber_center(&packed, first_owner, 2);
    let second_center_z = packed_fiber_center(&packed, second_owner, 2);
    let first_diameter = 2.0 * packed.segment_radii[worst.first_segment as usize];
    let second_diameter = 2.0 * packed.segment_radii[worst.second_segment as usize];
    let centerline_separation = 0.5 * (first_diameter + second_diameter) - inter_fiber_penetration;
    let near_centerline_crossing =
        centerline_separation <= 0.05 * first_diameter.min(second_diameter);
    let first_contacts = capture
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.first_segment == worst.first_segment
                || candidate.second_segment == worst.first_segment
        })
        .count();
    let second_contacts = capture
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.first_segment == worst.second_segment
                || candidate.second_segment == worst.second_segment
        })
        .count();
    let assembly = app
        .get_resource_ref::<FiberAssembly>()
        .expect("assembly resource");
    let first = &assembly.topology.fibers[first_owner];
    let second = &assembly.topology.fibers[second_owner];
    println!(
        "  contact audit: worst inter-fiber penetration {:.3} um between fibers {} (layer {:?}, {:.1} um diameter) and {} (layer {:?}, {:.1} um diameter)",
        1.0e6 * inter_fiber_penetration,
        first.id.0,
        first.formation_layer,
        1.0e6 * first_diameter,
        second.id.0,
        second.formation_layer,
        1.0e6 * second_diameter,
    );
    println!(
        "  contact audit: segments {} and {} have {} and {} penetrating neighbors; crossing angle {:.2} deg, closest centerlines {:.3} um apart, fiber-center z separation {:.3} um{}{}",
        worst.first_segment,
        worst.second_segment,
        first_contacts,
        second_contacts,
        worst.crossing_angle.to_degrees(),
        1.0e6 * centerline_separation.max(0.0),
        1.0e6 * (first_center_z - second_center_z).abs(),
        if near_centerline_crossing {
            "; near-exact centerline crossing"
        } else {
            ""
        },
        if capture.overflow {
            "; capture capacity overflowed"
        } else {
            ""
        }
    );
    for segment in [worst.first_segment, worst.second_segment] {
        for candidate in capture.candidates.iter().filter(|candidate| {
            candidate.first_segment == segment || candidate.second_segment == segment
        }) {
            let other = if candidate.first_segment == segment {
                candidate.second_segment
            } else {
                candidate.first_segment
            };
            let other_owner = packed.segment_fibers[other as usize] as usize;
            let other_fiber = &assembly.topology.fibers[other_owner];
            println!(
                "    segment {segment}: {:.3} um penetration with segment {other} / fiber {} / layer {:?} at {:.2} deg",
                -1.0e6 * candidate.surface_gap,
                other_fiber.id.0,
                other_fiber.formation_layer,
                candidate.crossing_angle.to_degrees()
            );
        }
    }
    if inter_fiber_penetration + 0.05e-6 < reported_maximum {
        println!(
            "  contact audit: the {:.3} um solver maximum is larger, so the limiting residual is nonadjacent same-fiber contact",
            1.0e6 * reported_maximum
        );
    }
}

pub(super) fn packed_fiber_center(
    packed: &tangle_relax::PackedAssembly,
    fiber: usize,
    axis: usize,
) -> f32 {
    let first = packed.fiber_vertex_spans[2 * fiber] as usize;
    let count = packed.fiber_vertex_spans[2 * fiber + 1] as usize;
    let mut sum = 0.0;
    let mut active = 0;
    for vertex in first..first + count {
        if packed.vertex_active[vertex] != 0 {
            sum += packed.positions[3 * vertex + axis];
            active += 1;
        }
    }
    assert!(active > 0);
    sum / active as f32
}
