//! Exhaustive host-side collision audit for a saved fake-needled checkpoint.

use std::{collections::BTreeMap, collections::BTreeSet, path::PathBuf};

use tangle_checkpoint::load_checkpoint;
use tangle_relax::{DeviceWorld, RelaxationConfig};

const CASE_ID: &str = "fake_needled_2-v19-staged-solve-policy";

fn main() {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("output/fake_needled_2.restart")
        });
    let checkpoint = load_checkpoint(&path, CASE_ID)
        .unwrap_or_else(|error| panic!("could not audit {}: {error}", path.display()));
    let packed = &checkpoint.device.packed;
    let maximum_rest_length = packed
        .segment_rest_lengths
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let maximum_current_length = (0..packed.segment_count())
        .map(|segment| {
            let (start, end) = endpoints(packed, segment);
            segment_distance(start, start, end, end)
        })
        .fold(0.0_f32, f32::max);
    let maximum_radius = packed.segment_radii.iter().copied().fold(0.0_f32, f32::max);
    let requested_cell_size =
        maximum_rest_length.max(maximum_current_length) + 2.0 * maximum_radius + 2.0 * 2.0e-6;
    let cell_counts = [0, 1, 2].map(|axis| {
        ((packed.cell_upper[axis] - packed.cell_lower[axis]) / requested_cell_size)
            .floor()
            .max(1.0) as u32
    });
    let active = packed
        .segment_active
        .iter()
        .enumerate()
        .filter_map(|(segment, active)| (*active != 0).then_some(segment))
        .collect::<Vec<_>>();
    let mut contacts = 0_usize;
    let mut over_tolerance = 0_usize;
    let mut maximum_penetration = 0.0_f32;
    let mut worst = None;
    let mut cpu_contacts = BTreeMap::new();

    for (offset, first) in active.iter().copied().enumerate() {
        for second in active[offset + 1..].iter().copied() {
            if packed.segment_fibers[first] == packed.segment_fibers[second] {
                continue;
            }
            let (first_start, first_end) = endpoints(packed, first);
            let (mut second_start, mut second_end) = endpoints(packed, second);
            for axis in 0..3 {
                if packed.cell_periodic[axis] == 0 {
                    continue;
                }
                let width = packed.cell_upper[axis] - packed.cell_lower[axis];
                let first_midpoint = 0.5 * (first_start[axis] + first_end[axis]);
                let second_midpoint = 0.5 * (second_start[axis] + second_end[axis]);
                let shift = ((second_midpoint - first_midpoint) / width).round() * width;
                second_start[axis] -= shift;
                second_end[axis] -= shift;
            }
            let distance = segment_distance(first_start, first_end, second_start, second_end);
            let penetration = packed.segment_radii[first] + packed.segment_radii[second] - distance;
            if penetration > 0.0 {
                cpu_contacts.insert((first, second), penetration);
                contacts += 1;
                over_tolerance += usize::from(penetration > 0.30e-6);
                if penetration > maximum_penetration {
                    maximum_penetration = penetration;
                    worst = Some((first, second, distance));
                }
            }
        }
    }

    println!("checkpoint: {}", path.display());
    println!("iteration: {}", checkpoint.relaxation.iterations);
    println!("active segments: {}", active.len());
    println!(
        "broad-phase requested size: {:.3} um; grid: {} x {} x {}",
        1.0e6 * requested_cell_size,
        cell_counts[0],
        cell_counts[1],
        cell_counts[2]
    );
    println!("cross-fiber contacts: {contacts}");
    println!("contacts over 0.30 um: {over_tolerance}");
    println!("maximum penetration: {:.6} um", 1.0e6 * maximum_penetration);
    if let Some((first, second, distance)) = worst {
        println!(
            "worst pair: reserved segments {first} and {second}, fibers {} and {}, centerline distance {:.6} um",
            packed.segment_fibers[first],
            packed.segment_fibers[second],
            1.0e6 * distance
        );
    }
    println!(
        "stored solver residual: {:.6} um",
        1.0e6 * checkpoint.relaxation.max_penetration
    );

    let config = RelaxationConfig {
        max_step: 2.0e-6,
        ..RelaxationConfig::flexible()
    };
    let mut world = DeviceWorld::restore(&config, checkpoint.device.clone());
    let captured = world.capture_contacts(0.0, 50_000);
    let captured_maximum = captured
        .candidates
        .iter()
        .map(|contact| (-contact.surface_gap).max(0.0))
        .fold(0.0_f32, f32::max);
    let gpu_contacts = captured
        .candidates
        .iter()
        .map(|contact| {
            let first = contact.first_segment as usize;
            let second = contact.second_segment as usize;
            (first.min(second), first.max(second))
        })
        .collect::<BTreeSet<_>>();
    let missed = cpu_contacts
        .iter()
        .filter(|(pair, _)| !gpu_contacts.contains(pair))
        .collect::<Vec<_>>();
    println!(
        "GPU broad-phase contacts: {}{}",
        captured.candidates.len(),
        if captured.overflow {
            " (capture overflow)"
        } else {
            ""
        }
    );
    println!(
        "GPU captured maximum penetration: {:.6} um",
        1.0e6 * captured_maximum
    );
    println!("unique GPU contact pairs: {}", gpu_contacts.len());
    println!("CPU contacts missed by GPU: {}", missed.len());
    for (pair, penetration) in missed.iter().take(5) {
        let first_midpoint = midpoint(packed, pair.0);
        let second_midpoint = midpoint(packed, pair.1);
        let first_direction = direction(packed, pair.0);
        let second_direction = direction(packed, pair.1);
        let cosine = (dot(first_direction, second_direction)
            / (dot(first_direction, first_direction) * dot(second_direction, second_direction))
                .sqrt())
        .abs()
        .clamp(0.0, 1.0);
        println!(
            "  missed segments {} and {}: penetration {:.6} um, cells {:?} {:?}, lengths {:.3}/{:.3} um, angle {:.6} deg",
            pair.0,
            pair.1,
            1.0e6 * **penetration,
            cell_index(packed, first_midpoint, cell_counts),
            cell_index(packed, second_midpoint, cell_counts),
            1.0e6 * segment_length(packed, pair.0),
            1.0e6 * segment_length(packed, pair.1),
            cosine.acos().to_degrees(),
        );
    }
}

fn midpoint(packed: &tangle_relax::PackedAssembly, segment: usize) -> [f32; 3] {
    let (start, end) = endpoints(packed, segment);
    scale(add(start, end), 0.5)
}

fn segment_length(packed: &tangle_relax::PackedAssembly, segment: usize) -> f32 {
    let (start, end) = endpoints(packed, segment);
    dot(subtract(end, start), subtract(end, start)).sqrt()
}

fn direction(packed: &tangle_relax::PackedAssembly, segment: usize) -> [f32; 3] {
    let (start, end) = endpoints(packed, segment);
    subtract(end, start)
}

fn cell_index(
    packed: &tangle_relax::PackedAssembly,
    position: [f32; 3],
    counts: [u32; 3],
) -> [u32; 3] {
    [0, 1, 2].map(|axis| {
        let width = (packed.cell_upper[axis] - packed.cell_lower[axis]) / counts[axis] as f32;
        let raw = ((position[axis] - packed.cell_lower[axis]) / width).floor() as i32;
        if packed.cell_periodic[axis] != 0 {
            ((raw % counts[axis] as i32 + counts[axis] as i32) % counts[axis] as i32) as u32
        } else {
            raw.clamp(0, counts[axis] as i32 - 1) as u32
        }
    })
}

fn endpoints(packed: &tangle_relax::PackedAssembly, segment: usize) -> ([f32; 3], [f32; 3]) {
    let first = packed.segment_vertices[2 * segment] as usize;
    let second = packed.segment_vertices[2 * segment + 1] as usize;
    (
        [
            packed.positions[3 * first],
            packed.positions[3 * first + 1],
            packed.positions[3 * first + 2],
        ],
        [
            packed.positions[3 * second],
            packed.positions[3 * second + 1],
            packed.positions[3 * second + 2],
        ],
    )
}

fn segment_distance(p1: [f32; 3], q1: [f32; 3], p2: [f32; 3], q2: [f32; 3]) -> f32 {
    let d1 = subtract(q1, p1);
    let d2 = subtract(q2, p2);
    let relative = subtract(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, relative);
    let epsilon = 1.0e-20_f32;
    let (mut first_coordinate, second_coordinate) = if a <= epsilon && e <= epsilon {
        (0.0, 0.0)
    } else if a <= epsilon {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, relative);
        if e <= epsilon {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denominator = a * e - b * b;
            let mut first = if denominator.abs() > epsilon {
                ((b * f - c * e) / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let projected = (b * first + f) / e;
            let second = if projected < 0.0 {
                first = (-c / a).clamp(0.0, 1.0);
                0.0
            } else if projected > 1.0 {
                first = ((b - c) / a).clamp(0.0, 1.0);
                1.0
            } else {
                projected
            };
            (first, second)
        }
    };
    first_coordinate = first_coordinate.clamp(0.0, 1.0);
    let first_point = add(p1, scale(d1, first_coordinate));
    let second_point = add(p2, scale(d2, second_coordinate));
    dot(
        subtract(second_point, first_point),
        subtract(second_point, first_point),
    )
    .sqrt()
}

fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn scale(vector: [f32; 3], factor: f32) -> [f32; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
