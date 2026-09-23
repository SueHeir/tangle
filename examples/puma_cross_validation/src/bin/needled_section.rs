//! Offline section analysis from the final TANGLE capsule export.
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};
use tangle_characterize::{characterize_assembly, write_analysis_json};
use tangle_core::{FiberAssembly, FiberId, PeriodicCell, Section, Vec3};
use tangle_export::{write_puma_bundle, PumaVoxelExportConfig};

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source =
        root.join("examples/felt_20ply_control_vs_needled/output/needled_polish_capsules.data");
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("examples/puma_cross_validation/output/needled_section"));
    fs::create_dir_all(&output)?;
    let (assembly, maximum_join_gap) = read_capsules(&source)?;
    let lower = assembly.cell.origin;
    let lengths: Vec3 = std::array::from_fn(|i| assembly.cell.basis[i][i]);
    let upper: Vec3 = std::array::from_fn(|i| lower[i] + lengths[i]);
    assembly.validate()?;
    let full = characterize_assembly(&assembly);
    println!(
        "export: {} fibers, {} segments, maximum periodic join gap {:.6e} m",
        full.fibers, full.segments, maximum_join_gap
    );
    write_analysis_json(&full, output.join("full_specimen_analysis.json"))?;

    let width = 240e-6;
    let low: Vec3 = std::array::from_fn(|i| (lower[i] + upper[i] - width) * 0.5);
    let high: Vec3 = std::array::from_fn(|i| low[i] + width);
    let mut cell = PeriodicCell::orthorhombic([width; 3], [false; 3]);
    cell.origin = low;
    // Measurement fibers below are individual clipped segments. Their count is
    // NOT a physical fiber count, and their curvature is not used in the report.
    let mut measured = FiberAssembly::new(cell);
    measured.materials = assembly.materials.clone();
    measured.sections = assembly.sections.clone();
    // Full source segments are retained for voxelization, including centers
    // outside the ROI whose swept radii intersect it. This avoids cut-end caps.
    let mut raster = FiberAssembly::new(cell);
    raster.materials = assembly.materials.clone();
    raster.sections = assembly.sections.clone();
    let mut source_ids = BTreeSet::new();
    let mut raster_source_map = Vec::new();
    for fiber in &assembly.topology.fibers {
        let radius = match assembly.sections.entries[fiber.section.0 as usize] {
            Section::Circular { radius } => radius,
            _ => return Err("section analysis requires circular fibers".into()),
        };
        let start = fiber.vertices.start as usize;
        let end = fiber.vertices.checked_end().unwrap() as usize;
        for (segment, points) in assembly.geometry.placed.positions[start..end]
            .windows(2)
            .enumerate()
        {
            let a = points[0];
            let b = points[1];
            let ranges: [(i32, i32); 3] = std::array::from_fn(|i| {
                if !assembly.cell.periodic[i] {
                    return (0, 0);
                }
                (
                    ((low[i] - radius - 1.5e-6 - a[i].max(b[i])) / lengths[i]).ceil() as i32,
                    ((high[i] + radius + 1.5e-6 - a[i].min(b[i])) / lengths[i]).floor() as i32,
                )
            });
            for x in ranges[0].0..=ranges[0].1 {
                for y in ranges[1].0..=ranges[1].1 {
                    for z in ranges[2].0..=ranges[2].1 {
                        let image = [x, y, z];
                        let aa = std::array::from_fn(|i| a[i] + image[i] as f64 * lengths[i]);
                        let bb = std::array::from_fn(|i| b[i] + image[i] as f64 * lengths[i]);
                        if clip(
                            aa,
                            bb,
                            low.map(|v| v - radius - 1.5e-6),
                            high.map(|v| v + radius + 1.5e-6),
                        )
                        .is_none()
                        {
                            continue;
                        }
                        let id = FiberId(raster.topology.fibers.len() as u32 + 1);
                        raster.add_fiber(
                            id,
                            fiber.material,
                            fiber.section,
                            &[aa, bb],
                            &[aa, bb],
                        )?;
                        raster_source_map.push(serde_json::json!({"export_fiber_id": id.0, "source_fiber_id": fiber.id.0, "source_segment": segment, "periodic_image": image}));
                        if let Some((t0, t1)) = clip(aa, bb, low, high) {
                            let p = lerp(aa, bb, t0);
                            let q = lerp(aa, bb, t1);
                            if distance(p, q) <= 0.0 {
                                continue;
                            }
                            let id = FiberId(measured.topology.fibers.len() as u32 + 1);
                            let intrinsic = &assembly.geometry.intrinsic.positions
                                [start + segment..start + segment + 2];
                            measured.add_fiber(
                                id,
                                fiber.material,
                                fiber.section,
                                &[
                                    lerp(intrinsic[0], intrinsic[1], t0),
                                    lerp(intrinsic[0], intrinsic[1], t1),
                                ],
                                &[p, q],
                            )?;
                            source_ids.insert(fiber.id.0);
                        }
                    }
                }
            }
        }
    }
    let metrics = characterize_assembly(&measured);
    write_analysis_json(&metrics, output.join("section_centerline_analysis.json"))?;
    let lateral_area: f64 = metrics
        .fiber_metrics
        .iter()
        .map(|f| std::f64::consts::PI * f.equivalent_diameter * f.placed_length)
        .sum();
    let metadata = serde_json::json!({
        "source_path": source, "source_format": "TANGLE adaptive-capsule DEM-BPM export",
        "maximum_periodic_endpoint_join_error_m": maximum_join_gap,
        "rest_geometry_available": false, "solver_state_available": false,
        "source_cell_low_m": lower, "source_cell_high_m": upper,
        "crop_low_m": low, "crop_high_m": high, "crop_width_m": width,
        "physical_fibers_with_centerline_in_crop": source_ids.len(),
        "source_fiber_ids": source_ids, "clipped_segment_count": metrics.segments,
        "nominal_lateral_surface_area_m2": lateral_area,
        "nominal_specific_lateral_area_per_m": lateral_area / width.powi(3),
        "notes": ["Crop is bounded, not periodic; source periodic images are included before cropping.",
        "Legacy checkpoint is not decodable by current structs; final capsule export is the geometry source. Rest/strain and admissibility metrics are unavailable.",
        "Measurement assembly contains clipped segments; its fiber count and curvature are not physical fiber statistics.",
        "Bundle tangle_analysis.json describes halo raster segments; use section_centerline_analysis.json for crop-native metrics.",
        "Voxel owner IDs refer to segments; raster_source_map.json maps them to physical fibers. Ambiguity counts include shared vertices."]
    });
    fs::write(
        output.join("section_metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    fs::write(
        output.join("raster_source_map.json"),
        serde_json::to_vec_pretty(&raster_source_map)?,
    )?;
    println!(
        "crop: {} physical fibers, {} clipped segments, nominal Vf {:.6}, A_V {:?}",
        source_ids.len(),
        metrics.segments,
        metrics.nominal_swept_volume_fraction,
        metrics.volume_weighted_orientation_tensor
    );
    for h_um in [3, 2, 1] {
        let report = write_puma_bundle(
            &raster,
            &PumaVoxelExportConfig::new(output.join(format!("h_{h_um}um")), h_um as f64 * 1e-6),
        )?;
        println!(
            "h = {h_um} um: voxel Vf {:.6}, {} occupied voxels",
            report.voxel_volume_fraction, report.occupied_voxels
        );
    }
    Ok(())
}

fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}
fn distance(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|i| (b[i] - a[i]).powi(2)).sum::<f64>().sqrt()
}
fn clip(a: Vec3, b: Vec3, low: Vec3, high: Vec3) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for i in 0..3 {
        let d = b[i] - a[i];
        if d == 0.0 {
            if a[i] < low[i] || a[i] > high[i] {
                return None;
            }
        } else {
            let u = (low[i] - a[i]) / d;
            let v = (high[i] - a[i]) / d;
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
            if t1 <= t0 {
                return None;
            }
        }
    }
    Some((t0, t1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_preserves_length_under_endpoint_reversal() {
        let a = [-2.0, 0.5, 0.5];
        let b = [2.0, 0.5, 0.5];
        for (a, b) in [(a, b), (b, a)] {
            let (t0, t1) = clip(a, b, [0.0; 3], [1.0; 3]).unwrap();
            assert!((distance(lerp(a, b, t0), lerp(a, b, t1)) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn crop_rejects_disjoint_and_point_only_intersections() {
        assert!(clip([-1.0, 2.0, 0.5], [2.0, 2.0, 0.5], [0.0; 3], [1.0; 3]).is_none());
        assert!(clip([-1.0; 3], [0.0; 3], [0.0; 3], [1.0; 3]).is_none());
        assert_eq!(
            clip([0.25; 3], [0.75; 3], [0.0; 3], [1.0; 3]),
            Some((0.0, 1.0))
        );
    }
}

// Rebuild consecutive centerlines while checking every exported bond and
// endpoint. Only the x/y boundaries are periodic in this specimen's TOML.
fn read_capsules(path: &Path) -> Result<(FiberAssembly, f64), Box<dyn Error>> {
    let input = fs::read_to_string(path)?;
    let mut low = [0.0; 3];
    let mut high = [0.0; 3];
    let mut atoms = BTreeMap::<u32, (u32, usize, f64, Vec3)>::new();
    let mut shapes = BTreeMap::<u32, (f64, Vec3)>::new();
    let mut bonds = BTreeSet::new();
    let mut mode = "";
    for line in input.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        if s.starts_with("Atoms") {
            mode = "atoms";
            continue;
        }
        if s == "Capsules" {
            mode = "capsules";
            continue;
        }
        if s == "Bonds" {
            mode = "bonds";
            continue;
        }
        let v: Vec<&str> = s.split_whitespace().collect();
        if mode.is_empty() {
            for (i, label) in ["xlo", "ylo", "zlo"].iter().enumerate() {
                if v.len() == 4 && v[2] == *label {
                    low[i] = v[0].parse()?;
                    high[i] = v[1].parse()?;
                }
            }
        } else if mode == "atoms" {
            atoms.insert(
                v[0].parse()?,
                (
                    v[1].parse()?,
                    v[2].parse::<usize>()? - 1,
                    v[3].parse::<f64>()? * 0.5,
                    [v[5].parse()?, v[6].parse()?, v[7].parse()?],
                ),
            );
        } else if mode == "capsules" {
            shapes.insert(
                v[0].parse()?,
                (v[1].parse()?, [v[2].parse()?, v[3].parse()?, v[4].parse()?]),
            );
        } else {
            bonds.insert((v[2].parse::<u32>()?, v[3].parse::<u32>()?));
        }
    }
    let lengths: Vec3 = std::array::from_fn(|i| high[i] - low[i]);
    let mut cell = PeriodicCell::orthorhombic(lengths, [true, true, false]);
    cell.origin = low;
    let mut result = FiberAssembly::new(cell);
    for name in ["fine_7um", "coarse_19um"] {
        result.materials.add(name);
    }
    for radius in [3.5e-6, 9.5e-6] {
        result.sections.add(Section::Circular { radius });
    }
    let mut fibers = BTreeMap::<u32, (usize, Vec<Vec3>, u32)>::new();
    let mut maximum_gap = 0.0_f64;
    for (id, (molecule, material, radius, center)) in &atoms {
        let (half, axis) = shapes.get(id).ok_or("capsule shape missing")?;
        if (axis.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() > 1e-10 {
            return Err("nonunit capsule axis".into());
        }
        if *material > 1 || (*radius - [3.5e-6, 9.5e-6][*material]).abs() > 1e-12 {
            return Err("unexpected material/diameter".into());
        }
        let mut a: Vec3 = std::array::from_fn(|i| center[i] - half * axis[i]);
        let mut b: Vec3 = std::array::from_fn(|i| center[i] + half * axis[i]);
        let entry = fibers
            .entry(*molecule)
            .or_insert((*material, Vec::new(), *id));
        if let Some(last) = entry.1.last() {
            if !bonds.remove(&(entry.2, *id)) {
                return Err("nonconsecutive or missing intra-fiber bond".into());
            }
            for i in 0..2 {
                let shift = ((last[i] - a[i]) / lengths[i]).round() * lengths[i];
                a[i] += shift;
                b[i] += shift;
            }
            maximum_gap = maximum_gap.max(distance(*last, a));
        } else {
            entry.1.push(a);
        }
        entry.1.push(b);
        entry.2 = *id;
    }
    if maximum_gap > 1e-10 || !bonds.is_empty() {
        return Err(format!(
            "geometry continuity check failed: gap {maximum_gap}, unused bonds {}",
            bonds.len()
        )
        .into());
    }
    for (id, (material, points, _)) in fibers {
        result.add_fiber(
            FiberId(id),
            tangle_core::MaterialId(material as u32),
            tangle_core::SectionId(material as u32),
            &points,
            &points,
        )?;
    }
    result.provenance.source = path.display().to_string();
    result.provenance.notes.push(
        "Placed geometry reconstructed from final capsule export; rest geometry unavailable."
            .into(),
    );
    Ok((result, maximum_gap))
}
