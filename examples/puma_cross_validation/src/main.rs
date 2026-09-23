use std::error::Error;
use std::path::PathBuf;

use tangle_characterize::characterize_assembly;
use tangle_core::{FiberAssembly, FiberId, PeriodicCell, Section};
use tangle_export::{write_puma_bundle, PumaVoxelExportConfig};

const CELL_LENGTH: f64 = 100.0e-6;
const FIBER_LENGTH: f64 = 80.0e-6;
const FIBER_RADIUS: f64 = 5.0e-6;
const AXIS_SEPARATION: f64 = 12.0e-6;
const VOXEL_SIZE: f64 = 2.0e-6;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("output/crossed.puma"));

    let assembly = orthogonal_pair()?;
    let analysis = characterize_assembly(&assembly);
    let export = write_puma_bundle(&assembly, &PumaVoxelExportConfig::new(&output, VOXEL_SIZE))?;

    println!("TANGLE -> PuMA cross-validation fixture");
    println!(
        "  {} fibers, {} segments, nominal Vf {:.6}",
        analysis.fibers, analysis.segments, analysis.nominal_swept_volume_fraction
    );
    println!(
        "  length-weighted orientation: {:?}",
        analysis.orientation_tensor
    );
    println!(
        "  {:?} voxels, voxel Vf {:.6}, {} ownership ties",
        export.voxel_counts, export.voxel_volume_fraction, export.ambiguous_voxels
    );
    println!("  bundle: {}", export.output_directory.display());
    Ok(())
}

fn orthogonal_pair() -> Result<FiberAssembly, Box<dyn Error>> {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([CELL_LENGTH; 3], [false; 3]));
    let material = assembly.materials.add("validation fiber");
    let center = 0.5 * CELL_LENGTH;
    let half_length = 0.5 * FIBER_LENGTH;
    let fibers = [
        [
            [center - half_length, center, center - 0.5 * AXIS_SEPARATION],
            [center + half_length, center, center - 0.5 * AXIS_SEPARATION],
        ],
        [
            [center, center - half_length, center + 0.5 * AXIS_SEPARATION],
            [center, center + half_length, center + 0.5 * AXIS_SEPARATION],
        ],
    ];
    for (index, placed) in fibers.into_iter().enumerate() {
        // Keep a separate section-table entry per source fiber, matching the
        // construction performed by the Python collection insertion API.
        let section = assembly.sections.add(Section::Circular {
            radius: FIBER_RADIUS,
        });
        assembly.add_fiber(
            FiberId(index as u32 + 1),
            material,
            section,
            &placed,
            &placed,
        )?;
    }
    assembly.provenance.source = "puma_cross_validation".into();
    assembly.provenance.version = env!("CARGO_PKG_VERSION").into();
    Ok(assembly)
}
