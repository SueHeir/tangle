//! Small path helpers shared by TANGLE's executable examples.
//!
//! This crate deliberately does not construct or run a GRASS app. The plugin
//! schedule remains visible in each teaching example.

use std::path::{Path, PathBuf};

use tangle_export::{BpmExportConfig, BpmExportMode, OvitoColoring, OvitoTrajectoryConfig};

/// Conventional output files for one example or example case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExampleOutput {
    /// Root directory for this run.
    pub directory: PathBuf,
    /// Bonded-particle LAMMPS data file.
    pub dem_data: PathBuf,
    /// Multi-frame OVITO-readable LAMMPS dump.
    pub ovito_dump: PathBuf,
    /// Portable OVITO Python viewing recipe.
    pub ovito_view_script: PathBuf,
    /// Session generated when the viewing recipe is run.
    pub ovito_session: PathBuf,
}

impl ExampleOutput {
    /// Creates paths beneath `<manifest>/output/<case>`.
    ///
    /// An empty case places files directly beneath `<manifest>/output`.
    pub fn for_case(manifest_dir: impl AsRef<Path>, case: impl AsRef<Path>) -> Self {
        let mut directory = manifest_dir.as_ref().join("output");
        if !case.as_ref().as_os_str().is_empty() {
            directory.push(case);
        }
        Self {
            dem_data: directory.join("relaxed_fibers_bpm.data"),
            ovito_dump: directory.join("relaxation.dump"),
            ovito_view_script: directory.join("relaxation_view.py"),
            ovito_session: directory.join("relaxation.ovito"),
            directory,
        }
    }

    /// Creates an endpoint-preserving bonded-sphere BPM output.
    pub fn sphere_bpm(&self, density: f64) -> BpmExportConfig {
        BpmExportConfig::new(&self.dem_data)
            .with_mode(BpmExportMode::SpheresDynamic)
            .with_sphere_spacing_over_radius(1.0)
            .with_density(density)
    }

    /// Creates the exact active-segment spherocylinder BPM output.
    pub fn capsule_bpm(&self, density: f64) -> BpmExportConfig {
        BpmExportConfig::new(&self.dem_data).with_density(density)
    }

    /// Creates an oriented-segment trajectory with a generated viewing recipe.
    pub fn ovito_segments(
        &self,
        frame_interval: usize,
        coloring: OvitoColoring,
    ) -> OvitoTrajectoryConfig {
        OvitoTrajectoryConfig::fiber_segments(&self.ovito_dump, frame_interval)
            .with_coloring(coloring)
            .with_viewing_files(&self.ovito_view_script, &self.ovito_session)
    }
}

/// Whether the conventional optional OVITO trajectory was requested.
pub fn debug_ovito_requested() -> bool {
    std::env::args().any(|argument| argument == "--debug-ovito")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_case_scoped_output_paths() {
        let output = ExampleOutput::for_case("/tmp/example", "aligned");
        assert_eq!(
            output.dem_data,
            PathBuf::from("/tmp/example/output/aligned/relaxed_fibers_bpm.data")
        );
        assert_eq!(
            output.ovito_session,
            PathBuf::from("/tmp/example/output/aligned/relaxation.ovito")
        );
    }
}
