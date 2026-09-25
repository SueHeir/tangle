use crate::{
    write_ovito_assembly_frame, write_ovito_view_script, OvitoRepresentation,
    OvitoTrajectoryConfig, OvitoTrajectoryReport,
};
use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::FiberAssembly;
use tangle_relax::{PackedAssembly, RelaxationConfig, RelaxationState, WorkflowControl};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct OvitoTrajectoryState {
    initialized: bool,
    final_written: bool,
    next_frame_index: usize,
    last_written_step: Option<usize>,
}

/// Optional GRASS plugin that turns explicitly requested CubeCL snapshot
/// readbacks into an OVITO trajectory.
///
/// The snapshot interval in [`RelaxationConfig`] must match the frame
/// interval here. With no debug plugin, the solver performs no intermediate
/// geometry readbacks.
pub struct OvitoTrajectoryPlugin {
    /// Trajectory configuration shared with the host reference recorder.
    pub config: OvitoTrajectoryConfig,
}

impl Plugin for OvitoTrajectoryPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            self.config.frame_interval > 0,
            "OVITO trajectory frame_interval must be positive"
        );
        if let OvitoRepresentation::DemParticles { spacing_ratio } = self.config.representation {
            assert!(
                spacing_ratio.is_finite() && spacing_ratio > 0.0 && spacing_ratio <= 1.0,
                "OVITO trajectory spacing_ratio must be in (0, 1]"
            );
        }
        assert!(
            self.config.atom_type > 0,
            "OVITO trajectory atom_type must be positive"
        );

        app.add_resource(self.config.clone())
            .add_resource(OvitoTrajectoryState::default())
            .add_resource(OvitoTrajectoryReport {
                dump_path: self.config.dump_path.clone(),
                view_script_path: self.config.view_script_path.clone(),
                session_path: self.config.session_path.clone(),
                ..OvitoTrajectoryReport::default()
            })
            .add_update_system(record_device_trajectory, TanglePhase::Observe);
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_DEBUG_OUTPUT.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![
            TANGLE_WORKFLOW.clone(),
            TANGLE_ASSEMBLY.clone(),
            TANGLE_RELAXATION.clone(),
        ]
    }
}

fn record_device_trajectory(
    assembly: Res<FiberAssembly>,
    stage: Res<CurrentState<TangleStage>>,
    solver_config: Res<RelaxationConfig>,
    mut relaxation: ResMut<RelaxationState>,
    workflow: Res<WorkflowControl>,
    config: Res<OvitoTrajectoryConfig>,
    mut state: ResMut<OvitoTrajectoryState>,
    mut report: ResMut<OvitoTrajectoryReport>,
) {
    assert_eq!(
        solver_config.debug_snapshot_interval,
        Some(config.frame_interval),
        "CubeCL debug_snapshot_interval must match the OVITO frame_interval"
    );

    if !state.initialized {
        if config.write_initial_frame {
            write_ovito_assembly_frame(&assembly, &config, 0, false)
                .unwrap_or_else(|error| panic!("OVITO trajectory output failed: {error}"));
            state.last_written_step = Some(0);
            state.next_frame_index = 1;
            report.frames = 1;
            report.last_timestep = Some(0);
        } else if let Err(error) = std::fs::remove_file(&config.dump_path) {
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::NotFound,
                "could not reset OVITO trajectory {}: {error}",
                config.dump_path.display()
            );
        }
        if let Some(script_path) = &config.view_script_path {
            write_ovito_view_script(&config, script_path)
                .unwrap_or_else(|error| panic!("OVITO viewing-recipe output failed: {error}"));
        }
        if let Some(session_path) = &config.session_path {
            if let Err(error) = std::fs::remove_file(session_path) {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::NotFound,
                    "could not reset OVITO session target {}: {error}",
                    session_path.display()
                );
            }
        }
        state.initialized = true;
        if stage.0 == TangleStage::Generate {
            return;
        }
    }

    if stage.0 != TangleStage::Relax || state.final_written {
        return;
    }

    let packed = PackedAssembly::from_assembly(&assembly)
        .unwrap_or_else(|error| panic!("CubeCL OVITO snapshot packing failed: {error}"));
    let mut frame_assembly = assembly.clone();
    // Frames are append-only once written, so release their large host-side
    // assemblies immediately. Detailed trajectories can contain hundreds of
    // frames while the authoritative geometry remains resident on the GPU.
    let snapshots = std::mem::take(&mut relaxation.snapshots);
    for snapshot in snapshots {
        if let Some(snapshot_assembly) = &snapshot.assembly {
            frame_assembly = snapshot_assembly.clone();
        } else {
            packed
                .unpack_positions(&snapshot.positions, None, &mut frame_assembly)
                .unwrap_or_else(|error| panic!("CubeCL OVITO snapshot was invalid: {error}"));
        }
        write_ovito_assembly_frame(&frame_assembly, &config, state.next_frame_index, true)
            .unwrap_or_else(|error| panic!("OVITO trajectory output failed: {error}"));
        state.next_frame_index += 1;
        state.last_written_step = Some(snapshot.iteration);
        report.frames += 1;
        report.last_timestep = Some(snapshot.iteration);
    }
    let terminal = (relaxation.converged && !workflow.hold_relax_stage)
        || relaxation.cell_list_overflow
        || relaxation.iterations >= solver_config.max_iterations;
    if terminal && state.last_written_step != Some(relaxation.iterations) {
        write_ovito_assembly_frame(&assembly, &config, state.next_frame_index, true)
            .unwrap_or_else(|error| panic!("OVITO trajectory output failed: {error}"));
        state.next_frame_index += 1;
        state.last_written_step = Some(relaxation.iterations);
        report.frames += 1;
        report.last_timestep = Some(relaxation.iterations);
    }
    state.final_written = terminal;
}
