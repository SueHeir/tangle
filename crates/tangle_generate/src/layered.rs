use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use tangle_app::prelude::*;
use tangle_core::FiberAssembly;

use tangle_relax::{DeviceState, RelaxationState, WorkflowControl};

/// GRASS-controlled planar formation schedule applied to resident GPU fibers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayeredFormationConfig {
    /// Layer-normal Cartesian axis.
    pub axis: usize,
    /// Device iterations at the original layer spacing.
    pub initial_relax_iterations: usize,
    /// Number of discrete reductions in layer spacing.
    pub compaction_steps: usize,
    /// Device iterations performed at each compacted spacing.
    pub iterations_per_step: usize,
    /// Final layer spacing divided by generated layer spacing.
    pub final_spacing_scale: f32,
    /// Fraction of fiber-center error translated before each batch.
    pub tether_stiffness: f32,
    /// Maximum normal translation applied to one fiber before a batch.
    pub max_translation: f32,
}

impl Default for LayeredFormationConfig {
    fn default() -> Self {
        Self {
            axis: 2,
            initial_relax_iterations: 128,
            compaction_steps: 4,
            iterations_per_step: 128,
            final_spacing_scale: 0.8,
            tether_stiffness: 0.5,
            max_translation: 0.01,
        }
    }
}

/// Observable progress of the layered manufacturing protocol.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayeredFormationState {
    /// Original mean coordinate of every generated layer.
    pub initial_targets: Vec<f32>,
    /// Target-spacing scale used for the next relaxation batch.
    pub current_spacing_scale: f32,
    /// Whether layer-target commands have ended for final global relaxation.
    pub released: bool,
    /// Bytes uploaded in small layer-target command buffers.
    pub command_bytes_uploaded: usize,
    /// Most recent formation phase whose target command was dispatched.
    pub last_command_phase: Option<usize>,
}

/// Formation controller that adjusts small layer-target buffers between stages
/// while the GPU applies the active target every solver iteration and keeps
/// fiber geometry resident.
pub struct LayeredFormationPlugin {
    /// Staged compaction parameters.
    pub config: LayeredFormationConfig,
}

impl Plugin for LayeredFormationPlugin {
    fn build(&self, app: &mut App) {
        assert!(self.config.axis < 3);
        assert!(self.config.initial_relax_iterations > 0);
        assert!(self.config.compaction_steps > 0);
        assert!(self.config.iterations_per_step > 0);
        assert!(self.config.final_spacing_scale > 0.0);
        assert!(self.config.final_spacing_scale <= 1.0);
        assert!(self.config.tether_stiffness > 0.0 && self.config.tether_stiffness <= 1.0);
        assert!(self.config.max_translation > 0.0);
        app.add_resource(self.config)
            .add_resource(LayeredFormationState::default())
            .add_update_system(
                control_layered_formation.run_if(in_state(TangleStage::Relax)),
                TanglePhase::BuildBroadPhase,
            );
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_FORMATION.clone()]
    }

    fn requires_capabilities(&self) -> Vec<CapabilityId> {
        vec![
            TANGLE_WORKFLOW.clone(),
            TANGLE_ASSEMBLY.clone(),
            TANGLE_RELAXATION.clone(),
        ]
    }
}

fn control_layered_formation(
    config: Res<LayeredFormationConfig>,
    assembly: Res<FiberAssembly>,
    mut device: ResMut<DeviceState>,
    mut relaxation: ResMut<RelaxationState>,
    mut workflow: ResMut<WorkflowControl>,
    mut state: ResMut<LayeredFormationState>,
) {
    if state.initial_targets.is_empty() {
        state.initial_targets = layer_targets(&assembly, config.axis);
        state.current_spacing_scale = 1.0;
    }
    if state.released || relaxation.cell_list_overflow {
        if let Some(world) = device.world.as_mut() {
            world.clear_layer_targets();
        }
        workflow.hold_relax_stage = false;
        workflow.batch_iteration_limit = None;
        workflow.force_full_batch = false;
        return;
    }

    let formation_iterations =
        config.initial_relax_iterations + config.compaction_steps * config.iterations_per_step;
    if relaxation.iterations >= formation_iterations {
        state.released = true;
        device
            .world
            .as_mut()
            .expect("layered formation requires an uploaded CubeCL device world")
            .clear_layer_targets();
        workflow.hold_relax_stage = false;
        workflow.batch_iteration_limit = None;
        workflow.force_full_batch = false;
        // A previous batch may have converged under its layer tether. Release
        // it and explicitly request a final unconstrained batch.
        relaxation.converged = false;
        return;
    }

    workflow.hold_relax_stage = true;
    workflow.force_full_batch = true;
    relaxation.converged = false;
    let (phase, scale, next_boundary) = if relaxation.iterations < config.initial_relax_iterations {
        (0, 1.0, config.initial_relax_iterations)
    } else {
        let compacted_iteration = relaxation.iterations - config.initial_relax_iterations;
        let step = compacted_iteration / config.iterations_per_step + 1;
        let fraction = step as f32 / config.compaction_steps as f32;
        (
            step,
            1.0 - fraction * (1.0 - config.final_spacing_scale),
            config.initial_relax_iterations + step * config.iterations_per_step,
        )
    };
    workflow.batch_iteration_limit = Some(next_boundary - relaxation.iterations);
    state.current_spacing_scale = scale;
    let center = 0.5
        * (assembly.cell.origin[config.axis]
            + assembly.cell.origin[config.axis]
            + assembly.cell.basis[config.axis][config.axis]) as f32;
    if state.last_command_phase == Some(phase) {
        return;
    }
    let targets: Vec<f32> = state
        .initial_targets
        .iter()
        .map(|target| center + (*target - center) * scale)
        .collect();
    state.command_bytes_uploaded += targets.len() * size_of::<f32>();
    state.last_command_phase = Some(phase);
    device
        .world
        .as_mut()
        .expect("layered formation requires an uploaded CubeCL device world")
        .apply_layer_targets(
            config.axis,
            &targets,
            config.tether_stiffness,
            config.max_translation,
        );
}

pub(crate) fn layer_targets(assembly: &FiberAssembly, axis: usize) -> Vec<f32> {
    let layer_count = assembly
        .topology
        .fibers
        .iter()
        .filter_map(|fiber| fiber.formation_layer)
        .max()
        .map_or(0, |maximum| maximum as usize + 1);
    assert!(
        layer_count > 0,
        "layered formation requires formation-layer labels"
    );
    let mut sums = vec![0.0_f64; layer_count];
    let mut counts = vec![0_usize; layer_count];
    for fiber in &assembly.topology.fibers {
        let Some(layer) = fiber.formation_layer.map(|layer| layer as usize) else {
            continue;
        };
        let start = fiber.vertices.start as usize;
        let end = fiber.vertices.checked_end().expect("valid fiber span") as usize;
        let positions = &assembly.geometry.placed.positions[start..end];
        let center =
            positions.iter().map(|point| point[axis]).sum::<f64>() / positions.len() as f64;
        sums[layer] += center;
        counts[layer] += 1;
    }
    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            assert!(
                count > 0,
                "formation layers must be contiguous and populated"
            );
            (sum / count as f64) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell, Section};

    #[test]
    fn calculates_explicit_layer_centers() {
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
        assert_eq!(layer_targets(&assembly, 2), vec![0.2, 0.8]);
    }
}
