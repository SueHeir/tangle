//! GRASS workflow orchestration for TANGLE fiber-assembly operations.
//!
//! This crate owns ordering, lifecycle, and plugin contracts. Geometry remains
//! in tangle_core, while generation, contact, relaxation, junction, and export
//! crates can supply replaceable systems at the phases defined here.

#![warn(missing_docs)]

use grass_app::prelude::*;
use grass_derive::ScheduleSet;
use grass_scheduler::prelude::*;
use tangle_core::{FiberAssembly, PeriodicCell};

/// Capability provided by the base TANGLE workflow plugin.
pub const TANGLE_WORKFLOW: CapabilityId = CapabilityId::new("tangle.workflow");
/// Capability provided when a FiberAssembly resource is installed.
pub const TANGLE_ASSEMBLY: CapabilityId = CapabilityId::new("tangle.assembly");
/// Capability provided by a fiber-generation plugin.
pub const TANGLE_GENERATION: CapabilityId = CapabilityId::new("tangle.generation");
/// Capability provided by a contact-detection plugin.
pub const TANGLE_CONTACTS: CapabilityId = CapabilityId::new("tangle.contacts");
/// Capability provided by a relaxation plugin.
pub const TANGLE_RELAXATION: CapabilityId = CapabilityId::new("tangle.relaxation");
/// Capability provided by a junction-formation plugin.
pub const TANGLE_JUNCTIONS: CapabilityId = CapabilityId::new("tangle.junctions");
/// Capability provided by an export plugin.
pub const TANGLE_EXPORT: CapabilityId = CapabilityId::new("tangle.export");
/// Capability provided by optional diagnostic output plugins.
pub const TANGLE_DEBUG_OUTPUT: CapabilityId = CapabilityId::new("tangle.debug_output");
/// Capability provided by assembly-characterization plugins.
pub const TANGLE_CHARACTERIZATION: CapabilityId = CapabilityId::new("tangle.characterization");
/// Capability provided by plugins that reproduce a manufacturing/formation protocol.
pub const TANGLE_FORMATION: CapabilityId = CapabilityId::new("tangle.formation");

/// Coarse workflow state for preparing or consuming a fiber assembly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TangleStage {
    /// Create intrinsic fibers and their initial placement.
    #[default]
    Generate,
    /// Iteratively detect and resolve invalid overlaps.
    Relax,
    /// Convert selected final contacts into persistent junctions.
    FormJunctions,
    /// Measure the completed assembly.
    Characterize,
    /// Hold a prepared assembly for a configured downstream action.
    Ready,
    /// Apply a mechanical loading or deformation process.
    Deform,
    /// Write or hand off a derived representation.
    Export,
    /// End the GRASS application loop.
    Done,
}

/// Ordered phases available to TANGLE workflow plugins.
///
/// A stage selects which systems are active. These phases then order the
/// active systems within one GRASS update.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ScheduleSet)]
pub enum TanglePhase {
    /// Generate or import fibers.
    Generate,
    /// Rebuild segment geometry derived from the assembly.
    PrepareGeometry,
    /// Build a spatial broad-phase index.
    BuildBroadPhase,
    /// Produce exact geometric contact data.
    DetectContacts,
    /// Apply one relaxation or mechanics iteration.
    ApplyConstraints,
    /// Commit solver results to placed geometry.
    UpdateGeometry,
    /// Observe the current geometry for diagnostics or trajectory output.
    Observe,
    /// Measure residuals and request stage transitions.
    MeasureConvergence,
    /// Create persistent junctions from selected contacts.
    FormJunctions,
    /// Export or hand off the assembly.
    Export,
    /// Apply pending workflow-state transitions.
    Transition,
    /// Stop the application after entering the done state.
    Stop,
}

/// Installs the TANGLE state machine and application stop condition.
pub struct TangleWorkflowPlugin {
    /// Initial workflow stage.
    pub initial: TangleStage,
}

impl Default for TangleWorkflowPlugin {
    fn default() -> Self {
        Self {
            initial: TangleStage::Generate,
        }
    }
}

impl Plugin for TangleWorkflowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(StatesPlugin::new(self.initial, TanglePhase::Transition))
            .add_update_system(finish_when_done, TanglePhase::Stop);
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_WORKFLOW.clone()]
    }

    fn schedule_labels(&self) -> Vec<&'static str> {
        vec![
            "Generate",
            "PrepareGeometry",
            "BuildBroadPhase",
            "DetectContacts",
            "ApplyConstraints",
            "UpdateGeometry",
            "Observe",
            "MeasureConvergence",
            "FormJunctions",
            "Export",
            "Transition",
            "Stop",
        ]
    }

    fn extension_points(&self) -> Vec<&'static str> {
        vec!["TangleStage", "TanglePhase", "FiberAssembly resource"]
    }
}

/// Installs an initially empty FiberAssembly resource in a chosen cell.
pub struct TangleAssemblyPlugin {
    /// Cell used by the new assembly.
    pub cell: PeriodicCell,
}

/// Installs an existing, fully constructed [`FiberAssembly`] as the canonical
/// assembly resource.
///
/// This is the import-side counterpart to [`TangleAssemblyPlugin`]. It is
/// useful for front ends such as the Python bindings, which construct fiber
/// collections before handing the complete staged assembly to the GPU
/// workflow.
pub struct TanglePreparedAssemblyPlugin {
    /// Assembly installed when the plugin is built.
    pub assembly: FiberAssembly,
}

impl TangleAssemblyPlugin {
    /// Creates an assembly plugin for the supplied cell.
    pub fn new(cell: PeriodicCell) -> Self {
        Self { cell }
    }
}

impl Plugin for TangleAssemblyPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(FiberAssembly::new(self.cell));
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_ASSEMBLY.clone()]
    }
}

impl Plugin for TanglePreparedAssemblyPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(self.assembly.clone());
    }

    fn provides_capabilities(&self) -> Vec<CapabilityId> {
        vec![TANGLE_ASSEMBLY.clone()]
    }
}

fn finish_when_done(
    stage: Res<CurrentState<TangleStage>>,
    mut scheduler: ResMut<SchedulerManager>,
) {
    if stage.0 == TangleStage::Done {
        scheduler.state = SchedulerState::End;
    }
}

/// Common imports for TANGLE application and plugin authors.
pub mod prelude {
    pub use crate::{
        TangleAssemblyPlugin, TanglePhase, TanglePreparedAssemblyPlugin, TangleStage,
        TangleWorkflowPlugin, TANGLE_ASSEMBLY, TANGLE_CHARACTERIZATION, TANGLE_CONTACTS,
        TANGLE_DEBUG_OUTPUT, TANGLE_EXPORT, TANGLE_FORMATION, TANGLE_GENERATION, TANGLE_JUNCTIONS,
        TANGLE_RELAXATION, TANGLE_WORKFLOW,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Counts {
        generate: usize,
        relax: usize,
        junctions: usize,
        export: usize,
    }

    fn generate(mut counts: ResMut<Counts>, mut next: ResMut<NextState<TangleStage>>) {
        counts.generate += 1;
        next.set(TangleStage::Relax);
    }

    fn relax(mut counts: ResMut<Counts>, mut next: ResMut<NextState<TangleStage>>) {
        counts.relax += 1;
        if counts.relax == 3 {
            next.set(TangleStage::FormJunctions);
        }
    }

    fn form_junctions(mut counts: ResMut<Counts>, mut next: ResMut<NextState<TangleStage>>) {
        counts.junctions += 1;
        next.set(TangleStage::Export);
    }

    fn export(mut counts: ResMut<Counts>, mut next: ResMut<NextState<TangleStage>>) {
        counts.export += 1;
        next.set(TangleStage::Done);
    }

    #[test]
    fn derived_phase_order_matches_declaration_order() {
        assert_eq!(TanglePhase::Generate.to_index(), 0);
        assert_eq!(TanglePhase::Observe.to_index(), 6);
        assert_eq!(TanglePhase::MeasureConvergence.to_index(), 7);
        assert_eq!(TanglePhase::Stop.to_index(), 11);
        assert_eq!(TanglePhase::FormJunctions.name(), "FormJunctions");
    }

    #[test]
    fn workflow_orders_one_shot_and_iterative_stages() {
        let mut app = App::new();
        app.add_resource(Counts::default())
            .add_plugins(TangleWorkflowPlugin::default())
            .add_update_system(
                generate.run_if(in_state(TangleStage::Generate)),
                TanglePhase::Generate,
            )
            .add_update_system(
                relax.run_if(in_state(TangleStage::Relax)),
                TanglePhase::ApplyConstraints,
            )
            .add_update_system(
                form_junctions.run_if(in_state(TangleStage::FormJunctions)),
                TanglePhase::FormJunctions,
            )
            .add_update_system(
                export.run_if(in_state(TangleStage::Export)),
                TanglePhase::Export,
            );

        app.start();

        let counts = app.get_resource_ref::<Counts>().unwrap();
        assert_eq!(counts.generate, 1);
        assert_eq!(counts.relax, 3);
        assert_eq!(counts.junctions, 1);
        assert_eq!(counts.export, 1);
    }
}
