# tangle_app

GRASS workflow orchestration for TANGLE.

The crate defines two orthogonal ordering concepts:

- TangleStage selects the active coarse operation, such as generation,
  relaxation, junction formation, deformation, or export.
- TanglePhase orders systems within each application update, including
  contact preparation, constraint application, convergence measurement, and
  state transitions.

Scientific algorithms remain ordinary functions in their owning crates.
Plugins use this crate to install those functions into a shared workflow.
