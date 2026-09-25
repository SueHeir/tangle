//! Solver-neutral geometry and persistent topology for fibrous assemblies.
//!
//! FiberAssembly is intentionally not a mechanics model. It records what
//! fibers exist, their intrinsic and placed centerlines, and which material
//! locations participate in persistent Junctions. Contact candidates, solver
//! histories, GPU buffers, meshes, and voxels are derived resources.

#![warn(missing_docs)]

mod admissibility;
mod assembly;
mod cell;
mod curvature;
mod director;
mod error;
mod fiber;
mod ids;
mod junction;
mod material;
mod math;
mod provenance;
mod section;
mod validation;

pub use admissibility::{FiberAdmissibility, FiberBendLimit};
pub use assembly::FiberAssembly;
pub use cell::PeriodicCell;
pub use curvature::{maximum_polyline_curvature, measure_vertex_curvature, VertexCurvature};
pub use director::{
    default_director, default_directors, orthonormalize_directors, polyline_tangents,
    project_director,
};
pub use error::{AnchorError, BuildError};
pub use fiber::{Fiber, FiberGeometry, FiberTopology, GeometryState};
pub use ids::{
    FiberId, JunctionId, JunctionLawId, JunctionParameterId, MaterialId, SectionId, Span,
};
pub use junction::{
    FiberAnchor, Junction, JunctionLawDescriptor, JunctionLawTable, JunctionTable, ResolvedAnchor,
};
pub use material::{MaterialDescriptor, MaterialTable};
pub use provenance::Provenance;
pub use section::{Section, SectionTable};
pub use validation::{ValidationIssue, ValidationReport};

/// A three-dimensional point or vector.
pub type Vec3 = [f64; 3];
