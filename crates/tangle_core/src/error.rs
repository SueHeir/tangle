use std::error::Error;
use std::fmt;

use crate::{FiberId, JunctionId, JunctionLawId, MaterialId, SectionId};

/// Failures returned while incrementally constructing an assembly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// Intrinsic and placed centerlines use different vertex counts.
    GeometryLengthMismatch {
        /// Number of intrinsic vertices.
        intrinsic: usize,
        /// Number of placed vertices.
        placed: usize,
    },
    /// A fiber had fewer than two vertices.
    TooFewVertices(usize),
    /// A junction had fewer than two anchors.
    TooFewAnchors(usize),
    /// A fiber ID was reused.
    DuplicateFiberId(FiberId),
    /// A junction ID was reused.
    DuplicateJunctionId(JunctionId),
    /// A referenced fiber does not exist.
    UnknownFiber(FiberId),
    /// A referenced material does not exist.
    UnknownMaterial(MaterialId),
    /// A referenced cross-section does not exist.
    UnknownSection(SectionId),
    /// A referenced junction law does not exist.
    UnknownJunctionLaw(JunctionLawId),
    /// A flat table cannot be represented with 32-bit dense indices.
    IndexOverflow,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeometryLengthMismatch { intrinsic, placed } => write!(
                f,
                "intrinsic vertex count {intrinsic} differs from placed count {placed}"
            ),
            Self::TooFewVertices(count) => {
                write!(f, "a fiber needs at least 2 vertices, got {count}")
            }
            Self::TooFewAnchors(count) => {
                write!(f, "a junction needs at least 2 anchors, got {count}")
            }
            Self::DuplicateFiberId(id) => write!(f, "fiber ID {:?} already exists", id),
            Self::DuplicateJunctionId(id) => write!(f, "junction ID {:?} already exists", id),
            Self::UnknownFiber(id) => write!(f, "unknown fiber ID {:?}", id),
            Self::UnknownMaterial(id) => write!(f, "unknown material ID {:?}", id),
            Self::UnknownSection(id) => write!(f, "unknown section ID {:?}", id),
            Self::UnknownJunctionLaw(id) => write!(f, "unknown junction law ID {:?}", id),
            Self::IndexOverflow => f.write_str("assembly exceeds 32-bit dense-index capacity"),
        }
    }
}

impl Error for BuildError {}

/// Failures returned while resolving a material anchor.
#[derive(Clone, Debug, PartialEq)]
pub enum AnchorError {
    /// The stable fiber ID does not exist.
    UnknownFiber(FiberId),
    /// The fiber's flat vertex span is invalid.
    InvalidFiberSpan(FiberId),
    /// The arc-length coordinate is negative or non-finite.
    InvalidArcLength(f64),
    /// The polyline contains no segment of positive length.
    DegenerateFiber,
    /// The coordinate lies beyond the fiber's intrinsic length.
    BeyondFiber {
        /// Requested material coordinate.
        requested: f64,
        /// Total intrinsic fiber length.
        length: f64,
    },
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFiber(id) => write!(f, "unknown fiber ID {:?}", id),
            Self::InvalidFiberSpan(id) => write!(f, "fiber {:?} has an invalid vertex span", id),
            Self::InvalidArcLength(value) => write!(f, "invalid rest arc length {value}"),
            Self::DegenerateFiber => f.write_str("fiber has no positive-length segment"),
            Self::BeyondFiber { requested, length } => {
                write!(
                    f,
                    "rest arc length {requested} exceeds fiber length {length}"
                )
            }
        }
    }
}

impl Error for AnchorError {}
