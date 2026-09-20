use std::f64::consts::PI;

use tangle_core::{FiberAssembly, Section};
use tangle_relax::{CompactionEnergyModel, CompactionKinematics, CompactionMetrics};

/// Observable quantity that ends one closed-loop compaction operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompactionTarget {
    /// Nominal fiber volume divided by the current cell volume.
    NominalVolumeFraction(f64),
    /// Current orthorhombic cell volume.
    CellVolume(f64),
    /// Three final cell edge lengths.
    CellLengths([f64; 3]),
    /// Arithmetic mean of the three directional penalty pressures.
    MeanPressure(f32),
    /// Per-axis penalty pressure; zero disables an axis target.
    DirectionalPressure([f32; 3]),
    /// Total contact, stretch, and excess-bending penalty energy.
    PenaltyEnergy(f32),
}

/// Rule used to distribute each scalar compaction increment among cell axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompactionPath {
    /// Fixed nonnegative logarithmic-strain ratios.
    AxisWeights([f32; 3]),
    /// Adapt strain rates to approach equal pressure on selected axes.
    EqualPressure {
        /// Axes permitted to shrink.
        active_axes: [bool; 3],
        /// Positive regularization for initially unloaded directions.
        pressure_floor: f32,
    },
    /// Adapt strain rates to approach the supplied directional pressure ratio.
    StressRatio {
        /// Desired nonnegative pressure ratio; zero disables an axis.
        ratio: [f32; 3],
        /// Positive regularization for initially unloaded directions.
        pressure_floor: f32,
    },
    /// Put the next increment on the currently least expensive active axis.
    /// Directional pressure is the derivative of boundary work with respect
    /// to cell shortening in the formation penalty model.
    MinimumIncrementalWork {
        /// Axes permitted to shrink.
        active_axes: [bool; 3],
    },
}

impl Default for CompactionPath {
    fn default() -> Self {
        Self::AxisWeights([1.0; 3])
    }
}

/// Adaptive step-size policy for a compact-relax-measure loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveCompactionIncrement {
    /// Initial total logarithmic volume strain per compaction step.
    pub initial_log_strain: f32,
    /// Smallest permitted logarithmic volume-strain step.
    pub minimum_log_strain: f32,
    /// Largest permitted logarithmic volume-strain step.
    pub maximum_log_strain: f32,
    /// Multiplier after an inexpensive successful relaxation.
    pub growth_factor: f32,
    /// Multiplier after a difficult relaxation window.
    pub shrink_factor: f32,
    /// Relaxation iterations allowed before the controller reassesses a step.
    pub relax_iterations: usize,
    /// Largest total cell shortening in one step, divided by the smallest
    /// active fiber diameter. This geometric cap prevents a nominal strain
    /// increment from moving a wall through an entire fine fiber.
    pub maximum_shortening_over_minimum_diameter: f32,
}

impl Default for AdaptiveCompactionIncrement {
    fn default() -> Self {
        Self {
            initial_log_strain: 0.02,
            minimum_log_strain: 0.001,
            maximum_log_strain: 0.05,
            growth_factor: 1.25,
            shrink_factor: 0.5,
            relax_iterations: 64,
            maximum_shortening_over_minimum_diameter: 0.5,
        }
    }
}

/// Hard limits that can stop an infeasible compaction path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionGuards {
    /// Maximum accepted capsule penetration after a relaxation window.
    pub maximum_penetration: f32,
    /// Maximum accepted current/admissible curvature ratio.
    pub maximum_bend_ratio: f32,
    /// Maximum accepted pressure on any cell axis.
    pub maximum_pressure: f32,
    /// Maximum accepted total formation penalty energy.
    pub maximum_penalty_energy: f32,
    /// Maximum accepted cell-deformation increments.
    pub maximum_steps: usize,
    /// Relaxation windows allowed for one increment before declaring jamming.
    pub maximum_relax_windows: usize,
}

impl Default for CompactionGuards {
    fn default() -> Self {
        Self {
            maximum_penetration: 2.0e-4,
            maximum_bend_ratio: 1.001,
            maximum_pressure: f32::INFINITY,
            maximum_penalty_energy: f32::INFINITY,
            maximum_steps: 1_000,
            maximum_relax_windows: 4,
        }
    }
}

/// One closed-loop, device-resident cell-compaction command.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionConfig {
    /// Quantity used to end compaction.
    pub target: CompactionTarget,
    /// Directional strain-selection rule.
    pub path: CompactionPath,
    /// How active fiber geometry follows moving cell faces.
    pub kinematics: CompactionKinematics,
    /// Fixed point of each cell-axis contraction: zero keeps the lower face
    /// fixed, one keeps the upper face fixed, and one half moves both equally.
    pub cell_anchor: [f32; 3],
    /// Distribute motion between opposing nonperiodic faces according to their
    /// measured resistance instead of using `cell_anchor` unchanged.
    pub balance_opposing_faces: bool,
    /// Positive pressure regularizer used by inverse-resistance face motion.
    pub face_pressure_floor: f32,
    /// Blend from the fixed anchor (zero) toward fully inverse-resistance face
    /// motion (one). One half limits either face to 75% when the base is 0.5.
    pub face_balance_strength: f32,
    /// Adaptive strain-step parameters.
    pub increment: AdaptiveCompactionIncrement,
    /// Physical and numerical safety limits.
    pub guards: CompactionGuards,
    /// Penalty stiffnesses used for pressure, work, and energy control.
    pub energy_model: CompactionEnergyModel,
    /// Relative tolerance used for scalar target comparisons.
    pub target_tolerance: f32,
}

impl CompactionConfig {
    /// Creates volume-fraction compaction with prescribed axis ratios.
    pub fn volume_fraction(target: f64, axis_weights: [f32; 3]) -> Self {
        Self {
            target: CompactionTarget::NominalVolumeFraction(target),
            path: CompactionPath::AxisWeights(axis_weights),
            kinematics: CompactionKinematics::RigidFiberCenters,
            cell_anchor: [0.5; 3],
            balance_opposing_faces: false,
            face_pressure_floor: 1.0e-12,
            face_balance_strength: 0.5,
            increment: AdaptiveCompactionIncrement::default(),
            guards: CompactionGuards::default(),
            energy_model: CompactionEnergyModel::default(),
            target_tolerance: 1.0e-3,
        }
    }
}

/// Why a compaction operation stopped.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompactionStopReason {
    /// The requested target was reached.
    TargetReached,
    /// A named safety guard was reached first.
    GuardReached(String),
    /// Relaxation could not produce an admissible state.
    Jammed,
}

/// Host-visible record of one accepted compact-relax-measure increment.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompactionStepReport {
    /// Formation-recipe operation owning this increment.
    pub operation: usize,
    /// One-based increment number within the operation.
    pub step: usize,
    /// Device relaxation iteration at the measurement checkpoint.
    pub iteration: usize,
    /// Current cell lengths.
    pub cell_lengths: [f64; 3],
    /// Current nominal volume fraction of active fibers.
    pub nominal_volume_fraction: f64,
    /// Total logarithmic volume strain applied by this increment.
    pub log_volume_strain: f32,
    /// Maximum capsule penetration when this increment was accepted.
    pub max_penetration: f32,
    /// Maximum current/admissible curvature ratio when accepted.
    pub max_bend_ratio: f32,
    /// Directional and energetic response reduced on the GPU.
    pub metrics: CompactionMetrics,
    /// Accumulated directional boundary work.
    pub cumulative_work: [f32; 3],
}

/// Final report for one compaction recipe operation.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompactionReport {
    /// Formation-recipe operation index.
    pub operation: usize,
    /// Reason compaction stopped.
    pub reason: CompactionStopReason,
    /// Number of accepted increments.
    pub steps: usize,
    /// Final cell lengths.
    pub cell_lengths: [f64; 3],
    /// Final nominal active-fiber volume fraction.
    pub nominal_volume_fraction: f64,
    /// Final GPU-reduced response.
    pub metrics: CompactionMetrics,
    /// Accumulated directional boundary work.
    pub cumulative_work: [f32; 3],
}

/// Nominal solid volume of fibers active through the supplied formation step.
///
/// The measure integrates cross-sectional area along intrinsic centerline arc
/// length and intentionally excludes end caps and junction overlap corrections.
pub fn nominal_fiber_volume(assembly: &FiberAssembly, active_through: Option<u32>) -> f64 {
    assembly
        .topology
        .fibers
        .iter()
        .filter(|fiber| active_through.is_none_or(|step| fiber.formation_step <= step))
        .map(|fiber| {
            let area = match assembly.sections.entries[fiber.section.0 as usize] {
                Section::Circular { radius } => PI * radius * radius,
                Section::Elliptical { semi_axes } => PI * semi_axes[0] * semi_axes[1],
            };
            let start = fiber.vertices.start as usize;
            let end = start + fiber.vertices.len as usize;
            let length = assembly.geometry.intrinsic.positions[start..end]
                .windows(2)
                .map(|pair| {
                    let delta = [
                        pair[1][0] - pair[0][0],
                        pair[1][1] - pair[0][1],
                        pair[1][2] - pair[0][2],
                    ];
                    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
                })
                .sum::<f64>();
            area * length
        })
        .sum()
}

pub(crate) fn validate_compaction(config: &CompactionConfig) {
    match config.target {
        CompactionTarget::NominalVolumeFraction(value) => assert!(value > 0.0 && value < 1.0),
        CompactionTarget::CellVolume(value) => assert!(value > 0.0 && value.is_finite()),
        CompactionTarget::CellLengths(values) => {
            assert!(values.iter().all(|value| *value > 0.0 && value.is_finite()))
        }
        CompactionTarget::MeanPressure(value) | CompactionTarget::PenaltyEnergy(value) => {
            assert!(value >= 0.0 && value.is_finite())
        }
        CompactionTarget::DirectionalPressure(values) => {
            assert!(values.iter().any(|value| *value > 0.0));
            assert!(values
                .iter()
                .all(|value| *value >= 0.0 && value.is_finite()));
        }
    }
    match config.path {
        CompactionPath::AxisWeights(weights) => validate_weights(weights),
        CompactionPath::EqualPressure {
            active_axes,
            pressure_floor,
        } => {
            assert!(active_axes.iter().any(|active| *active));
            assert!(pressure_floor > 0.0 && pressure_floor.is_finite());
        }
        CompactionPath::StressRatio {
            ratio,
            pressure_floor,
        } => {
            validate_weights(ratio);
            assert!(pressure_floor > 0.0 && pressure_floor.is_finite());
        }
        CompactionPath::MinimumIncrementalWork { active_axes } => {
            assert!(active_axes.iter().any(|active| *active));
        }
    }
    let increment = config.increment;
    assert!(increment.minimum_log_strain > 0.0);
    assert!(increment.initial_log_strain >= increment.minimum_log_strain);
    assert!(increment.maximum_log_strain >= increment.initial_log_strain);
    assert!(increment.growth_factor >= 1.0);
    assert!(increment.shrink_factor > 0.0 && increment.shrink_factor < 1.0);
    assert!(increment.relax_iterations > 0);
    assert!(
        increment
            .maximum_shortening_over_minimum_diameter
            .is_finite()
            && increment.maximum_shortening_over_minimum_diameter > 0.0
    );
    assert!(config.guards.maximum_penetration >= 0.0);
    assert!(config.guards.maximum_bend_ratio >= 1.0);
    assert!(config.guards.maximum_pressure >= 0.0);
    assert!(config.guards.maximum_penalty_energy >= 0.0);
    assert!(config.guards.maximum_steps > 0);
    assert!(config.guards.maximum_relax_windows > 0);
    assert!(config.energy_model.contact_stiffness >= 0.0);
    assert!(config.energy_model.stretch_stiffness >= 0.0);
    assert!(config.energy_model.bending_stiffness >= 0.0);
    assert!(config
        .cell_anchor
        .iter()
        .all(|anchor| anchor.is_finite() && (0.0..=1.0).contains(anchor)));
    assert!(config.face_pressure_floor > 0.0 && config.face_pressure_floor.is_finite());
    assert!((0.0..=1.0).contains(&config.face_balance_strength));
    assert!(config.target_tolerance >= 0.0 && config.target_tolerance < 1.0);
}

fn validate_weights(weights: [f32; 3]) {
    assert!(weights
        .iter()
        .all(|weight| *weight >= 0.0 && weight.is_finite()));
    assert!(weights.iter().any(|weight| *weight > 0.0));
}

pub(crate) fn axis_weights(path: CompactionPath, metrics: CompactionMetrics) -> [f32; 3] {
    let raw = match path {
        CompactionPath::AxisWeights(weights) => weights,
        CompactionPath::EqualPressure {
            active_axes,
            pressure_floor,
        } => std::array::from_fn(|axis| {
            if active_axes[axis] {
                1.0 / (metrics.pressure[axis] + pressure_floor)
            } else {
                0.0
            }
        }),
        CompactionPath::StressRatio {
            ratio,
            pressure_floor,
        } => std::array::from_fn(|axis| {
            if ratio[axis] > 0.0 {
                ratio[axis] / (metrics.pressure[axis] + pressure_floor)
            } else {
                0.0
            }
        }),
        CompactionPath::MinimumIncrementalWork { active_axes } => {
            let selected = (0..3)
                .filter(|axis| active_axes[*axis])
                .min_by(|first, second| {
                    metrics.pressure[*first].total_cmp(&metrics.pressure[*second])
                })
                .unwrap();
            std::array::from_fn(|axis| f32::from(axis == selected))
        }
    };
    let sum = raw.iter().sum::<f32>();
    raw.map(|weight| weight / sum)
}

pub(crate) fn diameter_limited_log_strain(
    requested: f32,
    weights: [f32; 3],
    lengths: [f64; 3],
    minimum_diameter: f64,
    maximum_shortening_over_minimum_diameter: f32,
) -> f32 {
    let maximum_shortening = minimum_diameter * maximum_shortening_over_minimum_diameter as f64;
    let mut limited = requested;
    for axis in 0..3 {
        if weights[axis] > 0.0 {
            let retained = (1.0 - maximum_shortening / lengths[axis]).clamp(f64::MIN_POSITIVE, 1.0);
            limited = limited.min((-retained.ln() / weights[axis] as f64) as f32);
        }
    }
    limited.max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tangle_core::{FiberId, PeriodicCell};

    #[test]
    fn nominal_volume_uses_intrinsic_arc_length_and_section_area() {
        let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([2.0; 3], [false; 3]));
        let material = assembly.materials.add("fiber");
        let section = assembly.sections.add(Section::Circular { radius: 0.5 });
        assembly
            .add_fiber(
                FiberId(1),
                material,
                section,
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                &[[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [1.0, 1.0, 0.5]],
            )
            .unwrap();
        assert!((nominal_fiber_volume(&assembly, None) - 0.5 * PI).abs() < 1.0e-12);
    }

    #[test]
    fn minimum_work_selects_the_softest_active_axis() {
        let weights = axis_weights(
            CompactionPath::MinimumIncrementalWork {
                active_axes: [true, false, true],
            },
            CompactionMetrics {
                pressure: [2.0, 0.0, 1.0],
                ..CompactionMetrics::default()
            },
        );
        assert_eq!(weights, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn diameter_limit_caps_cell_shortening() {
        let length = 1.0e-3;
        let strain = diameter_limited_log_strain(0.01, [0.0, 0.0, 1.0], [length; 3], 7.0e-6, 0.25);
        let shortening = length * (1.0 - (-(strain as f64)).exp());
        assert!((shortening - 1.75e-6).abs() < 1.0e-10);
    }
}
