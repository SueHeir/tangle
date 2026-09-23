"""Python interface to TANGLE fiber assembly and CubeCL formation recipes.

Configuration objects follow three naming conventions:

``*Settings``
    Run-wide configuration that applies to every step of a recipe, such as
    ``RelaxationSettings`` or ``CheckpointSettings``.
``*Policy``
    A named rule that one recipe step follows, such as the convergence
    targets in ``SolvePolicy`` or the capture rule in ``JunctionPolicy``.
``*Overrides``
    Temporary deltas that apply only to the step they are passed to, such as
    ``RelaxationOverrides.preset("curvature_cleanup")``.

Every settings, policy and overrides class accepts keyword arguments in its
constructor and has a ``replace(**changes)`` method that returns a modified
copy. The small option classes (orientations, positions, compaction targets
and paths, needle footprints) are immutable values. Lengths are
in meters; ``tangle.units`` has ``um`` and ``mm`` helpers.
"""

__version__ = "0.1.0"

from . import units
from ._tangle import (
    AdaptiveSegmentationSettings,
    AlignedOrientation,
    AnalysisReport,
    Assembly,
    AxisWeightsPath,
    Cell,
    CellLengthsTarget,
    CellVolumeTarget,
    CheckpointSettings,
    CircularFootprint,
    CompactionSettings,
    DensityGradientPosition,
    DirectionalPressureTarget,
    EqualPressurePath,
    FiberCollection,
    FiberPopulation,
    FiberSelection,
    HeldTargets,
    IsotropicOrientation,
    JunctionPolicy,
    LayeredBiaxialOrientation,
    LayeredPosition,
    Material,
    MeanPressureTarget,
    MinimumWorkPath,
    NeighborReport,
    PenaltyEnergyTarget,
    PlanarOrientation,
    PumaExportReport,
    RandomFiberFraction,
    Recipe,
    RecipeError,
    RelaxationOverrides,
    RelaxationSettings,
    RunResult,
    ShapeReport,
    SolvePolicy,
    StressRatioPath,
    UniformPosition,
    VolumeFractionTarget,
    generate_fiber_pair_crossing,
    generate_fiber_population,
    generate_multisegment_crossing,
    generate_point_crossing,
)

__all__ = [
    "AdaptiveSegmentationSettings",
    "AlignedOrientation",
    "AnalysisReport",
    "Assembly",
    "AxisWeightsPath",
    "Cell",
    "CellLengthsTarget",
    "CellVolumeTarget",
    "CheckpointSettings",
    "CircularFootprint",
    "CompactionSettings",
    "DensityGradientPosition",
    "DirectionalPressureTarget",
    "EqualPressurePath",
    "FiberCollection",
    "FiberPopulation",
    "FiberSelection",
    "HeldTargets",
    "IsotropicOrientation",
    "JunctionPolicy",
    "LayeredBiaxialOrientation",
    "LayeredPosition",
    "Material",
    "MeanPressureTarget",
    "MinimumWorkPath",
    "NeighborReport",
    "PenaltyEnergyTarget",
    "PlanarOrientation",
    "PumaExportReport",
    "RandomFiberFraction",
    "Recipe",
    "RecipeError",
    "RelaxationOverrides",
    "RelaxationSettings",
    "RunResult",
    "ShapeReport",
    "SolvePolicy",
    "StressRatioPath",
    "UniformPosition",
    "VolumeFractionTarget",
    "generate_fiber_pair_crossing",
    "generate_fiber_population",
    "generate_multisegment_crossing",
    "generate_point_crossing",
    "units",
    "__version__",
]
