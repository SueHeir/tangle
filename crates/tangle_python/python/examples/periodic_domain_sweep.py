"""Periodic domain-size sweep: layered stacks of one fiber type for DIRT.

Every configuration stacks the same wavy fibers (length 10, one material) in
a cell that is periodic in x and y, then relaxes, compacts, and polishes it
for a DIRT through-thickness tension test. Only the in-plane side of the
periodic cell changes, from 10 (one fiber length) down to 1 (a tenth of a
fiber length). Areal fiber density, layer count, and final volume fraction are
held fixed, so every specimen has the same nominal thickness and the fiber
count scales with the cell area: 200 fibers (5,000 segments) at side 10 and
2 fibers at side 1. On small cells each fiber wraps the periodic cell several
times and meets its own image; that interaction is the domain effect under
study, not an error.

Fibers (diameter 0.2) use 25 segments (length 0.4) wherever the cell allows
it. A segment plus one diameter must stay below half the cell side, or a
segment could touch two periodic images of the same neighbor and the
minimum-image contact (and a DIRT bond or contact) becomes ambiguous; the
smallest cell therefore refines to 40 segments per fiber. Segments must also
be at least one diameter long: only adjacent segments of a fiber are excluded
from contact, so shorter segments make every next-nearest pair overlap.

Each fiber undulates out of plane (three waves along its length, 1.5
diameters in amplitude, so about one layer spacing peak to peak) with a random
phase, so neighboring layers nest into each other under compaction and carry
load through the thickness. Straight in-plane layers only touch by friction
and have almost no through-thickness strength.

A fiber that meets one of its own periodic images (every fiber on side
1, and fibers close to a lattice direction on the next few sides) would sit
on itself in one plane and cannot relax apart. Each such fiber is deposited
as a gentle straight ramp that rises a little more than one diameter between
the two strands that would touch, so they start separated through the
thickness; a ramp already crosses the layers, so it carries no wave.
Lengths are unitless; scale every length setting together to change units.

Each configuration writes ``side_XX/`` with the relaxed geometry (OVITO dump),
multi-material spherocylinder and bond data for DIRT
(``side_XX_capsules.data``), and a JSON summary. ``summary.csv`` collects one
row per configuration.
"""

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import tangle


OUTPUT = Path(__file__).parent / "output" / "periodic_domain_sweep"
FIBER_LENGTH = 10.0
DIAMETER = 0.2
MAX_SEGMENT_LENGTH = 0.4
# A segment plus one diameter must fit within this fraction of the cell side.
IMAGE_CLEARANCE = 0.45
FIBERS_PER_AREA = 2.0
LAYER_COUNT = 5
VOLUME_FRACTION = 0.20
DOMAIN_SIDES = tuple(range(10, 0, -1))
SEED = 20_260_923
# In fiber diameters: the clearance between staged layers, the gap between
# the top of one deposited layer and the bottom of the next (a flat layer of
# crossing fibers is about two diameters thick), the convergence tolerance
# used during deposition and compaction, and the final and DEM-handoff
# penetration targets.
STAGING_SPACING = 4.0
LAYER_GAP = 2.0
FORMATION_TOLERANCE = 0.02
CONTACT_TOLERANCE = 0.01
DEM_CONTACT_TOLERANCE = 0.002
# Rise between self-touching strands of a ramped fiber, in diameters.
RAMP_MARGIN = 1.2
# Out-of-plane waviness: amplitude in diameters and full waves per fiber.
WAVE_AMPLITUDE = 1.5
WAVES_PER_FIBER = 3.0
# In-plane distance, in diameters, below which two strands count as touching.
SELF_CONTACT_CLEARANCE = 1.1
DENSITY = 1_000.0


@dataclass(frozen=True)
class SweepConfig:
    """One domain-size configuration."""

    side: float
    fiber_length: float = FIBER_LENGTH
    diameter: float = DIAMETER
    max_segment_length: float = MAX_SEGMENT_LENGTH
    fibers_per_area: float = FIBERS_PER_AREA
    layer_count: int = LAYER_COUNT
    volume_fraction: float = VOLUME_FRACTION
    wave_amplitude: float = WAVE_AMPLITUDE
    waves_per_fiber: float = WAVES_PER_FIBER
    seed: int = SEED

    @property
    def fiber_count(self) -> int:
        return max(1, round(self.fibers_per_area * self.side**2))

    @property
    def segments_per_fiber(self) -> int:
        """Coarsest uniform segmentation the cell admits, capped by
        ``max_segment_length``."""
        allowed = min(
            self.max_segment_length, IMAGE_CLEARANCE * self.side - self.diameter
        )
        if allowed <= 0.0:
            raise ValueError(
                f"diameter {self.diameter:g} is too large for cell side {self.side:g}"
            )
        segments = math.ceil(self.fiber_length / allowed - 1.0e-9)
        if self.fiber_length / segments < self.diameter * (1.0 - 1.0e-9):
            raise ValueError(
                f"cell side {self.side:g} needs segments shorter than the "
                f"diameter {self.diameter:g}; use a thinner fiber or larger cell"
            )
        return segments

    @property
    def segment_count(self) -> int:
        return self.fiber_count * self.segments_per_fiber

    @property
    def segment_length(self) -> float:
        return self.fiber_length / self.segments_per_fiber

    @property
    def target_thickness(self) -> float:
        """Stack thickness at the target volume fraction."""
        fiber_volume = self.fiber_count * self.fiber_length * math.pi * self.diameter**2 / 4
        return fiber_volume / (self.volume_fraction * self.side**2)

    @property
    def name(self) -> str:
        return f"side_{self.side:05.2f}".replace(".", "p").replace("p00", "")

    def validate(self) -> None:
        if min(self.side, self.fiber_length, self.diameter, self.max_segment_length) <= 0.0:
            raise ValueError(
                "side, fiber length, diameter, and max segment length must be positive"
            )
        if self.layer_count < 1:
            raise ValueError("layer_count must be at least 1")
        if not 0.0 < self.volume_fraction < 1.0:
            raise ValueError("volume_fraction must lie between 0 and 1")
        if self.wave_amplitude < 0.0 or self.waves_per_fiber < 0.0:
            raise ValueError("wave_amplitude and waves_per_fiber must not be negative")
        self.segments_per_fiber


def material(config: SweepConfig) -> tangle.Material:
    return tangle.Material("fiber", diameter=config.diameter)


def layer_counts(config: SweepConfig) -> list[int]:
    """Fibers per layer, spread as evenly as the count allows.

    Cells with fewer fibers than layers get one fiber per layer.
    """
    layers = min(config.layer_count, config.fiber_count)
    base, extra = divmod(config.fiber_count, layers)
    return [base + (layer < extra) for layer in range(layers)]


def self_contact_arc(config: SweepConfig, angle: float) -> float | None:
    """Shortest distance along a flat fiber to a point touching its own image.

    A straight fiber in direction ``u`` meets its periodic image shifted by
    lattice vector ``n * side`` where the in-plane distance
    ``|n * side x u|`` drops below the contact clearance, at arc length
    ``n * side . u``. Returns ``None`` when no such point lies on the fiber.
    """
    ux, uy = math.cos(angle), math.sin(angle)
    reach = config.fiber_length + config.diameter
    limit = math.ceil(reach / config.side)
    clearance = SELF_CONTACT_CLEARANCE * config.diameter
    shortest = None
    for nx in range(-limit, limit + 1):
        for ny in range(-limit, limit + 1):
            if nx == ny == 0:
                continue
            arc = config.side * (nx * ux + ny * uy)
            offset = config.side * abs(nx * uy - ny * ux)
            if 0.0 < arc <= config.fiber_length and offset < clearance:
                shortest = arc if shortest is None else min(shortest, arc)
    return shortest


def ramp_slope(config: SweepConfig, angle: float) -> float:
    """Rise per unit in-plane length, or 0 when a flat fiber clears itself."""
    arc = self_contact_arc(config, angle)
    return 0.0 if arc is None else RAMP_MARGIN * config.diameter / arc


@dataclass(frozen=True)
class FiberPlacement:
    """In-plane pose, ramp, and out-of-plane wave of one fiber."""

    angle: float
    center: tuple[float, float]
    slope: float
    tilt: float
    amplitude: float = 0.0
    phase: float = 0.0

    def rise(self, config: SweepConfig) -> float:
        """Height between the lowest and highest point of the centerline."""
        ramp = config.fiber_length * self.slope / math.hypot(1.0, self.slope)
        return ramp + 2.0 * self.amplitude

    def span(self, config: SweepConfig) -> float:
        """In-plane extent of the fiber.

        A wave shortens the span; it is solved so the fiber carries exactly
        ``waves_per_fiber`` waves over its length.
        """
        if not self.amplitude:
            return config.fiber_length / math.hypot(1.0, self.slope + self.tilt)
        span = config.fiber_length
        samples = [math.cos(2.0 * math.pi * (i + 0.5) / 256) for i in range(256)]
        for _ in range(50):
            steepness = self.amplitude * 2.0 * math.pi * config.waves_per_fiber / span
            stretch = sum(math.hypot(1.0, steepness * c) for c in samples) / 256
            span = config.fiber_length / stretch
        return span

    def height(self, config: SweepConfig, x: float, span: float) -> float:
        """Height above the layer base at in-plane distance ``x`` from the
        start of a fiber that spans ``span`` in plane."""
        wavenumber = 2.0 * math.pi * config.waves_per_fiber / span
        return (
            (x - 0.5 * span) * self.tilt
            + x * self.slope
            + self.amplitude * (1.0 + math.sin(wavenumber * x + self.phase))
        )


def sample_layers(config: SweepConfig, rng: random.Random) -> list[list[FiberPlacement]]:
    """Random in-plane angle and position for every fiber, grouped by layer."""
    layers = []
    for count in layer_counts(config):
        placements = []
        for _ in range(count):
            angle = rng.uniform(0.0, math.pi)
            center = (rng.uniform(0.0, config.side), rng.uniform(0.0, config.side))
            # A tiny tilt keeps crossings on flat fibers from sharing one
            # height exactly.
            tilt = rng.uniform(-0.01, 0.01) * config.diameter / config.fiber_length
            phase = rng.uniform(0.0, 2.0 * math.pi)
            slope = ramp_slope(config, angle)
            amplitude = 0.0 if slope else config.wave_amplitude * config.diameter
            placements.append(
                FiberPlacement(angle, center, slope, tilt, amplitude, phase)
            )
        layers.append(placements)
    return layers


def layer_rise(config: SweepConfig, placements: list[FiberPlacement]) -> float:
    return max(placement.rise(config) for placement in placements)


def profile(config: SweepConfig, placement: FiberPlacement) -> list[tuple[float, float]]:
    """In-plane distance and height of every vertex, with every segment
    exactly ``segment_length`` long so the fiber keeps its full length.

    """
    span = placement.span(config)
    segment = config.segment_length
    points = [(0.0, placement.height(config, 0.0, span))]
    for _ in range(config.segments_per_fiber):
        x0, z0 = points[-1]

        def reach(x: float) -> float:
            return math.hypot(x - x0, placement.height(config, x, span) - z0)

        # The chord grows with the in-plane step for these gentle curves.
        low, high = 0.0, segment
        for _ in range(60):
            middle = 0.5 * (low + high)
            if reach(x0 + middle) < segment:
                low = middle
            else:
                high = middle
        x = x0 + 0.5 * (low + high)
        points.append((x, placement.height(config, x, span)))
    return points


def layer_fibers(
    config: SweepConfig,
    layer: int,
    placements: list[FiberPlacement],
    z: float,
) -> tangle.FiberCollection:
    """Fibers of one layer, lowest point near height ``z``.

    Fibers keep their full length even when it exceeds the cell; the
    centerlines cross the periodic faces as many times as they need. Wavy
    and ramped fibers keep that length along the curve.
    """
    fiber = material(config)
    collection = tangle.FiberCollection(f"layer {layer}")
    for placement in placements:
        points = profile(config, placement)
        middle = 0.5 * points[-1][0]
        direction = (math.cos(placement.angle), math.sin(placement.angle))
        centerline = [
            [
                placement.center[0] + (x - middle) * direction[0],
                placement.center[1] + (x - middle) * direction[1],
                z + height,
            ]
            for x, height in points
        ]
        collection.add_fiber(centerline, fiber, formation_layer=layer)
    return collection


def compaction(config: SweepConfig) -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        config.volume_fraction,
        kinematics="moving_walls",
        # Close both z walls so the stack is pressed from above and below
        # instead of being pushed against the top wall alone.
        cell_anchor=[0.0, 0.0, 0.5],
        balance_opposing_faces=True,
        face_balance_strength=0.5,
        initial_log_strain=0.02,
        min_log_strain=0.000_5,
        max_log_strain=0.05,
        growth_factor=1.2,
        shrink_factor=0.5,
        relax_iterations=2_000,
        max_shortening_over_min_diameter=0.5,
        max_penetration=5 * CONTACT_TOLERANCE * config.diameter,
        max_curvature_ratio=1.0e6,
        max_steps=512,
        max_relax_windows=16,
    )


def contact_stage(
    recipe: tangle.Recipe,
    config: SweepConfig,
    name: str,
    target: float,
    budget: int,
) -> None:
    """Solve contacts toward ``target`` fiber diameters of penetration.

    A policy finishes as soon as its max penetration holds, so the stage only
    accepts its own target. Penetration is a soft limit: a stage that stalls
    warns and continues, and the run summary records what was reached.
    """
    recipe.solve(
        tangle.SolvePolicy(
            name,
            target_penetration=target * config.diameter,
            hard_penetration=False,
            max_iterations=budget,
            on_budget_exhausted="continue_if_hard_ok",
        ),
        tangle.RelaxationOverrides.preset("contact_cleanup"),
    )


def build(
    side: float = 10.0, **changes
) -> tuple[tangle.Recipe, tangle.RelaxationSettings, SweepConfig]:
    """Recipe, settings, and configuration for one cell side."""
    config = SweepConfig(side=side, **changes)
    config.validate()
    rng = random.Random(f"{config.seed}:{config.side!r}")
    layers = sample_layers(config, rng)
    rises = [layer_rise(config, placements) for placements in layers]
    spacing = STAGING_SPACING * config.diameter
    gaps = [
        LAYER_GAP * config.diameter + 0.5 * (below + above)
        for below, above in zip(rises, rises[1:])
    ]
    # Staged stack height once every layer is lowered onto the one below.
    stack = config.diameter + 0.5 * (rises[0] + rises[-1]) + sum(gaps)
    bases = []
    top = 0.0
    for rise in rises:
        bases.append(top + spacing)
        top = bases[-1] + rise
    cell = tangle.Cell(
        [config.side, config.side, top + spacing + config.target_thickness],
        periodic="xy",
    )
    recipe = tangle.Recipe(cell)
    for layer, placements in enumerate(layers):
        recipe.insert(layer_fibers(config, layer, placements, bases[layer]))
        recipe.relax_for(200)
        if layer == 0:
            continue
        # The gap is between layer center planes; ramped and wavy layers are
        # centered half their rise above their base.
        with recipe.place_layer_above(
            layer,
            gap=gaps[layer - 1],
            stiffness=1.0,
            max_translation=0.05 * config.diameter,
        ):
            recipe.settle_targets(tolerance=0.05 * config.diameter, max_iterations=6_000)
        recipe.relax_for(200)
    # Compaction first waits for a relaxed baseline; fixed-length deposition
    # relaxes do not guarantee one, and without it compaction stops at once.
    contact_stage(recipe, config, "baseline/contact", FORMATION_TOLERANCE, 30_000)
    # Drop the empty staging headroom so both walls start on the stack; a
    # thin stack otherwise floats below the top wall with an empty bottom.
    # A stack thinner than the target keeps enough headroom for compaction
    # to finish at the target thickness.
    recipe.fit_cell_to_active_fibers(
        padding=max(0.05 * config.diameter, 0.5 * (config.target_thickness - stack))
    )
    # Once the layers touch, the default averaged correction cannot clear a
    # compaction increment within one window; use the contact stages' solver.
    recipe.compact(
        compaction(config), tangle.RelaxationOverrides.preset("contact_cleanup")
    )
    contact_stage(recipe, config, "final/contact", CONTACT_TOLERANCE, 60_000)
    contact_stage(recipe, config, "polish/dem-contact", DEM_CONTACT_TOLERANCE, 100_000)

    settings = tangle.RelaxationSettings(
        constraint_iterations=8,
        contact_aggregation="uniform_average",
        correction_fraction=0.5,
        penetration_tolerance=FORMATION_TOLERANCE * config.diameter,
        max_step=0.05 * config.diameter,
        max_iterations=500_000,
        iterations_per_batch=10,
        # Keep the exported discretization identical across the sweep.
        adaptive_segmentation=None,
    )
    return recipe, settings, config


def run(
    side: float = 10.0,
    *,
    output: Path = OUTPUT,
    backend: str | None = None,
    bpm_mode: str = "spherocylinders_exact",
    **changes,
) -> dict:
    """Generate, relax, and export one configuration; return its summary."""
    recipe, settings, config = build(side, **changes)
    if backend is not None:
        settings.backend = backend
    directory = output / config.name
    directory.mkdir(parents=True, exist_ok=True)
    result = recipe.run(settings)
    result.write_ovito(
        directory / f"{config.name}.dump",
        view_script_path=directory / f"{config.name}_view.py",
        coloring="curvature_ratio",
    )
    particles, bonds = result.export_bpm(
        directory / f"{config.name}_capsules.data",
        mode=bpm_mode,
        density=DENSITY,
    )
    analysis = result.characterize()
    neighbors = result.characterize_neighbors(0.05 * config.diameter)
    cell = result.assembly.cell
    compaction_event = next(
        (event for event in result.events if event.startswith("compact in")), ""
    )
    compacted = "TargetReached" in compaction_event
    if not compacted:
        print(f"warning: {config.name} did not reach the target volume fraction: "
              f"{compaction_event or 'no compaction event'}")
    summary = {
        **asdict(config),
        "name": config.name,
        "side_over_fiber_length": config.side / config.fiber_length,
        "fiber_count": config.fiber_count,
        "segments_per_fiber": config.segments_per_fiber,
        "segment_count": config.segment_count,
        "converged": result.converged,
        "iterations": result.iterations,
        "max_penetration": result.max_penetration,
        "dem_contact_target_met": result.max_penetration
        <= DEM_CONTACT_TOLERANCE * config.diameter * (1 + 1.0e-3),
        "cell_lengths": cell.lengths,
        "thickness": cell.lengths[2],
        "nominal_volume_fraction": analysis.nominal_swept_volume_fraction,
        "compacted_to_target": compacted,
        "compaction": compaction_event,
        "contacts_per_length": neighbors.contacts_per_length,
        "contact_ratio_to_random": neighbors.contact_ratio_to_random,
        "bpm_mode": bpm_mode,
        "bpm_particles": particles,
        "bpm_bonds": bonds,
        "warnings": result.warnings,
    }
    (directory / f"{config.name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


SUMMARY_COLUMNS = (
    "name",
    "side",
    "side_over_fiber_length",
    "fiber_count",
    "segments_per_fiber",
    "segment_count",
    "converged",
    "iterations",
    "max_penetration",
    "dem_contact_target_met",
    "thickness",
    "nominal_volume_fraction",
    "compacted_to_target",
    "contacts_per_length",
    "contact_ratio_to_random",
    "bpm_particles",
    "bpm_bonds",
)


def sweep(sides=DOMAIN_SIDES, *, output: Path = OUTPUT, **options) -> list[dict]:
    """Run every side in turn and write ``summary.csv``."""
    summaries = []
    output.mkdir(parents=True, exist_ok=True)
    for side in sides:
        summary = run(side, output=output, **options)
        summaries.append(summary)
        print(
            f"side {side:g}: {summary['fiber_count']} fibers, "
            f"converged={summary['converged']}, "
            f"thickness={summary['thickness']:.3f}, "
            f"max penetration={summary['max_penetration']:.2e}"
        )
        with open(output / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sides",
        type=float,
        nargs="+",
        default=list(DOMAIN_SIDES),
        help="periodic cell sides to run (default: 10 9 ... 1)",
    )
    parser.add_argument("--fiber-length", type=float, default=FIBER_LENGTH)
    parser.add_argument("--diameter", type=float, default=DIAMETER)
    parser.add_argument("--max-segment-length", type=float, default=MAX_SEGMENT_LENGTH)
    parser.add_argument("--fibers-per-area", type=float, default=FIBERS_PER_AREA)
    parser.add_argument("--layers", type=int, default=LAYER_COUNT)
    parser.add_argument("--volume-fraction", type=float, default=VOLUME_FRACTION)
    parser.add_argument(
        "--wave-amplitude",
        type=float,
        default=WAVE_AMPLITUDE,
        help="out-of-plane wave amplitude in diameters (0 for straight fibers)",
    )
    parser.add_argument("--waves-per-fiber", type=float, default=WAVES_PER_FIBER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--backend", choices=("wgpu", "cpu"), default=None)
    parser.add_argument(
        "--bpm-mode",
        default="spherocylinders_exact",
        help="export_bpm mode; spheres_exact writes bpm/sphere data instead",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sweep(
        args.sides,
        output=args.output,
        backend=args.backend,
        bpm_mode=args.bpm_mode,
        fiber_length=args.fiber_length,
        diameter=args.diameter,
        max_segment_length=args.max_segment_length,
        fibers_per_area=args.fibers_per_area,
        layer_count=args.layers,
        volume_fraction=args.volume_fraction,
        wave_amplitude=args.wave_amplitude,
        waves_per_fiber=args.waves_per_fiber,
        seed=args.seed,
    )
