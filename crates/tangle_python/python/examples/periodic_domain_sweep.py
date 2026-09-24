"""Periodic domain-size sweep: biased in-plane fiber stacks for DIRT.

Every configuration fills a cell that is periodic in x and y with the same
straight fibers (length 10, one material), then relaxes, compacts, and
polishes it for a DIRT through-thickness tension test. Only the in-plane side
of the periodic cell changes, from 10 (one fiber length) down to 2 (a fifth of
a fiber length); side 1 is available on request, but its two fibers never
touch. Areal fiber density and final volume fraction are held fixed, so every
specimen has the same nominal thickness and the fiber count scales with the
cell area: 200 fibers (5,000 segments) at side 10 and 8 at side 2. On small
cells each fiber wraps the periodic cell several times and meets its own
image; that interaction is the domain effect under study, not an error.

Fibers are placed all at once, not deposited layer by layer: each gets a
random in-plane direction, a random tilt of at most 10 degrees out of the
plane (the in-plane bias), and a random position anywhere in a cell twice the
final thickness. The stack is relaxed, trimmed to the fibers, and compacted
from both z walls to the target volume fraction. Tilted fibers cross the
thickness, so neighbors interlock instead of lying in separate layers.

Fibers (diameter 0.2) use 25 segments (length 0.4) wherever the cell allows
it. A segment plus one diameter must stay below half the cell side, or a
segment could touch two periodic images of the same neighbor and the
minimum-image contact (and a DIRT bond or contact) becomes ambiguous; the
side-1 cell therefore refines to 40 segments per fiber. Segments must also be
at least one diameter long: only adjacent segments of a fiber are excluded
from contact, so shorter segments make every next-nearest pair overlap.

A fiber that meets one of its own periodic images (every fiber on side 1, and
fibers close to a lattice direction on the next few sides) would lie on
itself if it were too flat and could not relax apart. Such a fiber is tilted
at least enough to rise a little more than one diameter between the two
strands that would touch. Lengths are unitless; scale every length setting
together to change units.

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
# Largest out-of-plane tilt, in degrees.
MAX_TILT = 10.0
VOLUME_FRACTION = 0.20
# Fibers are placed in a cell this dilute, then compacted to VOLUME_FRACTION.
PLACEMENT_VOLUME_FRACTION = 0.10
# Side 1 holds two fibers that never touch; run it with --sides 1 if needed.
DOMAIN_SIDES = tuple(range(10, 1, -1))
SEED = 20_260_923
# In fiber diameters: the convergence tolerance used during placement and
# compaction, and the final and DEM-handoff penetration targets.
FORMATION_TOLERANCE = 0.02
CONTACT_TOLERANCE = 0.01
DEM_CONTACT_TOLERANCE = 0.002
# Rise between self-touching strands of one fiber, in diameters.
SELF_RISE = 1.2
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
    max_tilt: float = MAX_TILT
    volume_fraction: float = VOLUME_FRACTION
    placement_volume_fraction: float = PLACEMENT_VOLUME_FRACTION
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

    def thickness(self, volume_fraction: float) -> float:
        """Stack thickness at ``volume_fraction``."""
        fiber_volume = self.fiber_count * self.fiber_length * math.pi * self.diameter**2 / 4
        return fiber_volume / (volume_fraction * self.side**2)

    @property
    def target_thickness(self) -> float:
        return self.thickness(self.volume_fraction)

    @property
    def placement_thickness(self) -> float:
        return self.thickness(self.placement_volume_fraction)

    @property
    def name(self) -> str:
        return f"side_{self.side:05.2f}".replace(".", "p").replace("p00", "")

    def validate(self) -> None:
        if min(self.side, self.fiber_length, self.diameter, self.max_segment_length) <= 0.0:
            raise ValueError(
                "side, fiber length, diameter, and max segment length must be positive"
            )
        if not 0.0 <= self.max_tilt < 90.0:
            raise ValueError("max_tilt must lie in [0, 90) degrees")
        if not 0.0 < self.placement_volume_fraction <= self.volume_fraction < 1.0:
            raise ValueError(
                "need 0 < placement_volume_fraction <= volume_fraction < 1"
            )
        self.segments_per_fiber


def material(config: SweepConfig) -> tangle.Material:
    return tangle.Material("fiber", diameter=config.diameter)


def self_contact_arc(config: SweepConfig, angle: float) -> float | None:
    """Shortest in-plane distance along a fiber to a point touching its own
    image.

    A fiber heading in in-plane direction ``u`` meets its periodic image
    shifted by lattice vector ``n * side`` where the in-plane distance
    ``|n * side x u|`` drops below the contact clearance, at in-plane distance
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


def min_tilt(config: SweepConfig, angle: float) -> float:
    """Smallest tilt, in radians, that lifts a fiber clear of its own image."""
    arc = self_contact_arc(config, angle)
    return 0.0 if arc is None else math.atan(SELF_RISE * config.diameter / arc)


@dataclass(frozen=True)
class FiberPlacement:
    """Center, in-plane direction, and signed out-of-plane tilt of one fiber."""

    angle: float
    tilt: float
    center: tuple[float, float, float]

    def direction(self) -> tuple[float, float, float]:
        flat = math.cos(self.tilt)
        return (
            flat * math.cos(self.angle),
            flat * math.sin(self.angle),
            math.sin(self.tilt),
        )

    def rise(self, config: SweepConfig) -> float:
        return config.fiber_length * abs(math.sin(self.tilt))


def sample_fibers(config: SweepConfig, rng: random.Random) -> list[FiberPlacement]:
    """Random direction, tilt, and position for every fiber.

    Tilts are uniform within ``max_tilt`` of the plane, raised where needed so
    no fiber lies on its own periodic image. Centers are uniform in plane and
    through the placement thickness, with every fiber inside the z walls.
    """
    height = config.placement_thickness
    placements = []
    for _ in range(config.fiber_count):
        angle = rng.uniform(0.0, math.pi)
        tilt = math.radians(rng.uniform(-config.max_tilt, config.max_tilt))
        floor = min_tilt(config, angle)
        if abs(tilt) < floor:
            tilt = math.copysign(floor, tilt)
        placement = FiberPlacement(angle, tilt, (0.0, 0.0, 0.0))
        margin = 0.5 * placement.rise(config) + 0.5 * config.diameter
        if 2.0 * margin > height:
            raise ValueError(
                f"a fiber tilted {math.degrees(tilt):.1f} degrees does not fit "
                f"a {height:.3g}-thick placement cell; lower max_tilt or "
                "placement_volume_fraction"
            )
        center = (
            rng.uniform(0.0, config.side),
            rng.uniform(0.0, config.side),
            rng.uniform(margin, height - margin),
        )
        placements.append(FiberPlacement(angle, tilt, center))
    return placements


def fiber_collection(
    config: SweepConfig, placements: list[FiberPlacement]
) -> tangle.FiberCollection:
    """Straight fibers along each placement.

    Fibers keep their full length even when it exceeds the cell; the
    centerlines cross the periodic faces as many times as they need.
    """
    fiber = material(config)
    collection = tangle.FiberCollection("fibers")
    for placement in placements:
        direction = placement.direction()
        centerline = [
            [
                placement.center[axis]
                + (index * config.segment_length - 0.5 * config.fiber_length)
                * direction[axis]
                for axis in range(3)
            ]
            for index in range(config.segments_per_fiber + 1)
        ]
        collection.add_fiber(centerline, fiber)
    return collection


def compaction(config: SweepConfig) -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        config.volume_fraction,
        kinematics="moving_walls",
        # Close both z walls so the stack is pressed from above and below.
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
    placements = sample_fibers(config, rng)
    cell = tangle.Cell(
        [config.side, config.side, config.placement_thickness], periodic="xy"
    )
    recipe = tangle.Recipe(cell)
    recipe.insert(fiber_collection(config, placements))
    # Randomly placed fibers overlap; separate them before compaction, which
    # waits for a relaxed baseline and otherwise stops at once.
    contact_stage(recipe, config, "placement/contact", FORMATION_TOLERANCE, 60_000)
    # Trim the empty space above and below the stack so both walls start on
    # the fibers.
    recipe.fit_cell_to_active_fibers(padding=0.05 * config.diameter)
    # Once the fibers touch, the default averaged correction cannot clear a
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
        help="periodic cell sides to run (default: 10 9 ... 2)",
    )
    parser.add_argument("--fiber-length", type=float, default=FIBER_LENGTH)
    parser.add_argument("--diameter", type=float, default=DIAMETER)
    parser.add_argument("--max-segment-length", type=float, default=MAX_SEGMENT_LENGTH)
    parser.add_argument("--fibers-per-area", type=float, default=FIBERS_PER_AREA)
    parser.add_argument(
        "--max-tilt", type=float, default=MAX_TILT, help="degrees out of plane"
    )
    parser.add_argument("--volume-fraction", type=float, default=VOLUME_FRACTION)
    parser.add_argument(
        "--placement-volume-fraction", type=float, default=PLACEMENT_VOLUME_FRACTION
    )
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
        max_tilt=args.max_tilt,
        volume_fraction=args.volume_fraction,
        placement_volume_fraction=args.placement_volume_fraction,
        seed=args.seed,
    )
