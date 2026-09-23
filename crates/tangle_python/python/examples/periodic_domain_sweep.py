"""Periodic domain-size sweep: layered stacks of one fiber type for DIRT.

Every configuration stacks the same straight fibers (length 10, one material)
in a cell that is periodic in x and y, then relaxes, compacts, and polishes it
for a DIRT through-thickness tension test. Only the in-plane side of the
periodic cell changes, from 10 (one fiber length) down to 1 (a tenth of a
fiber length). Areal fiber density, layer count, and final volume fraction are
held fixed, so every specimen has the same nominal thickness and the fiber
count scales with the cell area: 200 fibers (5,000 segments) at side 10 and
2 fibers at side 1. On small cells each fiber wraps the periodic cell several
times and meets its own image; that interaction is the domain effect under
study, not an error.

Fibers use 25 segments (length 0.4) wherever the cell allows it. A segment
plus one diameter must stay below half the cell side, or a segment could touch
two periodic images of the same neighbor and the minimum-image contact (and a
DIRT bond or contact) becomes ambiguous; the smallest cell therefore refines
to 50 segments per fiber. Lengths are unitless; scale every length setting
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
DIAMETER = 0.25
MAX_SEGMENT_LENGTH = 0.4
# A segment plus one diameter must fit within this fraction of the cell side.
IMAGE_CLEARANCE = 0.45
FIBERS_PER_AREA = 2.0
LAYER_COUNT = 5
VOLUME_FRACTION = 0.20
DOMAIN_SIDES = tuple(range(10, 0, -1))
SEED = 20_260_923
# In fiber diameters: starting spacing of the deposited layers, the gap each
# layer is lowered to, the convergence tolerance used during deposition and
# compaction, and the final and DEM-handoff penetration targets.
STAGING_SPACING = 4.0
LAYER_GAP = 0.5
FORMATION_TOLERANCE = 0.02
CONTACT_TOLERANCE = 0.01
DEM_CONTACT_TOLERANCE = 0.002
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
        return math.ceil(self.fiber_length / allowed - 1.0e-9)

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
    def staging_thickness(self) -> float:
        return (self.layer_count + 1) * STAGING_SPACING * self.diameter + self.target_thickness

    @property
    def name(self) -> str:
        return f"side_{self.side:05.2f}".replace(".", "p").replace("p00", "")

    def validate(self) -> None:
        if min(self.side, self.fiber_length, self.diameter) <= 0.0:
            raise ValueError("side, fiber length, and diameter must be positive")
        if self.layer_count < 1:
            raise ValueError("layer_count must be at least 1")
        if not 0.0 < self.volume_fraction < 1.0:
            raise ValueError("volume_fraction must lie between 0 and 1")
        self.segments_per_fiber


def material(config: SweepConfig) -> tangle.Material:
    return tangle.Material("fiber", diameter=config.diameter)


def layer_assignment(config: SweepConfig, rng: random.Random) -> list[int]:
    """Spread fibers over the layers as evenly as the count allows."""
    layers = [index % config.layer_count for index in range(config.fiber_count)]
    rng.shuffle(layers)
    return layers


def straight_layer_fibers(
    config: SweepConfig, layer: int, count: int, rng: random.Random
) -> tangle.FiberCollection:
    """Straight in-plane fibers at random angle and position in one layer.

    Fibers keep their full length even when it exceeds the cell; the
    centerlines cross the periodic faces as many times as they need.
    """
    fiber = material(config)
    collection = tangle.FiberCollection(f"layer {layer}")
    z = (layer + 1) * STAGING_SPACING * config.diameter
    step = config.segment_length
    for _ in range(count):
        angle = rng.uniform(0.0, math.pi)
        direction = (math.cos(angle), math.sin(angle))
        center = (rng.uniform(0.0, config.side), rng.uniform(0.0, config.side))
        # A tiny tilt keeps self-crossings on the smallest cells from sharing
        # one height exactly.
        tilt = rng.uniform(-0.01, 0.01) * config.diameter / config.fiber_length
        start = -0.5 * config.fiber_length
        centerline = [
            [
                center[0] + (start + index * step) * direction[0],
                center[1] + (start + index * step) * direction[1],
                z + (start + index * step) * tilt,
            ]
            for index in range(config.segments_per_fiber + 1)
        ]
        collection.add_fiber(centerline, fiber, formation_layer=layer)
    return collection


def compaction(config: SweepConfig) -> tangle.CompactionSettings:
    return tangle.CompactionSettings.volume_fraction(
        config.volume_fraction,
        kinematics="moving_walls",
        cell_anchor=[0.0, 0.0, 0.0],
        initial_log_strain=0.02,
        min_log_strain=0.002,
        max_log_strain=0.05,
        growth_factor=1.2,
        shrink_factor=0.5,
        relax_iterations=1_000,
        max_shortening_over_min_diameter=0.5,
        max_penetration=5 * CONTACT_TOLERANCE * config.diameter,
        max_curvature_ratio=1.0e6,
        max_steps=512,
        max_relax_windows=16,
    )


def build(
    side: float = 10.0, **changes
) -> tuple[tangle.Recipe, tangle.RelaxationSettings, SweepConfig]:
    """Recipe, settings, and configuration for one cell side."""
    config = SweepConfig(side=side, **changes)
    config.validate()
    rng = random.Random(config.seed * 1_000 + round(100 * side))
    cell = tangle.Cell(
        [config.side, config.side, config.staging_thickness], periodic="xy"
    )
    recipe = tangle.Recipe(cell)
    layers = layer_assignment(config, rng)
    placed = 0
    for layer in range(config.layer_count):
        count = layers.count(layer)
        if count == 0:
            continue
        recipe.insert(straight_layer_fibers(config, layer, count, rng))
        recipe.relax_for(200)
        if placed > 0:
            with recipe.place_layer_above(
                layer,
                gap=LAYER_GAP * config.diameter,
                stiffness=1.0,
                max_translation=0.05 * config.diameter,
            ):
                recipe.settle_targets(
                    tolerance=0.05 * config.diameter, max_iterations=6_000
                )
            recipe.relax_for(200)
        placed += 1
    # Compaction first waits for a relaxed baseline; fixed-length deposition
    # relaxes do not guarantee one, and without it compaction stops at once.
    recipe.solve(
        tangle.SolvePolicy(
            "baseline/contact",
            target_penetration=FORMATION_TOLERANCE * config.diameter,
            max_penetration=5 * CONTACT_TOLERANCE * config.diameter,
            max_iterations=30_000,
            on_budget_exhausted="continue_if_hard_ok",
        )
    )
    recipe.compact(compaction(config))
    # Each stage tightens penetration but keeps going on a stalled budget as
    # long as its hard limit holds; the summary records what was reached.
    for name, target, limit, budget in [
        ("final/contact", CONTACT_TOLERANCE, 5 * CONTACT_TOLERANCE, 60_000),
        ("polish/dem-contact", DEM_CONTACT_TOLERANCE, CONTACT_TOLERANCE, 100_000),
    ]:
        recipe.solve(
            tangle.SolvePolicy(
                name,
                target_penetration=target * config.diameter,
                max_penetration=limit * config.diameter,
                max_iterations=budget,
                on_budget_exhausted="continue_if_hard_ok",
            ),
            tangle.RelaxationOverrides.preset("contact_cleanup"),
        )

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
        seed=args.seed,
    )
