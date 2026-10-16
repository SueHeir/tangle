"""LAMMPS input scripts for a bonded-particle file of capsules (``atom_style bacillus``): ``in.atom_types``, which
sets up every fiber type's contacts and bonds, and ``in.tensile``, an example that pulls the fibers apart.

    >>> relaxed.export_bpm(run, "relaxed/fibers_bpm.data")   # also writes relaxed/in.atom_types and in.tensile
    >>> ct.write_bpm_lammps_inputs("relaxed/fibers_bpm.data")  # the same, for a file written some other way

Then, in that folder: ``lmp -in in.tensile``, or ``include in.atom_types`` after ``read_data`` in a script of your own.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._bpm import _DataFile, _Particles

ATOM_TYPES = "in.atom_types"
TENSILE = "in.tensile"
_FIBER_TYPE = re.compile(r'^# atom type (\d+) and bond type (\d+) = fiber type "(.*)" \((\d+) capsules\)')
_BOND_TYPE = re.compile(r"^# bond type (\d+) = (.*)$")


@dataclass(frozen=True)
class _FiberType:
    atom_type: int
    bond_type: int
    name: str
    label: str
    count: int
    diameter: float  # median, meters
    density: float  # median, kg/m^3
    shortest: float  # shortest capsule, pole to pole (meters)


def write_bpm_lammps_inputs(path: str | os.PathLike, directory: str | os.PathLike | None = None) -> tuple[Path, Path]:
    """Write ``in.atom_types`` and ``in.tensile`` for a capsule file that ``export_bpm`` wrote (``atom_style
    bacillus``), into ``directory`` (default: the data file's folder). ``FitResult.export_bpm`` calls it, and
    :func:`add_bpm_binder_bonds` calls it again when it adds a bond type to a file that has them.

    ``in.atom_types`` is included after ``read_data``. It sets what the bonded-particle model needs
    (``special_bonds``, ``comm_modify``, ``neighbor``), groups and type labels named after the fiber types, the
    contacts (``pair_style granular/bacillus``) and bonds (``bond_style bpm/guo``) of each type, and ``dt``, a stable
    timestep. Each material value is an ``index`` variable with a placeholder default (Young's modulus ``E_1``,
    Poisson's ratio ``nu_1``, ...): define it before the include, or run with ``-var E_1 3.0e9``, to set it without
    editing the file.

    ``in.tensile`` reads the data file, includes ``in.atom_types``, holds the fibers near each end of the sample
    along ``axis`` (default x) and pulls one end away at a constant speed until the strain reaches ``max_strain``,
    writing the strain, stress and grip forces to ``tensile.txt``. Each file is overwritten; a folder holds the
    inputs for one data file. Returns both paths.
    """
    path = Path(path)
    directory = Path(directory) if directory is not None else path.parent
    data = _DataFile(path)
    if "Bacilli" not in data.sections:
        raise ValueError(f"{path} is not a capsule file for atom_style bacillus; in.atom_types and in.tensile are "
                         "written for those only")
    particles = _Particles.read(data)
    types = _fiber_types(data, particles)
    directory.mkdir(parents=True, exist_ok=True)
    atom_types = directory / ATOM_TYPES
    tensile = directory / TENSILE
    atom_types.write_text(_atom_types_script(path, data, particles, types))
    tensile.write_text(_tensile_script(path, directory))
    return atom_types, tensile


def _refresh(path: Path) -> None:
    """Write the inputs again if the data file's folder has an ``in.atom_types`` for this file."""
    existing = path.parent / ATOM_TYPES
    if existing.exists() and f"of {path.name} " in existing.read_text().splitlines()[0]:
        write_bpm_lammps_inputs(path)


def _label(name: str, taken: set[str]) -> str:
    """A LAMMPS type label (no spaces, not starting with a digit) for a fiber type's name, unique in ``taken``."""
    label = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "fiber"
    if label[0].isdigit():
        label = "fiber_" + label
    unique, k = label, 2
    while unique in taken:
        unique, k = f"{label}_{k}", k + 1
    taken.add(unique)
    return unique


def _fiber_types(data: _DataFile, particles: _Particles) -> list[_FiberType]:
    """The file's fiber types from its header comments, or one per atom type when it has none."""
    named = [match.groups() for match in map(_FIBER_TYPE.match, data.header) if match]
    if not named:
        named = [(str(kind), str(kind), f"type {kind}", "0") for kind in range(1, data.count("atom types") + 1)]
    types, taken = [], set()
    for atom_type, bond_type, name, _ in named:
        mine = particles.types == int(atom_type)
        if not mine.any():
            continue
        diameters = 2.0 * particles.radii[mine]
        lengths = 2.0 * particles.half[mine]
        types.append(_FiberType(int(atom_type), int(bond_type), name, _label(name, taken), int(mine.sum()),
                                float(np.median(diameters)), float(np.median(particles.densities[mine])),
                                float(lengths.min())))
    if not types:
        raise ValueError(f"{data.path} has no capsules")
    return types


def _g(value: float) -> str:
    return f"{value:.6g}"


def _min(a: str, b: str) -> str:
    """The smaller of two LAMMPS formulas (min() in a formula reduces a compute or fix, not two numbers)."""
    return f"ternary(({a})<({b}),{a},{b})"


def _atom_types_script(path: Path, data: _DataFile, particles: _Particles, types: list[_FiberType]) -> str:
    atom_count, bond_count = max(data.count("atom types"), 1), data.count("bond types")
    index_of = {atom: index for index, atom in enumerate(particles.ids.tolist())}
    bonds = data.rows("Bonds")
    longest = max((float(np.linalg.norm(particles.centers[index_of[int(row[2])]] -
                                        particles.centers[index_of[int(row[3])]])) for row in bonds), default=0.0)
    reach = float(np.max(2.0 * particles.half + 2.0 * particles.radii))
    thinnest = min(fiber.diameter for fiber in types)
    first = types[0].atom_type
    by_atom = {fiber.atom_type: fiber for fiber in types}
    by_bond = {fiber.bond_type: fiber for fiber in types}
    notes = {int(match.group(1)): match.group(2) for match in map(_BOND_TYPE.match, data.header) if match}

    out = [
        f"# LAMMPS set-up of the fiber types of {path.name} (atom_style bacillus, SI units), written by Tangle.",
        "# Include it right after read_data:",
        f"#   read_data {path.name}",
        f"#   include {ATOM_TYPES}",
        "# It sets what the bonded-particle model needs, a group and a type label per fiber type, each type's contacts",
        "# (pair_style granular/bacillus) and bonds (bond_style bpm/guo), and dt, a stable timestep.",
        "#",
        "# The material values below are placeholders. Each is an index variable, so a value defined before the",
        "# include, or given on the command line (lmp -in in.tensile -var E_1 3.0e9), takes its place.",
        "#   E_<type>, nu_<type>    Young's modulus (Pa) and Poisson's ratio of the fiber; G = E / 2(1 + nu)",
        "#   zeta_<type>            damping ratio of the bonds (stretch, shear, bend and twist)",
        "#   sigma_max_<type>,      tensile and shear strength (Pa) at which a bond breaks (bpm/guo's guo model:",
        "#   tau_max_<type>         normal plus bending stress, shear plus twist stress); 0 = never breaks",
        "#   kn_<type>, mu_<type>   contact stiffness (N/m; default E times the diameter) and friction coefficient",
        "",
        "# bpm/guo needs these, whatever the script: no contact between bonded neighbors, ghost velocities, and",
        "# ghosts reaching across the longest bond.",
        "special_bonds   lj 0.0 1.0 1.0 coul 1.0 1.0 1.0",
        f"comm_modify     vel yes cutoff {_g(1.25 * max(longest, reach))}",
        f"neighbor        {_g(0.25 * thinnest)} bin",
        "",
    ]
    for fiber in types:
        out += [
            f"# atom type {fiber.atom_type} and bond type {fiber.bond_type} = fiber type \"{fiber.name}\": "
            f"{fiber.count} capsules, {_g(fiber.diameter)} m across, density {_g(fiber.density)} kg/m^3, "
            f"shortest {_g(fiber.shortest)} m",
            f"labelmap        atom {fiber.atom_type} {fiber.label}",
            f"labelmap        bond {fiber.bond_type} {fiber.label}",
            f"group           {fiber.label} type {fiber.atom_type}",
        ]
        k = fiber.atom_type
        out += [
            f"variable        E_{k} index 1.0e9",
            f"variable        nu_{k} index 0.3",
            f"variable        zeta_{k} index 0.1",
            f"variable        sigma_max_{k} index 0",
            f"variable        tau_max_{k} index 0",
            f"variable        kn_{k} index $(v_E_{k}*{_g(fiber.diameter)})",
            f"variable        mu_{k} index 0.5",
            "",
        ]
    out += [
        "# A timestep well inside the fastest bond and contact: a fiftieth of a sound wave's crossing of the",
        "# shortest capsule (or diameter), and a tenth of the stiffest contact's period over 2 pi.",
    ]
    limits = []
    for fiber in types:
        k = fiber.atom_type
        mass = fiber.density * np.pi * (0.25 * fiber.diameter ** 2 * fiber.shortest + fiber.diameter ** 3 / 6.0)
        wave = f"0.02*{_g(min(fiber.shortest, fiber.diameter))}/sqrt(v_E_{k}/{_g(fiber.density)})"
        out.append(f"variable        dt_{k} equal {_min(wave, f'0.1*sqrt({_g(mass)}/v_kn_{k})')}")
        limits.append(f"v_dt_{k}")
    dt = limits[-1]
    for limit in reversed(limits[:-1]):
        dt = _min(limit, dt)
    out.append(f"variable        dt equal {dt}")
    out.append("# the slowest sound speed, for scripts that time things by wave crossings")
    speeds = [f"sqrt(v_E_{fiber.atom_type}/{_g(fiber.density)})" for fiber in types]
    slowest = speeds[-1]
    for speed in reversed(speeds[:-1]):
        slowest = _min(speed, slowest)
    out += [f"variable        sound_speed equal {slowest}", ""]

    out += ["pair_style      granular/bacillus"]
    for kind in range(1, atom_count + 1):
        k = by_atom[kind].atom_type if kind in by_atom else first
        note = "" if kind in by_atom else f"   # not a fiber type: fiber type {first}'s contacts"
        out.append(f"pair_coeff      {kind} {kind} hooke ${{kn_{k}}} 0.0 tangential linear_history "
                   f"$(v_kn_{k}/2) 0.0 ${{mu_{k}}}{note}")
    out.append("# pairs of types mix their contact values")
    out.append("")
    if bond_count:
        out.append("bond_style      bpm/guo")
        for kind in range(1, bond_count + 1):
            if kind in by_bond:
                k = str(by_bond[kind].atom_type)
            else:
                k = f"bond_{kind}"
                out += [
                    f"# bond type {kind} = {notes.get(kind, 'not a fiber type')}",
                    f"variable        E_{k} index $(v_E_{first})",
                    f"variable        nu_{k} index $(v_nu_{first})",
                    f"variable        zeta_{k} index $(v_zeta_{first})",
                    f"variable        sigma_max_{k} index 0",
                    f"variable        tau_max_{k} index 0",
                ]
            moduli = (f"${{E_{k}}} $(v_E_{k}/(2*(1+v_nu_{k}))) ${{E_{k}}} $(v_E_{k}/(2*(1+v_nu_{k}))) "
                      f"${{zeta_{k}}} ${{zeta_{k}}} ${{zeta_{k}}} ${{zeta_{k}}}")
            out.append(f"bond_coeff      {kind} {moduli}")
            # A zero strength never breaks: as a huge one.
            out.append(f"if \"${{sigma_max_{k}}} > 0 || ${{tau_max_{k}}} > 0\" then &\n"
                       f"  \"bond_coeff      {kind} {moduli} break_model guo "
                       f"$(v_sigma_max_{k}+(v_sigma_max_{k}<=0)*1.0e300) $(v_tau_max_{k}+(v_tau_max_{k}<=0)*1.0e300)\"")
        out.append("")
    return "\n".join(out)


def _tensile_script(path: Path, directory: Path) -> str:
    data = os.path.relpath(path, directory)
    axes = {"x": ("ly*lz", "0", "INF ${grip_lo} INF INF INF INF", "${grip_hi} INF INF INF INF INF"),
            "y": ("lx*lz", "1", "INF INF INF ${grip_lo} INF INF", "INF INF ${grip_hi} INF INF INF"),
            "z": ("lx*ly", "2", "INF INF INF INF INF ${grip_lo}", "INF INF INF INF ${grip_hi} INF")}
    out = [
        f"# Tensile pull of {data} along an axis (atom_style bacillus, SI units), written by Tangle. In this folder:",
        f"#   lmp -in {TENSILE}",
        f"#   lmp -in {TENSILE} -var axis z -var max_strain 0.05 -var E_1 3.0e9",
        "# The capsules whose centers are within grip of either end of the fibers' extent along the axis are the",
        "# grips: they don't rotate, the low one stays put and the high one moves away at a constant speed. The rest",
        "# move freely, with a little viscous damping. Every `every` steps, tensile.txt gets the strain (grip",
        "# separation over the length between the innermost held capsules' centers), the stress (the mean of the",
        "# two grip forces over the box's cross-section), both grip forces and the number of bonds left. First the",
        "# fibers settle for `settle` sound crossings with both grips held, so leftover overlaps relax before the",
        f"# pull. The material values are in {ATOM_TYPES}: set them there, or with -var.",
        "",
        "variable        axis index x          # x, y or z",
        "variable        max_strain index 0.02",
        "variable        grip index 0.1        # share of the fibers' extent held at each end",
        "variable        transits index 30     # pull speed: sound crossings of the sample per 0.5% strain; more is",
        "                                      # slower, closer to quasi-static",
        "variable        settle index 5        # sound crossings of the sample to settle before the pull",
        "variable        every index 1000      # steps between output lines",
        "",
        "units           si",
        "atom_style      bacillus",
        "boundary        m m m",
        "newton          on off",
        "special_bonds   lj 0.0 1.0 1.0 coul 1.0 1.0 1.0",
        f"read_data       {data}",
        f"include         {ATOM_TYPES}",
        "timestep        ${dt}",
        "",
        "# The fibers' extent and the box's cross-section across the pull",
    ]
    for axis, (area, _, _, _) in axes.items():
        out.append(f"if \"${{axis}} == {axis}\" then &\n"
                   f"  \"variable        low equal $(bound(all,{axis}min))\" &\n"
                   f"  \"variable        high equal $(bound(all,{axis}max))\" &\n"
                   f"  \"variable        area equal $({area})\"")
    out += [
        "if \"$(is_defined(variable,low)) == 0\" then \"print 'axis must be x, y or z'\" \"quit 1\"",
        "variable        grip_lo equal $(v_low+v_grip*(v_high-v_low))",
        "variable        grip_hi equal $(v_high-v_grip*(v_high-v_low))",
        "",
        "# The grips",
    ]
    for axis, (_, component, low, high) in axes.items():
        velocity = ["0.0", "0.0", "0.0"]
        velocity[int(component)] = "v_speed"  # set below, once the grips are known
        out.append(f"if \"${{axis}} == {axis}\" then &\n"
                   f"  \"region          rlo block {low}\" &\n"
                   f"  \"region          rhi block {high}\" &\n"
                   f"  \"variable        vx equal {velocity[0]}\" &\n"
                   f"  \"variable        vy equal {velocity[1]}\" &\n"
                   f"  \"variable        vz equal {velocity[2]}\"")
    out += [
        "group           grip_lo region rlo",
        "group           grip_hi region rhi",
        "group           grips union grip_lo grip_hi",
        "group           free subtract all grips",
        "if \"$(count(grip_lo)) == 0 || $(count(grip_hi)) == 0\" then &\n"
        "  \"print 'A grip holds no capsules: raise grip.'\" \"quit 1\"",
        "# The free length is between the innermost held capsules' centers, where the bonds that stretch start",
    ]
    for axis in axes:  # one line per axis: ${axis} isn't substituted inside $()
        out.append(f"if \"${{axis}} == {axis}\" then &\n"
                   f"  \"variable        free_length equal $(bound(grip_hi,{axis}min)-bound(grip_lo,{axis}max))\"")
    out += [
        "",
        "# Pull speed and damping, from the time a sound wave takes to cross the sample",
        "variable        transit equal $((v_high-v_low)/v_sound_speed)",
        "variable        steps equal $(ceil(v_transits*v_transit*v_max_strain/0.005/dt))",
        "variable        speed equal $(v_max_strain*v_free_length/(v_steps*dt))",
        "variable        gamma equal $(mass(all)/count(all)/v_transit)",
        "",
        "fix             integrate free nve/bacillus",
        "fix             damp free viscous ${gamma}",
        "fix             hold grip_lo move linear 0.0 0.0 0.0",
        "fix             pull grip_hi move linear 0.0 0.0 0.0",
        "compute         kinetic free ke",
        "thermo_style    custom step time c_kinetic bonds",
        "thermo_modify   norm no",
        "thermo          ${every}",
        "",
        "# Settle with both grips held",
        "run             $(ceil(v_settle*v_transit/dt))",
        "",
        "# Pull",
        "unfix           pull",
        "fix             pull grip_hi move linear ${vx} ${vy} ${vz}",
        "compute         force_lo grip_lo reduce sum f${axis}",
        "compute         force_hi grip_hi reduce sum f${axis}",
        "variable        separation equal xcm(grip_hi,${axis})-xcm(grip_lo,${axis})",
        "variable        separation0 equal $(v_separation)",
        "variable        strain equal (v_separation-v_separation0)/v_free_length",
        "variable        stress equal 0.5*(c_force_lo-c_force_hi)/v_area",
        "variable        f_lo equal c_force_lo",
        "variable        f_hi equal c_force_hi",
        "variable        nbonds equal bonds",
        "variable        now equal step",
        "fix             out all print ${every} \"${now} ${strain} ${stress} ${f_lo} ${f_hi} ${nbonds}\" &\n"
        "                file tensile.txt screen no title \"# step strain stress_Pa force_low_N force_high_N bonds\"",
        "",
        "thermo_style    custom step time v_strain v_stress c_force_lo c_force_hi c_kinetic bonds",
        "print           \"Pulling along ${axis} to strain ${max_strain} in ${steps} steps of ${dt} s at ${speed} m/s\"",
        "run             ${steps}",
        "write_data      tensile_end.data",
        "",
    ]
    return "\n".join(out)
