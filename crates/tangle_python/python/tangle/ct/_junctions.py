"""Try every way the loose fiber ends in an unsure region can be joined up.

After the unsure stretches are cut out (``_regrow.cut_unsure``), every sure
piece that ends in a region has a loose end there, a **port**. Each port
either connects to another port of the same fiber type, through a smooth
bridge that keeps within the bend limit, or the fiber ends in the region
(the end grows along its own direction until the scan stops being fiber or
it runs onto another piece). Every combination of these choices is scored
on the host, in nats, as in ``_ends``:

* the squared residual between the scan and the rendered fibers over the
  region, divided by the evidence scale ``2 σ² π r²``;
* the voxels where two fibers overlap, one nat per fiber cross-section;
* every fiber end inside the scan, at the fiber-length prior's price for a
  fiber of the length it would have (``_ends.length_end_cost``: dear for a
  short fiber, cheap near the typical length; at least one nat, so a join
  is preferred when it explains the scan as well);
* every join, by how far past the typical length it makes the fiber
  (``_ends.length_join_cost``).

A piece whose own end already lies in a region is a port too, so it can be
joined; left unjoined, it stays as it is.

Combinations are ranked by that score; the fitter builds the best one, and
the next best when a region's first choice fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._geometry import polyline_length, resample
from ._refine import curvature_ratio


@dataclass
class Port:
    piece: int
    end: int  # 0 = start, -1 = end of the piece
    point: np.ndarray
    direction: np.ndarray  # outward, into the region
    kind: int
    radius: float


@dataclass
class Plan:
    """One region's choice: bridges between port pairs, and extensions of the other ports."""

    score: float
    pairs: list[tuple[int, int]]  # indices into the region's ports
    ends: int = 0  # fiber ends it leaves inside the scan
    details: dict = field(default_factory=dict)


def region_ports(
    pieces: list[np.ndarray],
    cut_ends: list[tuple[int, int]],
    kinds: np.ndarray,
    radii: np.ndarray,
    regions: list[tuple[np.ndarray, np.ndarray]],
) -> list[list[Port]]:
    """The loose ends in every region (a cut end belongs to the nearest region box)."""
    lows = np.array([low for low, _ in regions], dtype=np.float64).reshape(-1, 3)
    highs = np.array([high for _, high in regions], dtype=np.float64).reshape(-1, 3)
    ports: list[list[Port]] = [[] for _ in regions]
    for piece, end in cut_ends:
        line = pieces[piece]
        if len(line) < 2 or not len(regions):
            continue
        tip = line[end]
        inner = line[min(2, len(line) - 1)] if end == 0 else line[max(len(line) - 3, 0)]
        direction = tip - inner
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        gap = np.maximum(np.maximum(lows - tip, tip - highs), 0.0)
        region = int(np.argmin(np.linalg.norm(gap, axis=1)))
        ports[region].append(
            Port(
                piece,
                end,
                tip.copy(),
                direction / norm,
                int(kinds[piece]),
                float(radii[piece]),
            )
        )
    return ports


def bridge(a: Port, b: Port, spacing: float) -> np.ndarray:
    """A cubic Hermite curve leaving ``a`` along its direction and entering ``b`` against its own."""
    length = float(np.linalg.norm(b.point - a.point))
    count = max(int(np.ceil(length / max(spacing, 1e-6))), 2) + 1
    t = np.linspace(0.0, 1.0, count)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00 * a.point
        + h10 * length * a.direction
        + h01 * b.point
        + h11 * length * (-b.direction)
    )


def turn_cost(a: Port, b: Port, length: float, bend_radius: float, *, floor_degrees: float = 10.0) -> float:
    """Nats for how far a join from port ``a`` to port ``b`` turns.

    The turn is the angle between ``a``'s outward direction and the
    direction ``b``'s fiber arrives in (0 for a straight continuation). It
    is priced as a half-normal: half its square over the turn a fiber at the
    bend limit makes along a bridge of ``length`` (at least
    ``floor_degrees``), so a join that turns by a crossing angle costs
    several nats while a gentle continuation costs almost none.
    """
    cosine = float(np.clip(-a.direction @ b.direction, -1.0, 1.0))
    turn = float(np.arccos(cosine))
    allowed = max(length / max(bend_radius, 1e-9), np.radians(floor_degrees))
    return 0.5 * (turn / allowed) ** 2


def allowed_pairs(
    ports: list[Port],
    bends: np.ndarray,
    spacing: float,
    max_length: float,
    max_ratio: float = 1.2,
) -> dict[tuple[int, int], np.ndarray]:
    """Port pairs that can be joined: same type, different pieces, a bridge within the bend limit."""
    pairs = {}
    for i in range(len(ports)):
        for j in range(i + 1, len(ports)):
            a, b = ports[i], ports[j]
            if a.kind != b.kind or a.piece == b.piece:
                continue
            if np.linalg.norm(b.point - a.point) > max_length:
                continue
            if (
                float(a.direction @ (b.point - a.point)) <= 0.0
                or float(b.direction @ (a.point - b.point)) <= 0.0
            ):
                continue  # the ends face away from each other
            curve = bridge(a, b, spacing)
            if curvature_ratio([curve], float(bends[a.kind]))[0] > max_ratio:
                continue
            pairs[(i, j)] = curve
    return pairs


def matchings(
    count: int, allowed: list[tuple[int, int]], cap: int = 4096
) -> list[list[tuple[int, int]]]:
    """Every set of disjoint ``allowed`` pairs over ``count`` ports (the empty set included), up to ``cap``."""
    partners: dict[int, list[int]] = {i: [] for i in range(count)}
    for i, j in allowed:
        partners[i].append(j)
    found: list[list[tuple[int, int]]] = []
    # Depth first with an explicit stack (a region can have more ports than
    # Python's recursion limit), children pushed in reverse so they are
    # visited in order: port i ending here first, then each partner.
    stack: list[tuple[int, frozenset[int], tuple[tuple[int, int], ...]]] = [(0, frozenset(), ())]
    while stack and len(found) < cap:
        i, used, chosen = stack.pop()
        while i < count and i in used:
            i += 1
        if i >= count:
            found.append(list(chosen))
            continue
        children = [(i + 1, used | {i}, chosen)]  # port i ends here
        children += [(i + 1, used | {i, j}, chosen + ((i, j),)) for j in partners[i] if j not in used]
        stack.extend(reversed(children))
    return found


def rank_plans(
    image: np.ndarray,
    box: tuple[np.ndarray, np.ndarray],
    ports: list[Port],
    pairs: dict[tuple[int, int], np.ndarray],
    extensions: list[np.ndarray],
    base_lines: list[np.ndarray],
    base_radii: np.ndarray,
    *,
    margin: float,
    scale: float,
    end_costs: np.ndarray,
    interior: list[bool],
    join_costs: dict[tuple[int, int], float] | None = None,
    grey: np.ndarray | None = None,
    port_profiles: list[np.ndarray] | None = None,
    base_profiles: list[np.ndarray] | None = None,
    void: float = 0.0,
    cap: int = 1024,
    exact: int = 24,
) -> list[Plan]:
    """Every combination for one region, best (lowest score) first.

    ``extensions[k]`` is where port ``k``'s fiber goes if it ends in the
    region (possibly empty), ``interior[k]`` whether that end is inside
    the scan (an end on the scan boundary costs nothing) and ``end_costs[k]``
    its price in nats. ``join_costs`` prices a pair's join (default 0).
    ``base_lines`` are the fixed fibers near the box.

    With ``grey`` (the denoised scan) and profiles for the ports and the
    base fibers, the residual compares the scan's grey with the fibers
    drawn with their profiles (``_grey.render_grey``, brighter fiber where
    two meet) instead of ``image`` with plain occupancy.

    Every combination gets a quick score from each element's own effect
    (its change to the residual and its overlap with the fixed fibers,
    measured once); the best ``exact`` are then drawn whole and scored
    exactly, and come first. The rest keep their quick scores
    (``details["quick"]``).
    """
    from ._moves import render_occupancy

    upper = np.array(image.shape[::-1])
    low = np.clip(np.floor(box[0]).astype(int), 0, upper)
    high = np.clip(np.ceil(box[1]).astype(int), 0, upper)
    if np.any(high <= low):
        return [Plan(0.0, [])]
    observed = image[low[2] : high[2], low[1] : high[1], low[0] : high[0]].astype(
        np.float64
    )
    base = render_occupancy(
        low, high, base_lines, np.asarray(base_radii, dtype=np.float64) + margin
    )
    base_count = (base >= 0.5).astype(np.int32)

    # Each element is drawn over its own sub-box of the region (everything
    # its capsule and soft edge can reach), so scoring it touches only
    # those voxels, not the whole region.
    def reach_box(line: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...]]:
        pad = radius + max(margin, 0.0) + 2.0 * 1.2 + 1.0
        sub_low = np.clip(np.floor(line.min(axis=0) - pad).astype(int), low, high)
        sub_high = np.clip(np.ceil(line.max(axis=0) + pad).astype(int), low, high)
        where = tuple(slice(int(sub_low[a] - low[a]), int(sub_high[a] - low[a])) for a in (2, 1, 0))
        return sub_low, sub_high, where

    def element(line: np.ndarray, radius: float) -> tuple[tuple[slice, ...], np.ndarray] | None:
        if len(line) < 2:
            return None
        sub_low, sub_high, where = reach_box(line, radius)
        if np.any(sub_high <= sub_low):
            return None  # nothing of it in the region
        return where, render_occupancy(sub_low, sub_high, [line], np.array([radius + margin]))

    bridge_occupancy = {
        key: element(curve, ports[key[0]].radius) for key, curve in pairs.items()
    }
    tails = [
        np.vstack([ports[k].point[None], extensions[k]]) if len(extensions[k]) else extensions[k]
        for k in range(len(ports))
    ]
    extension_occupancy = [element(tails[k], ports[k].radius) for k in range(len(ports))]
    drawn_base = bridge_grey = extension_grey = None
    if grey is not None:
        from ._grey import render_grey

        observed = grey[low[2] : high[2], low[1] : high[1], low[0] : high[0]].astype(np.float64)
        drawn_base = render_grey(low, high, base_lines, base_radii, base_profiles, void)

        def drawn(line: np.ndarray, k: int) -> np.ndarray | None:
            if len(line) < 2:
                return None
            sub_low, sub_high, _ = reach_box(line, ports[k].radius)
            if np.any(sub_high <= sub_low):
                return None
            return render_grey(sub_low, sub_high, [line], [ports[k].radius], [port_profiles[k]], void)

        bridge_grey = {key: drawn(curve, key[0]) for key, curve in pairs.items()}
        extension_grey = [drawn(tails[k], k) for k in range(len(ports))]
    radius = float(np.mean([p.radius for p in ports])) if ports else 1.0
    area = np.pi * radius * radius
    shown_base = drawn_base if drawn_base is not None else base
    base_residual = float(((observed - shown_base) ** 2).sum())

    def pieces_of(chosen: list[tuple[int, int]], paired: set[int]) -> list[tuple]:
        """((sub-box, occupancy), drawn) of every element a plan adds."""
        items = [(bridge_occupancy[key], bridge_grey[key] if bridge_grey else None) for key in chosen]
        items += [
            (extension_occupancy[k], extension_grey[k] if extension_grey else None)
            for k in range(len(ports))
            if k not in paired
        ]
        return [item for item in items if item[0] is not None]

    # Each element's own change to the residual and its overlap with the fixed
    # fibers, measured once: a plan's quick score adds them up (exact while its
    # elements don't overlap one another), and only the best ``exact`` plans
    # are drawn whole.
    delta: dict[int, tuple[float, float]] = {}

    def alone(item: tuple) -> tuple[float, float]:
        key = id(item[0])
        if key not in delta:
            (where, occupancy), drawn = item
            shown = drawn if drawn_base is not None else occupancy
            seen, before = observed[where], shown_base[where]
            change = float(((seen - np.maximum(before, shown)) ** 2).sum() - ((seen - before) ** 2).sum())
            overlap = float(((occupancy >= 0.5) & (base_count[where] >= 1)).sum())
            delta[key] = (change, overlap)
        return delta[key]

    candidates = []
    for chosen in matchings(len(ports), list(pairs), cap=cap):
        paired = {i for pair in chosen for i in pair}
        ends = [k for k in range(len(ports)) if k not in paired and interior[k]]
        end_nats = float(sum(end_costs[k] for k in ends))
        join_nats = float(sum(join_costs.get(pair, 0.0) for pair in chosen)) if join_costs else 0.0
        items = pieces_of(chosen, paired)
        changes = [alone(item) for item in items]
        quick = (base_residual + sum(c for c, _ in changes)) / scale + sum(o for _, o in changes) / area
        candidates.append((quick + end_nats + join_nats, chosen, paired, ends, end_nats, join_nats, items))
    candidates.sort(key=lambda c: c[0])

    plans = []
    for rank, (quick, chosen, paired, ends, end_nats, join_nats, items) in enumerate(candidates):
        if rank >= exact:
            plans.append(Plan(quick, chosen, len(ends), {"quick": True, "end_nats": end_nats, "join_nats": join_nats}))
            continue
        rendered = shown_base.copy()
        count = base_count.copy()
        for (where, occupancy), drawn in items:
            np.maximum(rendered[where], drawn if drawn_base is not None else occupancy, out=rendered[where])
            count[where] += occupancy >= 0.5
        residual = float(((observed - rendered) ** 2).sum())
        overlap = float(np.maximum(count - np.maximum(base_count, 1), 0).sum())
        overlap_nats = overlap / area
        score = residual / scale + overlap_nats + end_nats + join_nats
        plans.append(
            Plan(
                score,
                chosen,
                len(ends),
                {
                    "residual_nats": residual / scale,
                    "overlap_nats": overlap_nats,
                    "end_nats": end_nats,
                    "join_nats": join_nats,
                },
            )
        )
    head = sorted(plans[:exact], key=lambda plan: plan.score)
    return head + plans[exact:]


def assemble(
    pieces: list[np.ndarray],
    connections: list[tuple[int, int, int, int, np.ndarray]],
    extensions: dict[tuple[int, int], np.ndarray],
    spacing: float,
) -> tuple[list[np.ndarray], list[int]]:
    """Chain pieces through the bridges into fibers.

    ``connections`` are ``(piece, end, other piece, other end, bridge)``
    with the bridge running from the first end to the second;
    ``extensions`` maps a ``(piece, end)`` to the points appended there.
    Returns the fibers and, for each, the first piece in it (for its type
    and radius). A connection that would close a loop is left out.
    """
    link: dict[tuple[int, int], tuple[int, int, np.ndarray]] = {}
    root = list(range(len(pieces)))

    def find(k: int) -> int:
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for piece, end, other, other_end, curve in connections:
        a, b = find(piece), find(other)
        if a == b:
            continue  # would close a loop
        root[b] = a
        link[(piece, end)] = (other, other_end, curve)
        link[(other, other_end)] = (piece, end, curve[::-1])

    def oriented(piece: int, entry: int) -> np.ndarray:
        line = pieces[piece]
        return line if entry == 0 else line[::-1]

    seen: set[int] = set()
    fibers: list[np.ndarray] = []
    first: list[int] = []
    starts = [
        p for p in range(len(pieces)) if (p, 0) not in link or (p, -1) not in link
    ]
    for start in starts:
        if start in seen:
            continue
        entry = 0 if (start, 0) not in link else -1
        parts: list[np.ndarray] = []
        head = extensions.get((start, entry))
        if head is not None and len(head):
            parts.append(head[::-1])
        piece, into = start, entry
        while True:
            seen.add(piece)
            parts.append(oriented(piece, into))
            out = -1 if into == 0 else 0
            nxt = link.get((piece, out))
            if nxt is None:
                tail = extensions.get((piece, out))
                if tail is not None and len(tail):
                    parts.append(tail)
                break
            other, other_end, curve = nxt
            parts.append(curve[1:-1])
            piece, into = other, other_end
        line = np.vstack([p for p in parts if len(p)])
        steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
        line = line[np.concatenate([[True], steps > 1e-9])]
        fibers.append(resample(line, spacing) if polyline_length(line) > 0 else line)
        first.append(start)
    return fibers, first
