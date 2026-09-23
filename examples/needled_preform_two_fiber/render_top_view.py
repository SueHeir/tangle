#!/usr/bin/env python3
"""Build a compact periodic top-view diagnostic from a final OVITO dump."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_dump(path: Path):
    lines = path.read_text().splitlines()
    count_index = lines.index("ITEM: NUMBER OF ATOMS")
    count = int(lines[count_index + 1])
    bounds_index = next(i for i, line in enumerate(lines) if line.startswith("ITEM: BOX BOUNDS"))
    bounds = [tuple(map(float, lines[bounds_index + axis + 1].split())) for axis in range(3)]
    periodic = lines[bounds_index].split()[-3:]
    atoms_index = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    columns = lines[atoms_index].split()[2:]
    records = [line.split() for line in lines[atoms_index + 1 : atoms_index + 1 + count]]
    return bounds, periodic, columns, records


def periodic_smooth(grid: list[list[float]], passes: int = 2) -> list[list[float]]:
    size = len(grid)
    for _ in range(passes):
        grid = [
            [
                sum(grid[(row + dr) % size][(column + dc) % size] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                / 9.0
                for column in range(size)
            ]
            for row in range(size)
        ]
    return grid


def analyze(path: Path, fiber_limit: int = 1_000, bins: int = 40):
    bounds, periodic, columns, records = read_dump(path)
    index = {name: offset for offset, name in enumerate(columns)}
    x_low, x_high = bounds[0]
    y_low, y_high = bounds[1]
    width = x_high - x_low
    height = y_high - y_low

    fiber_ids = sorted({int(row[index["mol"]]) for row in records})
    stride = max(1, math.ceil(len(fiber_ids) / fiber_limit))
    selected = set(fiber_ids[::stride])
    segments = []
    density = [[0.0 for _ in range(bins)] for _ in range(bins)]
    radii = sorted({round(float(row[index["shapex"]]), 12) for row in records})
    radius_type = {radius: offset for offset, radius in enumerate(radii)}

    for row in records:
        radius = round(float(row[index["shapex"]]), 12)
        length = float(row[index["shapez"]])
        qx = float(row[index["quati"]])
        qy = float(row[index["quatj"]])
        qz = float(row[index["quatk"]])
        qw = float(row[index["quatw"]])
        axis_x = 2.0 * (qx * qz + qw * qy)
        axis_y = 2.0 * (qy * qz - qw * qx)
        x = float(row[index["x"]])
        y = float(row[index["y"]])
        center_x = ((x - x_low) % width) / width
        center_y = ((y - y_low) % height) / height
        column = min(bins - 1, int(center_x * bins))
        grid_row = min(bins - 1, int(center_y * bins))
        density[grid_row][column] += length

        if int(row[index["mol"]]) in selected:
            half_dx = 0.5 * length * axis_x / width
            half_dy = 0.5 * length * axis_y / height
            segments.append(
                [
                    round(center_x - half_dx, 5),
                    round(center_y - half_dy, 5),
                    round(center_x + half_dx, 5),
                    round(center_y + half_dy, 5),
                    radius_type[radius],
                ]
            )

    density = periodic_smooth(density)
    flat = [value for row in density for value in row]
    mean = sum(flat) / len(flat)
    normalized = [round(value / mean, 3) for value in flat]
    coefficient_of_variation = math.sqrt(sum((value - mean) ** 2 for value in flat) / len(flat)) / mean
    edge_values = []
    interior_values = []
    edge_width = max(1, bins // 10)
    for row in range(bins):
        for column in range(bins):
            target = (
                edge_values
                if row < edge_width
                or row >= bins - edge_width
                or column < edge_width
                or column >= bins - edge_width
                else interior_values
            )
            target.append(density[row][column])
    edge_ratio = (sum(edge_values) / len(edge_values)) / (sum(interior_values) / len(interior_values))
    return {
        "segments": segments,
        "density": normalized,
        "bins": bins,
        "edge_ratio": edge_ratio,
        "coefficient_of_variation": coefficient_of_variation,
        "periodic": periodic,
        "fiber_count": len(fiber_ids),
        "shown_fibers": len(selected),
        "segment_count": len(records),
        "radii_um": [radius * 1.0e6 for radius in radii],
    }


def build_fragment(result: dict) -> str:
    data = json.dumps(result, separators=(",", ":"))
    return f"""<div id="merino-top-view">
  <h2>MERINO top view</h2>
  <div class="viz-grid">
    <div class="card viz-stat"><div class="text-muted">Edge / interior density</div><div class="viz-stat-value tabular-nums">{result['edge_ratio']:.3f}</div><div class="text-small text-muted">1.0 means no edge depletion</div></div>
    <div class="card viz-stat"><div class="text-muted">Smoothed areal CV</div><div class="viz-stat-value tabular-nums">{100.0 * result['coefficient_of_variation']:.1f}%</div><div class="text-small text-muted">Periodic 3×3-bin smoothing</div></div>
    <div class="card viz-stat"><div class="text-muted">Boundary flags</div><div class="viz-stat-value tabular-nums">x/y periodic</div><div class="text-small text-muted">z bounded</div></div>
  </div>
  <div class="merino-panels">
    <section><h3>Periodic fiber projection</h3><canvas data-fibers role="img" aria-label="Top projection of a deterministic sample of thin and thick fibers with periodic wrapping"></canvas><div class="text-small text-muted">{result['shown_fibers']:,} / {result['fiber_count']:,} fibers shown</div></section>
    <section><h3>Centerline-length density</h3><canvas data-density role="img" aria-label="Heatmap of normalized projected centerline length density across the periodic x-y cell"></canvas><div class="text-small text-muted">{result['segment_count']:,} segments · domain mean = 1</div></section>
  </div>
  <div class="viz-row merino-legend"><span><i data-thin></i>7 µm fibers</span><span><i data-thick></i>19 µm fibers</span><span><i data-low></i><i data-high></i>density: low → high</span></div>
</div>
<style>
#merino-top-view {{ width:100%; min-width:0; color:var(--foreground); overflow-wrap:anywhere; }}
#merino-top-view .viz-grid {{ margin:0 0 16px; }}
#merino-top-view .merino-panels {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
#merino-top-view section {{ min-width:0; }}
#merino-top-view canvas {{ display:block; box-sizing:border-box; width:100%; aspect-ratio:1; background:var(--background); border:1px solid var(--border); }}
#merino-top-view .merino-legend {{ display:flex; flex-wrap:wrap; margin-top:12px; gap:12px 18px; }}
#merino-top-view .merino-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
#merino-top-view .merino-legend i {{ display:inline-block; width:18px; height:3px; background:var(--viz-series-1); }}
#merino-top-view .merino-legend i[data-thick] {{ height:7px; background:var(--viz-series-2); }}
#merino-top-view .merino-legend i[data-low] {{ width:12px; height:12px; opacity:.15; }}
#merino-top-view .merino-legend i[data-high] {{ width:12px; height:12px; margin-left:-6px; opacity:1; }}
@media (max-width:620px) {{
  #merino-top-view .viz-grid, #merino-top-view .merino-panels {{ grid-template-columns:1fr; }}
  #merino-top-view .merino-legend {{ align-items:flex-start; }}
}}
</style>
<script>
(() => {{
  const root = document.getElementById('merino-top-view');
  const DATA = {data};
  const colors = [
    getComputedStyle(root.querySelector('[data-thin]')).backgroundColor,
    getComputedStyle(root.querySelector('[data-thick]')).backgroundColor
  ];
  function setup(canvas) {{
    const size = Math.max(300, Math.round(canvas.getBoundingClientRect().width));
    const ratio = window.devicePixelRatio || 1;
    canvas.width = size * ratio; canvas.height = size * ratio;
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
    const canvasStyle = getComputedStyle(canvas);
    ctx.fillStyle = canvasStyle.backgroundColor; ctx.fillRect(0, 0, size, size);
    return [ctx, size, canvasStyle.borderColor];
  }}
  function fibers() {{
    const [ctx, size, border] = setup(root.querySelector('[data-fibers]'));
    ctx.lineCap = 'round';
    DATA.segments.forEach(s => {{
      ctx.strokeStyle = colors[s[4]]; ctx.lineWidth = s[4] ? 1.8 : 0.8; ctx.globalAlpha = s[4] ? 0.55 : 0.38;
      for (let ox=-1; ox<=1; ox++) for (let oy=-1; oy<=1; oy++) {{
        ctx.beginPath(); ctx.moveTo((s[0]+ox)*size, (1-s[1]-oy)*size); ctx.lineTo((s[2]+ox)*size, (1-s[3]-oy)*size); ctx.stroke();
      }}
    }});
    ctx.globalAlpha=1; ctx.strokeStyle=border; ctx.lineWidth=1; ctx.strokeRect(.5,.5,size-1,size-1);
  }}
  function density() {{
    const [ctx, size, border] = setup(root.querySelector('[data-density]'));
    const n=DATA.bins, cell=size/n, max=Math.max(...DATA.density), min=Math.min(...DATA.density);
    ctx.fillStyle=colors[0];
    DATA.density.forEach((v,i) => {{ ctx.globalAlpha=.08+.82*(v-min)/(max-min||1); const x=i%n, y=Math.floor(i/n); ctx.fillRect(x*cell,size-(y+1)*cell,cell+.5,cell+.5); }});
    ctx.globalAlpha=1; ctx.strokeStyle=border; ctx.lineWidth=1; ctx.strokeRect(.5,.5,size-1,size-1);
  }}
  function draw() {{ fibers(); density(); }}
  draw(); new ResizeObserver(draw).observe(root);
}})();
</script>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.dump)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_fragment(result))
    print(
        f"edge/interior={result['edge_ratio']:.3f}, "
        f"smoothed CV={100.0 * result['coefficient_of_variation']:.1f}%"
    )


if __name__ == "__main__":
    main()
