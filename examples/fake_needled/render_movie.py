"""Render the fake-needled OVITO trajectory to an MP4."""

from pathlib import Path

from ovito.io import import_file
from ovito.modifiers import (
    AssignColorModifier,
    ColorCodingModifier,
    ExpressionSelectionModifier,
)
from ovito.vis import ParticlesVis, TachyonRenderer, Viewport


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

pipeline = import_file(str(OUTPUT / "relaxation.dump"), sort_particles=True)
pipeline.modifiers.append(
    ColorCodingModifier(property="curvature_ratio", start_value=0.0, end_value=1.0)
)
pipeline.modifiers.append(ExpressionSelectionModifier(expression="curvature_ratio > 1.0"))
pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.0, 0.0)))
pipeline.add_to_scene(name="TANGLE fake needled felt")

data = pipeline.compute(0)
data.particles.vis.shape = ParticlesVis.Shape.Spherocylinder
data.cell.vis.enabled = True
data.cell.vis.line_width = 0.004

viewport = Viewport(type=Viewport.Type.Perspective)
viewport.camera_pos = (2.15, -2.15, 1.75)
viewport.camera_dir = (-1.65, 1.65, -1.25)
viewport.camera_up = (0.0, 0.0, 1.0)
viewport.zoom_all(size=(1280, 720))

renderer = TachyonRenderer(
    ambient_occlusion=True,
    ambient_occlusion_samples=8,
    antialiasing=True,
    antialiasing_samples=8,
    shadows=True,
)

movie_path = OUTPUT / "fake_needled.mp4"
viewport.render_anim(
    filename=str(movie_path),
    size=(1280, 720),
    fps=8,
    background=(0.97, 0.97, 0.97),
    renderer=renderer,
)
print(f"Rendered OVITO movie: {movie_path}")
