"""Overlay fiber labels on the raw scan, one color per fiber."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fiber_palette(count: int, seed: int = 0) -> np.ndarray:
    """``count + 1`` RGB colors in [0, 1]; row 0 (void) is black.

    Hues follow the golden-ratio sequence so neighboring ids differ strongly.
    """
    import colorsys

    rng = np.random.default_rng(seed)
    start = rng.uniform()
    colors = [(0.0, 0.0, 0.0)]
    for i in range(count):
        hue = (start + 0.61803398875 * i) % 1.0
        saturation = 0.65 + 0.3 * ((i * 7) % 3) / 2
        value = 0.95 - 0.2 * ((i * 5) % 2)
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return np.asarray(colors)


def _gray(volume: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if low is None or high is None:
        sample = volume[:: max(1, volume.shape[0] // 32)] if volume.ndim == 3 else volume
        low, high = np.percentile(sample, [0.5, 99.5])
    return np.clip((volume - low) / max(high - low, 1e-12), 0.0, 1.0)


def overlay_slice(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    palette: np.ndarray | None = None,
    alpha: float = 0.45,
    outline: bool = True,
    window: tuple[float, float] | None = None,
) -> np.ndarray:
    """RGB float image of a 2D slice with each fiber's region tinted."""
    from scipy.ndimage import grey_dilation, grey_erosion

    labels = np.asarray(labels)
    if palette is None:
        palette = fiber_palette(int(labels.max()))
    gray = _gray(image, *(window or (None, None)))
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    mask = labels > 0
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * palette[labels[mask]]
    if outline:
        edge = mask & (grey_dilation(labels, size=3) != grey_erosion(labels, size=3))
        rgb[edge] = 0.75 * palette[labels[edge]] + 0.25
    return np.clip(rgb, 0.0, 1.0)


def overlay_volume(volume: np.ndarray, labels: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """``uint8`` RGB stack ``(z, y, x, 3)`` for viewing in Fiji/Napari/ParaView."""
    palette = fiber_palette(int(labels.max()))
    low, high = np.percentile(np.asarray(volume)[:: max(1, volume.shape[0] // 32)], [0.5, 99.5])
    out = np.empty(volume.shape + (3,), dtype=np.uint8)
    for z in range(volume.shape[0]):
        out[z] = (255 * overlay_slice(volume[z], labels[z], palette=palette, alpha=alpha, window=(low, high))).astype(np.uint8)
    return out


def save_overlay_figure(
    path: str | Path,
    volume: np.ndarray,
    labels: np.ndarray,
    *,
    title: str | None = None,
    slices: tuple[int, int, int] | None = None,
) -> Path:
    """Three orthogonal slices, raw scan above and overlay below (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, y, x = slices or tuple(n // 2 for n in volume.shape)
    palette = fiber_palette(int(labels.max()))
    low, high = np.percentile(np.asarray(volume)[:: max(1, volume.shape[0] // 32)], [0.5, 99.5])
    views = [
        (f"z = {z}", volume[z], labels[z]),
        (f"y = {y}", volume[:, y], labels[:, y]),
        (f"x = {x}", volume[:, :, x], labels[:, :, x]),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.8))
    for column, (name, image, label) in enumerate(views):
        axes[0, column].imshow(_gray(image, low, high), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"scan, {name}")
        axes[1, column].imshow(overlay_slice(image, label, palette=palette, window=(low, high)))
        axes[1, column].set_title(f"fitted fibers, {name}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
