"""Reading fibers and materials from an Assembly for both meshers."""

from __future__ import annotations

import numpy as np


def as_assembly(source):
    """The ``Assembly`` behind an ``Assembly`` or a recipe's ``RunResult``."""
    if hasattr(source, "fiber_ids") and hasattr(source, "centerlines"):
        return source
    assembly = getattr(source, "assembly", None)
    if assembly is not None and hasattr(assembly, "fiber_ids"):
        return assembly
    raise TypeError(f"expected a tangle.Assembly or RunResult, got {type(source).__name__}")


def material_table(assembly) -> tuple[tuple[str, ...], np.ndarray]:
    """The distinct material names in fiber order, and each fiber's index
    into them."""
    names: list[str] = []
    index: dict[str, int] = {}
    per_fiber = []
    for name in assembly.fiber_materials():
        if name not in index:
            index[name] = len(names)
            names.append(name)
        per_fiber.append(index[name])
    return tuple(names), np.asarray(per_fiber, dtype=np.int64)


def binder_name(materials: tuple[str, ...]) -> str:
    """``"binder"``, or a variant no fiber material already uses."""
    name, suffix = "binder", 2
    while name in materials:
        name, suffix = f"binder_{suffix}", suffix + 1
    return name
