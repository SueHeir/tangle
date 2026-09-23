"""Length units for building TANGLE inputs, expressed in meters.

    >>> from tangle.units import um, mm
    >>> Material("fine_7um", diameter=7 * um)
"""

m = 1.0
mm = 1e-3
um = 1e-6
nm = 1e-9

__all__ = ["m", "mm", "um", "nm"]
