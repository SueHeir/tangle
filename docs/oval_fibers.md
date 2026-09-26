# Oval fibers

Tangle fibers can have an elliptical (oval) cross-section as well as a round
one. This page describes how ovals are stored, relaxed and exported.

## Making oval fibers

In Python, give a material a `thickness` smaller than its `diameter`. The
diameter is the long width and the thickness is the short one:

```python
oval = tangle.Material("oval fiber", diameter=30 * um, thickness=20 * um)
collection.add_fiber(points, oval)                      # long axis lies flat
collection.add_fiber(points, oval, long_axis=[0, 0, 1])  # long axis given
```

`FiberPopulation(material=oval, ...)` keeps the thickness-to-width ratio for
every sampled diameter. `Assembly.long_axes()` and
`FiberCollection.long_axes()` return the long-axis direction at every vertex.
The crossing generators still build round fibers only.

In Rust, a fiber with `Section::Elliptical { semi_axes }` stores one unit
director per vertex in `GeometryState::directors`. The first semi-axis lies
along the director and the second across it. Directors are optional. When
none are stored, the default director is perpendicular to the fiber and lies
in the xy plane, so the oval's short axis is as close to z as it can be.

## Relaxation

The GPU solver models an oval as a row of round lanes side by side along its
director:

- Each lane has the short semi-axis as its radius.
- The outer lanes sit at the long semi-axis minus the short one from the
  centerline.
- There are 2 to 5 lanes, spaced so that the flat sides dip by less than 5 %.

Contact is resolved between every pair of lanes. A contact on an outer lane
moves the centerline and also turns the oval about its own axis.
`RelaxationSettings.twist_stiffness`, 0.1 by default, keeps the long axis
smoothly varying along the fiber. Walls touch the outer lane nearest to them,
so a flat oval rests on a wall with its thin side.

Round fibers are unchanged: they have one lane on the centerline.

Not yet supported for ovals:

- Adaptive segmentation.
- The CT image force (`ImageRelaxer`).
- Contact capture used by contact metrics, which still measures gaps with the
  long semi-axis.

## Exports

- **PuMA** voxelizes an elliptical tube around every segment, with
  ellipsoidal caps at the fiber ends.
- **OVITO** draws each oval as its row of lane capsules, the same geometry the
  solver sees.
- **DIRT** (DEM-BPM) export still rejects oval fibers.

The oval example is
[`oval_fibers.py`](../crates/tangle_python/python/examples/oval_fibers.py).
