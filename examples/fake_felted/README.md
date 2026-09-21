# Fake felted material

This example builds the larger twenty-ply control and needled specimens used
by the paired DIRT through-thickness tension experiments. It is a hypothetical
material-generation study rather than a calibrated material model.

| Specimen | Formation video |
| --- | --- |
| Unneedled control | [![Twenty-ply control formation](../../docs/media/felted-control.png)](../../docs/media/felted-control.mp4) |
| Needled continuation | [![Twenty-ply needled formation](../../docs/media/felted-needled.png)](../../docs/media/felted-needled.mp4) |

The exported needled specimen can then be loaded in DIRT for through-thickness
tension. [Click the preview below to play the DIRT simulation.](../../docs/media/dirt-needled-tension.mp4)

[![Needled fake felt undergoing through-thickness tension in DIRT](../../docs/media/dirt-needled-tension.png)](../../docs/media/dirt-needled-tension.mp4)

The control is constructed one physically separated ply at a time, with each
ply approached and relaxed against the existing stack. The needled path uses
the same population and deposition sequence, then applies layer-scoped needle
pulls before final cleanup. Both paths export multi-material capsule and bond
data for DIRT.

Run the control formation and cleanup:

```console
cargo run --release -p fake_felted
cargo run --release -p fake_felted -- --cleanup-from-checkpoint
cargo run --release -p fake_felted -- --dem-polish
```

Run the layer-by-layer needled continuation and its matching cleanup:

```console
cargo run --release -p fake_felted -- --needled
cargo run --release -p fake_felted -- --needled-cleanup
cargo run --release -p fake_felted -- --needled-dem-polish
```

Long stages support `--resume`, `--resume-cleanup`, `--resume-needled`, and
`--resume-needled-cleanup` as appropriate. Generated checkpoints, trajectories,
and solver exports remain under `output/` and are intentionally excluded from
version control.
