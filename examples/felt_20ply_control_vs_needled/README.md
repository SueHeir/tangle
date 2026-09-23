# Twenty-ply control versus needled felt

Start with the **[Python notebook](../../crates/tangle_python/python/examples/native/felt_20ply_control_vs_needled.ipynb)**
or [editable script](../../crates/tangle_python/python/examples/native/felt_20ply_control_vs_needled.py).
See the [installation guide](../../crates/tangle_python/README.md) first.
The commands and output paths below describe the native Rust version; Python
uses the same solver but can have different export/debug defaults and paths.

This example builds the larger twenty-ply control and needled specimens used
by the paired DIRT through-thickness tension experiments. It is a hypothetical
material-generation study rather than a calibrated material model.

| Specimen | Formation video |
| --- | --- |
| Unneedled control | [![Twenty-ply control formation](../../docs/media/felted-control.png)](../../docs/media/felted-control.mp4) |
| Needled continuation | [![Twenty-ply needled formation](../../docs/media/felted-needled.png)](../../docs/media/felted-needled.mp4) |

The exported needled specimen can then be loaded in DIRT for through-thickness
tension. [Click the preview below to play the DIRT simulation.](../../docs/media/dirt-needled-tension.mp4)

[![Needled felt undergoing through-thickness tension in DIRT](../../docs/media/dirt-needled-tension.png)](../../docs/media/dirt-needled-tension.mp4)

The control is constructed one physically separated ply at a time, with each
ply approached and relaxed against the existing stack. The needled path uses
the same population and deposition sequence, then applies layer-scoped needle
pulls before final cleanup. Both paths export multi-material capsule and bond
data for DIRT.

Each run picks a specimen and a stage. `--specimen` is `control` (default) or
`needled`; `--stage` is `form` (default), `cleanup`, or `polish`:

```console
cargo run --release -p felt_20ply_control_vs_needled -- --specimen control --stage form
cargo run --release -p felt_20ply_control_vs_needled -- --specimen control --stage cleanup
cargo run --release -p felt_20ply_control_vs_needled -- --specimen control --stage polish

cargo run --release -p felt_20ply_control_vs_needled -- --specimen needled --stage form
cargo run --release -p felt_20ply_control_vs_needled -- --specimen needled --stage cleanup
cargo run --release -p felt_20ply_control_vs_needled -- --specimen needled --stage polish
```

| Stage | Starts from | Recipe |
| --- | --- | --- |
| `form` | an empty cell | ply-by-ply deposition (plus needling for `needled`), compaction, final admissibility relaxation |
| `cleanup` | the same specimen's `form` checkpoint | `settle/*` then the `cleanup/1-curvature-coarse` … `cleanup/6-contact-final` ladder |
| `polish` | the same specimen's `cleanup` checkpoint | `polish/1-contact-coarse`, `polish/2-contact-final` for DEM contact startup |

Each specimen/stage pair has its own checkpoint case id,
`felt-20ply-{specimen}-{stage}`, and writes `output/{specimen}_{stage}.restart`,
`.dump`, `_view.py`, `.ovito`, and `_capsules.data` (for example
`output/needled_polish_capsules.data`). Starting a stage clears that stage's
previous outputs. Add `--resume` to continue a stage from its own checkpoint,
or `--retry` to restart a rejected `cleanup` or `polish` recipe from that
stage's last saved geometry. `--debug-ovito` records a trajectory for the
control `form` stage; every other run records one by default.

The two populations use the material names `fine_7um` (7 µm, 200 µm minimum
bend radius) and `coarse_19um` (19 µm, 60 µm minimum bend radius). Generated
checkpoints, trajectories, and solver exports remain under `output/` and are
intentionally excluded from version control.
