# Fake needled felt

[![Layer-by-layer insertion and needling](../../docs/media/layered-needling.png)](../../docs/media/layered-needling.mp4)

*Click the preview to play the OVITO rendering.*

This teaching example builds a hypothetical needled fibrous material one
planar layer at a time while the complete reserved topology stays on the GPU.
It is a process demonstration, not a calibrated material model.

The recipe starts with two active layers, lowers the second onto the first,
then repeats this cycle for each later layer:

1. activate the next prepacked deposition layer;
2. place it one fiber-scale gap above the existing stack;
3. relax contacts;
4. release the layer-center tethers so the whole stack can respond;
5. deterministically select one internal vertex on 30% of the layer fibers;
6. pull those vertices downward by two layer gaps while every contacted fiber
   remains free to translate and deform;
7. release the temporary needle targets and relax again.

The initial `1 × 1 × 0.32` manufacturing envelope keeps the six-layer preform
compact without pre-compressing it to the final density. After all layers are
deposited, the recipe releases manufacturing targets and lowers the top cell
face using closed-loop GPU compaction. The requested
nominal fiber volume fraction is 40%. Safety guards may stop the example first
if that trial target cannot be reached without excessive overlap or bending.

Run the recipe and write its DEM-BPM model:

```console
cargo run --release -p fake_needled
```

Record an OVITO-readable trajectory:

```console
cargo run --release -p fake_needled -- --debug-ovito
```

Curvature use is normalized from zero to one. Values above the admissible bend
limit are colored red by the generated `output/relaxation_view.py` recipe.
