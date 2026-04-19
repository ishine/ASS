# TF-MLPNet (edge TIGER backbone)

This folder packages the **TF-MLPNet–style**, NPU-oriented TIGER separator that was drafted in `context.md` (chat export). The **implemented code** is the **v2 “conservative export”** variant: no channel norm in the separator path, smaller time states, no `expand` on global state, and `forward_sequence` unrolled **frame-by-frame** through `forward_cell` so training matches deployment.

## Layout

| Path | Role |
|------|------|
| `context.md` | Original discussion / v1 sketch / review |
| `tf_mlpnet/tiger_edge_mlp.py` | **v2** `TIGEREdgeMLP`, `EdgeTFMLPSeparator`, mixers |
| `tf_mlpnet/legacy_v1.py` | **v1** archive: `V1TIGEREdgeMLP`, `V1Edge*` (norm + expand + chunk sequence) |
| `tf_mlpnet/export_onnx.py` | Single-frame ONNX cell wrapper + export helpers |
| `tf_mlpnet/npu_utils.py` | Streaming state byte count, conv receptive-field assert |
| `tests/test_tiger_edge_mlp_smoke.py` | Smoke + ONNX checker |

## Alignment with `prj_context.md`

- **Operators:** depthwise / pointwise `Conv2d`, `cat`, slice, `+`, `*`, `sigmoid`, `ReLU` (no attention / `bmm` in this path).
- **Rank:** activations stay **4D** `[B, C, F, T]`; no folding of batch into other axes for 4D tensors.
- **Causal time:** left cache + `Conv2d` on time without future padding; `(k_t - 1) * d < 14` enforced for time dilations.
- **State:** dummy `past_kvs` / `past_valid_mask` plus real `prev_states_*` / `prev_global_states`; sizes should stay under **192 KiB** fp32 for chosen `nband` / channels (validate per config).
- **Export:** use `TIGEREdgeMLPCellExportWrapper` + `build_tiger_edge_mlp_dummy_inputs` so all caches are explicit ONNX inputs/outputs.

## Import path

Python cannot import a hyphenated top-level module. Add **`TF-MLPNet`** (this directory) to `sys.path`, then:

```python
from tf_mlpnet import TIGEREdgeMLP
from tf_mlpnet import V1TIGEREdgeMLP  # archived v1 (see tf_mlpnet/legacy_v1.py)
```

From the ASS repo root, tests insert both `ASS/` and `TF-MLPNet/` on `sys.path`.

## Commands

```bash
cd /app/ASS
./.venv/bin/python -m pytest TF-MLPNet/tests/test_tiger_edge_mlp_smoke.py -q
```

Optional ONNX path (also covered by pytest):

```bash
cd /app/ASS
PYTHONPATH="/app/ASS:/app/ASS/TF-MLPNet" ./.venv/bin/python -c "
from tf_mlpnet import TIGEREdgeMLP, TIGEREdgeMLPCellExportWrapper
from tf_mlpnet.export_onnx import build_tiger_edge_mlp_dummy_inputs, export_tiger_edge_mlp_to_onnx
m = TIGEREdgeMLP(out_channels=24, in_channels=96, num_blocks=4, upsampling_depth=2, num_sources=2, need_streaming=True, edge_hidden_channels=24, edge_num_blocks=4)
w = TIGEREdgeMLPCellExportWrapper(m.eval())
export_tiger_edge_mlp_to_onnx(w, build_tiger_edge_mlp_dummy_inputs(m), '/tmp/tiger_edge.onnx')
"
```

## Note on base `TIGER`

`TIGEREdgeMLP` subclasses `TIGER` from `TIGER.tiger_online`. `TIGER.__init__` still builds the original `RecurrentKV` separator before it is **replaced** by `EdgeTFMLPSeparator`; this wastes a bit of construction time. A future refactor can add an optional hook on `TIGER` to skip building the default separator when an edge subclass is used.

## Deviations from the raw ``context.md`` snippet (required for this repo)

1. **Encoder layout:** ``RecurrentKV`` consumes ``[B, nband, feature_dim, T]``, not ``[B, feature_dim, nband, T]``. The separator permutes at the boundary so internal blocks still use ``[B, C, F, T]`` with ``F = nband``.
2. **Separator output channels:** ``_decode_masks`` expects the same layout as stock TIGER: ``[B, nband, feature_dim, T]``. The chat draft used ``num_output * input_dim``; here ``out_proj`` maps to **``feature_dim``** only; per-stem expansion remains in ``MaskBlock`` inside ``_decode_masks``.
3. **Export asserts:** Python ``assert``s on tensor shapes can trigger ``TracerWarning`` during ONNX export; they are kept for eager safety. For stricter graphs, gate them with ``not torch.onnx.is_in_onnx_export()`` or use ``torch._assert``.
