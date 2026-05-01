"""Smoke tests for ``TIGEREdgeMLP`` (TF-MLPNet-style edge backbone)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Package root: ``TF-MLPNet/`` (parent of ``tf_mlpnet/``)
_TF_ROOT = Path(__file__).resolve().parent.parent
_ASS_ROOT = _TF_ROOT.parent
if str(_ASS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASS_ROOT))
if str(_TF_ROOT) not in sys.path:
    sys.path.insert(0, str(_TF_ROOT))

from tf_mlpnet.export_onnx import (  # noqa: E402
    TIGEREdgeMLPCellExportWrapper,
    build_tiger_edge_mlp_dummy_inputs,
    export_tiger_edge_mlp_to_onnx,
    precheck_tiger_edge_mlp_export,
)
from tf_mlpnet.npu_utils import assert_conv_receptive_field, streaming_state_bytes_fp32  # noqa: E402
from tf_mlpnet.legacy_v1 import V1TIGEREdgeMLP  # noqa: E402
from tf_mlpnet.tiger_edge_mlp import TIGEREdgeMLP  # noqa: E402


def test_forward_cell_and_streaming_budget():
    model = TIGEREdgeMLP(
        out_channels=32,
        in_channels=128,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=32,
        edge_num_blocks=4,
        edge_time_dilations=(1, 2, 4),
    )
    model.eval()
    assert_conv_receptive_field(model, limit=14)

    states = model.init_streaming_state(batch_size=1)
    nbytes = streaming_state_bytes_fp32(states)
    assert nbytes <= 192 * 1024, f"streaming state {nbytes} B exceeds 192 KiB"

    ri = torch.randn(1, 1, model.enc_dim * 2, 1)
    out = model.forward_cell(ri, *states)
    assert len(out) == 7
    assert out[0].dim() == 4


def test_sequence_matches_unrolled_cell():
    torch.manual_seed(0)
    model = TIGEREdgeMLP(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=24,
        edge_num_blocks=4,
    )
    model.eval()
    T = 5
    ri = torch.randn(1, 1, model.enc_dim * 2, T)

    seq_out, *seq_tail = model.forward_sequence(ri, detach_state=False)

    s = model.init_streaming_state(1, device=ri.device, dtype=ri.dtype)
    past_kvs, past_valid, p0, p1, p2, pg = s
    frames = []
    for t in range(T):
        fo, past_kvs, past_valid, p0, p1, p2, pg = model.forward_cell(
            ri[:, :, :, t : t + 1],
            past_kvs=past_kvs,
            past_valid_mask=past_valid,
            prev_states_0=p0,
            prev_states_1=p1,
            prev_states_2=p2,
            prev_global_states=pg,
        )
        frames.append(fo)
    cell_cat = torch.cat(frames, dim=-1)

    assert torch.allclose(seq_out, cell_cat, atol=1e-5, rtol=1e-4)


def test_v1_forward_cell_smoke():
    """Archived v1: channel norm + larger time state + chunk sequence path."""
    model = V1TIGEREdgeMLP(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=24,
        edge_num_blocks=4,
    )
    model.eval()
    states = model.init_streaming_state(batch_size=1)
    ri = torch.randn(1, 1, model.enc_dim * 2, 1)
    out = model.forward_cell(ri, *states)
    assert len(out) == 7
    assert out[0].dim() == 4


def test_onnx_export_roundtrip_smoke(tmp_path: Path):
    model = TIGEREdgeMLP(
        out_channels=24,
        in_channels=96,
        num_blocks=4,
        upsampling_depth=2,
        num_sources=2,
        need_streaming=True,
        edge_hidden_channels=24,
        edge_num_blocks=4,
    )
    model.eval()
    wrapper = TIGEREdgeMLPCellExportWrapper(model)
    inputs = build_tiger_edge_mlp_dummy_inputs(model, batch_size=1)
    precheck_tiger_edge_mlp_export(wrapper, inputs)
    out_onnx = tmp_path / "tiger_edge_mlp_cell.onnx"
    export_tiger_edge_mlp_to_onnx(wrapper, inputs, out_onnx, opset_version=17)
    assert out_onnx.is_file()

    import onnx

    onnx.checker.check_model(onnx.load(str(out_onnx)))


if __name__ == "__main__":
    import tempfile

    test_forward_cell_and_streaming_budget()
    test_sequence_matches_unrolled_cell()
    test_v1_forward_cell_smoke()
    test_onnx_export_roundtrip_smoke(Path(tempfile.mkdtemp()))
    print("[ok] TF-MLPNet smoke tests passed")
