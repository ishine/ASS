"""
Full validation for ``TIGERNPUEdgeV1`` (NPU edge preset).

Runs:
- conv receptive-field constraints (prj_context)
- fp32 streaming state byte budget (192 KiB target)
- no PReLU / no Dropout after sanitization
- sequence vs single-frame cell numerical consistency
- ONNX export + onnx checker
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TIGER.npu_edge_utils import (
    assert_conv_constraints,
    collect_conv_constraint_violations,
    streaming_state_bytes_fp32,
    streaming_state_elements,
)
from TIGER.streaming_io import build_causal_ri_sequence
from TIGER.tiger_npu_edge import TIGERNPUEdgeV1, export_tiger_npu_edge_onnx
from TIGER.tiger_online import TIGERCtxStreamingTrainingWrapper


def _assert_no_prelu_or_dropout(root: nn.Module) -> None:
    for m in root.modules():
        if isinstance(m, nn.PReLU):
            raise AssertionError("PReLU still present after NPU edge build")
        if isinstance(m, nn.Dropout):
            raise AssertionError("Dropout still present after NPU edge build")


def run_sequence_vs_cell(model: TIGERNPUEdgeV1, subband_spec_ri_seq: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        seq_out, seq_kv, seq_valid, seq_ctx = model.forward_sequence(subband_spec_ri_seq)

        cell_kv, cell_valid, cell_ctx = model.init_streaming_state(
            batch_size=subband_spec_ri_seq.shape[0],
            device=subband_spec_ri_seq.device,
            dtype=subband_spec_ri_seq.dtype,
        )
        outs = []
        for t in range(subband_spec_ri_seq.shape[-1]):
            cell_out, cell_kv, cell_valid, cell_ctx = model.forward_cell(
                subband_spec_ri_seq[..., t : t + 1],
                past_kvs=cell_kv,
                past_valid_mask=cell_valid,
                time_ctx=cell_ctx,
            )
            outs.append(cell_out)
        cell_out = torch.cat(outs, dim=-1)

    return (seq_out - cell_out).abs().max().item()


def run_onnx_check(model: TIGERNPUEdgeV1, path: Path) -> None:
    export_tiger_npu_edge_onnx(model, path)
    import onnx

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TIGERNPUEdgeV1 for TV NPU edge constraints")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--onnx-out", type=Path, default=REPO_ROOT / "TIGER" / "artifacts" / "tiger_npu_edge_v1.onnx")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    print("[build] TIGERNPUEdgeV1")
    model = TIGERNPUEdgeV1()
    model.eval()

    print("[check] no PReLU / Dropout in module tree")
    _assert_no_prelu_or_dropout(model)

    print("[check] conv / avgpool receptive field (k-1)*d < 14 per axis")
    bad = collect_conv_constraint_violations(model, limit=14)
    if bad:
        for v in bad:
            print(f"  VIOLATION: {v.module_name} axis={v.axis} eff={v.effective}")
        assert_conv_constraints(model)

    states = model.init_streaming_state(batch_size=1)
    nbytes = streaming_state_bytes_fp32(states)
    nelem = streaming_state_elements(states)
    budget = 192 * 1024
    print(f"[budget] streaming state tensors: {len(states)}")
    for i, s in enumerate(states):
        print(f"  [{i}] shape={tuple(s.shape)} numel={s.numel()}")
    print(f"[budget] total numel={nelem} fp32_bytes={nbytes} (limit {budget} = 192 KiB)")
    if nbytes > budget:
        raise AssertionError(f"Streaming state {nbytes} bytes exceeds {budget} bytes budget")

    waveform = torch.randn(1, 1, 256 + (args.frames - 1) * model.stride)
    subband = build_causal_ri_sequence(
        waveform,
        win=model.win,
        hop=model.stride,
        startup_packet=256,
        analysis_window=None,
    )
    print(f"[data] RI shape={tuple(subband.shape)}")

    wrapper = TIGERCtxStreamingTrainingWrapper(model, detach_state=False, chunk_size=4)
    with torch.no_grad():
        w_out, *_ = wrapper(subband)
    print(f"[wrapper] out shape={tuple(w_out.shape)}")

    max_diff = run_sequence_vs_cell(model, subband)
    print(f"[consistency] max |seq - cell| = {max_diff:.6e}")
    if max_diff > 1e-4:
        raise AssertionError(f"sequence vs cell mismatch too large: {max_diff}")

    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_onnx:
        print(f"[onnx] export -> {args.onnx_out}")
        run_onnx_check(model, args.onnx_out)
        print("[onnx] checker passed")

        mlir_tool = shutil.which("onnx-mlir")
        if mlir_tool:
            mlir_out = args.onnx_out.with_suffix(".mlir")
            subprocess.run([mlir_tool, str(args.onnx_out), "-o", str(mlir_out)], check=True, capture_output=True)
            print(f"[mlir] wrote {mlir_out}")
        else:
            print("[mlir] onnx-mlir not in PATH (skip; use Docker /workdir toolchain per prj_context.md)")

    print("[ok] all NPU edge v1 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
