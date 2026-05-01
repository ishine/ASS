"""
Full validation for ``TIGERNPUEdgeV1`` (NPU edge preset).

Runs:
- conv receptive-field constraints (prj_context)
- fp32 streaming state byte budget (192 KiB target)
- no PReLU / no Dropout after sanitization
- sequence vs single-frame cell numerical consistency
- ONNX export + onnx checker
- onnx-mlir ``--EmitMLIR`` and a heuristic scan of emitted MLIR for patterns
  that commonly break embedded NPUs (see ``prj_context.md``: avoid dynamic
  control flow / graph loops in ONNX; this checks leftovers after lowering).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import Counter
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


_DEFAULT_ONNX_MLIR = Path("/workdir/onnx-mlir/build/Debug/bin/onnx-mlir")


def resolve_onnx_mlir_tool(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        return cli_path if cli_path.is_file() else None
    w = shutil.which("onnx-mlir")
    if w:
        return Path(w)
    if _DEFAULT_ONNX_MLIR.is_file():
        return _DEFAULT_ONNX_MLIR
    return None


def run_onnx_mlir_emit_mlir(onnx_path: Path, o_path: Path, tool: Path) -> Path:
    """
    Run onnx-mlir in ``--EmitMLIR`` mode and return the path to the textual MLIR.

    onnx-mlir names outputs as ``< -o argument >.onnx.mlir``; we also parse the
    ``Generated "..."`` line from stdout/stderr for forward compatibility.
    """
    proc = subprocess.run(
        [str(tool), "--EmitMLIR", str(onnx_path), "-o", str(o_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in blob.splitlines():
        m = re.search(r'Generated\s+"([^"]+)"', line)
        if m:
            out = Path(m.group(1))
            if out.is_file():
                return out
    guess = o_path.with_name(o_path.name + ".onnx.mlir")
    if guess.is_file():
        return guess
    raise AssertionError(
        "onnx-mlir --EmitMLIR finished but output MLIR not found "
        f"(tried parsed path and {guess}); stdout/stderr:\n{blob[-4000:]}"
    )


# Ops / dialects that usually indicate ONNX dynamic control flow or graph
# loops still present after lowering — at odds with ``prj_context.md`` guidance
# to avoid if/else and for/while in the export path. Tune for your stack.
_MLIR_FAIL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bonnx\.If\b"), "onnx.If (dynamic branch)"),
    (re.compile(r"\bonnx\.Scan\b"), "onnx.Scan (explicit loop)"),
    (re.compile(r"\bonnx\.Loop\b"), "onnx.Loop"),
    (re.compile(r"\bonnx\.Sequence"), "onnx.Sequence*"),
    (re.compile(r"\bonnx\.NonZero\b"), "onnx.NonZero (dynamic indices)"),
    (re.compile(r"\bonnx\.RNN\b"), "onnx.RNN"),
    (re.compile(r"\bonnx\.LSTM\b"), "onnx.LSTM"),
    (re.compile(r"\bonnx\.GRU\b"), "onnx.GRU"),
    (re.compile(r"\bscf\.(if|for|while)\b"), "scf loop/branch (explicit host-style control flow)"),
    (re.compile(r"\bcf\.cond_br\b"), "cf.cond_br (CFG branch)"),
)

# Many TV NPUs have no native transcendental path; flag for visibility (warn only).
_MATH_OP = re.compile(r"\bmath\.([a-zA-Z0-9_]+)\b")


def scan_emit_mlir_for_npu_red_flags(mlir_path: Path) -> tuple[list[str], Counter[str]]:
    """
    Return (hard_fail_reasons, math_op_counts).

    ``affine.for`` / ``krnl.*`` are *not* failures here: onnx-mlir lowers conv
    and elementwise ops to nested affine loops by design.
    """
    text = mlir_path.read_text(encoding="utf-8", errors="replace")
    fails: list[str] = []
    for rx, label in _MLIR_FAIL_PATTERNS:
        if rx.search(text):
            fails.append(label)
    math_ops: Counter[str] = Counter()
    for m in _MATH_OP.finditer(text):
        math_ops[m.group(1)] += 1
    return fails, math_ops


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TIGERNPUEdgeV1 for TV NPU edge constraints")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--onnx-out", type=Path, default=REPO_ROOT / "TIGER" / "artifacts" / "tiger_npu_edge_v1.onnx")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument(
        "--skip-mlir",
        action="store_true",
        help="Skip onnx-mlir --EmitMLIR and MLIR scan (faster CI without /workdir toolchain).",
    )
    parser.add_argument(
        "--onnx-mlir",
        type=Path,
        default=None,
        help="Path to onnx-mlir binary (default: PATH, else /workdir/onnx-mlir/build/Debug/bin/onnx-mlir).",
    )
    parser.add_argument(
        "--mlir-out-stub",
        type=Path,
        default=None,
        help="Passed to onnx-mlir -o; actual file will be <stub>.onnx.mlir (default: alongside --onnx-out).",
    )
    parser.add_argument(
        "--fail-on-math-ops",
        action="store_true",
        help="Fail if any math.* ops appear in emitted MLIR (strict; many NPUs lack exp/log/sqrt).",
    )
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

        if args.skip_mlir:
            print("[mlir] skipped (--skip-mlir)")
        else:
            tool = resolve_onnx_mlir_tool(args.onnx_mlir)
            if not tool:
                raise AssertionError(
                    "onnx-mlir not found: install it, put it on PATH, or pass "
                    "--onnx-mlir /workdir/onnx-mlir/build/Debug/bin/onnx-mlir "
                    "(see prj_context.md /workdir)."
                )
            stub = args.mlir_out_stub or args.onnx_out.with_suffix(".emit.mlir")
            stub.parent.mkdir(parents=True, exist_ok=True)
            print(f"[mlir] onnx-mlir --EmitMLIR ({tool}) -> -o {stub}")
            mlir_path = run_onnx_mlir_emit_mlir(args.onnx_out, stub, tool)
            print(f"[mlir] generated {mlir_path} ({mlir_path.stat().st_size} bytes)")

            fails, math_ops = scan_emit_mlir_for_npu_red_flags(mlir_path)
            if fails:
                raise AssertionError(
                    "Emitted MLIR contains patterns discouraged for embedded NPU ONNX "
                    f"(prj_context): {', '.join(fails)}"
                )
            print("[mlir] scan: no onnx If/Scan/Loop/Sequence, scf.*, or cf.cond_br hits")

            if math_ops:
                top = ", ".join(f"{k}={v}" for k, v in math_ops.most_common(8))
                msg = f"[mlir] scan: math.* ops present (informational): {top}"
                if args.fail_on_math_ops:
                    raise AssertionError(msg.replace("(informational)", "FAIL (--fail-on-math-ops)"))
                print(msg)
            else:
                print("[mlir] scan: no math.* dialect ops")

    print("[ok] all NPU edge v1 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
