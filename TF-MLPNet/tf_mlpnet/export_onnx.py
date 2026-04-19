"""Single-frame ONNX export for ``TIGEREdgeMLP`` (streaming cell)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TIGEREdgeMLPCellExportWrapper(nn.Module):
    """Wrap ``forward_cell`` for ``T=1`` ONNX export (explicit state I/O)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        prev_states_0: torch.Tensor,
        prev_states_1: torch.Tensor,
        prev_states_2: torch.Tensor,
        prev_global_states: torch.Tensor,
    ):
        if subband_spec_RIs.shape[-1] != 1:
            raise RuntimeError(
                f"TIGEREdgeMLPCellExportWrapper expects T=1, got {tuple(subband_spec_RIs.shape)}"
            )
        return self.model.forward_cell(
            subband_spec_RIs=subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )


def build_tiger_edge_mlp_dummy_inputs(
    model: nn.Module,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Build dummy inputs using ``init_streaming_state`` (exact shapes)."""
    model = model.to(device=device, dtype=dtype).eval()

    (
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    ) = model.init_streaming_state(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    if not hasattr(model, "enc_dim"):
        raise RuntimeError("Model is missing attribute 'enc_dim'")

    num_input = int(getattr(model, "num_input", 1))

    subband_spec_RIs = torch.randn(
        batch_size,
        num_input,
        model.enc_dim * 2,
        1,
        device=device,
        dtype=dtype,
    )

    return (
        subband_spec_RIs,
        past_kvs,
        past_valid_mask,
        prev_states_0,
        prev_states_1,
        prev_states_2,
        prev_global_states,
    )


def precheck_tiger_edge_mlp_export(wrapper: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    wrapper.eval()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    for idx, out in enumerate(outputs):
        if not isinstance(out, torch.Tensor):
            continue
        print(f"precheck output[{idx}] shape = {tuple(out.shape)}")


def export_tiger_edge_mlp_to_onnx(
    wrapper: nn.Module,
    inputs: tuple[Any, ...],
    onnx_path: str | Path,
    opset_version: int = 17,
) -> None:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = [
        "subband_spec_RIs",
        "past_kvs",
        "past_valid_mask",
        "prev_states_0",
        "prev_states_1",
        "prev_states_2",
        "prev_global_states",
    ]
    output_names = [
        "band_masked_output",
        "new_kvs",
        "new_valid_mask",
        "new_states_0",
        "new_states_1",
        "new_states_2",
        "new_global_states",
    ]
    dynamic_axes = {
        "subband_spec_RIs": {0: "B"},
        "past_kvs": {0: "B"},
        "past_valid_mask": {0: "B"},
        "prev_states_0": {0: "B"},
        "prev_states_1": {0: "B"},
        "prev_states_2": {0: "B"},
        "prev_global_states": {0: "B"},
        "band_masked_output": {0: "B"},
        "new_kvs": {0: "B"},
        "new_valid_mask": {0: "B"},
        "new_states_0": {0: "B"},
        "new_states_1": {0: "B"},
        "new_states_2": {0: "B"},
        "new_global_states": {0: "B"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print(f"exported ONNX to {onnx_path}")
