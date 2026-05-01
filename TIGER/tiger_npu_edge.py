"""
TV edge NPU deployment variant (v1).

This module keeps the original ``tiger_online.py`` intact and provides a
preset + post-init sanitization path aligned with ``prj_context.md``:
ReLU instead of PReLU, no Dropout in the exported graph, and conservative
hyper-parameters so fp32 streaming state typically stays under 192 KiB
(verify at runtime with ``npu_edge_utils.streaming_state_bytes_fp32``).

The architecture follows ``TIGERNPULargeDeployable`` (stacked explicit stages,
ctx-based time path, sliding KV frame attention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .npu_edge_utils import sanitize_for_npu_edge
from .tiger_online import TIGERNPULargeDeployable


class TIGERNPUEdgeV1(TIGERNPULargeDeployable):
    """
    Edge NPU preset: smaller channels, narrow attention hidden dims, and
    ``time_dilations=(1, 1, 2)`` to shorten the time-block context tail so
    ``time_ctx`` stays small for 67 subbands @ 44.1 kHz / 2048 STFT.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("sample_rate", 44100)
        kwargs.setdefault("num_sources", 3)
        kwargs.setdefault("need_streaming", True)
        kwargs.setdefault("win", 2048)
        kwargs.setdefault("stride", 512)

        # ``out_channels`` must be divisible by ``num_sources`` (MaskBlock depthwise groups).
        kwargs.setdefault("out_channels", 66)
        kwargs.setdefault("in_channels", 132)
        kwargs.setdefault("upsampling_depth", 4)
        kwargs.setdefault("num_stages", 2)

        kwargs.setdefault("att_n_head", 4)
        kwargs.setdefault("att_hid_chan", 2)
        kwargs.setdefault("att_val_hid_chan", 2)
        kwargs.setdefault("kv_window_size", 4)

        kwargs.setdefault("time_hidden_channels", 8)
        kwargs.setdefault("time_dilations", (1, 1, 2))
        kwargs.setdefault("time_kernel_size", 3)
        kwargs.setdefault("time_fusion_kernel_size", 1)
        kwargs.setdefault("time_global_kernel_size", 5)

        super().__init__(*args, **kwargs)
        sanitize_for_npu_edge(self)


class NPUEdgeCtxExportWrapper(nn.Module):
    """ONNX export wrapper: fixed inputs, no optional streaming tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        time_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_cell(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            time_ctx=time_ctx,
        )


def export_tiger_npu_edge_onnx(
    model: TIGERNPUEdgeV1,
    export_path: Path,
    opset: int = 14,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dummy_ri = torch.zeros(1, 1, model.enc_dim * 2, 1, device=device, dtype=dtype)
    past_kvs, past_valid_mask, time_ctx = model.init_streaming_state(batch_size=1, device=device, dtype=dtype)
    wrapper = NPUEdgeCtxExportWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ri, past_kvs, past_valid_mask, time_ctx),
            str(export_path),
            export_params=True,
            opset_version=opset,
            input_names=["subband_spec_RIs", "past_kvs", "past_valid_mask", "time_ctx"],
            output_names=["band_masked_output", "new_kv", "new_valid_mask", "new_time_ctx"],
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
