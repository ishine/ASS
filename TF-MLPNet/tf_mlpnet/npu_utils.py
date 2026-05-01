"""Helpers aligned with ``prj_context.md`` (streaming state budget, conv RF)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def streaming_state_bytes_fp32(states: Sequence[torch.Tensor]) -> int:
    """Sum of numel * element size for fp32 tensors (matches TIGER ``npu_edge_utils`` idea)."""
    total = 0
    for t in states:
        if t.dtype == torch.float32:
            total += t.numel() * 4
        else:
            total += t.numel() * t.element_size()
    return int(total)


def streaming_state_elements(states: Sequence[torch.Tensor]) -> int:
    return int(sum(s.numel() for s in states))


def assert_conv_receptive_field(model: nn.Module, limit: int = 14) -> None:
    """Assert (kernel_size - 1) * dilation < limit for Conv2d / AvgPool2d (prj_context)."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            for axis, k, d in (
                ("H", m.kernel_size[0], m.dilation[0]),
                ("W", m.kernel_size[1], m.dilation[1]),
            ):
                eff = (k - 1) * d
                if eff >= limit:
                    raise AssertionError(
                        f"{name}: Conv2d axis {axis} (k-1)*d = {eff} >= {limit}"
                    )
        if isinstance(m, nn.AvgPool2d):
            if m.kernel_size is None:
                continue
            kh, kw = (
                m.kernel_size
                if isinstance(m.kernel_size, tuple)
                else (m.kernel_size, m.kernel_size)
            )
            for axis, k in (("H", kh), ("W", kw)):
                eff = k - 1
                if eff >= limit:
                    raise AssertionError(
                        f"{name}: AvgPool2d axis {axis} (k-1) = {eff} >= {limit}"
                    )
