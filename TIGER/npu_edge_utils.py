"""
Utilities for TV edge NPU constraints (see prj_context.md):
- operator hygiene (ReLU vs PReLU, no Dropout in deploy graph)
- receptive field checks for Conv / pooling
- streaming state byte budget (fp32 default)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn


def sanitize_for_npu_edge(root: nn.Module) -> nn.Module:
    """
    Replace common unsupported / problematic modules for static export:
    - PReLU -> ReLU
    - Dropout -> Identity
    """

    def _walk(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, nn.PReLU):
                setattr(m, name, nn.ReLU(inplace=False))
            elif isinstance(child, nn.Dropout):
                setattr(m, name, nn.Identity())
            else:
                _walk(child)

    _walk(root)
    return root


def _pair(x: Any) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise TypeError(f"expected int or pair, got {type(x)}: {x!r}")


@dataclass
class ConvConstraintViolation:
    module_name: str
    cls: str
    kernel: Tuple[int, int]
    dilation: Tuple[int, int]
    axis: str
    effective: int
    limit: int


def collect_conv_constraint_violations(
    root: nn.Module,
    limit: int = 14,
) -> List[ConvConstraintViolation]:
    """
    Enforce (kernel_size - 1) * dilation < limit on each spatial axis (height, width).
    Matches prj_context: conv2d (kernel_size-1)*dilation < 14.
    """
    violations: List[ConvConstraintViolation] = []

    for name, mod in root.named_modules():
        if isinstance(mod, nn.Conv2d):
            kh, kw = _pair(mod.kernel_size)
            dh, dw = _pair(mod.dilation)
            eff_h = (kh - 1) * dh
            eff_w = (kw - 1) * dw
            if eff_h >= limit:
                violations.append(
                    ConvConstraintViolation(name, mod.__class__.__name__, (kh, kw), (dh, dw), "H", eff_h, limit)
                )
            if eff_w >= limit:
                violations.append(
                    ConvConstraintViolation(name, mod.__class__.__name__, (kh, kw), (dh, dw), "W", eff_w, limit)
                )
        elif isinstance(mod, nn.Conv1d):
            ks = mod.kernel_size
            k = int(ks[0]) if isinstance(ks, tuple) else int(ks)
            ds = mod.dilation
            d = int(ds[0]) if isinstance(ds, tuple) else int(ds)
            eff = (k - 1) * d
            if eff >= limit:
                violations.append(
                    ConvConstraintViolation(name, mod.__class__.__name__, (k, 1), (d, 1), "T", eff, limit)
                )
        elif isinstance(mod, nn.AvgPool2d):
            kh, kw = _pair(mod.kernel_size)
            dh, dw = _pair(getattr(mod, "dilation", (1, 1)))
            eff_h = (kh - 1) * dh
            eff_w = (kw - 1) * dw
            if eff_h >= limit:
                violations.append(
                    ConvConstraintViolation(name, mod.__class__.__name__, (kh, kw), (dh, dw), "H", eff_h, limit)
                )
            if eff_w >= limit:
                violations.append(
                    ConvConstraintViolation(name, mod.__class__.__name__, (kh, kw), (dh, dw), "W", eff_w, limit)
                )

    return violations


def assert_conv_constraints(root: nn.Module, limit: int = 14) -> None:
    bad = collect_conv_constraint_violations(root, limit=limit)
    if bad:
        lines = [f"{v.module_name} ({v.cls}) axis={v.axis} (k-1)*d={v.effective} >= {v.limit}" for v in bad]
        raise AssertionError("Conv / AvgPool receptive field violations:\n" + "\n".join(lines))


def streaming_state_bytes_fp32(state_tuple: Iterable[torch.Tensor]) -> int:
    total = 0
    for t in state_tuple:
        total += int(t.numel()) * 4
    return total


def streaming_state_elements(state_tuple: Iterable[torch.Tensor]) -> int:
    return sum(int(t.numel()) for t in state_tuple)
