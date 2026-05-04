"""Opt-in foreground count heads for modified DeFT USS models.

DCASE 2026 Task 4 includes zero-target mixtures and permits up to three
foreground target events.  The waveform/PIT objective alone does not directly
teach the USS stage to decide whether the mixture contains 0, 1, 2, or 3
foreground targets.  This module adds a small config-level opt-in count head
without changing the public output schema used by the existing USS losses and
S5 pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.deft.modified_deft_bf16_safe import (
    ModifiedDeFTUSSMemoryEfficient as _BF16SafeModifiedDeFTUSSMemoryEfficient,
    ModifiedDeFTUSSMemoryEfficientTemporal as _BF16SafeModifiedDeFTUSSMemoryEfficientTemporal,
)


class ForegroundCountHead(nn.Module):
    """Predict foreground count from per-slot class/activity summaries.

    Input features per foreground slot:
        - class logits for the 18 foreground classes
        - active probability from the historical ``silence_logits`` tensor
        - waveform RMS for that USS foreground slot

    Output classes are count = 0, 1, 2, 3 by default.
    """

    def __init__(self, n_foreground: int = 3, n_classes: int = 18, hidden_dim: int = 64, max_count: int = 3):
        super().__init__()
        self.n_foreground = int(n_foreground)
        self.n_classes = int(n_classes)
        self.max_count = int(max_count)
        in_dim = self.n_foreground * (self.n_classes + 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.max_count + 1),
        )

    def forward(self, output_dict):
        class_logits = output_dict["class_logits"].float()
        active_prob = output_dict["silence_logits"].float().sigmoid().unsqueeze(-1)
        waveform_rms = output_dict["foreground_waveform"][:, :, 0].float().pow(2).mean(dim=-1).sqrt().unsqueeze(-1)
        features = torch.cat([class_logits, active_prob, waveform_rms], dim=-1)
        return self.net(features.reshape(features.shape[0], -1))


class _USSCountHeadMixin:
    def _init_count_head(self, n_foreground: int, n_classes: int, count_hidden_dim: int, max_count: int) -> None:
        self.count_head = ForegroundCountHead(
            n_foreground=n_foreground,
            n_classes=n_classes,
            hidden_dim=count_hidden_dim,
            max_count=max_count,
        )

    def _add_count_logits(self, output_dict):
        output_dict = dict(output_dict)
        output_dict["count_logits"] = self.count_head(output_dict)
        return output_dict


class ModifiedDeFTUSSMemoryEfficientCountHead(
    _USSCountHeadMixin,
    _BF16SafeModifiedDeFTUSSMemoryEfficient,
):
    """BF16-safe memory-efficient USS with opt-in foreground count head."""

    def __init__(
        self,
        *args,
        n_foreground: int = 3,
        n_classes: int = 18,
        count_hidden_dim: int = 64,
        max_count: int = 3,
        **kwargs,
    ):
        super().__init__(*args, n_foreground=n_foreground, n_classes=n_classes, **kwargs)
        self._init_count_head(
            n_foreground=n_foreground,
            n_classes=n_classes,
            count_hidden_dim=count_hidden_dim,
            max_count=max_count,
        )

    def forward(self, input_dict):
        return self._add_count_logits(super().forward(input_dict))


class ModifiedDeFTUSSMemoryEfficientTemporalCountHead(
    _USSCountHeadMixin,
    _BF16SafeModifiedDeFTUSSMemoryEfficientTemporal,
):
    """Temporal memory-efficient USS with opt-in foreground count head."""

    def __init__(
        self,
        *args,
        n_foreground: int = 3,
        n_classes: int = 18,
        count_hidden_dim: int = 64,
        max_count: int = 3,
        **kwargs,
    ):
        super().__init__(*args, n_foreground=n_foreground, n_classes=n_classes, **kwargs)
        self._init_count_head(
            n_foreground=n_foreground,
            n_classes=n_classes,
            count_hidden_dim=count_hidden_dim,
            max_count=max_count,
        )

    def forward(self, input_dict):
        return self._add_count_logits(super().forward(input_dict))
