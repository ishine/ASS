"""Opt-in residual/trash-slot wrappers for modified DeFT USS models.

These classes keep the public foreground interface unchanged while allocating one
or more extra non-foreground slots to absorb leftover, leaked, or label-noise
energy.  The residual/trash slots are carved out of the internal interference
object bank and exposed as ``residual_waveform``.  They should not be submitted
as foreground estimates and should not be passed to the SC/TSE stages.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.models.deft.modified_deft_bf16_safe import (
    ModifiedDeFTUSSMemoryEfficient,
    ModifiedDeFTUSSMemoryEfficientTemporal,
)


class _ResidualSlotMixin:
    """Post-process USS outputs to expose residual/trash slots separately.

    Parent USS classes already know how to predict ``n_interference`` generic
    non-foreground objects.  We instantiate the parent with
    ``task_n_interference + n_residual`` and then split the last ``n_residual``
    non-foreground objects out as residual/trash outputs.
    """

    task_n_interference: int
    n_residual: int

    def _split_residual_slots(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.n_residual <= 0:
            output["residual_waveform"] = output["foreground_waveform"].new_zeros(
                output["foreground_waveform"].shape[0],
                0,
                output["foreground_waveform"].shape[2],
                output["foreground_waveform"].shape[3],
            )
            return output

        interference = output["interference_waveform"]
        expected = self.task_n_interference + self.n_residual
        if interference.shape[1] < expected:
            raise RuntimeError(
                "Residual wrapper expected at least "
                f"{expected} non-foreground slots, got {interference.shape[1]}"
            )

        residual_start = self.task_n_interference
        residual_end = residual_start + self.n_residual
        output["interference_waveform"] = interference[:, : self.task_n_interference]
        output["residual_waveform"] = interference[:, residual_start:residual_end]

        if "interference_activity_logits" in output:
            activity = output["interference_activity_logits"]
            output["interference_activity_logits"] = activity[:, : self.task_n_interference]
            output["residual_activity_logits"] = activity[:, residual_start:residual_end]

        return output

    def forward(self, input_dict):  # type: ignore[override]
        output = super().forward(input_dict)
        return self._split_residual_slots(output)


class ModifiedDeFTUSSMemoryEfficientResidual(_ResidualSlotMixin, ModifiedDeFTUSSMemoryEfficient):
    """Memory-efficient spatial USS with opt-in residual/trash slots.

    Args:
        n_interference: Number of regular interference slots kept in
            ``output["interference_waveform"]``.
        n_residual: Number of extra residual/trash slots exposed in
            ``output["residual_waveform"]``. These are internal non-foreground
            slots and should not be classified or submitted as foreground.
    """

    def __init__(self, *args, n_interference: int = 2, n_residual: int = 1, **kwargs):
        task_n_interference = int(n_interference)
        n_residual = int(n_residual)
        if task_n_interference < 0 or n_residual < 0:
            raise ValueError("n_interference and n_residual must be non-negative")
        super().__init__(
            *args,
            n_interference=task_n_interference + n_residual,
            **kwargs,
        )
        self.task_n_interference = task_n_interference
        self.n_residual = n_residual


class ModifiedDeFTUSSMemoryEfficientTemporalResidual(
    _ResidualSlotMixin,
    ModifiedDeFTUSSMemoryEfficientTemporal,
):
    """Temporal memory-efficient spatial USS with residual/trash slots."""

    def __init__(self, *args, n_interference: int = 2, n_residual: int = 1, **kwargs):
        task_n_interference = int(n_interference)
        n_residual = int(n_residual)
        if task_n_interference < 0 or n_residual < 0:
            raise ValueError("n_interference and n_residual must be non-negative")
        super().__init__(
            *args,
            n_interference=task_n_interference + n_residual,
            **kwargs,
        )
        self.task_n_interference = task_n_interference
        self.n_residual = n_residual
