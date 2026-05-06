"""Per-slot spatial heads for modified DeFT USS models.

The classes in this module are opt-in wrappers around the existing FOA spatial
feature + count-head USS models.  They do not change the separation graph.
Instead, they capture the object features produced by ``object_conv`` and add
foreground-slot spatial outputs:

    spatial_embedding: [B, n_foreground, D]
    doa_vector:        [B, n_foreground, 3], L2-normalized xyz direction

These outputs are consumed by optional losses in ``uss_loss.py`` to reduce
same-class slot collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deft.foa_spatial_features import ModifiedDeFTUSSMemoryEfficientSpatialFeaturesCountHead


class ForegroundSpatialHead(nn.Module):
    """Predict a per-slot spatial embedding and unit DoA vector from slot features."""

    def __init__(self, hidden_channels: int = 96, embedding_dim: int = 16):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, self.embedding_dim),
        )
        self.doa = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, foreground_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            foreground_features: [B, n_foreground, C, T, F]

        Returns:
            spatial_embedding: [B, n_foreground, embedding_dim], L2-normalized
            doa_vector:        [B, n_foreground, 3], L2-normalized
        """

        batch_size, n_foreground, channels, time_steps, freq_bins = foreground_features.shape
        pooled = self.pool(foreground_features.reshape(batch_size * n_foreground, channels, time_steps, freq_bins))
        spatial_embedding = self.embedding(pooled).view(batch_size, n_foreground, self.embedding_dim)
        doa_vector = self.doa(pooled).view(batch_size, n_foreground, 3)
        return F.normalize(spatial_embedding, dim=-1, eps=1e-8), F.normalize(doa_vector, dim=-1, eps=1e-8)


class _ObjectFeatureCaptureMixin:
    """Capture object_conv outputs from existing DeFT forward implementations."""

    def _capture_object_features_during_forward(self, input_dict):
        captured = []

        def _hook(_module, _inputs, output):
            captured.append(output)

        handle = self.object_conv.register_forward_hook(_hook)
        try:
            output = super().forward(input_dict)
        finally:
            handle.remove()

        if not captured:
            return output, None
        object_tensor = captured[-1]
        batch_size, _, time_steps, freq_bins = object_tensor.shape
        object_features = object_tensor.view(batch_size, self.n_objects, -1, time_steps, freq_bins)
        return output, object_features


class ModifiedDeFTUSSMemoryEfficientSpatialFeaturesCountSpatialHead(
    _ObjectFeatureCaptureMixin,
    ModifiedDeFTUSSMemoryEfficientSpatialFeaturesCountHead,
):
    """FOA spatial-feature count-head USS with per-foreground-slot spatial heads."""

    def __init__(
        self,
        *args,
        hidden_channels: int = 96,
        n_foreground: int = 3,
        spatial_embedding_dim: int = 16,
        **kwargs,
    ):
        super().__init__(*args, hidden_channels=hidden_channels, n_foreground=n_foreground, **kwargs)
        self.spatial_head = ForegroundSpatialHead(
            hidden_channels=hidden_channels,
            embedding_dim=spatial_embedding_dim,
        )

    def forward(self, input_dict):
        output, object_features = self._capture_object_features_during_forward(input_dict)
        if object_features is None:
            return output
        foreground_features = object_features[:, : self.n_foreground]
        spatial_embedding, doa_vector = self.spatial_head(foreground_features)
        output = dict(output)
        output["spatial_embedding"] = spatial_embedding
        output["doa_vector"] = doa_vector
        return output
