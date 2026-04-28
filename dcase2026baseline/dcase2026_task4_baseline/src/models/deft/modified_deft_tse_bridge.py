from __future__ import annotations

import torch
import torch.nn as nn

from .modified_deft import ModifiedDeFTTSEMemoryEfficientTemporal


class BridgeModifiedDeFTTSEMemoryEfficientTemporal(ModifiedDeFTTSEMemoryEfficientTemporal):
    def __init__(
        self,
        *args,
        label_dim: int = 18,
        bridge_condition_dim: int = 256,
        bridge_hidden_dim: int = 256,
        bridge_label_scale: float = 0.5,
        bridge_dropout: float = 0.0,
        require_bridge_condition: bool = False,
        **kwargs,
    ):
        super().__init__(*args, label_dim=label_dim, **kwargs)
        self.label_dim = int(label_dim)
        self.bridge_condition_dim = int(bridge_condition_dim)
        self.bridge_label_scale = float(bridge_label_scale)
        self.require_bridge_condition = bool(require_bridge_condition)
        self.bridge_to_label = nn.Sequential(
            nn.LayerNorm(self.bridge_condition_dim),
            nn.Linear(self.bridge_condition_dim, int(bridge_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(bridge_dropout)),
            nn.Linear(int(bridge_hidden_dim), self.label_dim),
        )

    def _get_bridge(self, input_dict):
        for key in ("bridge_condition", "tse_condition", "proposal_condition"):
            if key in input_dict:
                return input_dict[key]
        return None

    def _match_bridge_shape(self, bridge: torch.Tensor, label_vector: torch.Tensor) -> torch.Tensor:
        bridge = bridge.to(device=label_vector.device, dtype=label_vector.dtype)
        if label_vector.dim() == 2:
            if bridge.dim() == 3:
                bridge = bridge[:, 0]
            if bridge.dim() != 2:
                raise ValueError("bridge_condition must be [B,D] or [B,S,D]")
            return bridge
        if label_vector.dim() == 3:
            if bridge.dim() == 2:
                bridge = bridge.unsqueeze(1).expand(-1, label_vector.shape[1], -1)
            if bridge.dim() != 3:
                raise ValueError("bridge_condition must be [B,D] or [B,S,D]")
            if bridge.shape[1] < label_vector.shape[1]:
                pad = bridge.new_zeros(bridge.shape[0], label_vector.shape[1] - bridge.shape[1], bridge.shape[-1])
                bridge = torch.cat([bridge, pad], dim=1)
            elif bridge.shape[1] > label_vector.shape[1]:
                bridge = bridge[:, : label_vector.shape[1]]
            return bridge
        raise ValueError(f"Unsupported label_vector shape: {tuple(label_vector.shape)}")

    def _prepare_input(self, input_dict):
        bridge = self._get_bridge(input_dict)
        if bridge is None:
            if self.require_bridge_condition:
                raise ValueError("bridge_condition is required")
            return input_dict
        label_vector = input_dict["label_vector"]
        bridge = self._match_bridge_shape(bridge, label_vector)
        if bridge.shape[-1] < self.bridge_condition_dim:
            pad = bridge.new_zeros(*bridge.shape[:-1], self.bridge_condition_dim - bridge.shape[-1])
            bridge = torch.cat([bridge, pad], dim=-1)
        elif bridge.shape[-1] > self.bridge_condition_dim:
            bridge = bridge[..., : self.bridge_condition_dim]
        delta = torch.tanh(self.bridge_to_label(bridge)) * self.bridge_label_scale
        out = dict(input_dict)
        out["label_vector"] = label_vector + delta
        out["bridge_label_delta"] = delta
        return out

    def forward(self, input_dict):
        return super().forward(self._prepare_input(input_dict))
