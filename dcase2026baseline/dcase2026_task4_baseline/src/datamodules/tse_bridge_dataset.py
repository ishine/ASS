import os

import torch

from src.datamodules.tse_dataset import EstimatedEnrollmentTSEDataset, TSEDataset


class USSBridgeFeatureMixin:
    """Mixin that loads per-soundscape USS bridge features from .pt files.

    Expected file layout:
        bridge_feature_dir/<soundscape>.pt

    The .pt file is produced by ``src/tools/export_uss_bridge_features.py`` and
    may contain:
        tse_condition:                [S, D]
        foreground_embedding:         [S, D]
        foreground_audio_embedding:   [S, D]
        pred_doa_vector:              [S, 3]
        used_spatial_vector:          [S, 3]
        proposal_quality_logits:      [S]
        class_logits:                 [S, C]
        silence_logits:               [S]
    """

    def _init_bridge_feature_loader(
        self,
        bridge_feature_dir=None,
        bridge_feature_key="tse_condition",
        bridge_feature_dim=256,
        require_bridge_feature=False,
    ):
        self.bridge_feature_dir = bridge_feature_dir
        self.bridge_feature_key = bridge_feature_key
        self.bridge_feature_dim = int(bridge_feature_dim)
        self.require_bridge_feature = bool(require_bridge_feature)

    def _soundscape_name_from_item(self, item, idx):
        if "soundscape" in item:
            return item["soundscape"]
        if hasattr(self.base_dataset, "data"):
            info = self.base_dataset.data[idx]
            if isinstance(info, dict) and "soundscape" in info:
                return info["soundscape"]
        return f"soundscape_{idx:08d}"

    def _zero_bridge_condition(self):
        if not hasattr(self.base_dataset, "n_sources"):
            raise ValueError("base_dataset must expose n_sources for bridge feature padding")
        return torch.zeros(self.base_dataset.n_sources, self.bridge_feature_dim, dtype=torch.float32)

    def _load_bridge_condition(self, soundscape):
        if self.bridge_feature_dir is None:
            if self.require_bridge_feature:
                raise ValueError("bridge_feature_dir is required when require_bridge_feature=True")
            return self._zero_bridge_condition()

        path = os.path.join(self.bridge_feature_dir, f"{soundscape}.pt")
        if not os.path.exists(path):
            if self.require_bridge_feature:
                raise FileNotFoundError(f"Missing USS bridge feature file: {path}")
            return self._zero_bridge_condition()

        obj = torch.load(path, map_location="cpu")
        if self.bridge_feature_key not in obj:
            if self.require_bridge_feature:
                raise KeyError(f"{path} does not contain key '{self.bridge_feature_key}'")
            return self._zero_bridge_condition()

        bridge = obj[self.bridge_feature_key].to(torch.float32)
        if bridge.dim() != 2:
            raise ValueError(f"Bridge feature must have shape [S,D], got {tuple(bridge.shape)} from {path}")

        n_sources = self.base_dataset.n_sources
        if bridge.shape[0] < n_sources:
            pad = bridge.new_zeros(n_sources - bridge.shape[0], bridge.shape[-1])
            bridge = torch.cat([bridge, pad], dim=0)
        elif bridge.shape[0] > n_sources:
            bridge = bridge[:n_sources]

        if bridge.shape[-1] < self.bridge_feature_dim:
            pad = bridge.new_zeros(n_sources, self.bridge_feature_dim - bridge.shape[-1])
            bridge = torch.cat([bridge, pad], dim=-1)
        elif bridge.shape[-1] > self.bridge_feature_dim:
            bridge = bridge[:, : self.bridge_feature_dim]
        return bridge

    def _attach_bridge_condition(self, item, idx):
        soundscape = self._soundscape_name_from_item(item, idx)
        item["soundscape"] = soundscape
        item["bridge_condition"] = self._load_bridge_condition(soundscape)
        return item


class BridgeTSEDataset(USSBridgeFeatureMixin, TSEDataset):
    """Oracle-enrollment TSE dataset with optional USS bridge conditions."""

    def __init__(
        self,
        base_dataset,
        bridge_feature_dir=None,
        bridge_feature_key="tse_condition",
        bridge_feature_dim=256,
        require_bridge_feature=False,
    ):
        super().__init__(base_dataset=base_dataset)
        self._init_bridge_feature_loader(
            bridge_feature_dir=bridge_feature_dir,
            bridge_feature_key=bridge_feature_key,
            bridge_feature_dim=bridge_feature_dim,
            require_bridge_feature=require_bridge_feature,
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return self._attach_bridge_condition(item, idx)


class BridgeEstimatedEnrollmentTSEDataset(USSBridgeFeatureMixin, EstimatedEnrollmentTSEDataset):
    """Estimated-enrollment TSE dataset with optional USS bridge conditions.

    This is the recommended USS -> TSE opt-in path: enrollment waveform comes
    from USS/S5 estimates, while ``bridge_condition`` carries the semantic,
    acoustic, class-probability and spatial proposal vector exported by the USS
    bridge model.
    """

    def __init__(
        self,
        base_dataset,
        label_source="oracle",
        crop_seconds=None,
        random_crop=True,
        require_estimate_for_active=True,
        bridge_feature_dir=None,
        bridge_feature_key="tse_condition",
        bridge_feature_dim=256,
        require_bridge_feature=False,
    ):
        super().__init__(
            base_dataset=base_dataset,
            label_source=label_source,
            crop_seconds=crop_seconds,
            random_crop=random_crop,
            require_estimate_for_active=require_estimate_for_active,
        )
        self._init_bridge_feature_loader(
            bridge_feature_dir=bridge_feature_dir,
            bridge_feature_key=bridge_feature_key,
            bridge_feature_dim=bridge_feature_dim,
            require_bridge_feature=require_bridge_feature,
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return self._attach_bridge_condition(item, idx)
