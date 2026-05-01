"""Opt-in semantic-acoustic USS bridge models for DeFT.

The classes in this module intentionally leave ``modified_deft.py`` unchanged.
They add a stronger USS -> TSE bridge by exporting slot-level proposal features:

    foreground_embedding       semantic slot embedding for class/prototype loss
    foreground_audio_embedding acoustic slot embedding aligned to waveform branch
    prototype_logits           class-prototype logits from proposal embeddings
    pred_doa_vector            optional slot-level DoA / spatial proposal
    used_spatial_vector        spatial clue used to condition object features
    tse_condition              normalized TSE-ready proposal condition vector

Existing input/output contracts remain valid. A plain call
``model({"mixture": mixture})`` works. During training, callers may optionally
pass ``spatial_vector`` derived from oracle event_position; the model can mix it
with predicted DoA using scheduled sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modified_deft import ModifiedDeFTUSSSpatialTemporal


class ObjectEmbeddingHead(nn.Module):
    def __init__(self, channels: int, emb_dim: int = 256, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(channels, emb_dim)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        batch_size, n_objects, channels, time_steps, freq_bins = object_features.shape
        x = object_features.reshape(batch_size * n_objects, channels, time_steps, freq_bins)
        emb = self.net(x)
        emb = F.normalize(emb, dim=-1)
        return emb.view(batch_size, n_objects, -1)


class ClassPrototypeHead(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int, scale: float = 10.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, emb_dim))
        self.scale = float(scale)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return self.scale * (emb @ prototypes.t())


class ObjectDoAHead(nn.Module):
    def __init__(self, channels: int, spatial_dim: int = 3, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(channels // 2, 32)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spatial_dim),
        )

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        batch_size, n_objects, channels, time_steps, freq_bins = object_features.shape
        x = object_features.reshape(batch_size * n_objects, channels, time_steps, freq_bins)
        doa = self.net(x)
        doa = F.normalize(doa, dim=-1)
        return doa.view(batch_size, n_objects, -1)


class ObjectSpatialConditioner(nn.Module):
    def __init__(self, spatial_dim: int, channels: int):
        super().__init__()
        self.beta = nn.Linear(spatial_dim, channels)
        self.gamma = nn.Linear(spatial_dim, channels)

    def forward(self, spatial_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta(spatial_vector)[:, :, :, None, None]
        gamma = self.gamma(spatial_vector)[:, :, :, None, None]
        return beta, gamma


def _normalize_or_pad_spatial_vector(
    spatial_vector: torch.Tensor,
    batch_size: int,
    n_objects: int,
    spatial_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    spatial_vector = spatial_vector.to(device=device, dtype=dtype)
    if spatial_vector.dim() == 2:
        if spatial_vector.shape[1] == spatial_dim:
            spatial_vector = spatial_vector.unsqueeze(1).expand(-1, n_objects, -1)
        elif spatial_vector.shape[1] % n_objects == 0:
            spatial_vector = spatial_vector.view(batch_size, n_objects, -1)
        else:
            raise ValueError("spatial_vector rank-2 shape must be [B,D] or [B,N*D]")
    elif spatial_vector.dim() != 3:
        raise ValueError("spatial_vector must have shape [B,D], [B,N,D], or [B,N*D]")

    if spatial_vector.shape[0] != batch_size:
        raise ValueError("spatial_vector batch dimension does not match mixture")
    if spatial_vector.shape[1] < n_objects:
        pad = spatial_vector.new_zeros(batch_size, n_objects - spatial_vector.shape[1], spatial_vector.shape[-1])
        spatial_vector = torch.cat([spatial_vector, pad], dim=1)
    elif spatial_vector.shape[1] > n_objects:
        spatial_vector = spatial_vector[:, :n_objects]

    cur_dim = spatial_vector.shape[-1]
    if cur_dim < spatial_dim:
        pad = spatial_vector.new_zeros(batch_size, n_objects, spatial_dim - cur_dim)
        spatial_vector = torch.cat([spatial_vector, pad], dim=-1)
    elif cur_dim > spatial_dim:
        spatial_vector = spatial_vector[..., :spatial_dim]

    norm = spatial_vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return spatial_vector / norm


def _foa_intensity_fallback(mixture: torch.Tensor, n_objects: int, spatial_dim: int) -> torch.Tensor:
    batch_size = mixture.shape[0]
    if mixture.shape[1] < 4:
        return mixture.new_zeros(batch_size, n_objects, spatial_dim)
    w = mixture[:, 0].float()
    xyz = mixture[:, 1:4].float()
    direction = (w[:, None, :] * xyz).mean(dim=-1)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = direction.to(device=mixture.device, dtype=mixture.dtype)
    if spatial_dim < 3:
        direction = direction[:, :spatial_dim]
    elif spatial_dim > 3:
        pad = direction.new_zeros(batch_size, spatial_dim - 3)
        direction = torch.cat([direction, pad], dim=-1)
    return direction.unsqueeze(1).expand(batch_size, n_objects, spatial_dim)


class SemanticAcousticBridgeMixin:
    def _init_bridge(
        self,
        object_feature_channels: int,
        n_classes: int,
        embedding_dim: int = 256,
        prototype_scale: float = 10.0,
        use_audio_embedding: bool = True,
        use_doa_head: bool = True,
        use_spatial_conditioning: bool = True,
        spatial_dim: int = 3,
        spatial_conditioning_scale: float = 1.0,
        predicted_spatial_prob: float = 0.0,
        spatial_mix_fallback_prob: float = 0.0,
        detach_predicted_spatial_for_condition: bool = False,
        tse_condition_dim: int = 256,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.use_audio_embedding = bool(use_audio_embedding)
        self.use_doa_head = bool(use_doa_head)
        self.use_spatial_conditioning = bool(use_spatial_conditioning)
        self.spatial_dim = int(spatial_dim)
        self.spatial_conditioning_scale = float(spatial_conditioning_scale)
        self.predicted_spatial_prob = float(predicted_spatial_prob)
        self.spatial_mix_fallback_prob = float(spatial_mix_fallback_prob)
        self.detach_predicted_spatial_for_condition = bool(detach_predicted_spatial_for_condition)

        self.semantic_embedding_head = ObjectEmbeddingHead(object_feature_channels, emb_dim=embedding_dim)
        self.audio_embedding_head = ObjectEmbeddingHead(object_feature_channels, emb_dim=embedding_dim)
        self.prototype_head = ClassPrototypeHead(n_classes=n_classes, emb_dim=embedding_dim, scale=prototype_scale)
        self.doa_head = ObjectDoAHead(object_feature_channels, spatial_dim=spatial_dim)
        self.spatial_conditioner = ObjectSpatialConditioner(spatial_dim=spatial_dim, channels=object_feature_channels)

        # TSE bridge projection consumes [semantic emb, audio emb, class probs, doa].
        self.tse_condition_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2 + n_classes + spatial_dim, tse_condition_dim),
            nn.LayerNorm(tse_condition_dim),
            nn.ReLU(),
            nn.Linear(tse_condition_dim, tse_condition_dim),
            nn.LayerNorm(tse_condition_dim),
        )

    def set_predicted_spatial_prob(self, value: float) -> None:
        self.predicted_spatial_prob = float(value)

    def _get_oracle_spatial_vector(self, input_dict: dict, mixture: torch.Tensor, n_objects: int) -> torch.Tensor | None:
        spatial_vector = input_dict.get("spatial_vector", None)
        if spatial_vector is None:
            spatial_vector = input_dict.get("spatial_clue", None)
        if spatial_vector is None:
            spatial_vector = input_dict.get("doa_vector", None)
        if spatial_vector is None:
            return None
        return _normalize_or_pad_spatial_vector(
            spatial_vector=spatial_vector,
            batch_size=mixture.shape[0],
            n_objects=n_objects,
            spatial_dim=self.spatial_dim,
            device=mixture.device,
            dtype=mixture.dtype,
        )

    def _choose_spatial_condition(self, input_dict: dict, mixture: torch.Tensor, pred_doa_vector: torch.Tensor) -> torch.Tensor:
        batch_size, n_objects, _ = pred_doa_vector.shape
        oracle = self._get_oracle_spatial_vector(input_dict, mixture, n_objects)
        predicted = pred_doa_vector if self.use_doa_head else _foa_intensity_fallback(mixture, n_objects, self.spatial_dim)
        predicted_for_condition = predicted.detach() if self.detach_predicted_spatial_for_condition else predicted

        if not self.training:
            return predicted_for_condition
        if oracle is None:
            return predicted_for_condition

        p_pred = float(self.predicted_spatial_prob)
        if p_pred <= 0.0:
            mixed = oracle
        elif p_pred >= 1.0:
            mixed = predicted_for_condition
        else:
            use_pred = torch.rand(batch_size, n_objects, 1, device=mixture.device) < p_pred
            mixed = torch.where(use_pred, predicted_for_condition, oracle)

        if self.spatial_mix_fallback_prob > 0.0:
            fallback = _foa_intensity_fallback(mixture, n_objects, self.spatial_dim)
            use_fb = torch.rand(batch_size, n_objects, 1, device=mixture.device) < self.spatial_mix_fallback_prob
            mixed = torch.where(use_fb, fallback, mixed)
        return F.normalize(mixed, dim=-1)

    def _apply_spatial_conditioning(self, object_features: torch.Tensor, spatial_vector: torch.Tensor) -> torch.Tensor:
        if not self.use_spatial_conditioning:
            return object_features
        beta, gamma = self.spatial_conditioner(spatial_vector)
        scale = self.spatial_conditioning_scale
        return object_features * (1.0 + scale * gamma) + scale * beta

    def _bridge_outputs(self, object_features: torch.Tensor, class_logits: torch.Tensor, used_spatial_vector: torch.Tensor) -> dict:
        semantic_emb = self.semantic_embedding_head(object_features)
        audio_emb = self.audio_embedding_head(object_features) if self.use_audio_embedding else semantic_emb
        fg_semantic = semantic_emb[:, : self.n_foreground]
        fg_audio = audio_emb[:, : self.n_foreground]
        fg_doa = used_spatial_vector[:, : self.n_foreground]
        prototype_logits = self.prototype_head(fg_semantic)
        class_probs = torch.softmax(class_logits, dim=-1)
        tse_in = torch.cat([fg_semantic, fg_audio, class_probs, fg_doa], dim=-1)
        tse_condition = F.normalize(self.tse_condition_proj(tse_in), dim=-1)
        return {
            "object_embedding": semantic_emb,
            "object_audio_embedding": audio_emb,
            "foreground_embedding": fg_semantic,
            "foreground_audio_embedding": fg_audio,
            "prototype_logits": prototype_logits,
            "tse_condition": tse_condition,
        }


class SemanticBridgeModifiedDeFTUSSSpatialTemporal(SemanticAcousticBridgeMixin, ModifiedDeFTUSSSpatialTemporal):
    """Most-capable opt-in ModifiedDeFT USS bridge for downstream TSE.

    It preserves the original input/output keys while adding extra proposal keys.
    """

    def __init__(
        self,
        *args,
        hidden_channels: int = 96,
        sample_rate: int = 32000,
        embedding_dim: int = 256,
        prototype_scale: float = 10.0,
        use_audio_embedding: bool = True,
        use_doa_head: bool = True,
        use_spatial_conditioning: bool = True,
        spatial_dim: int = 3,
        spatial_conditioning_scale: float = 1.0,
        predicted_spatial_prob: float = 0.0,
        spatial_mix_fallback_prob: float = 0.0,
        detach_predicted_spatial_for_condition: bool = False,
        tse_condition_dim: int = 256,
        **kwargs,
    ):
        super().__init__(*args, hidden_channels=hidden_channels, sample_rate=sample_rate, **kwargs)
        self._init_bridge(
            object_feature_channels=hidden_channels,
            n_classes=self.n_classes,
            embedding_dim=embedding_dim,
            prototype_scale=prototype_scale,
            use_audio_embedding=use_audio_embedding,
            use_doa_head=use_doa_head,
            use_spatial_conditioning=use_spatial_conditioning,
            spatial_dim=spatial_dim,
            spatial_conditioning_scale=spatial_conditioning_scale,
            predicted_spatial_prob=predicted_spatial_prob,
            spatial_mix_fallback_prob=spatial_mix_fallback_prob,
            detach_predicted_spatial_for_condition=detach_predicted_spatial_for_condition,
            tse_condition_dim=tse_condition_dim,
        )

    def _forward_full(self, input_dict: dict) -> dict:
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(torch.cat([real, imag], dim=1))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)

        pred_doa_vector = self.doa_head(x)
        used_spatial_vector = self._choose_spatial_condition(input_dict, mixture, pred_doa_vector)
        x = self._apply_spatial_conditioning(x, used_spatial_vector)

        waveform = self._spatial_mask_to_waveform(x, real, imag, samples)
        activity_logits = self._activity_logits(x)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))

        bridge = self._bridge_outputs(x, class_logits, used_spatial_vector)
        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
            "foreground_activity_logits": activity_logits[:, : self.n_foreground],
            "interference_activity_logits": activity_logits[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_activity_logits": activity_logits[:, -1:],
            "duration_sec": duration_sec,
            "pred_doa_vector": pred_doa_vector[:, : self.n_foreground],
            "used_spatial_vector": used_spatial_vector[:, : self.n_foreground],
            **bridge,
        }

    def forward(self, input_dict: dict) -> dict:
        return self._forward_full(input_dict)
