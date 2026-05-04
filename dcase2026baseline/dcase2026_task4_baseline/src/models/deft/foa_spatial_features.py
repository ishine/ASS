"""Opt-in FOA spatial feature frontends for modified DeFT USS models.

This module keeps the existing DeFT forward graphs unchanged. The base DeFT
models already pass raw complex FOA STFT features to ``self.encoder`` as
``[real(WXYZ), imag(WXYZ)]``. The opt-in encoder wrapper below accepts that
same tensor, appends explicit FOA spatial cues, and then applies the same
Conv2d-BN-GELU stem shape as the original encoder.

For 4-channel FOA input the default feature set is:

    raw complex FOA:  8 channels  [Re(WXYZ), Im(WXYZ)]
    log magnitude:    4 channels
    normalized AIV:   3 channels  Re(conj(W) * [X, Y, Z]) / FOA energy
    normalized IPD:   6 channels  cos/sin angle([X, Y, Z]) - angle(W)

Total: 21 encoder input channels. The public model output dict is unchanged,
so the existing USS lightning module and loss can consume these classes as a
pure config-level opt-in replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.deft.modified_deft_bf16_safe import (
    ModifiedDeFTUSSMemoryEfficient as _BF16SafeModifiedDeFTUSSMemoryEfficient,
    ModifiedDeFTUSSMemoryEfficientTemporal as _BF16SafeModifiedDeFTUSSMemoryEfficientTemporal,
)


class FOASpatialFeatureEncoder(nn.Module):
    """Encoder stem that augments raw complex FOA STFT with AIV/IPD cues.

    Args:
        input_channels: Number of input audio channels. AIV/IPD require
            4-channel FOA order [W, X, Y, Z].
        hidden_channels: Output channels of the DeFT encoder stem.
        include_logmag: Append log1p magnitude for every input channel.
        include_aiv: Append normalized acoustic intensity vector features.
        include_ipd: Append normalized cos/sin inter-channel phase-difference
            features between W and X/Y/Z.
        eps: Numerical floor used by magnitude/normalization operations.
    """

    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 96,
        include_logmag: bool = True,
        include_aiv: bool = True,
        include_ipd: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.include_logmag = bool(include_logmag)
        self.include_aiv = bool(include_aiv)
        self.include_ipd = bool(include_ipd)
        self.eps = float(eps)

        if self.input_channels != 4 and (self.include_aiv or self.include_ipd):
            raise ValueError(
                "FOA AIV/IPD features require input_channels=4 with channel order [W, X, Y, Z]. "
                "Disable include_aiv/include_ipd for non-FOA input."
            )

        feature_channels = self.input_channels * 2
        if self.include_logmag:
            feature_channels += self.input_channels
        if self.include_aiv:
            feature_channels += 3
        if self.include_ipd:
            feature_channels += 6
        self.feature_channels = feature_channels

        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

    def _split_raw_complex(self, raw_complex: torch.Tensor):
        expected = self.input_channels * 2
        if raw_complex.dim() != 4:
            raise ValueError(
                f"Expected raw complex tensor [B, {expected}, T, F], got shape {tuple(raw_complex.shape)}"
            )
        if raw_complex.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} raw complex channels for input_channels={self.input_channels}, "
                f"got {raw_complex.shape[1]}"
            )
        real = raw_complex[:, : self.input_channels]
        imag = raw_complex[:, self.input_channels :]
        return real, imag

    def _log_magnitude(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(real.square() + imag.square() + self.eps)
        return torch.log1p(mag)

    def _foa_aiv(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Normalized FOA acoustic intensity vector features.

        Uses Re(conj(W) * X/Y/Z), normalized by total FOA TF-bin energy. This
        provides a compact direction-of-arrival-like cue per time-frequency bin.
        """

        w_r, x_r, y_r, z_r = real[:, 0], real[:, 1], real[:, 2], real[:, 3]
        w_i, x_i, y_i, z_i = imag[:, 0], imag[:, 1], imag[:, 2], imag[:, 3]

        ix = w_r * x_r + w_i * x_i
        iy = w_r * y_r + w_i * y_i
        iz = w_r * z_r + w_i * z_i
        energy = real.square().sum(dim=1) + imag.square().sum(dim=1) + self.eps
        return torch.stack([ix / energy, iy / energy, iz / energy], dim=1)

    def _foa_ipd(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Cos/sin IPD features between W and directional FOA channels."""

        w_r = real[:, 0]
        w_i = imag[:, 0]
        w_mag = torch.sqrt(w_r.square() + w_i.square() + self.eps)

        ipd_feats = []
        for ch in (1, 2, 3):
            a_r = real[:, ch]
            a_i = imag[:, ch]
            cross_r = w_r * a_r + w_i * a_i
            cross_i = w_r * a_i - w_i * a_r
            a_mag = torch.sqrt(a_r.square() + a_i.square() + self.eps)
            denom = w_mag * a_mag + self.eps
            cos_ipd = torch.clamp(cross_r / denom, min=-1.0, max=1.0)
            sin_ipd = torch.clamp(cross_i / denom, min=-1.0, max=1.0)
            ipd_feats.extend([cos_ipd, sin_ipd])
        return torch.stack(ipd_feats, dim=1)

    def build_features(self, raw_complex: torch.Tensor) -> torch.Tensor:
        real, imag = self._split_raw_complex(raw_complex)
        features = [raw_complex]
        if self.include_logmag:
            features.append(self._log_magnitude(real, imag))
        if self.include_aiv:
            features.append(self._foa_aiv(real, imag))
        if self.include_ipd:
            features.append(self._foa_ipd(real, imag))
        return torch.cat(features, dim=1)

    def forward(self, raw_complex: torch.Tensor) -> torch.Tensor:
        return self.net(self.build_features(raw_complex))


class _FOASpatialFeatureEncoderMixin:
    """Mixin that replaces the original 8-channel encoder with the opt-in stem."""

    def _replace_encoder_with_foa_spatial_features(
        self,
        input_channels: int,
        hidden_channels: int,
        include_logmag: bool,
        include_aiv: bool,
        include_ipd: bool,
        spatial_feature_eps: float,
    ) -> None:
        self.encoder = FOASpatialFeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            include_logmag=include_logmag,
            include_aiv=include_aiv,
            include_ipd=include_ipd,
            eps=spatial_feature_eps,
        )


class ModifiedDeFTUSSMemoryEfficientSpatialFeatures(
    _FOASpatialFeatureEncoderMixin,
    _BF16SafeModifiedDeFTUSSMemoryEfficient,
):
    """BF16-safe memory-efficient USS with opt-in FOA logmag/AIV/IPD features."""

    def __init__(
        self,
        *args,
        input_channels: int = 4,
        hidden_channels: int = 96,
        include_logmag: bool = True,
        include_aiv: bool = True,
        include_ipd: bool = True,
        spatial_feature_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, input_channels=input_channels, hidden_channels=hidden_channels, **kwargs)
        self._replace_encoder_with_foa_spatial_features(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            include_logmag=include_logmag,
            include_aiv=include_aiv,
            include_ipd=include_ipd,
            spatial_feature_eps=spatial_feature_eps,
        )


class ModifiedDeFTUSSMemoryEfficientTemporalSpatialFeatures(
    _FOASpatialFeatureEncoderMixin,
    _BF16SafeModifiedDeFTUSSMemoryEfficientTemporal,
):
    """Temporal memory-efficient USS with opt-in FOA logmag/AIV/IPD features."""

    def __init__(
        self,
        *args,
        input_channels: int = 4,
        hidden_channels: int = 96,
        include_logmag: bool = True,
        include_aiv: bool = True,
        include_ipd: bool = True,
        spatial_feature_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, input_channels=input_channels, hidden_channels=hidden_channels, **kwargs)
        self._replace_encoder_with_foa_spatial_features(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            include_logmag=include_logmag,
            include_aiv=include_aiv,
            include_ipd=include_ipd,
            spatial_feature_eps=spatial_feature_eps,
        )
