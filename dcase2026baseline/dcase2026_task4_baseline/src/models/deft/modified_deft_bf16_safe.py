"""BF16-safe wrappers for modified DeFT USS models.

The original modified_deft models are kept unchanged.  These wrappers are meant
for bf16-mixed training, where STFT/ISTFT and phase normalisation should remain
in fp32 to avoid NaN/Inf on near-silent bins and zero-target training examples.
"""

import torch
import torch.nn.functional as F

from src.models.deft.modified_deft import (
    ModifiedDeFTUSSMemoryEfficient as _ModifiedDeFTUSSMemoryEfficient,
    ModifiedDeFTUSSMemoryEfficientTemporal as _ModifiedDeFTUSSMemoryEfficientTemporal,
)


def _safe_magphase(real, imag, eps: float = 1e-8):
    """Return magnitude and unit phase in fp32 with an explicit epsilon.

    torchlibrosa.stft.magphase can be numerically fragile under autocast on
    near-zero complex bins.  Keeping this calculation in fp32 avoids 0/0 phase
    normalisation and prevents NaNs from entering ISTFT and the waveform loss.
    """

    real = real.float()
    imag = imag.float()
    mag = torch.sqrt(real.square() + imag.square() + eps)
    cos = real / mag
    sin = imag / mag
    return mag, cos, sin


class _BF16SafeSpectralMixin:
    def waveform_to_complex(self, waveform):
        device_type = waveform.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            real, imag = self.stft(waveform.float())
        return real, imag

    def complex_to_waveform(self, real, imag, length):
        device_type = real.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return self.istft(real.float(), imag.float(), length)

    def _spatial_mask_to_waveform(self, object_features, real, imag, samples):
        batch_size, n_objects, _, time_steps, freq_bins = object_features.shape

        mask = self.audio_head(
            object_features.reshape(batch_size * n_objects, -1, time_steps, freq_bins)
        )

        device_type = mask.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            mask = mask.float().view(
                batch_size,
                n_objects,
                self.input_channels,
                self.mask_components,
                time_steps,
                freq_bins,
            )

            mask_mag = torch.sigmoid(mask[:, :, :, 0])
            mask_real = torch.tanh(mask[:, :, :, 1])
            mask_imag = torch.tanh(mask[:, :, :, 2])
            _, mask_cos, mask_sin = _safe_magphase(mask_real, mask_imag)

            mixture_mag, mixture_cos, mixture_sin = _safe_magphase(real, imag)
            out_mag = F.relu(mixture_mag[:, None] * mask_mag)
            out_cos = mixture_cos[:, None] * mask_cos - mixture_sin[:, None] * mask_sin
            out_sin = mixture_sin[:, None] * mask_cos + mixture_cos[:, None] * mask_sin

            est_real = out_mag * out_cos
            est_imag = out_mag * out_sin

            est_real = est_real.reshape(
                batch_size * n_objects,
                self.input_channels,
                time_steps,
                freq_bins,
            )
            est_imag = est_imag.reshape(
                batch_size * n_objects,
                self.input_channels,
                time_steps,
                freq_bins,
            )

            est_real = self.out_conv(est_real)
            est_imag = self.out_conv(est_imag)

        waveform = self.complex_to_waveform(
            est_real.reshape(
                batch_size * n_objects * self.output_channels,
                1,
                time_steps,
                freq_bins,
            ),
            est_imag.reshape(
                batch_size * n_objects * self.output_channels,
                1,
                time_steps,
                freq_bins,
            ),
            samples,
        )
        return waveform.view(batch_size, n_objects, self.output_channels, samples)


class ModifiedDeFTUSSMemoryEfficient(
    _BF16SafeSpectralMixin,
    _ModifiedDeFTUSSMemoryEfficient,
):
    """BF16-safe memory-efficient USS model."""


class ModifiedDeFTUSSMemoryEfficientTemporal(
    _BF16SafeSpectralMixin,
    _ModifiedDeFTUSSMemoryEfficientTemporal,
):
    """BF16-safe memory-efficient temporal USS model."""
