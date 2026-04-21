import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT


def _stack_complex(real, imag):
    return torch.cat([real, imag], dim=1)


class ResFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, beta=None, gamma=None):
        residual = x
        out = self.bn1(self.conv1(x))
        if beta is not None:
            out = out + beta
        if gamma is not None:
            out = out * (1.0 + gamma)
        out = F.leaky_relu(out, negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        return residual + out


class DeFTBlock(nn.Module):
    """Modified DeFT block without Mamba, following the 2025 challenge report."""

    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.freq_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.resfilm = ResFiLM(channels)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x, beta=None, gamma=None):
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x + self.local(x)

        x_f = x.permute(0, 2, 3, 1).reshape(batch_size * time_steps, freq_bins, channels)
        x_f = self.freq_attn(x_f)
        x_f = x_f.reshape(batch_size, time_steps, freq_bins, channels).permute(0, 3, 1, 2)

        x_t = x.permute(0, 3, 2, 1).reshape(batch_size * freq_bins, time_steps, channels)
        x_t = self.time_attn(x_t)
        x_t = x_t.reshape(batch_size, freq_bins, time_steps, channels).permute(0, 3, 2, 1)

        x = self.norm(x + x_f + x_t)
        return self.resfilm(x, beta=beta, gamma=gamma)


class ClassConditioner(nn.Module):
    def __init__(self, label_dim, channels):
        super().__init__()
        self.beta = nn.Linear(label_dim, channels)
        self.gamma = nn.Linear(label_dim, channels)

    def forward(self, label_onehot):
        beta = self.beta(label_onehot)[:, :, None, None]
        gamma = self.gamma(label_onehot)[:, :, None, None]
        return beta, gamma


class _BaseSpectralModel(nn.Module):
    def __init__(self, window_size=1024, hop_size=320):
        super().__init__()
        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

    def waveform_to_complex(self, waveform):
        real, imag = self.stft(waveform)
        return real, imag

    def complex_to_waveform(self, real, imag, length):
        return self.istft(real, imag, length)


class ModifiedDeFTUSS(_BaseSpectralModel):
    def __init__(
        self,
        input_channels=4,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        n_foreground=3,
        n_interference=2,
        n_classes=18,
        window_size=1024,
        hop_size=320,
    ):
        super().__init__(window_size=window_size, hop_size=hop_size)
        self.input_channels = input_channels
        self.n_foreground = n_foreground
        self.n_interference = n_interference
        self.n_noise = 1
        self.n_objects = n_foreground + n_interference + self.n_noise
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DeFTBlock(hidden_channels, n_heads=n_heads) for _ in range(n_deft_blocks)]
        )
        self.object_conv = nn.Conv2d(hidden_channels, hidden_channels * self.n_objects, kernel_size=1)
        self.audio_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, n_classes),
        )
        self.silence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(_stack_complex(real, imag))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)

        audio_mask = self.audio_head(x.reshape(batch_size * self.n_objects, -1, time_steps, freq_bins))
        audio_mask = torch.tanh(audio_mask)
        audio_mask = audio_mask.view(batch_size, self.n_objects, 2, time_steps, freq_bins)

        ref_real = real[:, :1].expand(-1, self.n_objects, -1, -1)
        ref_imag = imag[:, :1].expand(-1, self.n_objects, -1, -1)
        est_real = audio_mask[:, :, 0] * ref_real - audio_mask[:, :, 1] * ref_imag
        est_imag = audio_mask[:, :, 0] * ref_imag + audio_mask[:, :, 1] * ref_real

        waveform = self.complex_to_waveform(
            est_real.reshape(batch_size * self.n_objects, 1, time_steps, freq_bins),
            est_imag.reshape(batch_size * self.n_objects, 1, time_steps, freq_bins),
            samples,
        ).view(batch_size, self.n_objects, 1, samples)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
        }


class ModifiedDeFTTSE(_BaseSpectralModel):
    def __init__(
        self,
        mixture_channels=4,
        enrollment_channels=1,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        label_dim=18,
        window_size=1024,
        hop_size=320,
    ):
        super().__init__(window_size=window_size, hop_size=hop_size)
        self.mixture_channels = mixture_channels
        self.enrollment_channels = enrollment_channels
        self.label_dim = label_dim

        self.encoder = nn.Sequential(
            nn.Conv2d((mixture_channels + enrollment_channels) * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DeFTBlock(hidden_channels, n_heads=n_heads) for _ in range(n_deft_blocks)]
        )
        self.class_conditioner = ClassConditioner(label_dim, hidden_channels)
        self.audio_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        enroll = input_dict["enrollment"]
        label_vector = input_dict["label_vector"]
        batch_size, n_queries, _, samples = enroll.shape

        mix_real, mix_imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = mix_real.shape
        mix_real = mix_real.view(batch_size, self.mixture_channels, time_steps, freq_bins)
        mix_imag = mix_imag.view(batch_size, self.mixture_channels, time_steps, freq_bins)

        enr_real, enr_imag = self.waveform_to_complex(enroll.reshape(-1, samples))
        enr_real = enr_real.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)
        enr_imag = enr_imag.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)

        mix_real = mix_real[:, None].expand(-1, n_queries, -1, -1, -1)
        mix_imag = mix_imag[:, None].expand(-1, n_queries, -1, -1, -1)
        joint_real = torch.cat([mix_real, enr_real], dim=2)
        joint_imag = torch.cat([mix_imag, enr_imag], dim=2)
        features = _stack_complex(
            joint_real.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
            joint_imag.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
        )

        x = self.encoder(features)
        beta, gamma = self.class_conditioner(label_vector.reshape(batch_size * n_queries, -1))
        for block in self.blocks:
            x = block(x, beta=beta, gamma=gamma)

        mask = torch.tanh(self.audio_head(x))
        mix_ref_real = mix_real[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        mix_ref_imag = mix_imag[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        est_real = mask[:, 0:1] * mix_ref_real - mask[:, 1:2] * mix_ref_imag
        est_imag = mask[:, 0:1] * mix_ref_imag + mask[:, 1:2] * mix_ref_real

        waveform = self.complex_to_waveform(est_real, est_imag, samples)
        waveform = waveform.view(batch_size, n_queries, 1, samples)
        return {"waveform": waveform}
