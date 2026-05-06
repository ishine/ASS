"""Opt-in residual-aware USS loss for DCASE 2026 Task 4.

Adds optional residual non-foreground supervision, foreground waveform L1 /
MR-STFT losses, mixture consistency, and cross-source anti-crosstalk loss.
Existing configs are unaffected unless they select this loss module.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.training.loss.class_aware_pit import (
    inactive_source_energy_loss,
    matched_pairwise_mean,
    pairwise_sa_sdr_loss,
    pit_from_pairwise_loss,
    source_activity_mask,
    unmatched_prediction_mask,
)
from src.training.loss.uss_loss import _class_pair_loss, _si_snr_loss_per_source


def _src(x: Tensor, ref_channel: int = 0) -> Tensor:
    if x.dim() == 4:
        return x[:, :, ref_channel, :]
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected source bank [B,S,C,T] or [B,S,T], got {tuple(x.shape)}")


def _mix(x: Tensor, ref_channel: int = 0) -> Tensor:
    if x.dim() == 3:
        return x[:, ref_channel, :]
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected mixture [B,C,T] or [B,T], got {tuple(x.shape)}")


def _sum_src(x: Tensor | None, ref_channel: int = 0) -> Tensor | None:
    if x is None:
        return None
    y = _src(x, ref_channel)
    if y.shape[1] == 0:
        return y.new_zeros(y.shape[0], y.shape[-1])
    return y.sum(dim=1)


def _gather(pred: Tensor, best_perm: Tensor) -> Tensor:
    return torch.stack([pred[b, best_perm[b]] for b in range(pred.shape[0])], dim=0)


def _stft_l1(pred: Tensor, target: Tensor, n_fft: int, hop: int) -> Tensor:
    window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
    p = torch.stft(pred.float(), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, center=True, return_complex=True)
    t = torch.stft(target.float(), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, center=True, return_complex=True)
    return F.l1_loss(p.abs(), t.abs())


def _mrstft(pred: Tensor, target: Tensor, fft_sizes: Sequence[int]) -> Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    vals = []
    for n_fft in fft_sizes:
        vals.append(_stft_l1(pred, target, int(n_fft), max(1, int(n_fft) // 4)))
    return torch.stack(vals).mean()


def _matched_l1(pred: Tensor, target: Tensor, best_perm: Tensor, active: Tensor, ref_channel: int) -> Tensor:
    p = _gather(_src(pred, ref_channel), best_perm)
    t = _src(target, ref_channel)
    a = active.to(device=p.device, dtype=torch.bool)
    if not a.any():
        return p.new_zeros(())
    per = (p - t).abs().mean(dim=-1)
    return (per * a.float()).sum() / a.float().sum().clamp_min(1.0)


def _matched_mrstft(pred: Tensor, target: Tensor, best_perm: Tensor, active: Tensor, ref_channel: int, fft_sizes: Sequence[int]) -> Tensor:
    p = _gather(_src(pred, ref_channel), best_perm)
    t = _src(target, ref_channel)
    a = active.to(device=p.device, dtype=torch.bool)
    if not a.any():
        return p.new_zeros(())
    return _mrstft(p[a], t[a], fft_sizes)


def _cross_talk_loss(fg_est: Tensor, fg_ref: Tensor, best_perm: Tensor, active: Tensor, ref_channel: int, eps: float = 1e-8) -> Tensor:
    est = _gather(_src(fg_est, ref_channel).float(), best_perm)
    ref = _src(fg_ref, ref_channel).float()
    a = active.to(device=est.device, dtype=torch.bool)
    if a.sum() < 2:
        return est.new_zeros(())
    est = est / est.norm(dim=-1, keepdim=True).clamp_min(eps)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp_min(eps)
    corr = torch.abs(torch.matmul(est, ref.transpose(1, 2)))
    s = corr.shape[1]
    mask = a[:, :, None] & a[:, None, :] & (~torch.eye(s, device=est.device, dtype=torch.bool)[None])
    if not mask.any():
        return est.new_zeros(())
    return (corr * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _residual_ref(target: dict, ref_channel: int) -> Tensor:
    mixture = _mix(target["mixture"], ref_channel)
    known = torch.zeros_like(mixture)
    for key in ("foreground_waveform", "interference_waveform", "noise_waveform"):
        if key in target:
            known = known + _sum_src(target[key], ref_channel).to(device=known.device, dtype=known.dtype)
    return (mixture - known).detach()


def _residual_loss(output: dict, target: dict, ref_channel: int, fft_sizes: Sequence[int]) -> tuple[Tensor, Tensor, Tensor]:
    if "residual_waveform" not in output or output["residual_waveform"].shape[1] == 0:
        z = output["foreground_waveform"].new_zeros(())
        return z, z, z
    est = _src(output["residual_waveform"], ref_channel).sum(dim=1)
    ref = _residual_ref(target, ref_channel).to(est.device)
    mae = F.l1_loss(est.float(), ref.float())
    stft = _mrstft(est, ref, fft_sizes)
    return mae + stft, mae, stft


def _mix_loss(output: dict, target: dict, ref_channel: int) -> Tensor:
    if "mixture" not in target:
        return output["foreground_waveform"].new_zeros(())
    ref = _mix(target["mixture"], ref_channel).to(output["foreground_waveform"].device)
    recon = torch.zeros_like(ref)
    for key in ("foreground_waveform", "interference_waveform", "noise_waveform", "residual_waveform"):
        if key in output:
            recon = recon + _sum_src(output[key], ref_channel).to(device=recon.device, dtype=recon.dtype)
    return F.l1_loss(recon.float(), ref.float())


def get_loss_func(
    lambda_non_fg: float = 0.01,
    lambda_class: float = 0.8,
    lambda_mae: float = 0.0,
    lambda_stft: float = 0.0,
    lambda_cross_talk: float = 0.0,
    lambda_inactive: float = 0.05,
    lambda_interference_inactive: float = 0.01,
    lambda_noise_inactive: float = 0.01,
    lambda_residual_slot: float = 0.0,
    lambda_mix: float = 0.0,
    lambda_silence: float = 1.0,
    lambda_kl: float = 1.0,
    stft_fft_sizes: Iterable[int] = (512, 1024, 2048),
    ref_channel: int = 0,
    active_energy_eps: float = 1e-8,
):
    fft_sizes = tuple(int(x) for x in stft_fft_sizes)

    def loss_func(output, target):
        device_type = output["foreground_waveform"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            fg_est = output["foreground_waveform"].float()
            int_est = output["interference_waveform"].float()
            noise_est = output["noise_waveform"][:, :, 0].float()
            fg_ref = target["foreground_waveform"].float()
            int_ref = target["interference_waveform"].float()
            noise_ref = target["noise_waveform"][:, :, 0].float()
            class_logits = output["class_logits"].float()
            class_index = target["class_index"]
            active = ~target["is_silence"].bool()

            pair_wave = pairwise_sa_sdr_loss(fg_est, fg_ref)
            pair_class = _class_pair_loss(class_logits, class_index)
            _, best_perm = pit_from_pairwise_loss(pair_wave + float(lambda_class) * pair_class, active_mask=active)
            loss_fg_wave = matched_pairwise_mean(pair_wave, best_perm, active)
            loss_ce = matched_pairwise_mean(pair_class, best_perm, active)
            loss_mae = _matched_l1(fg_est, fg_ref, best_perm, active, ref_channel)
            loss_stft = _matched_mrstft(fg_est, fg_ref, best_perm, active, ref_channel, fft_sizes)
            loss_cross_talk = _cross_talk_loss(fg_est, fg_ref, best_perm, active, ref_channel)
            inactive_mask = unmatched_prediction_mask(best_perm, active, fg_est.shape[1])
            loss_inactive = inactive_source_energy_loss(fg_est, inactive_mask)
            slot_active = ~inactive_mask
            loss_silence = F.binary_cross_entropy_with_logits(output["silence_logits"].float(), slot_active.float())
            if inactive_mask.any():
                lp = F.log_softmax(class_logits[inactive_mask], dim=-1)
                uni = torch.full_like(lp, 1.0 / lp.shape[-1])
                loss_kl = F.kl_div(lp, uni, reduction="batchmean")
            else:
                loss_kl = class_logits.new_zeros(())

            loss_fg = (loss_fg_wave + float(lambda_class) * loss_ce + float(lambda_mae) * loss_mae
                       + float(lambda_stft) * loss_stft + float(lambda_cross_talk) * loss_cross_talk
                       + float(lambda_inactive) * loss_inactive)

            int_active = source_activity_mask(int_ref, energy_eps=active_energy_eps)
            int_pair = pairwise_sa_sdr_loss(int_est, int_ref)
            loss_int_match, int_perm = pit_from_pairwise_loss(int_pair, active_mask=int_active)
            loss_int_wave = loss_int_match.mean()
            int_inactive = unmatched_prediction_mask(int_perm, int_active, int_est.shape[1])
            loss_int_inactive = inactive_source_energy_loss(int_est, int_inactive)
            loss_interference = loss_int_wave + float(lambda_interference_inactive) * loss_int_inactive

            noise_active = source_activity_mask(noise_ref, energy_eps=active_energy_eps)
            noise_per = _si_snr_loss_per_source(noise_est, noise_ref)
            loss_noise_wave = (noise_per * noise_active.float()).sum() / noise_active.float().sum().clamp_min(1.0) if noise_active.any() else noise_per.new_zeros(())
            loss_noise_inactive = inactive_source_energy_loss(noise_est, ~noise_active)
            loss_noise = loss_noise_wave + float(lambda_noise_inactive) * loss_noise_inactive

            loss_residual, loss_residual_mae, loss_residual_stft = _residual_loss(output, target, ref_channel, fft_sizes)
            loss_mix = _mix_loss(output, target, ref_channel)

        loss = (loss_fg + float(lambda_non_fg) * (loss_interference + loss_noise)
                + float(lambda_residual_slot) * loss_residual + float(lambda_mix) * loss_mix
                + float(lambda_silence) * loss_silence + float(lambda_kl) * loss_kl)
        return {
            "loss": loss,
            "loss_fg": loss_fg,
            "loss_fg_wave": loss_fg_wave,
            "loss_ce": loss_ce,
            "loss_fg_mae": loss_mae,
            "loss_fg_stft": loss_stft,
            "loss_cross_talk": loss_cross_talk,
            "loss_fg_inactive": loss_inactive,
            "loss_interference": loss_interference,
            "loss_int_wave": loss_int_wave,
            "loss_int_inactive": loss_int_inactive,
            "loss_noise": loss_noise,
            "loss_noise_wave": loss_noise_wave,
            "loss_noise_inactive": loss_noise_inactive,
            "loss_residual": loss_residual,
            "loss_residual_mae": loss_residual_mae,
            "loss_residual_stft": loss_residual_stft,
            "loss_mix": loss_mix,
            "loss_silence": loss_silence,
            "loss_kl": loss_kl,
        }

    return loss_func
