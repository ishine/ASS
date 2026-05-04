import torch
import torch.nn.functional as F

from src.training.loss.class_aware_pit import (
    inactive_source_energy_loss,
    matched_pairwise_mean,
    pairwise_sa_sdr_loss,
    pit_from_pairwise_loss,
    source_activity_mask,
    unmatched_prediction_mask,
)
from src.temporal import align_spans_to_predictions
from src.training.loss.temporal import temporal_activity_loss


def _safe_energy(x):
    x = x.float()
    return torch.sum(x**2, dim=-1) + 1e-8


def _si_snr_loss_per_source(est, target):
    est = est.float()
    target = target.float()
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + 1e-8
    scale = torch.sum(est * target, dim=-1, keepdim=True) / target_energy
    target_proj = scale * target
    noise = est - target_proj
    ratio = (_safe_energy(target_proj) / (_safe_energy(noise) + 1e-8)).clamp_min(1e-8)
    return -10.0 * torch.log10(ratio)


def _class_pair_loss(class_logits, class_index):
    class_logits = class_logits.float()
    batch_size, n_pred, n_classes = class_logits.shape
    n_target = class_index.shape[1]
    neg_log_probs = -F.log_softmax(class_logits, dim=-1)
    expanded = neg_log_probs.unsqueeze(1).expand(batch_size, n_target, n_pred, n_classes)
    gather_idx = class_index[:, :, None, None].expand(batch_size, n_target, n_pred, 1)
    return expanded.gather(dim=-1, index=gather_idx).squeeze(-1)


def _safe_active_mean(values, active_mask):
    active_mask = active_mask.to(device=values.device, dtype=torch.bool)
    if active_mask.any():
        return (values * active_mask.float()).sum() / active_mask.float().sum().clamp_min(1.0)
    return values.new_zeros(())


def _residual_consistency_loss(output, target):
    if "mixture" not in target:
        return output["foreground_waveform"].new_zeros(())
    mixture_ref = target["mixture"][:, 0].float()
    recon = output["foreground_waveform"][:, :, 0].float().sum(dim=1)
    recon = recon + output["interference_waveform"][:, :, 0].float().sum(dim=1)
    recon = recon + output["noise_waveform"][:, 0, 0].float()
    return F.mse_loss(recon, mixture_ref)


def _source_activity_loss(output, target, output_key, span_key, active_mask=None, best_perm=None, pos_weight=1.0):
    if output_key not in output or span_key not in target:
        ref = output.get(output_key)
        if ref is None:
            ref = next(iter(output.values()))
        return ref.float().new_zeros(())
    activity_logits = output[output_key].float()
    span_sec = target[span_key].to(device=activity_logits.device, dtype=activity_logits.dtype)
    if best_perm is not None and active_mask is not None:
        span_sec = align_spans_to_predictions(best_perm, span_sec, active_mask, activity_logits.shape[1])
    return temporal_activity_loss(
        {
            "activity_logits": activity_logits,
            "duration_sec": output.get("duration_sec"),
        },
        {"span_sec": span_sec},
        pos_weight=pos_weight,
    )


def _foreground_count_target(is_silence, max_count):
    count_target = (~is_silence.bool()).long().sum(dim=1)
    return count_target.clamp(max=max_count)


def _foreground_count_loss(output, target):
    if "count_logits" not in output:
        return output["foreground_waveform"].float().new_zeros(())
    count_logits = output["count_logits"].float()
    max_count = count_logits.shape[-1] - 1
    count_target = _foreground_count_target(target["is_silence"], max_count=max_count)
    count_target = count_target.to(device=count_logits.device)
    return F.cross_entropy(count_logits, count_target)


def get_loss_func(
    lambda_non_foreground=0.01,
    # lambda_class_match=1.0,
    lambda_class_pit=0.05,
    lambda_class_ce=0.1,
    lambda_kl=1.0,
    lambda_silence=1.0,
    lambda_count=0.0,
    lambda_inactive_foreground=0.05,
    lambda_inactive_interference=0.01,
    lambda_inactive_noise=0.01,
    lambda_residual=0.0,
    lambda_activity_foreground=0.0,
    lambda_activity_interference=0.0,
    lambda_activity_noise=0.0,
    activity_pos_weight=1.0,
    active_energy_eps=1e-8,
):
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
            silence_logits = output["silence_logits"].float()
            class_index = target["class_index"]
            fg_active = ~target["is_silence"].bool()

            fg_pair_wave = pairwise_sa_sdr_loss(fg_est, fg_ref)
            fg_pair_class = _class_pair_loss(class_logits, class_index)
            fg_pair_total = fg_pair_wave + lambda_class_pit * fg_pair_class
            loss_fg_match, fg_best_perm = pit_from_pairwise_loss(fg_pair_total, active_mask=fg_active)
            loss_fg_wave = matched_pairwise_mean(fg_pair_wave, fg_best_perm, fg_active)
            loss_ce = matched_pairwise_mean(fg_pair_class, fg_best_perm, fg_active)
            fg_inactive_mask = unmatched_prediction_mask(fg_best_perm, fg_active, fg_est.shape[1])
            loss_fg_inactive = inactive_source_energy_loss(fg_est, fg_inactive_mask)

            fg_pred_active = ~fg_inactive_mask
            loss_silence = F.binary_cross_entropy_with_logits(silence_logits, fg_pred_active.float())
            if fg_inactive_mask.any():
                log_probs = F.log_softmax(class_logits[fg_inactive_mask], dim=-1)
                uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
                loss_kl = F.kl_div(log_probs, uniform, reduction="batchmean")
            else:
                loss_kl = class_logits.new_zeros(())

            loss_count = _foreground_count_loss(output, target)
            loss_fg = loss_fg_wave + lambda_class_ce * loss_ce + lambda_inactive_foreground * loss_fg_inactive
            loss_fg_activity = _source_activity_loss(
                output,
                target,
                "foreground_activity_logits",
                "foreground_span_sec",
                active_mask=fg_active,
                best_perm=fg_best_perm,
                pos_weight=activity_pos_weight,
            )

            int_active = source_activity_mask(int_ref, energy_eps=active_energy_eps)
            int_pair_wave = pairwise_sa_sdr_loss(int_est, int_ref)
            loss_int_match, int_best_perm = pit_from_pairwise_loss(int_pair_wave, active_mask=int_active)
            loss_int_wave = loss_int_match.mean()
            int_inactive_mask = unmatched_prediction_mask(int_best_perm, int_active, int_est.shape[1])
            loss_int_inactive = inactive_source_energy_loss(int_est, int_inactive_mask)
            loss_int = loss_int_wave + lambda_inactive_interference * loss_int_inactive
            loss_int_activity = _source_activity_loss(
                output,
                target,
                "interference_activity_logits",
                "interference_span_sec",
                active_mask=int_active,
                best_perm=int_best_perm,
                pos_weight=activity_pos_weight,
            )

            noise_active = source_activity_mask(noise_ref, energy_eps=active_energy_eps)
            noise_loss_per_source = _si_snr_loss_per_source(noise_est, noise_ref)
            loss_noise_wave = _safe_active_mean(noise_loss_per_source, noise_active)
            loss_noise_inactive = inactive_source_energy_loss(noise_est, ~noise_active)
            loss_noise = loss_noise_wave + lambda_inactive_noise * loss_noise_inactive
            loss_noise_activity = _source_activity_loss(
                output,
                target,
                "noise_activity_logits",
                "noise_span_sec",
                active_mask=noise_active,
                pos_weight=activity_pos_weight,
            )

            loss_residual = _residual_consistency_loss(output, target)
        loss = (
            loss_fg
            + lambda_non_foreground * (loss_int + loss_noise)
            + lambda_kl * loss_kl
            + lambda_silence * loss_silence
            + lambda_count * loss_count
            + lambda_residual * loss_residual
            + lambda_activity_foreground * loss_fg_activity
            + lambda_activity_interference * loss_int_activity
            + lambda_activity_noise * loss_noise_activity
        )
        return {
            "loss": loss,
            "loss_fg": loss_fg,
            "loss_fg_wave": loss_fg_wave,
            "loss_fg_inactive": loss_fg_inactive,
            "loss_int": loss_int,
            "loss_int_wave": loss_int_wave,
            "loss_int_inactive": loss_int_inactive,
            "loss_noise": loss_noise,
            "loss_noise_wave": loss_noise_wave,
            "loss_noise_inactive": loss_noise_inactive,
            "loss_ce": loss_ce,
            "loss_kl": loss_kl,
            "loss_silence": loss_silence,
            "loss_count": loss_count,
            "loss_residual": loss_residual,
            "loss_fg_activity": loss_fg_activity,
            "loss_int_activity": loss_int_activity,
            "loss_noise_activity": loss_noise_activity,
        }

    return loss_func
