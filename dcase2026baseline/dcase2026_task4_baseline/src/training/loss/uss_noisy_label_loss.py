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
from src.training.loss.uss_loss import (
    _safe_active_mean,
    _si_snr_loss_per_source,
    _residual_consistency_loss,
)


def _weighted_mean(values, weight, active_mask):
    weight = weight.to(device=values.device, dtype=values.dtype)
    active = active_mask.to(device=values.device, dtype=values.dtype)
    w = weight * active
    return (values * w).sum() / w.sum().clamp_min(1.0)


def _hard_class_pair_loss(class_logits, class_index):
    class_logits = class_logits.float()
    batch_size, n_pred, n_classes = class_logits.shape
    n_target = class_index.shape[1]
    neg_log_probs = -F.log_softmax(class_logits, dim=-1)
    expanded = neg_log_probs.unsqueeze(1).expand(batch_size, n_target, n_pred, n_classes)
    gather_idx = class_index[:, :, None, None].expand(batch_size, n_target, n_pred, 1)
    return expanded.gather(dim=-1, index=gather_idx).squeeze(-1)


def _soft_class_pair_loss(class_logits, soft_target):
    class_logits = class_logits.float()
    log_probs = F.log_softmax(class_logits, dim=-1)
    batch_size, n_pred, n_classes = log_probs.shape
    n_target = soft_target.shape[1]
    target = soft_target.to(device=log_probs.device, dtype=log_probs.dtype)
    log_probs = log_probs.unsqueeze(1).expand(batch_size, n_target, n_pred, n_classes)
    target = target.unsqueeze(2).expand(batch_size, n_target, n_pred, n_classes)
    return -(target * log_probs).sum(dim=-1)


def _class_pair_loss(output, target, use_soft_targets=True):
    class_logits = output["class_logits"].float()
    if use_soft_targets and "soft_class_target" in target:
        soft = target["soft_class_target"].to(device=class_logits.device, dtype=class_logits.dtype)
        if soft.sum(dim=-1).amax() > 0:
            return _soft_class_pair_loss(class_logits, soft)
    return _hard_class_pair_loss(class_logits, target["class_index"])


def _gather_pairwise(pairwise, perm, active_mask):
    batch_size, n_target, _ = pairwise.shape
    idx = perm[:, :n_target].unsqueeze(-1)
    gathered = torch.gather(pairwise, dim=2, index=idx).squeeze(-1)
    return gathered * active_mask.to(device=pairwise.device, dtype=pairwise.dtype)


def _semantic_truncation_weight(losses, active_mask, quantile=0.8, drop_weight=0.1):
    active = active_mask.to(device=losses.device, dtype=torch.bool)
    weight = torch.ones_like(losses)
    if not active.any():
        return weight
    vals = losses[active].detach()
    if vals.numel() < 2:
        return weight
    q = torch.quantile(vals.float(), float(quantile)).to(losses.device)
    high = (losses.detach() > q) & active
    weight[high] = float(drop_weight)
    return weight


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
        {"activity_logits": activity_logits, "duration_sec": output.get("duration_sec")},
        {"span_sec": span_sec},
        pos_weight=pos_weight,
    )


def _weighted_bce_with_logits(logits, target, weight):
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = weight.to(device=raw.device, dtype=raw.dtype)
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def get_loss_func(
    lambda_non_foreground=0.01,
    lambda_class_match=0.25,
    lambda_kl=1.0,
    lambda_silence=1.0,
    lambda_inactive_foreground=0.05,
    lambda_inactive_interference=0.01,
    lambda_inactive_noise=0.01,
    lambda_residual=0.0,
    lambda_activity_foreground=0.0,
    lambda_activity_interference=0.0,
    lambda_activity_noise=0.0,
    activity_pos_weight=1.0,
    active_energy_eps=1e-8,
    use_soft_targets=True,
    use_class_confidence=True,
    uncertain_slot_silence_weight=0.2,
    bad_slot_silence_weight=0.0,
    robust_semantic_mode="soft_truncated",
    semantic_warmup_epochs=3,
    semantic_truncation_quantile=0.8,
    semantic_truncation_drop_weight=0.1,
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
            fg_active = ~target["is_silence"].bool()

            class_conf = target.get("class_confidence", None)
            if class_conf is None or not use_class_confidence:
                class_conf = torch.ones_like(fg_active, dtype=torch.float32)
            class_conf = class_conf.to(device=fg_est.device, dtype=fg_est.dtype)
            class_conf = torch.where(fg_active, class_conf, torch.zeros_like(class_conf))

            uncertain = target.get("uncertain_slot_mask", torch.zeros_like(fg_active)).to(device=fg_est.device).bool()
            bad = target.get("bad_slot_mask", torch.zeros_like(fg_active)).to(device=fg_est.device).bool()

            fg_pair_wave = pairwise_sa_sdr_loss(fg_est, fg_ref)
            fg_pair_class = _class_pair_loss(output, target, use_soft_targets=use_soft_targets)
            fg_pair_total = fg_pair_wave + lambda_class_match * fg_pair_class * class_conf[:, :, None]
            loss_fg_match, fg_best_perm = pit_from_pairwise_loss(fg_pair_total, active_mask=fg_active)
            loss_fg_wave = matched_pairwise_mean(fg_pair_wave, fg_best_perm, fg_active)

            matched_class = _gather_pairwise(fg_pair_class, fg_best_perm, fg_active)
            semantic_weight = class_conf.clone()
            current_epoch = int(target.get("current_epoch", 0))
            if robust_semantic_mode in {"truncated", "soft_truncated"} and current_epoch >= int(semantic_warmup_epochs):
                semantic_weight = semantic_weight * _semantic_truncation_weight(
                    matched_class,
                    fg_active,
                    quantile=semantic_truncation_quantile,
                    drop_weight=semantic_truncation_drop_weight,
                )
            loss_ce = _weighted_mean(matched_class, semantic_weight, fg_active)

            fg_inactive_mask = unmatched_prediction_mask(fg_best_perm, fg_active, fg_est.shape[1])
            loss_fg_inactive = inactive_source_energy_loss(fg_est, fg_inactive_mask)

            fg_pred_active = ~fg_inactive_mask
            silence_weight = torch.ones_like(silence_logits, dtype=fg_est.dtype)
            silence_weight = torch.where(uncertain, silence_weight * float(uncertain_slot_silence_weight), silence_weight)
            silence_weight = torch.where(bad, silence_weight * float(bad_slot_silence_weight), silence_weight)
            loss_silence = _weighted_bce_with_logits(silence_logits, fg_pred_active.float(), silence_weight)

            if fg_inactive_mask.any():
                log_probs = F.log_softmax(class_logits[fg_inactive_mask], dim=-1)
                uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
                loss_kl = F.kl_div(log_probs, uniform, reduction="batchmean")
            else:
                loss_kl = class_logits.new_zeros(())

            loss_fg = loss_fg_wave + lambda_class_match * loss_ce + lambda_inactive_foreground * loss_fg_inactive
            loss_fg_activity = _source_activity_loss(
                output, target, "foreground_activity_logits", "foreground_span_sec",
                active_mask=fg_active, best_perm=fg_best_perm, pos_weight=activity_pos_weight,
            )

            int_active = source_activity_mask(int_ref, energy_eps=active_energy_eps)
            int_pair_wave = pairwise_sa_sdr_loss(int_est, int_ref)
            loss_int_match, int_best_perm = pit_from_pairwise_loss(int_pair_wave, active_mask=int_active)
            loss_int_wave = loss_int_match.mean()
            int_inactive_mask = unmatched_prediction_mask(int_best_perm, int_active, int_est.shape[1])
            loss_int_inactive = inactive_source_energy_loss(int_est, int_inactive_mask)
            loss_int = loss_int_wave + lambda_inactive_interference * loss_int_inactive
            loss_int_activity = _source_activity_loss(
                output, target, "interference_activity_logits", "interference_span_sec",
                active_mask=int_active, best_perm=int_best_perm, pos_weight=activity_pos_weight,
            )

            noise_active = source_activity_mask(noise_ref, energy_eps=active_energy_eps)
            noise_loss_per_source = _si_snr_loss_per_source(noise_est, noise_ref)
            loss_noise_wave = _safe_active_mean(noise_loss_per_source, noise_active)
            loss_noise_inactive = inactive_source_energy_loss(noise_est, ~noise_active)
            loss_noise = loss_noise_wave + lambda_inactive_noise * loss_noise_inactive
            loss_noise_activity = _source_activity_loss(
                output, target, "noise_activity_logits", "noise_span_sec",
                active_mask=noise_active, pos_weight=activity_pos_weight,
            )

            loss_residual = _residual_consistency_loss(output, target)

        loss = (
            loss_fg
            + lambda_non_foreground * (loss_int + loss_noise)
            + lambda_kl * loss_kl
            + lambda_silence * loss_silence
            + lambda_residual * loss_residual
            + lambda_activity_foreground * loss_fg_activity
            + lambda_activity_interference * loss_int_activity
            + lambda_activity_noise * loss_noise_activity
        )
        quality_resamples = target.get("quality_resample_attempts", None)
        if quality_resamples is not None:
            quality_resamples = quality_resamples.float().mean().to(loss.device)
        else:
            quality_resamples = loss.new_zeros(())
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
            "loss_residual": loss_residual,
            "loss_fg_activity": loss_fg_activity,
            "loss_int_activity": loss_int_activity,
            "loss_noise_activity": loss_noise_activity,
            "semantic_weight_mean": class_conf[fg_active].mean() if fg_active.any() else loss.new_zeros(()),
            "quality_resample_attempts": quality_resamples,
        }

    return loss_func
