import torch
import torch.nn.functional as F

from src.training.loss.temporal import temporal_activity_loss


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def _loss_truncation_weights(losses, quantile=0.8, drop_weight=0.1, min_keep_ratio=0.5):
    if losses.numel() <= 1:
        return torch.ones_like(losses)

    quantile = float(max(0.0, min(1.0, quantile)))
    drop_weight = float(max(0.0, min(1.0, drop_weight)))
    min_keep_ratio = float(max(0.0, min(1.0, min_keep_ratio)))

    detached = losses.detach()
    threshold = torch.quantile(detached.float(), quantile).to(device=losses.device, dtype=losses.dtype)
    weights = torch.where(
        detached <= threshold,
        torch.ones_like(losses),
        torch.full_like(losses, drop_weight),
    )

    min_keep = max(1, int(torch.ceil(losses.new_tensor(losses.numel() * min_keep_ratio)).item()))
    if weights.gt(drop_weight).sum().item() < min_keep:
        order = torch.argsort(detached)
        weights = torch.full_like(losses, drop_weight)
        weights[order[:min_keep]] = 1.0
    return weights


def get_loss_func(
    lambda_energy=0.0,
    m_in=-6.0,
    m_out=-1.0,
    lambda_activity=0.0,
    activity_pos_weight=1.0,
    lambda_duplicate_recall=0.0,
    duplicate_m_in=-8.0,
    label_smoothing=0.0,
    robust_loss_mode="none",
    truncation_quantile=0.8,
    truncation_warmup_epochs=0,
    truncation_drop_weight=0.1,
    min_keep_ratio=0.5,
    robust_apply_to_validation=False,
):
    if robust_loss_mode not in (None, "none", "truncation"):
        raise ValueError(f"Unsupported robust_loss_mode: {robust_loss_mode}")

    def loss_func(output, target):
        class_index = target["class_index"]
        is_silence = target["is_silence"].bool()
        robust_enabled = robust_loss_mode not in (None, "none")
        current_epoch = int(target.get("current_epoch", 0))
        is_training = bool(target.get("is_training", True))
        robust_active = (
            robust_enabled
            and robust_loss_mode == "truncation"
            and current_epoch >= int(truncation_warmup_epochs)
            and (is_training or robust_apply_to_validation)
        )

        loss = output["plain_logits"].new_tensor(0.0)
        metrics = {}
        sample_weights = torch.ones_like(is_silence, dtype=output["plain_logits"].dtype)

        if (~is_silence).any():
            active_losses = F.cross_entropy(
                output["logits"][~is_silence],
                class_index[~is_silence],
                reduction="none",
                label_smoothing=label_smoothing,
            )
            if robust_active:
                active_weights = _loss_truncation_weights(
                    active_losses,
                    quantile=truncation_quantile,
                    drop_weight=truncation_drop_weight,
                    min_keep_ratio=min_keep_ratio,
                )
                sample_weights[~is_silence] = active_weights
                loss_arc = _weighted_mean(active_losses, active_weights)
                metrics["loss_arcface_raw"] = active_losses.mean()
                metrics["loss_truncation_weight_mean"] = active_weights.mean()
                metrics["loss_truncation_kept_ratio"] = active_weights.gt(truncation_drop_weight).to(active_weights.dtype).mean()
            else:
                loss_arc = active_losses.mean()
                metrics["loss_arcface_raw"] = loss_arc
                metrics["loss_truncation_weight_mean"] = loss.new_tensor(1.0)
                metrics["loss_truncation_kept_ratio"] = loss.new_tensor(1.0)
            loss = loss + loss_arc
            metrics["loss_arcface"] = loss_arc
        else:
            metrics["loss_arcface"] = loss
            metrics["loss_arcface_raw"] = loss
            metrics["loss_truncation_weight_mean"] = loss.new_tensor(1.0)
            metrics["loss_truncation_kept_ratio"] = loss.new_tensor(1.0)

        if is_silence.any():
            log_probs = F.log_softmax(output["plain_logits"][is_silence], dim=-1)
            uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
            loss_kl = F.kl_div(log_probs, uniform, reduction="batchmean")
            loss = loss + loss_kl
            metrics["loss_kl"] = loss_kl
        else:
            metrics["loss_kl"] = loss.new_tensor(0.0)

        if lambda_energy > 0.0:
            energy = output["energy"]
            loss_in = energy[~is_silence] - m_in if (~is_silence).any() else energy.new_zeros(1)
            loss_out = m_out - energy[is_silence] if is_silence.any() else energy.new_zeros(1)
            hinge_in = torch.clamp(loss_in, min=0.0).pow(2).mean() if (~is_silence).any() else energy.new_tensor(0.0)
            hinge_out = torch.clamp(loss_out, min=0.0).pow(2).mean() if is_silence.any() else energy.new_tensor(0.0)
            loss_energy = hinge_in + hinge_out
            loss = loss + lambda_energy * loss_energy
            metrics["loss_energy"] = loss_energy
        else:
            metrics["loss_energy"] = loss.new_tensor(0.0)

        duplicate_mask = target.get("is_duplicate_class", None)
        if duplicate_mask is not None:
            duplicate_mask = duplicate_mask.to(device=output["energy"].device, dtype=torch.bool) & (~is_silence)
        if lambda_duplicate_recall > 0.0 and duplicate_mask is not None and duplicate_mask.any():
            duplicate_energy = output["energy"][duplicate_mask]
            duplicate_penalty = torch.clamp(duplicate_energy - duplicate_m_in, min=0.0).pow(2)
            duplicate_weights = sample_weights.to(device=duplicate_penalty.device, dtype=duplicate_penalty.dtype)[duplicate_mask]
            loss_duplicate_recall = _weighted_mean(duplicate_penalty, duplicate_weights)
            loss = loss + lambda_duplicate_recall * loss_duplicate_recall
            metrics["loss_duplicate_recall"] = loss_duplicate_recall
        else:
            metrics["loss_duplicate_recall"] = loss.new_tensor(0.0)

        if lambda_activity > 0.0:
            loss_activity = temporal_activity_loss(output, target, pos_weight=activity_pos_weight)
            if loss_activity is None:
                loss_activity = loss.new_tensor(0.0)
            loss = loss + lambda_activity * loss_activity
            metrics["loss_activity"] = loss_activity
        else:
            metrics["loss_activity"] = loss.new_tensor(0.0)

        metrics["loss"] = loss
        return metrics

    return loss_func
