import torch

from src.training.loss.class_aware_pit import (
    class_aware_pit_loss,
    inactive_source_energy_loss,
    infer_active_mask_from_label,
    pairwise_sa_sdr_loss,
    unmatched_prediction_mask,
)
from src.temporal import align_spans_to_predictions
from src.training.loss.temporal import temporal_activity_loss


def masked_snr_loss(est, target, active_mask):
    """Backward-compatible active-only negative SDR loss.

    New Task4 2026 training should use ``get_loss_func`` below, which also
    handles duplicated labels and inactive/silence output suppression.
    """

    est = est[:, :, 0]
    target = target[:, :, 0]
    power_t = torch.sum(target**2, dim=-1) + 1e-8
    power_n = torch.sum((target - est) ** 2, dim=-1) + 1e-8
    snr = 10.0 * torch.log10(power_t / power_n)
    active_mask = active_mask.float()
    return -(snr * active_mask).sum() / active_mask.sum().clamp_min(1.0)


def get_loss_func(lambda_inactive=0.05, lambda_activity=0.0, activity_pos_weight=1.0):
    def loss_func(output, target):
        label_vector = target.get("label_vector")
        if label_vector is None:
            active_mask = target["active_mask"].bool()
            label_vector = active_mask.to(output["waveform"].dtype).unsqueeze(-1)
        else:
            active_mask = target.get("active_mask", infer_active_mask_from_label(label_vector)).bool()

        loss_active, best_perm, active_count = class_aware_pit_loss(
            waveform_pred=output["waveform"],
            waveform_target=target["waveform"],
            label=label_vector,
            active_mask=active_mask,
            pairwise_loss_func=pairwise_sa_sdr_loss,
            eval_func="min",
        )
        inactive_mask = unmatched_prediction_mask(best_perm, active_mask, output["waveform"].shape[1])
        loss_inactive = inactive_source_energy_loss(output["waveform"], inactive_mask)
        loss_waveform = loss_active.mean()
        if "activity_logits" in output and "span_sec" in target:
            span_sec = align_spans_to_predictions(
                best_perm,
                target["span_sec"].to(device=output["activity_logits"].device, dtype=output["activity_logits"].dtype),
                active_mask,
                output["activity_logits"].shape[1],
            )
            loss_activity = temporal_activity_loss(
                {"activity_logits": output["activity_logits"], "duration_sec": output.get("duration_sec")},
                {"span_sec": span_sec},
                pos_weight=activity_pos_weight,
            )
        else:
            loss_activity = output["waveform"].sum() * 0.0
        loss = loss_waveform + lambda_inactive * loss_inactive + lambda_activity * loss_activity
        return {
            "loss": loss,
            "loss_waveform": loss_waveform,
            "loss_inactive": loss_inactive,
            "loss_activity": loss_activity,
            "active_sources": active_count.float().mean(),
        }

    return loss_func
