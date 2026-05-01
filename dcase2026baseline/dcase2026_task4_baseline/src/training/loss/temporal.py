import torch
import torch.nn.functional as F

from src.temporal import spans_to_frame_targets


def temporal_activity_loss(output, target, pos_weight=1.0):
    if "activity_logits" not in output or "span_sec" not in target:
        return None

    logits = output["activity_logits"]
    span_sec = target["span_sec"].to(device=logits.device, dtype=logits.dtype)
    duration_sec = output.get("duration_sec")
    if duration_sec is not None:
        duration_sec = duration_sec.to(device=logits.device, dtype=logits.dtype)
    frame_targets = spans_to_frame_targets(
        span_sec,
        num_frames=logits.shape[-1],
        duration_sec=duration_sec,
    )
    valid = (span_sec[..., 0] >= 0.0) & (span_sec[..., 1] > span_sec[..., 0])
    if not valid.any():
        return logits.sum() * 0.0

    weight = torch.ones_like(frame_targets)
    if pos_weight != 1.0:
        weight = torch.where(frame_targets > 0.5, weight * float(pos_weight), weight)
    loss = F.binary_cross_entropy_with_logits(logits, frame_targets, weight=weight, reduction="none")
    return loss[valid].mean()
