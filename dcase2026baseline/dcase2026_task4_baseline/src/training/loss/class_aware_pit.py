from itertools import permutations
from typing import Callable, Literal

import torch
from torch import Tensor
from torchmetrics.functional.audio import signal_noise_ratio as snr


def _flatten_sources(waveform: Tensor) -> Tensor:
    if waveform.dim() < 3:
        raise ValueError(f"Expected waveform with shape [B, S, ...], got {tuple(waveform.shape)}")
    return waveform.flatten(start_dim=2)


def infer_active_mask_from_label(label: Tensor) -> Tensor:
    return (label != 0).any(dim=-1)


def source_activity_mask(waveform: Tensor, energy_eps: float = 1e-8) -> Tensor:
    flat = _flatten_sources(waveform)
    return flat.pow(2).sum(dim=-1) > energy_eps


def source_energy_loss(waveform: Tensor) -> Tensor:
    flat = _flatten_sources(waveform)
    return flat.pow(2).mean(dim=-1)


def inactive_source_energy_loss(waveform: Tensor, inactive_mask: Tensor) -> Tensor:
    inactive_mask = inactive_mask.bool()
    energy = source_energy_loss(waveform)
    if inactive_mask.any():
        return (energy * inactive_mask.float()).sum() / inactive_mask.float().sum().clamp_min(1.0)
    return energy.sum() * 0.0


def pairwise_sa_sdr_loss(waveform_pred: Tensor, waveform_target: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-aware negative SDR for every target/prediction pair.

    Returns [B, S_target, S_pred]. This function should only be used for active
    references; zero references are handled by explicit inactive-energy losses.
    """

    pred = _flatten_sources(waveform_pred)
    target = _flatten_sources(waveform_target)
    err = pred.unsqueeze(1) - target.unsqueeze(2)
    target_power = target.pow(2).sum(dim=-1).unsqueeze(2).clamp_min(eps)
    noise_power = err.pow(2).sum(dim=-1).clamp_min(eps)
    return -10.0 * torch.log10(target_power / noise_power)


def snr_loss_return_batch(preds: Tensor, target: Tensor) -> Tensor:
    metric = -snr(preds, target)
    if metric.dim() == 1:
        return metric
    return metric.flatten(start_dim=1).mean(dim=1)


def pit_from_pairwise_loss(
    pairwise_loss: Tensor,
    active_mask: Tensor | None = None,
    valid_pair_mask: Tensor | None = None,
    eval_func: Literal["min", "max"] = "min",
) -> tuple[Tensor, Tensor]:
    """Select a one-to-one assignment from a pairwise cost/score matrix.

    ``active_mask`` marks target rows that should contribute to the objective.
    Inactive target rows are ignored here and should be handled by an explicit
    suppression term on unmatched predictions.
    """

    if pairwise_loss.dim() != 3:
        raise ValueError(f"Expected pairwise_loss [B, S, S], got {tuple(pairwise_loss.shape)}")
    batch_size, n_targets, n_preds = pairwise_loss.shape
    if n_targets != n_preds:
        raise ValueError("Current PIT helper expects the same number of target and prediction slots")
    if eval_func not in {"min", "max"}:
        raise ValueError(f"eval_func must be 'min' or 'max', got {eval_func}")

    device = pairwise_loss.device
    if active_mask is None:
        active_mask = torch.ones(batch_size, n_targets, device=device, dtype=torch.bool)
    else:
        active_mask = active_mask.to(device=device, dtype=torch.bool)
    if valid_pair_mask is not None:
        valid_pair_mask = valid_pair_mask.to(device=device, dtype=torch.bool)

    perms = torch.tensor(list(permutations(range(n_preds))), device=device, dtype=torch.long)
    identity = torch.arange(n_preds, device=device, dtype=torch.long)
    batch_losses = []
    best_perms = []

    for batch_idx in range(batch_size):
        active_idx = torch.nonzero(active_mask[batch_idx], as_tuple=False).flatten()
        if active_idx.numel() == 0:
            batch_losses.append(pairwise_loss[batch_idx].sum() * 0.0)
            best_perms.append(identity)
            continue

        valid_losses = []
        valid_perms = []
        for perm in perms:
            pred_idx = perm[active_idx]
            if valid_pair_mask is not None and not valid_pair_mask[batch_idx, active_idx, pred_idx].all():
                continue
            valid_losses.append(pairwise_loss[batch_idx, active_idx, pred_idx].mean())
            valid_perms.append(perm)

        if not valid_losses:
            pred_idx = identity[active_idx]
            valid_losses = [pairwise_loss[batch_idx, active_idx, pred_idx].mean()]
            valid_perms = [identity]

        losses = torch.stack(valid_losses)
        best_idx = torch.argmin(losses) if eval_func == "min" else torch.argmax(losses)
        batch_losses.append(losses[best_idx])
        best_perms.append(valid_perms[int(best_idx.item())])

    return torch.stack(batch_losses, dim=0), torch.stack(best_perms, dim=0)


def matched_pairwise_mean(pairwise_loss: Tensor, best_perm: Tensor, active_mask: Tensor) -> Tensor:
    active_mask = active_mask.to(device=pairwise_loss.device, dtype=torch.bool)
    values = []
    for batch_idx in range(pairwise_loss.shape[0]):
        active_idx = torch.nonzero(active_mask[batch_idx], as_tuple=False).flatten()
        if active_idx.numel() == 0:
            values.append(pairwise_loss[batch_idx].sum() * 0.0)
            continue
        pred_idx = best_perm[batch_idx, active_idx]
        values.append(pairwise_loss[batch_idx, active_idx, pred_idx].mean())
    return torch.stack(values, dim=0).mean()


def unmatched_prediction_mask(best_perm: Tensor, active_mask: Tensor, n_preds: int | None = None) -> Tensor:
    active_mask = active_mask.to(device=best_perm.device, dtype=torch.bool)
    if n_preds is None:
        n_preds = best_perm.shape[1]
    matched = torch.zeros(best_perm.shape[0], n_preds, device=best_perm.device, dtype=torch.bool)
    for batch_idx in range(best_perm.shape[0]):
        active_idx = torch.nonzero(active_mask[batch_idx], as_tuple=False).flatten()
        if active_idx.numel() > 0:
            matched[batch_idx, best_perm[batch_idx, active_idx]] = True
    return ~matched


def class_aware_pit_loss(
    waveform_pred: Tensor,
    waveform_target: Tensor,
    label: Tensor,
    active_mask: Tensor | None = None,
    pairwise_loss_func: Callable[[Tensor, Tensor], Tensor] = pairwise_sa_sdr_loss,
    eval_func: Literal["min", "max"] = "min",
) -> tuple[Tensor, Tensor, Tensor]:
    if active_mask is None:
        active_mask = infer_active_mask_from_label(label)
    same_label = (label.unsqueeze(2) == label.unsqueeze(1)).all(dim=-1)
    active_mask = active_mask.to(device=waveform_pred.device, dtype=torch.bool)
    valid_pair_mask = same_label.to(device=waveform_pred.device) & active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
    pairwise_loss = pairwise_loss_func(waveform_pred, waveform_target)
    batch_loss, best_perm = pit_from_pairwise_loss(
        pairwise_loss=pairwise_loss,
        active_mask=active_mask,
        valid_pair_mask=valid_pair_mask,
        eval_func=eval_func,
    )
    active_count = active_mask.sum(dim=1)
    return batch_loss, best_perm, active_count


def permutation_invariant_loss(
    waveform_pred: Tensor,
    waveform_target: Tensor,
    active_mask: Tensor | None = None,
    pairwise_loss_func: Callable[[Tensor, Tensor], Tensor] = pairwise_sa_sdr_loss,
    eval_func: Literal["min", "max"] = "min",
) -> tuple[Tensor, Tensor, Tensor]:
    if active_mask is None:
        active_mask = source_activity_mask(waveform_target)
    pairwise_loss = pairwise_loss_func(waveform_pred, waveform_target)
    batch_loss, best_perm = pit_from_pairwise_loss(
        pairwise_loss=pairwise_loss,
        active_mask=active_mask,
        valid_pair_mask=None,
        eval_func=eval_func,
    )
    active_count = active_mask.sum(dim=1)
    return batch_loss, best_perm, active_count


def class_aware_permutation_invariant_training(
    waveform_pred: Tensor,
    waveform_target: Tensor,
    label: Tensor,
    metric_func: Callable[[Tensor, Tensor], Tensor],
    eval_func: Literal["max", "min"] = "max",
) -> tuple[Tensor, Tensor]:
    batch_size, n_sources = waveform_pred.shape[0], waveform_pred.shape[1]
    pairwise_metric = []
    for target_idx in range(n_sources):
        row = []
        for pred_idx in range(n_sources):
            row.append(metric_func(waveform_pred[:, pred_idx], waveform_target[:, target_idx]))
        pairwise_metric.append(torch.stack(row, dim=1))
    pairwise_metric = torch.stack(pairwise_metric, dim=1)

    active_mask = infer_active_mask_from_label(label)
    same_label = (label.unsqueeze(2) == label.unsqueeze(1)).all(dim=-1)
    valid_pair_mask = same_label & active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
    batch_metric, best_perm = pit_from_pairwise_loss(
        pairwise_loss=pairwise_metric,
        active_mask=active_mask,
        valid_pair_mask=valid_pair_mask,
        eval_func=eval_func,
    )
    if batch_metric.shape[0] != batch_size:
        raise RuntimeError("Unexpected CAPI-PIT batch shape")
    return batch_metric, best_perm


def get_loss_func(lambda_inactive: float = 0.05):
    def loss_func(output, target):
        active_mask = infer_active_mask_from_label(target["label_vector"])
        loss_active, best_perm, _ = class_aware_pit_loss(
            waveform_pred=output["waveform"],
            waveform_target=target["waveform"],
            label=target["label_vector"],
            active_mask=active_mask,
            eval_func="min",
        )
        inactive_mask = unmatched_prediction_mask(best_perm, active_mask, output["waveform"].shape[1])
        loss_inactive = inactive_source_energy_loss(output["waveform"], inactive_mask)
        loss_waveform = loss_active.mean()
        loss = loss_waveform + lambda_inactive * loss_inactive
        return {
            "loss": loss,
            "loss_waveform": loss_waveform,
            "loss_inactive": loss_inactive,
        }

    return loss_func


def get_metric_func():
    def loss_func(output, target):
        loss_val_all_sources, _ = class_aware_permutation_invariant_training(
            waveform_pred=output["waveform"],
            waveform_target=target["waveform"],
            label=target["label_vector"],
            metric_func=snr_loss_return_batch,
            eval_func="min",
        )
        return {"loss": loss_val_all_sources}

    return loss_func
