import torch

from .base_lightningmodule import BaseLightningModule
from src.evaluation.metrics.s5_validation_breakdown import S5ValidationBreakdownMetric


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _labels_from_index_and_mask(class_index, active_mask):
    labels = []
    class_index = class_index.detach().cpu()
    active_mask = active_mask.detach().cpu()
    for sample_index, sample_active in zip(class_index, active_mask):
        sample_labels = []
        for idx, is_active in zip(sample_index.tolist(), sample_active.tolist()):
            sample_labels.append(str(int(idx)) if bool(is_active) else "silence")
        labels.append(sample_labels)
    return labels


def _labels_from_logits_and_mask(class_logits, active_mask):
    pred_index = class_logits.detach().float().argmax(dim=-1).cpu()
    active_mask = active_mask.detach().cpu()
    labels = []
    for sample_index, sample_active in zip(pred_index, active_mask):
        sample_labels = []
        for idx, is_active in zip(sample_index.tolist(), sample_active.tolist()):
            sample_labels.append(str(int(idx)) if bool(is_active) else "silence")
        labels.append(sample_labels)
    return labels


class USSLightning(BaseLightningModule):
    def _get_target_dict(self, batch_data_dict):
        target_dict = {
            "mixture": batch_data_dict["mixture"],
            "foreground_waveform": batch_data_dict["foreground_waveform"],
            "interference_waveform": batch_data_dict["interference_waveform"],
            "noise_waveform": batch_data_dict["noise_waveform"],
            "class_index": batch_data_dict["class_index"],
            "is_silence": batch_data_dict["is_silence"],
        }
        for key in (
            "foreground_span_sec",
            "interference_span_sec",
            "noise_span_sec",
            "foreground_doa",
            "foreground_doa_mask",
        ):
            if key in batch_data_dict:
                target_dict[key] = batch_data_dict[key]
        return target_dict

    def _capi_validation_breakdown(self, output_dict, target_dict, pred_active):
        """Batch-local CAPI-SDRi breakdown for USS training validation.

        We use integer class ids converted to strings because the USS dataloader
        provides class indices, not class names. CAPI-SDRi only needs consistent
        labels between predictions and references, so stringified ids are enough.

        The validation metric logs both official-compatible raw-SDR assignment
        CAPI-SDRi and paper-definition SDRi assignment diagnostics.  Existing
        ``valid/capi_sdri_*`` keys remain official-compatible.
        """

        metric = S5ValidationBreakdownMetric(metricfunc="sdr", prefix="valid", assignment_mode="compare")
        est_labels = _labels_from_logits_and_mask(output_dict["class_logits"], pred_active)
        ref_active = ~target_dict["is_silence"].bool()
        ref_labels = _labels_from_index_and_mask(target_dict["class_index"], ref_active)
        metric.update(
            batch_est_labels=est_labels,
            batch_est_waveforms=output_dict["foreground_waveform"][:, :, 0, :].detach().cpu(),
            batch_ref_labels=ref_labels,
            batch_ref_waveforms=target_dict["foreground_waveform"][:, :, 0, :].detach().cpu(),
            batch_mixture=target_dict["mixture"][:, 0, :].detach().cpu(),
        )
        summary = metric.compute(is_print=False)
        # Keep Lightning logging numeric.  None appears when a mini-batch has no
        # sample from a given scene bucket.
        return {k: v for k, v in summary.items() if v is not None}

    def _validation_breakdown(self, output_dict, target_dict):
        """Validation diagnostics for 2026 zero-target and same-class cases."""

        fg_active = ~target_dict["is_silence"].bool()
        ref_count = fg_active.long().sum(dim=1)
        zero_target = ref_count == 0
        one_target = ref_count == 1
        multi_target = ref_count > 1

        class_index = target_dict["class_index"]
        same_class = []
        for b in range(class_index.shape[0]):
            active_classes = class_index[b, fg_active[b]]
            same_class.append(active_classes.unique().numel() < active_classes.numel() if active_classes.numel() > 1 else False)
        same_class = ref_count.new_tensor(same_class, dtype=fg_active.dtype).bool()
        distinct_class = multi_target & ~same_class

        diag = {
            "valid_zero_target_ratio": zero_target.float().mean(),
            "valid_one_target_ratio": one_target.float().mean(),
            "valid_same_class_ratio": same_class.float().mean(),
            "valid_distinct_class_ratio": distinct_class.float().mean(),
        }

        active_prob = output_dict["silence_logits"].float().sigmoid()
        slot_energy = output_dict["foreground_waveform"][:, :, 0].float().pow(2).mean(dim=-1).sqrt()
        # Energy-only and active-logit diagnostics are not final inference gates;
        # they help tune the explicit Kwon2025S5 gate thresholds.
        pred_active_prob = active_prob > 0.5
        pred_active_energy = slot_energy > 1e-4
        pred_active = pred_active_prob & pred_active_energy
        pred_count = pred_active.long().sum(dim=1)

        diag.update({
            "valid_count_mae_slot_gate": (pred_count - ref_count).abs().float().mean(),
            "valid_zero_target_fp_rate_slot_gate": (pred_count[zero_target] > 0).float().mean() if zero_target.any() else ref_count.float().new_zeros(()),
            "valid_slot_active_rate": pred_active.float().mean(),
            "valid_slot_energy_mean": slot_energy.mean(),
            # Aliases matching the offline evaluator's requested log names.
            "valid/foreground_slot_active_rate": pred_active.float().mean(),
            "valid/foreground_leakage_energy": slot_energy[~pred_active].mean() if (~pred_active).any() else slot_energy.new_zeros(()),
        })

        if "count_logits" in output_dict:
            count_pred = output_dict["count_logits"].float().argmax(dim=-1)
            diag.update({
                "valid_count_acc": (count_pred == ref_count.clamp(max=output_dict["count_logits"].shape[-1] - 1)).float().mean(),
                "valid_count_mae": (count_pred - ref_count).abs().float().mean(),
                "valid_zero_target_fp_rate_count": (count_pred[zero_target] > 0).float().mean() if zero_target.any() else ref_count.float().new_zeros(()),
                "valid_same_class_count_mae": (count_pred[same_class] - ref_count[same_class]).abs().float().mean() if same_class.any() else ref_count.float().new_zeros(()),
                "valid_distinct_class_count_mae": (count_pred[distinct_class] - ref_count[distinct_class]).abs().float().mean() if distinct_class.any() else ref_count.float().new_zeros(()),
            })

        if "foreground_doa_mask" in target_dict:
            doa_mask = target_dict["foreground_doa_mask"].bool()
            diag["valid_foreground_doa_supervision_rate"] = doa_mask.float().mean()
            diag["valid_same_class_doa_supervision_rate"] = doa_mask[same_class].float().mean() if same_class.any() else ref_count.float().new_zeros(())

        diag.update(self._capi_validation_breakdown(output_dict, target_dict, pred_active))
        return diag

    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model({"mixture": batch_data_dict["mixture"]})
        target_dict = self._get_target_dict(batch_data_dict)
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model({"mixture": batch_data_dict["mixture"]})
        target_dict = self._get_target_dict(batch_data_dict)
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: _to_float(v) for k, v in loss_dict.items()}
        for k, v in self._validation_breakdown(output_dict, target_dict).items():
            loss_dict[k] = _to_float(v)

        if self.metric_func:
            metric = self.metric_func(output_dict, target_dict)
            for k, v in metric.items():
                loss_dict[k] = v.mean().item()

        return batchsize, loss_dict
