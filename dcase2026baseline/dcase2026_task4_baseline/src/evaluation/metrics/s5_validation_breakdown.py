"""DCASE 2026 Task 4 validation breakdown metrics.

This module complements the official CAPI-SDRi scorer with category-level
aggregates that expose the new 2026 failure modes:

    - zero-target mixtures
    - one-target mixtures
    - 2--3 target mixtures with all classes distinct
    - 2--3 target mixtures with at least one same-class duplicate

It intentionally reuses ``S5ClassAwareMetric`` for per-sample CAPI-SDRi so the
reported category scores remain compatible with the official-baseline scorer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional

import torch

from src.evaluation.metrics.s5capi_metric import S5ClassAwareMetric


def _as_label_list(labels) -> List[str]:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    return [str(label) for label in labels]


def _active_labels(labels: Iterable[str]) -> List[str]:
    return [str(label) for label in labels if str(label) != "silence"]


def _scene_bucket(ref_labels) -> str:
    active = _active_labels(ref_labels)
    n_active = len(active)
    if n_active == 0:
        return "zero_target"
    if n_active == 1:
        return "one_target"
    if len(set(active)) < n_active:
        return "same_class_duplicate"
    return "distinct_class"


def _mean(values) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


class S5ValidationBreakdownMetric:
    """Aggregate CAPI-SDRi and detection diagnostics by scene type."""

    metric_name = "S5 validation breakdown"

    def __init__(self, metricfunc: str = "sdr", prefix: str = "valid"):
        self.prefix = str(prefix).rstrip("/")
        self.sample_metric = S5ClassAwareMetric(metricfunc=metricfunc)
        self.reset()

    def reset(self):
        self.values = defaultdict(list)
        self.sample_values = []
        self.silence_tp = 0
        self.silence_fp = 0
        self.silence_fn = 0

    def update(self, batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture):
        batch_values = []
        for est_labels, est_wf, ref_labels, ref_wf, mixture in zip(
            batch_est_labels,
            batch_est_waveforms,
            batch_ref_labels,
            batch_ref_waveforms,
            batch_mixture,
        ):
            sample_value = self.compute_sample(est_labels, est_wf, ref_labels, ref_wf, mixture)
            self.sample_values.append(sample_value)
            batch_values.append(sample_value)
        return batch_values

    def _add_capi_value(self, bucket: str, capi_value):
        if capi_value is None:
            return
        self.values["capi_sdri_all"].append(float(capi_value))
        if bucket == "one_target":
            self.values["capi_sdri_1target"].append(float(capi_value))
        elif bucket == "distinct_class":
            self.values["capi_sdri_distinct_class"].append(float(capi_value))
        elif bucket == "same_class_duplicate":
            self.values["capi_sdri_same_class_duplicate"].append(float(capi_value))
        elif bucket == "zero_target":
            # True zero-target silence predictions are undefined/excluded by the
            # official scorer.  Zero-target false positives receive a 0 dB value.
            self.values["capi_sdri_zero_target_fp"].append(float(capi_value))

    def _add_silence_counts(self, est_labels: List[str], ref_labels: List[str]):
        """Count silence precision/recall by slot counts, not class labels.

        Estimated slots are not guaranteed to be ordered like reference slots, so
        we compare the number of silent slots.  This makes the diagnostic
        permutation-insensitive and directly reflects unused-slot/zero-target
        behavior.
        """

        n_slots = max(len(est_labels), len(ref_labels), 1)
        est_silence = sum(label == "silence" for label in est_labels)
        ref_silence = sum(label == "silence" for label in ref_labels)
        # Missing padded positions are treated as silence so variable-length
        # outputs can still be diagnosed consistently.
        est_silence += max(n_slots - len(est_labels), 0)
        ref_silence += max(n_slots - len(ref_labels), 0)
        matched = min(est_silence, ref_silence)
        self.silence_tp += matched
        self.silence_fp += max(est_silence - ref_silence, 0)
        self.silence_fn += max(ref_silence - est_silence, 0)

    def compute_sample(self, est_lb, est_wf, ref_lb, ref_wf, mixture):
        est_labels = _as_label_list(est_lb)
        ref_labels = _as_label_list(ref_lb)
        est_active = _active_labels(est_labels)
        ref_active = _active_labels(ref_labels)
        bucket = _scene_bucket(ref_labels)

        capi_value = self.sample_metric.compute_sample(est_labels, est_wf, ref_labels, ref_wf, mixture)
        self._add_capi_value(bucket, capi_value)
        self._add_silence_counts(est_labels, ref_labels)

        est_count = len(est_active)
        slot_active_rate = est_count / max(len(est_labels), 1)
        self.values["foreground_slot_active_rate"].append(float(slot_active_rate))
        self.values[f"foreground_slot_active_rate_{bucket}"].append(float(slot_active_rate))

        if bucket == "zero_target":
            self.values["capi_sdri_zero_target_fp_rate"].append(float(est_count > 0))

        if est_wf is not None and len(est_wf) > 0:
            est_wf = est_wf.float()
            active_mask = torch.tensor([label != "silence" for label in est_labels], dtype=torch.bool, device=est_wf.device)
            inactive_mask = ~active_mask
            if inactive_mask.any():
                leakage = est_wf[inactive_mask].pow(2).mean().sqrt().item()
            else:
                leakage = 0.0
            self.values["foreground_leakage_energy"].append(float(leakage))
            self.values[f"foreground_leakage_energy_{bucket}"].append(float(leakage))

        return capi_value

    def compute(self, is_print=False):
        result = {}
        for key, values in sorted(self.values.items()):
            value = _mean(values)
            if value is not None:
                result[f"{self.prefix}/{key}"] = value

        precision = _safe_div(self.silence_tp, self.silence_tp + self.silence_fp)
        recall = _safe_div(self.silence_tp, self.silence_tp + self.silence_fn)
        if precision is not None:
            result[f"{self.prefix}/silence_precision"] = precision
        if recall is not None:
            result[f"{self.prefix}/silence_recall"] = recall

        # Explicit aliases for the four scene buckets requested in experiment logs.
        for key in (
            "capi_sdri_all",
            "capi_sdri_1target",
            "capi_sdri_distinct_class",
            "capi_sdri_same_class_duplicate",
            "capi_sdri_zero_target_fp_rate",
            "foreground_slot_active_rate",
            "foreground_leakage_energy",
            "silence_precision",
            "silence_recall",
        ):
            result.setdefault(f"{self.prefix}/{key}", None)

        if is_print:
            for key, value in result.items():
                if value is None:
                    print(f"{key}: None")
                else:
                    print(f"{key}: {value:.3f}")
        return result
