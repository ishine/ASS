"""DCASE 2026 Task 4 validation breakdown metrics.

This module complements the official CAPI-SDRi scorer with category-level
aggregates that expose the new 2026 failure modes:

    - zero-target mixtures
    - one-target mixtures
    - 2--3 target mixtures with all classes distinct
    - 2--3 target mixtures with at least one same-class duplicate

By default it keeps the official-baseline-compatible assignment behavior.  Set
``assignment_mode='compare'`` to additionally log the paper-definition SDRi
assignment values and their difference from the raw-SDR assignment values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional

import torch

from src.evaluation.metrics.s5capi_metric import S5ClassAwareMetric, S5ClassAwareMetricSDRiAssignment


_CAPI_BUCKET_SUFFIX = {
    "one_target": "1target",
    "distinct_class": "distinct_class",
    "same_class_duplicate": "same_class_duplicate",
    "zero_target": "zero_target_fp",
}


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
    """Aggregate CAPI-SDRi and detection diagnostics by scene type.

    Args:
        metricfunc: waveform metric backend.  Currently only ``'sdr'`` is
            supported by the underlying S5 CAPI metric.
        prefix: logging prefix.
        assignment_mode:
            ``'official'`` or ``'raw_sdr'`` keeps the official-baseline
            compatible behavior: select same-class permutations by raw SDR and
            report SDRi.
            ``'sdri'`` selects same-class permutations by SDRi and writes those
            values to the standard ``capi_sdri_*`` keys.
            ``'compare'`` logs both assignment variants.  The existing
            ``capi_sdri_*`` keys remain raw-SDR-assignment compatible, while
            ``capi_sdri_sdri_assignment_*`` and
            ``capi_sdri_assignment_delta_*`` expose the paper-definition
            diagnostic.
    """

    metric_name = "S5 validation breakdown"

    def __init__(self, metricfunc: str = "sdr", prefix: str = "valid", assignment_mode: str = "official"):
        self.prefix = str(prefix).rstrip("/")
        assignment_mode = str(assignment_mode).lower()
        if assignment_mode == "raw_sdr":
            assignment_mode = "official"
        if assignment_mode not in {"official", "sdri", "compare"}:
            raise ValueError(
                "assignment_mode must be one of 'official', 'raw_sdr', 'sdri', or 'compare', "
                f"got {assignment_mode!r}"
            )
        self.assignment_mode = assignment_mode
        self.raw_assignment_metric = S5ClassAwareMetric(metricfunc=metricfunc)
        self.sdri_assignment_metric = S5ClassAwareMetricSDRiAssignment(metricfunc=metricfunc)
        self.reset()

    def reset(self):
        self.values = defaultdict(list)
        self.sample_values = []
        self.silence_tp = 0
        self.silence_fp = 0
        self.silence_fn = 0
        self.raw_assignment_metric.reset()
        self.sdri_assignment_metric.reset()

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

    def _add_capi_value(self, bucket: str, capi_value, key_prefix: str = "capi_sdri"):
        if capi_value is None:
            return
        capi_value = float(capi_value)
        self.values[f"{key_prefix}_all"].append(capi_value)
        bucket_suffix = _CAPI_BUCKET_SUFFIX.get(bucket)
        if bucket_suffix is not None:
            # True zero-target silence predictions are undefined/excluded by the
            # official scorer.  Zero-target false positives receive a 0 dB value.
            self.values[f"{key_prefix}_{bucket_suffix}"].append(capi_value)

    def _add_assignment_delta(self, bucket: str, raw_value, sdri_value):
        if raw_value is None or sdri_value is None:
            return
        delta = float(sdri_value) - float(raw_value)
        self.values["capi_sdri_assignment_delta_all"].append(delta)
        bucket_suffix = _CAPI_BUCKET_SUFFIX.get(bucket)
        if bucket_suffix is not None:
            self.values[f"capi_sdri_assignment_delta_{bucket_suffix}"].append(delta)

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

    def _compute_assignment_values(self, est_labels, est_wf, ref_labels, ref_wf, mixture):
        raw_value = None
        sdri_value = None
        if self.assignment_mode in {"official", "compare"}:
            raw_value = self.raw_assignment_metric.compute_sample(est_labels, est_wf, ref_labels, ref_wf, mixture)
        if self.assignment_mode in {"sdri", "compare"}:
            sdri_value = self.sdri_assignment_metric.compute_sample(est_labels, est_wf, ref_labels, ref_wf, mixture)
        return raw_value, sdri_value

    def compute_sample(self, est_lb, est_wf, ref_lb, ref_wf, mixture):
        est_labels = _as_label_list(est_lb)
        ref_labels = _as_label_list(ref_lb)
        est_active = _active_labels(est_labels)
        bucket = _scene_bucket(ref_labels)

        raw_value, sdri_value = self._compute_assignment_values(est_labels, est_wf, ref_labels, ref_wf, mixture)
        if self.assignment_mode == "official":
            capi_value = raw_value
            self._add_capi_value(bucket, raw_value, key_prefix="capi_sdri")
        elif self.assignment_mode == "sdri":
            capi_value = sdri_value
            self._add_capi_value(bucket, sdri_value, key_prefix="capi_sdri")
            self._add_capi_value(bucket, sdri_value, key_prefix="capi_sdri_sdri_assignment")
        else:
            # Keep the historical/official-compatible keys unchanged while also
            # exposing the paper-definition assignment diagnostic.
            capi_value = raw_value
            self._add_capi_value(bucket, raw_value, key_prefix="capi_sdri")
            self._add_capi_value(bucket, raw_value, key_prefix="capi_sdri_raw_assignment")
            self._add_capi_value(bucket, sdri_value, key_prefix="capi_sdri_sdri_assignment")
            self._add_assignment_delta(bucket, raw_value, sdri_value)

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

        # Explicit aliases for common experiment logs.  Values remain ``None``
        # until a batch contains the corresponding scene bucket/assignment mode.
        for key in (
            "capi_sdri_all",
            "capi_sdri_1target",
            "capi_sdri_distinct_class",
            "capi_sdri_same_class_duplicate",
            "capi_sdri_zero_target_fp",
            "capi_sdri_zero_target_fp_rate",
            "capi_sdri_raw_assignment_all",
            "capi_sdri_raw_assignment_1target",
            "capi_sdri_raw_assignment_distinct_class",
            "capi_sdri_raw_assignment_same_class_duplicate",
            "capi_sdri_raw_assignment_zero_target_fp",
            "capi_sdri_sdri_assignment_all",
            "capi_sdri_sdri_assignment_1target",
            "capi_sdri_sdri_assignment_distinct_class",
            "capi_sdri_sdri_assignment_same_class_duplicate",
            "capi_sdri_sdri_assignment_zero_target_fp",
            "capi_sdri_assignment_delta_all",
            "capi_sdri_assignment_delta_1target",
            "capi_sdri_assignment_delta_distinct_class",
            "capi_sdri_assignment_delta_same_class_duplicate",
            "capi_sdri_assignment_delta_zero_target_fp",
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
