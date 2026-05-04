import torch

from src.evaluation.metrics.s5_validation_breakdown import S5ValidationBreakdownMetric


def _waves(n_sources, length=64):
    if n_sources == 0:
        return torch.zeros(0, length)
    waves = []
    base = torch.linspace(-1.0, 1.0, length)
    for idx in range(n_sources):
        waves.append(base.roll(idx + 1) * (idx + 1) / n_sources)
    return torch.stack(waves, dim=0)


def test_s5_validation_breakdown_reports_requested_keys():
    metric = S5ValidationBreakdownMetric(prefix="valid")
    batch_est_labels = [
        ["silence", "silence", "silence"],
        ["Speech", "silence", "silence"],
        ["Speech", "Dog", "silence"],
        ["Speech", "Speech", "silence"],
    ]
    batch_ref_labels = [
        ["silence", "silence", "silence"],
        ["Speech", "silence", "silence"],
        ["Speech", "Dog", "silence"],
        ["Speech", "Speech", "silence"],
    ]
    batch_est_waveforms = torch.stack([
        _waves(3),
        _waves(3),
        _waves(3),
        _waves(3),
    ])
    batch_ref_waveforms = batch_est_waveforms.clone()
    batch_mixture = batch_ref_waveforms.sum(dim=1)

    metric.update(
        batch_est_labels=batch_est_labels,
        batch_est_waveforms=batch_est_waveforms,
        batch_ref_labels=batch_ref_labels,
        batch_ref_waveforms=batch_ref_waveforms,
        batch_mixture=batch_mixture,
    )
    summary = metric.compute()

    for key in (
        "valid/capi_sdri_all",
        "valid/capi_sdri_zero_target_fp_rate",
        "valid/capi_sdri_1target",
        "valid/capi_sdri_distinct_class",
        "valid/capi_sdri_same_class_duplicate",
        "valid/foreground_slot_active_rate",
        "valid/foreground_leakage_energy",
        "valid/silence_precision",
        "valid/silence_recall",
    ):
        assert key in summary

    assert summary["valid/capi_sdri_zero_target_fp_rate"] == 0.0
    assert summary["valid/silence_precision"] == 1.0
    assert summary["valid/silence_recall"] == 1.0


def test_s5_validation_breakdown_zero_target_false_positive_rate():
    metric = S5ValidationBreakdownMetric(prefix="valid")
    est_waveforms = torch.stack([_waves(3)])
    ref_waveforms = torch.zeros_like(est_waveforms)
    mixture = torch.zeros(1, est_waveforms.shape[-1])

    metric.update(
        batch_est_labels=[["Dog", "silence", "silence"]],
        batch_est_waveforms=est_waveforms,
        batch_ref_labels=[["silence", "silence", "silence"]],
        batch_ref_waveforms=ref_waveforms,
        batch_mixture=mixture,
    )
    summary = metric.compute()

    assert summary["valid/capi_sdri_zero_target_fp_rate"] == 1.0
    assert summary["valid/capi_sdri_all"] == 0.0
