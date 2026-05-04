import torch

from src.models.deft.foa_spatial_features import (
    ModifiedDeFTUSSMemoryEfficientSpatialFeaturesCountHead,
)
from src.models.deft.uss_count_head import ForegroundCountHead


def test_combined_foa_spatial_count_head_model_outputs_count_logits():
    model = ModifiedDeFTUSSMemoryEfficientSpatialFeaturesCountHead(
        input_channels=4,
        output_channels=1,
        hidden_channels=8,
        n_deft_blocks=1,
        n_heads=1,
        n_foreground=3,
        n_interference=1,
        n_classes=18,
        window_size=64,
        hop_size=16,
        time_window_size=16,
        freq_group_size=16,
        shift_windows=False,
        sample_rate=16000,
        inference_chunk_seconds=None,
        count_hidden_dim=16,
        max_count=3,
    )
    model.eval()
    mixture = torch.randn(2, 4, 512)
    with torch.no_grad():
        out = model({"mixture": mixture})

    assert "count_logits" in out
    assert out["count_logits"].shape == (2, 4)
    assert out["foreground_waveform"].shape[:3] == (2, 3, 1)
    assert torch.isfinite(out["count_logits"]).all()


def test_foreground_count_head_accepts_foa_spatial_output_dict_shape():
    output = {
        "foreground_waveform": torch.randn(2, 3, 1, 128),
        "class_logits": torch.randn(2, 3, 18),
        "silence_logits": torch.randn(2, 3),
    }
    head = ForegroundCountHead(n_foreground=3, n_classes=18, hidden_dim=16, max_count=3)
    logits = head(output)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
