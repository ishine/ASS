import numpy as np
import torch

from src.datamodules.uss_dataset import _event_position_to_unit_vector, _extract_spatial_targets
from src.models.deft.spatial_heads import ForegroundSpatialHead


def test_event_position_to_unit_vector_from_xyz_aliases():
    event = {"event_position": [[3.0, 4.0, 0.0]]}
    direction = _event_position_to_unit_vector(event)
    assert direction is not None
    np.testing.assert_allclose(direction, np.array([0.6, 0.8, 0.0], dtype=np.float32), atol=1e-6)


def test_event_position_to_unit_vector_from_nested_metadata_and_angles():
    event = {"metadata": {"azimuth_deg": 90.0, "elevation_deg": 0.0}}
    direction = _event_position_to_unit_vector(event)
    assert direction is not None
    np.testing.assert_allclose(direction, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)


def test_extract_spatial_targets_masks_missing_positions():
    directions, mask = _extract_spatial_targets(
        [
            {"event_position": [1.0, 0.0, 0.0]},
            {"metadata": {"label": "Speech"}},
        ],
        n_expected=3,
    )
    assert directions.shape == (3, 3)
    assert mask.tolist() == [True, False, False]
    assert torch.allclose(directions[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(directions[1], torch.zeros(3))


def test_foreground_spatial_head_shapes_and_normalization():
    head = ForegroundSpatialHead(hidden_channels=12, embedding_dim=8)
    features = torch.randn(2, 3, 12, 5, 7)
    spatial_embedding, doa_vector = head(features)

    assert spatial_embedding.shape == (2, 3, 8)
    assert doa_vector.shape == (2, 3, 3)
    assert torch.isfinite(spatial_embedding).all()
    assert torch.isfinite(doa_vector).all()
    assert torch.allclose(spatial_embedding.norm(dim=-1), torch.ones(2, 3), atol=1e-5)
    assert torch.allclose(doa_vector.norm(dim=-1), torch.ones(2, 3), atol=1e-5)
