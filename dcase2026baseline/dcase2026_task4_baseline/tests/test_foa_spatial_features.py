import pytest
import torch

from src.models.deft.foa_spatial_features import FOASpatialFeatureEncoder


def test_foa_spatial_feature_encoder_default_shape():
    encoder = FOASpatialFeatureEncoder(input_channels=4, hidden_channels=16)
    raw_complex = torch.randn(2, 8, 5, 9)

    features = encoder.build_features(raw_complex)
    encoded = encoder(raw_complex)

    assert encoder.feature_channels == 21
    assert features.shape == (2, 21, 5, 9)
    assert encoded.shape == (2, 16, 5, 9)
    assert torch.isfinite(features).all()
    assert torch.isfinite(encoded).all()


def test_foa_spatial_feature_encoder_can_disable_features():
    encoder = FOASpatialFeatureEncoder(
        input_channels=4,
        hidden_channels=16,
        include_logmag=False,
        include_aiv=True,
        include_ipd=False,
    )
    raw_complex = torch.randn(2, 8, 5, 9)
    assert encoder.feature_channels == 11
    assert encoder.build_features(raw_complex).shape == (2, 11, 5, 9)


def test_aiv_ipd_requires_foa_input():
    with pytest.raises(ValueError, match="input_channels=4"):
        FOASpatialFeatureEncoder(input_channels=1, include_aiv=True, include_ipd=True)
