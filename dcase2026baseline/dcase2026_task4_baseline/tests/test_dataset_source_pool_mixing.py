import random
from collections import Counter

from src.datamodules.dataset import DatasetS3


def _base_config(weights=(0.8, 0.2)):
    return {
        "mode": "generate",
        "spatial_sound_scene": {
            "duration": 10.0,
            "sr": 32000,
            "max_event_overlap": 3,
            "max_event_dur": 10.0,
            "ref_db": -55,
            "foreground_dir": "official/sound_event/train",
            "background_dir": "official/noise/train",
            "interference_dir": "official/interference/train",
            "room_config": {"module": "unused", "main": "UnusedRoom", "args": {}},
        },
        "spatial_sound_scene_sources": {
            "sampling_mode": "scene_weighted",
            "pools": [
                {
                    "name": "official",
                    "weight": weights[0],
                    "foreground_dir": "official/sound_event/train",
                    "background_dir": "official/noise/train",
                    "interference_dir": "official/interference/train",
                },
                {
                    "name": "audioset_strong",
                    "weight": weights[1],
                    "foreground_dir": "audioset/sound_event/train",
                    "background_dir": "official/noise/train",
                    "interference_dir": "audioset/interference/train",
                },
            ],
        },
        "dupse_rate": 0.5,
        "dupse_min_angle": 60.0,
        "max_n_dupse": 2,
        "dupse_exclusion_folder_depth": 1,
        "snr_range": [5, 20],
        "nevent_range": [0, 3],
        "ninterference_range": [0, 2],
        "inteference_snr_range": [0, 15],
        "dataset_length": 8,
        "shuffle_label": True,
        "fg_return": {"dry": True, "dry_channel": 0, "metadata": True},
    }


def _dataset(config):
    return DatasetS3(
        config=config,
        n_sources=3,
        label_set="dcase2026t4",
        return_source=True,
        label_vector_mode="stack",
        silence_label_mode="zeros",
    )


def test_weighted_source_pool_selection_uses_active_ratios():
    random.seed(0)
    dataset = _dataset(_base_config(weights=(0.8, 0.2)))

    counts = Counter(dataset._select_spatial_sound_scene()[1] for _ in range(1000))

    assert 700 < counts["official"] < 900
    assert 100 < counts["audioset_strong"] < 300


def test_zero_weight_source_pool_is_not_selected():
    dataset = _dataset(_base_config(weights=(1.0, 0.0)))

    selected = [dataset._select_spatial_sound_scene() for _ in range(20)]

    assert {pool_name for _scene, pool_name in selected} == {"official"}
    assert all(scene["foreground_dir"] == "official/sound_event/train" for scene, _pool_name in selected)
