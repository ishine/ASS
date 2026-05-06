import numpy as np
import torch

from src.datamodules.dataset import DatasetS3
from src.temporal import event_to_span_sec, pad_spans


def _extract_waveforms(events, n_expected, length):
    waveforms = []
    for event in events[:n_expected]:
        wav = event.get("waveform_dry", event.get("waveform", None))
        if wav is None:
            continue
        wav = np.asarray(wav)
        if wav.ndim == 1:
            wav = wav[None, :]
        waveforms.append(wav.astype(np.float32))
    while len(waveforms) < n_expected:
        waveforms.append(np.zeros((1, length), dtype=np.float32))
    return torch.from_numpy(np.stack(waveforms, axis=0))


def _extract_spans(events, n_expected):
    return pad_spans([event_to_span_sec(event) for event in events[:n_expected]], n_expected)


def _first_present(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_position_array(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[0]
    if arr.shape[0] < 3:
        return None
    return arr[:3].astype(np.float32)


def _event_position_to_unit_vector(event):
    """Extract a unit xyz direction vector from a synthesized event metadata dict.

    DatasetS3 generate mode stores positions as ``event_position`` inside the
    event dict before synthesis.  Metadata files may contain slightly different
    names, so this helper accepts common aliases and nested ``metadata`` fields.
    If no valid position exists, callers receive ``None`` and should mask the
    spatial loss for that source.
    """

    candidates = [event]
    if isinstance(event, dict) and isinstance(event.get("metadata"), dict):
        candidates.append(event["metadata"])

    for candidate in candidates:
        position = _first_present(
            candidate,
            (
                "event_position",
                "source_position",
                "position",
                "xyz",
                "location",
                "source_location",
            ),
        )
        position = _as_position_array(position)
        if position is None:
            continue
        norm = np.linalg.norm(position)
        if norm > 1e-8:
            return (position / norm).astype(np.float32)

    # Fall back to azimuth/elevation if future metadata stores angles instead of
    # xyz positions.  Values are accepted in degrees or radians.
    for candidate in candidates:
        azimuth = _first_present(candidate, ("azimuth", "azimuth_deg", "azi", "theta"))
        elevation = _first_present(candidate, ("elevation", "elevation_deg", "ele", "phi"))
        if azimuth is None:
            continue
        azimuth = float(np.asarray(azimuth).reshape(-1)[0])
        elevation = 0.0 if elevation is None else float(np.asarray(elevation).reshape(-1)[0])
        if abs(azimuth) > 2 * np.pi or abs(elevation) > 2 * np.pi:
            azimuth = np.deg2rad(azimuth)
            elevation = np.deg2rad(elevation)
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float32,
        )
        norm = max(float(np.linalg.norm(direction)), 1e-8)
        return (direction / norm).astype(np.float32)
    return None


def _extract_spatial_targets(events, n_expected):
    directions = []
    mask = []
    for event in events[:n_expected]:
        direction = _event_position_to_unit_vector(event)
        if direction is None:
            directions.append(np.zeros(3, dtype=np.float32))
            mask.append(False)
        else:
            directions.append(direction)
            mask.append(True)
    while len(directions) < n_expected:
        directions.append(np.zeros(3, dtype=np.float32))
        mask.append(False)
    return torch.from_numpy(np.stack(directions, axis=0)), torch.tensor(mask, dtype=torch.bool)


class USSDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        args = base_dataset["args"].copy()
        args["return_meta"] = True
        self.base_dataset = DatasetS3(**args)
        self.labels = self.base_dataset.labels
        self.collate_fn = self.base_dataset.collate_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        metadata = item.get("metadata", {})
        length = item["mixture"].shape[-1]

        fg_events = metadata.get("fg_events", metadata.get("foreground_events", []))
        int_events = metadata.get("int_events", metadata.get("interference_events", []))
        background = metadata.get("bg_events", metadata.get("background", metadata.get("bg_event", {})))
        background_events = [background] if isinstance(background, dict) else background

        labels = item["label"]
        class_index = []
        is_silence = []
        for label in labels:
            is_silence.append(label == "silence")
            class_index.append(0 if label == "silence" else self.labels.index(label))

        foreground_doa, foreground_doa_mask = _extract_spatial_targets(fg_events, self.base_dataset.n_sources)
        foreground_doa_mask = foreground_doa_mask & ~torch.tensor(is_silence, dtype=torch.bool)

        item["foreground_waveform"] = item["dry_sources"].to(torch.float32)
        item["interference_waveform"] = _extract_waveforms(int_events, 2, length)
        item["noise_waveform"] = _extract_waveforms(background_events, 1, length)
        item["foreground_span_sec"] = item.get("span_sec", torch.full((self.base_dataset.n_sources, 2), -1.0, dtype=torch.float32))
        item["interference_span_sec"] = _extract_spans(int_events, 2)
        item["noise_span_sec"] = _extract_spans(background_events, 1)
        item["class_index"] = torch.tensor(class_index, dtype=torch.long)
        item["is_silence"] = torch.tensor(is_silence, dtype=torch.bool)
        item["foreground_doa"] = foreground_doa.to(torch.float32)
        item["foreground_doa_mask"] = foreground_doa_mask
        return item
