import numpy as np
import torch

from src.datamodules.uss_dataset import USSDataset


def _event_position_to_unit_vector(event):
    """Extract a 3-D unit direction vector from a SpAudSyn event.

    Common SpAudSyn format is ``event["event_position"] == [[x, y, z]]``.
    The function also checks common metadata aliases and returns zeros for
    silence / missing / invalid positions.
    """
    candidates = []
    if isinstance(event, dict):
        for key in ("event_position", "position", "source_position"):
            if key in event:
                candidates.append(event[key])
        metadata = event.get("metadata", None)
        if isinstance(metadata, dict):
            for key in ("event_position", "position", "source_position"):
                if key in metadata:
                    candidates.append(metadata[key])

    for pos in candidates:
        arr = np.asarray(pos, dtype=np.float32)
        if arr.size < 3:
            continue
        arr = arr.reshape(-1, arr.shape[-1])[0, :3]
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            return arr / norm
    return np.zeros(3, dtype=np.float32)


def _extract_spatial_vectors(events, n_expected):
    vectors = [_event_position_to_unit_vector(event) for event in events[:n_expected]]
    while len(vectors) < n_expected:
        vectors.append(np.zeros(3, dtype=np.float32))
    return torch.from_numpy(np.stack(vectors, axis=0)).to(torch.float32)


class USSSpatialDataset(USSDataset):
    """USS dataset wrapper that additionally returns foreground spatial vectors.

    This class is opt-in and leaves ``USSDataset`` unchanged. It relies on the
    ``return_meta=True`` behavior already used by USSDataset to access SpAudSyn
    foreground event metadata.

    Output key:
        spatial_vector: FloatTensor [n_foreground, 3]

    The vector is a unit Cartesian DoA-like direction derived from event_position.
    Padded / silence slots are zeros.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        metadata = item.get("metadata", {})
        fg_events = metadata.get("fg_events", metadata.get("foreground_events", []))
        item["spatial_vector"] = _extract_spatial_vectors(
            fg_events,
            n_expected=self.base_dataset.n_sources,
        )
        return item
