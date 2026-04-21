import torch
from src.datamodules.dataset import DatasetS3


class SourceClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = DatasetS3(**base_dataset["args"]) if isinstance(base_dataset, dict) else base_dataset
        self.labels = self.base_dataset.labels
        self.collate_fn = self._collate_fn

    def __len__(self):
        return len(self.base_dataset) * self.base_dataset.n_sources

    def __getitem__(self, idx):
        mixture_idx = idx // self.base_dataset.n_sources
        source_idx = idx % self.base_dataset.n_sources
        item = self.base_dataset[mixture_idx]
        label = item["label"][source_idx]
        is_silence = label == "silence"
        class_index = 0 if is_silence else self.labels.index(label)
        return {
            "waveform": item["dry_sources"][source_idx],
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "is_silence": torch.tensor(is_silence, dtype=torch.bool),
        }

    def _collate_fn(self, items):
        return {
            "waveform": torch.stack([x["waveform"] for x in items], dim=0),
            "class_index": torch.stack([x["class_index"] for x in items], dim=0),
            "is_silence": torch.stack([x["is_silence"] for x in items], dim=0),
        }
