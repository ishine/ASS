import torch

from src.datamodules.dataset import DatasetS3


class TSEDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        args = base_dataset["args"].copy()
        args["return_source"] = True
        self.base_dataset = DatasetS3(**args)
        self.collate_fn = self.base_dataset.collate_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        labels = item["label"]
        label_stack = []
        active_mask = []
        for source_vec, label in zip(item["label_vector"].view(self.base_dataset.n_sources, -1), labels):
            if label == "silence":
                label_stack.append(torch.zeros_like(source_vec))
                active_mask.append(False)
            else:
                label_stack.append(source_vec)
                active_mask.append(True)
        item["label_vector"] = torch.stack(label_stack, dim=0)
        item["enrollment"] = item["dry_sources"].clone()
        item["waveform"] = item["dry_sources"]
        item["active_mask"] = torch.tensor(active_mask, dtype=torch.bool)
        if "span_sec" not in item:
            item["span_sec"] = torch.full((self.base_dataset.n_sources, 2), -1.0, dtype=torch.float32)
        return item
