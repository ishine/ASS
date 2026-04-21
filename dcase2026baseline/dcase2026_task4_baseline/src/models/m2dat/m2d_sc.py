import torch
import torch.nn as nn
import torch.nn.functional as F

from .portable_m2d import PortableM2D


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-8))
        phi = cosine * torch.cos(torch.tensor(self.m, device=x.device)) - sine * torch.sin(torch.tensor(self.m, device=x.device))
        one_hot = F.one_hot(labels, num_classes=cosine.shape[-1]).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s


class M2DSingleClassifier(PortableM2D):
    def __init__(
        self,
        weight_file,
        num_classes=18,
        embedding_dim=512,
        finetuning_layers="2_blocks",
        energy_thresholds=None,
        ref_channel=None,
    ):
        super().__init__(weight_file, num_classes=None, freeze_embed=False, flat_features=None)
        self.num_classes = num_classes
        self.ref_channel = ref_channel
        self.energy_thresholds = energy_thresholds or {}

        self.embedding = nn.Sequential(
            nn.Linear(self.cfg.feature_d, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.arc_head = ArcMarginProduct(embedding_dim, num_classes=num_classes)

        modules = [self.backbone.cls_token, self.backbone.pos_embed, self.backbone.patch_embed, self.backbone.pos_drop, self.backbone.patch_drop, self.backbone.norm_pre]
        for block in self.backbone.blocks:
            modules.append(block)
        modules.extend([self.backbone.norm, self.backbone.fc_norm, self.backbone.head_drop, self.embedding, self.arc_head])

        finetuning_modules_idx = {"head": len(modules) - 2, "2_blocks": len(modules) - 4, "all": 0}
        modules_idx = finetuning_modules_idx.get(finetuning_layers, len(modules) - 2)
        for i, module in enumerate(modules):
            if isinstance(module, torch.nn.parameter.Parameter):
                module.requires_grad = i >= modules_idx
            else:
                for param in module.parameters():
                    param.requires_grad = i >= modules_idx

    def _prepare_audio(self, waveform):
        if waveform.dim() == 3:
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            else:
                assert self.ref_channel is not None
                waveform = waveform[:, self.ref_channel]
        return waveform

    def forward(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        features = self.encode(waveform, average_per_time_frame=False).mean(1)
        embedding = self.embedding(features)
        logits = self.arc_head(embedding, input_dict.get("class_index"))
        plain_logits = self.arc_head(embedding, None)
        energy = -torch.logsumexp(plain_logits, dim=-1)
        return {
            "embedding": embedding,
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
        }

    def predict(self, input_dict):
        output = self.forward(input_dict)
        probs = torch.softmax(output["plain_logits"], dim=-1)
        values, indices = torch.max(probs, dim=-1)
        labels = F.one_hot(indices, num_classes=self.num_classes).float()

        silence = []
        for idx, energy in zip(indices.tolist(), output["energy"].tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else energy > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        return {
            "label_vector": labels,
            "probabilities": values,
            "energy": output["energy"],
            "silence": silence,
        }
