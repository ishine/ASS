import json
import os
import random
from typing import Any, Dict, List, Optional

import torch

from src.datamodules.uss_dataset import USSDataset


_GROUP_ALIASES = {
    "good": "clean",
    "trusted": "clean",
    "ok": "clean",
    "ambiguous": "uncertain",
    "weak": "uncertain",
    "dirty": "bad",
    "bled": "bad",
    "blooded": "bad",
    "impure": "bad",
    "exclude": "bad",
}


def _norm_group(value: Any, default: str = "clean") -> str:
    if value is None:
        return default
    group = str(value).strip().lower()
    return _GROUP_ALIASES.get(group, group if group in {"clean", "uncertain", "bad"} else default)


def _path_keys(path: str) -> List[str]:
    if not path:
        return []
    p = os.path.normpath(str(path))
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    return list(dict.fromkeys([p, p.replace("\\", "/"), base, stem]))


class SourceQualityManifest:
    """Lookup table for foreground stem quality.

    Supported manifest formats:
      - JSON list of dicts
      - JSON dict mapping path -> entry
      - JSONL, one dict per line

    Recommended entry fields:
      path, declared_label, quality_group, semantic_confidence, soft_target
    """

    def __init__(self, path: Optional[str], labels: List[str], default_group: str = "clean"):
        self.labels = list(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.default_group = default_group
        self.entries: Dict[str, Dict[str, Any]] = {}
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if path.endswith(".jsonl"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                records = []
                for key, value in obj.items():
                    if isinstance(value, dict):
                        rec = dict(value)
                        rec.setdefault("path", key)
                        records.append(rec)
            elif isinstance(obj, list):
                records = obj
            else:
                raise ValueError("source quality manifest must be JSON list, JSON dict, or JSONL")
        for rec in records:
            for key in _path_keys(str(rec.get("path", rec.get("source_file", "")))):
                self.entries[key] = rec

    def _event_candidate_keys(self, event: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        stack = [event]
        meta = event.get("metadata") if isinstance(event, dict) else None
        if isinstance(meta, dict):
            stack.append(meta)
        for obj in stack:
            for key in (
                "path", "source_path", "source_file", "file", "filename", "file_name",
                "audio_path", "wav_path", "source", "source_name",
            ):
                value = obj.get(key)
                if isinstance(value, (str, os.PathLike)):
                    candidates.extend(_path_keys(str(value)))
                elif isinstance(value, dict):
                    for subkey in ("path", "source_file", "filename", "file"):
                        if isinstance(value.get(subkey), (str, os.PathLike)):
                            candidates.extend(_path_keys(str(value[subkey])))
        return list(dict.fromkeys(candidates))

    def lookup_event(self, event: Dict[str, Any], label: str) -> Dict[str, Any]:
        rec = None
        for key in self._event_candidate_keys(event):
            if key in self.entries:
                rec = self.entries[key]
                break
        if rec is None:
            rec = {}
        group = _norm_group(rec.get("quality_group", rec.get("group")), self.default_group)
        conf = rec.get("semantic_confidence", rec.get("class_confidence", None))
        if conf is None:
            conf = 1.0 if group == "clean" else 0.35 if group == "uncertain" else 0.0
        conf = float(max(0.0, min(1.0, conf)))
        soft = torch.zeros(len(self.labels), dtype=torch.float32)
        soft_target = rec.get("soft_target", rec.get("teacher_probs", None))
        if isinstance(soft_target, dict):
            for name, value in soft_target.items():
                if name in self.label_to_idx:
                    soft[self.label_to_idx[name]] = float(value)
        elif isinstance(soft_target, list) and len(soft_target) == len(self.labels):
            soft = torch.tensor(soft_target, dtype=torch.float32)
        if soft.sum() <= 0 and label in self.label_to_idx:
            soft[self.label_to_idx[label]] = 1.0
        elif soft.sum() > 0:
            soft = soft / soft.sum().clamp_min(1e-8)
        return {"quality_group": group, "class_confidence": conf, "soft_target": soft}


class QualityAwareUSSDataset(torch.utils.data.Dataset):
    """Opt-in USS dataset wrapper for noisy / bled foreground labels.

    It preserves all USSDataset keys and adds optional tensors:
      class_confidence:    [S]
      soft_class_target:   [S, C]
      uncertain_slot_mask: [S]
      bad_slot_mask:       [S]

    In generate mode it can resample scenes containing bad foreground stems.
    """

    def __init__(
        self,
        base_dataset,
        source_quality_manifest: Optional[str] = None,
        default_quality_group: str = "clean",
        resample_bad_foreground: bool = True,
        max_resample_attempts: int = 8,
        uncertain_class_weight: float = 0.35,
        bad_class_weight: float = 0.0,
    ):
        self.uss_dataset = USSDataset(base_dataset=base_dataset)
        self.base_dataset = self.uss_dataset.base_dataset
        self.labels = self.uss_dataset.labels
        self.collate_fn = self.uss_dataset.collate_fn
        self.manifest = SourceQualityManifest(source_quality_manifest, self.labels, default_group=default_quality_group)
        self.resample_bad_foreground = bool(resample_bad_foreground)
        self.max_resample_attempts = int(max_resample_attempts)
        self.uncertain_class_weight = float(uncertain_class_weight)
        self.bad_class_weight = float(bad_class_weight)

    def __len__(self):
        return len(self.uss_dataset)

    def _fg_events(self, item):
        meta = item.get("metadata", {})
        return meta.get("fg_events", meta.get("foreground_events", []))

    def _quality_tensors(self, item):
        labels = item["label"]
        events = list(self._fg_events(item))
        n_sources = self.base_dataset.n_sources
        n_classes = len(self.labels)
        class_conf = torch.ones(n_sources, dtype=torch.float32)
        soft = torch.zeros(n_sources, n_classes, dtype=torch.float32)
        uncertain = torch.zeros(n_sources, dtype=torch.bool)
        bad = torch.zeros(n_sources, dtype=torch.bool)
        groups: List[str] = []
        for idx in range(n_sources):
            label = labels[idx] if idx < len(labels) else "silence"
            if label == "silence":
                class_conf[idx] = 0.0
                groups.append("silence")
                continue
            event = events[idx] if idx < len(events) else {}
            q = self.manifest.lookup_event(event, label)
            group = q["quality_group"]
            groups.append(group)
            if group == "uncertain":
                class_conf[idx] = min(float(q["class_confidence"]), self.uncertain_class_weight)
                uncertain[idx] = True
            elif group == "bad":
                class_conf[idx] = min(float(q["class_confidence"]), self.bad_class_weight)
                bad[idx] = True
            else:
                class_conf[idx] = float(q["class_confidence"])
            soft[idx] = q["soft_target"]
        return class_conf, soft, uncertain, bad, groups

    def _attach_quality(self, item):
        class_conf, soft, uncertain, bad, groups = self._quality_tensors(item)
        item["class_confidence"] = class_conf
        item["soft_class_target"] = soft
        item["uncertain_slot_mask"] = uncertain
        item["bad_slot_mask"] = bad
        item["source_quality_group"] = groups
        return item

    def __getitem__(self, idx):
        item = self.uss_dataset[idx]
        item = self._attach_quality(item)
        mode = getattr(self.base_dataset, "config", {}).get("mode")
        if self.resample_bad_foreground and mode == "generate":
            attempts = 0
            while item["bad_slot_mask"].any() and attempts < self.max_resample_attempts:
                item = self._attach_quality(self.uss_dataset[random.randrange(len(self.uss_dataset))])
                attempts += 1
            item["quality_resample_attempts"] = torch.tensor(attempts, dtype=torch.long)
        return item
