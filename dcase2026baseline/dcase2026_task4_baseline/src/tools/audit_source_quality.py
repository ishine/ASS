from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import librosa
import numpy as np
from tqdm import tqdm

from src.utils import LABELS


def iter_wavs(root: str) -> Iterable[str]:
    for cur, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".wav"):
                yield os.path.join(cur, name)


def infer_declared_label(path: str, foreground_dir: str, labels: List[str]) -> str:
    rel = os.path.relpath(path, foreground_dir)
    parts = rel.split(os.sep)
    for part in parts[:-1]:
        if part in labels:
            return part
    for label in labels:
        if label in os.path.basename(path):
            return label
    return parts[0] if len(parts) > 1 else "unknown"


def load_teacher_predictions(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    records: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                rec = dict(value) if isinstance(value, dict) else {"probs": value}
                rec.setdefault("path", key)
                records.append(rec)
    elif path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        raise ValueError("teacher predictions must be .jsonl, .json, or .csv")

    out = {}
    for rec in records:
        p = rec.get("path") or rec.get("source_file") or rec.get("filename") or rec.get("file")
        if not p:
            continue
        keys = {os.path.normpath(str(p)), os.path.basename(str(p)), os.path.splitext(os.path.basename(str(p)))[0]}
        for key in keys:
            out[key] = rec
    return out


def probs_from_record(rec: Dict[str, Any], labels: List[str]) -> Dict[str, float]:
    probs = rec.get("probs") or rec.get("teacher_probs") or rec.get("soft_target")
    if isinstance(probs, dict):
        return {label: float(probs.get(label, 0.0)) for label in labels}
    if isinstance(probs, list) and len(probs) == len(labels):
        return {label: float(v) for label, v in zip(labels, probs)}
    out = defaultdict(float)
    for label in labels:
        for key in (label, f"prob_{label}", f"p_{label}"):
            if key in rec:
                out[label] = float(rec[key])
                break
    if sum(out.values()) > 0:
        return dict(out)
    top1 = rec.get("teacher_top1") or rec.get("top1") or rec.get("pred_label")
    top1_prob = float(rec.get("teacher_top1_prob", rec.get("top1_prob", rec.get("prob", 0.0))) or 0.0)
    if top1 in labels and top1_prob > 0:
        out[top1] = top1_prob
        top2 = rec.get("teacher_top2") or rec.get("top2")
        top2_prob = float(rec.get("teacher_top2_prob", rec.get("top2_prob", 0.0)) or 0.0)
        if top2 in labels and top2_prob > 0:
            out[top2] = top2_prob
        return dict(out)
    return {}


def audio_stats(path: str, sr: Optional[int]) -> Dict[str, float]:
    y, actual_sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return {"duration_sec": 0.0, "energy_db": -120.0, "active_ratio": 0.0}
    duration = float(y.size) / float(actual_sr)
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    energy_db = float(20.0 * np.log10(max(rms, 1e-8)))
    frame = max(1, int(0.02 * actual_sr))
    hop = max(1, int(0.01 * actual_sr))
    if y.size < frame:
        active_ratio = float(np.mean(np.abs(y) > 0.01))
    else:
        frames = librosa.util.frame(y, frame_length=frame, hop_length=hop)
        frms = np.sqrt(np.mean(np.square(frames), axis=0) + 1e-12)
        active_ratio = float(np.mean(frms > max(0.01, rms * 0.25)))
    return {"duration_sec": duration, "energy_db": energy_db, "active_ratio": active_ratio}


def classify_quality(
    declared_label: str,
    probs: Dict[str, float],
    clean_prob: float,
    clean_margin: float,
    bad_prob: float,
    uncertain_secondary_prob: float,
) -> Dict[str, Any]:
    if not probs:
        soft = {declared_label: 1.0}
        return {
            "teacher_top1": declared_label,
            "teacher_top1_prob": 1.0,
            "teacher_top2": None,
            "teacher_top2_prob": 0.0,
            "teacher_margin": 1.0,
            "declared_label_prob": 1.0,
            "quality_group": "clean",
            "semantic_confidence": 1.0,
            "soft_target": soft,
        }
    total = sum(max(0.0, v) for v in probs.values())
    if total <= 0:
        return classify_quality(declared_label, {}, clean_prob, clean_margin, bad_prob, uncertain_secondary_prob)
    probs = {k: max(0.0, float(v)) / total for k, v in probs.items()}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1, p1 = ranked[0]
    top2, p2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
    declared_prob = float(probs.get(declared_label, 0.0))
    margin = float(p1 - p2)
    if top1 == declared_label and declared_prob >= clean_prob and margin >= clean_margin and p2 < uncertain_secondary_prob:
        group = "clean"
        conf = min(1.0, declared_prob)
    elif declared_prob < bad_prob or (top1 != declared_label and p1 >= clean_prob):
        group = "bad"
        conf = max(0.0, declared_prob)
    else:
        group = "uncertain"
        conf = max(0.05, min(0.7, declared_prob))
    return {
        "teacher_top1": top1,
        "teacher_top1_prob": float(p1),
        "teacher_top2": top2,
        "teacher_top2_prob": float(p2),
        "teacher_margin": margin,
        "declared_label_prob": declared_prob,
        "quality_group": group,
        "semantic_confidence": conf,
        "soft_target": probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label_set", default="dcase2026t4")
    parser.add_argument("--teacher_predictions", default=None)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clean_prob", type=float, default=0.65)
    parser.add_argument("--clean_margin", type=float, default=0.25)
    parser.add_argument("--bad_prob", type=float, default=0.20)
    parser.add_argument("--uncertain_secondary_prob", type=float, default=0.25)
    args = parser.parse_args()

    labels = LABELS[args.label_set]
    teacher = load_teacher_predictions(args.teacher_predictions)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    counts = defaultdict(int)
    with open(args.output, "w", encoding="utf-8") as f:
        for path in tqdm(sorted(iter_wavs(args.foreground_dir)), desc="audit source quality"):
            declared = infer_declared_label(path, args.foreground_dir, labels)
            keys = [os.path.normpath(path), os.path.basename(path), os.path.splitext(os.path.basename(path))[0]]
            rec = next((teacher[k] for k in keys if k in teacher), {})
            probs = probs_from_record(rec, labels)
            q = classify_quality(
                declared,
                probs,
                clean_prob=args.clean_prob,
                clean_margin=args.clean_margin,
                bad_prob=args.bad_prob,
                uncertain_secondary_prob=args.uncertain_secondary_prob,
            )
            stats = audio_stats(path, args.sample_rate)
            out = {"path": path, "declared_label": declared, **q, **stats}
            counts[out["quality_group"]] += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("source quality counts:", dict(counts))


if __name__ == "__main__":
    main()
