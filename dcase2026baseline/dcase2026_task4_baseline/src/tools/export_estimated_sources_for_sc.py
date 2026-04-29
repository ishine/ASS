from __future__ import annotations

import argparse
import csv
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

from scipy.io import wavfile

from src.datamodules.dataset import DatasetS3
from src.utils import initialize_config, parse_yaml


def _load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = parse_yaml(config_path)
    if "lightning_module" in cfg:
        model_cfg = cfg["lightning_module"]["args"]["model"]
    elif "model" in cfg:
        model_cfg = cfg["model"]
    else:
        raise KeyError("Config must contain either lightning_module.args.model or model")
    model = initialize_config(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    clean_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        clean_state[key] = value
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[export_estimated_sources_for_sc] missing model keys: {len(missing)}")
    if unexpected:
        print(f"[export_estimated_sources_for_sc] unexpected model keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _build_dataset(args):
    return DatasetS3(
        config={
            "mode": "waveform",
            "soundscape_dir": args.soundscape_dir,
            "oracle_target_dir": args.oracle_target_dir,
            "sr": args.sample_rate,
        },
        n_sources=args.n_sources,
        label_set=args.label_set,
        return_source=True,
        label_vector_mode="stack",
        silence_label_mode="zeros",
        return_meta=False,
    )


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _flatten_sources(x: torch.Tensor) -> torch.Tensor:
    if x.dim() < 3:
        raise ValueError(f"Expected source tensor [B,S,...], got {tuple(x.shape)}")
    return x.flatten(start_dim=2).float()


def _pairwise_sa_sdr_score(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise SA-SDR score in dB, higher is better.

    Args:
        est: [B, S_est, 1, T] or [B, S_est, T]
        ref: [B, S_ref, 1, T] or [B, S_ref, T]

    Returns:
        [B, S_ref, S_est]
    """
    est_f = _flatten_sources(est)
    ref_f = _flatten_sources(ref)
    err = est_f.unsqueeze(1) - ref_f.unsqueeze(2)
    ref_power = ref_f.pow(2).sum(dim=-1).unsqueeze(2).clamp_min(eps)
    err_power = err.pow(2).sum(dim=-1).clamp_min(eps)
    ratio = (ref_power / err_power).clamp_min(eps)
    return 10.0 * torch.log10(ratio)


def _energy_db(source: torch.Tensor, eps: float = 1e-12) -> float:
    x = source.float().flatten()
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return float(20.0 * torch.log10(rms.clamp_min(1e-8)).item())


def _best_assignment(scores: torch.Tensor, active_ref_indices: List[int], n_est: int) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Brute-force target->estimate assignment for up to three foreground slots."""
    if not active_ref_indices:
        return {}, {}
    n_match = min(len(active_ref_indices), n_est)
    active_ref_indices = active_ref_indices[:n_match]
    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(n_est), n_match):
        vals = [scores[ref_idx, est_idx] for ref_idx, est_idx in zip(active_ref_indices, perm)]
        mean_score = torch.stack(vals).mean()
        if best_score is None or mean_score > best_score:
            best_score = mean_score
            best_perm = perm
    assignment = {ref_idx: int(est_idx) for ref_idx, est_idx in zip(active_ref_indices, best_perm)}
    matched_scores = {ref_idx: float(scores[ref_idx, est_idx].item()) for ref_idx, est_idx in assignment.items()}
    return assignment, matched_scores


def _match_margin(scores: torch.Tensor, ref_idx: int, est_idx: int) -> float:
    row = scores[ref_idx]
    if row.numel() <= 1:
        return float("inf")
    best = row[est_idx]
    others = torch.cat([row[:est_idx], row[est_idx + 1 :]])
    return float((best - others.max()).item())


def _write_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    if sf is not None:
        sf.write(path, waveform, sample_rate)
    else:  # fallback; scipy expects int or float accepted in recent versions
        wavfile.write(path, sample_rate, waveform)


def _safe_label(label: str) -> str:
    return str(label).replace("/", "_").replace(" ", "")


def _export_batch(output, batch, args, manifest_rows):
    est = output[args.output_key].detach().cpu().float()
    if est.dim() == 4 and est.shape[2] != 1:
        est = est[:, :, :1]
    ref = batch["dry_sources"].detach().cpu().float()
    labels = batch["label"]
    soundscapes = batch["soundscape"]
    scores = _pairwise_sa_sdr_score(est, ref).cpu()

    for b, soundscape in enumerate(soundscapes):
        active_ref = [i for i, label in enumerate(labels[b]) if label != "silence"]
        assignment, matched_scores = _best_assignment(scores[b], active_ref, est.shape[1])
        used_est = set()
        output_slot = 0
        for ref_idx in active_ref:
            if ref_idx not in assignment:
                continue
            est_idx = assignment[ref_idx]
            used_est.add(est_idx)
            label = labels[b][ref_idx]
            score = matched_scores[ref_idx]
            margin = _match_margin(scores[b], ref_idx, est_idx)
            energy = _energy_db(est[b, est_idx])
            keep = True
            if score < args.min_match_sdr:
                keep = False
            if margin < args.min_match_margin:
                keep = False
            if energy < args.min_energy_db:
                keep = False
            quality = "clean" if keep else "uncertain"
            if keep or args.save_uncertain:
                filename = f"{soundscape}_{output_slot:02d}_{_safe_label(label)}.wav"
                wav_path = os.path.join(args.output_dir, filename)
                _write_wav(wav_path, est[b, est_idx, 0].numpy(), args.sample_rate)
                output_slot += 1
            manifest_rows.append({
                "soundscape": soundscape,
                "oracle_slot": ref_idx,
                "estimate_slot": est_idx,
                "label": label,
                "match_sa_sdr": score,
                "match_margin": margin,
                "energy_db": energy,
                "quality_group": quality,
                "saved": bool(keep or args.save_uncertain),
            })
        if args.save_unmatched_manifest:
            for est_idx in range(est.shape[1]):
                if est_idx not in used_est:
                    manifest_rows.append({
                        "soundscape": soundscape,
                        "oracle_slot": -1,
                        "estimate_slot": est_idx,
                        "label": "silence",
                        "match_sa_sdr": "",
                        "match_margin": "",
                        "energy_db": _energy_db(est[b, est_idx]),
                        "quality_group": "unmatched",
                        "saved": False,
                    })


def _write_manifest(rows: List[Dict], path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "soundscape",
        "oracle_slot",
        "estimate_slot",
        "label",
        "match_sa_sdr",
        "match_margin",
        "energy_db",
        "quality_group",
        "saved",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Export USS estimated sources with oracle-matched labels for SC training.")
    parser.add_argument("--config", required=True, help="USS model training config containing lightning_module.args.model")
    parser.add_argument("--checkpoint", required=True, help="USS checkpoint path")
    parser.add_argument("--soundscape_dir", required=True, help="Directory of mixture wav files")
    parser.add_argument("--oracle_target_dir", required=True, help="Directory of oracle target wav files with labels in filenames")
    parser.add_argument("--output_dir", required=True, help="Output estimate_target_dir for EstimatedSourceClassifierDataset")
    parser.add_argument("--manifest_path", default=None, help="Optional CSV manifest for match quality")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_sources", type=int, default=3)
    parser.add_argument("--label_set", default="dcase2026t4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_key", default="foreground_waveform", help="Model output key to export")
    parser.add_argument("--min_match_sdr", type=float, default=-10.0, help="Reject matches below this SA-SDR")
    parser.add_argument("--min_match_margin", type=float, default=-1000000000.0, help="Reject ambiguous matches below this margin")
    parser.add_argument("--min_energy_db", type=float, default=-60.0, help="Reject estimates below this RMS dB")
    parser.add_argument("--save_uncertain", action="store_true", help="Also write low-quality matches, but mark them uncertain in manifest")
    parser.add_argument("--save_unmatched_manifest", action="store_true", help="Record unmatched estimate slots in manifest")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = _load_model(args.config, args.checkpoint, device)
    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    rows: List[Dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="export estimated sources for SC"):
            device_batch = _to_device(batch, device)
            output = model({"mixture": device_batch["mixture"]})
            _export_batch(output, batch, args, rows)
    if args.manifest_path:
        _write_manifest(rows, args.manifest_path)
    print(f"Exported estimated sources to: {args.output_dir}")
    if args.manifest_path:
        print(f"Wrote match manifest to: {args.manifest_path}")


if __name__ == "__main__":
    main()
