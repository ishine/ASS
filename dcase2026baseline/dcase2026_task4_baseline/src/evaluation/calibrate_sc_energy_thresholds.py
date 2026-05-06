import argparse
import copy
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.utils import initialize_config, parse_yaml


ENERGY_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    if set(model_state.keys()) != set(state_dict.keys()):
        matched = {}
        for key, value in state_dict.items():
            if key in model_state:
                matched[key] = value
            elif key.startswith("model.") and key[len("model.") :] in model_state:
                matched[key[len("model.") :]] = value
        state_dict = matched
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint did not match model cleanly: missing={list(missing)}, unexpected={list(unexpected)}"
        )


def extract_model_config(config):
    if "lightning_module" in config:
        return copy.deepcopy(config["lightning_module"]["args"]["model"])
    if "model" in config and config["model"].get("main") == "Kwon2025S5":
        return copy.deepcopy(config["model"]["args"]["sc_config"])
    if "model" in config:
        return copy.deepcopy(config["model"])
    raise KeyError("Could not find an SC model config in the YAML file.")


def extract_val_dataset_config(config):
    if "datamodule" not in config:
        raise KeyError("Calibration expects a training label config with datamodule.args.val_dataloader.")
    return copy.deepcopy(config["datamodule"]["args"]["val_dataloader"])


def slot_batches(config, batch_size=None, num_workers=None):
    val_config = extract_val_dataset_config(config)
    if batch_size is not None:
        val_config["batch_size"] = batch_size
    if num_workers is not None:
        val_config["num_workers"] = num_workers
        val_config["persistent_workers"] = bool(num_workers)
    dataset = initialize_config(val_config["dataset"])
    return DataLoader(
        dataset=dataset,
        batch_size=val_config["batch_size"],
        collate_fn=dataset.collate_fn,
        num_workers=val_config["num_workers"],
        pin_memory=True,
        persistent_workers=val_config.get("persistent_workers", False),
        shuffle=False,
    )


def choose_threshold(energies, positives, beta=1.0, max_fpr=None, min_precision=None):
    if not energies:
        return None

    rows = sorted(zip(energies, positives), key=lambda x: x[0])
    total_pos = sum(1 for _, is_pos in rows if is_pos)
    total_neg = len(rows) - total_pos
    beta2 = beta * beta

    candidates = [rows[0][0] - 1e-6] + [energy for energy, _ in rows]
    best = None
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for energy, is_pos in rows:
            accepted = energy <= threshold
            if accepted and is_pos:
                tp += 1
            elif accepted and not is_pos:
                fp += 1
            elif not accepted and is_pos:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        if max_fpr is not None and fpr > max_fpr:
            continue
        if min_precision is not None and precision < min_precision:
            continue
        if precision + recall == 0.0:
            fbeta = 0.0
        else:
            fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall)

        score = (fbeta, -fp, tp, -threshold)
        if best is None or score > best["score"]:
            best = {
                "threshold": threshold,
                "score": score,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "fbeta": fbeta,
                "n_pos": total_pos,
                "n_neg": total_neg,
            }
    return best


def calibrate_thresholds(records, labels, beta=1.0, max_fpr=None, min_precision=None, fallback_default=None):
    per_class = {idx: {"energies": [], "positives": []} for idx in range(len(labels))}
    for record in records:
        pred = record["pred_class"]
        per_class[pred]["energies"].append(record["energy"])
        per_class[pred]["positives"].append(record["target_class"] == pred and not record["is_silence"])

    thresholds = {}
    stats = []
    for idx, label in enumerate(labels):
        best = choose_threshold(
            per_class[idx]["energies"],
            per_class[idx]["positives"],
            beta=beta,
            max_fpr=max_fpr,
            min_precision=min_precision,
        )
        row = {
            "class_index": idx,
            "class_name": label,
            "threshold": fallback_default,
            "n_predicted_as_class": len(per_class[idx]["energies"]),
            "n_pos": 0,
            "n_neg": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": 0.0,
            "fbeta": 0.0,
            "used_fallback": True,
        }
        if best is not None:
            row.update({key: best[key] for key in row.keys() if key in best})
            row["used_fallback"] = False
            thresholds[idx] = float(best["threshold"])
        elif fallback_default is not None:
            thresholds[idx] = float(fallback_default)
        stats.append(row)
    return thresholds, stats


def _is_estimated_source_dataset(dataset):
    return dataset.__class__.__name__ == "EstimatedSourceClassifierDataset" or getattr(dataset, "source_prefix", None) == "est"


def collect_records(model, dataloader, device, max_batches=None, is_estimated_source=None):
    records = []
    model.eval()
    model.to(device)
    if hasattr(model, "energy_thresholds"):
        model.energy_thresholds = {}
    if is_estimated_source is None:
        is_estimated_source = _is_estimated_source_dataset(dataloader.dataset)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            waveform = batch["waveform"].to(device)
            output = model.predict({"waveform": waveform})
            energy = output["energy"].detach().cpu()
            pred_class = output["class_indices"].detach().cpu()
            target_class = batch["class_index"].detach().cpu()
            is_silence = batch["is_silence"].detach().cpu().bool()
            for idx in range(energy.numel()):
                records.append(
                    {
                        "energy": float(energy[idx].item()),
                        "pred_class": int(pred_class[idx].item()),
                        "target_class": int(target_class[idx].item()),
                        "is_silence": bool(is_silence[idx].item()),
                        "is_estimated_source": bool(is_estimated_source),
                    }
                )
    return records


def _quantile_summary(name, records):
    values = [float(record["energy"]) for record in records]
    row = {"bucket": name, "n": len(values)}
    for quantile in ENERGY_QUANTILES:
        row[f"q{int(round(quantile * 100)):02d}"] = None
    row["mean"] = None
    row["std"] = None
    if not values:
        return row

    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor(ENERGY_QUANTILES, dtype=torch.float32))
    for quantile, value in zip(ENERGY_QUANTILES, quantiles.tolist()):
        row[f"q{int(round(quantile * 100)):02d}"] = float(value)
    row["mean"] = float(tensor.mean().item())
    row["std"] = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    return row


def energy_diagnostics(records):
    """Summarize SC logit-energy distributions for calibration debugging."""

    buckets = [
        (
            "true_active_correct_predictions",
            [
                record
                for record in records
                if (not record["is_silence"]) and record["pred_class"] == record["target_class"]
            ],
        ),
        (
            "true_active_wrong_predictions",
            [
                record
                for record in records
                if (not record["is_silence"]) and record["pred_class"] != record["target_class"]
            ],
        ),
        ("true_silence_slots", [record for record in records if record["is_silence"]]),
        ("estimated_source_slots", [record for record in records if record.get("is_estimated_source", False)]),
    ]
    return [_quantile_summary(name, bucket_records) for name, bucket_records in buckets]


def write_outputs(output_dir, thresholds, stats, default_threshold=None, diagnostics=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_block = {"energy_thresholds": {int(k): float(v) for k, v in thresholds.items()}}
    if default_threshold is not None:
        threshold_block["energy_thresholds"]["default"] = float(default_threshold)

    yaml_path = output_dir / "energy_thresholds.yaml"
    json_path = output_dir / "energy_thresholds.json"
    csv_path = output_dir / "energy_threshold_stats.csv"
    diagnostics_csv_path = output_dir / "energy_diagnostics.csv"
    diagnostics_json_path = output_dir / "energy_diagnostics.json"
    yaml_path.write_text(yaml.safe_dump(threshold_block, sort_keys=False))
    json_path.write_text(json.dumps(threshold_block, indent=2) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    diagnostics = diagnostics or []
    if diagnostics:
        with diagnostics_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(diagnostics)
    else:
        diagnostics_csv_path.write_text("")
    diagnostics_json_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return yaml_path, json_path, csv_path, diagnostics_csv_path, diagnostics_json_path


def main():
    parser = argparse.ArgumentParser(description="Calibrate M2D-SC per-class energy thresholds on validation slots.")
    parser.add_argument("-c", "--config", required=True, help="SC training label config YAML.")
    parser.add_argument("--checkpoint", help="Fine-tuned SC checkpoint to load.")
    parser.add_argument("--output-dir", default="workspace/calibration/sc_energy", help="Directory for YAML/CSV outputs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0, help="F-beta objective; beta < 1 favors precision.")
    parser.add_argument("--max-fpr", type=float, help="Optional per-class false-positive-rate constraint.")
    parser.add_argument("--min-precision", type=float, help="Optional per-class precision constraint.")
    parser.add_argument("--default-threshold", type=float, help="Optional default threshold to include in output.")
    parser.add_argument("--max-batches", type=int, help="Debug limit.")
    args = parser.parse_args()

    config = parse_yaml(args.config)
    model_config = extract_model_config(config)
    model = initialize_config(model_config)
    if args.checkpoint:
        load_model_state(model, args.checkpoint)

    dataloader = slot_batches(config, batch_size=args.batch_size, num_workers=args.num_workers)
    records = collect_records(model, dataloader, torch.device(args.device), max_batches=args.max_batches)
    labels = dataloader.dataset.labels
    thresholds, stats = calibrate_thresholds(
        records,
        labels,
        beta=args.beta,
        max_fpr=args.max_fpr,
        min_precision=args.min_precision,
        fallback_default=args.default_threshold,
    )
    diagnostics = energy_diagnostics(records)
    paths = write_outputs(
        Path(args.output_dir),
        thresholds,
        stats,
        default_threshold=args.default_threshold,
        diagnostics=diagnostics,
    )

    print(f"Collected {len(records)} validation source slots.")
    print("Wrote:")
    for path in paths:
        print(f"  {path}")
    print("\nEnergy diagnostics:")
    for row in diagnostics:
        if row["q50"] is None:
            print(f"  {row['bucket']}: n=0")
        else:
            print(
                f"  {row['bucket']}: "
                f"n={row['n']}, q10={row['q10']:.3f}, q50={row['q50']:.3f}, q90={row['q90']:.3f}"
            )
    print("\nPaste this block under the SC model args in an eval config:")
    print(yaml.safe_dump({"energy_thresholds": {int(k): float(v) for k, v in thresholds.items()}}, sort_keys=False))


if __name__ == "__main__":
    main()
