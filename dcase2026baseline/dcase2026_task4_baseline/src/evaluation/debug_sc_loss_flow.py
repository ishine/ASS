import argparse
from typing import Dict, Tuple

import torch

from src.utils import initialize_config, parse_yaml


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _active_loss_terms(output: Dict, target: Dict) -> Dict[str, float]:
    is_silence = target["is_silence"].bool()
    class_index = target["class_index"]
    active = (~is_silence)
    active_count = int(active.sum().item())
    slot_count = int(is_silence.numel())
    silence_count = slot_count - active_count

    metrics = {
        "slot_count": float(slot_count),
        "active_count": float(active_count),
        "silence_count": float(silence_count),
        "active_ratio": float(active_count / max(slot_count, 1)),
        "loss_arcface_active_mean": 0.0,
        "loss_plain_ce_active_mean": 0.0,
        "loss_arcface_active_sum": 0.0,
        "loss_plain_ce_active_sum": 0.0,
    }
    if active_count == 0:
        return metrics

    logits = output["logits"].float()[active]
    plain_logits = output["plain_logits"].float()[active]
    y = class_index[active]
    loss_arcface = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    loss_plain = torch.nn.functional.cross_entropy(plain_logits, y, reduction="none")
    metrics["loss_arcface_active_mean"] = float(loss_arcface.mean().item())
    metrics["loss_plain_ce_active_mean"] = float(loss_plain.mean().item())
    metrics["loss_arcface_active_sum"] = float(loss_arcface.sum().item())
    metrics["loss_plain_ce_active_sum"] = float(loss_plain.sum().item())
    return metrics


def _accumulate(acc: Dict[str, float], values: Dict[str, float], prefix: str, active_weight: int) -> None:
    acc[f"{prefix}/num_batches"] += 1.0
    acc[f"{prefix}/slot_count"] += values["slot_count"]
    acc[f"{prefix}/active_count"] += values["active_count"]
    acc[f"{prefix}/silence_count"] += values["silence_count"]
    acc[f"{prefix}/loss_arcface_active_sum"] += values["loss_arcface_active_sum"]
    acc[f"{prefix}/loss_plain_ce_active_sum"] += values["loss_plain_ce_active_sum"]
    acc[f"{prefix}/batch_avg_loss_arcface"] += values["loss_arcface_active_mean"]
    acc[f"{prefix}/batch_avg_loss_plain_ce"] += values["loss_plain_ce_active_mean"]
    acc[f"{prefix}/batch_avg_active_ratio"] += values["active_ratio"]
    # "slot-weighted" approximation of lightning's default epoch average for a metric
    # that is already mean-reduced inside each batch.
    acc[f"{prefix}/slot_weighted_loss_arcface_like_logger"] += values["loss_arcface_active_mean"] * active_weight
    acc[f"{prefix}/slot_weighted_loss_plain_ce_like_logger"] += values["loss_plain_ce_active_mean"] * active_weight
    acc[f"{prefix}/slot_weight_total"] += float(active_weight)


def _finalize(acc: Dict[str, float], prefix: str) -> Dict[str, float]:
    batches = max(acc[f"{prefix}/num_batches"], 1.0)
    active = max(acc[f"{prefix}/active_count"], 1.0)
    slots = max(acc[f"{prefix}/slot_count"], 1.0)
    slot_weight_total = max(acc[f"{prefix}/slot_weight_total"], 1.0)
    return {
        "num_batches": acc[f"{prefix}/num_batches"],
        "slot_count": acc[f"{prefix}/slot_count"],
        "active_count": acc[f"{prefix}/active_count"],
        "silence_count": acc[f"{prefix}/silence_count"],
        "active_ratio_global": acc[f"{prefix}/active_count"] / slots,
        "loss_arcface_active_global_mean": acc[f"{prefix}/loss_arcface_active_sum"] / active,
        "loss_plain_ce_active_global_mean": acc[f"{prefix}/loss_plain_ce_active_sum"] / active,
        "loss_arcface_batch_mean": acc[f"{prefix}/batch_avg_loss_arcface"] / batches,
        "loss_plain_ce_batch_mean": acc[f"{prefix}/batch_avg_loss_plain_ce"] / batches,
        "active_ratio_batch_mean": acc[f"{prefix}/batch_avg_active_ratio"] / batches,
        "loss_arcface_like_logger": acc[f"{prefix}/slot_weighted_loss_arcface_like_logger"] / slot_weight_total,
        "loss_plain_ce_like_logger": acc[f"{prefix}/slot_weighted_loss_plain_ce_like_logger"] / slot_weight_total,
    }


def _analyze_loader(
    model,
    loader,
    max_batches: int,
    device: torch.device,
    split_name: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Compare model.train() vs model.eval() on the exact same batches.
    train_mode_acc = {
        f"{split_name}/num_batches": 0.0,
        f"{split_name}/slot_count": 0.0,
        f"{split_name}/active_count": 0.0,
        f"{split_name}/silence_count": 0.0,
        f"{split_name}/loss_arcface_active_sum": 0.0,
        f"{split_name}/loss_plain_ce_active_sum": 0.0,
        f"{split_name}/batch_avg_loss_arcface": 0.0,
        f"{split_name}/batch_avg_loss_plain_ce": 0.0,
        f"{split_name}/batch_avg_active_ratio": 0.0,
        f"{split_name}/slot_weighted_loss_arcface_like_logger": 0.0,
        f"{split_name}/slot_weighted_loss_plain_ce_like_logger": 0.0,
        f"{split_name}/slot_weight_total": 0.0,
    }
    eval_mode_acc = dict(train_mode_acc)
    eval_mode_acc = {k.replace(f"{split_name}/", f"{split_name}_evalmode/"): v for k, v in eval_mode_acc.items()}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = _to_device(batch, device)
        input_dict = {
            "waveform": batch["waveform"],
            "class_index": batch["class_index"],
        }
        target = {
            "class_index": batch["class_index"],
            "is_silence": batch["is_silence"],
        }
        active_weight = int((~batch["is_silence"].bool()).sum().item())

        model.train()
        with torch.no_grad():
            out_trainmode = model(input_dict)
        metrics_trainmode = _active_loss_terms(out_trainmode, target)
        _accumulate(train_mode_acc, metrics_trainmode, split_name, active_weight)

        model.eval()
        with torch.no_grad():
            out_evalmode = model(input_dict)
        metrics_evalmode = _active_loss_terms(out_evalmode, target)
        _accumulate(eval_mode_acc, metrics_evalmode, f"{split_name}_evalmode", active_weight)

    train_final = _finalize(train_mode_acc, split_name)
    eval_final = _finalize(eval_mode_acc, f"{split_name}_evalmode")
    return train_final, eval_final


def main():
    parser = argparse.ArgumentParser(description="Debug SC loss gap between train and validation flows.")
    parser.add_argument("-c", "--config", required=True, help="Training config yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional lightning checkpoint to load.")
    parser.add_argument(
        "--m2d-weight-file",
        default=None,
        help="Optional override for model.args.weight_file when config path is unavailable locally.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override dataloader num_workers for debug stability.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for train/val batch size.",
    )
    args = parser.parse_args()

    cfg = parse_yaml(args.config)
    if args.m2d_weight_file:
        cfg["lightning_module"]["args"]["model"]["args"]["weight_file"] = args.m2d_weight_file
    cfg["datamodule"]["args"]["train_dataloader"]["num_workers"] = int(args.num_workers)
    cfg["datamodule"]["args"]["train_dataloader"]["persistent_workers"] = False
    cfg["datamodule"]["args"]["val_dataloader"]["num_workers"] = int(args.num_workers)
    cfg["datamodule"]["args"]["val_dataloader"]["persistent_workers"] = False
    if args.batch_size is not None:
        cfg["datamodule"]["args"]["train_dataloader"]["batch_size"] = int(args.batch_size)
        cfg["datamodule"]["args"]["val_dataloader"]["batch_size"] = int(args.batch_size)
    datamodule = initialize_config(cfg["datamodule"])
    lightning = initialize_config(cfg["lightning_module"])
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        lightning.load_state_dict(state_dict, strict=False)
    model = lightning.model

    device = torch.device(args.device)
    model.to(device)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("=== Tracing train loader ===", flush=True)
    train_as_trainmode, train_as_evalmode = _analyze_loader(
        model=model,
        loader=train_loader,
        max_batches=args.max_batches,
        device=device,
        split_name="train_loader",
    )
    print(train_as_trainmode, flush=True)
    print(train_as_evalmode, flush=True)

    print("=== Tracing val loader ===", flush=True)
    val_as_trainmode, val_as_evalmode = _analyze_loader(
        model=model,
        loader=val_loader,
        max_batches=args.max_batches,
        device=device,
        split_name="val_loader",
    )
    print(val_as_trainmode, flush=True)
    print(val_as_evalmode, flush=True)

    print("=== Key comparisons ===", flush=True)
    print(
        {
            "train_loader_evalmode/loss_arcface_active_global_mean": train_as_evalmode["loss_arcface_active_global_mean"],
            "val_loader_evalmode/loss_arcface_active_global_mean": val_as_evalmode["loss_arcface_active_global_mean"],
            "train_loader_evalmode/active_ratio_global": train_as_evalmode["active_ratio_global"],
            "val_loader_evalmode/active_ratio_global": val_as_evalmode["active_ratio_global"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
