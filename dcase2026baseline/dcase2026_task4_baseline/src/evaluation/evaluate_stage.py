import argparse
import copy
import json
import os

import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import initialize_config
from .metrics import label_metric, s5capi_metric


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _resolve_existing_path(path, config_dir):
    if path is None:
        return None
    if os.path.exists(path):
        return path
    candidate = os.path.join(config_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path


def _load_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    if set(state_dict.keys()) != set(model_state.keys()):
        one_model_key = next(iter(model_state.keys()))
        suffix_matches = [k for k in state_dict if isinstance(k, str) and k.endswith(one_model_key)]
        if suffix_matches:
            prefix = suffix_matches[0][:-len(one_model_key)]
            state_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if isinstance(k, str) and k.startswith(prefix)
            }
        else:
            for prefix in ("model.", "module.", "net."):
                stripped = {
                    k[len(prefix):]: v
                    for k, v in state_dict.items()
                    if isinstance(k, str) and k.startswith(prefix)
                }
                if stripped and any(k in model_state for k in stripped):
                    state_dict = stripped
                    break
    model.load_state_dict(state_dict)


def _stage_model_config(config, stage):
    if "model" in config and f"{stage}_config" in config["model"].get("args", {}):
        args = config["model"]["args"]
        return copy.deepcopy(args[f"{stage}_config"]), args.get(f"{stage}_ckpt")
    if "lightning_module" in config:
        return copy.deepcopy(config["lightning_module"]["args"]["model"]), None
    if "model" in config:
        return copy.deepcopy(config["model"]), None
    raise KeyError("Could not find a model config")


def _dataset_config(config, split):
    if "dataset" in config:
        return copy.deepcopy(config["dataset"])
    if "datamodule" in config:
        dm_args = config["datamodule"]["args"]
        key = f"{split}_dataloader"
        if key not in dm_args:
            raise KeyError(f"Training config has no datamodule.args.{key}")
        return copy.deepcopy(dm_args[key]["dataset"])
    raise KeyError("Could not find a dataset config")


def _label_from_vectors(label_vectors, labels):
    batch_labels = []
    for sample_vectors in label_vectors.detach().cpu():
        sample_labels = []
        for vector in sample_vectors:
            if float(vector.abs().sum()) == 0.0:
                sample_labels.append("silence")
            else:
                sample_labels.append(labels[int(torch.argmax(vector).item())])
        batch_labels.append(sample_labels)
    return batch_labels


def _uss_labels(output, labels):
    class_probs = torch.softmax(output["class_logits"], dim=-1)
    probabilities, indices = class_probs.max(dim=-1)
    active_logits = output.get("silence_logits")
    if active_logits is None:
        active = torch.ones_like(indices, dtype=torch.bool)
        active_probs = torch.ones_like(probabilities)
    else:
        active_probs = torch.sigmoid(active_logits)
        active = active_logits > 0.0

    batch_labels = []
    for sample_indices, sample_active in zip(indices.detach().cpu(), active.detach().cpu()):
        sample_labels = []
        for idx, is_active in zip(sample_indices.tolist(), sample_active.tolist()):
            sample_labels.append(labels[int(idx)] if is_active else "silence")
        batch_labels.append(sample_labels)
    return batch_labels, probabilities * active_probs


def _classification_counts(est_labels, ref_labels):
    total = 0
    correct = 0
    active_total = 0
    active_correct = 0
    silence_total = 0
    silence_correct = 0
    tp = 0
    fp = 0
    fn = 0
    for sample_est, sample_ref in zip(est_labels, ref_labels):
        for est, ref in zip(sample_est, sample_ref):
            total += 1
            correct += int(est == ref)
            if ref == "silence":
                silence_total += 1
                silence_correct += int(est == ref)
            else:
                active_total += 1
                active_correct += int(est == ref)
            if est == ref and ref != "silence":
                tp += 1
            else:
                fp += int(est != "silence")
                fn += int(ref != "silence")
    precision = 100.0 * tp / max(tp + fp, 1)
    recall = 100.0 * tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "source_top1_accuracy": 100.0 * correct / max(total, 1),
        "active_source_top1_accuracy": 100.0 * active_correct / max(active_total, 1),
        "silence_accuracy": 100.0 * silence_correct / max(silence_total, 1),
        "source_precision": precision,
        "source_recall": recall,
        "source_f1": f1,
        "num_sources": total,
        "num_active_sources": active_total,
        "num_silence_sources": silence_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _compute_metric(metric_func, is_print=True):
    summary = metric_func.compute(is_print=is_print)
    if summary is not None:
        return summary
    values = [v for v in getattr(metric_func, "metric_values", []) if v is not None]
    if values:
        return {"mean": sum(values) / len(values)}
    return None


class StageEvaluator:
    def __init__(
        self,
        config_path,
        stage,
        checkpoint=None,
        split="val",
        waveform_output_dir="",
        result_dir="",
        batch_size=2,
        use_cpu=False,
        num_workers=None,
    ):
        self.config_path = config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.config = _load_yaml(config_path)
        self.stage = stage
        self.filename = os.path.basename(config_path)[:-5]
        self.batch_size = batch_size
        self.result_dir = result_dir
        self.waveform_output_dir = (
            os.path.join(waveform_output_dir, self.filename, stage)
            if waveform_output_dir
            else waveform_output_dir
        )
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

        if self.waveform_output_dir:
            os.makedirs(self.waveform_output_dir, exist_ok=True)

        dataset_config = _dataset_config(self.config, split)
        self.dataset = initialize_config(dataset_config, reload=True)
        self.labels = getattr(self.dataset, "labels", None)
        if self.labels is None and hasattr(self.dataset, "base_dataset"):
            self.labels = self.dataset.base_dataset.labels
        self.sr = getattr(self.dataset, "sr", None)
        if self.sr is None and hasattr(self.dataset, "base_dataset"):
            self.sr = self.dataset.base_dataset.sr

        workers = batch_size * 2 if num_workers is None else num_workers
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=getattr(self.dataset, "collate_fn", None),
            num_workers=workers,
        )

        model_config, config_checkpoint = _stage_model_config(self.config, stage)
        checkpoint = checkpoint or config_checkpoint
        checkpoint = _resolve_existing_path(checkpoint, self.config_dir)
        self.model = initialize_config(model_config, reload=True)
        _load_checkpoint(self.model, checkpoint)
        self.model.eval()
        self.model.to(self.device)

    def _write_waveforms(self, batch, labels, waveforms):
        if not self.waveform_output_dir:
            return
        for sample_labels, sample_waveforms, soundscape_name in zip(labels, waveforms, batch["soundscape"]):
            for source_idx, (label, waveform) in enumerate(zip(sample_labels, sample_waveforms)):
                if label == "silence":
                    continue
                wavpath = os.path.join(
                    self.waveform_output_dir,
                    f"{soundscape_name}_{source_idx}_{label}.wav",
                )
                sf.write(wavpath, waveform.detach().cpu().numpy(), self.sr)

    def _evaluate_uss_batch(self, batch):
        mixture = batch["mixture"].to(self.device)
        with torch.no_grad():
            output = self.model({"mixture": mixture})
        est_waveforms = output["foreground_waveform"][:, :, 0, :].detach().cpu()
        est_labels, probabilities = _uss_labels(output, self.labels)
        return est_labels, est_waveforms, probabilities.detach().cpu(), batch["label"]

    def _evaluate_sc_batch(self, batch):
        if "dry_sources" in batch:
            ref_waveforms = batch["dry_sources"][:, :, 0, :]
            batch_size, n_sources, samples = ref_waveforms.shape
            flat = ref_waveforms.reshape(batch_size * n_sources, samples)
            ref_labels = batch["label"]
        else:
            flat = batch["waveform"]
            if flat.dim() == 3 and flat.shape[1] == 1:
                flat = flat[:, 0]
            batch_size, n_sources = flat.shape[0], 1
            ref_labels = []
            for class_index, is_silence in zip(batch["class_index"].tolist(), batch["is_silence"].tolist()):
                ref_labels.append(["silence" if is_silence else self.labels[int(class_index)]])
        flat = flat.to(self.device)
        with torch.no_grad():
            output = self.model.predict({"waveform": flat})
        label_vector = output["label_vector"].view(batch_size, n_sources, -1)
        probabilities = output["probabilities"].view(batch_size, n_sources)
        est_labels = _label_from_vectors(label_vector, self.labels)
        return est_labels, None, probabilities.detach().cpu(), ref_labels

    def _evaluate_tse_batch(self, batch):
        mixture = batch["mixture"].to(self.device)
        enrollment = batch["enrollment"] if "enrollment" in batch else batch["dry_sources"]
        enrollment = enrollment.to(self.device)
        label_vector = batch["label_vector"].to(self.device)
        label_dim = getattr(self.model, "label_dim", None)
        if label_dim is not None and label_vector.shape[-1] > label_dim:
            label_vector = label_vector[..., :label_dim]
        with torch.no_grad():
            output = self.model({
                "mixture": mixture,
                "enrollment": enrollment,
                "label_vector": label_vector,
            })
        est_waveforms = output["waveform"][:, :, 0, :].detach().cpu()
        probabilities = torch.ones(est_waveforms.shape[:2], dtype=torch.float32)
        return batch["label"], est_waveforms, probabilities, batch["label"]

    def evaluate(self):
        results = []
        label_metrics = label_metric.LabelMetric()
        separation_metric = s5capi_metric.S5ClassAwareMetric(metricfunc="sdr")
        metric_funcs = [label_metrics] if self.stage == "sc" else [separation_metric, label_metrics]
        for metric_func in metric_funcs:
            metric_func.reset()

        all_est_labels = []
        all_ref_labels = []
        for batch in tqdm(self.dataloader):
            if self.stage == "uss":
                est_labels, est_waveforms, probabilities, ref_labels = self._evaluate_uss_batch(batch)
            elif self.stage == "sc":
                est_labels, est_waveforms, probabilities, ref_labels = self._evaluate_sc_batch(batch)
            elif self.stage == "tse":
                est_labels, est_waveforms, probabilities, ref_labels = self._evaluate_tse_batch(batch)
            else:
                raise ValueError(f"Unsupported stage: {self.stage}")

            all_est_labels.extend(est_labels)
            all_ref_labels.extend(ref_labels)
            ref_waveforms = batch.get("dry_sources")
            mixture = batch.get("mixture")

            metric_values = []
            for metric_func in metric_funcs:
                kwargs = {
                    "batch_est_labels": est_labels,
                    "batch_ref_labels": ref_labels,
                }
                if metric_func is separation_metric:
                    kwargs.update({
                        "batch_est_waveforms": est_waveforms,
                        "batch_ref_waveforms": ref_waveforms[:, :, 0, :],
                        "batch_mixture": mixture[:, 0, :],
                    })
                metric_values.append(metric_func.update(**kwargs))

            if est_waveforms is not None:
                self._write_waveforms(batch, est_labels, est_waveforms)

            if self.result_dir:
                for i, soundscape in enumerate(batch.get("soundscape", [None] * len(est_labels))):
                    reobj = {
                        "soundscape": soundscape,
                        "stage": self.stage,
                        "ref_labels": ref_labels[i],
                        "est_labels": est_labels[i],
                        "probabilities": _as_list(probabilities[i]),
                        "metrics": [],
                    }
                    for mval, mfunc in zip(metric_values, metric_funcs):
                        reobj["metrics"].append({
                            "metric": getattr(mfunc, "metric_name", None),
                            "value": mval[i],
                        })
                    results.append(reobj)

        summary = {}
        for metric_func in metric_funcs:
            summary[getattr(metric_func, "metric_name", metric_func.__class__.__name__)] = _compute_metric(
                metric_func,
                is_print=True,
            )
        if self.stage == "sc":
            summary["source_classification"] = _classification_counts(all_est_labels, all_ref_labels)
            print("Source top-1 accuracy: %.3f" % summary["source_classification"]["source_top1_accuracy"])
            print("Active source top-1 accuracy: %.3f" % summary["source_classification"]["active_source_top1_accuracy"])
            print("Silence accuracy: %.3f" % summary["source_classification"]["silence_accuracy"])
            print("Source F1: %.3f" % summary["source_classification"]["source_f1"])

        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)
            stem = f"{self.filename}_{self.stage}"
            with open(os.path.join(self.result_dir, f"{stem}_results.json"), "w") as outfile:
                json.dump(results, outfile, indent=4)
            with open(os.path.join(self.result_dir, f"{stem}_summary.json"), "w") as outfile:
                json.dump(summary, outfile, indent=4)


def main(args):
    evaluator = StageEvaluator(
        args.config,
        args.stage,
        checkpoint=args.checkpoint,
        split=args.split,
        waveform_output_dir=args.waveform_output_dir,
        result_dir=args.result_dir,
        batch_size=args.batchsize,
        use_cpu=args.cpu,
        num_workers=args.num_workers,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--stage", choices=["uss", "sc", "tse"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--waveform_output_dir", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batchsize", "-b", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    print("START")
    main(args)
