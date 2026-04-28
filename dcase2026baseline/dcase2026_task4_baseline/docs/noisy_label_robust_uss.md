# Noisy / Bled Label Robust USS

This is an opt-in path for handling impure foreground dry stems and noisy class labels in DCASE 2026 Task 4. Existing datasets, losses, models, and configs are unchanged.

## Motivation

Some nominal foreground dry stems may contain additional foreground/interference content. A stem named as `Cough` may also contain `Speech`. In the original USS objective, waveform PIT, class matching, silence supervision, and inactive-slot suppression all trust the declared foreground labels. That can punish a useful extra slot that explains leaked content and can force the model to keep leakage inside the wrong class slot.

The robust path keeps waveform PIT strong but makes semantic supervision quality-aware.

## 1. Build a source quality manifest

If you already have teacher predictions from BEATs/M2D/PaSST/AST, pass them to the audit tool:

```bash
python -m src.tools.audit_source_quality \
  --foreground_dir data/dev_set/sound_event/train \
  --teacher_predictions workspace/manifests/teacher_predictions.jsonl \
  --output workspace/manifests/source_quality_manifest.jsonl \
  --label_set dcase2026t4
```

If teacher predictions are not available yet, the tool can still create a clean default manifest from audio statistics:

```bash
python -m src.tools.audit_source_quality \
  --foreground_dir data/dev_set/sound_event/train \
  --output workspace/manifests/source_quality_manifest.jsonl \
  --label_set dcase2026t4
```

Recommended teacher prediction fields:

```json
{"path": ".../Cough/example.wav", "teacher_probs": {"Cough": 0.70, "Speech": 0.20}}
```

The output manifest stores:

- `quality_group`: `clean`, `uncertain`, or `bad`
- `semantic_confidence`
- `soft_target`
- teacher top-1/top-2/margin fields
- duration/energy/activity statistics

## 2. Train quality-aware USS

```bash
python -m src.train \
  -c config/separation/modified_deft_uss_temporal_noisylabel_min.yaml \
  -w workspace/uss_noisylabel
```

This config uses:

- `src.datamodules.noisy_label_dataset.QualityAwareUSSDataset`
- `src.training.lightningmodule.uss_noisy_label.NoisyLabelUSSLightning`
- `src.training.loss.uss_noisy_label_loss.get_loss_func`

## 3. What changes during training

The wrapper dataset preserves all original `USSDataset` keys and adds:

- `class_confidence`: per-foreground-slot semantic confidence
- `soft_class_target`: per-slot soft class distribution
- `uncertain_slot_mask`: foreground slots with uncertain semantics
- `bad_slot_mask`: foreground slots marked bad/impure
- `quality_resample_attempts`: how often bad generated scenes were resampled

The robust loss keeps waveform PIT unchanged but modifies semantic terms:

- class pair cost is multiplied by `class_confidence`
- optional soft targets replace hard CE for uncertain stems
- after warmup, high semantic-loss active slots are downweighted
- silence supervision is reduced for uncertain/bad foreground slots

## 4. Practical settings

Conservative first run:

```yaml
lambda_class_match: 0.25
uncertain_slot_silence_weight: 0.2
bad_slot_silence_weight: 0.0
semantic_warmup_epochs: 3
semantic_truncation_quantile: 0.8
semantic_truncation_drop_weight: 0.1
```

If USS becomes too semantically weak, increase:

```yaml
lambda_class_match: 0.5
uncertain_class_weight: 0.5
```

If false positives increase, raise silence weight cautiously:

```yaml
uncertain_slot_silence_weight: 0.3
```

## 5. Important sanity checks

```bash
python -m py_compile \
  src/datamodules/noisy_label_dataset.py \
  src/training/lightningmodule/uss_noisy_label.py \
  src/training/loss/uss_noisy_label_loss.py \
  src/tools/audit_source_quality.py
```

Then run one fast development pass before a full training job.

## 6. Evaluation

Compare at least:

1. original temporal USS config
2. noisy-label USS with clean-only/default manifest
3. noisy-label USS with real teacher manifest
4. noisy-label USS + robust SC + quality-aware TSE/bridge path

Track:

- USS-only SDRi / CAPI-SDRi
- zero-target false positives
- source-classifier accuracy on estimated sources
- final S5 CAPI-SDRi
- same-class duplicate recall
