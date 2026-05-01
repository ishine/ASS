# USS -> TSE Semantic-Acoustic Bridge

This is an opt-in path. Existing USS/TSE configs and model classes remain unchanged.

## 1. Train bridge-aware USS

```bash
python -m src.train \
  -c config/separation/modified_deft_uss_temporal_bridge_min.yaml \
  -w workspace/uss_bridge
```

This uses:

- `src.datamodules.uss_spatial_dataset.USSSpatialDataset`
- `src.models.deft.modified_deft_semantic_bridge.SemanticBridgeModifiedDeFTUSSSpatialTemporal`
- `src.training.lightningmodule.uss_bridge.USSBridgeLightning`
- `src.training.loss.uss_bridge_loss.get_loss_func`

The USS bridge model emits extra proposal keys:

- `foreground_embedding`
- `foreground_audio_embedding`
- `prototype_logits`
- `pred_doa_vector`
- `used_spatial_vector`
- `tse_condition`

## 2. Export bridge features

```bash
python -m src.tools.export_uss_bridge_features \
  --config config/separation/modified_deft_uss_temporal_bridge_min.yaml \
  --checkpoint workspace/uss_bridge/checkpoints/last.ckpt \
  --soundscape_dir workspace/sc_finetune/soundscape \
  --output_dir workspace/sc_finetune/uss_bridge_features \
  --batch_size 4
```

The exporter writes one file per soundscape:

```text
workspace/sc_finetune/uss_bridge_features/<soundscape>.pt
```

Each file can contain `tse_condition`, embeddings, DoA vectors, and logits.

## 3. Fine-tune TSE with bridge features

```bash
python -m src.train \
  -c config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment_bridge_min.yaml \
  -w workspace/tse_bridge
```

This uses:

- `src.datamodules.tse_bridge_dataset.BridgeEstimatedEnrollmentTSEDataset`
- `src.training.lightningmodule.tse_bridge.TSEBridgeLightning`
- `src.models.deft.modified_deft_tse_bridge.BridgeModifiedDeFTTSEMemoryEfficientTemporal`

The TSE wrapper preserves the old input contract. If `bridge_condition` is absent, it behaves like the base TSE model. If present, it projects the bridge feature into an additive residual over the original `label_vector` condition.

## Recommended schedule

For USS bridge training:

- start with `predicted_spatial_prob: 0.0` for a stable warmup if training is unstable;
- use `0.3` after warmup;
- increase to `0.5` only after DoA and class losses are stable.

For TSE bridge fine-tuning:

- start with `bridge_label_scale: 0.3` to avoid overpowering the class condition;
- use `0.5` once the exported USS features are reliable;
- keep `pretrained_model_strict: false` because the bridge wrapper adds new projection weights.
