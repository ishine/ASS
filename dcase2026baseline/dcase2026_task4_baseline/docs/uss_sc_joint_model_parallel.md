# USS + SC Model-Parallel Joint Fine-tuning

This document explains the opt-in joint training path for the USS separator and the SC source-classifier.

The goal is to let a strong SC model provide semantic feedback to USS separated foreground waveforms while avoiding the memory cost of putting both large models on one GPU.

This path is **opt-in**. It does not modify or replace the existing USS configs, SC configs, USS model classes, SC model classes, or standalone training workflows.

---

## 1. Problem this solves

The current USS model can improve SA-SDR/SDR while its internal class head remains weak. This happens because waveform separation receives dense gradients, while classification is a smaller auxiliary head and is affected by PIT assignment, noisy labels, and bled dry stems.

The SC model can classify better, but if it is trained only on oracle clean source audio, it often performs badly on USS separated audio. USS separated slots contain artifacts, leakage, missing parts, residual interference, and sometimes partial events.

The joint path solves the two issues together:

```text
USS produces foreground estimates
    -> SC classifies those estimates
    -> SC semantic loss flows back into USS
```

This asks the separator to produce waveforms that are not only high-SDR but also semantically recognizable by a stronger source classifier.

---

## 2. High-level architecture

Model placement:

```text
USS model -> cuda:0
SC model  -> cuda:1
```

Forward pass:

```text
mixture on cuda:0
    -> USS
    -> foreground_waveform on cuda:0
    -> differentiable copy to cuda:1
    -> SC
    -> SC classification loss on cuda:1
    -> gradient flows back through copy
    -> USS updated on cuda:0
```

PyTorch autograd supports gradients through a tensor copy such as:

```python
sep_for_sc = sep.to("cuda:1")
```

So the SC loss can update USS even though USS and SC live on different GPUs.

---

## 3. Why this is not DDP

Do **not** use ordinary DDP for this mode.

DDP would replicate the full LightningModule on every GPU:

```text
GPU0: USS + SC
GPU1: USS + SC
```

That does not reduce memory. It makes the memory problem worse.

This joint path uses **model parallelism**:

```text
GPU0: USS only
GPU1: SC only
```

Therefore the config intentionally uses:

```yaml
trainer:
  accelerator: gpu
  devices: 1
  strategy: auto
```

Even though the trainer uses `devices: 1`, the module manually uses both `cuda:0` and `cuda:1` internally.

Launch with two visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
  -c config/separation/modified_deft_uss_sc_joint_model_parallel_min.yaml \
  -w workspace/uss_sc_joint
```

Inside the process, these are seen as:

```text
cuda:0 -> first visible GPU
cuda:1 -> second visible GPU
```

---

## 4. Added files

Main Lightning module:

```text
src/training/lightningmodule/uss_sc_joint_model_parallel.py
```

Main config:

```text
config/separation/modified_deft_uss_sc_joint_model_parallel_min.yaml
```

Related estimated-source export utility:

```text
src/tools/export_estimated_sources_for_sc.py
```

Related matching utility:

```text
src/tools/estimated_source_matching.py
```

---

## 5. Recommended full workflow

Do not train USS and SC jointly from scratch. The stable recipe is staged training.

### Stage 1: train USS alone

Train the USS model with your preferred config. For noisy or bled-label data, use the robust USS config:

```bash
python -m src.train \
  -c config/separation/modified_deft_uss_temporal_noisylabel_min.yaml \
  -w workspace/uss_noisylabel
```

Expected checkpoint:

```text
workspace/uss_noisylabel/checkpoints/last.ckpt
```

### Stage 2: export USS estimated sources for SC training

Use oracle waveform matching to label the estimated sources. Do **not** use the weak USS class head labels.

```bash
python -m src.tools.export_estimated_sources_for_sc \
  --config config/separation/modified_deft_uss_temporal_noisylabel_min.yaml \
  --checkpoint workspace/uss_noisylabel/checkpoints/last.ckpt \
  --soundscape_dir workspace/sc_finetune/soundscape \
  --oracle_target_dir workspace/sc_finetune/oracle_target \
  --output_dir workspace/sc_finetune/estimate_target \
  --manifest_path workspace/sc_finetune/estimate_target_manifest.csv \
  --match_metric sa_sdr \
  --min_match_score -10.0 \
  --min_energy_db -60.0 \
  --batch_size 4
```

This writes estimated source files like:

```text
workspace/sc_finetune/estimate_target/soundscape_00000001_00_Cough.wav
workspace/sc_finetune/estimate_target/soundscape_00000001_01_Doorbell.wav
```

The label in the filename is the oracle-matched label, not the USS prediction.

### Stage 3: train SC on estimated-source data

Use the existing estimated-source SC config, or your stronger SC config:

```bash
python -m src.train \
  -c config/label/m2d_sc_stage3_estimated_temporal_strong_robust.yaml \
  -w workspace/sc_stage3
```

Expected checkpoint:

```text
workspace/sc_stage3/checkpoints/last.ckpt
```

Important: for the final SC model, validation should also include estimated-source validation data, not only oracle dry stems. Otherwise the validation metric may look good but fail on USS outputs.

### Stage 4: joint fine-tune with frozen SC teacher

Update the checkpoint paths in:

```text
config/separation/modified_deft_uss_sc_joint_model_parallel_min.yaml
```

Default fields:

```yaml
uss_pretrained_ckpt: workspace/uss_noisylabel/checkpoints/last.ckpt
sc_pretrained_ckpt: workspace/sc_stage3/checkpoints/last.ckpt
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
  -c config/separation/modified_deft_uss_sc_joint_model_parallel_min.yaml \
  -w workspace/uss_sc_joint
```

This is the recommended first joint experiment.

### Stage 5: optional full joint fine-tuning

Only after frozen-SC training is stable, try:

```yaml
freeze_sc: false
sc_update_every: 4
```

This updates USS every step and SC once every four steps.

---

## 6. Main config fields

### Device placement

```yaml
uss_device: cuda:0
sc_device: cuda:1
```

USS is placed on `cuda:0`; SC is placed on `cuda:1`.

### Frozen SC teacher mode

```yaml
freeze_sc: true
sc_eval_mode_when_frozen: true
```

This freezes SC parameters and puts SC in eval mode. However, gradients can still flow through SC to its input waveform and then back to USS.

This is different from `torch.no_grad()`. Do not wrap SC forward with `torch.no_grad()` during training if you want SC loss to update USS.

### SC loss weight

```yaml
lambda_sc: 0.05
```

This controls how strongly SC classification loss affects USS. Start small.

Recommended values:

```text
safe:       0.01
normal:     0.05
aggressive: 0.10
```

If separation SDR drops quickly, reduce this value.

### Consistency loss

```yaml
lambda_consistency: 0.02
```

This encourages USS internal class logits to agree with SC logits.

Set to zero if you only want the SC waveform classification loss:

```yaml
lambda_consistency: 0.0
```

### Detach waveform switch

```yaml
detach_waveform_for_sc: false
```

Keep this false for joint USS semantic fine-tuning.

If set to true, the SC loss will not update USS. This can be useful only for debugging SC diagnostics.

### Matching metric

```yaml
match_metric: sa_sdr
```

Supported values:

```text
sa_sdr
si_sdr
```

This is used inside the joint module to assign oracle labels to predicted USS foreground slots.

### Match filtering

```yaml
min_match_score: -10.0
min_energy_db: -60.0
```

A predicted slot is used for SC loss only if it has sufficient match quality and energy. Low-quality or low-energy slots are treated as silence/unmatched for the SC loss.

### Full joint update frequency

```yaml
sc_update_every: 4
```

Only used when:

```yaml
freeze_sc: false
```

This means SC is updated less frequently than USS. It helps avoid SC overfitting to transient USS artifacts.

---

## 7. Label assignment inside joint training

SC targets are not taken from the USS class head.

The module uses oracle waveform matching:

```text
USS predicted foreground slots
    -> pairwise SA-SDR/SI-SDR with oracle foreground sources
    -> PIT assignment
    -> oracle class label assigned to predicted slot
```

This prevents weak USS class-head predictions from contaminating SC training.

Example:

```text
oracle target 0: Cough
oracle target 1: Doorbell

predicted slot 0 best matches oracle target 1 -> label Doorbell
predicted slot 1 best matches oracle target 0 -> label Cough
```

The SC receives the predicted slot waveform and the matched oracle label.

---

## 8. Losses used during joint training

The total loss is:

```text
loss = USS loss
     + lambda_sc * SC weighted CE on USS separated waveforms
     + lambda_consistency * USS/SC logit consistency
```

### USS loss

Configured under:

```yaml
uss_loss:
  module: src.training.loss.uss_noisy_label_loss
  main: get_loss_func
```

This can be replaced with the normal USS loss if desired.

### SC loss

Configured under:

```yaml
sc_loss:
  module: src.training.loss.m2d_sc_arcface
  main: get_loss_func
```

The configured SC loss is computed for diagnostics. The joint module also computes an explicit weighted CE term named:

```text
loss_sc_weighted
```

This explicit weighted CE is the term used for the SC-to-USS gradient.

### Consistency loss

If `lambda_consistency > 0`, the module computes KL consistency between:

```text
USS class_logits
SC plain_logits/logits
```

SC logits are used as the teacher by default.

---

## 9. Expected log keys

During training, monitor:

```text
step_train/loss
step_train/uss_loss
step_train/uss_loss_fg_wave
step_train/uss_loss_ce
step_train/loss_sc_weighted
step_train/sc_joint_top1
step_train/sc_active_weight_mean
step_train/loss_consistency
```

During validation, monitor:

```text
step_val/loss
step_val/uss_loss_fg_wave
step_val/loss_sc_weighted
step_val/sc_joint_top1
```

Interpretation:

```text
loss_sc_weighted finite:
    SC forward/loss works on USS outputs

sc_joint_top1 rising:
    USS separated slots are becoming more semantically recognizable

uss_loss_fg_wave stable or improving:
    semantic feedback is not destroying separation

sc_active_weight_mean near zero:
    matching/filtering is too strict, or USS outputs are too poor
```

---

## 10. Sanity checks

### Compile check

```bash
python -m py_compile \
  src/training/lightningmodule/uss_sc_joint_model_parallel.py \
  src/tools/estimated_source_matching.py \
  src/tools/export_estimated_sources_for_sc.py
```

### GPU visibility check

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

You need at least two visible GPUs.

### Tiny run

Before a full training job, set:

```yaml
batch_size: 1
lambda_sc: 0.01
lambda_consistency: 0.0
```

Then run one short job and verify the logs are finite.

---

## 11. Common problems and fixes

### Problem: `cuda:1` is unavailable

Cause: only one GPU is visible to the process.

Fix:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.train ...
```

Then keep:

```yaml
uss_device: cuda:0
sc_device: cuda:1
```

### Problem: DDP starts and both GPUs run out of memory

Cause: trainer strategy/devices were changed to DDP.

Fix:

```yaml
trainer:
  accelerator: gpu
  devices: 1
  strategy: auto
```

### Problem: SC loss does not affect USS

Check:

```yaml
detach_waveform_for_sc: false
```

Also ensure the SC forward during training is not wrapped in `torch.no_grad()`.

### Problem: separation SDR drops rapidly

Reduce semantic pressure:

```yaml
lambda_sc: 0.01
lambda_consistency: 0.0
```

Also use frozen SC first:

```yaml
freeze_sc: true
```

### Problem: `sc_active_weight_mean` is nearly zero

This means few predicted slots pass the matching filters.

Loosen:

```yaml
min_match_score: -20.0
min_energy_db: -70.0
```

or improve USS pretraining before joint fine-tuning.

### Problem: SC top-1 is poor but USS waveform loss is good

This may mean SC has not been adapted to USS estimated audio. Train or fine-tune SC on exported estimated sources first:

```bash
python -m src.tools.export_estimated_sources_for_sc ...
python -m src.train -c config/label/m2d_sc_stage3_estimated_temporal_strong_robust.yaml ...
```

### Problem: full joint mode is unstable

Return to frozen-SC mode:

```yaml
freeze_sc: true
```

Full joint mode should only be used after frozen-SC training is stable.

---

## 12. Recommended ablations

Run these in order:

```text
A. USS baseline alone
B. USS noisy-label robust alone
C. SC trained on oracle only
D. SC trained on exported estimated sources
E. USS + frozen SC joint fine-tuning, lambda_sc=0.01
F. USS + frozen SC joint fine-tuning, lambda_sc=0.05
G. USS + frozen SC + consistency, lambda_consistency=0.02
H. Optional full joint mode, freeze_sc=false, sc_update_every=4
```

Track:

```text
USS validation SDRi / CAPI-SDRi
zero-target false positive rate
same-class duplicate recall
SC top-1 on estimated sources
final S5 CAPI-SDRi
```

The desired outcome is:

```text
SC top-1 on USS outputs improves
USS SDR/CAPI-SDRi does not drop
zero-target false positives do not increase
final S5 CAPI-SDRi improves
```

---

## 13. Practical starting values

Safe first run:

```yaml
freeze_sc: true
lambda_sc: 0.01
lambda_consistency: 0.0
min_match_score: -20.0
min_energy_db: -70.0
```

Normal run:

```yaml
freeze_sc: true
lambda_sc: 0.05
lambda_consistency: 0.02
min_match_score: -10.0
min_energy_db: -60.0
```

Aggressive run:

```yaml
freeze_sc: false
sc_update_every: 4
lambda_sc: 0.05
lambda_consistency: 0.02
```

Use the safe run first.

---

## 14. Final recommendation

For this repo and task, the most reliable path is:

```text
Train USS alone
Train SC on oracle + estimated-source audio
Joint fine-tune with frozen SC on a second GPU
Only then try full joint updates
```

This gives semantic feedback to USS while preserving the existing code contracts and avoiding the memory cost of putting USS and SC on the same GPU.
