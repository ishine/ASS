# USS + SC Model-Parallel Joint Fine-tuning

This is an opt-in training path. It does not change existing USS or SC modules/configs.

## Goal

Use a strong source-classifier (SC) as a semantic teacher for USS separated foreground waveforms without fitting both large models on one GPU.

Model placement:

```text
USS model -> cuda:0
SC model  -> cuda:1
```

PyTorch autograd can backpropagate through the differentiable copy from `cuda:0` to `cuda:1`, so SC loss can update USS even when SC parameters are frozen.

## Recommended stages

1. Train USS alone.
2. Export USS estimated sources with oracle-matched labels using `src.tools.export_estimated_sources_for_sc`.
3. Train SC on oracle + estimated-source data.
4. Run this joint fine-tuning path with frozen SC first.
5. Only after stable training, optionally set `freeze_sc: false` for alternating USS/SC updates.

## Run

```bash
python -m src.train \
  -c config/separation/modified_deft_uss_sc_joint_model_parallel_min.yaml \
  -w workspace/uss_sc_joint
```

The config intentionally uses:

```yaml
trainer:
  accelerator: gpu
  devices: 1
  strategy: auto
```

Do not use DDP for this mode. DDP would replicate both USS and SC on each GPU, which defeats the purpose.

## Main module

```text
src/training/lightningmodule/uss_sc_joint_model_parallel.py
```

Main class:

```python
USSCSJointModelParallelLightning
```

It manually places the two models on different devices and uses manual optimization.

## Important settings

Frozen-SC teacher mode:

```yaml
freeze_sc: true
lambda_sc: 0.05
lambda_consistency: 0.02
detach_waveform_for_sc: false
```

`freeze_sc: true` freezes SC parameters, but gradients still flow through SC to the separated waveform and then back to USS. Do not set `detach_waveform_for_sc: true` unless you only want to train/evaluate SC diagnostics without updating USS from SC loss.

Full joint mode:

```yaml
freeze_sc: false
sc_update_every: 4
```

The module can return two optimizers and update SC less frequently than USS.

## Label assignment

SC labels are not taken from the USS class head. The module computes oracle waveform matching inside each batch:

```text
USS foreground estimate -> pairwise SA-SDR/SI-SDR to oracle foreground -> PIT assignment -> oracle class label
```

This avoids propagating the weak USS class-head predictions into SC.

## Sanity checks

```bash
python -m py_compile src/training/lightningmodule/uss_sc_joint_model_parallel.py
```

Then run a tiny test job with batch size 1 or 2. Watch:

- `step_train/loss_sc_weighted`
- `step_train/sc_joint_top1`
- `step_train/uss_loss_fg_wave`
- `epoch_val/loss`

If SDR drops quickly, reduce `lambda_sc` to `0.01`.
If SC loss has no effect on USS, confirm `detach_waveform_for_sc: false` and do not wrap SC forward in `torch.no_grad()` during training.
