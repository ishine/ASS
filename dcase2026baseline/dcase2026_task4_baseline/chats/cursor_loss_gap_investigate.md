# Loss discrepancy in SC model training
_Exported on 4/28/2026 at 00:20:24 GMT+8 from Cursor (3.1.17)_

---

**User**

when training the SC model with /home/cmj/works/dcase2026/dcase2026_task4_baseline/config/label/m2d_sc_stage1_strong.yaml, the loss bwteen training and evaluation is very big, e.g., epoch_training/loss_arcface is about 0.263, but the epoch_training/loss_arcface is about 4.36. After add a plain cross entropy loss for monitor, the difference are still big. Please help to investigate and try to fix

---

**Cursor**

I’ll trace how `loss_arcface` is computed in both train and eval paths for `m2d_sc_stage1_strong.yaml`, then reproduce the mismatch from code and patch the most likely inconsistency (usually margin/logit handling, label mapping, or model mode differences). I’m starting by locating the config and all arcface-related loss code.

I didn’t find matches at the repository root, so I’m now searching directly inside `dcase2026_task4_baseline` where your config lives and then I’ll inspect the exact train/eval loss code paths.

I’ve confirmed train and val both call the same `forward + loss_func`, so the next likely causes are label-index mapping or dataset-mode differences (`generate` vs `metadata`). I’m checking `DatasetS3` label construction and class-index handling next.

I found a likely measurement issue: `loss_arcface` is averaged only over active samples **inside each batch**, but epoch logging weights by full batch size, so silence-heavy train batches can artificially deflate train arcface a lot. I’m now quantifying silence/active ratios for your train config vs val metadata to confirm this mismatch before patching.

I found a concrete train/val mismatch in code: `eval_crop_seconds` is configured, but it is only used in `predict()` and **not** in `forward()`, while validation loss uses `forward()`. I’m patching `M2DSingleClassifierStrong.forward` so validation computes logits from the same crop-averaged representation used at inference, which should reduce the gap and make val loss more meaningful.

I’ve applied a targeted code fix in the SC model so validation now uses the configured eval-time crop averaging (`eval_crop_seconds`/`eval_crop_hop_seconds`) inside `forward`, not only in `predict`. I’m now summarizing the root cause and exactly what to rerun/check to confirm the loss gap improves.

The biggest issue I found is a **train/validation behavior mismatch** in the model path, and I patched it.

During training/validation, Lightning calls `model.forward(...)`, but in your SC model the eval-time crop logic (`eval_crop_seconds`, `eval_crop_hop_seconds`) was only used in `predict()`, not in `forward()`. So validation loss was being computed on a different inference path than intended.

I updated `forward()` to use crop-averaged logits/embeddings when `self.training == False` and crop config is enabled.

```651:671:src/models/m2dat/m2d_sc.py
def forward(self, input_dict):
    waveform = self._prepare_audio(input_dict["waveform"])
    if (not self.training) and (self.eval_crop_seconds is not None):
        embeddings = []
        plain_logits_all = []
        for crop in self._iter_eval_crops(waveform):
            crop_embedding, crop_plain_logits = self._plain_logits_from_waveform(crop)
            embeddings.append(crop_embedding)
            plain_logits_all.append(crop_plain_logits)
        embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
    else:
        embedding, plain_logits = self._plain_logits_from_waveform(waveform)
    logits = self.arc_head(embedding, input_dict.get("class_index"))
```

I also applied the same fix to the temporal strong variant for consistency:

```780:798:src/models/m2dat/m2d_sc.py
def forward(self, input_dict):
    waveform = self._prepare_audio(input_dict["waveform"])
    if (not self.training) and (self.eval_crop_seconds is not None):
        embeddings = []
        activity_logits_all = []
        plain_logits_all = []
        for crop in self._iter_eval_crops(waveform):
            crop_embedding, crop_activity_logits = self._embed_waveform_with_activity(crop)
            embeddings.append(crop_embedding)
            activity_logits_all.append(crop_activity_logits)
            plain_logits_all.append(self.arc_head(crop_embedding, None))
        embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        activity_logits = torch.stack(activity_logits_all, dim=0).mean(dim=0)
        plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
```

### What I observed in config/data
- Your config explicitly sets eval cropping in `m2d_sc_stage1_strong.yaml`, so this mismatch was real.
- Train uses `nevent_range: [0, 3]` (more silence-heavy), while `valid.json` has active ratio around `0.61` (computed from `fg_events_order` lengths). This can still keep train/eval loss apart even after the code fix, but the forward-path inconsistency was the first thing to fix.

### Next step
Please rerun training from scratch (or at least a few epochs) with this patch and compare:
- `epoch_train/loss_arcface`
- `epoch_val/loss_arcface`
- `epoch_train/loss_plain_ce`
- `epoch_val/loss_plain_ce`

If the gap is still large, I can do a second pass and patch logging/normalization so arcface metrics are weighted by active samples (not silence-heavy batch composition), which often makes train-vs-val curves much more interpretable.

---

**User**

why  ``` embedding, activity_logits = self._embed_waveform_with_activity(waveform) ``` is removed from the file?


---

**Cursor**

It wasn’t removed; it was moved into the `else` branch.

I changed `forward()` to support two paths:

- **Eval path** (`not self.training` and crop enabled): run multiple crops, then average embeddings/logits.
- **Default path** (training or no crop): keep original single-pass behavior.

So this line is still there:

```python
embedding, activity_logits = self._embed_waveform_with_activity(waveform)
```

It now executes in the default branch (`else`) instead of unconditionally at the top.  
Reason: in eval-crop mode, we must call `_embed_waveform_with_activity` **per crop** and then average results, so a single upfront call on full waveform would be inconsistent and unused.

---

**User**

I think you did not find the root cause. Could you try to trace the data flow in training and evaluation, and calculate the loss for comparing


---

**Cursor**

You’re right to push deeper. I’ll do a full trace from dataloader → model forward → loss logging for both train and val, then run a small reproducible script to compute per-batch and epoch-style aggregated losses side-by-side so we can identify the real source of the gap. I’m starting by checking your active terminal environments so I can run this with the same Python/Torch setup.

I’m setting up a reproducible loss-comparison script next; first I need the exact Python env with Torch from your training setup (the system `python3` here has no torch). I’ll locate that interpreter and then compute train-vs-val losses with identical code paths and explicit aggregation.

I’ve traced the end-to-end path and found we still need hard numbers from the *same* batches under train/eval modes to isolate root cause. I’m adding a dedicated debug script that computes comparable loss statistics for train-loader and val-loader (active ratio, ArcFace CE, plain CE, train-mode vs eval-mode on the same batch), so we can verify whether the gap is from data distribution, model mode, or logging aggregation.

I’m adding explicit **active-sample-weighted** train/val metrics in the SC Lightning module. This gives an apples-to-apples loss comparison (`*_active_weighted`) and should tell us immediately whether the big gap is real model behavior or mostly a logging/aggregation artifact.

You’re right — I did a deeper trace and added tooling so we can compare losses correctly.

### Data Flow Trace (train vs val)

- **Train dataloader**: `DatasetS3(mode=generate)` → `SourceClassifierDataset`  
  - config: `nevent_range: [0,3]`, `n_sources: 3`
  - source slot labels include padded `"silence"`  
- **Val dataloader**: `DatasetS3(mode=metadata)` with `valid.json` → `SourceClassifierDataset`
- **Both** go through same Lightning path:
  - `training_step_processing` / `validation_step_processing` in `single_label_classification.py`
  - model `forward(...)`
  - `loss_func(...)` in `m2d_sc_arcface.py`

### What I calculated

- Train expected active-slot ratio from config:
  - \((0+1+2+3)/4 / 3 = 0.5\)
- Val active-slot ratio from `valid.json`:
  - `3300 / 5400 = 0.6111`  
  - histogram from metadata: `{0:300, 1:300, 2:600, 3:600}`

Using your reported numbers (rough normalization sanity check):
- train active-normalized approx: `0.263 / 0.5 = 0.526`
- val active-normalized approx: `4.36 / 0.611 = 7.13`

So the gap is **not explained only by silence ratio**.

---

### Changes I made for proper comparison

1. **Added a debug tracer script**  
   `src/evaluation/debug_sc_loss_flow.py`
   - Computes, for both train-loader and val-loader:
     - active ratio
     - active-only ArcFace CE mean/sum
     - active-only plain CE mean/sum
   - Compares **model.train() vs model.eval() on the same batches** to isolate mode effects.

2. **Added active-weighted epoch metrics in Lightning**  
   `src/training/lightningmodule/single_label_classification.py`
   - New logs:
     - `epoch_train/loss_arcface_active_weighted`
     - `epoch_val/loss_arcface_active_weighted`
     - `epoch_train/loss_plain_ce_active_weighted`
     - `epoch_val/loss_plain_ce_active_weighted`
     - `epoch_train/active_ratio`
     - `epoch_val/active_ratio`
   - These are much better for train-vs-val comparison than raw batch-mean arcface logs.

---

### Run this in your training env

I couldn’t execute torch code in this shell (no torch installed in system python), so please run in your conda/env that you use for training:

```bash
python -m src.evaluation.debug_sc_loss_flow \
  -c config/label/m2d_sc_stage1_strong.yaml \
  --checkpoint <your_sc_ckpt_or_last.ckpt> \
  --device cuda \
  --max-batches 100
```

If you paste the output here, I can pinpoint whether the root cause is:
- true domain shift (generate train vs metadata val),
- eval-mode behavior,
- or remaining logging/aggregation artifacts.

---

**User**

you could try the venv for run the torch code


---

**Cursor**

Good call — I’ll locate and activate the project venv, then run the new loss-flow debug script there to produce concrete train-vs-val loss comparisons.

I tried exactly that with your project venv and it works for Torch:

- venv python: `dcase2026_task4_baseline/.venv/bin/python`
- torch version: `2.11.0+cu130`

Then I ran the debug tracer in that venv, but it failed because the configured M2D pretrained weight is missing locally:

- missing file:
`checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth`
- your `checkpoint/` folder currently only has `.gitignore`

I also improved the debug script so you can override that path without editing YAML:
- updated `src/evaluation/debug_sc_loss_flow.py` with `--m2d-weight-file`.

### Please run this (in repo root)
```bash
cd /home/cmj/works/dcase2026/dcase2026_task4_baseline
PYTHONPATH=. ./.venv/bin/python src/evaluation/debug_sc_loss_flow.py \
  -c config/label/m2d_sc_stage1_strong.yaml \
  --m2d-weight-file "<absolute_path_to_your_m2d_pretrained_weight>.pth" \
  --checkpoint "<optional_sc_checkpoint>.ckpt" \
  --device cuda \
  --max-batches 100
```

Once you paste that output, I can pinpoint the true cause from measured train/val flow (data ratio vs mode difference vs actual CE scale shift).

---

**User**

I had add the pretrained weights, you could retry

---

**Cursor**

Great — I’ll rerun the venv-based loss-flow trace now that weights are available, then I’ll report the measured train/val loss decomposition and propose a concrete fix based on those numbers.

The previous run got stuck in dataloader worker processes. I’m patching the debug script to force `num_workers=0` (and optional batch-size override) so we can get reliable results quickly, then rerunning immediately.

I retried in your `.venv` successfully and got concrete trace outputs.

### What I ran
- `PYTHONPATH=. ./.venv/bin/python src/evaluation/debug_sc_loss_flow.py -c config/label/m2d_sc_stage1_strong.yaml --device cpu --max-batches 2 --num-workers 0 --batch-size 2`
- It now runs end-to-end with your added pretrained M2D weights.

### Measured results (current run, **no SC finetune checkpoint loaded**)
- `train_loader_evalmode/loss_arcface_active_global_mean`: **18.05**
- `val_loader_evalmode/loss_arcface_active_global_mean`: **17.77**
- `train_loader_evalmode/active_ratio_global`: **0.75** (tiny sample)
- `val_loader_evalmode/active_ratio_global`: **0.50** (tiny sample)

So the train/val flow itself is consistent in this trace; I cannot reproduce the `0.263 vs 4.36` gap from just base pretrained initialization.

## Important blocker to reproduce your issue
Your large gap is from your **trained SC checkpoint**, but that checkpoint path is not provided (and no `.ckpt` is in this repo tree currently).  
Without loading that exact checkpoint, we can’t diagnose the true root cause of your observed epoch metrics.

### Next step
Please share your SC checkpoint path, then I’ll run:

```bash
cd /home/cmj/works/dcase2026/dcase2026_task4_baseline
PYTHONPATH=. ./.venv/bin/python src/evaluation/debug_sc_loss_flow.py \
  -c config/label/m2d_sc_stage1_strong.yaml \
  --checkpoint "<your_sc_checkpoint>.ckpt" \
  --device cuda \
  --max-batches 100 \
  --num-workers 0 \
  --batch-size 8
```

I already added this tracer and active-weighted logging hooks, so once I run with your checkpoint I can pinpoint whether the gap comes from:
- data/domain shift (`generate` train vs `metadata` val),
- mode mismatch (`train()` vs `eval()` behavior),
- or metric aggregation artifacts.

