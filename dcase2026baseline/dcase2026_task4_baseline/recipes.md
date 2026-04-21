# Recipes

## `kwo2025_top1_like`

This recipe is a closer implementation of the 2025 rank-1 KAIST system described in:

- DCASE 2025 technical report
- DCASE 2025 workshop paper

The key architectural choices now match the report more closely:

- `USS -> M2D-SC -> TSE -> M2D-SC -> TSE`
- modified `DeFT` blocks with the challenge-system simplification that removes the Mamba part
- `M2D-SC` as a single-label classifier with:
  - ArcFace training in step 1
  - KL loss on silence
  - energy-based silence learning in step 2
- `TSE` using:
  - raw enrollment waveform injection through complex-spectrogram concatenation
  - class one-hot conditioning through shared `Res-FiLM`
- final iterative refinement for stage 3 inference

Files:

- `src/models/deft/modified_deft.py`
- `src/models/m2dat/m2d_sc.py`
- `src/models/s5/kwo2025.py`
- `src/training/loss/m2d_sc_arcface.py`
- `src/training/loss/uss_loss.py`
- `src/training/loss/masked_snr.py`

Training sequence:

```bash
cd /home/cmj/works/dcase2026/dcase2026_task4_baseline

# stage 0: universal sound separation
python -m src.train -c config/separation/modified_deft_uss.yaml -w workspace/separation

# stage 1: single-label classifier
python -m src.train -c config/label/m2d_sc_stage1.yaml -w workspace/label

# stage 2: single-label classifier with energy hinge
python -m src.train -c config/label/m2d_sc_stage2.yaml -w workspace/label

# stage 3: target sound extractor
python -m src.train -c config/separation/modified_deft_tse.yaml -w workspace/separation
```

Evaluation:

```bash
cd /home/cmj/works/dcase2026/dcase2026_task4_baseline
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/kwo2025_top1_like.yaml --result_dir workspace/evaluation
```

Important notes:

- The public KAIST papers specify the system structure and losses, but do not release the full original training code or checkpoints.
- Their 2025 recipe also used external data changes including `VCTK` and `Pixabay`; these are not currently listed on the DCASE 2026 allowed external-resource page, so they are not part of the default 2026 submission recipe here.
- The current repo still needs the `SpAudSyn` module copied into `src/modules/spatial_audio_synthesizer` for on-the-fly training data generation, exactly as required by the official baseline.
- `modified_deft_tse.yaml` currently trains TSE with oracle foreground enrollments from the dataset wrapper. The 2025 paper trained TSE with USS-generated enrollments, so the next fidelity upgrade is to precompute or on-the-fly generate USS enrollments from a trained `ModifiedDeFTUSS` checkpoint.

Next high-value steps:

- replace oracle TSE enrollments with cached or on-the-fly USS enrollments
- calibrate per-class energy thresholds for `M2D-SC` from validation data instead of using a placeholder default
- add a 2026-legal stronger recipe using allowed pretrained models such as `AudioSep` or `BEATs`
