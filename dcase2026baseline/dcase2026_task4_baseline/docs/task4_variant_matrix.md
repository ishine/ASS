# Task 4 Variant Matrix

This table groups the current variants by the stage where they belong. The main
dependency rule is:

```text
USS variant -> estimated-slot cache -> SC stage3 adaptation -> TSE estimated-enrollment fine-tune -> S5 eval
```

Do not silently swap a later-stage model onto a cache from a different USS/TSE
distribution when judging final quality.

## Recommended Promotion Path

| Order | Stage | Config | Output checkpoint name to standardize | Purpose / gate |
| ---: | --- | --- | --- | --- |
| 1 | USS bootstrap/final candidate | `config/separation/modified_deft_uss_lite_6s.yaml` | `checkpoint/modified_deft_uss_lite_6s.ckpt` | First strong memory-efficient spatial USS; exports the estimated slots used by SC/TSE. |
| 2 | SC clean bootstrap | `config/label/m2d_sc_stage1_strong.yaml` -> `config/label/m2d_sc_stage2_strong.yaml` | `checkpoint/m2d_sc_stage2_strong.ckpt` | Clean-source strong M2D classifier; used for first S5/cache export. |
| 3 | TSE bootstrap | `config/separation/modified_deft_tse_lite_6s.yaml` | `checkpoint/modified_deft_tse_lite_6s.ckpt` | Oracle-enrollment TSE; enough to run the first S5, not final. |
| 4 | Cache export | `src/evaluation/export_sc_finetune_cache.py` with current S5 eval config | `workspace/sc_finetune/*` | Produces full 10s `soundscape/`, `oracle_target/`, `estimate_target/`. |
| 5 | SC estimated-source adaptation | `config/label/m2d_sc_stage3_estimated_strong.yaml` | `checkpoint/m2d_sc_stage3_estimated_strong.ckpt` | Adapts SC to distorted USS/TSE estimates. |
| 6 | TSE estimated-enrollment adaptation | `config/separation/modified_deft_tse_lite_6s_estimated_enrollment.yaml` | `checkpoint/modified_deft_tse_lite_6s_estimated_enrollment.ckpt` | Fine-tunes TSE on USS/S5 estimates as enrollment using random 6s crops. |
| 7 | TSE final 10s alignment | `config/separation/modified_deft_tse_lite_10s_estimated_enrollment.yaml` | final TSE candidate checkpoint | Short full-10s fine-tune to match official/eval clip length. |
| 8 | Main S5 eval | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml` | result directory | Full S5 validation; this is the real promotion gate. |
| 9 | Duplicate-recall eval | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc_duplicate_recall.yaml` | result directory | Tests same-class duplicate recovery. Promote only if CAPI-SDRi improves without false positives. |

## USS Variants

| Variant | Config | Model class | Works with next stage | Purpose / feature | Details |
| --- | --- | --- | --- | --- | --- |
| Legacy ResUNet baseline | `config/separation/resunetk_capisdr.yaml` | `ResUNet30` | Legacy S5 only | Baseline comparison. | Older path; not the current top-score branch. |
| Full spatial DeFT USS | `config/separation/modified_deft_uss.yaml` | `ModifiedDeFTUSSSpatial` | S5 / cache export | Stronger but heavier spatial DeFT USS. | Uses 4ch spatial STFT and phase-aware masks. Expensive; use as final candidate if A6000 memory/time allows. |
| 5s chunked spatial USS | `config/separation/modified_deft_uss_5s.yaml` | `ChunkedModifiedDeFTUSSSpatial` | S5 / cache export | Memory-reduced training with global DeFT blocks. | Trains 5s; eval chunks longer input. Useful middle ground. |
| 6s lite spatial USS | `config/separation/modified_deft_uss_lite_6s.yaml` | `ModifiedDeFTUSSMemoryEfficient` | Main cache, SC stage3, TSE estimated enrollment, lite S5 eval | Recommended first serious USS. | Same 96 channels / 6 blocks / 4 heads, but memory-efficient local time and grouped frequency attention. Spatial 4ch input, mono output. |
| Full temporal spatial USS | `config/separation/modified_deft_uss_temporal.yaml` | `ModifiedDeFTUSSSpatialTemporal` | Temporal S5 / temporal cache | Adds per-object activity heads. | Higher risk/heavier. Useful if temporal gating clearly helps. |
| 5s temporal chunked USS | `config/separation/modified_deft_uss_5s_temporal.yaml` | `ChunkedModifiedDeFTUSSSpatialTemporal` | Temporal S5 / temporal cache | Temporal activity with chunked eval. | Keeps global block style but adds activity outputs. |
| 6s lite temporal USS | `config/separation/modified_deft_uss_lite_6s_temporal.yaml` | `ModifiedDeFTUSSMemoryEfficientTemporal` | `Kwon2025TemporalS5`, temporal TSE, temporal SC | Recommended temporal USS candidate. | Memory-efficient plus foreground/interference/noise activity logits. |

## TSE Variants

| Variant | Config | Model class | Should work after / with | Purpose / feature | Details |
| --- | --- | --- | --- | --- | --- |
| Legacy DeFT-TSE-like | `config/separation/deft_tse_like.yaml` | `DeFTTSELikeSpatial` | Legacy ablation | Older label-query separator. | Not the current recommended S5 TSE. |
| Temporal DeFT-TSE-like | `config/separation/deft_tse_like_temporal.yaml` | `DeFTTSELikeSpatialTemporal` | Temporal ablation | Adds temporal activity head to legacy-like TSE. | Use only as ablation. |
| Full modified DeFT TSE | `config/separation/modified_deft_tse.yaml` | `ModifiedDeFTTSE` | Full USS / legacy top1-like eval | Oracle-enrollment TSE baseline. | Full/global path, heavier. |
| 6s lite TSE bootstrap | `config/separation/modified_deft_tse_lite_6s.yaml` | `ModifiedDeFTTSEMemoryEfficient` | First S5 and estimated-enrollment fine-tune | Recommended TSE bootstrap. | Trains with oracle dry enrollment; eval handles long input by chunking. |
| 6s lite estimated-enrollment TSE | `config/separation/modified_deft_tse_lite_6s_estimated_enrollment.yaml` | `ModifiedDeFTTSEMemoryEfficient` | 10s estimated-enrollment TSE or S5 eval | Aligns TSE with real S5 enrollment distribution. | Loads `checkpoint/modified_deft_tse_lite_6s.ckpt`; uses `estimate_target` as enrollment, oracle target as waveform, random 6s crops. |
| 10s lite estimated-enrollment TSE | `config/separation/modified_deft_tse_lite_10s_estimated_enrollment.yaml` | `ModifiedDeFTTSEMemoryEfficient` | Final non-temporal S5 | Final official-length alignment. | Loads `checkpoint/modified_deft_tse_lite_6s_estimated_enrollment.ckpt`; full cached 10s, batch 1, LR 2e-5, 10 epochs. |
| Full temporal modified DeFT TSE | `config/separation/modified_deft_tse_temporal.yaml` | `ModifiedDeFTTSETemporal` | Temporal S5 ablation | Adds per-query activity logits and time-FiLM conditioning. | Non-memory-efficient temporal path. |
| 6s lite temporal TSE bootstrap | `config/separation/modified_deft_tse_lite_6s_temporal.yaml` | `ModifiedDeFTTSEMemoryEfficientTemporal` | Temporal estimated-enrollment fine-tune | Temporal TSE bootstrap. | Oracle-enrollment training plus activity loss. |
| 6s lite temporal estimated-enrollment TSE | `config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment.yaml` | `ModifiedDeFTTSEMemoryEfficientTemporal` | 10s temporal estimated-enrollment TSE | Temporal TSE aligned to estimated enrollments. | Loads `checkpoint/modified_deft_tse_lite_6s_temporal.ckpt`; random 6s crops. |
| 10s lite temporal estimated-enrollment TSE | `config/separation/modified_deft_tse_lite_10s_temporal_estimated_enrollment.yaml` | `ModifiedDeFTTSEMemoryEfficientTemporal` | Final temporal S5 | Final temporal official-length alignment. | Loads `checkpoint/modified_deft_tse_lite_6s_temporal_estimated_enrollment.ckpt`; full cached 10s. |

## SC Variants

| Variant | Config | Model class | Should work after / with | Purpose / feature | Details |
| --- | --- | --- | --- | --- | --- |
| Legacy M2D SC stage1 | `config/label/m2d_sc_stage1.yaml` | `M2DSingleClassifier` | Legacy stage2 | Original/simple SC. | Keep for baseline only. |
| Legacy M2D SC stage2 | `config/label/m2d_sc_stage2.yaml` | `M2DSingleClassifier` | Legacy S5 eval | Simple SC with energy/silence. | Used by older eval configs. |
| Strong M2D SC stage1 | `config/label/m2d_sc_stage1_strong.yaml` | `M2DSingleClassifierStrong` | `m2d_sc_stage2_strong.yaml` | Recommended clean-source SC bootstrap. | Attentive statistics pooling, projection head, ArcFace. |
| Strong M2D SC stage2 | `config/label/m2d_sc_stage2_strong.yaml` | `M2DSingleClassifierStrong` | first S5/cache export or stage3 estimated SC | Strong clean-source classifier with energy/silence. | First reliable labeler before estimated-source adaptation. |
| Strong estimated SC | `config/label/m2d_sc_stage3_estimated_strong.yaml` | `M2DSingleClassifierStrong` | main S5 eval, calibration | Adapts classifier to estimated slots. | Trains from `workspace/sc_finetune/estimate_target`. |
| Strong estimated SC robust | `config/label/m2d_sc_stage3_estimated_strong_robust.yaml` | `M2DSingleClassifierStrong` | S5 eval if noisy cache hurts | Adds loss truncation for noisy estimated labels. | Use after baseline estimated SC exists. |
| Temporal strong stage1 | `config/label/m2d_sc_stage1_temporal_strong.yaml` | `M2DSingleClassifierTemporalStrong` | temporal stage2 | Temporal SC bootstrap. | Adds span-supervised activity head and activity-aware pooling. |
| Temporal strong stage2 | `config/label/m2d_sc_stage2_temporal_strong.yaml` | `M2DSingleClassifierTemporalStrong` | temporal estimated SC / temporal S5 bootstrap | Clean-source temporal SC with energy/silence. | Supplies activity probabilities to temporal S5. |
| Temporal estimated SC | `config/label/m2d_sc_stage3_estimated_temporal_strong.yaml` | `M2DSingleClassifierTemporalStrong` | `Kwon2025TemporalS5` | Temporal SC adapted to estimated slots. | Outputs labels, energy, and activity probabilities. |
| Temporal estimated SC robust | `config/label/m2d_sc_stage3_estimated_temporal_strong_robust.yaml` | `M2DSingleClassifierTemporalStrong` | temporal S5 if cache labels noisy | Robust temporal estimated-source SC. | Combines activity loss with truncation. |
| BEATs fusion stage1 | `config/label/m2d_sc_stage1_beats_fusion.yaml` | `M2DPretrainedFusionClassifier` | BEATs stage2 | Strong semantic pretrained SC upgrade. | Frozen BEATs branch plus M2D, default `concat_mlp`. |
| BEATs fusion stage2 | `config/label/m2d_sc_stage2_beats_fusion.yaml` | `M2DPretrainedFusionClassifier` | BEATs stage3 | BEATs fusion with energy/silence. | Needs `external/BEATs` and BEATs checkpoint. |
| BEATs estimated fusion | `config/label/m2d_sc_stage3_estimated_beats_fusion.yaml` | `M2DPretrainedFusionClassifier` | BEATs S5 eval | Best first pretrained SC upgrade. | Adapts M2D+BEATs to estimated slots. |
| BEATs estimated fusion robust | `config/label/m2d_sc_stage3_estimated_beats_fusion_robust.yaml` | `M2DPretrainedFusionClassifier` | BEATs S5 eval if noisy labels | Robust BEATs estimated-source SC. | Promote only after non-robust BEATs comparison. |
| fPaSST fusion stage1 | `config/label/m2d_sc_stage1_fpasst_fusion.yaml` | `M2DPretrainedFusionClassifier` | fPaSST stage2 | Frame-oriented pretrained SC upgrade. | Needs `external/PretrainedSED` and fPaSST checkpoint. |
| fPaSST fusion stage2 | `config/label/m2d_sc_stage2_fpasst_fusion.yaml` | `M2DPretrainedFusionClassifier` | fPaSST stage3 | fPaSST fusion with energy/silence. | Good ablation against BEATs. |
| fPaSST estimated fusion | `config/label/m2d_sc_stage3_estimated_fpasst_fusion.yaml` | `M2DPretrainedFusionClassifier` | fPaSST S5 eval | Estimated-source fPaSST SC adaptation. | Try if BEATs is weak or extra GPU time exists. |
| fPaSST estimated fusion robust | `config/label/m2d_sc_stage3_estimated_fpasst_fusion_robust.yaml` | `M2DPretrainedFusionClassifier` | fPaSST S5 eval if noisy labels | Robust fPaSST estimated-source SC. | Lower priority than BEATs robust unless fPaSST wins. |
| Old 1ch M2D-AT | `config/label/m2dat_1c.yaml`, `config/label/m2dat_1c_2blks.yaml` | `M2dAt` | legacy eval | Older 1-channel classifier. | Keep for baseline/compatibility. |
| Old 4ch M2D-AT | `config/label/m2dat_4c.yaml`, `config/label/m2dat_4c_2blks.yaml` | `M2dAtSpatial` | legacy eval | Older 4-channel classifier. | Assumes old 19-class silence-style contract in some paths. |

## S5 Evaluation Variants

| Variant | Config | Wrapper | Expects checkpoints | Purpose / feature | Notes |
| --- | --- | --- | --- | --- | --- |
| Full legacy Kwon-like | `src/evaluation/eval_configs/kwo2025_top1_like.yaml` | `Kwon2025S5` | `modified_deft_uss.ckpt`, `m2d_sc_stage2.ckpt`, `modified_deft_tse.ckpt` | Baseline full model. | Heavy/older path. |
| Chunked USS Kwon-like | `src/evaluation/eval_configs/kwo2025_top1_like_chunked_uss.yaml` | `Kwon2025S5` | `modified_deft_uss_5s.ckpt`, `m2d_sc_stage2.ckpt`, `modified_deft_tse.ckpt` | Tests 5s chunked USS with older SC/TSE. | Baseline ablation. |
| Lite USS/TSE with old SC | `src/evaluation/eval_configs/kwo2025_top1_like_lite_uss.yaml` | `Kwon2025S5` | `modified_deft_uss_lite_6s.ckpt`, `m2d_sc_stage2.ckpt`, `modified_deft_tse_lite_6s.ckpt` | Lite separation with older SC. | Compatibility/ablation. |
| Lite with clean strong SC | `src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml` | `Kwon2025S5` | lite USS/TSE + `m2d_sc_stage2_strong.ckpt` | First cache/export system. | Use before stage3 estimated SC is trained. |
| Lite clean strong SC + duplicate recall | `src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc_duplicate_recall.yaml` | `Kwon2025S5` | same as above | Tests same-class recovery before estimated SC. | Diagnostic only; not final if stage3 exists. |
| Lite with estimated strong SC | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml` | `Kwon2025S5` | lite USS/TSE + `m2d_sc_stage3_estimated_strong.ckpt` | Main non-temporal S5 candidate. | After TSE 10s alignment, update `tse_ckpt` to final aligned checkpoint. |
| Lite estimated strong SC + duplicate recall | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc_duplicate_recall.yaml` | `Kwon2025S5` | same as above | Tests duplicate same-class recovery. | Promote only if S5 metric improves. |
| Lite with BEATs estimated SC | `src/evaluation/eval_configs/kwo2025_top1_like_lite_beats_fusion_sc.yaml` | `Kwon2025S5` | lite USS/TSE + `m2d_sc_stage3_estimated_beats_fusion.ckpt` | Main pretrained SC upgrade candidate. | Calibrate thresholds before judging. |
| Lite BEATs + duplicate recall | `src/evaluation/eval_configs/kwo2025_top1_like_lite_beats_fusion_sc_duplicate_recall.yaml` | `Kwon2025S5` | same as above | Duplicate recall on BEATs SC. | Good if BEATs improves duplicate confidence. |
| Lite with fPaSST estimated SC | `src/evaluation/eval_configs/kwo2025_top1_like_lite_fpasst_fusion_sc.yaml` | `Kwon2025S5` | lite USS/TSE + `m2d_sc_stage3_estimated_fpasst_fusion.ckpt` | Frame-oriented pretrained SC ablation. | Lower priority unless BEATs underperforms. |
| Lite fPaSST + duplicate recall | `src/evaluation/eval_configs/kwo2025_top1_like_lite_fpasst_fusion_sc_duplicate_recall.yaml` | `Kwon2025S5` | same as above | Duplicate recall on fPaSST SC. | Promote only from measured S5 score. |
| Temporal lite S5 | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc.yaml` | `Kwon2025TemporalS5` | temporal lite USS/TSE + temporal estimated SC | Uses USS/SC/TSE activity for gating and time-FiLM conditioning. | Main temporal candidate. |
| Temporal lite S5 + duplicate recall | `src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_duplicate_recall.yaml` | `Kwon2025TemporalS5` | same as above | Duplicate recall requires temporal support. | Safer duplicate-recall variant. |
| Old m2dat/resunet 1ch | `src/evaluation/eval_configs/m2dat_1c_resunetk.yaml` | `S5` | old SC/separator stack | Legacy baseline. | Beware old silence contract assumptions. |
| Old m2dat/resunet 4ch | `src/evaluation/eval_configs/m2dat_4c_resunetk.yaml` | `S5` | old SC/separator stack | Legacy spatial baseline. | Compatibility only. |
| Self-guided M2D/DeFT | `src/evaluation/eval_configs/selfguided_m2dat_deft.yaml` | `SelfGuidedS5` | self-guided stack | Iterative/self-guided ablation. | Not current main branch. |

## Practical Notes

- Any SC checkpoint used in S5 should get per-class energy-threshold calibration
  before final comparison.
- If the final USS changes, regenerate `workspace/sc_finetune` and redo SC
  stage3 plus TSE estimated-enrollment fine-tunes.
- `*_duplicate_recall.yaml` should be treated as an S5-level evaluation variant,
  not a training stage.
- `*_robust.yaml` SC configs are stage3 estimated-source fine-tunes only; use
  them after the matching non-robust branch establishes a useful baseline.
- Gated fusion is enabled by changing `fusion_mode: gated_mlp` in a sibling
  BEATs/fPaSST config; do not overwrite the default concat configs.
