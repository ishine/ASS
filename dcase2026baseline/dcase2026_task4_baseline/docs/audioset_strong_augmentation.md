# AudioSet-Strong Augmentation Guide

This guide describes the opt-in workflow for adding clean AudioSet-Strong dry
sources to the DCASE 2026 Task 4 data pools, rebalancing their train/validation
split, and generating a validation metadata list that is compatible with the
existing `DatasetS3(mode=metadata)` pipeline.

All commands below assume the Docker container workspace:

```bash
cd /app/ASS/dcase2026baseline/dcase2026_task4_baseline
source /app/ASS/.venv/bin/activate
```

The AudioSet-Strong annotations are expected at:

```text
/app/ASS/dcase2026baseline/audioset_strong_annotations
```

## Outputs

The extraction script writes into the existing DCASE-style development-set
layout:

```text
data/dev_set/
  sound_event/{train,valid}/<DCASE2026 target label>/*.wav
  interference/{train,valid}/<interference label>/*.wav
  noise/{train,valid}/*.wav
  metadata/audioset_strong/*.jsonl|*.json
```

The validation metadata generation script writes:

```text
data/dev_set/metadata/audioset-strong-val.json
data/dev_set/metadata/audioset-strong-val/*.json
data/dev_set/metadata/audioset-strong-val/summary.json
```

The top-level JSON is a metadata list. Each entry points to one
SpAudSyn-style scene metadata file. This matches the config pattern already
used by validation datasets:

```yaml
mode: metadata
metadata_list: data/dev_set/metadata/audioset-strong-val.json
```

## 1. Inspect Annotation-Only Statistics

Start with a dry run. This parses the annotations, maps AudioSet labels to
DCASE target or interference labels, finds clean isolated candidates, and
writes manifests. It does not read raw audio.

```bash
python add_audioset_strong.py \
  --annotations_dir ../audioset_strong_annotations \
  --output_dir data/dev_set
```

Review the printed statistics before extraction:

- `Accepted by kind`: accepted foreground/interference/noise candidates.
- `Accepted by split`: AudioSet train/eval derived split.
- `Rejected by reason`: why annotations were discarded.
- Per-label foreground counts under `[sound_event]`.

The clean-source filter is intentionally conservative. A foreground event is
kept only when its annotated onset/offset does not overlap with another
annotated event in the same AudioSet clip. This is what makes the extracted WAV
usable as a dry source for synthetic scene generation.

Known AudioSet-Strong limitation: `Percussion` and `MusicalKeyboard` usually do
not appear as instrument-level strong labels in the current strong annotation
files. They may remain unavailable from AudioSet-Strong even though broader
`Music` annotations exist.

## 2. Extract WAVs

Run extraction only after checking the statistics. The input directory can
contain nested folders; the script searches recursively for audio files.

```bash
python add_audioset_strong.py \
  --annotations_dir ../audioset_strong_annotations \
  --input_dir /path/to/audioset/raw_wavs \
  --output_dir data/dev_set \
  --execute
```

Useful options:

```bash
--overwrite                 replace existing extracted files
--min_event_duration 0.15    discard shorter events
--overlap_margin 0.03        reject near-overlaps with this time margin
--no-trim_edges              keep exact annotated event boundaries
--amp_threshold 0.005        silence trimming threshold
--min_rms 1e-4               reject very quiet clips
```

All extracted WAVs are written at 32 kHz. By default the extractor writes mono
dry sources because these are later spatialized by SpAudSyn and the room RIRs.

## 3. Rebalance Train/Validation Split

The extractor preserves the annotation-derived split. If the validation split is
too small or too imbalanced, use the split adjustment script. It operates per
kind and per label, so each foreground class, interference class, and the noise
pool can get a stable validation ratio.

First run a dry run:

```bash
python adjust_audioset_strong_split.py \
  --data_root data/dev_set \
  --valid_ratio 0.15 \
  --seed 0
```

Then execute the planned file moves:

```bash
python adjust_audioset_strong_split.py \
  --data_root data/dev_set \
  --valid_ratio 0.15 \
  --seed 0 \
  --execute
```

The script writes a movement report by default:

```text
data/dev_set/metadata/audioset_strong/split_adjustment_summary.json
```

Use `--min_valid_per_group 0` if very small classes should be allowed to have
zero validation samples. The default is `1`, which keeps at least one validation
sample for any non-empty group.

## 4. Generate AudioSet-Strong Validation Metadata

Use the updated DCASE `valid.json` as a room template list. This reuses the
official room metadata and available SOFA positions instead of inventing a
separate room geometry.

Dry run:

```bash
python generate_audioset_strong_valid_json.py \
  --data_root data/dev_set \
  --room_template_list data/dev_set/metadata/valid.json \
  --valid_json data/dev_set/metadata/audioset-strong-val.json \
  --metadata_subdir audioset-strong-val \
  --num_scenes 1800 \
  --seed 0
```

Execute:

```bash
python generate_audioset_strong_valid_json.py \
  --data_root data/dev_set \
  --room_template_list data/dev_set/metadata/valid.json \
  --valid_json data/dev_set/metadata/audioset-strong-val.json \
  --metadata_subdir audioset-strong-val \
  --num_scenes 1800 \
  --seed 0 \
  --execute
```

If regenerating the same output path:

```bash
python generate_audioset_strong_valid_json.py \
  --data_root data/dev_set \
  --room_template_list data/dev_set/metadata/valid.json \
  --valid_json data/dev_set/metadata/audioset-strong-val.json \
  --metadata_subdir audioset-strong-val \
  --num_scenes 1800 \
  --seed 0 \
  --execute \
  --overwrite
```

The default `--metadata_backend spaudsyn` materializes each scene through:

```python
SpAudSyn.from_metadata(metadata).generate_metadata()
```

This keeps the generated files aligned with
`src.modules.spatial_audio_synthesizer` and the existing validation loader.
`--metadata_backend direct` exists only for debugging.

## 5. Validation-Set Rule Contract

For `--num_scenes 1800`, the generator enforces the DCASE validation-set scene
proportions exactly:

| Target count | Mixtures | Ratio |
| ---: | ---: | ---: |
| 0 | 300 | 16.7% |
| 1 | 300 | 16.7% |
| 2 | 600 | 33.3% |
| 3 | 600 | 33.3% |

Inside the two-target and three-target subsets, exactly 50% of scenes contain
multiple same-class target sources.

Other enforced settings:

- Mixture duration is 10 seconds.
- Sample rate is 32 kHz.
- Each mixture contains zero to three target events.
- Each mixture contains zero to two interference events.
- The maximum foreground plus interference overlap is three.
- Target SNR is uniformly sampled from 5 to 20 dB.
- Interference SNR is uniformly sampled from 0 to 15 dB.
- Same-class target source directions are separated by at least 60 degrees.
- Background is included by default.

`--num_scenes` must be divisible by 6 so the 0/1/2/3 target-count proportions
can be exact. The two-target and three-target subsets must also be even so the
same-class subset split can be exact.

## 6. Verify Generated Metadata

After generation, inspect the summary:

```bash
python -m json.tool \
  data/dev_set/metadata/audioset-strong-val/summary.json | head -80
```

The key fields to check are:

```text
target_count_scene_counts
same_class_subset_counts
max_overlap_counts
same_class_angle_violations
foreground_events
interference_events
background_events
metadata_backend
```

Expected full validation counts for 1,800 scenes:

```json
{
  "target_count_scene_counts": {
    "0": 300,
    "1": 300,
    "2": 600,
    "3": 600
  },
  "same_class_angle_violations": 0,
  "metadata_backend": "spaudsyn"
}
```

You can also smoke-test that the metadata is loadable by SpAudSyn:

```bash
python - <<'PY'
import json
from pathlib import Path
from src.modules.spatial_audio_synthesizer.spatial_audio_synthesizer import SpAudSyn

repo = Path(".")
metadata_list = repo / "data/dev_set/metadata/audioset-strong-val.json"
entries = json.loads(metadata_list.read_text())
first = repo / "data/dev_set/metadata" / entries[0]["metadata_path"]
metadata = json.loads(first.read_text())

synth = SpAudSyn.from_metadata(metadata)
generated = synth.generate_metadata()
mixture, fg, bg, inter = SpAudSyn.from_metadata(generated).synthesize()

print("metadata entries:", len(entries))
print("mixture shape:", mixture.shape)
print("foreground events:", len(fg))
print("background events:", len(bg))
print("interference events:", len(inter))
PY
```

For a 10-second FOA mixture at 32 kHz, `mixture shape` should be:

```text
(4, 320000)
```

## 7. Use AudioSet-Strong in Training

The existing training-generation configs already sample from these directories:

```yaml
foreground_dir: data/dev_set/sound_event/train
background_dir: data/dev_set/noise/train
interference_dir: data/dev_set/interference/train
room_config:
  module: src.modules.spatial_audio_synthesizer.room
  main: SofaRoom
  args:
    path: data/dev_set/room_ir/train
    direct_range_ms: [6, 50]
```

After extraction, the AudioSet-Strong WAVs live inside those same pools, so they
are automatically available to `DatasetS3(mode=generate)` as long as the config
points to `data/dev_set`. No default config has to be changed.

For a controlled ablation, keep separate data roots instead:

```text
data/dev_set                 official/current pool
data/dev_set_audioset_strong  AudioSet-Strong augmented pool
```

Then create a sibling config that points `foreground_dir`, `background_dir`,
and `interference_dir` to the augmented root. This keeps the baseline and the
AudioSet-Strong experiment comparable.

## 8. Use AudioSet-Strong in Validation

To validate against the generated AudioSet-Strong validation scenes, use a
sibling config or command-line override that changes only the metadata list:

```yaml
dataset:
  args:
    config:
      mode: metadata
      metadata_list: data/dev_set/metadata/audioset-strong-val.json
```

Keep `label_set: dcase2026t4`, `n_sources: 3`, and the model-side validation
settings unchanged unless the experiment needs a separate ablation.

## 9. Troubleshooting

If foreground labels are missing, rerun the extraction statistics and inspect
the per-label counts. AudioSet-Strong may simply not provide clean strong
annotations for every DCASE target class.

If `SpAudSyn` cannot be imported, confirm the module symlink and dependencies:

```bash
python - <<'PY'
import librosa
import sofa
from src.modules.spatial_audio_synthesizer.spatial_audio_synthesizer import SpAudSyn
print("SpAudSyn import OK")
PY
```

If SOFA room inspection fails, prefer this option:

```bash
--room_template_list data/dev_set/metadata/valid.json
```

This is the most robust path because it reuses metadata from known-good
validation scenes.

If generation fails because not enough labels are available, extract more clean
foreground clips or reduce the scene-generation target temporarily for a smoke
test:

```bash
python generate_audioset_strong_valid_json.py \
  --data_root data/dev_set \
  --room_template_list data/dev_set/metadata/valid.json \
  --valid_json /tmp/audioset-strong-val-smoke.json \
  --metadata_subdir audioset-strong-val-smoke \
  --num_scenes 60 \
  --seed 0
```

Use the 60-scene setting only for debugging. The validation-compatible setting
is `--num_scenes 1800`.
