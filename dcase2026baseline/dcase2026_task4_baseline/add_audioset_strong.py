#!/usr/bin/env python3
"""
AudioSet-Strong -> DCASE2026 Task4 augmentation extractor.

Default mode is annotation-only statistics. In that mode --input_dir is not
needed and no audio files are read. Real WAV extraction requires --execute and
then --input_dir must point to the raw AudioSet audio root.

Output layout, where --output_dir is a DCASE dev_set root:
  <output_dir>/sound_event/{train,valid}/<DCASE2026_LABEL>/*.wav
  <output_dir>/interference/{train,valid}/<DCASE_INTERFERENCE_LABEL>/*.wav
  <output_dir>/noise/{train,valid}/*.wav
  <output_dir>/metadata/audioset_strong/*.jsonl|*.json

All extracted WAV files are written at 32 kHz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    sf = None
    _SF_ERROR = exc
else:
    _SF_ERROR = None

try:
    import librosa
except Exception as exc:  # pragma: no cover
    librosa = None
    _LIBROSA_ERROR = exc
else:
    _LIBROSA_ERROR = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_: object):
        return x


TARGET_SR = 32000
AUDIOSET_SEGMENT_DURATION = 10.0
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac")

DCASE2026_LABELS = [
    "AlarmClock", "BicycleBell", "Blender", "Buzzer",
    "Clapping", "Cough", "CupboardOpenClose", "Dishes",
    "Doorbell", "FootSteps", "HairDryer", "MechanicalFans",
    "MusicalKeyboard", "Percussion", "Pour", "Speech",
    "Typing", "VacuumCleaner",
]

DCASE_INTERFERENCE_LABELS = [
    "Air conditioning", "Aircraft", "Bird flight, flapping wings", "Bleat", "Boiling", "Boom",
    "Burping, eructation", "Burst, pop", "Bus", "Camera", "Car passing by",
    "Cattle, bovinae", "Chainsaw", "Chewing, mastication", "Chink, clink", "Clip-clop",
    "Cluck", "Clunk", "Coin (dropping)", "Crack", "Crackle", "Creak", "Croak", "Crow",
    "Crumpling, crinkling", "Crushing", "Drill", "Drip", "Electric toothbrush", "Engine",
    "Fart", "Finger snapping", "Fire", "Fire alarm", "Firecracker", "Fireworks",
    "Fixed-wing aircraft, airplane", "Frog", "Gears", "Growling", "Gurgling", "Helicopter",
    "Hiss", "Hoot", "Howl", "Howl (wind)", "Jackhammer", "Keys jangling", "Lawn mower",
    "Light engine (high frequency)", "Microwave oven", "Moo", "Oink", "Packing tape, duct tape",
    "Pig", "Printer", "Purr", "Rain", "Rain on surface", "Raindrop", "Ratchet, pawl",
    "Rattle", "Sanding", "Sawing", "Scissors", "Screech", "Sheep", "Ship",
    "Shuffling cards", "Skateboard", "Slam", "Sliding door", "Sneeze", "Sniff", "Snoring",
    "Splinter", "Squeak", "Stream", "Subway, metro, underground", "Tap", "Tearing",
    "Thump, thud", "Tick", "Tick-tock", "Toothbrush", "Traffic noise, roadway noise",
    "Train", "Train horn", "Velcro, hook and loop fastener", "Waterfall",
    "Whoosh, swoosh, swish", "Wind", "Writing", "Zipper (clothing)",
]

# Conservative AudioSet display-name aliases for DCASE target classes.
AUDIOSET_TO_DCASE: Dict[str, str] = {
    "alarm clock": "AlarmClock",
    "bicycle bell": "BicycleBell",
    "blender": "Blender",
    "blender, food processor": "Blender",
    "buzzer": "Buzzer",
    "buzz": "Buzzer",
    "beep, bleep": "Buzzer",
    "clapping": "Clapping",
    "applause": "Clapping",
    "cough": "Cough",
    "cupboard open or close": "CupboardOpenClose",
    "drawer open or close": "CupboardOpenClose",
    "doorbell": "Doorbell",
    "ding-dong": "Doorbell",
    "walk, footsteps": "FootSteps",
    "footsteps": "FootSteps",
    "walking": "FootSteps",
    "hair dryer": "HairDryer",
    "fan": "MechanicalFans",
    "mechanical fan": "MechanicalFans",
    "keyboard (musical)": "MusicalKeyboard",
    "electric piano": "MusicalKeyboard",
    "piano": "MusicalKeyboard",
    "percussion": "Percussion",
    "drum": "Percussion",
    "drum kit": "Percussion",
    "drum roll": "Percussion",
    "snare drum": "Percussion",
    "bass drum": "Percussion",
    "tabla": "Percussion",
    "cymbal": "Percussion",
    "hi-hat": "Percussion",
    "pour": "Pour",
    "speech": "Speech",
    "male speech, man speaking": "Speech",
    "female speech, woman speaking": "Speech",
    "child speech, kid speaking": "Speech",
    "conversation": "Speech",
    "narration, monologue": "Speech",
    "typing": "Typing",
    "computer keyboard": "Typing",
    "typewriter": "Typing",
    "dishes, pots, and pans": "Dishes",
    "cutlery, silverware": "Dishes",
    "vacuum cleaner": "VacuumCleaner",
}


@dataclass(frozen=True)
class StrongEvent:
    uid: str
    segment_id: str
    ytid: str
    start: float
    end: float
    raw_label: str
    display_label: str
    dcase_label: Optional[str]
    interference_label: Optional[str]
    split: str
    annotation_file: str
    line_no: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Candidate:
    uid: str
    kind: str  # sound_event, interference, noise, reject
    split: str
    label: str
    segment_id: str
    start: float
    end: float
    duration: float
    annotation_file: str
    raw_label: str = ""
    display_label: str = ""
    reject_reason: str = ""


@dataclass
class ExtractResult:
    uid: str
    kind: str
    split: str
    label: str
    source_path: str
    output_path: str
    start: float
    end: float
    duration: float
    status: str
    reason: str = ""
    source_sr: Optional[int] = None
    target_sr: int = TARGET_SR
    peak: Optional[float] = None
    rms: Optional[float] = None


def norm_key(text: object) -> str:
    s = str(text or "").strip().lower().replace("#", "")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s.strip("_")


def norm_label(text: object) -> str:
    s = str(text or "").strip().replace("_", " ")
    return re.sub(r"\s+", " ", s)


def label_key(text: object) -> str:
    return norm_label(text).lower()


def safe_filename_part(text: object, max_len: int = 96) -> str:
    s = str(text or "unknown").strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._+\-=]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return (s or "unknown")[:max_len]


def parse_float(value: object) -> Optional[float]:
    try:
        value = float(str(value).strip())
    except Exception:
        return None
    return value if math.isfinite(value) else None


INTERFERENCE_BY_KEY = {label_key(x): x for x in DCASE_INTERFERENCE_LABELS}


def resolve_interference_label(display_or_raw: str) -> Optional[str]:
    return INTERFERENCE_BY_KEY.get(label_key(display_or_raw))


def resolve_dcase_label(display_or_raw: str) -> Optional[str]:
    label = norm_label(display_or_raw)
    if label in DCASE2026_LABELS:
        return label
    compact = re.sub(r"[^a-z0-9]", "", label.lower())
    for dcase in DCASE2026_LABELS:
        if re.sub(r"[^a-z0-9]", "", dcase.lower()) == compact:
            return dcase
    return AUDIOSET_TO_DCASE.get(label_key(label))


def choose_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = "".join(f.readline() for _ in range(20))
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_table(path: Path) -> Iterator[Tuple[int, Dict[str, str]]]:
    delimiter = choose_delimiter(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first_row: Optional[List[str]] = None
        first_line_no = 0
        for line_no, row in enumerate(reader, start=1):
            if row and any(cell.strip() for cell in row):
                first_row = row
                first_line_no = line_no
                break
        if first_row is None:
            return

        normalized = [norm_key(c) for c in first_row]
        header_tokens = {
            "segment_id", "ytid", "youtube_id", "video_id", "file_name", "filename", "audio_id",
            "start_time_seconds", "end_time_seconds", "start_seconds", "end_seconds",
            "onset", "offset", "label", "labels", "mid", "display_name", "positive_labels",
        }
        if set(normalized) & header_tokens:
            header = normalized
        elif len(first_row) >= 4:
            header = ["segment_id", "start_time_seconds", "end_time_seconds", "label"]
            if len(first_row) >= 5:
                header.append("display_name")
            yield first_line_no, {header[i]: first_row[i].strip() for i in range(min(len(header), len(first_row)))}
        else:
            return

        for line_no, row in enumerate(reader, start=first_line_no + 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#") and not any(cell.strip() for cell in row[1:]):
                continue
            yield line_no, {header[i]: row[i].strip() for i in range(min(len(header), len(row)))}


def discover_annotation_files(annotations_dir: Path) -> List[Path]:
    return sorted(
        p for p in annotations_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}
    )


def looks_like_mid_to_display_path(path: Path) -> bool:
    name = path.name.lower()
    return "mid_to_display" in name or "class_labels_indices" in name


def collect_mid_to_display_name(files: Sequence[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in files:
        try:
            delimiter = choose_delimiter(path)
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                first_row = next((row for row in reader if row and any(c.strip() for c in row)), None)
                if first_row is None:
                    continue
                header = [norm_key(c) for c in first_row]
                if "mid" in header and "display_name" in header:
                    mid_idx = header.index("mid")
                    name_idx = header.index("display_name")
                elif looks_like_mid_to_display_path(path) and len(first_row) >= 2:
                    mid_idx = 0
                    name_idx = 1
                    mid = first_row[mid_idx].strip()
                    name = first_row[name_idx].strip()
                    if mid and name:
                        out[mid] = name
                else:
                    continue
                for row in reader:
                    if len(row) <= max(mid_idx, name_idx):
                        continue
                    mid = row[mid_idx].strip()
                    name = row[name_idx].strip()
                    if mid and name:
                        out[mid] = name
        except Exception:
            continue
    return out


def is_mapping_file(path: Path, preview: Sequence[Tuple[int, Dict[str, str]]]) -> bool:
    if looks_like_mid_to_display_path(path):
        return True
    return any("mid" in row and "display_name" in row and "start_time_seconds" not in row for _, row in preview)


def first(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and str(row[key]).strip():
            return row[key]
    return ""


def infer_split(path: Path, train_keywords: Sequence[str], valid_keywords: Sequence[str]) -> str:
    low = "/".join(part.lower() for part in path.parts)
    if any(k.lower() in low for k in valid_keywords):
        return "valid"
    if any(k.lower() in low for k in train_keywords):
        return "train"
    return "train"


def split_label_values(raw: str, column_name: str) -> List[str]:
    raw = str(raw or "").strip().strip('"')
    if not raw:
        return []
    col = norm_key(column_name)
    if col in {"positive_labels", "labels", "mids"} or (raw.startswith("/m/") and "," in raw):
        return [x.strip().strip('"') for x in raw.split(",") if x.strip()]
    return [raw]


def parse_audioset_time_token(token: str) -> float:
    value = float(token)
    return value / 1000.0 if abs(value) >= 1000.0 else value


def parse_segment_offsets(segment_id: str) -> Tuple[str, Optional[float], Optional[float]]:
    stem = Path(str(segment_id)).stem.strip()
    if len(stem) > 12 and stem[11] in "_-":
        ytid = stem[:11]
        suffix = stem[12:]
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[_\-][0-9]+(?:\.[0-9]+)?)?", suffix):
            parts = re.split(r"[_\-]", suffix)
            clip_start = parse_audioset_time_token(parts[0])
            clip_end = parse_audioset_time_token(parts[1]) if len(parts) > 1 else clip_start + AUDIOSET_SEGMENT_DURATION
            return ytid, clip_start, clip_end
    m = re.match(r"^(.+?)[_\-]([0-9]+(?:\.[0-9]+)?)[_\-]([0-9]+(?:\.[0-9]+)?)$", stem)
    if m:
        return m.group(1), parse_audioset_time_token(m.group(2)), parse_audioset_time_token(m.group(3))
    return (stem[:11], None, None) if len(stem) >= 11 else (stem, None, None)


def audio_key_candidates(identifier: str) -> List[Tuple[str, str]]:
    stem = Path(str(identifier)).stem.strip()
    ytid, clip_start, clip_end = parse_segment_offsets(stem)
    out: List[Tuple[str, str]] = []

    def add(k: str, mode: str) -> None:
        pair = (k.strip(), mode)
        if pair[0] and pair not in out:
            out.append(pair)

    add(stem, "segment")
    if clip_start is not None and clip_end is not None:
        add(f"{ytid}_{clip_start:g}_{clip_end:g}", "segment")
        add(f"{ytid}_{int(round(clip_start)):06d}_{int(round(clip_end)):06d}", "segment")
        add(f"{ytid}-{int(round(clip_start)):06d}-{int(round(clip_end)):06d}", "segment")
        add(f"{ytid}_{int(round(clip_start * 1000)):d}", "segment")
        add(f"{ytid}-{int(round(clip_start * 1000)):d}", "segment")
    add(ytid, "ytid")
    if len(stem) >= 11:
        add(stem[:11], "ytid")
    return out


def parse_strong_annotations(
    annotations_dir: Path,
    train_keywords: Sequence[str],
    valid_keywords: Sequence[str],
) -> Tuple[List[StrongEvent], Dict[str, str], Counter]:
    files = discover_annotation_files(annotations_dir)
    mid_to_name = collect_mid_to_display_name(files)
    status = Counter()
    events: List[StrongEvent] = []

    for path in files:
        if "framed_posneg" in path.name.lower():
            status[f"skip_framed_posneg:{path.name}"] += 1
            continue
        try:
            rows = list(read_table(path))
        except Exception as exc:
            status[f"skip_read_error:{path.name}:{type(exc).__name__}"] += 1
            continue
        if not rows:
            status[f"skip_empty:{path.name}"] += 1
            continue
        if is_mapping_file(path, rows[:5]):
            status[f"skip_mapping:{path.name}"] += 1
            continue

        split = infer_split(path, train_keywords, valid_keywords)
        parsed = 0
        for line_no, row in rows:
            present = norm_key(row.get("present", ""))
            if present and present not in {"present", "positive", "true", "yes", "1"}:
                status[f"skip_non_present_frame:{path.name}"] += 1
                continue
            segment_id = first(row, ["segment_id", "ytid", "youtube_id", "video_id", "file_name", "filename", "audio_id", "wav"])
            start = parse_float(first(row, ["start_time_seconds", "start_seconds", "onset_seconds", "onset", "start"]))
            end = parse_float(first(row, ["end_time_seconds", "end_seconds", "offset_seconds", "offset", "end"]))
            label_col, raw_label = "", ""
            for col in ["label", "event_label", "class_label", "mid", "positive_labels", "labels", "display_name"]:
                if col in row and row[col].strip():
                    label_col, raw_label = col, row[col].strip()
                    break
            if not segment_id or start is None or end is None or not raw_label:
                status[f"skip_bad_row:{path.name}"] += 1
                continue
            if end <= start:
                status[f"skip_nonpositive_duration:{path.name}"] += 1
                continue
            for raw in split_label_values(raw_label, label_col):
                display = mid_to_name.get(raw, raw)
                interference_label = resolve_interference_label(display)
                dcase_label = None if interference_label else resolve_dcase_label(display)
                uid_base = f"{path.relative_to(annotations_dir)}:{line_no}:{segment_id}:{start:.6f}:{end:.6f}:{raw}"
                uid = hashlib.sha1(uid_base.encode("utf-8")).hexdigest()[:16]
                ytid, _, _ = parse_segment_offsets(segment_id)
                events.append(StrongEvent(uid, Path(str(segment_id)).stem.strip(), ytid, float(start), float(end),
                                          raw, display, dcase_label, interference_label, split,
                                          str(path.relative_to(annotations_dir)), line_no))
                parsed += 1
        status[f"parsed:{path.name}"] += parsed
    return events, mid_to_name, status


def overlaps(a0: float, a1: float, b0: float, b1: float, margin: float) -> bool:
    return max(a0 - margin, b0) < min(a1 + margin, b1)


def build_candidates(
    events: Sequence[StrongEvent],
    min_event_duration: float,
    max_event_duration: float,
    min_noise_duration: float,
    default_clip_duration: float,
    overlap_margin: float,
    keep_noise_gaps: bool,
    max_background_per_clip: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    by_segment: Dict[Tuple[str, str], List[StrongEvent]] = defaultdict(list)
    for e in events:
        by_segment[(e.split, e.segment_id)].append(e)

    accepted: List[Candidate] = []
    rejected: List[Candidate] = []

    for (split, seg), evs in by_segment.items():
        evs = sorted(evs, key=lambda x: (x.start, x.end, x.display_label))
        for i, e in enumerate(evs):
            reason = ""
            if e.duration < min_event_duration:
                reason = "too_short"
            elif e.duration > max_event_duration:
                reason = "too_long"
            elif any(i != j and overlaps(e.start, e.end, o.start, o.end, overlap_margin) for j, o in enumerate(evs)):
                reason = "overlap_with_other_annotated_event"

            label = e.dcase_label or e.interference_label or e.display_label
            if reason:
                rejected.append(Candidate(e.uid, "reject", split, label, seg, e.start, e.end, e.duration,
                                          e.annotation_file, e.raw_label, e.display_label, reason))
            elif e.dcase_label:
                accepted.append(Candidate(e.uid, "sound_event", split, e.dcase_label, seg, e.start, e.end,
                                          e.duration, e.annotation_file, e.raw_label, e.display_label))
            elif e.interference_label:
                accepted.append(Candidate(e.uid, "interference", split, e.interference_label, seg, e.start, e.end,
                                          e.duration, e.annotation_file, e.raw_label, e.display_label))
            else:
                rejected.append(Candidate(e.uid, "reject", split, e.display_label, seg, e.start, e.end,
                                          e.duration, e.annotation_file, e.raw_label, e.display_label,
                                          "unmapped_not_dcase_target_or_interference"))

        if keep_noise_gaps:
            _, clip_start, clip_end = parse_segment_offsets(seg)
            clip_duration = default_clip_duration
            if clip_start is not None and clip_end is not None and clip_end > clip_start:
                clip_duration = clip_end - clip_start
            if evs:
                clip_duration = max(clip_duration, max(e.end for e in evs))
            merged: List[Tuple[float, float]] = []
            for e in evs:
                s = max(0.0, e.start - overlap_margin)
                t = min(clip_duration, e.end + overlap_margin)
                if not merged or s > merged[-1][1]:
                    merged.append((s, t))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], t))
            cursor = 0.0
            gaps: List[Tuple[float, float]] = []
            for s, t in merged:
                if s - cursor >= min_noise_duration:
                    gaps.append((cursor, s))
                cursor = max(cursor, t)
            if clip_duration - cursor >= min_noise_duration:
                gaps.append((cursor, clip_duration))
            for k, (s, t) in enumerate(sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)[:max_background_per_clip]):
                uid = hashlib.sha1(f"noise:{split}:{seg}:{s:.6f}:{t:.6f}:{k}".encode()).hexdigest()[:16]
                accepted.append(Candidate(uid, "noise", split, "noise", seg, s, t, t - s,
                                          "<derived_from_strong_gaps>", "", "noise"))
    return accepted, rejected


def create_compatible_dirs(output_dir: Path) -> None:
    for split in ("train", "valid"):
        for label in DCASE2026_LABELS:
            (output_dir / "sound_event" / split / label).mkdir(parents=True, exist_ok=True)
        for label in DCASE_INTERFERENCE_LABELS:
            (output_dir / "interference" / split / label).mkdir(parents=True, exist_ok=True)
        (output_dir / "noise" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "valid").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "audioset_strong").mkdir(parents=True, exist_ok=True)


def summarize(events: Sequence[StrongEvent], accepted: Sequence[Candidate], rejected: Sequence[Candidate],
              parse_status: Counter, top_k: int) -> Dict[str, object]:
    duration_by_kind = defaultdict(float)
    duration_by_kind_label = defaultdict(float)
    count_by_kind_label: Dict[str, Counter] = defaultdict(Counter)
    for c in accepted:
        duration_by_kind[c.kind] += c.duration
        duration_by_kind_label[(c.kind, c.label)] += c.duration
        count_by_kind_label[c.kind][c.label] += 1
    labels: Dict[str, List[Dict[str, object]]] = {}
    for kind, counter in sorted(count_by_kind_label.items()):
        limit = top_k if kind == "interference" else None
        labels[kind] = [
            {"label": label, "count": count, "duration_hours": round(duration_by_kind_label[(kind, label)] / 3600.0, 4)}
            for label, count in counter.most_common(limit)
        ]
    return {
        "target_sr": TARGET_SR,
        "parse_status": dict(parse_status),
        "n_annotation_events": len(events),
        "n_candidates": len(accepted),
        "n_rejected": len(rejected),
        "accepted_by_kind": dict(Counter(c.kind for c in accepted)),
        "accepted_by_split": dict(Counter(c.split for c in accepted)),
        "rejected_by_reason": dict(Counter(c.reject_reason for c in rejected)),
        "duration_hours_by_kind": {k: round(v / 3600.0, 4) for k, v in sorted(duration_by_kind.items())},
        "labels": labels,
    }


def print_summary(summary: Dict[str, object]) -> None:
    print("\n================ AudioSet-Strong annotation statistics ================")
    print(f"Output sample rate       : {summary['target_sr']} Hz")
    print(f"Annotation events parsed : {summary['n_annotation_events']}")
    print(f"Accepted candidates      : {summary['n_candidates']}")
    print(f"Rejected annotations     : {summary['n_rejected']}")
    print(f"Accepted by kind         : {summary['accepted_by_kind']}")
    print(f"Accepted by split        : {summary['accepted_by_split']}")
    print(f"Duration hours by kind   : {summary['duration_hours_by_kind']}")
    print(f"Rejected by reason       : {summary['rejected_by_reason']}")
    for kind, rows in summary.get("labels", {}).items():
        print(f"  [{kind}]")
        for row in rows:
            print(f"    {row['label']}: count={row['count']}, hours={row['duration_hours']}")
    print("======================================================================\n")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_files(output_dir: Path, summary: Dict[str, object], accepted: Sequence[Candidate],
                        rejected: Sequence[Candidate]) -> None:
    meta = output_dir / "metadata" / "audioset_strong"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_jsonl(meta / "accepted_candidates.jsonl", accepted)
    write_jsonl(meta / "rejected_annotations.jsonl", rejected)


def build_audio_index(input_dir: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    index: Dict[str, Path] = {}
    for path in tqdm([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts],
                     desc="Index raw audio", unit="file"):
        for key, _mode in audio_key_candidates(path.stem):
            index.setdefault(key, path)
    print(f"Indexed {len(index)} audio lookup keys.")
    return index


def resolve_audio(candidate: Candidate, audio_index: Dict[str, Path]) -> Tuple[Optional[Path], float]:
    ytid, clip_start, _ = parse_segment_offsets(candidate.segment_id)
    for key, mode in audio_key_candidates(candidate.segment_id):
        if key in audio_index:
            offset = float(clip_start) if mode == "ytid" and key == ytid and clip_start is not None else 0.0
            return audio_index[key], offset
    return None, 0.0


def load_audio_segment(path: Path, start: float, end: float, mono: bool) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError(f"soundfile import failed: {_SF_ERROR}")
    with sf.SoundFile(str(path), "r") as f:
        sr = int(f.samplerate)
        start_frame = max(0, int(round(start * sr)))
        frames = max(1, int(round((end - start) * sr)))
        f.seek(min(start_frame, len(f)))
        audio = f.read(frames=frames, dtype="float32", always_2d=True).T
    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True).astype(np.float32)
    return audio.astype(np.float32, copy=False), sr


def trim_edges(audio: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0 or audio.size == 0:
        return audio
    m = np.max(np.abs(audio), axis=0)
    idx = np.flatnonzero(m >= threshold)
    return audio[:, idx[0]:idx[-1] + 1] if idx.size else audio[:, :0]


def resample_to_32k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    if librosa is None:
        raise RuntimeError(f"librosa is required for resampling: {_LIBROSA_ERROR}")
    return np.stack([librosa.resample(ch, orig_sr=sr, target_sr=TARGET_SR) for ch in audio], axis=0).astype(np.float32)


def output_path(output_dir: Path, c: Candidate) -> Path:
    name = (f"{safe_filename_part(c.segment_id, 80)}__{int(round(c.start * 1000)):08d}_"
            f"{int(round(c.end * 1000)):08d}__{safe_filename_part(c.display_label or c.label, 48)}__{c.uid}.wav")
    if c.kind == "noise":
        return output_dir / "noise" / c.split / name
    return output_dir / c.kind / c.split / c.label / name


def save_wav(path: Path, audio: np.ndarray) -> None:
    if sf is None:
        raise RuntimeError(f"soundfile import failed: {_SF_ERROR}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio[0] if audio.shape[0] == 1 else audio.T, TARGET_SR)


def extract(candidates: Sequence[Candidate], input_dir: Path, output_dir: Path, args: argparse.Namespace) -> List[ExtractResult]:
    create_compatible_dirs(output_dir)
    audio_index = build_audio_index(input_dir, args.audio_extensions)
    results: List[ExtractResult] = []
    for c in tqdm(candidates, desc="Extract candidates", unit="seg"):
        src, offset = resolve_audio(c, audio_index)
        dst = output_path(output_dir, c)
        if src is None:
            results.append(ExtractResult(c.uid, c.kind, c.split, c.label, "", str(dst), c.start, c.end, c.duration,
                                         "skip", "missing_raw_audio"))
            continue
        if dst.exists() and not args.overwrite:
            results.append(ExtractResult(c.uid, c.kind, c.split, c.label, str(src), str(dst), c.start, c.end,
                                         c.duration, "skip", "output_exists"))
            continue
        try:
            audio, sr = load_audio_segment(src, c.start + offset, c.end + offset, args.mono)
            if args.trim_edges:
                audio = trim_edges(audio, args.amp_threshold)
            if audio.size == 0:
                raise ValueError("empty_after_trim")
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if peak < args.amp_threshold:
                raise ValueError(f"low_peak:{peak:.6g}")
            if rms < args.min_rms:
                raise ValueError(f"low_rms:{rms:.6g}")
            audio = resample_to_32k(audio, sr)
            save_wav(dst, audio)
            results.append(ExtractResult(c.uid, c.kind, c.split, c.label, str(src), str(dst), c.start, c.end,
                                         c.duration, "saved", "", source_sr=sr, peak=peak, rms=rms))
        except Exception as exc:
            results.append(ExtractResult(c.uid, c.kind, c.split, c.label, str(src), str(dst), c.start, c.end,
                                         c.duration, "skip", f"{type(exc).__name__}:{exc}"))
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract clean AudioSet-Strong clips for DCASE2026 Task4.")
    p.add_argument("--annotations_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--input_dir", type=Path, default=None,
                   help="Raw AudioSet audio root. Required only with --execute; not used for statistics-only dry run.")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target_sr", type=int, default=TARGET_SR, choices=[TARGET_SR])
    p.add_argument("--min_event_duration", type=float, default=0.15)
    p.add_argument("--max_event_duration", type=float, default=10.0)
    p.add_argument("--min_noise_duration", type=float, default=1.0)
    p.add_argument("--default_clip_duration", type=float, default=10.0)
    p.add_argument("--overlap_margin", type=float, default=0.03)
    p.add_argument("--max_background_per_clip", type=int, default=2)
    p.add_argument("--keep_noise_gaps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trim_edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp_threshold", type=float, default=0.005)
    p.add_argument("--min_rms", type=float, default=1e-4)
    p.add_argument("--audio_extensions", nargs="+", default=list(AUDIO_EXTENSIONS))
    p.add_argument("--train_keywords", nargs="+", default=["train", "balanced", "unbalanced"])
    p.add_argument("--valid_keywords", nargs="+", default=["valid", "validation", "eval", "evaluate", "test"])
    p.add_argument("--top_k_interference_stats", type=int, default=40)
    p.add_argument("--no_write_manifests", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.annotations_dir.is_dir():
        raise FileNotFoundError(f"annotations_dir does not exist: {args.annotations_dir}")
    if args.execute and (args.input_dir is None or not args.input_dir.is_dir()):
        raise FileNotFoundError("--input_dir is required and must exist when --execute is used")

    print("====================================================")
    print("AudioSet-Strong -> DCASE2026 Task4")
    print(f"annotations_dir : {args.annotations_dir}")
    print(f"output_dir      : {args.output_dir}")
    print(f"input_dir       : {args.input_dir if args.input_dir else '<not needed for statistics>'}")
    print(f"execute         : {args.execute}")
    print(f"target_sr       : {TARGET_SR}")
    print("====================================================")

    create_compatible_dirs(args.output_dir)
    events, mid_to_name, status = parse_strong_annotations(args.annotations_dir, args.train_keywords, args.valid_keywords)
    print(f"Resolved {len(mid_to_name)} AudioSet MID -> display-name mappings.")

    accepted, rejected = build_candidates(
        events, args.min_event_duration, args.max_event_duration, args.min_noise_duration,
        args.default_clip_duration, args.overlap_margin, args.keep_noise_gaps, args.max_background_per_clip,
    )
    summary = summarize(events, accepted, rejected, status, args.top_k_interference_stats)
    print_summary(summary)

    if not args.no_write_manifests:
        write_summary_files(args.output_dir, summary, accepted, rejected)
        print(f"Wrote manifests to: {args.output_dir / 'metadata' / 'audioset_strong'}")

    if not args.execute:
        print("Statistics-only dry run complete. No audio was read or extracted.")
        return 0

    results = extract(accepted, args.input_dir, args.output_dir, args)
    print("\n================ Audio extraction summary ================")
    print(f"Output sample rate : {TARGET_SR} Hz")
    print(f"Segments processed : {len(results)}")
    print(f"By status          : {dict(Counter(r.status for r in results))}")
    print(f"Skip reasons       : {dict(Counter(r.reason for r in results if r.status != 'saved'))}")
    print(f"Saved by kind      : {dict(Counter(r.kind for r in results if r.status == 'saved'))}")
    print("==========================================================\n")

    if not args.no_write_manifests:
        meta = args.output_dir / "metadata" / "audioset_strong"
        write_jsonl(meta / "extraction_results.jsonl", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
