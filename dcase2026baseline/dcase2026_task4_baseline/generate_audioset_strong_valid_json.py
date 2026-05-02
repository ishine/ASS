#!/usr/bin/env python3
"""
Generate DatasetS3 metadata-list validation files from extracted AudioSet-Strong WAVs.

The output is compatible with configs that use:

  mode: metadata
  metadata_list: data/dev_set/metadata/<name>.json

Each list entry points to one SpAudSyn-style metadata JSON containing fixed
foreground/interference/background source selections from the valid split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DCASE2026_LABELS = [
    "AlarmClock", "BicycleBell", "Blender", "Buzzer",
    "Clapping", "Cough", "CupboardOpenClose", "Dishes",
    "Doorbell", "FootSteps", "HairDryer", "MechanicalFans",
    "MusicalKeyboard", "Percussion", "Pour", "Speech",
    "Typing", "VacuumCleaner",
]


@dataclass(frozen=True)
class WavInfo:
    path: Path
    label: str
    duration: float


def iter_label_wavs(root: Path) -> Dict[str, List[WavInfo]]:
    out: Dict[str, List[WavInfo]] = {}
    if not root.exists():
        return out
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        infos = []
        for path in sorted(label_dir.rglob("*.wav")):
            dur = wav_duration(path)
            if dur > 0:
                infos.append(WavInfo(path, label_dir.name, dur))
        if infos:
            out[label_dir.name] = infos
    return out


def iter_noise_wavs(root: Path) -> List[WavInfo]:
    if not root.exists():
        return []
    return [WavInfo(path, "noise", wav_duration(path)) for path in sorted(root.rglob("*.wav")) if wav_duration(path) > 0]


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as f:
            sr = f.getframerate()
            return f.getnframes() / float(sr) if sr else 0.0
    except Exception:
        return 0.0


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def choose_event_time(rng: random.Random, duration: float, event_duration: float, existing: Sequence[Dict], max_overlap: int) -> float:
    event_duration = min(event_duration, duration)
    if event_duration >= duration:
        candidate = {"event_time": 0.0, "event_duration": event_duration}
        if max_simultaneous_events([*existing, candidate]) <= max_overlap:
            return 0.0
        raise RuntimeError(f"No valid event time for full-duration event with max_overlap={max_overlap}.")

    for _ in range(2000):
        start = rng.uniform(0.0, duration - event_duration)
        candidate = {"event_time": start, "event_duration": event_duration}
        if max_simultaneous_events([*existing, candidate]) <= max_overlap:
            return start
    raise RuntimeError(
        f"No valid event time for duration={event_duration:.3f}s with max_overlap={max_overlap}."
    )


def make_scene_plan(num_scenes: int, rng: random.Random) -> List[Dict[str, object]]:
    if num_scenes % 6 != 0:
        raise ValueError("--num_scenes must be divisible by 6 to match DCASE 0/1/2/3-target proportions exactly")
    n_zero = num_scenes // 6
    n_one = num_scenes // 6
    n_two = num_scenes // 3
    n_three = num_scenes // 3
    if n_two % 2 or n_three % 2:
        raise ValueError("--num_scenes must make the 2-target and 3-target subsets even for exact 50% same-class scenes")

    plan: List[Dict[str, object]] = []
    plan.extend({"n_targets": 0, "same_class": False} for _ in range(n_zero))
    plan.extend({"n_targets": 1, "same_class": False} for _ in range(n_one))
    plan.extend({"n_targets": 2, "same_class": i < n_two // 2} for i in range(n_two))
    plan.extend({"n_targets": 3, "same_class": i < n_three // 2} for i in range(n_three))
    rng.shuffle(plan)
    return plan


def choose_target_labels(
    rng: random.Random,
    labels: Sequence[str],
    n_targets: int,
    same_class: bool,
) -> List[str]:
    if n_targets == 0:
        return []
    if n_targets == 1:
        return [rng.choice(list(labels))]
    if same_class:
        dup = rng.choice(list(labels))
        if n_targets == 2:
            return [dup, dup]
        other_pool = [label for label in labels if label != dup]
        if not other_pool:
            raise RuntimeError("Need at least two available labels for 3-target same-class scenes")
        return [dup, dup, rng.choice(other_pool)]
    if len(labels) < n_targets:
        raise RuntimeError(f"Need at least {n_targets} available foreground labels")
    return rng.sample(list(labels), k=n_targets)


def angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(x) * float(x) for x in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (na * nb)))
    return math.degrees(math.acos(cos_theta))


def max_simultaneous_events(events: Sequence[Dict]) -> int:
    points: List[Tuple[float, int]] = []
    for event in events:
        start = float(event["event_time"])
        end = start + float(event["event_duration"])
        points.append((start, +1))
        points.append((end, -1))
    points.sort(key=lambda x: (x[0], x[1]))
    current = 0
    max_seen = 0
    for _time, delta in points:
        current += delta
        max_seen = max(max_seen, current)
    return max_seen


def load_room_infos(args: argparse.Namespace) -> List[Dict[str, object]]:
    room_positions = None
    if args.room_positions_json:
        room_positions = json.loads(args.room_positions_json.read_text(encoding="utf-8"))
        if not isinstance(room_positions, list) or not room_positions:
            raise ValueError("--room_positions_json must contain a non-empty JSON list of [x, y, z] positions")

    if args.room_template_list:
        infos = load_room_infos_from_template_list(args.room_template_list)
        if room_positions is not None:
            for info in infos:
                info["_positions"] = room_positions
        return infos

    if args.room_metadata_json:
        data = json.loads(args.room_metadata_json.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "room_infos" in data:
            infos = list(data["room_infos"])
        elif isinstance(data, list):
            infos = data
        elif isinstance(data, dict) and "sofa_path" in data:
            infos = [data]
        else:
            raise ValueError("--room_metadata_json must contain a room_info dict, a list, or {'room_infos': [...]}")
        for info in infos:
            if room_positions is not None:
                info["_positions"] = room_positions
            elif "positions" in info:
                info["_positions"] = info["positions"]
            elif not args.allow_dummy_positions:
                raise ValueError(
                    "Room metadata does not include valid SOFA positions. Provide --room_positions_json, "
                    "include 'positions' in --room_metadata_json, or pass --allow_dummy_positions only for smoke tests."
                )
        return infos

    try:
        import sofa  # type: ignore
    except Exception:
        sofa = None

    def inspect_sofa(path: Path) -> Dict[str, object]:
        if sofa is None:
            if args.room_nchan is None or args.room_nrir is None or args.room_rir_len is None:
                raise ValueError(
                    "SOFA inspection needs the 'sofa' package. Install it, or provide --room_nchan, "
                    "--room_nrir, --room_rir_len, and --room_positions_json."
                )
            if room_positions is None and not args.allow_dummy_positions:
                raise ValueError("Provide --room_positions_json when SOFA inspection is unavailable.")
            return {
                "sofa_path": str(path),
                "sr": args.room_sr,
                "nchan": args.room_nchan,
                "nrir": args.room_nrir,
                "rir_len": args.room_rir_len,
                "direct_range_ms": args.direct_range_ms,
                "_positions": room_positions,
            }
        sofafile = sofa.Database.open(str(path), mode="r", parallel=False)
        dims = sofafile.Data.IR.dimensions()
        shape = sofafile.Data.IR.shape
        dim_sizes = dict(zip(dims, shape))
        sofa_sr = int(sofafile.Data.SamplingRate.get_values()[0])
        positions = sofafile.Source.Position.get_values(system="cartesian").tolist()
        sofafile.close()
        return {
            "sofa_path": str(path),
            "sr": sofa_sr,
            "nchan": dim_sizes["R"],
            "nrir": dim_sizes["M"],
            "rir_len": dim_sizes["N"],
            "direct_range_ms": args.direct_range_ms,
            "_positions": positions,
        }

    sofa_files = sorted(args.room_ir_dir.glob("*.sofa")) if args.room_ir_dir and args.room_ir_dir.exists() else []
    if not sofa_files:
        raise FileNotFoundError(
            "No room metadata source found. Provide --room_metadata_json or a --room_ir_dir containing .sofa files."
        )
    return [inspect_sofa(path) for path in sofa_files]


def load_room_infos_from_template_list(metadata_list: Path) -> List[Dict[str, object]]:
    metadata_dir = metadata_list.parent
    rows = json.loads(metadata_list.read_text(encoding="utf-8"))
    room_infos: Dict[str, Dict[str, object]] = {}
    positions: Dict[str, set] = {}
    for row in rows:
        meta_path = metadata_dir / row["metadata_path"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        room_info = dict(meta["room"]["args"]["metadata"])
        sofa_path = str(room_info["sofa_path"])
        room_infos.setdefault(sofa_path, room_info)
        positions.setdefault(sofa_path, set())
        for event in meta.get("fg_events", []) + meta.get("int_events", []):
            for pos in event.get("event_position") or []:
                positions[sofa_path].add(tuple(round(float(x), 10) for x in pos))

    out = []
    for sofa_path, info in sorted(room_infos.items()):
        grid = [list(pos) for pos in sorted(positions.get(sofa_path, []))]
        if not grid:
            raise ValueError(f"No event positions found for room template {sofa_path}")
        info["_positions"] = grid
        out.append(info)
    return out


def choose_position(
    rng: random.Random,
    room_info: Dict[str, object],
    label: Optional[str] = None,
    existing_events: Sequence[Dict] = (),
    min_same_class_angle: float = 60.0,
) -> List[List[float]]:
    positions = room_info.get("_positions")
    if isinstance(positions, list) and positions:
        candidates = list(positions)
        rng.shuffle(candidates)
        for pos in candidates:
            if label is None:
                return [pos]
            same_class_positions = [
                existing["event_position"][0]
                for existing in existing_events
                if existing.get("role") == "foreground"
                and existing.get("label") == label
                and existing.get("event_position")
            ]
            if all(angle_degrees(pos, other) >= min_same_class_angle for other in same_class_positions):
                return [pos]
        raise RuntimeError(f"No room position satisfies {min_same_class_angle} degree same-class separation for {label}")
    angle = rng.uniform(0.0, 2.0 * math.pi)
    z = rng.uniform(-0.2, 0.2)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return [[r * math.cos(angle), r * math.sin(angle), z]]


def make_event(
    rng: random.Random,
    item: WavInfo,
    role: str,
    source_root: Path,
    scene_duration: float,
    max_event_dur: float,
    existing_events: Sequence[Dict],
    max_event_overlap: int,
    snr_range: Sequence[float],
    room_info: Dict[str, object],
    all_scene_events: Sequence[Dict],
    min_same_class_angle: float,
) -> Dict[str, object]:
    event_duration = min(item.duration, max_event_dur, scene_duration)
    max_source_start = max(0.0, item.duration - event_duration)
    source_time = rng.uniform(0.0, max_source_start) if max_source_start > 1e-6 else 0.0
    event_time = choose_event_time(rng, scene_duration, event_duration, all_scene_events, max_event_overlap)
    event_position = choose_position(
        rng,
        room_info,
        label=item.label if role == "foreground" else None,
        existing_events=existing_events,
        min_same_class_angle=min_same_class_angle,
    )
    return {
        "label": item.label,
        "source_file": rel_or_abs(item.path, source_root),
        "source_time": source_time,
        "event_time": event_time,
        "event_duration": event_duration,
        "event_position": event_position,
        "snr": rng.uniform(float(snr_range[0]), float(snr_range[1])),
        "role": role,
    }


def make_background(item: WavInfo, background_root: Path, scene_duration: float) -> Dict[str, object]:
    max_source_start = max(0.0, item.duration - scene_duration)
    return {
        "label": None,
        "source_file": rel_or_abs(item.path, background_root),
        "source_time": 0.0 if max_source_start <= 1e-6 else 0.0,
        "event_time": 0,
        "event_duration": scene_duration,
        "event_position": None,
        "snr": 0,
        "role": "background",
    }


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def materialize_metadata(metadata: Dict[str, object], backend: str) -> Dict[str, object]:
    if backend == "direct":
        return metadata
    if backend != "spaudsyn":
        raise ValueError(f"Unknown metadata backend: {backend}")
    from src.modules.spatial_audio_synthesizer.spatial_audio_synthesizer import SpAudSyn

    return SpAudSyn.from_metadata(metadata).generate_metadata()


def parse_range(values: Sequence[str], cast=float) -> List:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range arguments must have exactly two values")
    return [cast(values[0]), cast(values[1])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AudioSet-Strong validation metadata JSON.")
    parser.add_argument("--data_root", type=Path, default=Path("data/dev_set"))
    parser.add_argument("--foreground_dir", type=Path, default=None)
    parser.add_argument("--interference_dir", type=Path, default=None)
    parser.add_argument("--background_dir", type=Path, default=None)
    parser.add_argument("--room_ir_dir", type=Path, default=Path("data/dev_set/room_ir/valid"))
    parser.add_argument(
        "--room_template_list",
        type=Path,
        default=None,
        help="Existing metadata list, e.g. data/dev_set/metadata/valid.json, used to reuse room metadata and positions.",
    )
    parser.add_argument("--room_metadata_json", type=Path, default=None)
    parser.add_argument("--room_positions_json", type=Path, default=None)
    parser.add_argument("--valid_json", type=Path, default=Path("data/dev_set/metadata/audioset_strong_valid.json"))
    parser.add_argument("--metadata_subdir", type=str, default="audioset_strong_valid")
    parser.add_argument("--num_scenes", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--n_sources", type=int, default=3)
    parser.add_argument("--nevent_min", type=int, default=1)
    parser.add_argument("--nevent_max", type=int, default=3)
    parser.add_argument("--ninterference_min", type=int, default=0)
    parser.add_argument("--ninterference_max", type=int, default=2)
    parser.add_argument("--max_event_overlap", type=int, default=3)
    parser.add_argument("--max_event_dur", type=float, default=10.0)
    parser.add_argument("--min_same_class_angle", type=float, default=60.0)
    parser.add_argument("--max_event_attempts", type=int, default=100)
    parser.add_argument("--ref_db", type=float, default=-55.0)
    parser.add_argument("--snr_range", nargs=2, default=[5.0, 20.0], type=float)
    parser.add_argument("--interference_snr_range", nargs=2, default=[0.0, 15.0], type=float)
    parser.add_argument("--direct_range_ms", nargs=2, default=[6, 50], type=int)
    parser.add_argument("--room_sr", type=int, default=48000)
    parser.add_argument("--room_nchan", type=int, default=None)
    parser.add_argument("--room_nrir", type=int, default=None)
    parser.add_argument("--room_rir_len", type=int, default=None)
    parser.add_argument("--allow_dummy_positions", action="store_true")
    parser.add_argument("--include_background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata_backend", choices=["spaudsyn", "direct"], default="spaudsyn")
    parser.add_argument("--execute", action="store_true", help="Write metadata files. Omit for dry run.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    foreground_dir = args.foreground_dir or (args.data_root / "sound_event" / "valid")
    interference_dir = args.interference_dir or (args.data_root / "interference" / "valid")
    background_dir = args.background_dir or (args.data_root / "noise" / "valid")
    metadata_dir = args.valid_json.parent / args.metadata_subdir

    fg_by_label = iter_label_wavs(foreground_dir)
    int_by_label = iter_label_wavs(interference_dir)
    bg_files = iter_noise_wavs(background_dir)
    room_infos = load_room_infos(args)

    available_fg_labels = [label for label in DCASE2026_LABELS if fg_by_label.get(label)]
    if not available_fg_labels:
        raise RuntimeError(f"No foreground validation wavs found under {foreground_dir}")
    if args.n_sources < 3:
        raise ValueError("--n_sources must be at least 3 for DCASE validation/test-style metadata")
    available_int_labels = sorted(int_by_label)

    rng = random.Random(args.seed)
    scene_plan = make_scene_plan(args.num_scenes, rng)
    entries = []
    label_counts = Counter()
    int_counts = Counter()
    skipped_interference = 0
    bg_count = 0
    target_count_scene_counts = Counter()
    same_class_subset_counts = Counter()
    max_overlap_counts = Counter()
    same_class_angle_violations = 0
    for idx, scene_spec in enumerate(scene_plan):
        room_info = rng.choice(room_infos)
        room_metadata = {k: v for k, v in room_info.items() if k not in {"_positions", "positions"}}
        n_targets = int(scene_spec["n_targets"])
        same_class = bool(scene_spec["same_class"])
        labels = choose_target_labels(rng, available_fg_labels, n_targets, same_class)
        rng.shuffle(labels)
        target_count_scene_counts[n_targets] += 1
        if n_targets in (2, 3):
            same_class_subset_counts[(n_targets, same_class)] += 1

        fg_events = []
        for label in labels:
            event = None
            last_error: Optional[Exception] = None
            for _attempt in range(args.max_event_attempts):
                item = rng.choice(fg_by_label[label])
                try:
                    event = make_event(
                        rng, item, "foreground", foreground_dir, args.duration, args.max_event_dur,
                        fg_events, args.max_event_overlap, args.snr_range, room_info,
                        fg_events, args.min_same_class_angle,
                    )
                    break
                except RuntimeError as exc:
                    last_error = exc
            if event is None:
                raise RuntimeError(f"Could not place foreground event {label} in scene {idx}: {last_error}")
            fg_events.append(event)
            label_counts[label] += 1

        int_events = []
        if available_int_labels and args.ninterference_max > 0:
            nint = rng.randint(args.ninterference_min, min(args.ninterference_max, len(available_int_labels)))
            for label in rng.sample(available_int_labels, k=nint):
                event = None
                for _attempt in range(args.max_event_attempts):
                    item = rng.choice(int_by_label[label])
                    try:
                        event = make_event(
                            rng, item, "interference", interference_dir, args.duration, args.max_event_dur,
                            int_events, args.max_event_overlap, args.interference_snr_range, room_info,
                            fg_events + int_events, args.min_same_class_angle,
                        )
                        break
                    except RuntimeError:
                        continue
                if event is None:
                    skipped_interference += 1
                    continue
                int_events.append(event)
                int_counts[label] += 1

        all_events = fg_events + int_events
        max_overlap = max_simultaneous_events(all_events)
        if max_overlap > args.max_event_overlap:
            raise RuntimeError(f"Scene {idx} exceeds max overlap: {max_overlap} > {args.max_event_overlap}")
        max_overlap_counts[max_overlap] += 1
        by_label: Dict[str, List[Dict]] = {}
        for event in fg_events:
            by_label.setdefault(event["label"], []).append(event)
        for same_label, events in by_label.items():
            if len(events) < 2:
                continue
            for i, event_a in enumerate(events):
                for event_b in events[i + 1:]:
                    angle = angle_degrees(event_a["event_position"][0], event_b["event_position"][0])
                    if angle + 1e-6 < args.min_same_class_angle:
                        same_class_angle_violations += 1

        bg_events = []
        if args.include_background and bg_files:
            bg_events.append(make_background(rng.choice(bg_files), background_dir, args.duration))
            bg_count += 1

        rel_meta = f"{args.metadata_subdir}/scene_{idx:06d}.json"
        metadata = {
            "config": {
                "duration": args.duration,
                "sr": args.sr,
                "max_event_overlap": args.max_event_overlap,
                "max_event_dur": args.max_event_dur,
                "ref_db": args.ref_db,
                "foreground_dir": str(foreground_dir),
                "background_dir": str(background_dir) if background_dir else None,
                "interference_dir": str(interference_dir) if interference_dir else None,
                "room_config": {
                    "module": "src.modules.spatial_audio_synthesizer.room",
                    "main": "SofaRoom",
                    "args": {"path": str(args.room_ir_dir), "direct_range_ms": args.direct_range_ms},
                },
                "verbose": False,
            },
            "fg_events": sorted(fg_events, key=lambda x: x["event_time"]),
            "bg_events": bg_events,
            "int_events": sorted(int_events, key=lambda x: x["event_time"]),
            "room": {
                "module": "src.modules.spatial_audio_synthesizer.room",
                "main": "SofaRoom.from_metadata",
                "args": {"metadata": room_metadata},
            },
        }
        metadata = materialize_metadata(metadata, args.metadata_backend)
        entries.append({"metadata_path": rel_meta, "labels": labels})
        if args.execute:
            meta_path = args.valid_json.parent / rel_meta
            if meta_path.exists() and not args.overwrite:
                raise FileExistsError(f"metadata exists, pass --overwrite: {meta_path}")
            write_json(meta_path, metadata)

    if args.execute:
        if args.valid_json.exists() and not args.overwrite:
            raise FileExistsError(f"valid_json exists, pass --overwrite: {args.valid_json}")
        write_json(args.valid_json, [{"metadata_path": e["metadata_path"]} for e in entries])
        summary_path = metadata_dir / "summary.json"
        write_json(
            summary_path,
            {
                "valid_json": str(args.valid_json),
                "metadata_dir": str(metadata_dir),
                "num_scenes": args.num_scenes,
                "target_count_scene_counts": {str(k): v for k, v in sorted(target_count_scene_counts.items())},
                "same_class_subset_counts": {f"{k[0]}_targets_same_class_{k[1]}": v for k, v in sorted(same_class_subset_counts.items())},
                "max_overlap_counts": {str(k): v for k, v in sorted(max_overlap_counts.items())},
                "same_class_angle_violations": same_class_angle_violations,
                "min_same_class_angle": args.min_same_class_angle,
                "foreground_events": dict(label_counts),
                "interference_events": dict(int_counts),
                "skipped_interference": skipped_interference,
                "background_events": bg_count,
                "foreground_dir": str(foreground_dir),
                "interference_dir": str(interference_dir),
                "background_dir": str(background_dir),
                "room_ir_dir": str(args.room_ir_dir),
                "metadata_backend": args.metadata_backend,
            },
        )

    print("====================================================")
    print("AudioSet-Strong validation metadata generation")
    print(f"foreground_dir : {foreground_dir}")
    print(f"interference_dir: {interference_dir}")
    print(f"background_dir : {background_dir}")
    print(f"room sources   : {len(room_infos)}")
    print(f"valid_json     : {args.valid_json}")
    print(f"metadata_subdir: {args.metadata_subdir}")
    print(f"num_scenes     : {args.num_scenes}")
    print(f"metadata_backend: {args.metadata_backend}")
    print(f"execute        : {args.execute}")
    print("====================================================")
    print(f"Available foreground labels: {len(available_fg_labels)}")
    print(f"Target-count scene distribution: {dict(sorted(target_count_scene_counts.items()))}")
    print(f"Same-class subset distribution: {dict(sorted(same_class_subset_counts.items()))}")
    print(f"Max-overlap distribution: {dict(sorted(max_overlap_counts.items()))}")
    print(f"Same-class angle violations: {same_class_angle_violations}")
    print(f"Foreground events: {sum(label_counts.values())} {dict(label_counts)}")
    print(f"Interference events: {sum(int_counts.values())}")
    print(f"Skipped interference events: {skipped_interference}")
    print(f"Background events: {bg_count}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to write metadata files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
