#!/usr/bin/env python3
"""
Adjust train/valid split for extracted AudioSet-Strong dry source WAVs.

This operates on the DCASE-compatible folder layout produced by
add_audioset_strong.py:

  <data_root>/sound_event/{train,valid}/<label>/*.wav
  <data_root>/interference/{train,valid}/<label>/*.wav
  <data_root>/noise/{train,valid}/*.wav

Default mode is a dry run. Pass --execute to move files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


AUDIO_KINDS = ("sound_event", "interference", "noise")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    kind: str
    split: str
    label: str


@dataclass(frozen=True)
class PlannedAction:
    action: str
    kind: str
    label: str
    src_split: str
    dst_split: str
    src: str
    dst: str


def stable_score(path: Path, seed: int) -> str:
    key = f"{seed}:{path.as_posix()}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()


def iter_wavs_from_tree(data_root: Path, kinds: Sequence[str]) -> Iterable[SourceFile]:
    for kind in kinds:
        for split in ("train", "valid"):
            base = data_root / kind / split
            if not base.exists():
                continue
            if kind == "noise":
                for path in sorted(base.rglob("*.wav")):
                    if path.is_file():
                        yield SourceFile(path, kind, split, "noise")
            else:
                for label_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                    for path in sorted(label_dir.rglob("*.wav")):
                        if path.is_file():
                            yield SourceFile(path, kind, split, label_dir.name)


def destination_for(data_root: Path, item: SourceFile, dst_split: str) -> Path:
    if item.kind == "noise":
        return data_root / item.kind / dst_split / item.path.name
    return data_root / item.kind / dst_split / item.label / item.path.name


def plan_split(
    data_root: Path,
    files: Sequence[SourceFile],
    valid_ratio: float,
    seed: int,
    min_valid_per_group: int,
) -> Tuple[List[PlannedAction], Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[SourceFile]] = defaultdict(list)
    for item in files:
        grouped[(item.kind, item.label)].append(item)

    actions: List[PlannedAction] = []
    group_summary = []
    for (kind, label), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda x: stable_score(x.path.relative_to(data_root), seed))
        desired_valid = int(round(len(ordered) * valid_ratio))
        if len(ordered) > 0:
            desired_valid = max(desired_valid, min(min_valid_per_group, len(ordered)))
        valid_set = {item.path for item in ordered[:desired_valid]}

        counts_before = Counter(item.split for item in ordered)
        counts_after = Counter()
        for item in ordered:
            dst_split = "valid" if item.path in valid_set else "train"
            counts_after[dst_split] += 1
            if item.split == dst_split:
                continue
            dst = destination_for(data_root, item, dst_split)
            actions.append(
                PlannedAction(
                    action="move",
                    kind=kind,
                    label=label,
                    src_split=item.split,
                    dst_split=dst_split,
                    src=str(item.path),
                    dst=str(dst),
                )
            )
        group_summary.append(
            {
                "kind": kind,
                "label": label,
                "total": len(ordered),
                "before": dict(counts_before),
                "after": dict(counts_after),
                "moves": sum(1 for a in actions if a.kind == kind and a.label == label),
            }
        )

    summary = {
        "data_root": str(data_root),
        "valid_ratio": valid_ratio,
        "seed": seed,
        "min_valid_per_group": min_valid_per_group,
        "n_files": len(files),
        "n_actions": len(actions),
        "before_by_kind_split": {
            f"{kind}/{split}": count
            for (kind, split), count in Counter((x.kind, x.split) for x in files).items()
        },
        "actions_by_direction": {
            f"{kind}/{src_split}->{dst_split}": count
            for (kind, src_split, dst_split), count in Counter((x.kind, x.src_split, x.dst_split) for x in actions).items()
        },
        "groups": group_summary,
    }
    return actions, summary


def execute_actions(actions: Sequence[PlannedAction], overwrite: bool) -> None:
    for action in actions:
        src = Path(action.src)
        dst = Path(action.dst)
        if not src.exists():
            raise FileNotFoundError(f"source missing before move: {src}")
        if dst.exists() and src.resolve() != dst.resolve():
            if not overwrite:
                raise FileExistsError(f"destination exists, pass --overwrite to replace: {dst}")
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-split extracted AudioSet-Strong WAVs.")
    parser.add_argument("--data_root", type=Path, default=Path("data/dev_set"))
    parser.add_argument("--valid_ratio", type=float, required=True, help="Target validation ratio per kind/label group.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kinds", nargs="+", default=list(AUDIO_KINDS), choices=AUDIO_KINDS)
    parser.add_argument("--min_valid_per_group", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Actually move files. Omit for dry run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report_path",
        type=Path,
        default=Path("data/dev_set/metadata/audioset_strong/split_adjustment_summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.valid_ratio <= 1.0:
        raise ValueError("--valid_ratio must be in [0, 1]")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")

    files = list(iter_wavs_from_tree(args.data_root, args.kinds))
    actions, summary = plan_split(
        args.data_root,
        files,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        min_valid_per_group=args.min_valid_per_group,
    )

    print("====================================================")
    print("AudioSet-Strong split adjustment")
    print(f"data_root    : {args.data_root}")
    print(f"valid_ratio  : {args.valid_ratio}")
    print(f"seed         : {args.seed}")
    print(f"execute      : {args.execute}")
    print(f"files        : {len(files)}")
    print(f"planned moves: {len(actions)}")
    print("====================================================")
    for group in summary["groups"]:
        if group["total"] == 0:
            continue
        print(
            f"{group['kind']}/{group['label']}: total={group['total']} "
            f"before={group['before']} after={group['after']} moves={group['moves']}"
        )

    write_json(args.report_path, {"summary": summary, "actions": [asdict(a) for a in actions]})
    print(f"Wrote report: {args.report_path}")

    if args.execute:
        execute_actions(actions, overwrite=args.overwrite)
        print("Split adjustment complete.")
    else:
        print("Dry run only. Re-run with --execute to move files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
