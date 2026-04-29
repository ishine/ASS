from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class MatchResult:
    soundscape: str
    oracle_slot: int
    estimate_slot: int
    label: str
    metric: str
    match_score: float
    second_best_score: float
    match_margin: float
    energy_db: float
    quality_group: str
    sample_weight: float
    saved: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def flatten_sources(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,S,...] source tensors to [B,S,T_flat]."""
    if x.dim() < 3:
        raise ValueError(f"Expected source tensor [B,S,...], got {tuple(x.shape)}")
    return x.flatten(start_dim=2).float()


def source_energy_db(source: torch.Tensor, eps: float = 1e-12) -> float:
    x = source.float().flatten()
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return float(20.0 * torch.log10(rms.clamp_min(1e-8)).item())


def pairwise_sa_sdr_score(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise source-aggregated SDR score in dB, higher is better.

    Args:
        est: [B,S_est,...]
        ref: [B,S_ref,...]

    Returns:
        scores: [B,S_ref,S_est]
    """
    est_f = flatten_sources(est)
    ref_f = flatten_sources(ref)
    err = est_f.unsqueeze(1) - ref_f.unsqueeze(2)
    ref_power = ref_f.pow(2).sum(dim=-1).unsqueeze(2).clamp_min(eps)
    err_power = err.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10((ref_power / err_power).clamp_min(eps))


def pairwise_si_sdr_score(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise scale-invariant SDR score in dB, higher is better.

    Args:
        est: [B,S_est,...]
        ref: [B,S_ref,...]

    Returns:
        scores: [B,S_ref,S_est]
    """
    est_f = flatten_sources(est)
    ref_f = flatten_sources(ref)
    est_zm = est_f - est_f.mean(dim=-1, keepdim=True)
    ref_zm = ref_f - ref_f.mean(dim=-1, keepdim=True)
    dot = (est_zm.unsqueeze(1) * ref_zm.unsqueeze(2)).sum(dim=-1, keepdim=True)
    ref_energy = ref_zm.pow(2).sum(dim=-1, keepdim=True).unsqueeze(2).clamp_min(eps)
    proj = dot * ref_zm.unsqueeze(2) / ref_energy
    noise = est_zm.unsqueeze(1) - proj
    ratio = proj.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def pairwise_match_score(est: torch.Tensor, ref: torch.Tensor, metric: str = "sa_sdr") -> torch.Tensor:
    metric = metric.lower()
    if metric in {"sa_sdr", "sasdr", "sdr"}:
        return pairwise_sa_sdr_score(est, ref)
    if metric in {"si_sdr", "sisdr"}:
        return pairwise_si_sdr_score(est, ref)
    raise ValueError(f"Unsupported matching metric: {metric}")


def active_ref_indices(labels: Sequence[str]) -> List[int]:
    return [idx for idx, label in enumerate(labels) if label != "silence"]


def best_pit_assignment(scores: torch.Tensor, refs: Iterable[int], n_est: int) -> Dict[int, int]:
    """Return target/oracle-slot -> estimated-slot assignment maximizing mean score.

    Brute-force is intentionally used because DCASE S5 has only up to 3 target
    foreground slots. This avoids a SciPy dependency and is exact for this task.
    """
    refs = list(refs)
    if not refs:
        return {}
    n_match = min(len(refs), n_est)
    refs = refs[:n_match]
    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(n_est), n_match):
        vals = torch.stack([scores[ref_idx, est_idx] for ref_idx, est_idx in zip(refs, perm)])
        score = vals.mean()
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    return {ref_idx: int(est_idx) for ref_idx, est_idx in zip(refs, best_perm)}


def second_best_and_margin(scores: torch.Tensor, ref_idx: int, est_idx: int) -> tuple[float, float]:
    row = scores[ref_idx]
    best = float(row[est_idx].item())
    if row.numel() <= 1:
        return float("-inf"), float("inf")
    others = torch.cat([row[:est_idx], row[est_idx + 1 :]])
    second = float(others.max().item())
    return second, best - second


def quality_and_weight(
    score: float,
    margin: float,
    energy_db: float,
    min_match_score: float,
    min_match_margin: float,
    min_energy_db: float,
    clean_match_score: float,
    clean_match_margin: float,
    uncertain_weight: float,
) -> tuple[str, float, bool]:
    if score < min_match_score or margin < min_match_margin or energy_db < min_energy_db:
        return "bad", 0.0, False
    if score >= clean_match_score and margin >= clean_match_margin:
        return "clean", 1.0, True
    return "uncertain", float(uncertain_weight), True


def match_one_soundscape(
    est_sources: torch.Tensor,
    ref_sources: torch.Tensor,
    labels: Sequence[str],
    soundscape: str,
    metric: str = "sa_sdr",
    min_match_score: float = -10.0,
    min_match_margin: float = -1e9,
    min_energy_db: float = -60.0,
    clean_match_score: float = 0.0,
    clean_match_margin: float = 2.0,
    uncertain_weight: float = 0.35,
    save_uncertain: bool = False,
    include_unmatched: bool = False,
) -> List[MatchResult]:
    """Match USS estimates to oracle sources and assign oracle labels.

    Args:
        est_sources: [S_est,1,T] or [S_est,T]
        ref_sources: [S_ref,1,T] or [S_ref,T]
        labels: oracle labels, with possible 'silence' padding

    Returns:
        one MatchResult per active oracle source, plus optional unmatched rows.
    """
    est_b = est_sources.unsqueeze(0)
    ref_b = ref_sources.unsqueeze(0)
    scores = pairwise_match_score(est_b, ref_b, metric=metric)[0].cpu()
    refs = active_ref_indices(labels)
    assignment = best_pit_assignment(scores, refs, est_sources.shape[0])
    used_est = set()
    rows: List[MatchResult] = []
    for ref_idx in refs:
        if ref_idx not in assignment:
            continue
        est_idx = assignment[ref_idx]
        used_est.add(est_idx)
        score = float(scores[ref_idx, est_idx].item())
        second, margin = second_best_and_margin(scores, ref_idx, est_idx)
        energy = source_energy_db(est_sources[est_idx])
        quality, weight, valid = quality_and_weight(
            score=score,
            margin=margin,
            energy_db=energy,
            min_match_score=min_match_score,
            min_match_margin=min_match_margin,
            min_energy_db=min_energy_db,
            clean_match_score=clean_match_score,
            clean_match_margin=clean_match_margin,
            uncertain_weight=uncertain_weight,
        )
        saved = bool(valid and (quality == "clean" or save_uncertain))
        rows.append(
            MatchResult(
                soundscape=str(soundscape),
                oracle_slot=int(ref_idx),
                estimate_slot=int(est_idx),
                label=str(labels[ref_idx]),
                metric=metric,
                match_score=score,
                second_best_score=second,
                match_margin=margin,
                energy_db=energy,
                quality_group=quality,
                sample_weight=weight,
                saved=saved,
            )
        )
    if include_unmatched:
        for est_idx in range(est_sources.shape[0]):
            if est_idx not in used_est:
                rows.append(
                    MatchResult(
                        soundscape=str(soundscape),
                        oracle_slot=-1,
                        estimate_slot=int(est_idx),
                        label="silence",
                        metric=metric,
                        match_score=float("nan"),
                        second_best_score=float("nan"),
                        match_margin=float("nan"),
                        energy_db=source_energy_db(est_sources[est_idx]),
                        quality_group="unmatched",
                        sample_weight=0.0,
                        saved=False,
                    )
                )
    return rows


def match_batch(
    est_sources: torch.Tensor,
    ref_sources: torch.Tensor,
    labels: Sequence[Sequence[str]],
    soundscapes: Sequence[str],
    **kwargs,
) -> List[List[MatchResult]]:
    results = []
    for batch_idx, soundscape in enumerate(soundscapes):
        results.append(
            match_one_soundscape(
                est_sources=est_sources[batch_idx].detach().cpu(),
                ref_sources=ref_sources[batch_idx].detach().cpu(),
                labels=labels[batch_idx],
                soundscape=soundscape,
                **kwargs,
            )
        )
    return results
