import torch


SILENCE_SPAN_SEC = (-1.0, -1.0)


def _as_float(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_to_span_sec(event):
    """Best-effort extraction of an event span from SpAudSyn-style metadata."""
    if event is None:
        return SILENCE_SPAN_SEC
    metadata = event.get("metadata", event) if isinstance(event, dict) else {}

    for source in (metadata, event if isinstance(event, dict) else {}):
        event_time = source.get("event_time")
        if isinstance(event_time, (list, tuple)) and len(event_time) >= 2:
            start = _as_float(event_time[0])
            end = _as_float(event_time[1])
            if start is not None and end is not None and end > start:
                return float(start), float(end)

    start = None
    end = None
    for key in ("event_time", "onset", "start", "start_time", "time_start"):
        start = _as_float(metadata.get(key, event.get(key) if isinstance(event, dict) else None))
        if start is not None:
            break

    for key in ("event_end", "offset", "end", "end_time", "time_end"):
        end = _as_float(metadata.get(key, event.get(key) if isinstance(event, dict) else None))
        if end is not None:
            break

    if end is None:
        duration = None
        for key in ("event_duration", "duration", "dur"):
            duration = _as_float(metadata.get(key, event.get(key) if isinstance(event, dict) else None))
            if duration is not None:
                break
        if start is not None and duration is not None:
            end = start + duration

    if start is None or end is None or end <= start:
        return SILENCE_SPAN_SEC
    return float(start), float(end)


def waveform_to_span_sec(waveform, sample_rate, energy_eps=1e-8):
    """Infer the non-silent support of a single source waveform."""
    tensor = torch.as_tensor(waveform)
    if tensor.numel() == 0:
        return SILENCE_SPAN_SEC
    while tensor.dim() > 1:
        tensor = tensor.abs().amax(dim=0)
    active = tensor.abs() > energy_eps
    if not bool(active.any()):
        return SILENCE_SPAN_SEC
    idx = torch.nonzero(active, as_tuple=False).flatten()
    start = float(idx[0].item()) / float(sample_rate)
    end = float(idx[-1].item() + 1) / float(sample_rate)
    return start, end


def pad_spans(spans, n_expected):
    spans = list(spans[:n_expected])
    while len(spans) < n_expected:
        spans.append(SILENCE_SPAN_SEC)
    return torch.tensor(spans, dtype=torch.float32)


def spans_to_frame_targets(span_sec, num_frames, duration_sec=None, frame_times_sec=None):
    """Convert [B, 2] spans into [B, T] binary activity targets."""
    original_shape = span_sec.shape[:-1]
    if span_sec.dim() > 2:
        span_sec = span_sec.reshape(-1, 2)
        if duration_sec is not None:
            duration_tensor = torch.as_tensor(duration_sec, device=span_sec.device)
            if duration_tensor.dim() == len(original_shape):
                duration_sec = duration_tensor.reshape(-1)
            elif duration_tensor.dim() == 1 and len(original_shape) > 1:
                duration_sec = duration_tensor[:, None].expand(original_shape).reshape(-1)
    if span_sec.dim() == 1:
        span_sec = span_sec.unsqueeze(0)
    device = span_sec.device
    dtype = span_sec.dtype if span_sec.is_floating_point() else torch.float32
    span_sec = span_sec.to(dtype=dtype)

    if frame_times_sec is None:
        if duration_sec is None:
            frame_times = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
        else:
            duration = torch.as_tensor(duration_sec, device=device, dtype=dtype)
            if duration.dim() == 0:
                duration = duration.expand(span_sec.shape[0])
            frame_unit = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
            frame_times = duration[:, None] * frame_unit[None, :]
    else:
        frame_times = frame_times_sec.to(device=device, dtype=dtype)
        if frame_times.dim() == 1:
            frame_times = frame_times.unsqueeze(0).expand(span_sec.shape[0], -1)

    start = span_sec[:, 0:1]
    end = span_sec[:, 1:2]
    valid = (start >= 0.0) & (end > start)
    targets = ((frame_times >= start) & (frame_times <= end) & valid).to(dtype)
    if len(original_shape) > 1:
        targets = targets.reshape(*original_shape, num_frames)
    return targets


def align_spans_to_predictions(best_perm, target_spans, active_mask, n_preds=None):
    """Move target spans into prediction-slot order using a PIT permutation."""
    if n_preds is None:
        n_preds = best_perm.shape[1]
    aligned = target_spans.new_full((target_spans.shape[0], n_preds, 2), -1.0)
    active_mask = active_mask.to(device=best_perm.device, dtype=torch.bool)
    for batch_idx in range(best_perm.shape[0]):
        active_idx = torch.nonzero(active_mask[batch_idx], as_tuple=False).flatten()
        for target_idx in active_idx.tolist():
            pred_idx = int(best_perm[batch_idx, target_idx].item())
            if 0 <= pred_idx < n_preds:
                aligned[batch_idx, pred_idx] = target_spans[batch_idx, target_idx]
    return aligned
