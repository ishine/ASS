import torch
import torch.nn.functional as F


def _safe_energy(x):
    return torch.sum(x ** 2, dim=-1) + 1e-8


def sa_sdr_loss(est, target):
    num = _safe_energy(target).sum(dim=1)
    den = _safe_energy(target - est).sum(dim=1) + 1e-8
    return -10.0 * torch.log10(num / den)


def si_snr_loss(est, target):
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
    scale = torch.sum(est * target, dim=-1, keepdim=True) / target_energy
    target_proj = scale * target
    noise = est - target_proj
    return -10.0 * torch.log10((_safe_energy(target_proj) / (_safe_energy(noise) + 1e-8)))


def get_loss_func(lambda_non_foreground=0.01, lambda_kl=1.0, lambda_silence=1.0):
    def loss_func(output, target):
        fg_est = output["foreground_waveform"][:, :, 0]
        int_est = output["interference_waveform"][:, :, 0]
        noise_est = output["noise_waveform"][:, 0, 0]

        fg_ref = target["foreground_waveform"][:, :, 0]
        int_ref = target["interference_waveform"][:, :, 0]
        noise_ref = target["noise_waveform"][:, 0, 0]

        loss_fg = sa_sdr_loss(fg_est, fg_ref).mean()
        loss_int = sa_sdr_loss(int_est, int_ref).mean()
        loss_noise = si_snr_loss(noise_est, noise_ref).mean()

        class_logits = output["class_logits"]
        class_index = target["class_index"]
        is_silence = target["is_silence"].bool()
        if (~is_silence).any():
            loss_ce = F.cross_entropy(class_logits[~is_silence], class_index[~is_silence])
        else:
            loss_ce = class_logits.new_tensor(0.0)
        if is_silence.any():
            log_probs = F.log_softmax(class_logits[is_silence], dim=-1)
            uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
            loss_kl = F.kl_div(log_probs, uniform, reduction="batchmean")
        else:
            loss_kl = class_logits.new_tensor(0.0)
        loss_silence = F.binary_cross_entropy_with_logits(output["silence_logits"], (~is_silence).float())

        loss = loss_fg + lambda_non_foreground * (loss_int + loss_noise) + loss_ce + lambda_kl * loss_kl + lambda_silence * loss_silence
        return {
            "loss": loss,
            "loss_fg": loss_fg,
            "loss_int": loss_int,
            "loss_noise": loss_noise,
            "loss_ce": loss_ce,
            "loss_kl": loss_kl,
            "loss_silence": loss_silence,
        }

    return loss_func
