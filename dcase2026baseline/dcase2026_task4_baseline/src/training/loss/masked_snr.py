import torch


def masked_snr_loss(est, target, active_mask):
    est = est[:, :, 0]
    target = target[:, :, 0]
    power_t = torch.sum(target ** 2, dim=-1) + 1e-8
    power_n = torch.sum((target - est) ** 2, dim=-1) + 1e-8
    snr = 10.0 * torch.log10(power_t / power_n)
    active_mask = active_mask.float()
    return -(snr * active_mask).sum() / active_mask.sum().clamp_min(1.0)


def get_loss_func():
    def loss_func(output, target):
        loss = masked_snr_loss(output["waveform"], target["waveform"], target["active_mask"])
        return {"loss": loss}

    return loss_func
