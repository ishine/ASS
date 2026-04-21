import torch
import torch.nn.functional as F


def get_loss_func(lambda_energy=0.0, m_in=-6.0, m_out=-1.0):
    def loss_func(output, target):
        class_index = target["class_index"]
        is_silence = target["is_silence"].bool()

        loss = output["plain_logits"].new_tensor(0.0)
        metrics = {}

        if (~is_silence).any():
            loss_arc = F.cross_entropy(output["logits"][~is_silence], class_index[~is_silence])
            loss = loss + loss_arc
            metrics["loss_arcface"] = loss_arc
        else:
            metrics["loss_arcface"] = loss

        if is_silence.any():
            log_probs = F.log_softmax(output["plain_logits"][is_silence], dim=-1)
            uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
            loss_kl = F.kl_div(log_probs, uniform, reduction="batchmean")
            loss = loss + loss_kl
            metrics["loss_kl"] = loss_kl
        else:
            metrics["loss_kl"] = loss.new_tensor(0.0)

        if lambda_energy > 0.0:
            energy = output["energy"]
            loss_in = energy[~is_silence] - m_in if (~is_silence).any() else energy.new_zeros(1)
            loss_out = m_out - energy[is_silence] if is_silence.any() else energy.new_zeros(1)
            hinge_in = torch.clamp(loss_in, min=0.0).pow(2).mean() if (~is_silence).any() else energy.new_tensor(0.0)
            hinge_out = torch.clamp(loss_out, min=0.0).pow(2).mean() if is_silence.any() else energy.new_tensor(0.0)
            loss_energy = hinge_in + hinge_out
            loss = loss + lambda_energy * loss_energy
            metrics["loss_energy"] = loss_energy
        else:
            metrics["loss_energy"] = loss.new_tensor(0.0)

        metrics["loss"] = loss
        return metrics

    return loss_func
