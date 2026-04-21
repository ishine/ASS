import torch

from src.utils import LABELS, initialize_config


class Kwon2025S5(torch.nn.Module):
    def __init__(
        self,
        label_set,
        uss_config,
        sc_config,
        tse_config,
        uss_ckpt=None,
        sc_ckpt=None,
        tse_ckpt=None,
    ):
        super().__init__()
        self.uss = initialize_config(uss_config)
        self.sc = initialize_config(sc_config)
        self.tse = initialize_config(tse_config)
        self.label_set = label_set
        self.labels = LABELS[label_set].copy()
        self.onehots = torch.eye(len(self.labels), requires_grad=False).float()

        if uss_ckpt is not None:
            self._load_ckpt(uss_ckpt, self.uss)
        if sc_ckpt is not None:
            self._load_ckpt(sc_ckpt, self.sc)
        if tse_ckpt is not None:
            self._load_ckpt(tse_ckpt, self.tse)

        self.uss.eval()
        self.sc.eval()
        self.tse.eval()

    def _load_ckpt(self, path, model):
        model_ckpt = torch.load(path, weights_only=False, map_location="cpu")["state_dict"]
        if set(model.state_dict().keys()) != set(model_ckpt.keys()):
            one_model_key = next(iter(model.state_dict().keys()))
            ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
            prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
            model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)}
        model.load_state_dict(model_ckpt)

    def _vector_to_label(self, label_vector):
        labels = []
        for source_vec in label_vector:
            if source_vec.sum() == 0:
                labels.append("silence")
            else:
                labels.append(self.labels[int(torch.argmax(source_vec))])
        return labels

    def _classify_sources(self, waveforms):
        batch_size, n_sources, _, samples = waveforms.shape
        flat = waveforms[:, :, 0].reshape(batch_size * n_sources, samples)
        out = self.sc.predict({"waveform": flat})
        label_vector = out["label_vector"].view(batch_size, n_sources, -1)
        probs = out["probabilities"].view(batch_size, n_sources)
        labels = [self._vector_to_label(label_vector[i]) for i in range(batch_size)]
        return labels, probs, label_vector

    def _run_tse(self, mixture, enroll, label_vector):
        return self.tse({
            "mixture": mixture,
            "enrollment": enroll,
            "label_vector": label_vector,
        })["waveform"]

    def predict_label_separate(self, mixture):
        with torch.no_grad():
            uss_out = self.uss({"mixture": mixture})
            stage1_waveform = uss_out["foreground_waveform"]
            stage1_labels, stage1_probs, stage1_vector = self._classify_sources(stage1_waveform)

            stage2_waveform = self._run_tse(mixture, stage1_waveform, stage1_vector)
            stage2_labels, stage2_probs, stage2_vector = self._classify_sources(stage2_waveform)

            stage3_waveform = self._run_tse(mixture, stage2_waveform, stage2_vector)
            stage3_labels, stage3_probs, _ = self._classify_sources(stage3_waveform)

            return {
                "label": stage3_labels,
                "probabilities": stage3_probs,
                "waveform": stage3_waveform,
            }
