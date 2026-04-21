import torch

from src.models.s5.s5 import S5
from src.utils import initialize_config


class SelfGuidedS5(S5):
    def __init__(
        self,
        mixture_tagger_config,
        separator_config,
        label_set,
        source_tagger_config=None,
        mixture_tagger_ckpt=None,
        source_tagger_ckpt=None,
        separator_ckpt=None,
        num_refinement_steps=2,
    ):
        super().__init__(
            tagger_config=mixture_tagger_config,
            separator_config=separator_config,
            label_set=label_set,
            tagger_ckpt=mixture_tagger_ckpt,
            separator_ckpt=separator_ckpt,
        )
        self.num_refinement_steps = max(1, int(num_refinement_steps))
        self.source_tagger = None
        if source_tagger_config is not None:
            self.source_tagger = initialize_config(source_tagger_config)
            if source_tagger_ckpt is not None:
                self._load_ckpt(source_tagger_ckpt, self.source_tagger)
            self.source_tagger.eval()

    def _refine_labels(self, waveforms):
        if self.source_tagger is None:
            return None

        batch_size, n_sources, _, n_samples = waveforms.shape
        source_audio = waveforms[:, :, 0, :].reshape(batch_size * n_sources, n_samples)
        output = self.source_tagger.predict({"waveform": source_audio})
        label_vector = output["label_vector"].view(batch_size, n_sources, -1)
        probabilities = output["probabilities"].view(batch_size, n_sources)
        labels = self._get_label(label_vector)
        return {
            "label": labels,
            "label_vector": label_vector[..., :-1],
            "probabilities": probabilities,
        }

    def predict_label_separate(self, mixture):
        with torch.no_grad():
            label_out = self.predict_label(mixture)
            batch_label = label_out["label"]
            batch_probs = label_out["probabilities"]
            batch_label_vector = label_out["label_vector"]

            for _ in range(self.num_refinement_steps):
                separator_labels = batch_label_vector.flatten(start_dim=1)
                predict_waveforms = self.separate(mixture, separator_labels)
                refined = self._refine_labels(predict_waveforms["waveform"])
                if refined is None:
                    return {
                        "label": batch_label,
                        "probabilities": batch_probs,
                        "waveform": predict_waveforms["waveform"],
                    }
                batch_label = refined["label"]
                batch_probs = refined["probabilities"]
                batch_label_vector = refined["label_vector"]

            return {
                "label": batch_label,
                "probabilities": batch_probs,
                "waveform": predict_waveforms["waveform"],
            }
