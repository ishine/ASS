from .uss_bridge import USSBridgeLightning


class NoisyLabelUSSLightning(USSBridgeLightning):
    """Opt-in USS lightning module for noisy-label robust losses."""

    def _get_target_dict(self, batch_data_dict):
        target_dict = super()._get_target_dict(batch_data_dict)
        for key in (
            "class_confidence",
            "soft_class_target",
            "uncertain_slot_mask",
            "bad_slot_mask",
            "quality_resample_attempts",
        ):
            if key in batch_data_dict:
                target_dict[key] = batch_data_dict[key]
        target_dict["current_epoch"] = self.current_epoch
        return target_dict
