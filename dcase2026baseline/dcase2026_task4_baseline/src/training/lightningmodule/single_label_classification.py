from .base_lightningmodule import BaseLightningModule


class SingleLabelClassificationLightning(BaseLightningModule):
    def _get_input_dict(self, batch_data_dict):
        input_dict = {
            "waveform": batch_data_dict["waveform"],
            "class_index": batch_data_dict["class_index"],
        }
        if "span_sec" in batch_data_dict:
            input_dict["span_sec"] = batch_data_dict["span_sec"]
        return input_dict

    def _get_target_dict(self, batch_data_dict):
        target_dict = {
            "class_index": batch_data_dict["class_index"],
            "is_silence": batch_data_dict["is_silence"],
        }
        for key in ("duplicate_class_count", "is_duplicate_class"):
            if key in batch_data_dict:
                target_dict[key] = batch_data_dict[key]
        if "span_sec" in batch_data_dict:
            target_dict["span_sec"] = batch_data_dict["span_sec"]
        return target_dict

    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["waveform"].shape[0]
        output_dict = self.model(self._get_input_dict(batch_data_dict))
        target_dict = self._get_target_dict(batch_data_dict)
        target_dict["current_epoch"] = self.current_epoch
        target_dict["is_training"] = True
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["waveform"].shape[0]
        output_dict = self.model(self._get_input_dict(batch_data_dict))
        target_dict = self._get_target_dict(batch_data_dict)
        target_dict["current_epoch"] = self.current_epoch
        target_dict["is_training"] = False
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if self.metric_func:
            metric = self.metric_func(output_dict, target_dict)
            for k, v in metric.items():
                loss_dict[k] = v.mean().item()

        return batchsize, loss_dict
