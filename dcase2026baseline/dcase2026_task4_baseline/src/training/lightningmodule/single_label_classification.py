from .base_lightningmodule import BaseLightningModule


class SingleLabelClassificationLightning(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["waveform"].shape[0]
        input_dict = {
            "waveform": batch_data_dict["waveform"],
            "class_index": batch_data_dict["class_index"],
        }
        output_dict = self.model(input_dict)
        target_dict = {
            "class_index": batch_data_dict["class_index"],
            "is_silence": batch_data_dict["is_silence"],
        }
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict
