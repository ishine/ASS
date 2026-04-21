from .base_lightningmodule import BaseLightningModule


class USSLightning(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model({"mixture": batch_data_dict["mixture"]})
        target_dict = {
            "foreground_waveform": batch_data_dict["foreground_waveform"],
            "interference_waveform": batch_data_dict["interference_waveform"],
            "noise_waveform": batch_data_dict["noise_waveform"],
            "class_index": batch_data_dict["class_index"],
            "is_silence": batch_data_dict["is_silence"],
        }
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict
