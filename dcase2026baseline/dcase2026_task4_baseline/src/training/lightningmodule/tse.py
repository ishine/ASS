from .base_lightningmodule import BaseLightningModule


class TSELightning(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict["mixture"].shape[0]
        output_dict = self.model({
            "mixture": batch_data_dict["mixture"],
            "enrollment": batch_data_dict["enrollment"],
            "label_vector": batch_data_dict["label_vector"],
        })
        target_dict = {
            "waveform": batch_data_dict["waveform"],
            "active_mask": batch_data_dict["active_mask"],
        }
        loss_dict = self.loss_func(output_dict, target_dict)
        return batchsize, loss_dict
