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

    def training_step(self, batch_data_dict, batch_idx):
        self.set_train_mode()
        batchsize, loss_dict = self.training_step_processing(batch_data_dict, batch_idx)
        loss = loss_dict["loss"]

        step_dict = {f"step_train/{name}": val.item() for name, val in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoch_dict = {f"epoch_train/{name}": val.item() for name, val in loss_dict.items()}
        self.log_dict(epoch_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)

        active_count = int((~batch_data_dict["is_silence"].bool()).sum().item())
        active_ratio = float(active_count / max(batchsize, 1))
        self.log("epoch_train/active_ratio", active_ratio, prog_bar=False, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        if active_count > 0:
            if "loss_arcface" in loss_dict:
                self.log(
                    "epoch_train/loss_arcface_active_weighted",
                    loss_dict["loss_arcface"].item(),
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    batch_size=active_count,
                )
            if "loss_plain_ce" in loss_dict:
                self.log(
                    "epoch_train/loss_plain_ce_active_weighted",
                    loss_dict["loss_plain_ce"].item(),
                    prog_bar=False,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    batch_size=active_count,
                )

        self.log_dict({"epoch/lr": self.optimizer.param_groups[0]["lr"]})
        return loss

    def _validation_step(self, batch_data_dict, batch_idx):
        self.model.eval()
        batchsize, loss_dict = self.validation_step_processing(batch_data_dict, batch_idx)

        step_dict = {f"step_val/{name}": metric for name, metric in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoch_dict = {f"epoch_val/{name}": metric for name, metric in loss_dict.items()}
        self.log_dict(epoch_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)

        active_count = int((~batch_data_dict["is_silence"].bool()).sum().item())
        active_ratio = float(active_count / max(batchsize, 1))
        self.log("epoch_val/active_ratio", active_ratio, prog_bar=False, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        if active_count > 0:
            if "loss_arcface" in loss_dict:
                self.log(
                    "epoch_val/loss_arcface_active_weighted",
                    float(loss_dict["loss_arcface"]),
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    batch_size=active_count,
                )
            if "loss_plain_ce" in loss_dict:
                self.log(
                    "epoch_val/loss_plain_ce_active_weighted",
                    float(loss_dict["loss_plain_ce"]),
                    prog_bar=False,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    batch_size=active_count,
                )
