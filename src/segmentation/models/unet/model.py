from segmentation.networks.unet.unet import UNet
from torch import optim
from typeguard import typechecked
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler


@typechecked
class Model(pl.LightningModule):
    """
    Simple boundary segmentation model, U-Net based
    """

    def __init__(self, hparams):
        """
        Initialises simple unet model
        :param dataset_names: names of the models
        :param hparams: task specific parameters indexed by task name
        """
        super(Model, self).__init__()
        self.hparams = hparams
        self.model = UNet(
            hparams["n_channels"],
            hparams["decoders"],
            hparams["multiple"],
            hparams["num_layers"],
            hparams["residual"],
        )

    def forward(self, x):
        """
        Forward pass through the model
        :param x: input features
        :return:
        """
        out = self.model(x)

        return out["classification"].squeeze()

    def training_step(self, batch, batch_idx):
        """
        Function used during pipelines
        :param batch:
        :param batch_idx:
        :return:
        """
        images, (label_mask, distance_mask, gt) = batch
        predicted_targets = self(images)
        loss = self.configuration["loss"](predicted_targets, label_mask, distance_mask)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """
        Computes statistics at the end of validation epoch
        :param outputs: a list of the form [{'loss': loss, 'metric': metric}, ...]
        :return:
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        images, (label_mask, distance_mask, gt) = batch
        predicted_targets = self(images)
        loss = self.configuration["loss"](predicted_targets, label_mask, distance_mask)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes statistics at the end of validation epoch
        :param outputs: a list of the form [{'loss': loss, 'metric': metric}, ...]
        :return:
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": log}

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        images, (label_mask, distance_mask, gt) = batch
        predicted_targets = self(images.cuda())

        predicted_postproc = self.configuration["postprocessing_func"](
            predicted_targets.cpu().numpy()
        )
        scores = self.configuration["metric"](predicted_postproc, gt.cpu().numpy())
        out = {
            "metrics": scores,
            "images": images.cpu().numpy().astype(np.uint8),
            "targets": label_mask.cpu().numpy(),
            "predictions": predicted_postproc,
        }

        return out

    def configure_optimizers(self):
        """
        This is required as part of pytorch-lightning
        :return:
        """
        optimizer_type = self.hparams["optimizer_type"]
        if optimizer_type == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )
        if optimizer_type == "ADAM":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        if self.hparams["scheduler_type"] == None:
            return [optimizer]
        else:
            if self.hparams["scheduler_type"] == "plateu":
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=5
                )
            elif self.hparams["scheduler_type"] == "one_cycle":
                scheduler = lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.hparams["lr"],
                    epochs=self.hparams["max_epochs"],
                    steps_per_epoch=self.hparams["steps_per_epoch"],
                )
            else:
                raise ValueError(
                    "Unspecified scheduler type: {}".format(
                        self.hparams["scheduler_type"]
                    )
                )

            return [optimizer], [scheduler]
