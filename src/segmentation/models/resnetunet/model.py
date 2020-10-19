from src.segmentation.networks.resnetunet.encoder import ResNet
from src.segmentation.networks.resnetunet.decoders import UnetDecoder, PSPDecoder
import importlib
from torch import optim
from typeguard import typechecked
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import lr_scheduler


@typechecked
class Model(pl.LightningModule):
    """
    Boundary segmentation model, Rsnet U-Net based
    """

    def __init__(self, hparams):
        """
        Initialises simple unet model
        :param dataset_names: names of the models
        :param hparams: task specific parameters indexed by task name
        """
        super(Model, self).__init__()
        self.hparams = hparams
        self.encoder, self.decoder = self.get_model(
            hparams["encoder_name"],
            bool(hparams["pretrained"]),
            hparams["decoder_name"],
            hparams["decoder_parameters"],
        )

    def forward(self, x):
        """
        Forward pass through the model
        :param x: input features
        :param dataset_name: task name
        :return:
        """
        layer0, layer1, layer2, layer3, layer4 = self.encoder(x)

        if isinstance(self.decoder, UnetDecoder):
            out = self.decoder(x, layer0, layer1, layer2, layer3, layer4)

        if isinstance(self.decoder, PSPDecoder):
            out = self.decoder(layer4)

        return out.squeeze()

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
            elif self.hparams["scheduler_type"] == "step":
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 50])
            else:
                raise ValueError(
                    "Unspecified scheduler type: {}".format(
                        self.hparams["scheduler_type"]
                    )
                )

            return [optimizer], [scheduler]

    def get_model(self, encoder_name, pretrained, decoder_name, decoder_parameters):

        # Get feature encoder
        encoder = ResNet(encoder_name, pretrained=pretrained)
        encoder = nn.Sequential(nn.BatchNorm2d(3), encoder)

        decoders = importlib.import_module("src.segmentation.networks.resnetunet.decoders")

        decoder = getattr(decoders, decoder_name)(**decoder_parameters)

        return encoder, decoder
