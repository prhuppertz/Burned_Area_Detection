from segmentation.networks.unet.unet import UNet
from segmentation.networks.adversarial.model import CNN
from torch import optim
import numpy as np
import torch
import pytorch_lightning as pl


class Model(pl.LightningModule):
    """
    Simple boundary segmentation model, U-Net based
    """
    def __init__(self, hparams):
        """

        :param dataset_names: names of the models
        :param params: task specific parameters indexed by task name
        :param encoder_name: either resnet18 or resnet50
        :param pretrained:
        """
        super(Model, self).__init__()
        self.hparams = hparams
        self.generator = UNet(hparams['n_channels'], hparams['decoders'], hparams['multiple'])
        self.discriminator = CNN(input_channels=hparams['discriminator_num_channels'])

    def forward(self, x):
        """
        Forward pass through the model
        :param x: input features
        :return:
        """
        out = self.generator(x)

        return out['classification'].squeeze()

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        TODO! explain all the steps!
        Function used during pipelines
        :param batch:
        :param batch_idx:
        :return:
        """
        images, targets = batch

        # First optimizes the segmentation network
        if optimizer_idx == 0:
            pred_targets = self(images).unsqueeze(1)

            valid = torch.ones(images.size(0), 1).cuda()

            disc_pred_targets = self.discriminator(pred_targets)

            g_loss = self.configuration['loss'](disc_pred_targets, valid)

            binary_loss = self.configuration['loss'](pred_targets.squeeze(), targets)

            loss = g_loss + binary_loss

        # Optimizes the discriminator network
        if optimizer_idx == 1:
            valid = torch.ones(images.size(0), 1).cuda()

            real_targets = self.discriminator(targets.unsqueeze(1))

            real_loss = self.configuration['loss'](real_targets, valid)

            fake = torch.zeros(images.size(0), 1).cuda()

            fake_targets = self.discriminator(self(images).detach().unsqueeze(1))

            fake_loss = self.configuration['loss'](fake_targets, fake)

            loss = (real_loss + fake_loss) / 2

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Function used during validation, applies postprocessing to evaluate the metric
        :param batch:
        :param batch_idx:
        :return:
        """
        images, targets = batch
        predicted_targets = self(images)
        loss = self.configuration['loss'](predicted_targets, targets).item()

        if self.configuration['metric']:
            predicted_postproc = self.configuration['postprocessing_func'](predicted_targets.cpu().numpy())
            metric = self.configuration['metric'](predicted_postproc,
                                               targets.cpu().numpy())
            out = {'val_loss': loss, 'metric': metric}
        else:
            out = {'val_loss': loss}

        return out

    def validation_epoch_end(self, outputs):
        """
        Computes statistics at the end of validation epoch
        :param outputs: a list of the form [{'loss': loss, 'metric': metric, 'dataset_name': dataset_name}, ...]
        :return:
        """
        loss = np.mean([x['val_loss'] for x in outputs])
        if self.configuration['metric']:
            metric = np.mean([x['metric'] for x in outputs])
            return {'val_loss': torch.Tensor([loss]), 'metric': torch.Tensor([metric])}
        else:
            return {'val_loss': torch.Tensor([loss])}

    def configure_optimizers(self):
        """
        This is required as part of pytorch-lightning
        :return:
        """
        opt_g = optim.Adam(self.generator.parameters(), lr=self.hparams['generator_lr'])
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.hparams['discriminator_lr'])
        return [opt_g, opt_d], []
