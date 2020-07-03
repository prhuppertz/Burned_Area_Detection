from segmentation.networks.unet.unet import UNet
from torch import optim
from typeguard import typechecked
import numpy as np
import torch
import pytorch_lightning as pl

@typechecked
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
        self.model = UNet(hparams['n_channels'], hparams['decoders'], hparams['multiple'])

    def forward(self, x):
        """
        Forward pass through the model
        :param x: input features
        :return:
        """
        out = self.model(x)

        cls = out['classification']
        hv = out['hover']
        return torch.cat([cls, hv], dim=1)

    def training_step(self, batch, batch_idx):
        """
        Function used during pipelines
        :param batch:
        :param batch_idx:
        :return:
        """
        images, targets = batch
        predicted_targets = self(images)
        loss = self.configuration['loss'](predicted_targets, targets, focus=targets[:,0,:,:])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Function used during validation, applies postprocessing to evaluate the metric
        :param batch:
        :param batch_idx:
        :return:
        """
        images, targets = batch
        # For evaluation during validation we need both semantic and instance mask
        # Semantic is indexed with 0 and instance mask is indexed at 1 #TODO! double check this!
        instance_mask, targets = targets[:,0,:,:], targets[:,0:,:,:]
        predicted_targets = self(images)
        loss = self.configuration['loss'](predicted_targets, targets).item()

        if self.configuration['metric']:
            predicted_postproc = self.configuration['postprocessing_func'](predicted_targets.cpu().numpy())
            metric = self.configuration['metric'](predicted_postproc,
                                               instance_mask.cpu().numpy())
            out = {'val_loss': loss, 'metric': metric}
        else:
            out = {'val_loss': loss}

        return out

    def validation_epoch_end(self, outputs):
        """
        Computes statistics at the end of validation epoch
        :param outputs: a list of the form [{'loss': loss, 'metric': metric}, ...]
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
        optimizer_type = self.hparams['optimizer_type']
        if optimizer_type == 'SGD':
            return optim.SGD(
                self.parameters(),
                lr=self.hparams['lr'],
                weight_decay=self.hparams['weight_decay'],
            )
        if optimizer_type == 'ADAM':
            return optim.Adam(self.parameters(), lr=self.hparams['lr'],
                              weight_decay=self.hparams['weight_decay'])
