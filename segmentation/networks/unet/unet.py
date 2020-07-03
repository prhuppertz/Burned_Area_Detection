from segmentation.networks.unet.unet_layers import InputConv, DownScale, UpScale, OutConv
from torch import nn
from typing import Dict, Optional, Union
from typeguard import typechecked

@typechecked
class UNet(nn.Module):

    def __init__(self, n_channels: int, decoders: Dict = {'classification': 1},
                 multiple: float = 1, num_layers: int = 2, residual = False):
        """
        Defines UNet neural network
        :param n_channels: How many input channels are there
        :param decoders: How many decoders there should be and how many channels they should output {name: num_channels}
        :param multiple: multiple of the model size
        """
        super(UNet, self).__init__()
        self.inc = InputConv(n_channels, int(8*multiple), num_layers = num_layers, residual = residual)
        self.down1 = DownScale(int(8*multiple), int(16*multiple), num_layers = num_layers, residual = residual)
        self.down2 = DownScale(int(16*multiple), int(32*multiple), num_layers = num_layers, residual = residual)
        self.down3 = DownScale(int(32*multiple), int(64*multiple), num_layers = num_layers, residual = residual)
        self.down4 = DownScale(int(64*multiple), int(64*multiple), num_layers = num_layers, residual = residual)

        self.decoders = nn.ModuleDict({})

        for name, num_channels in decoders.items():

            self.decoders.update({name: Decoder(num_channels, multiple, None, residual = residual)})

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out_dict = {}
        for name, decoder in self.decoders.items():
            out_dict[name] = decoder(x5, [x4, x3, x2, x1])
        return out_dict


class Decoder(nn.Module):

    def __init__(self, n_classes: int, multiple: float = 1, drop_prob: Optional[Union[int, float]]=None, residual = False):

        super(Decoder, self).__init__()

        self.up1 = UpScale(int(128*multiple), int(32*multiple), residual = residual)
        self.up2 = UpScale(int(64*multiple), int(16*multiple), residual = residual)
        self.up3 = UpScale(int(32*multiple), int(8*multiple), residual = residual)
        self.up4 = UpScale(int(16*multiple), int(8*multiple), residual = residual)

        if drop_prob:
            self.dropout = nn.Dropout(drop_prob)
        self.outc = OutConv(int(8*multiple), n_classes)

        self.drop_prob = drop_prob

    def forward(self, x, features):
        """

        :param x:
        :param features:
        :return:
        """
        x4, x3, x2, x1 = features
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.drop_prob:
            x = self.dropout(x)
        x = self.outc(x)
        return x

