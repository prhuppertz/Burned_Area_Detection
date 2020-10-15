import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, relu=True):
        """
        :param num_channels_in:
        :param num_channels_out:
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels_in, num_channels_out, 3, padding=1),
            nn.BatchNorm2d(num_channels_out),
        )
        self.last_relu = relu
        if self.last_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_layers=2, residual=False):
        """

        :param num_channels_in:
        :param num_channels_out:
        """
        super(DoubleConv, self).__init__()
        conv = []
        self.last_relu = True if residual == False else False
        self.residual = residual
        for k in range(num_layers):
            if k == 0:
                conv.append(ConvBlock(num_channels_in, num_channels_out))
            else:
                conv.append(
                    ConvBlock(num_channels_out, num_channels_out, relu=self.last_relu)
                )

        self.conv = nn.Sequential(*conv)

        if self.residual:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual:
            residual = x
        x = self.conv(x)
        if self.residual:
            x = x + residual
            x = self.relu(x)
        return x


class InputConv(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_layers=2, residual=False):
        """

        :param num_channels_in:
        :param num_channels_out:
        """
        super(InputConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels_in)
        self.conv = DoubleConv(
            num_channels_in, num_channels_out, num_layers=num_layers, residual=residual
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class DownScale(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_layers=2, residual=False):
        """

        :param in_ch:
        :param out_ch:
        """
        super(DownScale, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                num_channels_in,
                num_channels_out,
                num_layers=num_layers,
                residual=residual,
            ),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpScale(nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        bilinear=True,
        num_layers=2,
        residual=False,
    ):
        """

        :param in_ch:
        :param out_ch:
        :param bilinear:
        """
        super(UpScale, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                num_channels_in // 2, num_channels_in // 2, 2, stride=2
            )

        self.conv = DoubleConv(
            num_channels_in, num_channels_out, num_layers=num_layers, residual=residual
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        """

        :param in_ch:
        :param out_ch:
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(num_channels_in, num_channels_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
