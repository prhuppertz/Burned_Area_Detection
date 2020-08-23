import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from typeguard import typechecked


@typechecked
class PSPModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1024,
        sizes: Tuple[int, int, int, int] = (1, 2, 3, 6),
    ):
        """
        Performs pooling accross different spatial dimensions
        :param in_features:
        :param out_features:
        :param sizes:
        """
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            in_features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.relu = nn.ReLU()

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=stage(feats), size=(h, w), mode="bilinear", align_corners=True
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

    @staticmethod
    def _make_stage(in_features, size):
        """
        Makes a single psp module stage
        :param in_features: input feature dimension
        :param size:
        :return:
        """
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)


@typechecked
class PSPUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Upsampling module for psp network
        :param in_channels:
        :param out_channels:
        """
        super(PSPUpsample, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x, upsample_ratio=2):
        h, w = upsample_ratio * x.size(2), upsample_ratio * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode="bilinear", align_corners=True)
        return self.conv(p)


@typechecked
class PSPDecoder(nn.Module):
    """
    Segmentation decoder network
    Based on Pyramid Scene Parsing Network CVPR 2017
    http://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html
    """

    def __init__(
        self,
        input_dim: int = 2048,
        n_classes: int = 6,
        pooling_sizes: Tuple[int, int, int, int] = (1, 2, 3, 6),
    ):
        """
        :param input_dim: latent feature dimension received from the encoder
        :param n_classes: number of classes to classify into pixel-wise
        :param pooling_sizes: pooling dimensions
        """
        super(PSPDecoder, self).__init__()

        self.n_classes = n_classes

        self.psp = PSPModule(input_dim, 1024, pooling_sizes)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        p = self.psp(x)

        p = self.up_1(p)

        p = self.up_2(p, upsample_ratio=4)

        p = self.up_3(p, upsample_ratio=4)

        return self.final(p)
