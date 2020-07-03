import torch
from torch import nn

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UnetDecoder(nn.Module):

    def __init__(self, n_classes: int, encoder_name: str, multiple: int):
        """
        Unet like resnet decoder
        :param n_classes:
        :param multiple:
        """
        super(UnetDecoder, self).__init__()

        if encoder_name == "resnet18":
            latent_dims = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            latent_dims = [64, 256, 512, 1024, 2048]
        elif encoder_name == "resnet101":
            latent_dims = [64, 256, 512, 1024, 2048]

        multiple = int(multiple)

        self.layer0_1x1 = convrelu(latent_dims[0], 8 * multiple, 1, 0)
        self.layer1_1x1 = convrelu(latent_dims[1], 8 * multiple, 1, 0)
        self.layer2_1x1 = convrelu(latent_dims[2], 16 * multiple, 1, 0)
        self.layer3_1x1 = convrelu(latent_dims[3], 32 * multiple, 1, 0)
        self.layer4_1x1 = convrelu(latent_dims[4], 128 * multiple, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(32 * multiple + 128 * multiple, 128 * multiple, 3, 1)
        self.conv_up2 = convrelu(16 * multiple + 128 * multiple, 32 * multiple, 3, 1)
        self.conv_up1 = convrelu(8 * multiple + 32 * multiple, 32 * multiple, 3, 1)
        self.conv_up0 = convrelu(8 * multiple + 32 * multiple, 16 * multiple, 3, 1)

        self.conv_original_size0 = convrelu(3, 8 * multiple, 3, 1)

        self.conv_original_size2 = convrelu(8 * multiple + 16 * multiple, 8 * multiple, 3, 1)

        self.conv_last = nn.Conv2d(8 * multiple, n_classes, 1)

    def forward(self, x, layer0, layer1, layer2, layer3, layer4):
        x_original = self.conv_original_size0(x)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
