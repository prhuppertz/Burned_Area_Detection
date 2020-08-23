from torch import nn
import torchvision.models as models
from typeguard import typechecked


@typechecked
class ResNet(nn.Module):
    def __init__(self, encoder_name: str, pretrained: bool = False):
        """

        :param encoder_name:
        :param pretrained:
        """
        super(ResNet, self).__init__()

        self.base_layers = get_encoder(encoder_name, pretrained)

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(*self.base_layers[0:2])
        self.layer2 = self.base_layers[2]
        self.layer3 = self.base_layers[3]
        self.layer4 = self.base_layers[4]

    def forward(self, x):
        """
        Forward pass through feature extraction network
        :param x: Input image
        :return: Returns feature outputs at different stages of the networks
            resnet18
            Layer 0: torch.Size([1, 64, 112, 112]); Latent size multiple: 8.0
            Layer 1: torch.Size([1, 64, 56, 56]); Latent size multiple: 8.0
            Layer 2: torch.Size([1, 128, 28, 28]); Latent size multiple: 4.0
            Layer 3: torch.Size([1, 256, 14, 14]); Latent size multiple: 2.0
            Layer 4: torch.Size([1, 512, 7, 7]); Latent size multiple: 1.0 (4, 4 for 128x128 image)

            resnet50
            Layer 0: torch.Size([1, 64, 112, 112]); Latent size multiple: 32.0
            Layer 1: torch.Size([1, 256, 56, 56]); Latent size multiple: 8.0
            Layer 2: torch.Size([1, 512, 28, 28]); Latent size multiple: 4.0
            Layer 3: torch.Size([1, 1024, 14, 14]); Latent size multiple: 2.0
            Layer 4: torch.Size([1, 2048, 7, 7]); Latent size multiple: 1.0

        """
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer0, layer1, layer2, layer3, layer4


def get_encoder(encoder_name, pretrained):
    if encoder_name == "resnet18":
        base_model = models.resnet18(pretrained=pretrained)
    elif encoder_name == "resnet50":
        base_model = models.resnet50(pretrained=pretrained)
    elif encoder_name == "resnet101":
        base_model = models.resnet101(pretrained=pretrained)
    else:
        raise Exception("Unspecified model name: {}".format(encoder_name))
    return list(base_model.children())[3:8]
