"""
Adversary for adversarial segmentation loss
"""
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(32 * 5 * 5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 5 * 5)

        x = self.fc(x)

        return self.sigmoid(x)
