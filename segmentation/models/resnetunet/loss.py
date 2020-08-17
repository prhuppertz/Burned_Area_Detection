import torch.nn.functional as F
from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def loss(predicted_targets, label_mask, distance_mask):
    """

    :param predicted_targets:
    :param label_mask:
    :param distance_mask:
    :return:
    """
    print(predicted_targets.size())
    print(torch.squeeze(predicted_targets).size())
    print(label_mask.size())
    print(torch.squeeze(label_mask).size())
    #using torch.squeeze(tensor) instead of .squeeze()
    loss = F.binary_cross_entropy_with_logits(torch.squeeze(predicted_targets), torch.squeeze(label_mask), reduction='none')
   
    loss = loss * distance_mask

    return loss.mean() + DiceLoss().forward(predicted_targets, label_mask)
