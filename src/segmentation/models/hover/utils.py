import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_sobel_kernel(size):
    """
    Initialises kernel for sobel operator
    https://en.wikipedia.org/wiki/Sobel_operator
    :param size:
    :return:
    """
    assert size % 2 == 1, "Must be odd, get size=%d" % size

    h_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
    v_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
    h, v = np.meshgrid(h_range, v_range)
    kernel_h = h / (h * h + v * v + 1.0e-15)
    kernel_v = v / (h * h + v * v + 1.0e-15)
    return kernel_h, kernel_v


def get_gradient(x, y):
    """
    Computes gradient(sobel edge detection) of horizontal and vertical maps
    https://en.wikipedia.org/wiki/Sobel_operator
    :param x: Horizontal distance map
    :param y: Vertical distance map
    :return:
    """
    kernel_h, kernel_v = get_sobel_kernel(5)
    # Convert kernels to pytorch tensors
    kernel_h = torch.tensor(kernel_h, requires_grad=False)  # constant
    kernel_v = torch.tensor(kernel_v, requires_grad=False)  # constant
    # Reshape and place on cuda
    kernel_h = kernel_h.view(1, 1, 5, 5).to("cuda")  # constant
    kernel_v = kernel_v.view(1, 1, 5, 5).to("cuda")  # constant

    # Apply kernel
    horizontal = F.conv2d(x, kernel_h, padding=2)
    vertical = F.conv2d(y, kernel_v, padding=2)
    hv = torch.cat([horizontal, vertical], dim=1)

    return hv


def gradient_mse(pred_x, pred_y, x, y, focus):
    """
    Computes mean squarred error between horizontal/vertical gradient maps and ground truth gradient
    If focus is passed then will compute MSE only within focus areas.
    :param pred_x: Predicted horizontal map
    :param pred_y: Preticted vertical map
    :param x: True horizontal map
    :param y: True vertical map
    :param focus: Binary semantic segmentation mask
    :return:
    """
    pred_grad = get_gradient(pred_x, pred_y)

    true_grad = get_gradient(x, y)

    if focus != None:
        loss = pred_grad - true_grad
        loss = focus.unsqueeze(1).repeat(1, 2, 1, 1).float() * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
    else:
        loss = nn.MSELoss(reduction="mean")(pred_grad, true_grad)
    return loss


def hover_loss(predicted_targets, targets, focus=None):
    """
    Complete hovernet loss
    https://arxiv.org/abs/1812.06499
    The loss will compute binary cross entropy loss between segmentation masks, and MSE loss between distance maps
    as well as MSE between sobel applied distance maps
    :param predicted_targets: [num_channels, width, height] where num_channels = 3 for segmentation, horizontal and vertical maps
    :param targets: [num_channels, width, height] where num_channels = 3 for segmentation, horizontal and vertical maps
    :param focus: Binary semantic segmentation mask
    :return:
    """
    pred_seg_map, pred_x, pred_y = (
        predicted_targets[:, 0, :, :].unsqueeze(1),
        predicted_targets[:, 1, :, :].unsqueeze(1),
        predicted_targets[:, 2, :, :].unsqueeze(1),
    )

    seg_map, x, y = (
        targets[:, 0, :, :].unsqueeze(1),
        targets[:, 1, :, :].unsqueeze(1),
        targets[:, 2, :, :].unsqueeze(1),
    )

    seg_map_loss = nn.BCEWithLogitsLoss(reduction="mean")(pred_seg_map, seg_map)

    x_loss = nn.MSELoss(reduction="mean")(pred_x, x)

    y_loss = nn.MSELoss(reduction="mean")(pred_y, y)

    grad_loss = gradient_mse(pred_x, pred_y, x, y, focus)

    loss = seg_map_loss + x_loss + y_loss + grad_loss

    return loss
