from fastai import torch_core
import numpy as np
import torch


def dice(
    input: np.ndarray, targs: np.ndarray, iou: bool = False, eps: float = 1e-8
) -> float:
    "Returns the dice coefficient for a binary target"
    n = targs.shape[0]
    input = torch.from_numpy(input).argmax(dim=1).view(n, -1)
    targs = torch.from_numpy(targs).view(n, -1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input + targs).sum(dim=1).float()
    if not iou:
        l = 2.0 * intersect / union
    else:
        l = intersect / (union - intersect + eps)
    l[union == 0.0] = 1.0

    return l.mean().item()


def dice_array(input: np.ndarray, targs: np.ndarray, empty_score=1.0):

    input = np.asarray(input).astype(np.bool)
    target = np.asarray(targs).astype(np.bool)

    im_sum = np.abs(input.sum()) + np.abs(target.sum())
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(input, target)

    return 2.0 * np.abs(intersection.sum()) / im_sum


def iou(
    input: np.ndarray, targs: np.ndarray, iou: bool = True, eps: float = 1e-8
) -> float:
    "Returns the IoU metric for a binary target"
    n = targs.shape[0]
    input = torch.from_numpy(input).argmax(dim=1).view(n, -1)
    targs = torch.from_numpy(targs).view(n, -1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input + targs).sum(dim=1).float()
    if not iou:
        l = 2.0 * intersect / union
    else:
        l = intersect / (union - intersect + eps)
    l[union == 0.0] = 1.0

    return l.mean().item()


def iou_array(
    input: np.ndarray, targs: np.ndarray, eps: float = 1e-8, empty_score: float = 1.0
):

    input = np.asarray(input).astype(np.bool)
    target = np.asarray(targs).astype(np.bool)

    im_sum = input.sum() + target.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(input, target)

    return np.abs(intersection.sum()) / (np.abs(im_sum - intersection.sum()) + eps)


def dice_and_iou(input: np.ndarray, targs: np.ndarray) -> tuple:
    "Returns the dice and IoU metric in a tuple"
    iou_score = iou(input, targs)
    dice_score = dice(input, targs)
    return_tuple = (iou_score, dice_score)

    return return_tuple


def dice_and_iou_arrays(input: np.ndarray, targs: np.ndarray) -> tuple:
    "Returns the dice and IoU metric in a tuple"
    iou_score = iou_array(input, targs)
    dice_score = dice_array(input, targs)
    return_tuple = (iou_score, dice_score)

    return return_tuple
