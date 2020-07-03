import numpy as np
from scipy.special import expit
from typeguard import typechecked
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes
from segmentation.data.utils import batched


@typechecked
def sigmoid(targets: np.ndarray) -> np.ndarray:
    """
    Computes sigmoid probabilities over array elements
    :param targets:
    :return:
    """
    targets = expit(targets)
    return targets

@typechecked
def threshold_binary(targets: np.ndarray) -> np.ndarray:
    """
    Thresholds binary prediction
    :param targets:
    :return:
    """
    targets = targets > 0.5
    return targets.astype(np.int32)

@typechecked
def bg_to_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    Applies otsu thresholding, returns a binary mask
    :param targets:
    :return:
    """
    output = np.zeros(mask.shape)
    thresh = threshold_otsu(mask)
    binary = mask > thresh
    binary = binary_fill_holes(binary)
    output[binary] = 1
    return output

@batched(mean=False)
def batched_bg_to_segmentation(mask):
    out = bg_to_segmentation(mask)
    return out
