from typeguard import typechecked
import numpy as np
import cv2
from skimage.measure import label as _label
from skimage.filters import threshold_otsu
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects, watershed
from segmentation.data.utils import batched


@typechecked
def _otsu_threshold(targets: np.ndarray) -> np.ndarray:
    """
    Applies unet otsu thresholding, returns a binary mask
    :param targets:
    :return:
    """
    output = np.zeros(targets.shape)
    thresh = threshold_otsu(targets)
    binary = targets < thresh
    binary = binary_fill_holes(binary)
    output[binary] = 1
    return output


@typechecked
def _xy_watershed(targets: np.ndarray):
    """
    Watershed with marker initialisation from x,y map gradients.
    Applies sobel operators to horizontal and vetical maps and takes max of the both.
    Uses these as an initialisation for watershed algorithm.
    :param targets: Array of shape (num_channels, height, width) where num_channels = 3 for
        pixel probabilities, horizontal maps, and vertical maps respectively
    :return: Instance segmentation mask
    """
    # Split into segmentation, horizontal, and vertical predictions
    pixel_map = targets[0]
    horizontal = targets[1]
    vertical = targets[2]

    # Threshold pixel predictions
    blb = np.copy(pixel_map)
    blb[blb >= 0.5] = 1
    blb[blb < 0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # back ground is 0 already

    horizontal = cv2.normalize(
        horizontal, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    vertical = cv2.normalize(
        vertical, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # Applies Sobel operator
    horizontal_grad = cv2.Sobel(horizontal, cv2.CV_64F, 1, 0, ksize=21)
    vertical_grad = cv2.Sobel(vertical, cv2.CV_64F, 0, 1, ksize=21)

    horizontal_grad = 1 - (
        cv2.normalize(
            horizontal_grad,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )
    vertical_grad = 1 - (
        cv2.normalize(
            vertical_grad,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )

    # Takes the maximum between horizontal and vertical gradient maps
    overall = np.maximum(horizontal_grad, vertical_grad)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb

    # Field values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall[overall >= 0.4] = 1
    overall[overall < 0.4] = 0

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, marker, mask=blb)

    return proced_pred


@batched(mean=False)
def otsu_threshold(targets: np.ndarray) -> np.ndarray:
    res = _otsu_threshold(targets)
    return res


@batched(mean=False)
def label(targets_binary: np.ndarray) -> np.ndarray:
    res = _label(targets_binary)
    return res


@batched(mean=False)
def xy_watershed(targets_binary: np.ndarray) -> np.ndarray:
    res = _xy_watershed(targets_binary)
    return res
