from segmentation.data.postprocessing.classification import (
    sigmoid,
    batched_bg_to_segmentation,
    bg_to_segmentation,
)

# from segmentation.data.postprocessing.watershed import batched_watershed, unet_watershed


import numpy as np
from typeguard import typechecked
from typing import Tuple, Optional


@typechecked
class Postprocessor:
    def __init__(self, dilation_kernel: Optional[int] = None):

        self.dilation_kernel = dilation_kernel

    def postprocessing_func(self, pred: np.ndarray) -> np.ndarray:
        """
        Applies an ordred list of post-processing functions
        :param pred: (6. width, height) array, where first channel is background
        :return: returns a postprocessed array
        """
        # Get semantic segmentation mask
        segmentation_masks = batched_bg_to_segmentation(sigmoid(pred))

        # First apply instance segmentation post-processing
        # instance_masks = batched_watershed(segmentation_masks, dilatation_kernel=self.dilation_kernel)

        return segmentation_masks

    def postprocessing_func_single(self, pred: np.ndarray) -> np.ndarray:
        """
        Applies an ordred list of post-processing functions
        :param pred: (6. width, height) array, where first channel is background
        :return: returns a postprocessed array
        """
        # Get semantic segmentation mask
        segmentation_masks = bg_to_segmentation(sigmoid(pred))

        # First apply instance segmentation post-processing
        # instance_mask = unet_watershed(segmentation_masks, dilation_kernel=self.dilation_kernel)

        return segmentation_masks
