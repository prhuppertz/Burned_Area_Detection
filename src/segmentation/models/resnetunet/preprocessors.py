import torch
import numpy as np
from typeguard import typechecked
from typing import List, Callable, Union, Tuple
from src.segmentation.data.preprocessing.mask_functions import make_instance_mask
from src.segmentation.data.preprocessing.distance_maps.unet import get_unet_map
import torch


@typechecked
class Preprocessor:
    def __init__(self, w: int = 10, sigma: int = 5):
        self.w = w
        self.sigma = sigma

    def preprocessing_func(self, image: np.ndarray, target: np.ndarray):
        """
        Applies an ordred list of preprocessing functionse
        :param image:
        :param target:
        :return: a tuple with an image where channels are first
                and a mask, where first channels are labels, second channel is unet distance map, and the remaining
                channels are instance maps for ground truth computation
        """
        # Put image channel first
        image = image.transpose(2, 0, 1)

        distance_mask = get_unet_map(target, w0=self.w, sigma=self.sigma)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        target_mask = torch.from_numpy((target != 0)).float()

        distance_mask = torch.from_numpy(distance_mask).float()

        gt = torch.from_numpy(target)

        return image, (target_mask, distance_mask, gt)
