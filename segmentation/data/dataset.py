import os
import numpy as np
from typeguard import typechecked
from torch.utils.data import Dataset
import cv2
import pickle
import glob
from typing import Callable, Optional, Tuple
from segmentation.data.preprocessing.mask_functions import make_instance_mask


@typechecked
class SegmentationData(Dataset):
    def __init__(
        self,
        root: str,
        augmentation_func: Optional[Callable[[np.ndarray, np.ndarray], Tuple]] = None,
        preprocessing_func: Optional[Callable[[np.ndarray, np.ndarray], Tuple]] = None,
    ):
        """
        Instantiates pytorch dataset for any segmentation task
        :param root: path to where dataset is stored, generally "data/name"
        :param augmentation_func: callable that contains imgaug augmentations for mask and image
        :param preprocessing_func: callable that converts to differet ground truth types,
            such as segmentation, classification, and vertical and horizontal channels for example in HoVer-net
            or classification and segmentation channels with distance map for Unet, etc
        """
        self.root = root
        self.augmentation_func = augmentation_func
        self.preprocessing_func = preprocessing_func

        # Get image names
        self.image_names = glob.glob(os.path.join(root, "patches", "*.tif"))

        # Prepare instance based GT
        path = os.path.join(root, "annotations", "polygons.pkl")
        with open(path, "rb") as f:
            self.instance_dict = pickle.load(f)

    def __len__(self):
        return len(self.image_names)

    def _get_mask(self, img_name: str, img_size: Tuple[int, int]) -> np.ndarray:
        """
        Returns image mask
        :param img_name:
        :param img_size:
        :return:
        """
        polygons = self.instance_dict[img_name]

        instance_mask = make_instance_mask(polygons, img_size)
        return instance_mask

    def __getitem__(self, item):
        """
        Loads image and its mask
        :param item:
        :return:
        """
        image_path = self.image_names[item]
        image_name = os.path.basename(image_path)
        # Read image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Read mask

        targets = self._get_mask(image_name, image.shape[:-1])

        # Augment original image and mask
        if self.augmentation_func:
            image, targets = self.augmentation_func(image, targets)

        # Get the type of ground-truth we would like to work with
        if self.preprocessing_func:
            image, targets = self.preprocessing_func(image, targets)

        return image, targets


@typechecked
def get_segmentation_dataset(
    root: str,
    augmentation_func: Optional[Callable[[np.ndarray, np.ndarray], Tuple]] = None,
    preprocessing_func: Optional[Callable[[np.ndarray, np.ndarray], Tuple]] = None,
):
    """
    """
    dataset = SegmentationData(root, augmentation_func, preprocessing_func)

    return dataset
