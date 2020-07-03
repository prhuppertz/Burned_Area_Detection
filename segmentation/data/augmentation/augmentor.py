import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from typing import Tuple, Union
from typeguard import typechecked

@typechecked
class Augmentor():

    def __init__(self,
                 sequence: iaa.Sequential,
                 augment_target: bool = True):
        """
        Augmentation object
        :param sequence:
        :param augment_target:
        """
        self.sequence = sequence
        self.augment_target = augment_target

    def augmentation_func(self, image: np.ndarray, target: Union[np.ndarray, int]) -> Tuple[np.ndarray, Union[np.ndarray, int]]:
        """
        Augmentation function that transforms image and target if augment_target is True
        :param image:
        :param target:
        :return: returns a tuple of augmented images and target
        """
        if self.augment_target:
            # Check if target width and height make sense
            assert (image.shape[:-1] == target.shape), "Mask and image should be of the same " \
                                                            "shape to be augmented! Mask is of shape {}".format(target.shape)

            segmap = SegmentationMapsOnImage(target, shape=image.shape[:-1])

            image, target = self.sequence(image=image, segmentation_maps=segmap)

            target = target.get_arr()

        else:

            image = self.sequence(image=image)

        return image, target
