import cv2
import torch
import numpy as np
from typing import Tuple, Union, List
from typeguard import typechecked
from skimage.morphology import binary_erosion

@typechecked
def append_semantic_mask_add_xy(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns xy map as well as the original mask
    :param mask:
    :return:
    """
    x, y = get_xy_maps(mask)
    semantic_mask = erode_objects(mask)
    return image, np.stack([mask, semantic_mask, x, y])

@typechecked()
def erode_objects(mask: np.ndarray, area_threshold: int = 6) -> np.ndarray:
    """
    Erodes objects for segmentation, reduces instance segmentation mask to semantic segmentation mask
    i.e. instances are removed, and gaps are larger
    :param mask:
    :param area_threshold:
    :return:
    """
    number_of_objects = np.unique(mask)[1:]

    kernel1 = np.ones((3,3),np.uint8)
    # kernel2 is for objects with objects with area greater than area threshold
    kernel2 = np.ones((3,3),np.uint8)

    final_erosion = np.zeros((mask.shape[0], mask.shape[1]))

    if len(number_of_objects)!=0:
        for obj_num in number_of_objects:
            this_obj = mask==obj_num
            a , b = np.nonzero(this_obj)
            if len(a) < area_threshold:
                erosion = binary_erosion(this_obj, selem=kernel1, out=None)
                final_erosion = final_erosion + erosion

            else:
                erosion = binary_erosion(this_obj, selem=kernel2, out=None)
                final_erosion = final_erosion + erosion
        return final_erosion

    else:
        return final_erosion

@typechecked
def make_instance_mask(polygons, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Makes instance masks, where every field has its own unique id
    :param polygons: list of polygons
    :img_shape: tuple size of an image
    :return:
    """
    mask_instance = instance_mask_for_polygons(polygons, img_shape)
    #if mask_instance.ndim > 2:
    mask_multi = make_multi_channel(mask_instance)
    #else:
    #    mask_multi=mask_instance
    
    mask = np.amax(mask_multi, 0).astype(np.int32)
    return mask

@typechecked
def make_multi_channel(mask: np.ndarray) -> np.ndarray:
    """
    Converts single channel mask to multiple channel mask given the unique id's within it
    :param mask: mask where every shape's pixel are laveled with uniq number associated with that shape
    :return:
    """
    channels = []
    #ROBERT: changed from [1:] to [0:] because of errors with np.unique(mask)= [0] or [1]
    for instance in np.unique(mask)[0:]:
        channel = np.zeros(mask.shape)
        channel[mask == instance] = instance
        channels.append(erode_objects(channel)*instance)
    return np.stack(channels, 0)

@typechecked
def instance_mask_for_polygons(polygons, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon or multipolygon list back to an image mask ndarray
    :param polygons: list of polygons
    :img_size: tuple size of an image
    :return: img mask array
    """
    img_mask = np.zeros(img_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    for i, poly in enumerate(polygons):
        exteriors = [int_coords(poly.exterior.coords)]
        cv2.fillPoly(img_mask, exteriors, i+1)
    return img_mask
