import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_dilation
from scipy.ndimage.morphology import (binary_erosion,
                                    binary_dilation,
                                    binary_fill_holes,
                                    distance_transform_edt)
from typeguard import typechecked
from segmentation.data.utils import batched
from typing import Optional

@typechecked
def gen_inst_dst_map(ann: np.ndarray):
    """
    This function take instance-wise segmentation and generates
     distance map for each object
    :param ann: instance-wise segmentation map
    :return: distance map of segmntation
    """
    shape = ann.shape[:2]  # HW
    nuc_list = list(np.unique(ann))
    if 0 in nuc_list:
        nuc_list.remove(0)  # 0 is background

    canvas = np.zeros(shape, dtype=np.uint8)
    for nuc_id in nuc_list:
        nuc_map = np.copy(ann == nuc_id)
        nuc_dst = distance_transform_edt(nuc_map)
        nuc_dst = 255 * (nuc_dst / np.amax(nuc_dst))
        canvas += nuc_dst.astype('uint8')
    return canvas

@typechecked
def unet_watershed(thresh_pred: np.ndarray, dilation_kernel: Optional[int] = None):
    """
    Watershed based postprocessing
    :param thresh_pred: one channel image (preferably use for small objects) that has already been thresholded
    :param dilatation_kernel: if we need dilation, pass a dilation kernel size [Default: (3, 3)]
    :return: processed prediction
    """
    instance_mask = label(thresh_pred)

    # Generating distance map of ech object
    dist = gen_inst_dst_map(instance_mask)
    marker = np.copy(dist)

    # Thresholding values of distance transform to obtain marker for mc watershed
    marker[marker <= 125] = 0
    marker[marker > 125] = 1

    # Labeling marker
    marker = label(marker)

    # Apply watershed
    pred = watershed(-dist, marker, mask=thresh_pred)

    # Remove small objects from final output
    instance_mask = remove_small_objects(pred, min_size=10)
    #dilate objects
    if dilation_kernel:

        # Kernel for dilation
        kernel = np.ones((dilation_kernel, dilation_kernel))
        uniqe_ids = np.unique(instance_mask)
        uniqe_ids = np.delete(uniqe_ids, 0)
        if len(uniqe_ids)>0:
            dilated_instance_mask = np.zeros((pred.shape[0], pred.shape[1]))
            for id in uniqe_ids:
                this_mask = instance_mask==id
                this_mask = binary_dilation(this_mask , kernel)
                dilated_instance_mask = np.where(this_mask>0, id, dilated_instance_mask)
            instance_mask = dilated_instance_mask

    return instance_mask

@batched(mean=False)
def batched_watershed(pred, dilatation_kernel):
    instance_pred = unet_watershed(pred, dilatation_kernel)
    return instance_pred
