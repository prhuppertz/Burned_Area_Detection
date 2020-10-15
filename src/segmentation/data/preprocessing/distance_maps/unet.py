import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from typing import List


def get_unet_map(
    mask: np.ndarray, w0: int = 10, sigma: int = 5, wc: List[int] = [2, 1]
) -> np.ndarray:
    """

    :param mask: A single channel array with every instance having its pixels
        populated with a unique digit
    :param w0:
    :param sigma:
    :param wc: weights of the classes, first is background second element is nuclei
    :return: A single channel per pixel weights array
    """
    unique_list = np.unique(mask)[1:]

    wc = (mask == 0) * wc[0] + (mask != 0) * wc[1]

    stacked_inst_bgd_dst = np.zeros(mask.shape[:2] + (len(unique_list),))

    if len(unique_list) == 0:
        return np.zeros(mask.shape)

    for idx, unique in enumerate(unique_list):
        inst_bgd_map = np.array(mask != unique, np.uint8)
        inst_bgd_dst = distance_transform_edt(inst_bgd_map)
        stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

    near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)

    near2_dst = np.expand_dims(near1_dst, axis=2)

    near2_dst = stacked_inst_bgd_dst - near2_dst
    near2_dst[near2_dst == 0] = np.PINF  # very large
    near2_dst = np.amin(near2_dst, axis=2)
    near2_dst[mask > 0] = 0  # the instances
    near2_dst = near2_dst + near1_dst
    # to fix pixel where near1 == near2
    near2_eve = np.expand_dims(near1_dst, axis=2)
    # to avoide the warning of a / 0
    near2_eve = (1.0 + stacked_inst_bgd_dst) / (1.0 + near2_eve)
    near2_eve[near2_eve != 1] = 0  # Comment this to just have boundaries
    near2_eve = np.sum(near2_eve, axis=2)
    near2_dst[near2_eve > 1] = near1_dst[near2_eve > 1]

    pix_dst = near1_dst + near2_dst
    pen_map = pix_dst / sigma
    pen_map = w0 * np.exp(-(pen_map ** 2) / 2)
    pen_map[mask > 0] = 0  # inner instances zero

    return pen_map + wc
