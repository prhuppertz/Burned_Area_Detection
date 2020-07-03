from torch.utils.data import Dataset
import torch
import numpy as np
from skimage.util import view_as_windows
import rasterio
import cv2
from skimage.transform import resize
from shapely.ops import transform
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from collections import defaultdict
import os

def find_checkpoint(group):
    """
    Finds checkpoint path
    :param group:
    :return:
    """
    for root, dirs, files in os.walk("checkpoint/{}".format(group)):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)
    return checkpoint_path

def apply_to_batch(model, batch, batch_idx):
    """
    Applies neural network to a batch of samples
    :param net: Pytorch module
    :param batch: Pytorch Tensor array
    :return: Post processed image batch
    """
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            batch = batch.cuda()

        pred = model(batch).cpu().numpy()
        pred_postproc = model.configuration['postprocessing_func'](pred)

    pred_postproc = [resize(img, (122, 122), preserve_range=True).astype(np.int32)*(batch_idx+1) for img in pred_postproc]

    return pred_postproc

def prepare_patch(img, seq_test=None):
    """
    Assumes single image as an input. Returns torch tensor, with an extra dimension
    :param img: image to process
    :param seq_test: test time augmentation sequence
    :param transform: torch transform into tensor
    :return:
    """
    if seq_test:
        img = seq_test(image=img)
    img = resize(img, (128, 128))
    img = torch.from_numpy(img.transpose(2, 1, 0).astype(np.float32))
    return img.unsqueeze(0)

def normalise_patch(patch):
    """
    Normalises individual patch extracted from the test MGRS image
    :param patch:
    :return:and so my efforts are waste of time
    """
    patch = np.nan_to_num(patch)
    patch = ((patch - patch.min()) * (1 / (patch.max() - patch.min()) * 255))
    return patch

def load_stack(path_to_image):
    """
    Loads MGRS stack
    :return:
    """
    src = rasterio.open(path_to_image, 'r')
    # Read the bands we are interested in
    img_array = np.dstack(list(src.read([4, 3, 2])))
    return img_array

def split_into_patches(img_array, size=122, stride=122):
    """
    Splits large MGRS scene into multiple image patches
    :param img_array: MGRS scene
    :param size: Size of the patches
    :param stride: Stride to patch the image, ideally the same size as the image to not miss any information
    :return:
    """
    arrays = view_as_windows(img_array, (size, size, 3), stride).squeeze()
    return arrays

def reflection(x0):
    return lambda x, y: (x, 2*x0 - y)

def mask_to_polygons(mask, min_area=2.):
    """
    Convert a mask ndarray (binarized image) to Multipolygons
    :param mask:
    :param epsilon:
    param min_area:
    """
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        mask.copy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     if not hierarchy.all():
#         return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(transform(reflection(1), poly))
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons

class SceneDataset(Dataset):

    def __init__(self, arrays, seq_test=None):

        self.arrays = arrays
        self.seq_test = seq_test

    def __getitem__(self, item):
        img = self.arrays[item]
        img = normalise_patch(img)
        img = prepare_patch(img, seq_test=self.seq_test)
        return img.squeeze(), 1

    def __len__(self):
        return self.arrays.shape[0]
