"""
Usage:
        sort_patches.py <save_path> <all_extracted_path>  <path_to_annotations_source>

@ Robert Huppertz, Jevgenij Gamper,  Cervest 2020
Sort patches from all_extracted_path and annotations from path_to_annotations_source 
to save_path if they are not empty.

"""

import os
import pickle
import shutil
from pathlib import Path
from docopt import docopt
from src.preparedata.patchutils.coco import coco_to_shapely
from PIL import Image as pilimage
import numpy as np
from src.preparedata.patchutils import other
import rasterio as rio


def dump_polygons(annotation_dir_source, save_path, included):
    """
    Merge multiple MGRS annotation files into one pickle file 

    :param annotation_dir_source: directory path for the annotations
    :param save_path: Path to save the pickle file
    :param included: list of numbers of patches that are neither in the excluded.txt file nor empty
    :return:
    """
    all_annotations = {}
    mgrs_annotations_files = os.listdir(annotation_dir_source)

    for annotations_file in mgrs_annotations_files:
        if "annotations" in other.load_json(
            os.path.join(annotation_dir_source, annotations_file)
        ):
            annotations_file = coco_to_shapely(
                os.path.join(annotation_dir_source, annotations_file)
            )
            all_annotations = {**all_annotations, **annotations_file}

    included_annotations = {}

    for in_test_img in included:
        included_annotations[in_test_img + ".jpg"] = all_annotations[
            in_test_img + ".jpg"
        ]

    with open(save_path, "wb") as f:
        pickle.dump(included_annotations, f, pickle.HIGHEST_PROTOCOL)


def get_all(path):
    """
    Returns all files in path 

    :param path:
    :return:
    """
    files = [f.split(".")[0] for f in os.listdir(path)]
    return files


def main(
    save_path, all_extracted_path, path_to_annotations_source, delete=False,
):
    """
    :param save_path: path where newly sorted patches and their annotations will be stored
    :param excluded_path: path to where image names that will be excluded are stored
    :param all_extracted_path: path to where all images that have been extracted from MGRS files are stored
    :param path_to_annotations_source: path to where extracted source annotation files are stored
    :param delete: if all extracted (non-filtered) patches should be deleted [default=False]
    :return:
    """

    all_patches = get_all(all_extracted_path)
    all_patches = [i for i in all_patches if i]

    # Make directories to store patches and annotation files
    path_to_store_patches = os.path.join(save_path, "patches")
    os.makedirs(path_to_store_patches, exist_ok=True)
    path_to_store_annotations = os.path.join(save_path, "annotations")
    os.makedirs(path_to_store_annotations, exist_ok=True)

    included = []

    # Move patches if not empty
    for patch in all_patches:
        source_patch_path = os.path.join(all_extracted_path, patch + ".jpg")
        target_patch_path = os.path.join(path_to_store_patches, patch + ".jpg")

        img = np.asarray(pilimage.open(source_patch_path))
        # test if patch is empty
        if np.unique(img).size > 2:
            included.append(patch)
            shutil.copy(source_patch_path, target_patch_path)

    # print number of patches that are included
    print("Number of included patches {}".format(len(included)))

    # Combine annotations to a pickle file
    save_annotations_file = os.path.join(path_to_store_annotations, "polygons.pkl")
    dump_polygons(path_to_annotations_source, save_annotations_file, included)

    # Remove patches and their directory if not to be used for pipelines
    if delete:
        os.remove(str(Path(all_extracted_path).parents[0]))


if __name__ == "__main__":
    arguments = docopt(__doc__)
    save_path = arguments["<save_path>"]
    all_extracted_path = arguments["<all_extracted_path>"]
    path_to_annotations_source = arguments["<path_to_annotations_source>"]
    main(save_path, all_extracted_path, path_to_annotations_source)
