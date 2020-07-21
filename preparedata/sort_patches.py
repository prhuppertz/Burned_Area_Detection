"""Usage:
          sort_patches.py <save_path> <excluded_path> <all_extracted_path>  <path_to_annotations _source>

@ Jevgenij Gamper & Robert Huppertz 2020, Cervest
Sort patches and annotations to training folders if they are not excluded nor empty

Options:
  -h --help             Show help.
  --version             Show version.
  --cache=<cache_path>  Inference mode. 'roi' or 'wsi'. [default: data/]
  --link=<link>         Path to for symlink to points towards, if using remote storage
"""
import os
import pickle
import shutil
from pathlib import Path
from docopt import docopt
from patchutils.coco import coco_to_shapely
from PIL import Image as pilimage
import numpy as np
from patchutils import other


def dump_polygons(annotation_dir_source, save_path, included):
    """

    :param annotation_dir_source: 
    :param save_path: Path to file to save the annotations used for pipelines and testing eg polygons.pkl
    :param included:
    :return:
    """
    # Merge multiple mgrs annotation files
    all_annotations = {}
    mgrs_annotations_files = os.listdir(annotation_dir_source)
    for annotations_file in mgrs_annotations_files:
        path = os.path.join(annotation_dir_source, annotations_file)
        if "annotations" in other.load_json(os.path.join(annotation_dir_source, annotations_file)):
            annotations_file = coco_to_shapely(os.path.join(annotation_dir_source, annotations_file))
            all_annotations = {**all_annotations, **annotations_file}

    included_annotations = {}
    for in_test_img in included:
        included_annotations[in_test_img + ".jpg"] = all_annotations[
            in_test_img + ".jpg"
        ]

    with open(save_path, "wb") as f:
        pickle.dump(included_annotations, f, pickle.HIGHEST_PROTOCOL)


def read_excluded(path):
    """
    Reads all images that are meant to be excluded and returns a list
    :param path:
    :return:
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    lines = [line.split("\n")[0] for line in lines]
    return lines


def get_all(path):
    """
    Get all patches that have been extracted
    :param path:
    :return:
    """
    files = [f.split(".")[0] for f in os.listdir(path)]
    return files


def main(
    save_path, excluded_path, all_extracted_path, path_to_annotations_source, delete=False
):
    """
    :param save_path: path where newly sorted patches and their annotations will be stored
    :param excluded_path: path to where image names that will be excluded are stored
    :param all_extracted_path: path to where all images that have been extracted from MGRS files are stored
    :param path_to_annotations_source: path to where extracted source annotation files are stored
    :param delete: if all extracted (non-filtered) patches should be deleted [default=False]
    :return:
    """

    # Read files that have been selected to exclude from pipelines
    # Both all and excluded, returned without extension
    excluded_patches = read_excluded(excluded_path) if excluded_path else []
    print(len(excluded_patches))
    all_patches = get_all(all_extracted_path)
    all_patches = [i for i in all_patches if i]

    # Make directories to store filtered patches and ground truth
    path_to_store_patches = os.path.join(save_path, "patches")
    os.makedirs(path_to_store_patches, exist_ok=True)
    path_to_store_annotations = os.path.join(save_path, "annotations")
    os.makedirs(path_to_store_annotations , exist_ok=True)

    # Move image patches
    included = []  # To pass into dump_polygons
    for patch in all_patches:
        
        if patch not in excluded_patches:
            source_patch_path = os.path.join(all_extracted_path, patch + ".jpg")
            target_patch_path = os.path.join(path_to_store_patches, patch + ".jpg")
            #ROBERT:test if patch is empty
            img = np.asarray(pilimage.open(source_patch_path))
            if np.unique(img).size > 2:
                included.append(patch)
                shutil.move(source_patch_path, target_patch_path)
                
        
    print("Number of included patches {}".format(len(included)))
    # Combine and move annotations to new directory
    save_annotations_file = os.path.join(path_to_store_annotations, "polygons.pkl")
    dump_polygons(path_to_annotations_source, save_annotations_file, included)

    # Remove patches and their directory if not to be used for pipelines
    if delete:
        os.remove(str(Path(all_extracted_path).parents[0]))


if __name__ == "__main__":
    arguments = docopt(__doc__)
    save_path = arguments["<save_path>"]
    excluded_path = arguments["<excluded_path>"]
    all_extracted_path = arguments["<all_extracted_path>"]
    path_to_annotations_source = arguments["<path_to_annotations_source>"]
    main(save_path, excluded_path, all_extracted_path, path_to_annotations_source)
