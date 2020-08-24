from typing import Union, List, Dict
from pathlib import Path
import itertools
import os
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from PIL import Image as pilimage
from patchutils import other


def format_coco(chip_dfs: Dict, patch_size: int, row_name: str):
    """
    Format train and test chip geometries to COCO json format.
    COCO train and val set have specific ids.
    """
    chip_height, chip_width = patch_size, patch_size
    cocojson = {
        "info": {},
        "licenses": [],
        "categories": [
            {
                "supercategory": "Burned Areas",
                "id": 1,  # id needs to match category_id.
                "name": "agfields_singleclass",
            }
        ],
    }

    for key_idx, key in enumerate(chip_dfs.keys()):

        key_image = {
            "file_name": f"{key}.jpg",
            "id": int(key_idx),
            "height": chip_width,
            "width": chip_height,
        }
        cocojson.setdefault("images", []).append(key_image)

        for row_idx, row in chip_dfs[key]["chip_df"].iterrows():
            # Convert geometry to COCO segmentation format:
            # From shapely POLYGON ((x y, x1 y2, ..)) to COCO [[x, y, x1, y1, ..]].
            # The annotations were encoded by RLE, except for crowd region (iscrowd=1)
            coco_xy = list(
                itertools.chain.from_iterable(
                    (x, y) for x, y in zip(*row.geometry.exterior.coords.xy)
                )
            )
            coco_xy = [round(coords, 2) for coords in coco_xy]
            # Add COCO bbox in format [minx, miny, width, height]
            bounds = row.geometry.bounds  # COCO bbox
            coco_bbox = [
                bounds[0],
                bounds[1],
                bounds[2] - bounds[0],
                bounds[3] - bounds[1],
            ]
            coco_bbox = [round(coords, 2) for coords in coco_bbox]

            key_annotation = {
                "id": key_idx,
                "image_id": int(key_idx),
                "category_id": 1,  # with multiple classes use "category_id" : row.reclass_id
                "mycategory_name": "agfields_singleclass",
                "old_multiclass_category_name": row[row_name],
                "bbox": coco_bbox,
                "area": row.geometry.area,
                "iscrowd": 0,
                "segmentation": [coco_xy],
            }
            cocojson.setdefault("annotations", []).append(key_annotation)

    return cocojson


def coco_to_shapely(
    inpath_json: Union[Path, str], categories: List[int] = None
) -> Dict:
    """Transforms COCO annotations to shapely geometry format.
    Args:
        inpath_json: Input filepath coco json file.
        categories: Categories will filter to specific categories and images that contain at least one
        annotation of that category.
    Returns:
        Dictionary of image key and shapely Multipolygon.
    """
    data = other.load_json(inpath_json)
    if categories is not None:
        # Get image ids/file names that contain at least one annotation of the selected categories.
        image_ids = sorted(
            list(
                set(
                    [
                        x["image_id"]
                        for x in data["annotations"]
                        if x["category_id"] in categories
                    ]
                )
            )
        )
    else:
        image_ids = sorted(list(set([x["image_id"] for x in data["annotations"]])))
    file_names = [x["file_name"] for x in data["images"] if x["id"] in image_ids]

    # Extract selected annotations per image.
    extracted_geometries = {}
    for image_id, file_name in zip(image_ids, file_names):
        annotations = [x for x in data["annotations"] if x["image_id"] == image_id]
        if categories is not None:
            annotations = [x for x in annotations if x["category_id"] in categories]

        segments = [
            segment["segmentation"][0] for segment in annotations
        ]  # format [x,y,x1,y1,...]

        # Create shapely Multipolygons from COCO format polygons.
        mp = MultiPolygon(
            [
                Polygon(np.array(segment).reshape((int(len(segment) / 2), 2)))
                for segment in segments
            ]
        )
        extracted_geometries[str(file_name)] = mp

    return extracted_geometries


def plot_coco(inpath_json, inpath_image_folder, start=0, end=2):
    """Plot COCO annotations and image chips"""
    extracted = coco_to_shapely(inpath_json)

    for key in sorted(extracted.keys())[start:end]:
        print(key)
        plt.figure(figsize=(5, 5))
        plt.axis("off")

        img = np.asarray(pilimage.open(os.path.join(inpath_image_folder, key)))
        plt.imshow(img, interpolation="none")

        mp = extracted[key]
        patches = [
            PolygonPatch(p, ec="r", fill=False, alpha=1, lw=0.7, zorder=1) for p in mp
        ]
        plt.gca().add_collection(PatchCollection(patches, match_original=True))
        plt.show()


def save_gt_overlaid(inpath_json, inpath_image_folder, save_path):
    """
    Takes all the data and its ground truth and stores the overlaid image
    :param inpath_json:
    :param inpath_image_folder:
    :param raster_name:
    :param save_path:
    :return:
    """
    extracted = coco_to_shapely(inpath_json)
    for key in sorted(extracted.keys()):
        plt.figure(figsize=(5, 5))
        plt.axis("off")

        img = np.asarray(pilimage.open(os.path.join(inpath_image_folder, key)))
        plt.imshow(img, interpolation="none")

        # ADDED by Robert for skipping empty patches
        if np.unique(img).size > 2:
            mp = extracted[key]
            patches = [
                PolygonPatch(p, ec="r", fill=False, alpha=1, lw=0.7, zorder=1)
                for p in mp
            ]
            plt.gca().add_collection(PatchCollection(patches, match_original=True))
            plt.savefig(os.path.join(save_path, key), dpi=800)

        plt.close("all")