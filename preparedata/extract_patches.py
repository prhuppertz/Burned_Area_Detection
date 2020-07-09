"""Usage:
          extract_patches.py <shapefile>  <save_path> <class_name> <scenes_path> (-s <mgrs>)...

@ Jevgenij Gamper 2020, Cervest
Extract image patches from selected scences and stored the produced images along with their ground truth.

Options:
    --patch_size=<ps>  Size of patches, stands for both width and height [default: 128].
    --num_patches=<ns>  Number of patches to extract from a scene[default: 1000].
"""
from patching import (
    import_shapefile_for_patches,
    do_the_patching,
    store_coco_ground_truth,
    import_image,
)
from patchutils.coco import save_gt_overlaid
from docopt import docopt
import os


def main(
    shapefile,
    save_path,
    class_name,
    scenes_path,
    scenes,
    patch_size=128,
    num_patches=1000,
):

    # Make paths
    path_to_store_patches = os.path.join(save_path, "patches")
    os.makedirs(path_to_store_patches, exist_ok=True)
    path_to_store_anno = os.path.join(save_path, "anno")
    os.makedirs(path_to_store_anno, exist_ok=True)
    # What is field_vis?
    path_to_store_field_vis = os.path.join(save_path, "field_vis")
    os.makedirs(path_to_store_field_vis, exist_ok=True)

    for scene_mgrs in scenes:

        composite_path = os.path.join(
            scenes_path, scene_mgrs, "{}.jp2".format(scene_mgrs)
        )

        # Read composite and import its metadata
        raster = import_image(composite_path)
        raster_meta = raster.meta

        # Load the shapefile
        patch_dfs, patch_windows = import_shapefile_for_patches(
            shapefile, raster, raster_meta, patch_size, num_patches, scene_mgrs
        )

        # Patch the shapefile, and store image patches
        # What are the bands for?
        do_the_patching(
            raster,
            path_to_store_patches,
            patch_windows.keys(),
            patch_windows.values(),
            bands=[4, 3, 2],
        )

        # Save annotations
        store_coco_ground_truth(
            path_to_store_anno, patch_dfs, patch_size, class_name, scene_mgrs
        )

        try:
            save_gt_overlaid(
                os.path.join(path_to_store_anno, "anno{}.json".format(scene_mgrs)),
                path_to_store_patches,
                path_to_store_field_vis,
            )
        except Exception as e:
            print(e)
            print("MGRS tile without annotations: {}".format(scene_mgrs))

        raster.close()


if __name__ == "__main__":
    arguments = docopt(__doc__)

    shapefile = arguments["<shapefile>"]
    save_path = arguments["<save_path>"]
    class_name = arguments["<class_name>"]
    scenes_path = arguments["<scenes_path>"]
    scenes = arguments["<mgrs>"]

    main(
        shapefile,
        save_path,
        class_name,
        scenes_path,
        scenes,
        patch_size=128,
        num_patches=1000,
    )
