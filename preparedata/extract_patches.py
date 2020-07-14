"""Usage:
          extract_patches.py <shapefile>  <save_path> <class_name> <scenes_path> (-s <mgrs>)...

@ Jevgenij Gamper & Robert Huppertz 2020, Cervest
Extract image patches from selected scences and stored the produced images along with their ground truth.

Options:
    --patch_size=<ps>  Size of patches, stands for both width and height [default: 128].
    --num_patches=<ns> Maximum number of patches to extract from a scene[default: 1000].
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
import glob
from datetime import datetime

date = {}
date_datetime={}
date_string={}

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
    #path_to_store_field_vis = os.path.join(save_path, "field_vis")
    #os.makedirs(path_to_store_field_vis, exist_ok=True)

    for scene_mgrs in scenes:
        path_to_scene = os.path.join(
            scenes_path, scene_mgrs)
        
        list_of_paths = glob.glob(
            path_to_scene + "/*/*/*/" + "*.jp2".format(scene_mgrs)
        )

        for date_var in range (0, len(list_of_paths)):
            
            #Read the date of the loaded image
            date_index = list_of_paths[date_var].index("201")
            #TODO: Remove hard coding here
            date_end_index = list_of_paths[date_var].index("/B8A")
            date[date_var] = list_of_paths[date_var][date_index:date_end_index]
            date_datetime[date_var] = datetime.strptime(date[date_var], '%Y/%m/%d')
            date_string[date_var]=date_datetime[date_var].strftime('%Y_%m_%d')
            
            # Read composite and import its metadata
            raster = import_image(list_of_paths[date_var])
            raster_meta = raster.meta

            # Load the shapefile
            patch_dfs, patch_windows = import_shapefile_for_patches(
                shapefile, raster, raster_meta, patch_size, num_patches, scene_mgrs + date_string[date_var]
            )

            # Patch the shapefile, and store image patches
            # What are the bands for?
            do_the_patching(
                raster,
                path_to_store_patches,
                patch_windows.keys(),
                patch_windows.values(),
                bands=[3,2,1],
            )

            # Save annotations
            #give a date in the file 
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
