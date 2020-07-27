"""Usage:
          extract_patches.py <shapefile> <save_path> <class_name> <scenes_path> (-s <mgrs>)...

@ Jev Gamper & Robert Huppertz 2020, Cervest
Extract image patches from selected scences and store the produced images along with their ground truth.
<class_name>: Category of the shapefile data that should be tracked in the annotation json file 

Options:
    --patch_size=<ps>  Size of patches, stands for both width and height [default: 128].
    --num_patches=<ns> Maximum number of patches to extract from a scene[default: 2000].
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
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd

list_of_dates_as_datetime = []
list_of_dates_as_strings = []
date = []


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
    path_to_store_annotations = os.path.join(save_path, "annotations")
    os.makedirs(path_to_store_annotations, exist_ok=True)
    path_to_store_burn_vis = os.path.join(save_path, "burn_vis")
    os.makedirs(path_to_store_burn_vis, exist_ok=True)

    # folders for temporal sequence
    """
    path_to_store_patches_prev = os.path.join(save_path, "patches_prev")
    os.makedirs(path_to_store_patches_prev, exist_ok=True)
    path_to_store_annotations_prev = os.path.join(save_path, "annotations_prev")
    os.makedirs(path_to_store_annotations_prev, exist_ok=True)
    path_to_store_burn_vis_prev = os.path.join(save_path, "burn_vis_prev")
    os.makedirs(path_to_store_burn_vis_prev, exist_ok=True)
    """

    for scene in scenes:
        path_to_scene = os.path.join(scenes_path, scene)

        list_of_paths = glob.glob(
            path_to_scene + "/*/*/*/" + "*.tif"#.format(scene)
        )

        # create scene string with "_" instead of "/"
        scene_list = list(scene)
        scene_list[2] = "_"
        scene_list[4] = "_"
        scene_list.insert(0, "_")
        scene_list.insert(8, "_")
        scene_string = ""
        scene_string = scene_string.join(scene_list)

        #creating a list of dates from the paths of the files
        for path in range(0, len(list_of_paths)):
            #looking for the beginning index number of the year in the string path
            date_start_index = list_of_paths[path].find("20")
            date_end_index = list_of_paths[path].find("/B", date_start_index)
            date.append(list_of_paths[path][date_start_index:date_end_index])
            list_of_dates_as_datetime.append(datetime.strptime(date[path], "%Y/%m/%d"))

        # create sorted Pandas series for dates and paths
        date_file_mapping = {
            dt: filename for dt, filename in zip(list_of_dates_as_datetime, list_of_paths)
        }
        series_of_paths_sorted_by_date = pd.Series(date_file_mapping).sort_index()

        # create sorted date list with strings
        for path in range(0, len(series_of_paths_sorted_by_date)):
            list_of_dates_as_strings.append(series_of_paths_sorted_by_date.index[path].strftime("%Y-%m-%d"))

        for path in range(0, len(series_of_paths_sorted_by_date)):
            
            # import image
            raster = import_image(series_of_paths_sorted_by_date[path])
            raster_meta = raster.meta

            # load the shapefiles
            gdf = gpd.read_file(shapefile)

            # load the shapefile data of the days prior to the date of the mgrs scene
            shapefile_date = gdf[
                (gdf["DATE_ECLOS"] < series_of_paths_sorted_by_date.index[path].strftime("%Y-%m-%d")) & (gdf['DATE_ECLOS']>(series_of_paths_sorted_by_date.index[path]-timedelta(days=90)).strftime('%Y-%m-%d'))]

            # create windows from where shapefile and scenes overlap
            patch_dfs, patch_windows = import_shapefile_for_patches(
                shapefile_date,
                raster,
                raster_meta,
                patch_size,
                num_patches,
                scene_string + list_of_dates_as_strings[path],
            )

            # patch from the given scenes
            do_the_patching(
                raster,
                path_to_store_patches,
                patch_windows.keys(),
                patch_windows.values(),
                bands=[3, 2, 1],
            )

            # Save annotations
            store_coco_ground_truth(
                path_to_store_annotations,
                patch_dfs,
                patch_size,
                class_name,
                scene_string + list_of_dates_as_strings[path],
            )

            # create overlaid patches with ground truth
            try:
                save_gt_overlaid(
                    os.path.join(
                        path_to_store_annotations,
                        "annotations{}.json".format(scene_string + list_of_dates_as_strings[path]),
                    ),
                    path_to_store_patches,
                    path_to_store_burn_vis,
                )
            except Exception as e:
                print(e)
                print("MGRS tile without annotations: {}".format(scene_string))

            raster.close()

            # patch image from previous satelite image
            

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
        num_patches=2000,
    )
