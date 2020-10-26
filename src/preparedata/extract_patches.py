"""Usage:
          extract_patches.py <shapefile> <save_path> <class_name> <scenes_path> (-s <mgrs>)... [options]

@ Robert Huppertz, Jevgenij Gamper 2020, Cervest
Extract image patches from selected scences and store the produced images along with their ground truth.
<class_name>: Category of the shapefile data that should be tracked in the annotation json file 
Add list of mgrs scenes in path format (e.g. 29/S/PB)

Options:
    --patch_size=<ps>  Size of patches, stands for both width and height [default: 128]
    --num_patches=<ns> Maximum number of patches to extract from a scene[default: 2000]
    --time_filter=<tf> Timeframe (number of days) for matching ground truth to data [default: 30]
"""

from src.preparedata.patching import (
    import_shapefile_for_patches,
    do_the_patching,
    store_coco_ground_truth,
    import_image,
)
from src.preparedata.patchutils.coco import save_gt_overlaid
from docopt import docopt
import os
import glob
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
from pathlib import Path


def create_scene_string(scene):
    # create list of scene strings with "_" instead of "/" (e.g. _29_S_PB_ instead of /29/S/PB/)
    scene_list = list(scene)
    scene_list[2] = "_"
    scene_list[4] = "_"
    scene_list.insert(0, "_")
    scene_list.insert(8, "_")
    scene_string = ""
    scene_string = scene_string.join(scene_list)

    return scene_string


def create_date_series(list_of_paths):

    dates_dict = {}
    # creating a list of dates from the paths of the files
    for path in list_of_paths:
        # looking for the beginning index number of the year in the string path
        date_start_index = path.find("20")
        # looking for the index in the filepath string where the Band name starts (e.g. B12)
        date_end_index = path.find("/B", date_start_index)
        # append the date to the list dates
        dates_dict[path] = path[date_start_index:date_end_index]
    # convert the date into datetime format
    for path in dates_dict:
        dates_dict[path] = datetime.strptime(dates_dict[path], "%Y/%m/%d")
    # create a pandas series sorted by dates
    dates_series = pd.Series(dates_dict).sort_values()

    # return strings of dates
    return dates_series


def filter_shapefile(gdf, dates_series, path, time_filter):
    # transform to datetime
    # gdf["DHFim"]= pd.to_datetime(gdf["DHFim"], errors="coerce")

    # filter the ground truth dataframe for the ground truth
    shapefile_filtered = gdf[
        # finish date of the fire (DHFim) should be before the capture date of the image
        (pd.to_datetime(gdf["DHFim"], errors="coerce") < dates_series[path])
        # finish date of the fire (DHFim) should be within a timeframe before the capture date of the image
        & (
            pd.to_datetime(gdf["DHFim"], errors="coerce")
            > (dates_series[path] - timedelta(days=time_filter))
        )
        # burned area ground truth (AREA_HA) should be larger than 5 ha to filter very small burned areas
        & (gdf["AREA_HA"] > 5.0)
    ]
    return shapefile_filtered


def import_raster(path):
    # import images as raster
    raster = import_image(path)
    raster_meta = raster.meta

    return raster, raster_meta


def save_burn_vis(
    scene_string,
    dates_series,
    path,
    path_store_patches,
    path_store_burn_vis,
    path_store_annotations,
):
    # create images that overlay patch and ground truth
    try:
        save_gt_overlaid(
            os.path.join(
                path_store_annotations,
                "annotations{}.json".format(
                    scene_string + dates_series[path].strftime("%Y-%m-%d")
                ),
            ),
            path_store_patches,
            path_store_burn_vis,
        )
    except Exception as exception:
        print(exception)
        print(
            "MGRS tile without annotations: {}".format(
                scene_string + dates_series[path].strftime("%Y-%m-%d")
            )
        )


def extract_patches(
    path,
    dates_series,
    shapefile,
    patch_size,
    num_patches,
    scene_string,
    time_filter,
    path_store_patches,
    path_store_annotations,
    path_store_burn_vis,
):
    # open image as raster
    raster, raster_meta = import_raster(path)

    # load the ground truth (shapefile format) as a dataframe
    gdf = gpd.read_file(shapefile)

    # filter the shapefile with timedelta
    shapefile_filtered = filter_shapefile(gdf, dates_series, path, time_filter)

    # create small patch_windows where ground truth and images overlap
    patch_dfs, patch_windows = import_shapefile_for_patches(
        shapefile_filtered,
        raster,
        raster_meta,
        patch_size,
        num_patches,
        scene_string + dates_series[path].strftime("%Y-%m-%d"),
    )

    # cut out the windows to create patches
    do_the_patching(
        raster,
        path_store_patches,
        patch_windows.keys(),
        patch_windows.values(),
        bands=[1, 2, 3],
    )

    # Save ground truth data as annotation files
    store_coco_ground_truth(
        path_store_annotations,
        patch_dfs,
        patch_size,
        class_name,
        scene_string + dates_series[path].strftime("%Y-%m-%d"),
    )

    # save example images
    save_burn_vis(
        scene_string,
        dates_series,
        path,
        path_store_patches,
        path_store_burn_vis,
        path_store_annotations,
    )

    raster.close()


def main(
    shapefile,
    save_path,
    class_name,
    scenes_path,
    scenes,
    patch_size=128,
    num_patches=1000,
    time_filter=30,
):

    # Make paths
    path_store_patches = os.path.join(save_path, "patches")
    os.makedirs(path_store_patches, exist_ok=True)
    path_store_annotations = os.path.join(save_path, "annotations")
    os.makedirs(path_store_annotations, exist_ok=True)
    path_store_burn_vis = os.path.join(save_path, "burn_vis")
    os.makedirs(path_store_burn_vis, exist_ok=True)
    print(Path(path_store_annotations).parent)
    # loop for multiple mgrs scenes
    for scene in scenes:

        path_scene = os.path.join(scenes_path, scene)

        # get a list of all file paths for the MGRS scene
        list_of_paths = glob.glob(path_scene + "/*/*/*/" + "*.tif")

        # change format of scene name to e.g. _29_S_PB_ from 29/S/PB
        scene_string = create_scene_string(scene)

        dates_series = create_date_series(list_of_paths)

        for path in dates_series.index:
            extract_patches(
                path,
                dates_series,
                shapefile,
                patch_size,
                num_patches,
                scene_string,
                time_filter,
                path_store_patches,
                path_store_annotations,
                path_store_burn_vis,
            )


if __name__ == "__main__":
    arguments = docopt(__doc__)
    shapefile = arguments["<shapefile>"]
    save_path = arguments["<save_path>"]
    class_name = arguments["<class_name>"]
    scenes_path = arguments["<scenes_path>"]
    scenes = arguments["<mgrs>"]
    patch_size=arguments["<patch_size>"]
    num_patches=arguments["<num_patches>"]
    time_filter=arguments["<time_filter>"]

    main(
        shapefile,
        save_path,
        class_name,
        scenes_path,
        scenes,
        patch_size=128,
        num_patches=2000,
        time_filter=30,
    )
