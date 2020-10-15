"""Usage:
          extract_patches.py <shapefile> <save_path> <class_name> <scenes_path> (-s <mgrs>)...

@ Robert Huppertz, Jevgenij Gamper 2020, Cervest
Extract image patches from selected scences and store the produced images along with their ground truth.
<class_name>: Category of the shapefile data that should be tracked in the annotation json file 
Add list of mgrs scenes in path format (e.g. 29/S/PB)

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

    # loop for multiple mgrs scenes
    for scene in scenes:

        path_to_scene = os.path.join(scenes_path, scene)

        # get a list of all file paths for the MGRS scene
        list_of_paths = glob.glob(path_to_scene + "/*/*/*/" + "*.tif")

        # create list of scene strings with "_" instead of "/" (e.g. _29_S_PB_ instead of /29/S/PB/)
        scene_list = list(scene)
        scene_list[2] = "_"
        scene_list[4] = "_"
        scene_list.insert(0, "_")
        scene_list.insert(8, "_")
        scene_string = ""
        scene_string = scene_string.join(scene_list)

        # TODO:change explicit mentioning of the year and the filename to implicited values
        list_of_dates_as_datetime = []
        list_of_dates_as_strings = []
        dates = []
        # creating a list of dates from the paths of the files
        for path in range(0, len(list_of_paths)):
            # looking for the beginning index number of the year in the string path
            date_start_index = list_of_paths[path].find("20")
            # looking for the index in the filepath string where the Band name starts (e.g. B12)
            date_end_index = list_of_paths[path].find("/B", date_start_index)
            # append the date to the list dates
            dates.append(list_of_paths[path][date_start_index:date_end_index])
        # convert the date strings into datetime format
        for date in range(0, len(dates)):
            list_of_dates_as_datetime.append(datetime.strptime(dates[date], "%Y/%m/%d"))

        # create Pandas series for dates and paths to sort the paths by date
        date_file_mapping = {
            dt: filename
            for dt, filename in zip(list_of_dates_as_datetime, list_of_paths)
        }
        series_of_paths_sorted_by_date = pd.Series(date_file_mapping).sort_index()

        # create sorted date list with strings
        for path in range(0, len(series_of_paths_sorted_by_date)):
            list_of_dates_as_strings.append(
                series_of_paths_sorted_by_date.index[path].strftime("%Y-%m-%d")
            )

        for path in range(0, len(series_of_paths_sorted_by_date)):

            # import images as raster
            raster = import_image(series_of_paths_sorted_by_date[path])
            raster_meta = raster.meta

            # load the ground truth (shapefile format) as a dataframe
            gdf = gpd.read_file(shapefile)

            # filter the ground truth dataframe for the ground truth
            shapefile_date = gdf[
                # finish date of the fire (DHFim) should be before the capture date of the image
                (
                    gdf["DHFim"]
                    < series_of_paths_sorted_by_date.index[path].strftime("%Y-%m-%d")
                )
                # finish date of the fire (DHFim) should be within a 30-day timeframe before the capture date of the image
                & (
                    gdf["DHFim"]
                    > (
                        series_of_paths_sorted_by_date.index[path] - timedelta(days=30)
                    ).strftime("%Y-%m-%d")
                )
                # burned area ground truth (AREA_HA) should be larger than 5 ha to filter very small burned areas
                & (gdf["AREA_HA"] > 5.0)
            ]

            # create small patch_windows where ground truth and images overlap
            patch_dfs, patch_windows = import_shapefile_for_patches(
                shapefile_date,
                raster,
                raster_meta,
                patch_size,
                num_patches,
                scene_string + list_of_dates_as_strings[path],
            )

            # cut out the windows to create patches
            do_the_patching(
                raster,
                path_to_store_patches,
                patch_windows.keys(),
                patch_windows.values(),
                bands=[1, 2, 3],
            )

            # Save ground truth data as annotation files
            store_coco_ground_truth(
                path_to_store_annotations,
                patch_dfs,
                patch_size,
                class_name,
                scene_string + list_of_dates_as_strings[path],
            )

            # create images that overlay patch and ground truth
            try:
                save_gt_overlaid(
                    os.path.join(
                        path_to_store_annotations,
                        "annotations{}.json".format(
                            scene_string + list_of_dates_as_strings[path]
                        ),
                    ),
                    path_to_store_patches,
                    path_to_store_burn_vis,
                )
            except Exception as e:
                print(e)
                print(
                    "MGRS tile without annotations: {}".format(
                        scene_string + list_of_dates_as_strings[path]
                    )
                )

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
        num_patches=2000,
    )
