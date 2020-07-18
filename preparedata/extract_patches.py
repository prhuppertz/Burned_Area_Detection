"""Usage:
          extract_patches.py <shapefile>  <save_path> <class_name> <scenes_path> (-s <mgrs>)...

@ Jev Gamper & Robert Huppertz 2020, Cervest
Extract image patches from selected scences and stored the produced images along with their ground truth.

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
from datetime import datetime
import geopandas as gpd
import pandas as pd

date_datetime = []
date_string = []
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
    path_to_store_anno = os.path.join(save_path, "anno")
    os.makedirs(path_to_store_anno, exist_ok=True)
    path_to_store_burn_vis = os.path.join(save_path, "burn_vis")
    os.makedirs(path_to_store_burn_vis, exist_ok=True)

    # folders for temporal sequence
    """
    path_to_store_patches_prev = os.path.join(save_path, "patches_prev")
    os.makedirs(path_to_store_patches_prev, exist_ok=True)
    path_to_store_anno_prev = os.path.join(save_path, "anno_prev")
    os.makedirs(path_to_store_anno_prev, exist_ok=True)
    path_to_store_burn_vis_prev = os.path.join(save_path, "burn_vis_prev")
    os.makedirs(path_to_store_burn_vis_prev, exist_ok=True)
    """

    for scene_mgrs in scenes:
        path_to_scene = os.path.join(scenes_path, scene_mgrs)

        list_of_paths = glob.glob(
            path_to_scene + "/*/*/*/" + "*.tif".format(scene_mgrs)
        )

        # create scene string
        scene_list = list(scene_mgrs)
        scene_list[2] = "_"
        scene_list[4] = "_"
        scene_list.insert(0, "_")
        scene_list.insert(8, "_")
        scene_string = ""
        scene_string = scene_string.join(scene_list)

        for date_var1 in range(0, len(list_of_paths)):
            date_index = list_of_paths[date_var1].index("201")
            date_end_index = list_of_paths[date_var1].index("/B8A")
            date.append(list_of_paths[date_var1][date_index:date_end_index])
            date_datetime.append(datetime.strptime(date[date_var1], "%Y/%m/%d"))

        # create sorted Pandas series for dates and paths
        date_file_mapping = {
            dt: filename for dt, filename in zip(date_datetime, list_of_paths)
        }
        map_of_paths = pd.Series(date_file_mapping).sort_index()

        # create sorted date list with strings
        for date_var2 in range(0, len(map_of_paths)):
            date_string.append(map_of_paths.index[date_var2].strftime("%Y-%m-%d"))

        for date_var3 in range(0, len(map_of_paths)):
            # import image
            raster1 = import_image(map_of_paths[date_var3])
            raster_meta1 = raster1.meta

            # load the shapefiles
            gdf = gpd.read_file(shapefile)
            # load the shapefile data of the days prior to the date of the mgrs scene
            shapefile_date1 = gdf[
                (gdf["DATE_ECLOS"] < map_of_paths.index[date_var3].strftime("%Y-%m-%d"))
            ]  # &(gdf['DATE_ECLOS']>(map_of_paths.index[date_var3]-timedelta(days=45)).strftime('%Y-%m-%d'))]

            # create windows from where shapefile and scenes overlap
            patch_dfs1, patch_windows1 = import_shapefile_for_patches(
                shapefile_date1,
                raster1,
                raster_meta1,
                patch_size,
                num_patches,
                scene_string + date_string[date_var3],
            )

            # patch from the given scenes
            do_the_patching(
                raster1,
                path_to_store_patches,
                patch_windows1.keys(),
                patch_windows1.values(),
                bands=[3, 2, 1],
            )

            # Save annotations
            store_coco_ground_truth(
                path_to_store_anno,
                patch_dfs1,
                patch_size,
                class_name,
                scene_string + date_string[date_var3],
            )

            # create overlaid patches with ground truth
            try:
                save_gt_overlaid(
                    os.path.join(
                        path_to_store_anno,
                        "anno{}.json".format(scene_string + date_string[date_var3]),
                    ),
                    path_to_store_patches,
                    path_to_store_burn_vis,
                )
            except Exception as e:
                print(e)
                print("MGRS tile without annotations: {}".format(scene_string))

            raster1.close()

            # patch image from previous satelite image
            """
            if date_var3>1:
                for date_var4 in range (date_var3-2,date_var3):
                    raster_prev = import_image(map_of_paths[date_var4])
                    raster_meta_prev = raster_prev.meta
        
                    # load the shapefile data of the days prior to the date of the mgrs scene
                    shapefile_date_prev=gdf[(gdf["DATE_ECLOS"]<map_of_paths.index[date_var4].strftime('%Y-%m-%d'))]#&(gdf['DATE_ECLOS']>(map_of_paths.index[date_var4]-timedelta(days=45)).strftime('%Y-%m-%d'))]
        
                    patch_dfs_prev, patch_windows_prev = import_shapefile_for_patches(
                        shapefile_date_prev, raster_prev, raster_meta_prev, patch_size, num_patches, scene_string + date_string[date_var4]
                    )
    
                    do_the_patching(
                        raster_prev,
                        path_to_store_patches_prev,
                        patch_windows_prev.keys(),
                        patch_windows1.values(),
                        bands=[3,2,1],
                    )
    
                    # Save annotations
                    store_coco_ground_truth(
                        path_to_store_anno_prev, patch_dfs_prev, patch_size, class_name, scene_string + date_string[date_var4]
                    )
            

                    #create overlaid patches with ground truth
                    try:
                        save_gt_overlaid(
                            os.path.join(path_to_store_anno_prev, "anno{}.json".format(scene_string + date_string[date_var4])),
                            path_to_store_patches_prev,
                            path_to_store_burn_vis_prev,
                            )
                    except Exception as e:
                        print(e)
                        print("MGRS tile without annotations: {}".format(scene_string))

                    raster_prev.close()
            """


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
