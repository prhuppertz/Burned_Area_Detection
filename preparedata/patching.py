"""
Define functions for the other preparedata scripts
"""


import rasterio
import geopandas as gpd
import shapely
from patchutils import geo, img
from patchutils import other, coco
from typeguard import typechecked


@typechecked
def import_shapefile_for_patches(
    shapefile, raster, raster_meta, patch_size, first_n_patches, mgrs_tile
):
    """
    Import shapefile from file
    """
    # Read shapefile
    #!TODO: Maxime's comments - be explicit below, have individual line for every operation
    #!TODO: Maxime's comments - allows for easier errors, as it will point only to one line
    #ROBERT: chamged so a shapefile is passed, not the path
    sh_df = shapefile
    # Convert to raster crs
    sh_df = sh_df.to_crs(raster_meta["crs"])
    sh_df = geo.explode_mp(sh_df)
    sh_df = geo.buffer_zero(sh_df)
    sh_df = geo.close_holes(sh_df)

    # Clip shapefile to raster geometry
    sh_df = geo.clip(
        sh_df, clip_poly=shapely.geometry.box(*raster.bounds), explode_mp_=True
    )
    sh_df = sh_df.assign(
        geometry=lambda _df: _df.geometry.simplify(5, preserve_topology=True)
    )
    sh_df = geo.buffer_zero(sh_df)
    sh_df = geo.reduce_precision(sh_df, precision=4)
    sh_df = sh_df.reset_index(drop=True)
    sh_df = sh_df.assign(fid=lambda _df: range(0, len(_df.index)))

    patch_dfs = geo.cut_chip_geometries(
        vector_df=sh_df,
        raster_width=raster_meta["width"],
        raster_height=raster_meta["height"],
        raster_transform=raster_meta["transform"],
        chip_width=patch_size,
        chip_height=patch_size,
        first_n_chips=first_n_patches,
        mgrs_tile=mgrs_tile,
    )

    patch_windows = {
        chip_name: value["chip_window"] for chip_name, value in patch_dfs.items()
    }

    return patch_dfs, patch_windows


def do_the_patching(raster, output_patch_path, patch_names, patch_windows, bands):
    """
    """

    _ = img.cut_chip_images(
        raster=raster,
        output_patch_path=output_patch_path,
        patch_names=patch_names,
        patch_windows=patch_windows,
        bands=bands,
    )

    return


def store_coco_ground_truth(output_path, patch_dfs, patch_size, row_name, mgrs_tile):
    """
    Will save extract polygon ground truth in a json file
    """
    coco_dst = coco.format_coco(patch_dfs, patch_size, row_name)
    other.new_json(output_path + "/anno{}.json".format(mgrs_tile), coco_dst)


def import_image(path_to_tile):
    raster = rasterio.open(path_to_tile, "r")
    return raster
