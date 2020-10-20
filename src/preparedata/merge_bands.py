"""
Usage:
          merge_bands.py <source_path> (-s <mgrs>)... <save_path>

@ Robert Huppertz 2020, Cervest
Loads Selected Bands from source_path and produces a merged stack 
of those bands in save_path, feed list of mgrs scenes in path format (e.g. 29/S/PB) that should be included

"""

# adapted for use with S2 data on the NAS drive

import os
import rasterio
import glob
from tqdm import tqdm
from docopt import docopt
from datetime import datetime
from pathlib import Path


def create_source_paths(list_of_bands, path_to_scene):
    """
    Creates a dict of paths to all bands in path_to_scene
    :param list_of_bands: List of bands
    :param path_to_scene: path to the mgrs scene
    """
    bands_string=""
    dict_source_paths = {}
    meta_source = {}
    for band in list_of_bands:
        # create dictionary with paths to all files of each band
        dict_source_paths[band] = glob.glob(
            path_to_scene + "/*/*/*/*/" + band + "_sur.tif"
        )
        # store the metadata of one image of each band
        with rasterio.open(dict_source_paths[band][0]) as opened_band:
            #store the meta data of the opened band in the dictionary
            meta_source[band] = opened_band.meta
            #update the count to the number of bands
            meta_source[band].update(count=len(list_of_bands))
        bands_string+=band
        
    return dict_source_paths, bands_string, meta_source
        
def create_target_paths(dict_source_paths, save_path, mgrs_coordinate, bands_string, list_of_bands):
    
    """
    Creates a dict of target paths by date and bands
    :param list_of_bands: List of bands
    :param list_of_paths: List of all paths that belong to a certain band
    
    :param paths_target: path to the mgrs scene
    """
    dict_target_paths = {}
    date = {}
    dict_of_dates={}
    # create date list for the selected scene, create target files for stacking images
    
    for path in dict_source_paths[list_of_bands[0]]:
        date_index = path.index("201")
        date_end_index = path.index("/0")
        #create dict with source paths and dates
        dict_of_dates[path] = path[date_index:date_end_index]
        #create target path with date and bands_string
        os.makedirs(os.path.join(save_path, mgrs_coordinate, dict_of_dates[path]), exist_ok=True)
        #create dict with dates and target paths
        dict_target_paths[dict_of_dates[path]] = os.path.join(
            save_path, mgrs_coordinate, dict_of_dates[path], bands_string + ".tif"
        )
    return dict_target_paths

def write_bands(dict_source_paths, meta_source, dict_target_paths, list_of_bands):
    """
    Writes bands to target path
    :param list_of_bands: List of bands
    :param list_of_paths: List of all paths that belong to a certain band
    :param paths_target: path to the mgrs scene
    """
    list_target_paths=list(dict_target_paths.values())
    for x in list_target_paths:
        with rasterio.open(x, "w", **meta_source[list_of_bands[0]]) as target_file:
            for band in list_of_bands:
                with rasterio.open(dict_source_paths[band][list_target_paths.index(x)]) as source_file:
                    #write the source file into the target file's band in the order of the 
                    target_file.write_band((list_of_bands.index(band) + 1), source_file.read(1))


def stack_bands(source_path, mgrs_coordinate, save_path):
    """
    Stacks bands
    :param source_path: Path to source images
    :param mgrs_coordinates: List of MGRS coordinates, for example ['31/U/FQ', '31/U/FO']
    :param save_path: Directory where a stacked raster would be saved
    """

    #selecting the SWIR, NIR and Green band names from the remote sensing dataset
    dict_of_bands = {"SWIR-band":"B12","NIR-band":"B8A","Green-band":"B03"}
    list_of_bands = list(dict_of_bands.values())

    path_to_scene = os.path.join(source_path, mgrs_coordinate)

    dict_source_paths, bands_string, meta_source = create_source_paths(list_of_bands, path_to_scene)

    dict_target_paths = create_target_paths(dict_source_paths, save_path, mgrs_coordinate, bands_string, list_of_bands)
    
    write_bands(dict_source_paths, meta_source, dict_target_paths, list_of_bands)
    

def process_scenes(source_path, mgrs_coordinates, save_path):
    """
    Applies above function to multiple scenes
    :param source_path: Path to source images
    :param mgrs_coordinates: List of MGRS coordinates, for example ['31/U/FQ', '31/U/FO']
    :param save_path: Directory where a stacked raster would be saved
    """

    for mgrs_coordinate in mgrs_coordinates:
        stack_bands(source_path, mgrs_coordinate, save_path)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    source_path = arguments["<source_path>"]
    save_path = arguments["<save_path>"]
    coordinates = arguments["<mgrs>"]
    process_scenes(source_path, coordinates, save_path)
