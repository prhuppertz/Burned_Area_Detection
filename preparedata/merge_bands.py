
"""
Usage:
          merge_bands.py <source_path> (-s <mgrs>)... <save_path>

@ Robert Huppertz 2020, Cervest
Loads Selected Bands from image source and produces a merged stack 
of those bands in target folder
"""

#adapted for use with S2 data on the NAS drive

import os
import rasterio
import glob
from tqdm import tqdm
from docopt import docopt
from datetime import datetime

# selecting bands that are needed for the task at hand 
SELECTED_BANDS = ['B8A', 'B11', 'B12']

def stack_bands(source_path, mgrs_coordinate, save_path):
    """
    Reads from source and stacks image bands into a single raster file
    :param source_path: Path to image source
    :param mgrs_coordinate: MGRS coordinate in path format, for example 31/U/FQ
    :param save_path: Directory where a stacked raster would be saved
    :return:
    """
    path_to_scene = os.path.join(source_path, mgrs_coordinate)
    list_of_paths ={}
    date={}
    date_datetime={}
    date_string={}
    path_target={}
    meta_source={}
    
    #create a directory for the target file 
    os.makedirs(os.path.join(save_path, mgrs_coordinate), exist_ok=True)

    
    #create lists of paths to the processed images for all existing dates for each SELECTED_BAND 
    for n in range (0,(len(SELECTED_BANDS)-1)):
        list_of_paths[n] = glob.glob(path_to_scene+'/*/*/*/*/'+SELECTED_BANDS[n]+'_sur.tif')
        with rasterio.open(list_of_paths[n][0]) as src0:
            meta_source[n] = src0.meta
            meta_source[n].update(count=len(SELECTED_BANDS))
    #store the metadata of one BAND and adapt the length to the SELECTED_BANDS
    #WRONG! The metadata changes for the BANDS, depending on the resolution! -> width & height 
        
        with rasterio.open(list_of_paths[0][0]) as src0:
            meta_source = src0.meta
            meta_source.update(count=len(SELECTED_BANDS))
    #create date list for the selected scene, create target files for stacking images
    for i in range (0,len(list_of_paths[0])):
        date_index=list_of_paths[0][i].index('201')
        date_end_index=list_of_paths[0][i].index("/0")
        date[i]=list_of_paths[0][i][date_index:date_end_index]
        date_datetime[i] = datetime.strptime(date[i], '%Y/%m/%d')
        date_string[i]=date_datetime[i].strftime('%Y_%m_%d')
        #create target files for every date
        path_target[i] = os.path.join(save_path, mgrs_coordinate, date_string[i] +'.jp2')
        
    for x in range (0,(len(SELECTED_BANDS)-1)):
        for y in range (0,len(list_of_paths[0])):
            #write SELECTED_BANDS into the target files
            with rasterio.open(path_target[y], 'w', **meta_source) as dst:
                with rasterio.open(list_of_paths[x][y]) as src1:
                    dst.write_band((x+1), src1.read(1))
    

def process_scenes(source_path, mgrs_coordinates, save_path):
    """
    Applies above function to multiple scenes
    :param source_path: Path to source images
    :param mgrs_coordinates: List of MGRS coordinates, for example ['31/U/FQ', '31/U/FO']
    :param save_path: Directory where a stacked raster would be saved
    """
    for mgrs in mgrs_coordinates:
        stack_bands(source_path, mgrs, save_path)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    source_path = arguments['<source_path>']
    save_path = arguments['<save_path>']
    coordinates = arguments['<mgrs>']
    process_scenes(source_path, coordinates, save_path)






