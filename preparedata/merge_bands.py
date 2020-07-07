"""
Usage:
          merge_bands.py <source_path> <mgrs> <save_path>

@ Robert Huppertz 2020, Cervest
Loads calculated composites from image source, and produces a merged stack.
"""

#adapted for use with S2 data on the NAS drive

import os
import rasterio
import glob
from tqdm import tqdm
from docopt import docopt
from datetime import datetime


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
# selecting bands that are necessary for the task at hand 
SELECTED_BANDS = ['B8A', 'B11', 'B12']

def stack_bands(source_path, mgrs_coordinate, save_path):
    """
    Reads from source and stacks image bands into a single raster file
    :param source_path: Path to image source
    :param mgrs_coordinate: MGRS coordinate in path format, for example 31/U/FQ
    :param save_path: Directory where a stacked raster would be saved
    :param date: Date for the specific data in path format, for example 2017/10/15
    :return:
    """
    path_to_scene = os.path.join(source_path, mgrs_coordinate)
    list_of_paths ={}
    date={}

    #create lists of paths to the processed images for all existing dates for each SELECTED_BAND 
    for n in range (0,len(SELECTED_BANDS)):
        list_of_paths[n] = glob.glob(path_to_scene+'/*/*/*/*/'+SELECTED_BANDS[n]+'_sur.tif')
    
    #create date list for the selected scene
    for i in range (0,len(list_of_paths[0])):
        date_index=list_of_paths[0][i].index('201')
        date_end_index=list_of_paths[0][i].index("/0")
        date[i]=list_of_paths[0][i][date_index:date_end_index]
        date_datetime[i] = datetime.strptime[date[i], '%Y/%m/%d']
    
'''
    #add loop for stacking the BANDS for each date and storing the stack in a new directory
    with rasterio.open(list_of_paths[i]) as src0:
        meta_source = src0.meta

        meta_source.update(count=len(ALL_BANDS))
    os.makedirs(os.path.join(save_path, mgrs_coordinate, date[i]), exist_ok=True)
    path_target = os.path.join(save_path, mgrs_coordinate, date+'.jp2')

        with rasterio.open(path_target, 'w', **meta_source) as dst:
            for id, layer in enumerate(tqdm(ALL_BANDS), start=1):
                channel_source_path = os.path.join(path_source, "{}.tif".format(layer))
                with rasterio.open(channel_source_path) as src1:
                    dst.write_band(id, src1.read(1))
        
'''        

def process_scenes(source_path, mgrs_coordinates, save_path):
    """
    Applies above function to multiple scenes
    :param source_path: Path to source images
    :param mgrs_coordinates: List of MGRS coordinates, for example ['31UFQ', '31UFO']
    :param save_path: Directory where a stacked raster would be saved
    """
    for mgrs in mgrs_coordinates:
        stack_bands(source_path, mgrs, save_path)

 
    
if __name__ == "__main__":
    arguments = docopt(__doc__)
    source_path = arguments['<source_path>']
    save_path = arguments['<save_path>']
    mgrs_coordinates = arguments['<mgrs>']
    process_scenes(source_path, mgrs_coordinates, save_path)



