"""Usage:
          merge_bands.py <aws_composites_path>  <save_path> (-s <mgrs>)...

@ Jevgenij Gamper 2020, Cervest
Loads calculated composites from AWS, and produces a merged stack.
"""

#adapted for use with S2 data on the NAS drive

import os
import rasterio
from tqdm import tqdm
from docopt import docopt


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
# selecting bands that are necessary for the task at hand 
SELECTED_BANDS = ['B8A', 'B11', 'B12']
def stack_bands(aws_composites_path, mgrs_coordinate, date, save_path):
    """
    Reads from AWS and stacks image bands into a single raster file
    :param aws_composites_path: Path to composites on aws bucket
    :param mgrs_coordinate: MGRS coordinate in path format, for example 31/U/FQ
    :param save_path: Directory where a stacked raster would be saved
    :param date: Date for the specific data in path format, for example 2017/10/15
    :return:
    """
    path_source = os.path.join(aws_composites_path, mgrs_coordinate, date, "0")

    with rasterio.open(os.path.join(path_source, SELECTED_BANDS[0]+'_sur'+'.tif')) as src0:
        meta_source = src0.meta

    meta_source.update(count=len(SELECTED_BANDS))

    os.makedirs(os.path.join(save_path, mgrs_coordinate, date), exist_ok=True)
    path_target = os.path.join(save_path, mgrs_coordinate, date, mgrs_coordinate+'.jp2')

    with rasterio.open(path_target, 'w', **meta_source) as dst:
        for id, layer in enumerate(tqdm(ALL_BANDS), start=1):
            channel_source_path = os.path.join(path_source, "{}.tif".format(layer))
            with rasterio.open(channel_source_path) as src1:
                dst.write_band(id, src1.read(1))


def process_dates (aws_composites_path, mgrs_coordinate, dates, save_path):
    """
    Applies above function to multiple dates
    :param aws_composites_path: Path to composites on aws bucket
    :param mgrs_coordinate: MGRS coordinates, for example '31/T/CF'
    :param dates: List of dates, for example ['2017/12/13', '2018/6/9']
    :param save_path: Directory where a stacked raster would be saved
    """
    for date in dates:
        stack_bands(aws_composites_path, mgrs_coordinate, date, save_path)

 

if __name__ == "__main__":
    arguments = docopt(__doc__)
    aws_composites_path = arguments['<aws_composites_path>']
    save_path = arguments['<save_path>']
    coordinates = arguments['<mgrs>']
    process_dates(aws_composites_path, mgrs_oordinates, dates, save_path)
