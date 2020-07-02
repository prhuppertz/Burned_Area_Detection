"""Usage:
          merge_bans.py <aws_composites_path>  <save_path> (-s <mgrs>)...

@ Jevgenij Gamper 2020, Cervest
Loads calculated composites from AWS, and produces a merged stack.
"""
import os
import rasterio
from tqdm import tqdm
from docopt import docopt


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

def stack_bands(aws_composites_path, mgrs_coordinate, save_path):
    """
    Reads from AWS and stacks image bands into a single raster file
    :param aws_composites_path: Path to composites on aws bucket
    :param mgrs_coordinate: MGRS coordinate, for example 31UFQ
    :param save_path: Directory where a stacked raster would be saved
    :return:
    """
    path_source = os.path.join(aws_composites_path, mgrs_coordinate)

    with rasterio.open(os.path.join(path_source, ALL_BANDS[0]+'.tif')) as src0:
        meta_source = src0.meta

    meta_source.update(count=len(ALL_BANDS))

    os.makedirs(os.path.join(save_path, mgrs_coordinate), exist_ok=True)
    path_target = os.path.join(save_path, mgrs_coordinate, mgrs_coordinate+'.jp2')

    with rasterio.open(path_target, 'w', **meta_source) as dst:
        for id, layer in enumerate(tqdm(ALL_BANDS), start=1):
            channel_source_path = os.path.join(path_source, "{}.tif".format(layer))
            with rasterio.open(channel_source_path) as src1:
                dst.write_band(id, src1.read(1))

def process_scenes(aws_composites_path, mgrs_coordinates, save_path):
    """
    Applies above function to multiple scenes
    :param aws_composites_path: Path to composites on aws bucket
    :param mgrs_coordinates: List of MGRS coordinates, for example ['31UFQ', '31UFO']
    :param save_path: Directory where a stacked raster would be saved
    """
    for mgrs in mgrs_coordinates:
        stack_bands(aws_composites_path, mgrs, save_path)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    aws_composites_path = arguments['<aws_composites_path>']
    save_path = arguments['<save_path>']
    coordinates = arguments['<mgrs>']
    process_scenes(aws_composites_path, coordinates, save_path)
