"""
    Usage: upscale_forest_cover.py <landcover_data_path> <weather_data_path> <output_path>
"""

from docopt import docopt
import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr


def main(landcover_data_path, weather_data_path, output_path):
    """
    Takes a high-resolution landcover dataset, and a lower-resolution weather dataset. Binarises the landcover dataset,
    by whether a pixel corresponds to forest cover or not, and then matches each pixel in the landcover dataset to a
    pixel in the weather dataset. Landcover pixels are matched to the closest available weather data pixel. Then groups
    binarised landcover pixel by the weather data pixel they have been matched to and calculates the average of the
    binarised pixel values for each weather data pixel.

    This gives us an estimate of the percentage of land that is covered by forest for each pixel in the weather raster.
    """
    logging.info('Openining, clipping, and binarising required data')
    landcover_data, weather_data = open_required_data(landcover_data_path, weather_data_path)
    landcover_clipped = clip_landcover_to_weather_data(landcover_data, weather_data)
    forest_cover_binary = get_forest_cover_from_landcover(landcover_clipped)

    logging.info('Extracting coordinates and matching closest pixels')
    coordinate_grids = get_coordinate_grids(forest_cover_binary, weather_data)
    coordinate_lists = get_coordinate_lists_from_grids(**coordinate_grids)

    fc_group_ids, fc_group_coords = match_forest_pixels_to_closest_weather_pixel(coordinate_lists)

    logging.info('Compiling and sorting datacube')
    forest_cover_datacube = get_forest_cover_datacube(fc_group_ids, fc_group_coords, coordinate_grids['forest_cover'])
    forest_cover_datacube_sorted = sort_forest_cover_datacube(forest_cover_datacube)

    logging.info('Converting datacube to xarray.DataSet')
    forest_cover_dataarrays = get_forest_cover_dataarrays_from_cube(forest_cover_datacube_sorted)
    forest_cover_grouped_dataset = merge_forest_cover_data(forest_cover_dataarrays, forest_cover_binary)

    logging.info(f'Aggregating DataSet and saving output at {output_path}')
    forest_cover_aggregated_df = aggregate_grouped_forest_cover_data(forest_cover_grouped_dataset)
    convert_to_raster_and_write_to_output(forest_cover_aggregated_df, output_path)


def open_required_data(landcover_data_path, weather_data_path):
    """
    Open the landcover dataset to be upscaled and a reference weather data raster, which we wish to upscale to.
    """
    weather_data = xr.open_dataset(weather_data_path)
    weather_data = weather_data.assign_coords(
        longitude=(((weather_data.longitude + 180) % 360) - 180)
    )
    landcover_data = xr.open_dataset(landcover_data_path)
    return landcover_data, weather_data


def clip_landcover_to_weather_data(landcover_data, weather_data):
    """
    Clip the extent of the landcover data to match the extent of the weather data.
    """
    lat_max, lat_min = weather_data.R.latitude.max(), weather_data.R.latitude.min()
    lon_max, lon_min = weather_data.R.longitude.max(), weather_data.R.longitude.min()

    landcover_data_clipped = landcover_data.sortby('lon').sel(lon=slice(lon_min, lon_max))
    landcover_data_clipped = landcover_data_clipped.sortby('lat').sel(lat=slice(lat_min, lat_max))

    return landcover_data_clipped


def get_forest_cover_from_landcover(landcover_data):
    """
    Binarise the lccs_class variable in the landcover data (30-130, exclusive, corresponds to forest cover), and
    save in a new DataArray with the same coords as the original landcover data
    """
    forest_cover_binarised_pixels = (
            (landcover_data.lccs_class.values > 30) & (landcover_data.lccs_class.values < 130)
    ).astype(int)

    forest_cover_array = xr.DataArray(
        data=forest_cover_binarised_pixels, coords=dict(landcover_data.coords), dims=landcover_data.dims
    ).rename('forest_cover')

    return forest_cover_array


def get_coordinate_grids(forest_cover_data, weather_data):
    """
    Returns a two tuples of arrays, each containing a 2d array of latitude values at each grid point and longitude
    values at each grid point, one for the forest cover data and one for the weather data.
    """
    return {
        'forest_cover': np.meshgrid(forest_cover_data.lon, forest_cover_data.lat),
        'weather': np.meshgrid(weather_data.longitude, weather_data.latitude)
    }


def get_coordinate_lists_from_grids(forest_cover, weather):
    """
    Similar to get_coordinate_grids, but returns a flattened list of all lat/lon values instead of a 2d array.
    """
    return {
        'forest_cover': np.array([forest_cover[0].flatten(), forest_cover[1].flatten()]).T,
        'weather': np.array([weather[0].flatten(), weather[1].flatten()]).T
    }


def match_forest_pixels_to_closest_weather_pixel(coordinate_lists):
    """
    Uses a binary search tree (scipy.spatial.cKDTree) to match each point in the weather array to the nearest point in
    landcover data. Thus, each point in the landcover data is matched to the (lat,lon) coordinate of a point in the
    weather data, as well as a 'group_id' which corresponds to the index location of the (lat,lon) coordinate list
    given in coorindate_lists['weather'].
    """
    ckd_tree = cKDTree(coordinate_lists['weather'])
    _, forest_cover_coord_group_ids = ckd_tree.query(coordinate_lists['forest'])
    forest_cover_group_coords = coordinate_lists['weather'][forest_cover_coord_group_ids]
    return forest_cover_coord_group_ids, forest_cover_group_coords


def get_forest_cover_datacube(forest_cover_group_ids, fores_cover_group_coords, forest_cover_grid):
    """
    Compile the group_id, group_latitude, group_longitude, and forest_cover_pixel_latitude and
    forest_cover_pixel_longitude into a 3d datacube (array). This will allows us to easily sort everything by the
    latitude and longitude of the forest_cover pixels.
    """
    forest_cover_group_id_array = np.reshape(forest_cover_group_ids, forest_cover_grid[0].shape)
    forest_cover_group_lon_array = np.reshape(fores_cover_group_coords[:, 0], forest_cover_grid[0].shape)
    forest_cover_group_lat_array = np.reshape(fores_cover_group_coords[:, 1], forest_cover_grid[0].shape)

    forest_cover_datacube = np.stack([
        forest_cover_group_id_array,  # Group ID
        forest_cover_group_lon_array,  # Lon coord of group ID
        forest_cover_group_lat_array,  # Lat coord of group ID
        forest_cover_grid[0],  # Lat coord of pixel
        forest_cover_grid[1]  # Lon coord of pixel
    ], axis=0)

    return forest_cover_datacube


def sort_forest_cover_datacube(forest_cover_datacube):
    """
    Sort the datacube from get_forest_cover_datacube by latitude and longitude of the forest_cover pixels.
    """
    forest_cover_datacube_sorted = np.sort(forest_cover_datacube, axis=-1)
    forest_cover_datacube_sorted = np.sort(forest_cover_datacube_sorted, axis=-2)
    return forest_cover_datacube_sorted


def get_forest_cover_dataarrays_from_cube(forest_cover_datacube):
    """
    Create a dictionary of DataArrays from the given datacube, one containing group_id, one containing weather_lat (i.e.
    the group latitude value), and one containig weather_lon (i.e. the group longitude value). The coordinates of the
    DataArrays are the (lat,lon) coordinates of the forest_cover pixels.
    """
    weather_lon_values = forest_cover_datacube[1]
    weather_lat_values = forest_cover_datacube[2]
    group_id_values = forest_cover_datacube[0]

    pixel_lon_values = forest_cover_datacube[3][0, :]
    pixel_lat_values = forest_cover_datacube[4][:, 0]
    array_coords = {'lon': pixel_lon_values, 'lat': pixel_lat_values}

    name_variable_mapping = {
        'group_id': group_id_values, 'weather_lat': weather_lat_values, 'weather_lon': weather_lon_values
    }

    return {
        name: xr.DataArray(data=variable, coords=array_coords,  dims=['lat', 'lon'])
        for name, variable in name_variable_mapping.items()
    }


def merge_forest_cover_data(forest_cover_dataarrays, forest_cover_binary):
    """
    Merge together the DataArrays from get_forest_cover_dataarrays_from_cube to get a DataSet that represents the group
    (i.e. the weather data pixel) that each forest cover pixel has been assigned to. Then merge that DataSet with the
    binarised forest cover DataArray.
    """

    forest_cover_group_dataset = xr.Dataset({
        'group_id': forest_cover_dataarrays['group_id'],
        'group_lat': forest_cover_dataarrays['weather_lat'],
        'group_lon': forest_cover_dataarrays['weather_lon']
    })

    forest_cover_merged_dataset = xr.merge([
        forest_cover_group_dataset,
        forest_cover_binary
    ])

    return forest_cover_merged_dataset


def aggregate_grouped_forest_cover_data(forest_cover_grouped_dataset):
    """
    Takes a DataSet containing pixelwise binarised forest cover, a group_id, and group_lat and group_lon values,
    converts it to a pandas DataFrame, and groups it by group_id, returning a DataFrame containing the groupwise
    * average of the binarised forest cover
    * first available value for group_lat
    * first available value for group_lon
    """
    forest_cover_merged_stacked = forest_cover_grouped_dataset.stack({'space': ['lat', 'lon']})

    forest_cover_df = pd.DataFrame({
        'forest_cover_binary': forest_cover_merged_stacked.forest_cover_binary.values[0],
        'group_id': forest_cover_merged_stacked.group_id.values[0],
        'group_lat': forest_cover_merged_stacked.group_lat.values[0],
        'group_lon': forest_cover_merged_stacked.group_lon.values[0],
    }, index=pd.MultiIndex.from_tuples(list(forest_cover_merged_stacked.space.values)))

    forest_cover_df_grouped = forest_cover_df.groupby('group_id').aggregate(
        {'forest_cover_binary': pd.Series.mean, 'group_lat': lambda x: x.iloc[0], 'group_lon': lambda x: x.iloc[0]}
    )

    return forest_cover_df_grouped


def convert_to_raster_and_write_to_output(forest_cover_aggregated_df: pd.DataFrame, output_path: str):
    """
    Takes the DataFrame output by aggregate_grouped_forest_cover_data, containing groupwise mean binarised forest cover,
    group latitude and group longitude, for each group, converts it to an xarray DataArray, and writes the output
    to a NetCDF file at the location given by output_path.
    """
    forest_cover_lowres_xarray = forest_cover_aggregated_df.rename(
        columns={'group_lat': 'lat', 'group_lon': 'lon'}
    ).set_index(['lat', 'lon']).to_xarray()

    forest_cover_lowres_xarray.to_netcdf(output_path)


def parse_args(arguments):
    return {
        'landcover_data_path': arguments['<landcover_data_path>'],
        'weather_data_path': arguments['<weather_data_path>'],
        'output_path': arguments['<output_path>']
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = docopt(__doc__)
    parsed_args = parse_args(args)
    main(**parsed_args)
