import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio import features
from affine import Affine
import matplotlib.pyplot as plt 
import xarray as xr
import regionmask
import glob
from shapely.geometry import Polygon, MultiPolygon, shape
import re
import datetime


def open_shp_file_with_consistent_crs(shp_file, country, consistent_crs):
    '''
    Open shapefiles, and change CRS so they aisre all consistent.
    Italy does not have a CRS set to it but is provided on the website so it is set independently
    '''
    open_shp_file = gpd.read_file(shp_file)
    if country == 'italy':
        open_shp_file.crs = {'init' :'epsg:3003'} 
    wildfire_shp_file_changed_crs = open_shp_file.to_crs(consistent_crs)
    return wildfire_shp_file_changed_crs

def return_closest_datebin(date, date_bins):
    seperator = re.findall("[^0-9]", date)[0]
    dates_sep = re.findall("(\d+)", date)
    date_format = None
    if len(date) == 8:
        year = '20' + dates_sep[2] 
        date = dates_sep[0] + seperator + dates_sep[1] + seperator + year
        date_format = '%d{}%m{}%Y'.format(seperator, seperator)
    else:
        if len(dates_sep[0]) == 4:
            date_format = '%Y{}%m{}%d'.format(seperator, seperator)
        elif len(dates_sep[0]) == 2:
            date_format = '%d{}%m{}%Y'.format(seperator, seperator)
    
    
    date = datetime.datetime.strptime(date, date_format)
    return min(date_bins, key=lambda x: abs(x-date))

def read_WF_shp_file(file):
    WF_ground_truth = gpd.read_file(file)
    return WF_ground_truth

def get_consistent_crs_shp_file(file, crs):
    open_file = read_WF_shp_file(file)
    open_file_changed_crs = open_file.to_crs(crs)
    return open_file_changed_crs


def open_RH_file_for_date(date, base_path_to_weather_data):
    """
    opens relative humidity (RH) data for given date and path to weather data
    """
    RH_path_for_date = base_path_to_weather_data + date +"*.nc"
    RH_file_for_date = glob.glob(RH_path_for_date)
    open_RH_data = xr.open_dataset(RH_file_for_date[0])
    return open_RH_data

def get_noon_RH_data(RH_data):
    noon = RH_data.time[12]
    noon_RH_data = RH_data.where(RH_data.time == noon, drop=True)
    return noon_RH_data
    
def change_longitude_to_neg(RH_data):
    return RH_data.assign_coords({'longitude': (((RH_data.longitude +180)%360) -180)})

def mask_RH_data_to_region(all_nuts_regions, nuts_code, crs, RH_data):
    '''
    provide opened NUTS regio file, NUTS code of interest, CRS to change it to and the xarray data of interest
    
    '''
    nuts_code_region = get_nuts_shp_for_nuts_code(all_nuts_regions, nuts_code, crs)
    region_mask = create_mask_from_geom(geom=nuts_code_region.geometry.iloc[0], xrArray=RH_data)
    RH_data['mask'] = region_mask
    return RH_data

def get_wf_mask(noon_RH_boun, nearest_coord):
    noon_RH_copy = noon_RH_bound.copy()
    noon_RH_clipped = noon_RH_copy.where(noon_RH_copy.longitude == nearest_coord[0].values, 1).where(noon_RH_copy.latitude == nearest_coord[1].values, 1)
    return noon_RH_clipped.mask


def open_nuts_shp_with_consistent_crs(crs):
    all_nuts_regions_shp_1m_path = '/data/raw_data/nuts_regions_highres//NUTS_RG_01M_2016_3857.geojson'
    all_nuts_regions = gpd.read_file(all_nuts_regions_shp_1m_path)
    all_nuts_regions_changed_crs = all_nuts_regions.to_crs(crs)
    return all_nuts_regions_changed_crs

def get_nuts_shp_for_nuts_code(all_nuts_regions,nuts_code, crs):
    nuts_code_region = all_nuts_regions.query("NUTS_ID == '{}'".format(nuts_code))
    return nuts_code_region



def create_mask_from_geom(geom, xrArray):
    return regionmask.Regions([geom]).mask(xrArray.longitude, xrArray.latitude, lon_name = 'longitude', lat_name='latitude', wrap_lon=True)


def shift_longitude_to_negative_coords(xrArray):
    return xrArray.assign_coords({'longitude': (((xrArray.longitude + 180) %360)-180)})

def get_nearest_min_max_coords(WF_shp, noon_RH_bound):
    WF_shp_bounds = WF_shp.geometry.bounds
    nearest_min_long, nearest_min_lat = nearest_latlong(WF_shp_bounds[0], WF_shp_bounds[1], noon_RH_bound)
    nearest_max_long, nearest_max_lat = nearest_latlong(WF_shp_bounds[2], WF_shp_bounds[3], noon_RH_bound)
    nearest_min_coord = (nearest_min_long, nearest_min_lat)
    nearest_max_coord = (nearest_max_long, nearest_max_lat)           
    return nearest_min_coord, nearest_max_coord
    
def nearest_latlong(long, lat, xrArray):
    nearest_lat = xrArray.sel(latitude = lat, method='nearest')
    nearest_long = xrArray.sel(longitude = long, method='nearest')
    return nearest_long.longitude, nearest_lat.latitude

def get_bounded_landcover_map(landcover_chunked, bounds):
    lat_bounded = landcover_chunked.where(landcover_chunked.latitude >= bounds.miny.iloc[0], drop = True).where(landcover_chunked.latitude <= bounds.maxy.iloc[0], drop = True)
    landcover_map_bounded = lat_bounded.where(lat_bounded.longitude >= bounds.minx.iloc[0], drop=True).where(lat_bounded.longitude <= bounds.maxx.iloc[0], drop=True)
    return landcover_map_bounded

def get_name_of_date_column(wildfire_shp_file_changed_crs, part_date_name_country):
    column_list = wildfire_shp_file_changed_crs.columns
    for column_name in column_list:
        if part_date_name_country in column_name:
            return column_name
        
    return None
    
def change_dimension_names(landcover_map):
    landcover_map.coords['latitude'] = ('lat', landcover_map.lat)
    landcover_map.coords['longitude'] = ('lon', landcover_map.lon)
    landcover_map_changeddims = landcover_map.swap_dims({'lat': 'latitude', 'lon': 'longitude'})
    landcover_map_changeddims.drop(['lat','lon'])
    return landcover_map_changeddims
