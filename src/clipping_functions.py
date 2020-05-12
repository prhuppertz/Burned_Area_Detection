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




def open_relative_humidity_file(date, base_path_to_RH_data):
    RH_path_for_date = base_path_to_RH_data + date  +"*.nc"
    RH_file_for_date = glob.glob(RH_date_for_date)
    return RH_file_for_date

def create_mask_from_geom(geom, xrArray):
    return regionmask.Regions([geom]).mask(xrArray.longitude, xrArray.latitude, lon_name = 'longitude', lat_name='latitude')
def extract_noon_array(xrArray):
    noon = xrArray.time[12]
    noon_xrArray = xrArray.where(xrArray.time == noon, drop=True)
    return noon_xrArray

def shift_longitude_to_negative_coords(xrArray):
    return xrArray.assign_coords({'longitude': (((xrArray.longitude + 180) %360)-180)})


