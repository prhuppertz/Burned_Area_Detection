import xarray
import datetime
import clipping_functions as cf
import glob
from dateutil.relativedelta import relativedelta
import sys
sys.path.append("..")
from docopt import docopt
usage = '''
Usage: temporal_aggregation_RH_data.py init

'''
args = docopt(usage)
print(args)
if args['init']:
    NC_file_paths = glob.glob("/data/raw_data/era5/unpacked/R/*.nc")[8797:]
    date_bins = [datetime.datetime(2002, 1, 1) + i*relativedelta(days=5) for i in range(0,1093)]

    RH_array = xarray.open_mfdataset(NC_file_paths, combine='by_coords').load()
    five_day_average = RH_array.groupby_bins("time", date_bins).mean("time").persist()
    five_day_average = cf.change_longitude_to_neg(five_day_average)

    _, datasets = zip(*five_day_average.groupby('time_bins'))
    paths = ["../data/processed/5_day_averages/RH_{}.nc".format(str(date)[:10]) for date in date_bins]
    xarray.save_mfdataset(datasets, paths)

