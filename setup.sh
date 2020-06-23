# Assumes that data is filed analogously to the Cervest Science NAS

# Symlink to landcover data:
ln -fs /data/raw_data/un_fao_land_cover_map/unpacked/C3S-LC-L4-LCCS-Map-300m-P1Y-2018-v2.1.1.nc data/raw/landcover/UN_FAO_LCCS_2018.nc

# Symlink to reference raster for landcover upscaling:
ln -fs /data/raw_data/era5/unpacked/R/417331.R.2678.e5.oper.an.pl.128_157_r.ll025sc.1979111100_1979111123.nc data/raw/landcover/era5_0.25deg_ref_raster.nc
