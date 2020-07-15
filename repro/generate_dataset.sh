

dvc run -n merge_bands \
-O data/scenes \
"python preparedata/merge_bands.py /data/raw_data/s2_aws/tiles -s 31/T/GJ -s 31/T/GH data/scenes"

dvc run -n patch_scenes \
-d preparedata/ \
-d data/raw/wildfire-ground-truth \
-d data/scenes/ \
-O data/extracted/ \
"python preparedata/extract_patches.py /data/raw_data/wildfires-ground-truth/france/vars/N_DFCI_CONTOUR_FEUX_2017_S_083.shp /home/robert/ds-wildfire/data/extracted DATE_ECLOS /home/robert/ds-wildfire/data/scenes -s 31/T/GJ -s 31/T/GH"