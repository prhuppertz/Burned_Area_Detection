

dvc run -n merge_bands \
-d preparedata/merge_bands.py \
-d data/raw_data/s2_aws/tiles/31 \
-O data/scenes \
"python preparedata/merge_bands.py raw_data/s2_aws/tiles -s 31/T/GJ -s 31/T/GH data/scenes"

dvc run -n patch_scenes \
-d preparedata/extract_patches.py \
-d preparedata/patchutils \
-d preparedata/patching.py \
-d data/raw/wildfire-ground-truth/france/vars/N_DFCI_CONTOUR_FEUX_2017_S_083.shp \
-d data/scenes/ \
-O data/extracted/ \
"python preparedata/extract_patches.py raw_data/wildfires-ground-truth/france/vars/N_DFCI_CONTOUR_FEUX_2017_S_083.shp data/extracted DATE_ECLOS data/scenes -s 31/T/GJ -s 31/T/GH"