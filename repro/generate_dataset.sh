

dvc run -n merge_bands \
-O data/scenes \
"python preparedata/merge_bands.py /data/raw_data/s2_aws/tiles -s 31/T/GJ -s 31/T/GH data/scenes"

#dvc run -n patch_scenes \
#-d preparedata/ \
#-d data/shapefile/ \
#-d data/scenes/ \
#-O data/all_extracted/ \
#"python preparedata/extract_patches.py data/raw/france/vars/french_farms_full_cliped.shp data/all_extracted CODE_CULTU data/scenes -s 31UFQ -s 31UEQ"