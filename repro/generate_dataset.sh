export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

dvc run -n download_composites \
-O data/scenes \
"python preparedata/merge_bands.py s3://cervest-sentinel-s2-l1c-composites/80.0/4.0/10/  data/scenes -s 31UFQ -s 31UEQ"

dvc run -n patch_scenes \
-d preparedata/ \
-d data/shapefile/ \
-d data/scenes/ \
-O data/all_extracted/ \
"python preparedata/extract_patches.py data/shapefile/french_farms_full_cliped.shp data/all_extracted CODE_CULTU data/scenes -s 31UFQ -s 31UEQ"


dvc run -n sort_patches \
-d preparedata/ \
-d data/all_extracted/ \
-O data/training_patches/anno/ \
-O data/training_patches/patches/ \
"python preparedata/sort_patches.py data/training_patches/ excluded.txt data/all_extracted/patches data/all_extracted/anno"

dvc run -n split_data \
-d preparedata/ \
-d data/training_patches/patches/ \
-O data/training_patches/training_indices.json \
"python preparedata/split.py --root=data/training_patches/"
