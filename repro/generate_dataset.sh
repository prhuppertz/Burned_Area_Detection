dvc run -n merge_bands \
-d preparedata/merge_bands.py \
-o data/scenes \
"python preparedata/merge_bands.py /home/robert/ds-wildfire/raw_data/s2_aws/tiles -s 29/S/MC -s 29/S/MD -s 29/S/NB -s 29/S/NC -s 29/S/ND -s 29/S/PB -s 29/S/PC -s 29/S/PD -s 29/T/ME -s 29/T/NE -s 29/T/NF -s 29/T/NG -s 29/T/PE -s 29/T/PF  /home/robert/ds-wildfire/data/scenes"

dvc run -n patch_scenes \
-d preparedata/extract_patches.py \
-d preparedata/patchutils \
-d preparedata/patching.py \
-d raw_data/wildfires-ground-truth/portugal/AArdida2016_ETRS89PTTM06_20190813.shp \
-d data/scenes/ \
-o data/post_fire_model_Scenario2b/extracted/ \
"python preparedata/extract_patches.py /home/robert/ds-wildfire/raw_data/wildfires-ground-truth/portugal/AArdida2016_ETRS89PTTM06_20190813.shp /home/robert/ds-wildfire/data/post_fire_model_Scenario2b/extracted DHFim /home/robert/ds-wildfire/data/scenes -s 29/S/MC -s 29/S/MD -s 29/S/NB -s 29/S/NC -s 29/S/ND -s 29/S/PB -s 29/S/PC -s 29/S/PD -s 29/T/ME -s 29/T/NE -s 29/T/NF -s 29/T/NG -s 29/T/PE -s 29/T/PF"

dvc run -n sort_patches \
-d preparedata/sort_patches.py \
-d data/post_fire_model_Scenario2b/extracted/ \
-o data/post_fire_model_Scenario2b/training_patches/annotations/ \
-o data/post_fire_model_Scenario2b/training_patches/patches/ \
"python preparedata/sort_patches.py /home/robert/ds-wildfire/data/post_fire_model_Scenario2b/training_patches /home/robert/ds-wildfire/excluded.txt /home/robert/ds-wildfire/data/post_fire_model_Scenario2b/extracted/patches /home/robert/ds-wildfire/data/post_fire_model_Scenario2b/extracted/annotations"

dvc run -n split_data \
-d preparedata/split.py \
-d data/post_fire_model_Scenario2b/training_patches/patches/ \
-d data/post_fire_model_Scenario2b/training_patches/annotations/ \
-o data/post_fire_model_Scenario2b/training_patches/training_indices.json \
"python preparedata/split.py --root=data/post_fire_model_Scenario2b/training_patches/"
