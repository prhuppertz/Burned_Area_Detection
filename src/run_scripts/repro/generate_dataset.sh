dvc run -n merge_bands \
-d src/preparedata/merge_bands.py \

-o data/processed_data/scenes/29/S/MC/ \
-o data/processed_data/scenes/29/S/MD/ \
-o data/processed_data/scenes/29/S/NB/ \
-o data/processed_data/scenes/29/S/NC/ \
-o data/processed_data/scenes/29/S/ND/ \
-o data/processed_data/scenes/29/S/PB/ \
-o data/processed_data/scenes/29/S/PC/ \
-o data/processed_data/scenes/29/S/PD/ \
-o data/processed_data/scenes/29/T/ME/ \
-o data/processed_data/scenes/29/T/NE/ \
-o data/processed_data/scenes/29/T/NF/ \
-o data/processed_data/scenes/29/T/NG/ \
-o data/processed_data/scenes/29/T/PE/ \
-o data/processed_data/scenes/29/T/PF/ \
"python -m src.preparedata.merge_bands data/raw_data/tiles -s 29/S/MC -s 29/S/MD -s 29/S/NB -s 29/S/NC -s 29/S/ND -s 29/S/PB -s 29/S/PC -s 29/S/PD -s 29/T/ME -s 29/T/NE -s 29/T/NF -s 29/T/NG -s 29/T/PE -s 29/T/PF data/processed_data/scenes"

dvc run -n extract_patches \
-d data/processed_data/scenes/29/S/MC/ \
-d data/processed_data/scenes/29/S/MD/ \
-d data/processed_data/scenes/29/S/NB/ \
-d data/processed_data/scenes/29/S/NC/ \
-d data/processed_data/scenes/29/S/ND/ \
-d data/processed_data/scenes/29/S/PB/ \
-d data/processed_data/scenes/29/S/PC/ \
-d data/processed_data/scenes/29/S/PD/ \
-d data/processed_data/scenes/29/T/ME/ \
-d data/processed_data/scenes/29/T/NE/ \
-d data/processed_data/scenes/29/T/NF/ \
-d data/processed_data/scenes/29/T/NG/ \
-d data/processed_data/scenes/29/T/PE/ \
-d data/processed_data/scenes/29/T/PF/ \
-d data/raw_data/wildfires-ground-truth/portugal/AArdida2016_ETRS89PTTM06_20190813.shp \
-d src/preparedata/extract_patches.py \
-d src/preparedata/patching.py \
-d src/preparedata/patchutilspreparedata/extract_patches.py \

-o data/processed_data/extracted/ \
"python -m src.preparedata.extract_patches data/raw_data/wildfires-ground-truth/portugal/AArdida2016_ETRS89PTTM06_20190813.shp data/processed_data/extracted DHFim data/processed_data/scenes -s 29/S/MC -s 29/S/MD -s 29/S/NB -s 29/S/NC -s 29/S/ND -s 29/S/PB -s 29/S/PC -s 29/S/PD -s 29/T/ME -s 29/T/NE -s 29/T/NF -s 29/T/NG -s 29/T/PE -s 29/T/PF"

dvc run -n sort_patches \
-d data/processed_data/extracted/ \
-d src/preparedata/sort_patches.py \
-o data/processed_data/training_patches/annotations/ \
-o data/processed_data/training_patches/patches/ \
"python -m src.preparedata.sort_patches data/processed_data/training_patches data/processed_data/extracted/patches data/processed_data/extracted/annotations"

dvc run -n split_data \
-d data/processed_data/training_patches/annotations/ \
-d data/processed_data/training_patches/patches/ \
-d src/preparedata/split.py \
-o data/processed_data/training_patches/training_indices.json \
"python -m src.preparedata.split --root=data/processed_data/training_patches/"
