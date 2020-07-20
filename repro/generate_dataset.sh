dvc run -n merge_bands \
-d preparedata/merge_bands.py \
-d raw_data/s2_aws/tiles/31 \
-d raw_data/s2_aws/tiles/30 \
-o data/scenes \
"python preparedata/merge_bands.py /home/robert/ds-wildfire/raw_data/s2_aws/tiles -s 31/T/GH -s 31/T/GJ /home/robert/ds-wildfire/data/scenes"

dvc run -n patch_scenes \
-d preparedata/extract_patches.py \
-d preparedata/patchutils \
-d preparedata/patching.py \
-d raw_data/wildfires-ground-truth/france/vars/N_DFCI_CONTOUR_FEUX_2017_S_083.shp \
-d data/scenes/ \
-o data/extracted/ \
"python preparedata/extract_patches.py /home/robert/ds-wildfire/raw_data/wildfires-ground-truth/france/vars/N_DFCI_CONTOUR_FEUX_2017_S_083.shp /home/robert/ds-wildfire/data/extracted DATE_ECLOS /home/robert/ds-wildfire/data/scenes -s 31/T/GJ -s 31/T/GH"

dvc run -n sort_patches \
-d preparedata/sort_patches.py \
-d data/extracted/ \
-o data/training_patches/anno/ \
-o data/training_patches/patches/ \
"python preparedata/sort_patches.py /home/robert/ds-wildfire/data/training_patches /home/robert/ds-wildfire/excluded.txt /home/robert/ds-wildfire/data/extracted/patches /home/robert/ds-wildfire/data/extracted/anno"

dvc run -n split_data \
-d preparedata/split.py \
-d data/training_patches/patches/ \
-d data/training_patches/anno/ \
-o data/training_patches/training_indices.json \
"python preparedata/split.py --root=data/training_patches/"
