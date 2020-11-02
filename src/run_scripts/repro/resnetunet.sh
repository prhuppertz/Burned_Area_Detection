dvc run -fn resnetunet \
-d data/processed_data/training_patches/annotations/ \
-d data/processed_data/training_patches/patches/ \
-d data/processed_data/training_patches/training_indices.json \
-d src/run_scripts/run_training.py \
-d src/segmentation/data/ \
-d src/segmentation/evaluation/ \
-d src/segmentation/models/resnetunet/ \
-d src/segmentation/networks/resnetunet/ \

"python -m src.run_scripts.run_training --seed=11 --gpu=1 --save-images=1 --baseline=1 --model-name=resnetunet --group=resnetunet --save-path=data/results/"

