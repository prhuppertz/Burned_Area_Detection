dvc run -fn resnetunet \
-d data/post_fire_model_Scenario2b/training_patches/patches/ \
-d data/post_fire_model_Scenario2b/training_patches/training_indices.json \
-d data/post_fire_model_Scenario2b/training_patches/annotations/ \
-d segmentation/evaluation/ \
-d run_training.py \
-d segmentation/models/resnetunet/ \
-d segmentation/networks/resnetunet/ \
-d segmentation/data/ \
"python run_training.py --seed=11 --gpu=1 --save-images=1 --baseline=1 --model-name=resnetunet --group=resnetunet_Scenario2b --save-path=data/post_fire_model_Scenario2b/"

