dvc run -n resnetunet \
-d data/training_patches/patches/ \
-d data/training_patches/training_indices.json \
-d data/training_patches/anno/ \
-o checkpoint/ \
"python run_training.py --seed=11 --gpu=1 --save-images=1 --model-name=resnetunet --group=resnetunet --save-path=data/"
