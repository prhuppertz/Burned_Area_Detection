

dvc run -n merge_bands \
-O data/scenes \
"python preparedata/merge_bands.py /data/raw_data/s2_aws/tiles -s 31/T/GJ -s 31/T/GH data/scenes"

