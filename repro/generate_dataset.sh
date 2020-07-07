export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

dvc run -n merge_bands \
-O data/scenes \
"python preparedata/merge_bands.py /data/raw_data/s2_aws/tiles 31/T/GJ ['2016/7/11', '2016/7/14','2016/7/17','2016/7/21'] /data/temporary/robert/ds-wildfire/data/scenes"

