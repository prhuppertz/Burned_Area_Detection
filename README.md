# Burned Area Detection with Deep Learning and Sentinel-2
A pipeline to process Sentinel-2 remote sensing data and train a neural network for burned area detection.
## Pipeline Description
The pipeline is divided into four processing stages (1.-4.) and one training and validation stage (5.):
1. Merge bands: Merge the Green, NIR and SWIR bands of the atmospherically corrected Sentinel-2 data and create false-color images that emphasise burned area by transferring the spectral bands to RGB-channels and normalise the values to the 0-255 color scale.
2. Extract patches: Match the BA ground truth data to the false-color images by area and date; Extract small patches of size 128x128 from the false-color images and ground truth data. The smaller patch size leads to lower memory uptake when training.
3. Sort patches: Remove empty or broken patches from the datasets
4. Split patches: Randomly split the training patches into a training dataset (70% of the total data) and a validation dataset (30% of the total data)
5. run training: Take a neural network from the pytorch library (we used a Resnet-18 with a PSPNet in U-Net architecture) and train it on the processed patches and burned area ground truth. The best-performing epoch will be validated along a baseline model (see full doc) and the performance will be measured with a IoU and Dice coefficient. The training can be tracked with Weights and Biases (wandb.com).

## Directory Structure

```
raw_data
└───tiles
│   └───29
│       └───S
│           └───MC
│               └───2016
│                   └───11
|                       └───...
│                       └───26
│                           └───0
                                |   B01.tif
                                |   ...
                                |   B8A.tif
                                |   ...
                                |   B11.tif
                                |   ...                                                    
│                       └───29
│                           └───...   
└───wildfires-ground-truth
    │   file021.shp
    │   file022.shp
```

__Directories :__
- `data/` : Landsat-MODIS reflectance time series dataset and experiments outputs
- `repro/`: bash scripts to run data version control pipelines
- `src/`: modules to run reflectance patches extraction and deep reflectance ds-wildfire experiments
- `tests/`: unit testing
- `utils/`: miscellaneous utilities





## Installation

Code implemented in Python 3.8.2

### Setting up environment

Clone and go to repository

```bash
$ git clone https://github.com/prhuppertz/Burned_Area_Detection.git
$ cd ds-wildfire
```

Create and activate environment
```bash
$ pyenv virtualenv 3.8.2 ds-wildfire
$ pyenv activate ds-wildfire
$ (ds-wildfire)
```

Install dependencies
```bash
$ (ds-wildfire) pip install -r requirements.txt
```

#### Setting up dvc

From the environment and root project directory, you first need to build
symlinks to data directories as:
```bash
$ (ds-wildfire) dvc init -q
$ (ds-wildfire) python src/run_scripts/repro/dvc.py --link=where/data/stored --cache=where/dvc_cache/stored

```
if no `--link` specified, data will be stored by default into `data/` directory and default cache is `.dvc/cache`.

To reproduce a pipeline stage, execute:
```bash
$ (ds-wildfire) dvc repro -s stage_name
```
In case pipeline is broken, hidden bash files are provided under `repro` directory

## Obtain Data
First, obtain an atmospherically corrected version of the Sentinel-2 remote sensing data. You can do this either by downloading the Sentinel-1 Level 1C data product and atmospherically correct it yourself (e.g. with the SIAC algorithm) or download the atmospherically corrected Sentinel-2 Level-2 data product (that is corrected with the sen2cor algorithm).
All Sentinel-2 data is available at: https://scihub.copernicus.eu

For the pipeline to function properly, you'll have to make sure that the Level-2 Sentinel-2 tiles are structured in folders that show their MGRS coordinates and capture date, similar to the project structure above.

## Run the experiments
To run the pipeline you can either use the pre-constructed dvc pipeline or manually run the five pipeline steps.

### DVC (recommended)
To use DVC, simply run:
$dvc repro
and the pipeline should go through all six stages automatically and create the desired results for you 

### Run manually 
To create the correct symlinks to the raw remote sensing data, follow these steps:
1. $ln -s /path/to/where the  /path/to/link

### Running experiments

Setup YAML configuration files specifying experiment : dataset, model, optimizer, experiment. See [here](https://github.com/Cervest/ds-generative-reflectance-ds-wildfire/tree/master/src/deep_reflectance_ds-wildfire/config) for examples.

Execute __training__ on, say GPU 0, as:
```bash
$ python run_training.py --cfg=path/to/config.yaml --o=output/directory --device=0
```

Once training completed, specify model checkpoint to evaluate in previously defined YAML configuration file and run __evaluation__ as:

```bash
$ python run_testing.py --cfg=path/to/config.yaml --o=output/directory --device=0
```