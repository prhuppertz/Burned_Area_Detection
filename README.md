# Burned Area Detection with Deep Learning and Sentinel-2
A pipeline to process Sentinel-2 remote sensing data and train a neural network for burned area detection.

A full summary of the project, its methods, results and findings can be found here:
https://www.notion.so/Burned-Area-Detection-with-Deep-Learning-and-Sentinel-2-2b5913e4c79c4d8fa8a8c6dc7b1b4ac4

## Pipeline Description
The pipeline is divided into four processing stages (1.-4.) and one training and validation stage (5.):
1. Merge bands: Merge the Green, NIR and SWIR bands of the atmospherically corrected Sentinel-2 data and create false-color images that emphasise burned area by transferring the spectral bands to RGB-channels and normalise the values to the 0-255 color scale.
2. Extract patches: Match the BA ground truth data to the false-color images by area and date; Extract small patches of size 128x128 from the false-color images and ground truth data. The smaller patch size leads to lower memory uptake when training.
3. Sort patches: Remove empty or broken patches from the datasets
4. Split patches: Randomly split the training patches into a training dataset (70% of the total data) and a validation dataset (30% of the total data)
5. run training: Take a neural network from the pytorch library (we used a Resnet-18 with a PSPNet in U-Net architecture) and train it on the processed patches and burned area ground truth. The best-performing epoch will be validated along a baseline model (see full doc) and the performance will be measured with a IoU and Dice coefficient. The training can be tracked with Weights and Biases (wandb.com).

## Directory Structure
```
├── .dvc/
├── data/
│   ├── raw_data/
│       ├── tiles/
│       ├── wildfires-ground-truth/
├── src/
│   ├── preparedata/
│       ├── merge_bands.py
│       ├── extract_patches.py
│       ├── sort_patches.py
│       ├── split.py
│       └── ...
│   ├── run_scripts/
│       ├── run_training.py
│       ├── repro/
│       └── ...
│   └── segmentation/
├── dvc.yaml
└── ...
```
__Directories :__
- `data` : raw data, processed data and validation results
- `src/preparedata`: modules to run processing stages of pipeline
- `src/run_scripts`: scripts to run training/validation and testing stages of pipeline
- `src/run_scripts/repro`: bash scripts to run data version control (dvc) pipelines
- `src/segmentation`: networks, miscellaneous utilities, models

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
### Obtain Data
First, obtain an atmospherically corrected version of the Sentinel-2 remote sensing data. You can do this either by downloading the Sentinel-1 Level 1C data product and atmospherically correct it yourself (e.g. with the SIAC algorithm) or download the atmospherically corrected Sentinel-2 Level-2 data product (that is corrected with the sen2cor algorithm).
All Sentinel-2 data is available at: https://scihub.copernicus.eu

For the pipeline to function properly, you'll have to make sure that the Level-2 Sentinel-2 tiles are structured in folders that show their MGRS coordinates and capture date.

#### Example for raw data folder structure
For spectral bands B03, B8A, B11 of Sentinel-2 at the 26/11/2016 at MGRS scene 29/S/MC :
```
raw_data
└───tiles
│   └───29
│       └───S
│           └───MC
│               └───2016
│                   └───11
│                       └───26
│                           └───0
│                               └──B03.tif
│                               └──B8A.tif
│                               └──B11.tif                                                   
```


### Setting up Data Version Control (DVC)

We use DVC to execute and reproduce the constructed pipeline. See https://dvc.org/doc for more information!

From the environment and root project directory, you first need to build
symlinks to data directories as:
```bash
$ (ds-wildfire) dvc init -q
$ (ds-wildfire) python src/run_scripts/repro/dvc.py --cache=where/dvc_cache/is/stored --link=where/raw_data/is/stored

```
if no `--link` specified, data will be stored by default into `data/` directory and default cache is `.dvc/cache`.

### Setting up Weights & Biases (wandb)
We use Weights&Biases to track the model performance during training and for various experiments. See https://docs.wandb.com/quickstart for an overview on how to setup wandb on your machine.

## Run the experiments
To run the pipeline you can either use the pre-constructed dvc pipeline or manually run the five pipeline stages.

### DVC (recommended)
The __DVC.yaml__ files specifies the single stages, the respective python commands and dependencies and outputs. 

To reproduce the full pipeline, execute:
```bash
$ (ds-wildfire) dvc repro
```

To reproduce a pipeline stage, execute:
```bash
$ (ds-wildfire) dvc repro -s stage_name
```

### Manually

Execute __merge_bands__ (e.g. for all MGRS scenes that cover Portugal):
```bash
$ python -m src.preparedata.merge_bands data/raw_data/tiles -s 29/S/MC -s 29/S/MD -s 29/S/NB -s 29/S/NC -s 29/S/ND -s 29/S/PB -s 29/S/PC -s 29/S/PD -s 29/T/ME -s 29/T/NE -s 29/T/NF -s 29/T/NG -s 29/T/PE -s 29/T/PF data/processed_data/scenes
```
Execute __extract_patches__ (e.g. for all MGRS scenes that cover Portugal and the burned area ground truth shapefile for 2016):
```bash
$ python -m src.preparedata.extract_patches data/raw_data/wildfires-ground-truth/portugal/AArdida2016_ETRS89PTTM06_20190813.shp data/processed_data/extracted DHFim data/processed_data/scenes -s 29/S/MC -s 29/S/MD -s 29/S/NB -s 29/S/NC -s 29/S/ND -s 29/S/PB -s 29/S/PC -s 29/S/PD -s 29/T/ME -s 29/T/NE -s 29/T/NF -s 29/T/NG -s 29/T/PE -s 29/T/PF
```
Execute __sort_patches__:
```bash
$ python -m src.preparedata.sort_patches data/processed_data/training_patches data/processed_data/extracted/patches data/processed_data/extracted/annotations
```
Execute __split__:
```bash
$ python -m src.preparedata.split --root=data/processed_data/training_patches/
```
Execute __run_training__ (e.g with a ResNet in U-Net structure, a seed of 11, on gpu 1, and save the results along with baseline model results):
```bash
$ python -m src.run_scripts.run_training --seed=11 --gpu=1 --save-images=1 --baseline=1 --model-name=resnetunet --group=resnetunet --save-path=data/results/
```

## Results
The validation results will be available in data/results/
The models best-performing epoch and its parameters will be available in checkpoints/
