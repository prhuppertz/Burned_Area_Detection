# ds-wildfires

## Installation
Clone the repo:
$git clone git@github.com:prhuppertz/Burned_Area_Detection.git

Install the packages:
$pip install requirements.txt

## Obtain Sentinel-2 Data
First, obtain an atmospherically corrected version of the Sentinel-2 remote sensing data. You can do this either by downloading the Sentinel-1 Level 1C data product and atmospherically correct it yourself (e.g. with the SIAC algorithm) or download the atmospherically corrected Sentinel-2 Level-2 data product (that is corrected with the sen2cor algorithm).
All Sentinel-2 data is available at: https://scihub.copernicus.eu

For the pipeline to function properly, you'll have to make sure that the Level-2 Sentinel-2 tiles are structured in folders that show their MGRS coordinates and capture date.
For example, the Sentinel-2 data for the 26/11/2016 at the MGRS scene 29/S/MC would look like this:

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

# Pipeline
The Pipeline is structured into five different stages that process the raw data, train and validate a ResNet-18 (or any other model) on the data.

The processing steps are located in $src/preparedata/
1. Merge bands: Merge the Green, NIR and SWIR bands of the atmospherically corrected Sentinel-2 data and create false-color images that emphasise burned area by transferring the spectral bands to RGB-channels and normalise the values to the 0-255 color scale.
2. Extract patches: Match the BA ground truth data to the false-color images by area and date; Extract small patches of size 128x128 from the false-color images and ground truth data. The smaller patch size leads to lower memory uptake when training.
3. Sort patches: Remove empty or broken patches from the datasets
4. Split patches: Randomly split the training patches into a training dataset (70% of the total data) and a validation dataset (30% of the total data)

The training and validation stages are located in $src/run_scripts/
5. run training: Take a neural network from the pytorch library (we used a Resnet-18 with a PSPNet in U-Net architecture) and train it on the processed patches and burned area ground truth. The best-performing epoch will be validated along a baseline model (see full doc) and the performance will be measured with a IoU and Dice coefficient. The training can be tracked with Weights and Biases (wandb.com).

-> Each script has to be run with a certain degree of additional arguments and options (see the scripts command line interface in the scripts)

## Run the experiments
To run the pipeline you can either use the pre-constructed dvc pipeline or manually run the five pipeline steps.

### DVC (recommended)
To use DVC, simply run:
$dvc repro
and the pipeline should go through all six stages automatically and create the desired results for you 

### Run manually 
To run the pipeline manually, you can 
Each script has to be run with a certain degree of additional arguments and options (see the scripts command line interface in the scripts)
To create the correct symlinks to the raw remote sensing data, follow these steps:
1. $ln -s /path/to/where the  /path/to/link