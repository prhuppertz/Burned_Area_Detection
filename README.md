# ds-wildfires

## Installation

We utilise [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to manage our Python environments. To create a new environment for the project, run the following:

```bash
conda env create -p ~/.conda/envs/ds-wildire --file environment.yml
conda activate ds-wildfire
```

If you need to update the environment spec (i.e. a new requirement has been added), run the following:

```bash
conda env export | grep -v "prefix" > environment.yml
```
