# ds-wildfires

## Installation

We utilise [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to manage our Python environments. To create a new environment for the project, run the following:

```bash
conda create --name ds-wildfire --file .conda/env.txt
conda activate ds-wildfire
```

If you need to update the environment spec (i.e. a new requirement has been added), run the following:

```bash
conda list --explicit > .conda/env.txt
```
