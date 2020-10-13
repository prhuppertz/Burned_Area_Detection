"""
Usage:
          split.py [--seed=<seed>] [--frac=<frac>] (--root=<root>)

@ Robert Huppertz, Jevgenij Gamper - Cervest, 2020
Splits data in root directory into training, validation and test set 
by storing the patch IDs in a json file

Options:
  -h --help                                         Show help.
  --seed=<seed>                                     Seed selection for pipelines [default: 87]
  --frac=<frac>                                     Fraction of the the total dataset that is used for training [default: 0.7]
  --root=<root>                                     Location where data is stored
"""
import math
import json
from docopt import docopt
from sklearn.model_selection import train_test_split
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from segmentation.data.dataset import get_segmentation_dataset


def split(root, frac):

    dataset = get_segmentation_dataset(
        root, augmentation_func=None, preprocessing_func=None
    )

    dataset_len = len(dataset)

    all_ids = list(range(dataset_len))

    #splitting into training dataset and general testing data
    train_ids, test_ids = train_test_split(all_ids, train_size=frac, shuffle=True)

    #splitting the general testing data into a validation dataset (fraction of train_size) and a testing dataset
    valid_ids, test_ids = train_test_split(test_ids, train_size=0.9)

    splits = {"train": train_ids, "valid": valid_ids, "test": test_ids}

    #saving the splitted IDs in a json file
    with open(root + "training_indices.json", "w") as fp:
        json.dump(splits, fp)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    # Set seeds
    seed = int(arguments["--seed"])
    frac = float(arguments["--frac"])
    root = arguments["--root"]
    import torch
    import random
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    split(root, frac)
