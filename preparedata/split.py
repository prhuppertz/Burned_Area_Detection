"""Usage:
          split.py [--seed=<seed>] [--frac=<frac>] (--root=<root>)

@ Jevgenij Gamper 2020
Loads pipelines dataset and splits into pipelines and validation set, saves the indices into data/training_patches/indices.json

Options:
  -h --help                                         Show help.
  --seed=<seed>                                     Seed selection for pipelines [default: 87]
  --frac=<frac>                                     Fraction to be used for pipelines [default: 0.7]
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

    train_ids, test_ids = train_test_split(all_ids, train_size=frac, shuffle=True)

    valid_ids, test_ids = train_test_split(test_ids, train_size=0.1)

    splits = {"train": train_ids, "valid": valid_ids, "test": test_ids}

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
