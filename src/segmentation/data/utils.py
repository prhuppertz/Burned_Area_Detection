from torch.utils.data import Dataset, DataLoader
from typeguard import typechecked
from typing import Tuple, Dict, Optional, Callable
import numpy as np
import copy
import json
import os
from torchbearer.cv_utils import DatasetValidationSplitter, SubsetDataset


@typechecked
def split(
    dataset: Dataset, split_fraction: float, seed: int, return_test: bool = True
) -> Tuple[Dataset, ...]:
    """
    Splits the dataset into pipelines and validation (and testing if return_test is true)
    :param dataset: Dataset object
    :param split_fraction: Fraction of the whole dataset to be used for validation
    :param seed: Seed used for splitting
    :param return_test: if should split into three parts
    :return:
    """

    splitter = DatasetValidationSplitter(
        len(dataset), split_fraction, shuffle_seed=seed
    )

    trainset = splitter.get_train_dataset(dataset)
    valset = splitter.get_val_dataset(dataset)

    # Split the valset into test and validation
    if return_test:
        # Set split_fraction to low value such that testset
        valset, testset = split(
            valset, split_fraction=0.70, seed=seed, return_test=False
        )
        return trainset, valset, testset
    else:
        return trainset, valset


@typechecked
def get_splits(root: str):
    """
    Reads splits file indices.json
    :param root:
    :return:
    """
    with open(os.path.join(root, "training_indices.json")) as f:
        dict = json.load(f)
    return dict


def get_dataset(root, get_func, ids, aug_func, prep_func):

    dataset = get_func(root, augmentation_func=aug_func, preprocessing_func=prep_func)

    return SubsetDataset(dataset, ids)


@typechecked
def get_loaders(
    params: Dict, configuration: Dict, aug=True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    First gets the datasets with their specified processing functions and loads thm into dataloader
    :param params: Dictionary of hyper-parameters
    :param configuration: Dictionary of functions that are used in that specific task for dataloading and processing
    :param seed: Random seed to split the dataset
    :return: Training and validation dataloaders
    """
    root = params["root"]
    batch_size = params["batch_size"]
    aug_func = configuration["augmentation_func"] if aug == True else None
    prep_func = configuration["preprocessing_func"]
    get_dataset_func = configuration["get_dataset_func"]

    # Split into pipelines and validation
    splits = get_splits(root)
    train = get_dataset(root, get_dataset_func, splits["train"], aug_func, prep_func)
    valid = get_dataset(root, get_dataset_func, splits["valid"], None, prep_func)
    test = get_dataset(root, get_dataset_func, splits["test"], None, prep_func)

    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=10)

    valid = DataLoader(valid, batch_size=128, shuffle=False, num_workers=10)

    test = DataLoader(test, batch_size=128, shuffle=False, num_workers=10)

    return train, valid, test


@typechecked
def _change_preprocessing_func(dataset: Dataset) -> Dataset:
    """
    Copies original dataset, and re-assigns preprocessing function of the copy
    :param dataset:
    :param test_preprocessing_func: New pre-processing function
    :return:
    """
    dataset_copy = copy.deepcopy(dataset)
    dataset_copy.augmentation_func = None
    return dataset_copy


def batched(mean=True, argnum=1):
    """
    Wrapper for function that operate on instance to operate on a batch
    :param func: function applied to a single instance datapoint
    :return: same function wrapped to operate on a batch
    """

    def decorator(func):
        def apply_on_batch(*args, **kwargs):
            """
            :param arr: array of (batch_size, class_num, ...)
            :return:
            """

            if argnum == 2:
                pred, target = args[0], args[1]
            else:
                pred = args[0]
            result = []
            for idx in range(len(pred)):
                if argnum == 2:
                    res = func(pred[idx], target[idx], **kwargs)
                else:
                    res = func(pred[idx], **kwargs)
                result.append(res)
            if mean == True:
                return np.mean(result)
            if mean == False:
                return np.array(result)

        return apply_on_batch

    return decorator
