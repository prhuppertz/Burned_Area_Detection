from typeguard import typechecked
from typing import Dict, Tuple
import importlib
import yaml
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os


@typechecked
def assert_configuration(task_name: str, config_dict: Dict) -> bool:
    """
    Checks if task configuration file has the required specifications
    :params task_name:
    :param config_dict:
    :return:
    """
    required = {
        "augmentation_func",
        "preprocessing_func",
        "get_dataset_func",
        "test_preprocessing_func",
        "postprocessing_func",
        "metric",
    }
    keys = set(config_dict.keys())

    extra = keys - required
    if keys == required:
        return True
    elif extra:
        warnings.warn("Task {} has extra configurations: {}".format(task_name, extra))
        return True
    else:
        return False


@typechecked
def get_params(task_name: str) -> Dict:
    """
    TODO! add docstring
    :param task_name:
    :param path_to_data_dir:
    :return:
    """
    path_to_params = os.path.join(
        "src/segmentation/models/{}".format(task_name), "params.yml"
    )
    with open(path_to_params) as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


@typechecked
def get_configuration(task_name: str, hparams: Dict) -> Dict:
    """
    # TODO!
    :param task_name:
    :param hparams:
    :return:
    """
    configure_module = importlib.import_module(
        "src.segmentation.models.{}.configuration".format(task_name)
    )
    configuration = configure_module.configure(hparams)
    return configuration


@typechecked
def plot_results(
    image: np.ndarray,
    target_mask: np.ndarray,
    prediction: np.ndarray,
    scores: Tuple[float, float],
):
    """
    # TODO!
    :param image:
    :param target_mask:
    :param prediction:
    :param score:
    :return:
    """
    to_visualise = [image, target_mask, prediction]
    titles = ["Image", "Target", "Prediction"]
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.imshow(to_visualise[i])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(titles[i], fontdict={"fontsize": 16})
    fig.suptitle("IoU: {:.2f}; dice: {:.2f}".format(scores[0], scores[1]), fontsize=16)
    return fig
