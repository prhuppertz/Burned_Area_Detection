"""Usage:
          train.py [--seed=<seed>] [--gpu=<id>] [--save-images=<images>] [--baseline=<boolean>] (--model-name=<model-name>) (--group=<group) (--save-path=<save-path>)

@ Jevgenij Gamper 2020
Trains either selected model, and saves model checkpoints under `data/models/task-name/..

Options:
  -h --help                                         Show help.
  --seed=<seed>                                     Seed selection for pipelines [default: 87]
  --gpu=<id>                                        GPU list. [default: 0]
  --model-name=<model-name>                         Name of the model to train on, as in segmentation/models/ 
  --group=<group                                    Group to tag experiment for wandb
  --save-path=<save-path>                           Path to save results, logs and checkpoints
  --save-images=<images>                            If validation images should be saved [default: 0]
  --baseline=<boolean>                              If baseline should be stored on test images aswell [default:0]
"""
from docopt import docopt
import importlib
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from src.segmentation.evaluation.metrics.various_metrics import dice_and_iou_arrays

PROJECT = "burned_area_detection"


def load_best(model_module, configuration, save_path):
    checkpoint_path = os.path.join(save_path, "checkpoints")
    ckpt = [i for i in os.listdir(checkpoint_path) if ".ckpt" in i][0]
    checkpoint_path = os.path.join(checkpoint_path, ckpt)
    model = model_module.Model.load_from_checkpoint(checkpoint_path).cuda()
    model.configuration = configuration
    return model


def move_best(save_path, group):
    checkpoint_path = os.path.join(save_path, "checkpoints")
    ckpt = [i for i in os.listdir(checkpoint_path) if ".ckpt" in i][0]
    checkpoint_path = os.path.join(checkpoint_path, ckpt)
    move_to = "checkpoint"
    os.makedirs(os.path.join(move_to, group), exist_ok=True)
    shutil.copy(
        checkpoint_path, os.path.join(move_to, group, os.path.basename(checkpoint_path))
    )


def get_results(model, loader, logger, path_to_save, save_images, baseline):
    iterator = iter(loader)
    scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(iterator)):
            scoring_dict = model.test_step(batch, batch_idx)
            scores = scores + scoring_dict["metrics"]

            # Save images
            images = scoring_dict["images"]
            targets = scoring_dict["targets"]
            predictions = scoring_dict["predictions"]
            if save_images:
                for i in range(len(images)):
                    fig = plot_results(
                        images[i].transpose(1, 2, 0),
                        targets[i],
                        predictions[i],
                        scoring_dict["metrics"][i],
                    )
                    image_path = os.path.join(
                        path_to_save,
                        "res_{}_{}_IoU{:.2f}dice{:.2f}.png".format(
                            batch_idx,
                            i,
                            scoring_dict["metrics"][i][0],
                            scoring_dict["metrics"][i][1],
                        ),
                    )
                    fig.savefig(image_path, dpi=800, bbox_inches="tight")
                    plt.close()

            # save baseline
            if baseline:
                baseline_predictions = []
                baseline_metric = []
                for i in range(len(images)):
                    image_mean = images[i][1, :, :].mean()
                    baseline_prediction = np.zeros(images[i][1, :, :].shape)
                    for x in range(len(images[i][1, :, 0])):
                        for y in range(len(images[i][1, 0, :])):
                            if images[i][1, x, y] < image_mean:
                                baseline_prediction[x, y] = 1.0
                    baseline_predictions.append(baseline_prediction)

                for i in range(len(images)):
                    baseline_metric.append(
                        dice_and_iou_arrays(baseline_predictions[i], targets[i])
                    )

                for i in range(len(images)):
                    fig = plot_results(
                        images[i].transpose(1, 2, 0),
                        targets[i],
                        baseline_predictions[i],
                        baseline_metric[i],
                    )
                    baseline_path = os.path.join(path_to_save, "baseline")
                    os.makedirs(baseline_path, exist_ok=True)
                    image_path = os.path.join(
                        baseline_path,
                        "baseline_res_{}_{}_IoU{:.2f}dice{:.2f}.png".format(
                            batch_idx, i, baseline_metric[i][0], baseline_metric[i][1]
                        ),
                    )
                    fig.savefig(image_path, dpi=800, bbox_inches="tight")
                    plt.close()

    logger.experiment.log(
        {"sample_scores": wandb.Table(data=scores, columns=["IoU", "dice"])}
    )
    return scores


def save_metrics(metrics, metrics_save_path):
    metrics_save_path = os.path.join(metrics_save_path, "metrics.json")
    metrics_dict = {"IoU": [s[0] for s in metrics], "dice": [s[1] for s in metrics]}
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_dict, f)


def main(model_name, seed, group, save_path, save_images, baseline):
    """

    :return:
    """
    save_path = os.path.join(save_path, "experiments", group)
    os.makedirs(save_path, exist_ok=True)

    hparams = get_params(model_name)

    configuration_dict = get_configuration(model_name, hparams)

    # setup wandb pipeline
    wandb_logger = WandbLogger(
        name="{}-{}-{}".format(group, model_name, seed),
        save_dir=save_path,
        project=PROJECT,
        group=group,
        tags=group,
    )

    train, valid, test = get_loaders(hparams, configuration_dict)

    model_module = importlib.import_module(
        "src.segmentation.models.{}.model".format(model_name)
    )

    model = model_module.Model(hparams)
    model.configuration = configuration_dict

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=30, verbose=False, mode="min"
    )

    # Pytorch lightning trainer
    trainer = Trainer(
        gpus=1,
        weights_summary="top",
        max_epochs=50,
        logger=wandb_logger,
        early_stop_callback=early_stop_callback,
        num_sanity_val_steps=0,
        callbacks=[LearningRateLogger()] if hparams["scheduler_type"] != None else None,
        default_root_dir=save_path,
    )

    trainer.fit(model, train_dataloader=train, val_dataloaders=valid)

    del model
    torch.cuda.empty_cache()

    save_path = os.path.join(save_path, PROJECT, wandb_logger.__getstate__()["_id"])

    model = load_best(model_module, configuration_dict, save_path)

    scores = get_results(model, valid, wandb_logger, save_path, save_images, baseline)

    save_metrics(scores, save_path)

    move_best(save_path, group)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    # Set gpu devices, then import pytorch and set random seeds
    seed = int(arguments["--seed"])
    gpu = arguments["--gpu"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    import random
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Other imports
    from src.segmentation.models.utils import get_params, get_configuration, plot_results
    from src.segmentation.data.utils import get_loaders

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import LearningRateLogger
    import wandb

    task_name = arguments["--model-name"]

    main(
        task_name,
        seed,
        arguments["--group"],
        arguments["--save-path"],
        bool(int(arguments["--save-images"])),
        bool(int(arguments["--baseline"])),
    )
