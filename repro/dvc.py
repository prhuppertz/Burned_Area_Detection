"""Usage:
          run_training.py [--cache=<cache_path>]  [--link=None] [--link2=None]

@ Jevgenij Gamper 2020, Cervest
Trains a selected type of neural network on field boundary data

Options:
  -h --help             Show help.
  --version             Show version.
  --cache=<cache_path>  Inference mode. 'roi' or 'wsi'. [default: data/]
  --link=<link>         Path to for symlink to points towards, if using remote storage
  --link2=<link>        2nd Path for raw_data for symlink to points towards, if using remote storage
"""
import os
import subprocess
from docopt import docopt


def set_cache(cache_dir):
    """
    Sets dvc cache given config file
    :param cache_dir: path to cache directory
    :return:
    """
    p = subprocess.Popen("dvc cache dir {} --local".format(cache_dir), shell=True)
    p.communicate()


def set_symlink():
    p = subprocess.Popen("dvc config cache.type symlink --local", shell=True)
    p.communicate()
    p = subprocess.Popen("dvc config cache.protected true --local", shell=True)
    p.communicate()


def main(cache_path, link_path, link2_path):
    """
    Sets up dvc for large data
    :return:
    """
    set_symlink()
    if cache_path:
        os.makedirs(cache_path, exist_ok=True)
        set_cache(cache_path)
    # Get directory of current project
    cur_project_dir = os.getcwd()
    # Make path to store data
    if link_path:
        to_folder = os.path.join(cur_project_dir, "data")
        from_folder = link_path
        os.makedirs(from_folder, exist_ok=True)
        # Create a symlink
        os.symlink(from_folder, to_folder)
    # Make a path where raw_data is stored, if data is stored externally from the other directory
    if link2_path:
        to_folder = os.path.join(cur_project_dir, "raw_data")
        from_folder = link2_path
        os.makedirs(from_folder, exist_ok=True)
        # Create a symlink
        os.symlink(from_folder, to_folder)
    # If does not want to specify symlinks then just create data dir
    else:
        to_folder = os.path.join(cur_project_dir, "data")
        os.makedirs(to_folder, exist_ok=True)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    cache_path = arguments["--cache"]
    link_path = arguments["--link"]
    link2_path = arguments["--link2"]
    main(cache_path, link_path, link2_path)
