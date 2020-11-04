"""Usage:
        dvc.py [--cache=<cache_path>] [--link=None] [--link2=None] 

@ Jevgenij Gamper & Robert Huppertz 2020, Cervest
Creates symlinks/DVC structure

Options:
  -h --help             Show help.
  --version             Show version.
  --cache=<cache_path>  Inference mode. 'roi' or 'wsi'. [default: data/]
  --link=<link>         Source path for symlink to points towards, if using remote storage
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

def main(cache_path, link_path):
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

    # Make a path where raw_data is stored, if data is stored externally
    if link_path:
        to_folder = os.path.join(cur_project_dir, "data/raw_data/tiles")
        from_folder = link_path
        os.makedirs(from_folder, exist_ok=True)
        os.makedirs(to_folder, exist_ok=True)
        #Create a symlink
        os.symlink(from_folder, to_folder)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    cache_path = arguments["--cache"]
    link_path = arguments["--link"]

    main(cache_path, link_path)
