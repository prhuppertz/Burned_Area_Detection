"""Usage:
          run_testing.py (--model-name=<model-name>) (--group=<group) (--image=<path_to_image>) (--shp=<shp_name>) [--b_size=<batch_size>]

@ Robert Huppertz 2020, Cervest
Applies trained neural network to mgrs scene, and saves the shapefile

Options:
  -h --help                                  Show help.
  --model-name=<model-name>                    Name of the model to be used [default: unet]
  --group=<group                             Group to tag experiment
  --image=<path_to_image>                    Path/name to input image
  --shp=<shp_name>                           Path/name of the output shapefile
  --b_size=<batch_size>                      Size of batches when loading MGRS patches into the model [default: 32]
"""
from tqdm import tqdm
from docopt import docopt
import importlib
import numpy as np
from torch.utils.data import DataLoader
from segmentation.models.utils import get_params, get_configuration
from segmentation.data.test.test_time import (
    load_stack,
    split_into_patches,
    find_checkpoint,
)
from segmentation.data.test.test_time import (
    apply_to_batch,
    mask_to_polygons,
    SceneDataset,
)
from shapely.geometry import mapping
from shapely.errors import TopologicalError
import fiona

schema = {
    "geometry": "Polygon",
    "properties": {"id": "int"}
    # TODO! add spatial info!
    # 'crs': {'init': 'epsg:27700'},
    # 'crs_wkt': 'PROJCS["OSGB 1936 / British National Grid",GEOGCS["OSGB
    # 1936",DATUM["OSGB_1936",SPHEROID["Airy 1830",6377563.396,299.3249646,
    # AUTHORITY["EPSG","7001"]],AUTHORITY["EPSG","6277"]],
    # PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
    # UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
    # AUTHORITY["EPSG","4277"]],PROJECTION["Transverse_Mercator"],
    # PARAMETER["latitude_of_origin",49],PARAMETER["central_meridian",-2],
    # PARAMETER["scale_factor",0.9996012717],PARAMETER["false_easting",400000],
    # PARAMETER["false_northing",-100000],UNIT["metre",1,
    # AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],
    # AXIS["Northing",NORTH],AUTHORITY["EPSG","27700"]]'
}


def main(model_name, group, path_to_image, shp_name, batch_size):
    """

    :return:
    """
    hparams = get_params(model_name)

    configuration_dict = get_configuration(model_name, hparams)

    # Load module specific to model-name
    model_module = importlib.import_module(
        "segmentation.models.{}.model".format(model_name)
    )

    checkpoint_path = find_checkpoint(group)

    model = model_module.Model.load_from_checkpoint(checkpoint_path)
    model.configuration = configuration_dict

    # Load the mgrs scene of interest
    img_array = load_stack(path_to_image)

    # View array as patches
    arrays = split_into_patches(img_array)
    shape = arrays.shape  # Get the shape of the array
    arrays = arrays.reshape(
        (shape[0] * shape[1], *shape[2:])
    )  # Reshape into (num_patches, **dims))
    dts = SceneDataset(arrays)
    loader = DataLoader(dts, batch_size=batch_size, shuffle=False, drop_last=False)

    # Process patches through neural net and produce binary masks
    masks = []
    for batch, _ in tqdm(loader):
        out = apply_to_batch(model, batch).squeeze()
        masks.append(out)
    masks = np.concatenate(masks, 0)

    masks = masks.reshape((shape[0], shape[1], shape[2], shape[3]))

    # Merge masks into a single mask of size of img_array
    masks = masks.transpose(0, 3, 1, 2).reshape(-1, masks.shape[1] * masks.shape[3])

    # Convert the binary mask into polygons
    polygons = mask_to_polygons(masks, min_area=200)

    # Save as a shapefile
    with fiona.open(shp_name, "w", "ESRI Shapefile", schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        for i, poly in enumerate(tqdm(polygons)):
            try:
                c.write(
                    {"geometry": mapping(poly), "properties": {"id": i},}
                )
            except TopologicalError as e:
                continue


if __name__ == "__main__":
    arguments = docopt(__doc__)
    model_name = arguments["--task-name"]
    path_to_image = arguments["--image"]
    batch_size = int(arguments["--b_size"])
    shp_name = arguments["--shp"]

    main(model_name, path_to_image, shp_name, batch_size)
