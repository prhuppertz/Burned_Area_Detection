from segmentation.data.dataset import get_segmentation_dataset
from segmentation.data.augmentation import Augmentor, sequences
from segmentation.models.resnetunet.preprocessors import Preprocessor
from segmentation.models.resnetunet.postprocessor import Postprocessor
from segmentation.models.resnetunet.loss import loss
from segmentation.evaluation.metric import SegmentationMetric

def configure(hparams):
    augmentor = Augmentor(sequence = sequences.seq, augment_target = True)
    preprocessor = Preprocessor(w = hparams['w'], sigma = hparams['sigma'])
    postprocessor = Postprocessor()
    metric = SegmentationMetric()

    dictionary = {
        "get_dataset_func": get_segmentation_dataset,
        "augmentation_func": augmentor.augmentation_func,
        "preprocessing_func": preprocessor.preprocessing_func,
        "loss": loss,
        "postprocessing_func": postprocessor.postprocessing_func,
        "metric": metric.batch_compute,
        "postprocessing_func_single": postprocessor.postprocessing_func_single,
    }
    return dictionary
