from segmentation.data.dataset import get_segmentation_dataset
from segmentation.data.augmentation import Augmentor, sequences
from segmentation.data.preprocessing.mask_functions import convert_to_tensors_float, reduce_to_semantic_mask_add_xy, append_semantic_mask_add_xy
from segmentation.data.preprocessing.mask_functions import channel_first
from segmentation.data.preprocessing.preprocessor import Preprocessor
from segmentation.evaluation.pq_bf.panoptic_quality import single_channel_pq
from segmentation.data.postprocessing.postprocessor import Postprocessor
from segmentation.data.postprocessing.classification import sigmoid
from segmentation.data.postprocessing.hover import xy_watershed
from segmentation.models.hover.utils import hover_loss

augmentor = Augmentor(sequence=sequences.seq, augment_target=False)
preprocessor = Preprocessor(sequence=[reduce_to_semantic_mask_add_xy, channel_first, convert_to_tensors_float])
test_preprocessor = Preprocessor(sequence=[append_semantic_mask_add_xy, channel_first, convert_to_tensors_float])
postprocessor = Postprocessor(sequence=[sigmoid, xy_watershed])


dictionary = {
    "get_dataset_func": get_segmentation_dataset,
    "augmentation_func": augmentor.augmentation_func,
    "preprocessing_func": preprocessor.preprocessing_func,
    "test_preprocessing_func": test_preprocessor.preprocessing_func,
    "loss": hover_loss,
    "postprocessing_func": postprocessor.postprocessing_func,
    "metric": single_channel_pq
}




