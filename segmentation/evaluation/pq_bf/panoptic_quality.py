import numpy as np
import cv2
from typeguard import typechecked
from typing import List, Union, Tuple
from scipy.optimize import linear_sum_assignment

@typechecked
def get_contour(mask: np.ndarray):
    """
    Gets countour of the object in an mask
    :param mask: Single channel mask
    :return:
    """
    contours, _ = cv2.findContours(
        mask.copy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # findContours returns a list of numpy arrays
    contours_flattened = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            contours_flattened.append(contours[i][j][0].tolist())

    return contours_flattened


def calc_precision_recall(contours_a, contours_b, threshold):
    """
    Computes pixel-wise precision or recall for the contour boundaries
    :param contours_a:
    :param contours_b:
    :param threshold:
    :return:
    """
    top_count = 0
    try:
        for b in range(len(contours_b)):

            # find the nearest distance
            for a in range(len(contours_a)):
                dist = (contours_a[a][0] - contours_b[b][0]) * \
                       (contours_a[a][0] - contours_b[b][0])
                dist = dist + \
                       (contours_a[a][1] - contours_b[b][1]) * \
                       (contours_a[a][1] - contours_b[b][1])
                if dist < threshold * threshold:
                    top_count = top_count + 1
                    break

        precision_recall = top_count / len(contours_b)
    except Exception as e:
        precision_recall = 0

    return precision_recall

def compute_boundary_f_score(ground_truth: np.ndarray,
                             prediction: np.ndarray,
                             paired_gt_instances: List[int],
                             paired_pred_instances: List[int],
                             threshold: float = 2) -> float:
    """
    Computes boundary F-score
    From: http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
    :param ground_truth:
    :param prediction:
    :param paired_gt_instances:
    :param paired_pred_instances:
    :return:
    """
    f1s = []
    for idx, (true_id, pred_id) in enumerate(zip(paired_gt_instances, paired_pred_instances)):
        true = np.zeros(ground_truth.shape)
        true[ground_truth == true_id] = 1
        pred = np.zeros(ground_truth.shape)
        pred[prediction == pred_id] = 1

        true_contours = get_contour(true)
        pred_contours = get_contour(pred)

        precision = calc_precision_recall(true_contours,
                                          pred_contours,
                                          threshold)

        recall = calc_precision_recall(pred_contours,
                                       true_contours,
                                       threshold)

        try:
            f1 = 2 * recall * precision / (recall + precision)
        except ZeroDivisionError as e:
            f1 = 0
        f1s.append(f1)

    return np.mean(f1s)

@typechecked
def _single_channel_munkers_pq(ground_truth: np.ndarray,
                               prediction: np.ndarray,
                               include_bf: bool = True,
                               threshold: float = 1.3) -> Union[float, Tuple[float, Union[float, None]]]:
    """
    Computes panoptic quality metric based on Munkers algorithm matching
    :param ground_truth:
    :param prediction:
    :return:
    """
    ground_truth_unique_ids = np.unique(ground_truth)[1:].astype(np.int32).tolist()
    prediction_unique_ids = np.unique(prediction)[1:].astype(np.int32).tolist()

    pairwise_iou = np.zeros((len(ground_truth_unique_ids), len(prediction_unique_ids)))

    for idx_i, i in enumerate(ground_truth_unique_ids):
        temp_image = np.array(ground_truth)
        temp_image = temp_image == i
        matched_image = temp_image * prediction

        for j in np.unique(matched_image):
            if j == 0:
                pass
            else:
                idx_j = prediction_unique_ids.index(j)
                pred_temp = prediction == j
                intersection = sum(sum(temp_image * pred_temp))  # Evaluates two one, since both are boolean (and)
                union = np.clip((temp_image + pred_temp), 0,
                                1).sum()  # Adding two booleans i.e. where both are True (or)
                iou = intersection / union
                pairwise_iou[idx_i, idx_j] = iou

    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)

    paired_iou = pairwise_iou[paired_true, paired_pred]

    tp = len(paired_true)
    fp = len(ground_truth_unique_ids) - len(paired_true)
    fn = len(prediction_unique_ids) - len(paired_pred)

    # get the F1-score i.e DQ
    try:
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        pq = dq * sq

    except ZeroDivisionError as e:
        pq = 0
    if include_bf:
        paired_true_ids = [ground_truth_unique_ids[i] for i in paired_true]
        paired_pred_ids = [prediction_unique_ids[i] for i in paired_pred]
        bf = compute_boundary_f_score(ground_truth, prediction, paired_true_ids, paired_pred_ids, threshold)
        bf = dq * bf
    else:
        bf = None

    return pq, bf

@typechecked
class PanopticQuality(object):

    def __init__(self, num_classes: int, include_bf: bool = True):

        self.num_classes = num_classes
        self.include_bf = include_bf

    def _single_channel(self, ground_truth: np.ndarray, prediction: np.ndarray) -> Union[float, Tuple[float, Union[float, None]]]:
        """

        :param ground_truth: array (width, height) where every instance is a unique digit
        :param prediction: array (width, height) where every instance is a unique digit
        :return:
        """
        scores = _single_channel_munkers_pq(ground_truth, prediction)

        return scores

    def _multi_channel(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """

        :return:
        """
        assert len(ground_truth.shape) == len(prediction.shape), "Prediction and ground truth are not the same shape"
        assert prediction.shape[0] == self.num_classes
        assert ground_truth.shape[0] == self.num_classes

        cls_scores = []

        for channel_index in range(self.num_classes):

            scores = self._single_channel(ground_truth[channel_index], prediction[channel_index])
            cls_scores.append(scores)

        raise NotImplementedError

    def compute(self, ground_truth: np.ndarray, prediction: np.ndarray) -> Union[float, Tuple[float, Union[float, None]]]:
        if self.num_classes == 1:
            return self._single_channel(ground_truth, prediction)
        if self.num_classes > 1:
            return self._multi_channel(ground_truth, prediction)
