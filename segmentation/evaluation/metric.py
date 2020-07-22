from segmentation.evaluation.metrics.dice_and_iou import dice_and_iou

class SegmentationMetric():

    def __init__(self):


        self.num_classes = 1
        #self.metric = PanopticQuality(self.num_classes)
        self.scores = []


    def batch_compute(self, prediction, ground_truth):
        """

        :param ground_truth:
        :param prediction:
        :return:
        """
        scores = []
        for idx in range(len(ground_truth)):
            gt = ground_truth[idx]
            instance_mask = prediction[idx]
            score = dice_and_iou(instance_mask, gt)

            scores.append(score)

        return scores
