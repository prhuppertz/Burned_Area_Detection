from segmentation.evaluation.pq_bf.panoptic_quality import PanopticQuality

class SegmentationMetric():

    def __init__(self):


        self.num_classes = 1
        self.metric = PanopticQuality(self.num_classes)
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
            score = self.metric.compute(gt, instance_mask)

            scores.append(score)

        return scores
