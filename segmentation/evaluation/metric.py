from segmentation.evaluation.pq_bf.dice import dice_with_tensors

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
            score = dice_with_tensors(instance_mask, gt)

            scores.append(score)

        return scores
