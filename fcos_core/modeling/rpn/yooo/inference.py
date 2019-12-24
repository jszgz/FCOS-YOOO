
import torch


class YOOOPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes (two tensor).
    This is only used in the testing.
    """
    def __init__(
        self,
        num_classes,
    ):
        """
        Arguments:
            num_classes (int)
        """
        super(YOOOPostProcessor, self).__init__()
        self.num_classes = num_classes



    def forward_for_single_feature_map(
            self, box_cls,
            box_regression, centerness):
        """
        Arguments:
            box_cls: tensor of size N, C, H, W
            box_regression: tensor of size N, 2, H, W
            centerness: tensor of size N, 1, H, W
        """
        N, C, H, W = box_cls.shape

        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 2, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 2)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]  # torch.Size([1,80]) 1 will not be 1 if I modify the net output

            temp = torch.mean(per_box_cls, dim=0)
            score, class_index = torch.max(temp, 0)

            offset_left = box_regression[i][0][0]
            offset_right = box_regression[i][0][1]

            results.append({"score":torch.sqrt(score),
                            "index":class_index,
                            "EventStartTimeOffset":offset_left,
                            "EventEndTimeOffset":offset_right})

        return results

    def forward(self, box_cls, box_regression, centerness, train_fps, test_fps):
        """
        Arguments:

            box_cls: list[tensor]
            box_regression: list[tensor]
            centerness: list[tensor]

        Returns:
            [EventStartTime, EventEndTime]
        """
        sampled_boxes = [] #len(sampled_boxes) == feature_level_size
        for _, (cls, regress, center) in enumerate(zip(box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    cls, regress, center
                )
            )

        boxlists = list(zip(*sampled_boxes))
        # chwangteng: Explanation
        # sampled_boxes = [[1, 2, 3], [4, 5, 6]]
        # boxlists = list(zip(*sampled_boxes))
        # print(boxlists)  # [(1, 4), (2, 5), (3, 6)]

        #TODO:Maybe wrongMaybe wrongMaybe wrongMaybe wrongMaybe wrong
        new_boxlist = []
        for boxlist in boxlists: #multi level in a batch

            maxscore = boxlist[0]["score"]
            maxscoreindex = 0
            for index, box in enumerate(boxlist):#levels in multi level feature map
                if box["score"]>maxscore:
                    maxscoreindex = index
            new_boxlist.append(boxlist[maxscoreindex])

        return new_boxlist # list[diction,diction,diction]


def make_yooo_postprocessor(config):
    box_selector = YOOOPostProcessor(
        num_classes=config.MODEL.YOOO.NUM_CLASSES,
    )
    return box_selector