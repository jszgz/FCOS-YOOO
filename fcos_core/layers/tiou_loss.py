import torch
from torch import nn


class TIOULoss(nn.Module):
    def __init__(self, loss_type="tiou"):
        super(TIOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pass
        # TODO: TIOULOSS

        #one dimension(temporial) loss like IOU loss in an image in two dimensions.
        #intersection of union.
        #But what puzzles me is that the pred is a large number as in target(time in second or million second)?

        pred_left = pred[:, 0]
        pred_right = pred[:, 1]

        target_left = target[:, 0]
        target_right = target[:, 1]

        area_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        area_union = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        losses = -torch.log(ious)
        return losses.sum()