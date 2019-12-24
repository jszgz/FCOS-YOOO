
"""
This file contains specific functions for computing losses of YOOO
file
"""







import torch
from torch import nn
import os
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers import TIOULoss


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

class YOOOLossComputation(object):
    """
    This class computes the YOOO losses.
    """
    def __init__(self, cfg):
        self.cls_loss_func = nn.CrossEntropyLoss(reduction="sum")
        #     SigmoidFocalLoss(
        #     cfg.MODEL.FCOS.LOSS_GAMMA,
        #     cfg.MODEL.FCOS.LOSS_ALPHA
        # )

        self.iou_loss_type = cfg.MODEL.YOOO.IOU_LOSS_TYPE
        #self.norm_reg_targets = cfg.MODEL.YOOO.NORM_REG_TARGETS


        self.box_reg_loss_func = TIOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")


    def __call__(self, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList]) which is None in YOOO branch
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        num_classes = box_cls[0].size(1)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []


        labels_flatten = []
        labels_fla = self.compute_labels_flatten(targets)
        for l in range(len(box_cls)):
            labels_flatten.append(labels_fla)

        labels_flatten = torch.cat(labels_flatten, dim=0)
        #only select real event
        pos_inds = torch.nonzero( labels_flatten > 0).squeeze(1)


        box_cls_targets = []
        box_cls_targe = self.compute_box_cls_targets(targets, num_classes)



        reg_targets_flatten =[]
        centerness_targets = []

        if len(pos_inds>0):
            reg_targets_flat = self.compute_reg_targets(targets)
            centerness_targe = self.compute_centerness_targets(targets)


        for l in range(len(box_cls)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 2))
            centerness_flatten.append(centerness[l].reshape(-1))

            box_cls_targets.append(box_cls_targe)
            if len(pos_inds > 0):
                reg_targets_flatten.append(reg_targets_flat)
                centerness_targets.append(centerness_targe)



        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_cls_targets = torch.cat(box_cls_targets, dim=0)


        cls_loss = self.cls_loss_func(box_cls_flatten, box_cls_targets)
        cls_loss = cls_loss #/ num_classes # Because I use BCEloss, more classes, larger loss.

        reg_loss = torch.tensor(0.0)
        centerness_loss = torch.tensor(0.0)

        if len(pos_inds > 0):
            box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
            reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

            centerness_flatten = torch.cat(centerness_flatten, dim=0)
            centerness_targets = torch.cat(centerness_targets, dim=0)

            box_regression_flatten = box_regression_flatten[pos_inds]
            centerness_flatten = centerness_flatten[pos_inds]

            reg_loss = self.box_reg_loss_func(box_regression_flatten, reg_targets_flatten)
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets)


        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        return cls_loss  / num_pos_avg_per_gpu,\
               reg_loss / num_pos_avg_per_gpu, \
               centerness_loss / num_pos_avg_per_gpu

    def compute_box_cls_targets(self, targets, num_classes):
        # BCE onehot
        # xxxxxx = torch.tensor([[targets[i].get_field("EventLabel").item()] for i in range(len(targets))])
        #
        # m_zeros = torch.zeros(len(targets), num_classes)
        # one_hot = m_zeros.scatter_(1, xxxxxx, 1)  # (dim,index,value)
        # #return torch.tensor(one_hot, device= targets[0].get_field("EventLabel").device)
        # return one_hot.to(targets[0].get_field("EventLabel").device)
        xxxxxx = torch.tensor([targets[i].get_field("EventLabel").item() for i in range(len(targets))])
        return xxxxxx.to(targets[0].get_field("EventLabel").device)

    def compute_reg_targets(self, targets):
        reg_targets_flatten = []
        for i in range(len(targets)): #batch_size
            if targets[i].get_field("EventLabel").item() > 0:
                target = targets[i]
                reg_targets_flatten.append(
                    torch.FloatTensor([
                        [target.get_field("EventStartTimeOffsetInSecond"),
                        target.get_field("EventEndTimeOffsetInSecond")]]))
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        reg_targets_flatten = torch.tensor(reg_targets_flatten, device=targets[0].get_field("EventLabel").device)
        return reg_targets_flatten

    def compute_centerness_targets(self, targets):
        centerness_targets_flatten = []
        for i in range(len(targets)): #batch_size
            if targets[i].get_field("EventLabel").item() > 0:
                temp = torch.tensor([targets[i].get_field("EventStartTimeOffsetInSecond"),
                                     targets[i].get_field("EventEndTimeOffsetInSecond")])
                centerness_targets_flatten.append(torch.sqrt(temp.min()/temp.max()))

        centerness_targets_flatten = torch.tensor(centerness_targets_flatten, device=targets[0].get_field("EventLabel").device)
        return centerness_targets_flatten

    def compute_labels_flatten(self, targets):
        labels_flatten = []
        for i in range(len(targets)):
            target = targets[i]
            labels_flatten.append(target.get_field("EventLabel"))

        labels_flatten = torch.tensor(labels_flatten, device=labels_flatten[0].device)
        return labels_flatten

def make_yooo_loss_evaluator(cfg):
    loss_evaluator = YOOOLossComputation(cfg)
    return loss_evaluator