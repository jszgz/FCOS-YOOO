import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_yooo_postprocessor
from .loss import make_yooo_loss_evaluator


from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d

class YOOOHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(YOOOHead, self).__init__()
        num_classes = cfg.MODEL.YOOO.NUM_CLASSES # - 1  # if -1, its hard to calculate loss when there is no event in this frame, so NOEVENT is one class<=>one channel in the output, not like object event detection, positive sample is sparse in event
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES #[8, 16, 32, 64, 128]
        self.norm_reg_targets = cfg.MODEL.YOOO.NORM_REG_TARGETS      # True: normalizing the regression targets with FPN strides
        self.centerness_on_reg = cfg.MODEL.YOOO.CENTERNESS_ON_REG  #True
        self.use_dcn_in_tower = cfg.MODEL.YOOO.USE_DCN_IN_TOWER #False

        self.used_level = [0,1,2,3,4] # 0,1,2,3,4  p3,p4,p5,p6,p7
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.YOOO.NUM_CONVS): # range(4)
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.YOOO.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        #chwangteng: global average pooling to 1x1x256
        cls_tower.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        bbox_tower.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))


        self.add_module('cls_tower_event', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower_event', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1,
            padding=0
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 2, kernel_size=1, stride=1,
            padding=0
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=1, stride=1,
            padding=0
        )
        # initialization
        for modules in [self.cls_tower_event, self.bbox_tower_event,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        #scale exp()
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.used_level))])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []

        scale_index = -1
        for l, feature in enumerate(x):
            if l in self.used_level:
                scale_index = scale_index + 1
                cls_tower_event = self.cls_tower_event(feature)
                bbox_tower_event = self.bbox_tower_event(feature)

                logits.append(self.cls_logits(cls_tower_event))
                if self.centerness_on_reg:
                    centerness.append(self.centerness(bbox_tower_event))
                else:
                    centerness.append(self.centerness(cls_tower_event))

                bbox_pred = self.scales[scale_index](self.bbox_pred(bbox_tower_event))


                # if self.norm_reg_targets:
                #     bbox_pred = F.relu(bbox_pred)
                #     if self.training:
                #         bbox_reg.append(bbox_pred)
                #     else:
                #         bbox_reg.append(bbox_pred)
                # else:
                bbox_reg.append(torch.exp(bbox_pred))

        return logits, bbox_reg, centerness


class YOOOModule(torch.nn.Module):
    """
    Module for YOOO computation. Takes feature maps from the backbone and
    YOOO outputs and losses. Only Test on FPN now.
    """
    def __init__(self, cfg, in_channels):
        super(YOOOModule, self).__init__()

        head = YOOOHead(cfg, in_channels)

        box_selector_test = make_yooo_postprocessor(cfg)
        loss_evaluator = make_yooo_loss_evaluator(cfg)

        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES     #chwangteng [8, 16, 32, 64, 128]

        #TODO: timestamp sacle
        #Not tested, I want to scale the time if the input rate was not equal to the train rate, just intuition!just intuition!just intuition!
        self.train_fps = cfg.MODEL.YOOO.TRAIN_SampleRate
        self.test_fps = cfg.MODEL.YOOO.TEST_SampleRate

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image(stored in extra field for compatibility) (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)

        if self.training:
            return self._forward_train(
                box_cls, box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                 box_cls, box_regression,
                centerness, self.train_fps, self.test_fps
            )

    def _forward_train(self, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
             box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls_event": loss_box_cls,
            "loss_reg_event": loss_box_reg,
            "loss_centerness_event": loss_centerness
        }
        return None, losses

    def _forward_test(self, box_cls, box_regression, centerness, train_fps, test_fps):
        boxes = self.box_selector_test(
             box_cls, box_regression,
            centerness, train_fps, test_fps
        )
        return boxes, {} # the box is not eauql to box in FCOS(which is instance of BoxList Box), but just EventStartTime and EventEndTime



def build_yooo(cfg, in_channels):
    return YOOOModule(cfg, in_channels)