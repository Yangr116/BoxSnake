import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from detectron2.modeling.roi_heads.mask_head import (MaskRCNNConvUpsampleHead, mask_rcnn_inference, 
                                                    mask_rcnn_loss, ROI_MASK_HEAD_REGISTRY)
from detectron2.layers import ShapeSpec, cat
from detectron2.config import configurable
from detectron2.structures import Instances
from modeling.box_supervisor import BoxSupervisor


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHeadV2(MaskRCNNConvUpsampleHead):

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="",
                enable_box_sup=False, box_supervisor=None, **kwargs) -> None:
        super(MaskRCNNConvUpsampleHeadV2, self).__init__(input_shape, 
            num_classes=num_classes, conv_dims=conv_dims, conv_norm=conv_norm, **kwargs)
        # box_sup
        self.enable_box_sup = enable_box_sup
        self.box_supervisor = box_supervisor

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        # box sup
        ret_box_sup = {
            "enable_box_sup": cfg.MODEL.BOX_SUP.ENABLE,
            "box_supervisor": BoxSupervisor(cfg)
            }
        ret.update(ret_box_sup)
        return ret

    def forward(self, images, x, instances: List[Instances]):
        """
        Args:
            images: RGB/LAB images
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        pred_mask = self.layers(x)
        if self.training:
            if self.enable_box_sup:
                feat_in_boxes = x.detach()
                return self.box_supervisor(images, {'mask': pred_mask}, instances)
            return {"loss_mask": mask_rcnn_loss(pred_mask, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(pred_mask, instances)
            return instances