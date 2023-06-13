# Copyright (c) Facebook, Inc. and its affiliates.
# yr revised
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler

from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import select_foreground_proposals

from detectron2.structures.boxes import matched_pairwise_iou

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsV2(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *, # * 表示 后面必须以关键字传参数
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        enable_roi_jitter: bool = False,
        roi_jitter = None, 
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, 
                         box_predictor=box_predictor, mask_in_features=mask_in_features, mask_pooler=mask_pooler, 
                         mask_head=mask_head, keypoint_in_features=keypoint_in_features,
                         keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, **kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        # roi jitter
        self.enable_roi_jitter = enable_roi_jitter
        self.roi_jitter = roi_jitter

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.

        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        # roi jitter params
        ret_roi_jitter = {
            'enable_roi_jitter': cfg.MODEL.ROI_MASK_HEAD.ENABLE_ROI_JITTER,
            'roi_jitter': RoIJitter(
                num_jitter=cfg.MODEL.ROI_MASK_HEAD.NUM_ROI_JITTER,
                noise_scale=cfg.MODEL.ROI_MASK_HEAD.NOISE_SCALE,
                iou_thr=cfg.MODEL.ROI_MASK_HEAD.ROI_JITTER_IOU_THR,
                ), # add roi jitter in self._forward_mask_head
        }
        ret.update(ret_roi_jitter)
        return ret

    def forward(
        self,
        images: torch.Tensor,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        # del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(images, features, proposals, targets=targets))
            del targets
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(None, features, instances, None)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_mask(self, images, features: Dict[str, torch.Tensor], instances: List[Instances], targets):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            # roi jitter, how to adding mask target here ? 这里放的是 pos mask target
            # proposal_append_gt 会将 gt 加入到 instances 中，但是不知道会有几个 gt 
            if self.enable_roi_jitter:
                instances = self.roi_jitter(instances, targets) # TODO: adding roi jitter

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(images, features, instances)


class RoIJitter():
    def __init__(self, num_jitter=3, noise_scale=0.5, iou_thr=0.7):
        self.num_jitter = num_jitter # 每个 box 抖动几次
        self.noise_scale = noise_scale # 越小 抖动的越小
        self.iou_thr = iou_thr

    def __call__(self, instances, targets):
        """
        instances: List(Instances(), ...) 里面只包含 pos box
        """
        # 这里会修改 instance 
        # take out gt boxes 
        for idx, (instance, target) in enumerate(zip(instances, targets)):
            roi_instance = Instances(instance.image_size)
            num_instances = len(target)
            gt_boxes = (target.gt_boxes.tensor).repeat(self.num_jitter, 1) # (x1, y1, x2, y2) format
            _gt_boxes = Boxes(gt_boxes)
            rand_sign = torch.randint_like(gt_boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 # +1 or -1
            rand_part = torch.rand_like(gt_boxes) * rand_sign * self.noise_scale # this noise belong to (-noise_scale, noise_scale)
            wh = gt_boxes[:, 2:] - gt_boxes[:, :2] # shape=(N, 2)
            diff = torch.cat([wh, wh], dim=-1) / 2 # shape=(N, 4)
            roi_jitter = gt_boxes + torch.mul(diff, rand_part).type_as(gt_boxes) # 在 (x1, y1, x2, y2) 上的偏移 等价于 中心点偏移和尺度缩放
            roi_jitter = Boxes(roi_jitter)
            roi_jitter.clip(instance.image_size) # 限制box在图像范围内部
            # iou
            iou_flag = matched_pairwise_iou(roi_jitter, _gt_boxes) > self.iou_thr # return (N)
            num_reserved_inst = len(torch.nonzero(iou_flag))
            # prepare instances
            roi_instance.proposal_boxes = roi_jitter[iou_flag]
            roi_instance.objectness_logits = torch.ones(num_reserved_inst).type_as(instance.objectness_logits)
            roi_instance.gt_classes = torch.ones(num_reserved_inst).type_as(instance.gt_classes)
            roi_instance.gt_boxes = _gt_boxes[iou_flag]
            instances[idx] = instance.cat([instance, roi_instance])
        return instances