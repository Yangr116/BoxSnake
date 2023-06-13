# yr 2023.05.29
# most code copy from adelaidet CondInst

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances

import inspect
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.modeling.roi_heads import build_mask_head, ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures.boxes import Boxes

logger = logging.getLogger(__name__)
   

@ROI_HEADS_REGISTRY.register()
class PseudoRoIHead(nn.Module):
    @configurable
    def __init__(self,
                *, # * 表示 后面必须以关键字传参数
                mask_in_features: Optional[List[str]] = None,
                mask_pooler: Optional[ROIPooler] = None,
                mask_head: Optional[nn.Module] = None,
                max_proposals: int = 256,
                topk_proposals_per_im: int = 100,
                **kwargs,):
        super().__init__()
        
        self.max_proposals = max_proposals
        self.topk_proposals_per_im = topk_proposals_per_im
        
        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
            

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        # ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.

        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
            
        # FCOS param
        max_proposals = cfg.MODEL.ROI_HEADS_FCOS.MAX_PROPOSALS
        topk_proposals_per_im = cfg.MODEL.ROI_HEADS_FCOS.TOPK_PROPOSALS_PER_IM
        ret.update({
            'max_proposals': max_proposals,
            'topk_proposals_per_im': topk_proposals_per_im,
        })
        
        return ret
    
    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret
    
    def forward(self, images, features, proposals, gt_instances):
        if not self.training:
            return self._forward_mask_heads_test(features, proposals)
        return self._forward_polygon_head_train(images, features, proposals, gt_instances)
    
    def _forward_polygon_head_train(self, images, features, proposals, gt_instances):
        
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]
        
        # sample proposal
        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=features.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        # pred_instances.mask_head_params = pred_instances.top_feats
        # convert to per_image instances List[Instances]
        # Instances(proposal_boxes: Boxes, objectness_logits, gt_boxes: Boxes, gt_classes)
        _stride = 8 * (2 ** (pred_instances.fpn_levels.unsqueeze(-1))) # 这里的 fpn level 是从 0 开始的，所以不能 -1 
        _reg_targets = pred_instances.reg_targets * _stride # shape: (N, 4) * (N, 1) -> (N, 4)
        _reg_pred = pred_instances.reg_pred * _stride
        _loc = pred_instances.locations
        gt_boxes = torch.cat([_loc - _reg_targets[:, :2], _loc + _reg_targets[:, 2:]], dim=-1) # (N, 4)
        pred_boxes = torch.cat([_loc - _reg_pred[:, :2], _loc + _reg_pred[:, 2:]], dim=-1)
        object_ness_logits = torch.sigmoid(pred_instances.logits_pred).max(dim=1)[0] * torch.sigmoid(pred_instances.ctrness_pred)
        
        # 需要查看后处理部分
        gt_classes = pred_instances.labels
        
        img_inds = pred_instances.im_inds
        pred_instances_list = []
        img_hw = tuple(images['images'].shape[-2:])
        for i in range(len(gt_instances)):
            inst_per_img = Instances(img_hw)
            inst_per_img.proposal_boxes = Boxes(pred_boxes[img_inds==i])
            inst_per_img.objectness_logits = object_ness_logits[img_inds==i]
            inst_per_img.gt_classes = gt_classes[img_inds==i]
            inst_per_img.gt_boxes = Boxes(gt_boxes[img_inds==i])
            pred_instances_list.append(inst_per_img)
        
        # TODO: select the foreground samples? yes
        features = {f: features[f] for f in self.mask_in_features}
        # polygon head
        loss_polygon = self.mask_head(images, features, pred_instances_list)

        return pred_instances, loss_polygon
    
    def _forward_mask_heads_test(self, features, proposals):
        """
        features: 
        proposals: List(Instance), Instance filed {pred_boxes: Boxes(), scores:, pred_classes}
        """
        # prepare the inputs for mask heads
        # for im_id, per_im in enumerate(proposals):
            # per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = proposals # 可以直接输入 
        
        # pred_instances = Instances.cat(proposals) # 这里应该不用 cat
        # pred_instances.mask_head_params = pred_instances.top_feat
        
        # dosen't need to select the samples during inference
        features = {f: features[f] for f in self.mask_in_features}
        pred_instances_w_polys = self.mask_head(None, features, pred_instances)
        return pred_instances_w_polys, {}