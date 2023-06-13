import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.layers.diff_ras.polygon import SoftPolygon
from modeling.utils import get_union_box, rasterize_instances, POLY_LOSS_REGISTRY, inverse_sigmoid
from detectron2.layers import ROIAlign
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from modeling.diff_ras import ClippingStrategy
from typing import List
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask


class MaskCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mask_loss_type = cfg.MODEL.POLYGON_HEAD.POLY_LOSS.TYPE # support "ce", "dice"
        self.losses = [mask_loss_type, ]
        # whether to invoke our own rasterizer in "hard" mode.
        self.use_rasterized_gt = cfg.MODEL.DIFFRAS.USE_RASTERIZED_GT # False
        # self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.clip_to_proposal = not cfg.MODEL.ROI_HEADS.PROPOSAL_ONLY_GT
        self.predict_in_box_space = cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX
        
        if self.clip_to_proposal or not self.use_rasterized_gt:
            self.clipper = ClippingStrategy(cfg)
            self.gt_rasterizer = None
        else:
            self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")
        self.offset = 0.5
        self.loss_mask_weight = cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS
        
        self.mask_stride_sup = cfg.MODEL.POLYGON_HEAD.MASK_STRIDE_SUP # 到 P2 层去监督 ？
        self.mask_stride = cfg.MODEL.POLYGON_HEAD.MASK_STRIDE # P3 
        
        self.debug = False

    def loss_dice(self, pred_masks, target_masks):
        losses = {"loss_dice": dice_loss(pred_masks, target_masks)}
        del target_masks
        return losses

    def loss_ce(self, pred_masks, target_masks):
        """as mask rcnn"""
        return {"loss_ce": sigmoid_ce_loss(pred_masks, target_masks)}

    def get_loss(self, loss, pred_masks, targets):
        loss_map = {"dice": self.loss_dice, "ce": self.loss_ce}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](pred_masks, targets)
    
    @torch.no_grad()
    def _create_targets(self, instances, img_wh, lid=0):
        if self.predict_in_box_space:
            if self.clip_to_proposal or not self.use_rasterized_gt: # in coco, this is true
                clip_boxes = torch.cat([inst.proposal_boxes.tensor for inst in instances])
                masks = self.clipper(instances, clip_boxes=clip_boxes, lid=lid) # bitmask
            else:
                masks = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at) 
        else:
            masks = self.get_bitmasks(instances, img_h=img_wh[1], img_w=img_wh[0])
        return {"mask": masks.unsqueeze(1).float()}

    @torch.no_grad()
    def get_bitmasks(self, instances, img_h, img_w):
        gt_masks = []
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            # start = int(self.mask_stride // 2)
            start = int(self.mask_stride_sup // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                # per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, img_h, img_w)  
                    # TODO: 这里可以直接转换低分辨率的mask? 这里好像没有对齐？应该先到原图大小，然后再padding到原图大小？
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    bitmask = bitmask[start::self.mask_stride_sup, start::self.mask_stride_sup]
                    assert bitmask.size(0) * self.mask_stride_sup == img_h
                    assert bitmask.size(1) * self.mask_stride_sup == img_w
                    per_im_bitmasks.append(bitmask)

                gt_masks.append(torch.stack(per_im_bitmasks, dim=0))
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, img_w - w, 0, img_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_stride_sup, start::self.mask_stride_sup]
                
                gt_masks.append(bitmasks)
        return torch.cat(gt_masks, dim=0) # (N, H, W)

    def forward(self, images, pred_masks, instances):
        if not isinstance(pred_masks, List):
            pred_masks = [pred_masks]
        # targets
        img_wh = tuple(images['images_norm'].shape[-2:][::-1]) # padded wh
        del images
        self.device = pred_masks[0].device
        targets = self._create_targets(instances, img_wh, lid=0)
        
        upsampling_rate = self.mask_stride // self.mask_stride_sup
        assert upsampling_rate == 1, "not support"
                
        _losses = {}
        for lid, pred_masks_per_dec in enumerate(pred_masks): # 遍历4个decoder的pred_mask 
            # clamp pred masks ?
            pred_masks_per_dec = torch.clamp(pred_masks_per_dec, 0.00001, 0.99999)
            for loss in self.losses:
                l_dict = self.get_loss(loss, pred_masks_per_dec, targets["mask"])
                l_dict = {k + f"_{lid}": v for k, v in l_dict.items()}
                _losses.update(l_dict)
        # weights ? 
        losses = {"loss_mask": self.loss_mask_weight[0] * sum(_losses.values()) / len(_losses)}
        return losses


def dice_loss(input: torch.Tensor, target: torch.Tensor):
    smooth = 1.
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
    return loss


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def build_poly_losses(cfg, input_shape):
    """
    Build polygon losses `cfg.MODEL.POLYGON_HEAD.POLY_LOSS.NAMES`.
    """

    losses = []
    for name in cfg.MODEL.POLYGON_HEAD.POLY_LOSS.NAMES:
        losses.append(POLY_LOSS_REGISTRY.get(name)(cfg, input_shape))

    return losses