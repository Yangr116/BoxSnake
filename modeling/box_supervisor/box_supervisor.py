from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec, cat, ROIAlign
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from .target import create_box_targets, create_box_targets_p3, create_box_targets_crop
from .utils import get_images_color_similarity, unfold_wo_center
from .ciou import ciou_loss


class BoxSupervisor():
    def __init__(self, cfg):
        # box sup config
        if "Polygon" in cfg.MODEL.ROI_MASK_HEAD.NAME:
            mask_head_weights = cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS[0]
            is_logits = False
        else:
            mask_head_weights = cfg.MODEL.ROI_MASK_HEAD.MASK_HEAD_WEIGHTS 
            is_logits = True
        # box sup loss name
        boxsnake_loss = []
        if cfg.MODEL.BOX_SUP.LOSS_PROJ:
            boxsnake_loss.append('projection')
        if cfg.MODEL.BOX_SUP.LOSS_AVG_PROJ:
            boxsnake_loss.append('avg_projection')
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ:
            boxsnake_loss.append('points_projection')
        if cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE:
            boxsnake_loss.append("local_pairwise")
        if cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE:
            boxsnake_loss.append("global_pairwise")
        print(f"box_sup_loss: {boxsnake_loss}")

        # box sup loss weights
        boxsnake_loss_weights = {
            "loss_proj_dice": cfg.MODEL.BOX_SUP.LOSS_PROJ_DICE_WEIGHT,
            "loss_proj_ce": cfg.MODEL.BOX_SUP.LOSS_PROJ_CE_WEIGHT,
            "loss_points_proj": cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ_WEIGHT,
            "loss_avg_proj_dice": cfg.MODEL.BOX_SUP.LOSS_AVG_PROJ_DICE_WEIGHT,
            "loss_avg_proj_ce": cfg.MODEL.BOX_SUP.LOSS_AVG_PROJ_CE_WEIGHT,
            "loss_local_pairwise": cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_WEIGHT,
            "loss_global_pairwise": cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE_WEIGHT,
            
            }
        boxsnake_loss_weights = {k: v * mask_head_weights for k, v in boxsnake_loss_weights.items()}
        print(f"box_sup_loss_weights: {boxsnake_loss_weights}")

        # local pairwise loss param
        self.enable_local_pairwise_loss = cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE
        self.pairwise_warmup_iters = cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_WARMUP_ITER
        self.pairwise_cold_iters = cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_COLD_ITER
        self.local_pairwise_kernel_size = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_KERNEL_SIZE
        self.local_pairwise_dilation = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_DILATION
        self.local_pairwise_color_threshold = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_THR # for boxinst format pairwise
        self.local_pairwise_sigma = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_SIGMA
        
        self.crop_predicts = cfg.MODEL.BOX_SUP.CROP_PREDICTS
        self.crop_size = cfg.MODEL.BOX_SUP.CROP_SIZE
        self.mask_padding_size = cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE

        self.box_sup_loss = BoxSupLoss(
            boxsnake_loss, boxsnake_loss_weights, self.local_pairwise_kernel_size, 
            self.local_pairwise_dilation,
            is_logits=is_logits,
            loss_projection_type=cfg.MODEL.BOX_SUP.LOSS_PROJ_TYPE,
            local_pairwise_color_threshold=self.local_pairwise_color_threshold,
            loss_local_pairwise_type=cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_TYPE,
            crop_predicts=self.crop_predicts
            )
        
        self.mask_stride = cfg.MODEL.POLYGON_HEAD.MASK_STRIDE
        self.predict_in_box_space = cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX


    def __call__(self, images, preds, instances):
        """
        images: dict()
        preds: dict()
        instances: list
        Return: loss
        """
        assert 'mask' in preds, "There should be mask in preds"
        pred_masks = preds["mask"]
        if not isinstance(pred_masks, list):
            pred_masks = [pred_masks]
        pred_polygons = preds.get("polygon", [torch.empty_like(pred_masks[0]) for i in range(len(pred_masks))])

        total_num_masks = pred_masks[0].size(0)
        # create targets
        if self.predict_in_box_space: # boundaryformer method, this method does not exploit the background label
            mask_side_len = pred_masks[0].size(2)
            clip_boxes = torch.cat([inst.proposal_boxes.tensor for inst in instances])
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets(
                images["images"], images["images_norm"], instances, clip_boxes=clip_boxes,
                mask_size=mask_side_len, kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                sigma=self.local_pairwise_sigma,)
            tgt_masks = tgt_masks.unsqueeze(1).float()
        elif self.crop_predicts: # boxsnake take this method
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets_crop(
                images["images"], images["images_norm"], instances, self.mask_stride,
                kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                crop_size=self.crop_size, # add to config
                mask_padding_size=self.mask_padding_size,
                sigma=self.local_pairwise_sigma,
            )
            tgt_masks = tgt_masks.unsqueeze(1).float()
        else:  # rasterizerize the polygon to p3/p2 
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets_p3(
                images["images"], images["images_norm"], instances, self.mask_stride,
                kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                sigma=self.local_pairwise_sigma,
            )
            tgt_masks = tgt_masks.unsqueeze(1).float()
        
        del images
        
        targets = {"mask": tgt_masks, "imgs_sim": tgt_imgs_sim, "imgs": tgt_imgs, "gt_boxes": gt_boxes}
        
        # here, tgt_masks are the binary masks from bounding boxes

        losses_list = []
        
        for lid, (pred_masks_per_dec, pred_polys_per_dec) in enumerate(zip(pred_masks, pred_polygons)):
            #if it is not the cls agnostic mask
            if pred_masks_per_dec.size(1) != 1:
                indices = torch.arange(total_num_masks)
                gt_classes = cat([inst.gt_classes.to(dtype=torch.int64) for inst in instances], dim=0)
                # shape=(N, num_class, mask_size, mask_size) -> (N, 1, mask_size, mask_size)
                pred_masks_per_dec = pred_masks_per_dec[indices, gt_classes]
                pred_masks_per_dec = pred_masks_per_dec.unsqueeze(1)
            # cal loss
            l_dict = self.box_sup_loss(pred_masks_per_dec, pred_polys_per_dec, targets)
            losses_list.append(l_dict)
        
        losses = {} # to average the same loss from different decoder
        for k in list(losses_list[0].keys()):
            losses[k] = torch.stack([d[k] for d in losses_list]).mean()
        
        # pairwise warmup
        if self.enable_local_pairwise_loss:
            storage = get_event_storage()
            if storage.iter - self.pairwise_cold_iters > 0:
                pairwise_warmup_factor = min((storage.iter - self.pairwise_cold_iters) / float(self.pairwise_warmup_iters), 1.0)
            else:
                pairwise_warmup_factor = 0. # cold start
            losses["loss_local_pairwise"] *= pairwise_warmup_factor

        return losses


class BoxSupLoss():
    def __init__(self, losses, loss_weights, local_pairwise_kernel_size=3, local_pairwise_dilation=1, 
                local_pairwise_color_threshold=0.1, loss_local_pairwise_type="v1", is_logits=False, 
                loss_projection_type=["dice"], crop_predicts=False):
        super(BoxSupLoss, self).__init__()
        self.losses = losses
        self.loss_weights = loss_weights
        
        # projection
        self.loss_projection_type = loss_projection_type
        print(f"loss_projection_type: {loss_projection_type}")
        loss_map = {
            'projection': self.loss_projection,
            'avg_projection': self.loss_avg_projection,
            'points_projection': self.loss_points_proj, # this is the CIoU loss
        }
        
        # pairwise_loss
        loss_local_pairwise = {
            "v1": self.loss_local_pairwise,
            "v2": self.loss_local_pairwise_box_inst,
        }
        loss_map["local_pairwise"] = loss_local_pairwise.get(loss_local_pairwise_type, self.loss_local_pairwise)
        print(f"loss_local_pairwise_type: {loss_local_pairwise_type}")
        self.local_pairwise_kernel_size = local_pairwise_kernel_size
        self.local_pairwise_dilation = local_pairwise_dilation
        self.local_pairwise_color_threshold = local_pairwise_color_threshold # only for boxinst format pairiwse loss
        # global pairwise loss, also a levelset-like loss
        loss_map["global_pairwise"] = self.loss_global_pairwise

        self.loss_map = {k: loss_map[k] for k in self.losses}

        self.crop_predicts = crop_predicts
        self.is_logits = is_logits

    def loss_points_proj(self, pred_masks, targets, num_masks, **kwargs):
        pred_polys = kwargs['pred_polys']
        gt_boxes = targets['gt_boxes']
        # points proj
        proj_boxes = torch.cat([pred_polys.min(dim=1)[0], pred_polys.max(dim=1)[0]], dim=-1) # shape=(N, 4), x1y1x2y2 format
        # ciou loss
        loss = ciou_loss(proj_boxes, gt_boxes)
        loss = loss.mean() * self.loss_weights.get('loss_points_proj', 0.)
        return {'loss_points_proj': loss}

    def loss_projection(self, pred_masks, targets, num_masks, **kwargs):
        target_masks = targets["mask"]
        assert pred_masks.shape == target_masks.shape

        pred_proj_y = pred_masks.max(dim=-2, keepdim=True)[0].flatten(1)
        pred_proj_x = pred_masks.max(dim=-1, keepdim=True)[0].flatten(1)
        tgt_proj_y = target_masks.max(dim=-2, keepdim=True)[0].flatten(1)
        tgt_proj_x = target_masks.max(dim=-1, keepdim=True)[0].flatten(1)

        loss = {}
        if "dice" in self.loss_projection_type:
            mask_losses_dice_y = dice_loss_jit(pred_proj_y, tgt_proj_y, num_masks)
            mask_losses_dice_x = dice_loss_jit(pred_proj_x, tgt_proj_x, num_masks)
            loss.update({"loss_proj_dice": (mask_losses_dice_x + mask_losses_dice_y) * self.loss_weights.get('loss_proj_dice', 0.)})
        if "ce" in self.loss_projection_type:
            mask_losses_ce_y = sigmoid_ce_loss_jit(pred_proj_y, tgt_proj_y, num_masks)
            mask_losses_ce_x = sigmoid_ce_loss_jit(pred_proj_x, tgt_proj_x, num_masks)
            loss.update({"loss_proj_ce": (mask_losses_ce_x + mask_losses_ce_y) * self.loss_weights.get('loss_proj_ce', 0.)})
        del target_masks
        return loss

    def loss_avg_projection(self, pred_masks, targets, num_masks, **kwargs):
        target_masks = targets["mask"]
        assert pred_masks.shape == target_masks.shape
        # avg
        pred_proj_y = pred_masks.mean(dim=-2, keepdim=True)[0].flatten(1)
        pred_proj_x = pred_masks.mean(dim=-1, keepdim=True)[0].flatten(1)
        tgt_proj_y = target_masks.float().mean(dim=-2, keepdim=True)[0].flatten(1)
        tgt_proj_x = target_masks.float().mean(dim=-1, keepdim=True)[0].flatten(1)
        loss = {}
        if "dice" in self.loss_projection_type:
            mask_losses_dice_y = dice_loss_jit(pred_proj_y, tgt_proj_y, num_masks)
            mask_losses_dice_x = dice_loss_jit(pred_proj_x, tgt_proj_x, num_masks)
            loss.update({"loss_avg_proj_dice": (mask_losses_dice_x + mask_losses_dice_y) * self.loss_weights.get('loss_avg_proj_dice', 0.)})
        if "ce" in self.loss_projection_type:
            mask_losses_ce_y = sigmoid_ce_loss_jit(pred_proj_y, tgt_proj_y, num_masks)
            mask_losses_ce_x = sigmoid_ce_loss_jit(pred_proj_x, tgt_proj_x, num_masks)
            loss.update({"loss_avg_proj_ce": (mask_losses_ce_x + mask_losses_ce_y) * self.loss_weights.get('loss_avg_proj_ce', 0.)})
        return loss

    def loss_local_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        target_masks, imgs_sim = targets["mask"], targets["imgs_sim"]
        # fg_prob = torch.sigmoid(pred_masks) if self.is_logits else pred_masks
        fg_prob = pred_masks
        fg_prob_unfold = unfold_wo_center(
            fg_prob, kernel_size=self.local_pairwise_kernel_size,
            dilation=self.local_pairwise_dilation)
        pairwise_term = torch.abs(fg_prob[:, :, None] - fg_prob_unfold)[:, 0]
        weights = imgs_sim * target_masks.float() # limit to the box
        loss_local_pairwise = (weights * pairwise_term).sum() / weights.sum().clamp(min=1.0)
        # TODO: which one ?
        # loss_local_pairwise = (weights * pairwise_term).flatten(1).sum(-1) / weights.flatten(1).sum(-1).clamp(min=1.0)
        # loss_local_pairwise = loss_local_pairwise.sum() / num_masks

        loss = {"loss_local_pairwise": loss_local_pairwise * self.loss_weights.get("loss_local_pairwise", 0.)}
        # TODO: add different pairwise format
        del target_masks
        del imgs_sim
        return loss

    def loss_local_pairwise_box_inst(self, pred_masks, targets, num_masks, **kwargs):
        """
        L = y_e*log[P(y_e=1)] + (1-y_e)*log[P(y_e=0)]
        y_e = m_i * m_j + (1-m_i)*(1-m_j)
        The loss is minimized only if both points have the same score of 1 or the same score of 0
        """
        target_masks, imgs_sim = targets["mask"], targets["imgs_sim"]
        # fg_prob = torch.sigmoid(pred_masks) if self.is_logits else pred_masks
        pred_masks = pred_masks.clamp(min=1e-5, max=(1. - 1e-5)) # avoid nan in log()
        log_fg_prob = torch.log(pred_masks) # this is the sigmoid scores
        log_bg_prob = torch.log(1. - pred_masks)
        log_fg_prob_unfold = unfold_wo_center(
            log_fg_prob, kernel_size=self.local_pairwise_kernel_size,
            dilation=self.local_pairwise_dilation
        )
        log_bg_prob_unfold = unfold_wo_center(
            log_bg_prob, kernel_size=self.local_pairwise_kernel_size,
            dilation=self.local_pairwise_dilation
        )
        log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
        log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold
        
        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
        log_same_prob = torch.log(
            torch.exp(log_same_fg_prob - max_) +
            torch.exp(log_same_bg_prob - max_)
        ) + max_
        pairwise_term = -log_same_prob[:, 0]

        weights = (imgs_sim >= self.local_pairwise_color_threshold).float() * target_masks.float()
        loss_local_pairwise = (pairwise_term * weights).sum() / weights.sum().clamp(min=1.0)

        loss = {"loss_local_pairwise": loss_local_pairwise * self.loss_weights.get("loss_local_pairwise", 0.)}

        del target_masks
        del imgs_sim
        return loss
    
    def loss_global_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        """
        ref: https://www.math.ucla.edu/~lvese/PAPERS/JVCIR2000.pdf and boxlevelset method
        pred_masks: shpae=(N, 1, H, W)
        limit the mask into the box mask, imgs is cropped, it does't need mask gated
        """
        target_masks, imgs = targets["mask"], targets["imgs"] # shape=(N, 1, H, W), (N, 3, H, W)
        # prepare pred_masks
        pred_masks_back = 1.0 - pred_masks
        C_, H_, W_ = imgs.shape[1:]
        # imgs_wbox = imgs * target_masks # TODO, dose this matter in the cropped way?  
        level_set_energy = get_region_level_energy(imgs, pred_masks, C_) + \
                           get_region_level_energy(imgs, pred_masks_back, C_)

        pixel_num = float(H_ * W_)

        level_set_losses = torch.mean((level_set_energy) / pixel_num) # HW weights
        losses = {"loss_global_pairwise": level_set_losses * self.loss_weights.get('loss_global_pairwise', 0.)} # instances weights

        del target_masks
        del imgs
        return losses

    def get_loss(self, loss, pred_masks, target_masks, num_masks, **kwargs):
        assert loss in self.loss_map, f"do you really want to compute {loss} loss?"
        return self.loss_map[loss](pred_masks, target_masks, num_masks, **kwargs)

    def __call__(self, pred_masks, pred_polys, targets):
        """
        pred_polys: Tensor or None, Tensor shape=(N_inst, N_p, 2)
        """
        # NOTE: all nodes average or single node
        if self.is_logits: 
            pred_masks = pred_masks.sigmoid()
        num_masks = max(pred_masks.shape[0], 1.0)
        kwargs = {'pred_polys': pred_polys}
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, pred_masks, targets, num_masks, **kwargs))
        return losses


def dice_loss_(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid() # 这里错了
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1. - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(
    dice_loss_
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float
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
    loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def get_region_level_energy(imgs, masks, num_channel):
    # masks = masks.expand(-1, num_channel, -1, -1)  # shape=(N, C, H, W) 不用 expand, 自己扩展就行
    avg_sim = torch.sum(imgs * masks, dim=(2, 3), keepdim=True) / torch.sum(masks, dim=(2, 3), keepdim=True).clamp(min=1e-5)
    # shape=(N, C, 1, 1)
    region_level = torch.pow(imgs - avg_sim, 2) * masks  # shape=(N, C, H, W), 沿着channel相加，沿着 HW 求和（积分）
    return torch.sum(region_level, dim=(1, 2, 3)) / num_channel
