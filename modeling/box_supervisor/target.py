from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec, cat, ROIAlign
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from .utils import get_images_color_similarity, unfold_wo_center



@torch.no_grad()
def create_box_targets(images, images_norm, instances, clip_boxes, mask_size, kernel_size=3, dilation=1, sigma=2.0):
    """
    instances ([Instance(), ...]) : 
    clip_boxes (Tensor()) : 
    return: 
    """
    img_h, img_w = images.shape[-2:]
    # mask_size = 64
    gt_boxmasks, gt_imgs_sim, gt_clipped_imgs = [], [], []
    # targets = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)
    if clip_boxes is not None:
        clip_boxes = torch.split(clip_boxes, [len(inst) for inst in instances], dim=0) # tenor to (tensor, tensor)
    
    for idx, (per_img_inst, img, img_norm) in enumerate(zip(instances, images, images_norm)):
        N_inst = len(per_img_inst)
        if N_inst == 0:
            continue
        # convert the gt_box to bitmask
        per_img_gt_boxes = per_img_inst.gt_boxes.tensor
        per_img_bitmasks_full = []
        for per_gt_box in per_img_gt_boxes:
            bitmask_full = torch.zeros((img_h, img_w), device=per_img_gt_boxes.device, dtype=torch.float)        
            bitmask_full[int(per_gt_box[1]): int(per_gt_box[3])+1, int(per_gt_box[0]): int(per_gt_box[2])+1] = 1.0
            per_img_bitmasks_full.append(bitmask_full)
        per_img_bitmasks_full = torch.stack(per_img_bitmasks_full, dim=0) # shape=(N, H, W)

        # clipping mask by pred boxes 
        per_img_pred_boxes = clip_boxes[idx]
        batch_inds = torch.arange(len(per_img_pred_boxes)).type_as(per_img_pred_boxes)[:, None]
        rois = torch.cat([batch_inds, per_img_pred_boxes], dim=1)  # Nx5
        rois = rois.to(device=per_img_bitmasks_full.device)
        clipped_per_img_bitmasks = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(per_img_bitmasks_full[:, None, :, :], rois)
            .squeeze(1)
        ) # shape=(N, 1, H, W)
        clipped_per_img_bitmasks = clipped_per_img_bitmasks > 0.5
        gt_boxmasks.append(clipped_per_img_bitmasks)

        # clipping img by pred boxes
        clipped_per_img = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(img[None].repeat(N_inst, 1, 1, 1), rois)
            .squeeze(1)
        ) # shape=(N, C, H, W)
        # cal sim, (N, C, H, W) --> (N, kernel_size^2, H, W)
        gt_imgs_sim.append(get_images_color_similarity(clipped_per_img, None, kernel_size=kernel_size, dilation=dilation, sigma=sigma))
        
        # clip img_norm
        clipped_per_img_norm = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(img_norm[None].repeat(N_inst, 1, 1, 1), rois)
            .squeeze(1)
        )
        gt_clipped_imgs.append(clipped_per_img_norm)
        
    gt_boxes = [inst.gt_boxes.tensor for inst in instances]
    return torch.cat(gt_boxmasks), torch.cat(gt_imgs_sim), torch.cat(gt_clipped_imgs), torch.cat(gt_boxes)


@torch.no_grad()
def create_box_targets_p3(images, images_norm, instances, mask_stride, kernel_size=3, dilation=1, sigma=2.0):
    """
    instances ([Instance(), ...]) : 
    clip_boxes (Tensor()) : 
    return: 
    """
    img_h, img_w = images.shape[-2:]
    # downsampled image
    downsampled_images_norm = F.avg_pool2d(
        images_norm, kernel_size=mask_stride, 
        stride=mask_stride, padding=0)
    downsampled_images = F.avg_pool2d(
        images, kernel_size=mask_stride, 
        stride=mask_stride, padding=0)

    gt_boxmasks, gt_imgs_sim, gt_imgs= [], [], []
    start = int(mask_stride // 2)
    for idx, per_img_inst in enumerate(instances):
        N_inst = len(per_img_inst)
        if N_inst == 0:
            continue
        # convert the gt_box to bitmask
        per_img_gt_boxes = per_img_inst.gt_boxes.tensor
        per_img_bitmasks = []
        for per_gt_box in per_img_gt_boxes:
            bitmask_full = torch.zeros((img_h, img_w), device=per_img_gt_boxes.device, dtype=torch.float)        
            bitmask_full[int(per_gt_box[1]): int(per_gt_box[3])+1, int(per_gt_box[0]): int(per_gt_box[2])+1] = 1.0
            bitmask = bitmask_full[start::mask_stride, start::mask_stride]
            per_img_bitmasks.append(bitmask)
        per_img_bitmasks = torch.stack(per_img_bitmasks, dim=0) # shape=(N, H_p3, W_p3)
        gt_boxmasks.append(per_img_bitmasks)
        # img sim
        img_color_sim = get_images_color_similarity(
            downsampled_images[idx].unsqueeze(0), None, kernel_size=kernel_size, dilation=dilation, sigma=sigma
            ) # (1, k^2 - 1, H, W)
        gt_imgs_sim.append(img_color_sim.expand(N_inst, -1, -1, -1))
        img_norm = downsampled_images_norm[idx].unsqueeze(0)
        gt_imgs.append(img_norm.expand(N_inst, -1, -1, -1))
    gt_boxes = [inst.gt_boxes.tensor for inst in instances] # (N, 4) ?

    return torch.cat(gt_boxmasks), torch.cat(gt_imgs_sim), torch.cat(gt_imgs), torch.cat(gt_boxes)


@torch.no_grad()
def create_box_targets_crop(images, images_norm, instances, mask_stride, kernel_size=3, dilation=1, crop_size=64, mask_padding_size=4, sigma=2.0):
    """
    instances ([Instance(), ...])
    clip_boxes (Tensor()) : 
    return: 
    """
    img_h, img_w = images.shape[-2:]
    gt_boxmasks, gt_imgs_sim, gt_imgs= [], [], []
    start = int(mask_stride // 2)
    clipped_size = tuple([int(crop_size + 2 * mask_padding_size)] * 2)
    
    for idx, per_img_inst in enumerate(instances):
        N_inst = len(per_img_inst)
        if N_inst == 0:
            continue
        # convert the gt_box to bitmask
        per_img_gt_boxes = per_img_inst.gt_boxes.tensor  # (x1, y1, x2, y2) format

        # padding mask
        offset = mask_padding_size * (per_img_gt_boxes[:, 2:] - per_img_gt_boxes[:, :2]) / float(crop_size) # shape=(N, 2)
        per_img_enlarged_boxes = per_img_gt_boxes + torch.cat([-offset, offset], dim=-1) # shape=(N, 4)
        # box maybe out of the image, so we pad the image and add a bias to the enlarged box
        pad_size = int(torch.max(offset).item()+1)
        img_lab = images[idx]
        img_norm = images_norm[idx]
        img_lab_padded = F.pad(img_lab, (pad_size, pad_size, pad_size, pad_size), value=0.)
        img_norm_padded = F.pad(img_norm, (pad_size, pad_size, pad_size, pad_size), value=0.)
        padded_mask = torch.ones(img_lab.shape[-2:]).type_as(img_lab)
        padded_mask = F.pad(padded_mask, (pad_size, pad_size, pad_size, pad_size), value=0.)
        # We move the box because the image is enlarged after padding
        per_img_enlarged_boxes = per_img_enlarged_boxes + float(pad_size)
        
        batch_inds = torch.zeros(len(per_img_enlarged_boxes)).type_as(per_img_enlarged_boxes)[:, None]
        rois = torch.cat([batch_inds, per_img_enlarged_boxes], dim=1)
        
        # clipped_mask
        clipped_per_img_mask = ROIAlign(clipped_size, 1.0, 0, aligned=True).forward(
            padded_mask[None, None], rois) # (N, 1, H, W)
        # prepare image similarity
        clipped_per_img = ROIAlign(clipped_size, 1.0, 0, aligned=True).forward(
            img_lab_padded.unsqueeze(0), rois).squeeze(1) # shape=(N, C, clipped_size, clipped_size)
        gt_img_sim = get_images_color_similarity(clipped_per_img, clipped_per_img_mask, kernel_size=kernel_size, dilation=dilation, sigma=sigma)      
        gt_imgs_sim.append(gt_img_sim)
        # prepare image norm for level set 
        clipped_per_img_norm = (
            ROIAlign(clipped_size, 1.0, 0, aligned=True)
            .forward(img_norm_padded.unsqueeze(0), rois)
            .squeeze(1)
        )
        clipped_per_img_norm = clipped_per_img_norm * clipped_per_img_mask
        gt_imgs.append(clipped_per_img_norm)
        
    gt_boxes = [inst.gt_boxes.tensor for inst in instances] # (N, 4) ?
    gt_imgs_sim = torch.cat(gt_imgs_sim)
    gt_imgs = torch.cat(gt_imgs)
    gt_boxes = torch.cat(gt_boxes)
    # prepare bitmasks
    clipped_per_bitmasks = torch.zeros(clipped_size, device=gt_imgs.device, dtype=torch.float)
    _box = [mask_padding_size, mask_padding_size, mask_padding_size+crop_size, mask_padding_size+crop_size]
    clipped_per_bitmasks[int(_box[0]): int(_box[3])+1, int(_box[1]): int(_box[2])+1] = 1.0
    gt_boxmasks = clipped_per_bitmasks.unsqueeze(0).expand(gt_imgs.size(0), -1, -1)

    return gt_boxmasks, gt_imgs_sim, gt_imgs, gt_boxes
