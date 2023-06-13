# Copyright (c) Facebook, Inc. and its affiliates.
# yr copy from detectron GeneralizedRCNN, 2023.01.25

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like # 重新安装 detectron2
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .postprocessing import detector_postprocess

from skimage import color

__all__ = ["NewGeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class NewGeneralizedRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        pixel_std_bgr: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        enable_box_sup: bool = False,
        predict_in_box_space: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(NewGeneralizedRCNN, self).__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads,
            pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        
        self.is_rgb = True if input_format == 'RGB' else False

        self.register_buffer(
            'pixel_std_level_set', 
            torch.tensor(pixel_std_bgr).view(1, -1, 1, 1)) # (1, 3, 1, 1) 3 for channels
        self.enable_box_sup = enable_box_sup
        self.predict_in_box_space = predict_in_box_space

        self.is_debug = False

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "pixel_std_bgr": cfg.MODEL_PIXEL_STD_BGR,
            "enable_box_sup": cfg.MODEL.BOX_SUP.ENABLE,
            "predict_in_box_space": cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        # process batch image
        images = [x["image"].to(self.device) for x in batched_inputs]
        images_norm = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_norm = ImageList.from_tensors(
            images_norm,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        ### preparing the regularization needed image info
        images_info = {'images_norm': images_norm.tensor}
        if self.enable_box_sup:
            images_lab = [x["image_lab"].to(self.device) for x in batched_inputs]
            images_lab = ImageList.from_tensors(
                images_lab,
                self.backbone.size_divisibility,
                padding_constraints=self.backbone.padding_constraints,)
            images_info["images"] = images_lab.tensor
            
            # original_images = ImageList.from_tensors(
            #     images,
            #     self.backbone.size_divisibility,
            #     padding_constraints=self.backbone.padding_constraints,)
            # self.get_lab_images(images_info, original_images.tensor, is_rgb=self.is_rgb)  # get lab images for pairwise term
            # norm images for level set, since BGR format self.pixel_std eqals [1.0, 1.0, 1.0], we need to norm them again
            # images_level_set = [(x - self.pixel_mean) / self.pixel_std_level_set for x in images]
            # images_level_set = ImageList.from_tensors(
            #     images_level_set,
            #     self.backbone.size_divisibility,
            #     padding_constraints=self.backbone.padding_constraints,)
            
            # norm images for level set, since BGR format self.pixel_std eqals [1.0, 1.0, 1.0], we need to norm them again
            images_level_set = images_norm.tensor if self.is_rgb else images_norm.tensor / self.pixel_std_level_set
            images_info['images_norm'] = images_level_set
            
            # del gt_masks if exits
            if gt_instances[0].has('gt_masks'):
                for gt_inst in gt_instances: gt_inst.remove('gt_masks')
            ### preparing the image info end
        # else:
        #     if not self.predict_in_box_space:
        #         self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        # 占用显存太大

        features = self.backbone(images_norm.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images_norm, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        # yr: here add images_info to roi heads
        _, detector_losses = self.roi_heads(images_info, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals) 

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(None, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return NewGeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    def get_lab_images(self, images_info, original_images, is_rgb=False):
        if not is_rgb:
            original_images = original_images[:, [2, 1, 0]] # BGR to RGB
        images_lab = []
        for index in range(len(original_images)):
            if self.is_debug:
                import matplotlib.pyplot as plt
                # visualize
                img_path = f"/home/yr/BoundaryFormer/visualize/{str(index)}_ori_img.jpg"
                _img = original_images[index].byte().permute(1, 2, 0).cpu().numpy()
                plt.imsave(img_path, _img)

            image_lab = color.rgb2lab(original_images[index].byte().permute(1, 2, 0).cpu().numpy())
            image_lab = torch.as_tensor(image_lab, device=original_images[index].device, dtype=torch.float32) # fp16的时候怎么办？
            image_lab = image_lab.permute(2, 0, 1)[None].contiguous()
            images_lab.append(image_lab)
        images_info['images'] = torch.cat(images_lab)
        
    def add_bitmasks(self, instances, img_h, img_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            # start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                # per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, img_h, img_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    # start = int(self.mask_out_stride // 2)
                    # bitmask_full = bitmask
                    # bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    # assert bitmask.size(0) * self.mask_out_stride == img_h
                    # assert bitmask.size(1) * self.mask_out_stride == img_w

                    per_im_bitmasks.append(bitmask)
                    # per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks, dim=0)
                # per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, img_w - w, 0, img_h - h), "constant", 0)
                # bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                
                # per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full
