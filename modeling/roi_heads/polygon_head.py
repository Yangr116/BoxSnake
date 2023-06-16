import copy
import fvcore.nn.weight_init as weight_init
import imageio
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.structures import Boxes, Instances, BoxMode
from detectron2.modeling.poolers import ROIPooler

from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.nn.utils.rnn import pad_sequence

from modeling.diff_ras import MaskRasterizationLoss
from modeling.layers.deform_attn.modules import MSDeformAttn
from modeling.poolers import MultiROIPooler
from modeling.position_encoding import build_position_encoding
from modeling.tensor import NestedTensor
from modeling.transformer import DeformableTransformerDecoder, DeformableTransformerControlLayer, MLP, \
    point_encoding, UpsamplingDecoderLayer
from modeling.utils import (box_cxcywh_to_xyxy, box_xyxy_to_cxcywh,
                                   inverse_sigmoid, sample_ellipse_fast, POLY_LOSS_REGISTRY, _get_clones,
                                   sample_square)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

from detectron2.structures.instances import Instances
from detectron2.utils.events import get_event_storage

from modeling.layers.diff_ras.polygon import SoftPolygon
from modeling.utils import get_union_box, rasterize_instances, POLY_LOSS_REGISTRY, inverse_sigmoid
from detectron2.layers import ROIAlign

from modeling.box_supervisor import BoxSupLoss, create_box_targets, BoxSupervisor
from modeling.diff_ras import ClippingStrategy
from modeling.criterion import build_poly_losses

from modeling.soft_polygon import SoftPolygonBatch
from modeling.criterion import MaskCriterion


@ROI_MASK_HEAD_REGISTRY.register()
class PolygonHead(nn.Module):
    """
    polygon head from BoundaryFormer
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, in_features, vertex_loss_fns, vertex_loss_ws, mask_criterion,
                 box_supervisor, ref_init="ellipse",
                 model_dim=256, base_number_control_points=8, number_control_points=64, number_layers=4, vis_period=0,
                 is_upsampling=True, iterative_refinement=False, use_cls_token=False, use_p2p_attn=True, num_classes=80,
                 cls_agnostic=False,
                 predict_in_box_space=False, prepool=True, dropout=0.0, deep_supervision=True,
                 inv_smoothness=0.1, resolution_list=[], enable_box_sup=False, box_feat_pooler=None,
                 box_feat_refiner=None, mask_stride=8,
                 crop_predicts=False, crop_size=64, mask_padding_size=4,
                 idx_output=None, **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.in_features = in_features
        self.num_feature_levels = len(self.in_features)
        self.ref_init = ref_init

        self.batch_size_div = 16

        if not ref_init in ["ellipse", "random", "convex", "square"]:
            raise ValueError("unknown ref_init {0}".format(ref_init))

        self.base_number_control_points = base_number_control_points
        self.number_control_points = number_control_points
        self.model_dimension = model_dim
        self.is_upsampling = is_upsampling
        self.iterative_refinement = iterative_refinement or self.is_upsampling
        self.use_cls_token = use_cls_token
        self.use_p2p_attn = use_p2p_attn
        self.num_classes = num_classes
        self.cls_agnostic = cls_agnostic
        self.vis_period = vis_period
        self.predict_in_box_space = predict_in_box_space
        self.crop_predicts = crop_predicts
        self.prepool = prepool
        self.dropout = dropout
        self.deep_supervision = deep_supervision

        self.vertex_loss_fns = []
        for loss_fn in vertex_loss_fns:
            loss_fn_attr_name = "vertex_loss_fn_{0}".format(loss_fn.name)
            self.add_module(loss_fn_attr_name, loss_fn)

            self.vertex_loss_fns.append(getattr(self, loss_fn_attr_name))

        # add each as a module so it gets moved to the right device.
        self.vertex_loss_ws = vertex_loss_ws

        if len(self.vertex_loss_fns) != len(self.vertex_loss_ws):
            raise ValueError("vertex loss mismatch")

        self.position_embedding = build_position_encoding(self.model_dimension, kind="sine")
        self.level_embed = nn.Embedding(self.num_feature_levels, self.model_dimension)
        self.register_buffer("point_embedding",
                             point_encoding(self.model_dimension * 2, max_len=self.number_control_points))

        if self.use_cls_token:
            self.cls_token = nn.Embedding(self.num_classes, self.model_dimension * 2)

        self.xy_embed = MLP(self.model_dimension, self.model_dimension,
                            2 if self.cls_agnostic else 2 * self.num_classes, 3)

        if self.ref_init == "random":
            self.reference_points = nn.Linear(self.model_dimension, 2)
        else:
            nn.init.constant_(self.xy_embed.layers[-1].bias.data, 0.0)
            nn.init.constant_(self.xy_embed.layers[-1].weight.data, 0.0)

        if self.model_dimension != 256:
            self.feature_proj = nn.ModuleList([nn.Linear(256, self.model_dimension) for _ in range(self.num_feature_levels)])
        else:
            self.feature_proj = None

        activation = "relu"
        dec_n_points = 4
        nhead = 8

        self.feedforward_dimension = 1024
        decoder_layer = DeformableTransformerControlLayer(
            self.model_dimension, self.feedforward_dimension, self.dropout, activation, self.num_feature_levels, nhead,
            dec_n_points,
            use_p2p_attn=self.use_p2p_attn)

        if self.is_upsampling:
            # yr: to increase the midpoints between two neighborly points
            decoder_layer = UpsamplingDecoderLayer(
                self.model_dimension, self.base_number_control_points, self.number_control_points, decoder_layer)
            self.start_idxs = decoder_layer.idxs[0]
            number_layers = decoder_layer.number_iterations  # so we can get a final "layer".
            print(number_layers)
        else:
            number_layers = number_layers

        self.decoder = DeformableTransformerDecoder(
            decoder_layer, number_layers, return_intermediate=True, predict_in_box_space=self.predict_in_box_space)

        num_pred = self.decoder.num_layers

        if self.iterative_refinement:
            self.xy_embed = _get_clones(self.xy_embed, num_pred)
            nn.init.constant_(self.xy_embed[0].layers[-1].bias.data, 0.0)
            self.decoder.xy_embed = self.xy_embed
        else:
            self.xy_embed = nn.ModuleList([self.xy_embed for _ in range(num_pred)])

        # rasterizer
        self.inv_smoothness = inv_smoothness
        self.offset = 0.5
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        # self.pred_rasterizer = SoftPolygonBatch(inv_smoothness=self.inv_smoothness)
        self.register_buffer("rasterize_at", torch.from_numpy(np.array(resolution_list).reshape(-1, 2)))
        mask_criterion.rasterize_at = self.rasterize_at
        self.mask_criterion = mask_criterion

        self.mask_stride = mask_stride
        self.mask_stride_lvl_name = f'p{str(int(math.log(mask_stride, 2)))}'
        assert self.mask_stride_lvl_name in self.in_features

        # box_sup
        self.enable_box_sup = enable_box_sup
        self.box_supervisor = box_supervisor
        self.box_feat_pooler = box_feat_pooler
        self.box_feat_refiner = box_feat_refiner
        self.mask_padding_size = mask_padding_size
        self.crop_size = crop_size

        # inference
        self.idx_output = -1
        if idx_output is not None:
            assert -number_layers <= idx_output < number_layers
            self.idx_output = int(idx_output)

        self.debug = False

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if ("xy_embed" in name):
                continue

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.ref_init == "random":
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)

        normal_(self.level_embed.weight.data)

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features = cfg.MODEL.POLYGON_HEAD.IN_FEATURES
        enable_box_sup = cfg.MODEL.BOX_SUP.ENABLE

        ret = {
            "in_features": in_features,
            "ref_init": cfg.MODEL.POLYGON_HEAD.POLY_INIT,
            "model_dim": cfg.MODEL.POLYGON_HEAD.MODEL_DIM,
            "number_layers": cfg.MODEL.POLYGON_HEAD.NUM_DEC_LAYERS,
            "base_number_control_points": cfg.MODEL.POLYGON_HEAD.UPSAMPLING_BASE_NUM_PTS,
            "number_control_points": cfg.MODEL.POLYGON_HEAD.POLY_NUM_PTS,
            "vis_period": cfg.VIS_PERIOD,
            "vertex_loss_fns": build_poly_losses(cfg, input_shape),
            "vertex_loss_ws": cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS,
            "mask_criterion": MaskCriterion(cfg),
            "box_supervisor": BoxSupervisor(cfg),
            "is_upsampling": cfg.MODEL.POLYGON_HEAD.UPSAMPLING,
            "iterative_refinement": cfg.MODEL.POLYGON_HEAD.ITER_REFINE,
            "use_cls_token": cfg.MODEL.POLYGON_HEAD.USE_CLS_TOKEN,
            "use_p2p_attn": cfg.MODEL.POLYGON_HEAD.USE_P2P_ATTN,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic": cfg.MODEL.POLYGON_HEAD.CLS_AGNOSTIC_MASK,
            "predict_in_box_space": cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX,
            "prepool": cfg.MODEL.POLYGON_HEAD.PREPOOL,
            "dropout": cfg.MODEL.POLYGON_HEAD.DROPOUT,
            "deep_supervision": cfg.MODEL.POLYGON_HEAD.DEEP_SUPERVISION,
            "inv_smoothness": cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED[0],
            "resolution_list": cfg.MODEL.DIFFRAS.RESOLUTIONS,
            "enable_box_sup": enable_box_sup,
            "mask_stride": cfg.MODEL.POLYGON_HEAD.MASK_STRIDE,
            "crop_predicts": cfg.MODEL.BOX_SUP.CROP_PREDICTS,
            "crop_size": cfg.MODEL.BOX_SUP.CROP_SIZE,
            "mask_padding_size": cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE,
            # test
            "idx_output": cfg.MODEL.POLYGON_HEAD.IDX_OUTPUT,
        }

        ret.update(
            input_shape=input_shape,
        )
        return ret

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, images, x, instances: List[Instances]):
        """
        images (torch.Tensor): shape=(B, C, H, W)
        x (dict(Tensor)): fpn feature
        instances: List[]
        """
        mask_wh = torch.as_tensor(x[self.mask_stride_lvl_name].shape[-2:][::-1])
        x = [x[f] for f in self.in_features]
        device = x[0].device
        mask_wh = mask_wh.to(device=device)

        if self.prepool:
            if False:
                input_shapes = [x_.shape[-2:] for x_ in x]
                input_ys = [torch.linspace(-1, 1, s[0], device=device) for s in input_shapes]
                input_xs = [torch.linspace(-1, 1, s[1], device=device) for s in input_shapes]
                input_grid = [torch.stack(torch.meshgrid(y_, x_), dim=-1).unsqueeze(0).repeat(x[0].shape[0], 1, 1, 1)
                              for y_, x_ in zip(input_ys, input_xs)]
                x = [F.grid_sample(x_, grid_) for x_, grid_ in zip(x, input_grid)]
            else:
                # todo, find out how the core reason this works so well.
                aligner = MultiROIPooler(
                    list(itertools.chain.from_iterable([[tuple(x_.shape[-2:])] for x_ in x])),
                    scales=(0.25, 0.125, 0.0625, 0.03125),  # correspongding with scale of P2, P3, P4, and P5
                    sampling_ratio=0,
                    pooler_type="ROIAlignV2",
                    # pooler_type="ROIAlign",
                    assign_to_single_level=False)

                x = aligner(x,
                            [Boxes(torch.Tensor([[0, 0, inst.image_size[1], inst.image_size[0]]]).to(x[0].device)) for
                             inst in instances])

        if self.feature_proj is not None:  # default None
            x = [self.feature_proj[i](x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, x_ in enumerate(x)]

        number_levels = len(x)
        batch_size, feat_dim = x[0].shape[:2]

        # empty instance during inference
        if not self.training:
            no_instances = len(instances[0]) == 0
            if no_instances:
                instances[0].pred_masks = torch.zeros((0, 1, 4, 4), device=device)
                return instances

        masks = []
        pos_embeds = []
        srcs = []

        for l in range(number_levels):
            srcs.append(x[l])

            mask = torch.zeros((batch_size, x[l].shape[-2], x[l].shape[-1]), dtype=torch.bool, device=device)
            masks.append(mask)

            # todo, for non-pooled situation.. actually get the mask.
            f = NestedTensor(x[l], mask)
            pos_embeds.append(self.position_embedding(f))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        if self.is_upsampling:
            # make sure to pull this out correctly.
            query_embed, tgt = torch.split(self.point_embedding[self.start_idxs], self.model_dimension, dim=1)
        else:
            query_embed, tgt = torch.split(self.point_embedding, self.model_dimension, dim=1)

        number_instances = [len(inst) for inst in instances]
        max_instances = max(number_instances)
        box_preds_xyxy = pad_sequence([
            (inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor) / torch.Tensor(2 * inst.image_size[::-1]).to(device)
            for inst in instances], batch_first=True) 

        # padding different batch as DETR
        query_embed = query_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        tgt = tgt.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        cls_token = None

        if self.ref_init == "ellipse":
            reference_points = sample_ellipse_fast(
                0.5 * torch.ones((batch_size, max_instances), device=device),
                0.5 * torch.ones((batch_size, max_instances), device=device),
                0.49 * torch.ones((batch_size, max_instances), device=device),
                0.49 * torch.ones((batch_size, max_instances), device=device),
                count=len(self.start_idxs) if self.is_upsampling else self.number_control_points)
        elif self.ref_init == "square":
            reference_points = sample_square(
                batch_size, max_instances,
                count=len(self.start_idxs) if self.is_upsampling else self.number_control_points,
                device=device)
        else:
            raise ValueError("todo")

        # rescale
        if not self.predict_in_box_space:
            box_preds_xywh = [BoxMode.convert((inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor),
                                              BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) for inst in instances] # shape=(B, N, 4)
            padded_box_preds_xywh = pad_sequence(box_preds_xywh, batch_first=True)
            padded_box_xy, padded_box_wh = torch.split(padded_box_preds_xywh.unsqueeze(-2), 2, dim=-1)  # shape=(B, N, 1, 2)
            pad_img_wh = mask_wh * self.mask_stride  # devide
            reference_points = (reference_points * padded_box_wh + padded_box_xy) / pad_img_wh

        # decoder
        memory = src_flatten + lvl_pos_embed_flatten
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,
            cls_token=cls_token, reference_boxes=box_preds_xyxy)

        outputs_coords = []
        for lvl in range(len(hs)):
            xy = self.xy_embed[lvl](hs[lvl])  # 使用得到的 token 预测偏执

            outputs_coord = (inverse_sigmoid(inter_references[lvl]) + xy).sigmoid()  # 这里还是 (0, 1) 之间的

            # remove any padded entries.
            outputs_coords.append(outputs_coord)

        # unpad and flatten across batch dim.
        # take out valid instances in different batch
        for i in range(len(outputs_coords)):
            output_coords = outputs_coords[i]
            outputs_coords[i] = torch.cat([output_coords[j, :number_instances[j]] for j in range(batch_size)])

        # decoder end

        if self.training:
            if not self.deep_supervision:
                outputs_coords = [outputs_coords[-1]]
            
            # box prediction in BoundaryFormer
            if self.predict_in_box_space:
                pred_masks = [
                    self.pred_rasterizer(output_coords * float(self.rasterize_at[lid][1].item()) - self.offset,  # to P3
                                         self.rasterize_at[lid][1].item(),
                                         self.rasterize_at[lid][0].item(),
                                         1.0).unsqueeze(1)
                    for lid, output_coords in enumerate(outputs_coords)]
                pred_polygons = [output_coords * float(self.rasterize_at[lid][1].item())
                                 # NOTE: Does minusing offset matter?
                                 for lid, output_coords in
                                 enumerate(outputs_coords)]  # List(Tensor, ...) Tensor shape=(N_inst, Np, 2)
            # clipping strategy in BoxSnake
            elif self.crop_predicts:
                # we rasterize the predicted polygon to a mask in the ground-truth box, which has a fix size, like 64x64.
                # and padding 0 around it, such as padding size = (4, 4, 4, 4), the padded size is 72x72
                # before rasterize, we need to translate and scale the vertices to the 64x64 scope
                gt_boxes_xyxy = [BoxMode.convert((inst.gt_boxes.tensor), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) for inst in instances] # shape=(B, N, 4)
                gt_box_xy, gt_box_wh = torch.split(torch.cat(gt_boxes_xyxy, dim=0).unsqueeze(-2), 2, dim=-1)  # shape=(N, 1, 2)
                gt_box_wh = gt_box_wh.clamp(min=1.0)  # after padded sequence, wh may be zero.
                pred_masks = [self.pred_rasterizer((output_coords * pad_img_wh - gt_box_xy) / gt_box_wh * float(self.crop_size) - self.offset,
                                                   self.crop_size,  # w
                                                   self.crop_size,  # h
                                                   1.0).unsqueeze(1).contiguous()  # the output is (N, 1, H, W)
                              for lid, output_coords in enumerate(outputs_coords)]
                # here, we allow the vertices superpass the box. next, we padding the rasterized mask
                pred_masks = [
                    F.pad(pred_mask, tuple(4*[self.mask_padding_size]), mode='constant', value=0.)
                    for pred_mask in pred_masks]
                # pred_polygons lies in the original size
                pred_polygons = [output_coords * pad_img_wh for lid, output_coords in enumerate(outputs_coords)]  # original resoluation
                # List(Tensor, ...) Tensor shape=(N_inst, Np, 2)
            # pred in mask scale, like P3, P4
            else:
                pred_masks = [self.pred_rasterizer(output_coords * mask_wh - self.offset,
                                                   mask_wh[0].item(),  # w
                                                   mask_wh[1].item(),  # h
                                                   1.0).unsqueeze(1).contiguous()  # the output is (N, 1, H, W)
                              for lid, output_coords in enumerate(outputs_coords)]
                pred_polygons = [output_coords * pad_img_wh  # NOTE: Does minusing offset matter?
                                 for lid, output_coords in
                                 enumerate(outputs_coords)]  # List(Tensor, ...) Tensor shape=(N_inst, Np, 2)

            if self.enable_box_sup:
                preds = {'mask': pred_masks, 'polygon': pred_polygons}
                return self.box_supervisor(images, preds, instances)
            
            return self.mask_criterion(images, pred_masks, instances)

        pred_polys_per_image = outputs_coords[self.idx_output].split(number_instances, dim=0)
        for pred_polys, instance in zip(pred_polys_per_image, instances):
            # pred_polys \in (0, 1)
            # reference is normalized by the padded size
            # but, during inference, it will be multiply the unpadded size
            # Thus, we multiply the padded size to convert the pred_polys to the original coordinates
            pred_polys = pred_polys if self.predict_in_box_space else pred_polys * pad_img_wh
            instance.pred_polys = pred_polys

        return instances
