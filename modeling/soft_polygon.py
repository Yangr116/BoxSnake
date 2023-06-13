import torch
import torch.nn as nn


class SoftPolygonBatch(nn.Module):
    def __init__(self, inv_smoothness):
        super(SoftPolygonBatch, self).__init__()
        self.inv_smoothness = inv_smoothness

    def forward(self, vertices, width, height, p=1.0):
        H_ = height
        W_ = width
        N_ = vertices.size(0)
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=vertices.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=vertices.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # 生成特征图的坐标, shape=(H, W, 2), 2 表示坐标 (W, H)
        pnts = grid.unsqueeze(0).repeat(N_, 1, 1, 1).reshape(N_, -1, 1, 2).float()  # check 原文中有没有 + 0.5 TODO: + 0.5 ?
        # shape=(B, Np, 1, 2)

        # get winding number, return (B, Np) True or False 反应 points 是否在 polygon V 内部
        vex_start = vertices.unsqueeze(1)  # shape=(B, 1, Nv, 2)
        vex_end = torch.roll(vex_start, shifts=1, dims=-2)  # shape=(B, 1, Nv, 2)
        vex_end_start = vex_end - vex_start  # shape=(B, 1, Nv, 2)
        vex_end_start_x, vex_end_start_y = vex_end_start[..., 0], vex_end_start[..., 1]  # shape=(B, 1, Nv), (x1-x0), (y1-y0)
        pnts_vex_start = pnts - vex_start  # shape=(B, Np, Nv, 2), (x-x0), (y-y0)
        pnts_vex_start_x, pnts_vex_start_y = pnts_vex_start[..., 0], pnts_vex_start[..., 1]  # shape=(B, Np, Nv)

        with torch.no_grad():
            # (x1-x0)(y-y0) - (y1-y0)(x-x0)
            diff = vex_end_start_x * pnts_vex_start_y - vex_end_start_y * pnts_vex_start_x # shape=(B, Np, Nv)
            # condition 1: y > y0
            cond_a = pnts_vex_start[..., 1] > 0.0
            # condition 2: y < y1
            cond_b = (pnts[..., 1] - vex_end[..., 1]) < 0.0 # shape=(B, Np, 1) - (B, 1, Nv) = (B, Np, Nv)
            # condition 3: (x, y) on right, left, or
            cond_c = torch.sign(diff).to(dtype=torch.int32)
            pos_wn = (cond_a & cond_b & (cond_c > 0)).sum(dim=-1, dtype=torch.int)
            neg_wn = (~cond_a & ~cond_b & (cond_c < 0)).sum(dim=-1, dtype=torch.int)
            wn = pos_wn - neg_wn  # shape=(B, Np), wn == 0 means, pnts not in polygon V
            inside_outside = (wn != 0).float().reshape(N_, H_, W_)  # True 表示在内部， False 表示不在内部
            inside_outside[inside_outside == 0.] = -1. # inplace 操作, 不需要传播梯度 ？

        # distances from pnts to segments
        square_segment_length = torch.pow(vex_end_start_x, 2) + torch.pow(vex_end_start_y, 2) + 1e-5  # shape=(B, 1, Nv)
        param = (pnts_vex_start_x * vex_end_start_x + pnts_vex_start_y * vex_end_start_y) / square_segment_length 
        # shape=(B, Np, Nv) 点和segments的映射比例关系

        # cal distance
        # 垂足计算
        vex_proj = vex_start + param.unsqueeze(-1) * vex_end_start  # shape=(B, Np, Nv, 2)
        # 垂线
        pnts_proj = pnts - vex_proj  # shape=(B, Np, Nv, 2)
        pnts_vex_start = pnts - vex_start
        pnts_vex_end = pnts - vex_end

        # Does it matter here to compute the squared distance or true Euclidean distance?
        distance_a = torch.pow(pnts_proj[..., 0], 2) + torch.pow(pnts_proj[..., 1], 2)
        distance_b = torch.pow(pnts_vex_start[..., 0], 2) + torch.pow(pnts_vex_start[..., 1], 2)
        distance_c = torch.pow(pnts_vex_end[..., 0], 2) + torch.pow(pnts_vex_end[..., 1], 2)

        # convert to score
        distance = torch.where(param < 0., distance_b, distance_a)
        distance = torch.where(param > 1., distance_c, distance)  # shape=(B, Np, Nv)

        # convert to scores
        distance_min = torch.min(distance, dim=-1)[0]  # shape=(B, Np)
        score = torch.sigmoid(distance_min.view(N_, H_, W_) * inside_outside / self.inv_smoothness)
        
        return score