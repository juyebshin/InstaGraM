"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn

from data.utils import gen_dx_bx
from .base import CamEncode, BevEncode, InstaGraM
from .pointpillar import PointPillarEncoder


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class LiftSplat(nn.Module):
    def __init__(self, data_conf, segmentation, instance_seg, embedded_dim, direction_pred, distance_reg, vertex_pred, norm_layer_dict, lidar=False, refine=False):
        super(LiftSplat, self).__init__()
        self.data_conf = data_conf

        dx, bx, nx = gen_dx_bx(self.data_conf['xbound'],
                                              self.data_conf['ybound'],
                                              self.data_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 256
        self.lidar = lidar
        self.frustum = self.create_frustum()
        # D x H/downsample x D/downsample x 3
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.camC, self.D, data_conf['backbone'], norm_layer_dict['2d']) # self.downsample, 
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], norm_layer=norm_layer_dict['2d'], segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, distance_reg=distance_reg, vertex_pred=vertex_pred, cell_size=self.data_conf['cell_size'])
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], norm_layer=norm_layer_dict['2d'], segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, distance_reg=distance_reg, vertex_pred=vertex_pred, cell_size=self.data_conf['cell_size'])
        self.head = InstaGraM(data_conf, norm_layer_dict['1d'], distance_reg, refine)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_conf['image_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # *undo* post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3) # 4, 6, 41, 8, 22, 3
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points # B, N(=6), D(=41), H(=8), W(=22), 3

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x) # B*N, C*D, h, w
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample) # B, N, C, D, h, w
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()

        return x # B, N, D(=41), h, w, C(=64)

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # B, N(=6), D(=41), h, w, C(=64)
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # B x N x D x H/downsample x W/downsample x 3 // geom_feats: B, N(=6), D(=41), h, w, 3
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # x, y, z, b

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final # B, C(=64), Y, X

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # B x N x D x H/downsample x W/downsample x 3: (x,y,z) locations (in the ego frame)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # B x N x D x H/downsample x W/downsample x C: cam feats
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        # imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(), post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(), lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda()
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            x = torch.cat([x, lidar_feature], dim=1)
        x_seg, x_dt, x_vertex, x_embedded, x_direction = self.bevencode(x)
        return self.head(x_seg, x_dt, x_vertex, x_embedded, x_direction)
