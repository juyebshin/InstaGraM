import torch
from torch import nn

from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode, InstaGraM
from data.utils import gen_dx_bx


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size # 40, 80
        fv_dim = fv_size[0] * fv_size[1] # 8*22 = 176
        bv_dim = bv_size[0] * bv_size[1] # 40*80 = 3200
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape # batch, 6, 64, 8, 22
        feat = feat.view(B, N, C, H*W) # batch, 6, 64, 176
        outputs = []
        for i in range(N):
            # feat[:, i]: batch, 64, 176
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            # output: B, 64, 40, 80
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        # outputs: B, N, 64, 40, 80
        return outputs


class HDMapNet(nn.Module):
    def __init__(self, data_conf, norm_layer_dict, segmentation=True, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False, distance_reg=True, vertex_pred=True, refine=False):
        super(HDMapNet, self).__init__()
        self.camC = 64 # feature channel?
        self.downsample = 16

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        # nx = 0.15: tensor([400, 200,   1]), 0.3: tensor([200, 100,   1]), 0.6: tensor([100, 50,   1]) long
        # final_H: 200, final_W: 400
        final_H, final_W = nx[1].item(), nx[0].item()

        # EfficientNet-B0
        self.camencode = CamEncode(self.camC, backbone=data_conf['backbone'], norm_layer=norm_layer_dict['2d'])
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        # fv_size: (8, 22)
        bv_size = (final_H//5, final_W//5)
        # bv_size: 0.15: (40, 80), 0.3: (20, 40), 0.6: (10, 20); long: (80, 80)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4 # 0.15: 80*3 // 4 = 60, 0.3: 30, 0.6: 15; long: 80*3 // 4 = 60
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W] # 0.15: [-60, 60, 0.6], 0.3: [-30, 30, 0.6], 0.6: [-15, 15, 0.6]; long: [-60, 60, 0.6]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H] # 0.15: [-30, 30, 0.6], 0.3: [-15, 15, 0.6], 0.6: [-7.5, 7.5, 0.6]; long: [-60, 60, 0.6]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampler = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], norm_layer=norm_layer_dict['2d'], segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1, distance_reg=distance_reg, vertex_pred=vertex_pred, cell_size=data_conf['cell_size'])
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], norm_layer=norm_layer_dict['2d'], segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1, distance_reg=distance_reg, vertex_pred=vertex_pred, cell_size=data_conf['cell_size'])
        
        self.head = InstaGraM(data_conf, norm_layer_dict['1d'], distance_reg, refine)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape # batch, 6(surround), channel, 128, 352
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x) # batch*6, 64, 128, 352
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        # batch, 6, 64, 8, 22
        return x

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        x = self.get_cam_feats(img) # batch, 6, 64, 8, 22
        x = self.view_fusion(x) # batch, 6, 64, 40, 80
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        # Ks: batch, 6, eye(4, 4), RTs: batch, 6, RT(4, 4), post_RTs: None
        # RTs: ego to camera coordinate
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs) # b, 64, 100, 200; 200, 200
        topdown = self.up_sampler(topdown) # b, 64, 200, 400; 400, 400
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        x_seg, x_dt, x_vertex, x_embedded, x_direction = self.bevencode(topdown) # x, x_dt, x_vertex, x_embedded, x_direction
        return self.head(x_seg, x_dt, x_vertex, x_embedded, x_direction) #  -> InstaGraM
