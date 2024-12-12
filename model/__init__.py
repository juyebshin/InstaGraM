from .hdmapnet import HDMapNet
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar
# from .graphmap import VectorMapNet

import torch.nn as nn

def get_model(method, data_conf, norm_layer_dict, segmentation=True, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36, distance_reg=True, vertex_pred=True, refine=False):
    if method == 'lift_splat' or method == 'LSS':
        model = LiftSplat(data_conf, segmentation, instance_seg, embedded_dim, direction_pred, distance_reg, vertex_pred, norm_layer_dict, refine)
    elif method == 'IPM' or method == 'ipm':
        model = IPMNet(data_conf, segmentation, instance_seg, embedded_dim, direction_pred, distance_reg, vertex_pred, norm_layer_dict, refine)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, norm_layer_dict, segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, distance_reg=distance_reg, vertex_pred=vertex_pred, refine=refine)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, norm_layer_dict, segmentation=segmentation, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, distance_reg=distance_reg, vertex_pred=vertex_pred, refine=refine)
    else:
        raise NotImplementedError

    return model
