from cv2 import norm
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18, resnet50
from .graphmap import *

import data.utils as utils

# Precomputed aliases
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 2)),
        ('reduction_2', (2, 4)),
        ('reduction_3', (4, 6)),
        ('reduction_4', (6, 12))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 7)),
        ('reduction_3', (7, 11)),
        ('reduction_4', (11, 23)),
    ]
}


class EfficientNetExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layer_names, image_height, image_width, model_name='efficientnet-b4'):
        super().__init__()

        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in layer_names)

        idx_max = -1
        layer_to_idx = {}

        # Find which blocks to return
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in layer_names:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i

        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        # net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]

        # Only run needed blocks
        for idx in range(idx_max):
            l, r = MODELS[model_name][idx][1]

            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        self.layers = nn.Sequential(*blocks)
        self.layer_names = layer_names
        self.idx_pick = [layer_to_idx[l] for l in layer_names]

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True) # [B*N, 3, 128, 352]

        result = [] # [B*N, 48, 64, 176] -> [B*N, 32, 32, 88] -> [B*N, 56, 16, 44] -> [B*N, 112, 8, 22]

        for layer in self.layers:
            # if self.training:
            #     x = torch.utils.checkpoint.checkpoint(layer, x)
            # else:
            #     x = layer(x)

            x = layer(x)
            result.append(x)

        return [result[i] for i in self.idx_pick]


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]

        super().__init__(*layers)

        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)

        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class UpDT(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x1, x2):
        # x1: feature
        # x2: distance transform
        x1 = self.up(x1)
        x1 = torch.cat([x1, x2], dim=1)
        return self.conv(x1) # b, 3, 200, 400


class CamEncode(nn.Module):
    def __init__(self, C, D=None, backbone='efficientnet-b4', norm_layer=nn.BatchNorm2d):
        super(CamEncode, self).__init__()
        self.C = C
        self.D = D
        self.backbone = backbone

        if 'efficientnet' in backbone:
            self.trunk = EfficientNet.from_pretrained(backbone)
        elif backbone == 'resnet-18':
            self.trunk = resnet18(pretrained=True)
        elif backbone == 'resnet-50':
            self.trunk = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        
        if backbone == 'efficientnet-b0':
            channel = 320+112
        elif backbone == 'efficientnet-b4':
            channel = 448+160
        elif backbone == 'efficientnet-b7':
            channel = 640+224
        elif backbone == 'resnet-18':
            channel = 512+256
        elif backbone == 'resnet-50':
            channel = 2048+1024
        else:
            raise NotImplementedError

        self.up1 = Up(channel, self.C, norm_layer=norm_layer) # 320+112
        if D is not None:
            self.depthnet = nn.Conv2d(self.C, D + self.C, kernel_size=1, padding=0)

        """
        b0
        reduction_1: torch.Size([1, 16, 112, 112])
        reduction_2: torch.Size([1, 24, 56, 56])
        reduction_3: torch.Size([1, 40, 28, 28])
        reduction_4: torch.Size([1, 112, 14, 14])
        reduction_5: torch.Size([1, 320, 7, 7])
        reduction_6: torch.Size([1, 1280, 7, 7])

        b4
        reduction_1: torch.Size([1, 24, 112, 112])
        reduction_2: torch.Size([1, 32, 56, 56])
        reduction_3: torch.Size([1, 56, 28, 28])
        reduction_4: torch.Size([1, 160, 14, 14])
        reduction_5: torch.Size([1, 448, 7, 7])
        reduction_6: torch.Size([1, 1792, 7, 7])

        b7
        reduction_1: torch.Size([1, 32, 112, 112])
        reduction_2: torch.Size([1, 48, 56, 56])
        reduction_3: torch.Size([1, 80, 28, 28])
        reduction_4: torch.Size([1, 224, 14, 14])
        reduction_5: torch.Size([1, 640, 7, 7])
        reduction_6: torch.Size([1, 2560, 7, 7])

        r18
        x1: torch.Size([1, 64, 56, 56])
        x2: torch.Size([1, 128, 28, 28])
        x3: torch.Size([1, 256, 14, 14])
        x4: torch.Size([1, 512, 7, 7])
        
        r50
        x1: torch.Size([1, 256, 56, 56])
        x2: torch.Size([1, 512, 28, 28])
        x3: torch.Size([1, 1024, 14, 14])
        x4: torch.Size([1, 2048, 7, 7])
        """

    def get_eff_depth(self, x):
        # x: B*N, C, H, W [128, 352]
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        # Conv -> BN -> Swish
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def get_resnet_depth(self, x):
        # x: B*N, C, H, W
        # adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L266
        x = self.trunk.conv1(x) # [B*N, 64, H/2, W/2]
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x) # [B*N, 64, H/4, W/4]

        x1 = self.trunk.layer1(x) # [B*N, 64 or 246, H/4, W/4]
        x2 = self.trunk.layer2(x1) # [B*N, 128 or 512, H/8, W/8]
        x3 = self.trunk.layer3(x2) # [B*N, 256 or 1024, H/16, W/16]
        x4 = self.trunk.layer4(x3) # [B*N, 512 or 2048, H/32, W/32]

        x = self.up1(x4, x3)
        return x
    
    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)
    
    def get_depth_feat(self, x):
        if 'efficientnet' in self.backbone:
            x = self.get_eff_depth(x)
        elif 'resnet' in self.backbone:
            x = self.get_resnet_depth(x)
        else:
            raise NotImplementedError

        if self.D is not None:
            x = self.depthnet(x) # B*N, 41+64, 8, 22

            depth = self.get_depth_dist(x[:, :self.D]) # B*N, 41, 8, 22
            new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2) # [B*N, 1, 41, 8, 22] * [B*N, 64, 1, 8, 22]

            return new_x # [B*N, 64, 41, 8, 22]
        else:
            return x

    def forward(self, x):
        x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC, norm_layer=nn.BatchNorm2d, segmentation=True, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37, distance_reg=True, vertex_pred=True, cell_size=8):
        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4, norm_layer=norm_layer)

        self.segmentation = segmentation
        self.up2 = nn.Sequential( # final semantic segmentation prediction
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0), # outC = 4 (num_classes)
        )

        self.distance_reg = distance_reg
        if distance_reg:
            # self.up1_dt = Up(64 + 256, 256, scale_factor=4)
            self.up_dt = nn.Sequential( # distance transform prediction
                # b, 256, 100, 200
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                # b, 256, 200, 400
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                # b, 128, 200, 400
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, outC-1, kernel_size=1, padding=0), # outC = 3 no background
                # b, 3, 200, 400
            )
            self.up3 = UpDT(256 + outC-1, outC, scale_factor=2, norm_layer=norm_layer)
        else:
            self.up_bin = nn.Sequential( # distance transform prediction
                # b, 256, 100, 200
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                # b, 256, 200, 400
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                # b, 128, 200, 400
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, outC-1, kernel_size=1, padding=0), # outC = 3 no background
                # b, 3, 200, 400
            )

        self.vertex_pred = vertex_pred
        self.cell_size = cell_size
        if vertex_pred:
            self.vertex_head = nn.Sequential(
                # b, 256, 100, 200
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                # b, 128, 100, 200
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 256, 50, 100
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False), # 65: cell_size*cell_size + 1 (dustbin)
                # b, 128, 50, 100
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 128, 25, 50
                nn.Conv2d(128, cell_size*cell_size+1, kernel_size=1, padding=0), # 65: cell_size*cell_size + 1 (dustbin)
                # b, cs^2+1, 25, 50
            )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4, norm_layer=norm_layer)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4, norm_layer=norm_layer)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x): # x: b, 64, 200, 400; 400, 400
        x = self.conv1(x) # b, 64, 100, 200
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # b, 64, 100, 200
        x = self.layer2(x1) # b, 128, 50, 100
        x2 = self.layer3(x) # b, 256, 25, 50

        x = self.up1(x2, x1) # b, 256, 100, 200, apply distance transform after here

        if self.vertex_pred:
            x_vertex = self.vertex_head(x) # b, 65, 25, 50; 50, 50
        else:
            x_vertex = None

        if self.distance_reg:
            x_dt = self.up_dt(x) # b, 3, 200, 400; 400, 400
            # x: [b, 256, 100, 200], x_dt: [b, 3, 200, 400]
            # concat [x, x_dt] and upsample to get dense semantic prediction
            if self.segmentation:
                x_seg = self.up3(x, self.relu(x_dt)) # b, 4, 200, 400
            else:
                x_seg = None
        else:
            # x_dt = None # b, 4, 200, 400 # semantic segmentation prediction
            x_dt = x # for visual descriptor
            if self.segmentation:
                x_seg = self.up2(x) # b, 4, 200, 400 # semantic segmentation prediction
            else:
                x_seg = None
        # x = self.up2(x) # b, 4, 200, 400 # semantic segmentation prediction
        
        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1) # b, 256, 100, 200
            x_embedded = self.up2_embedded(x_embedded) # b, 16, 200, 400
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction) # b, 37, 200, 400
        else:
            x_direction = None

        return x_seg, x_dt, x_vertex, x_embedded, x_direction

class InstaGraM(nn.Module):
    def __init__(self, data_conf, norm_layer, distance_reg=True, refine=False) -> None:
        super(InstaGraM, self).__init__()

        self.num_classes = data_conf['num_channels'] # 4
        self.cell_size = data_conf['cell_size']
        self.dist_threshold = data_conf['dist_threshold']
        self.distance_reg = distance_reg
        self.xbound = data_conf['xbound'][:-1] # [-30.0, 30.0]
        self.ybound = data_conf['ybound'][:-1] # [-15.0, 15.0]
        self.resolution = data_conf['xbound'][-1] # 0.15
        self.vertex_threshold = data_conf['vertex_threshold'] # 0.015
        self.max_vertices = data_conf['num_vectors'] # 300
        self.feature_dim = data_conf['feature_dim'] # 256
        self.pos_freq = data_conf['pos_freq']
        self.sinkhorn_iters = data_conf['sinkhorn_iterations'] # 100 default 0: not using sinkhorn
        self.gnn_layers = data_conf['gnn_layers']
        self.refine = refine

        self.center = torch.tensor([self.xbound[0], self.ybound[0]]).cuda() # -30.0, -15.0

        # Positional encoding
        self.pe_fn, self.pe_dim = get_embedder(data_conf['pos_freq'])
        
        # Graph neural network
        self.venc = GraphEncoder(self.feature_dim, [self.pe_dim + 1, 64, 128, 256], norm_layer) # 43 -> 64 -> 128 -> 256 -> 256
        embedding_dim = (self.num_classes-1)*self.cell_size*self.cell_size if distance_reg else 256 # 192 or 256
        self.dtenc = GraphEncoder(self.feature_dim, [embedding_dim, 64, 128, 256], norm_layer) # 192/256 -> 128 -> 256 for visual descriptor
        self.gnn = AttentionalGNN(self.feature_dim, ['self']*self.gnn_layers, norm_layer)
        self.final_proj = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=True)

        if self.sinkhorn_iters > 0:
            bin_score = nn.Parameter(torch.tensor(1.))
            self.register_parameter('bin_score', bin_score)
        else:
            self.matchability = nn.Conv1d(self.feature_dim, 1, kernel_size=1, bias=True)

        self.cls_head = nn.Conv1d(self.feature_dim, self.num_classes-1, kernel_size=1, bias=True)
        if self.refine:
            self.offset_head = nn.Conv1d(self.feature_dim, 2, kernel_size=1, bias=True)

    def forward(self, semantic, distance, vertex, instance, direction):
        """ semantic, instance, direction are not used
        @ vertex: (b, 65, 25, 50); (..., 50, 50)
        @ distance: (b, 3, 200, 400); (..., 400, 400)
        """

        # Compute the dense vertices scores (heatmap)
        scores = F.softmax(vertex, 1) # (b, 65, 25, 50); (..., 50, 50)
        scores = scores[:, :-1] # b, 64, 25, 50; 50, 50
        b, _, h, w = scores.shape # b, 64, 25, 50; 50, 50
        mvalues, mindicies = scores.max(1, keepdim=True) # b, 1, 25, 50; 50, 50
        scores_max = scores.new_full(scores.shape, 0., dtype=scores.dtype)
        scores_max = scores_max.scatter_(1, mindicies, mvalues) # b, 64, 25, 50; 50, 50
        scores_max = scores_max.permute(0, 2, 3, 1).contiguous().reshape(b, h, w, self.cell_size, self.cell_size) # b, 25, 50, 64 -> b, 25, 50, 8, 8
        scores_max = scores_max.permute(0, 1, 3, 2, 4).contiguous().reshape(b, h*self.cell_size, w*self.cell_size) # b, 25, 8, 50, 8 -> b, 200, 400
        scores_max = simple_nms(scores_max, int(self.cell_size*0.5)) # b, 200, 400; 400, 400
        score_shape = scores_max.shape # b, 200, 400; 400, 400

        # [2] Extract vertices using NMS
        vertices = [torch.nonzero(s > self.vertex_threshold) for s in scores_max] # list of length b, [N, 2(row, col)] tensor
        scores = [s[tuple(v.t())] for s, v in zip(scores_max, vertices)] # list of length b, [N] tensor
        vertices_cell = [(v / self.cell_size).trunc().long() for v in vertices]

        # Extract distance transform
        if self.distance_reg:
            dt_embedding = sample_dt(vertices_cell, F.relu(distance).clamp(max=self.dist_threshold), self.cell_size) # list of [N, 193] tensor
        else:
            # distance: feature [b, 256, 100, 200]
            distance_down = F.interpolate(distance, scale_factor=0.25, mode='bilinear', align_corners=True) # [b, 256, 25, 50]
            dt_embedding = sample_feat(vertices_cell, distance_down) # list of [N, 256] tensor

        if self.max_vertices >= 0:
            vertices, scores, dt_embedding, masks = list(zip(*[
                top_k_vertices(v, s, d, self.max_vertices)
                for v, s, d in zip(vertices, scores, dt_embedding)
            ]))

        # Convert (h, w) to (x, y), normalized
        # v: [N, 2]
        vertices_norm = [normalize_vertices(torch.flip(v, [1]).float(), score_shape) for v in vertices] # list of [N, 2] tensor
        
        # Vertices in pixel coordinate
        vertices = torch.stack(vertices).flip([2]) # [b, N, 2] x, y

        # Positional embedding (x, y, c)
        pos_embedding = [torch.cat((self.pe_fn(v), s.unsqueeze(1)), 1) for v, s in zip(vertices_norm, scores)] # list of [N, pe_dim+1] tensor
        pos_embedding = torch.stack(pos_embedding) # [b, N, pe_dim+1]

        dt_embedding = torch.stack(dt_embedding) # [b, N, 64]
        masks = torch.stack(masks).unsqueeze(-1) # [b, N, 1]

        graph_embedding = self.venc(pos_embedding) + self.dtenc(dt_embedding) # for visual descriptor
        graph_embedding = self.gnn(graph_embedding, masks.transpose(1, 2)) # [b, 256, N], [b, L, 4, N, N]
        graph_embedding = self.final_proj(graph_embedding) # [b, 256, N]
        graph_cls = self.cls_head(graph_embedding) # [b, 3, N]
        if self.refine:
            offset = torch.tanh(self.offset_head(graph_embedding)) # [b, 2, N]

        # Adjacency matrix score as inner product of all nodes
        matches = torch.einsum('bdn,bdm->bnm', graph_embedding, graph_embedding)
        matches = matches / self.feature_dim**.5 # [b, N, N] [match.fill_diagonal_(0.0) for match in matches]
        
        # Don't care self matches
        b, m, n = matches.shape
        diag_mask = torch.eye(m).repeat(b, 1, 1).bool()
        matches[diag_mask] = -1e9

        # Don't care bin matches
        match_mask = torch.einsum('bnd,bmd->bnm', masks, masks) # [B, N, N]
        matches = matches.masked_fill(match_mask == 0, -1e9)
        
        # Matching layer
        if self.sinkhorn_iters > 0:
            matches = log_optimal_transport(matches, self.bin_score, self.sinkhorn_iters) # [b, N+1, N+1]
        else:
            z0 = self.matchability(graph_embedding) # [b, 1, N]
            matches = log_double_softmax(matches, z0, z0) # [b, N+1, N+1]
        # matches.exp() should be probability

        # Refinement offset in pixel coordinate
        if self.refine:
            _, h, w = score_shape
            offset = offset.permute(0, 2, 1).contiguous()*offset.new_tensor([self.cell_size, self.cell_size]) # [b, N, 2] [-cell_size ~ cell_size]
            vertices = torch.clamp(vertices + offset, max=offset.new_tensor([w-1, h-1]), min=offset.new_tensor([0, 0]))

        # return matches [b, N, N], vertices (pix coord) [b, N, 3], masks [b, N, 1]

        return F.log_softmax(graph_cls, dim=1), distance, vertex, instance, direction, (matches), vertices, masks # if NLLLoss: F.log_softmax(graph_cls, dim=1)