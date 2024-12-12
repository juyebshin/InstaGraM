import argparse
import mmcv
import tqdm
import torch
import os
import argparse
import numpy as np
import cv2
from PIL import Image
import time

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgb

from data.dataset import HDMapNetDataset, VectorMapDataset, vectormap_dataset
from data.const import CAMS, NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize_graph
from data.visualize import colorise, colors_plt
from data.image import denormalize_img
from export_pred_to_json import gen_dx_bx
from data.utils import get_proj_mat, perspective
from torchsummary import summary
from mmcv.runner.fp16_utils import wrap_fp16_model

def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]] # [0.15, 0.15]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]] # [-29.925, -14.925]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound]] # [400, 200]
    return dx, bx, nx

def test(dataset: HDMapNetDataset, model, args, data_conf):
    dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'])
    car_img = Image.open('icon/car.png')
    img_size = data_conf['image_size'] # [128, 352]
    intrin_scale = torch.tensor([img_size[1] / 1600., img_size[0] / 900., 1.0]) * torch.eye(3)

    model_time_sum = 0
    post_time_sum = 0
    total_time_sum = 0
    
    model.eval()
    with torch.no_grad():
        # warmp up
        print("GPU warm-up...")
        for idx in tqdm.tqdm(range(50)):
            imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, \
            yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt = dataset.__getitem__(idx)
            

            imgs = imgs.unsqueeze(0) # [1, 6, 3, H, W]
            trans = trans.unsqueeze(0) # [1, 6, 3]
            rots = rots.unsqueeze(0) # [1, 6, 3, 3]
            intrins = intrins.unsqueeze(0) # [1, 6, 3, 3]
            post_trans = post_trans.unsqueeze(0) # [1, 6, 3]
            post_rots = post_rots.unsqueeze(0) # [1, 6, 3, 3]
            lidar_data = torch.tensor(lidar_data).unsqueeze(0) # [1, ]
            lidar_mask = torch.tensor(lidar_mask).unsqueeze(0) # [1, ]
            car_trans = car_trans.unsqueeze(0) # [1, 3]
            yaw_pitch_roll = yaw_pitch_roll.unsqueeze(0) # [1, 3]

            semantic, distance, vertex, embedding, direction, matches, positions, masks = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                    lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
        print("Running test...")
        for idx in (range(dataset.__len__())):
            imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, \
            yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt = dataset.__getitem__(idx)
            rec = dataset.samples[idx]
            scene_name = dataset.nusc.get('scene', rec['scene_token'])['name']
            lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
            base_name = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0].split('_')[-1] # timestamp
            base_name = scene_name + '_' + base_name # {scene_name}_{timestamp}

            imgs = imgs.unsqueeze(0) # [1, 6, 3, H, W]
            trans = trans.unsqueeze(0) # [1, 6, 3]
            rots = rots.unsqueeze(0) # [1, 6, 3, 3]
            intrins = intrins.unsqueeze(0) # [1, 6, 3, 3]
            post_trans = post_trans.unsqueeze(0) # [1, 6, 3]
            post_rots = post_rots.unsqueeze(0) # [1, 6, 3, 3]
            lidar_data = torch.tensor(lidar_data).unsqueeze(0) # [1, ]
            lidar_mask = torch.tensor(lidar_mask).unsqueeze(0) # [1, ]
            car_trans = car_trans.unsqueeze(0) # [1, 3]
            yaw_pitch_roll = yaw_pitch_roll.unsqueeze(0) # [1, 3]

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                        post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                        lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            
                # torch.cuda.synchronize()
                # mid_time = time.perf_counter()
                # coords, confidences, line_types = vectorize_graph(positions[0], matches[0], semantic[0], masks[0], data_conf['match_threshold'])
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            # model_time = mid_time - start_time
            # post_time = end_time - mid_time
            total_time = end_time - start_time

            # model_time_sum += model_time
            # post_time_sum += post_time
            total_time_sum += total_time

            # if args.vis:
            #     impath = os.path.join(args.logdir, 'images')
            #     if not os.path.exists(impath):
            #         os.mkdir(impath)
            #     imname = os.path.join(impath, f'{base_name}.jpg')
            #     # print('saving', imname)

            #     fig = plt.figure(figsize=(8, 3))
            #     for i, (img, intrin, rot, tran, cam) in enumerate(zip(imgs[0], intrins[0], rots[0], trans[0], CAMS)):
            #         img = np.array(denormalize_img(img)) # h, w, 3
            #         intrin = intrin_scale @ intrin
            #         P = get_proj_mat(intrin, rot, tran)
            #         ax = fig.add_subplot(2, 3, i+1)
            #         ax.get_xaxis().set_visible(False)
            #         ax.get_yaxis().set_visible(False)
            #         for coord, confidence, line_type in zip(coords, confidences, line_types):
            #             coord = coord * dx + bx # [-30, -15, 30, 15]
            #             pts, pts_num = coord, coord.shape[0]
            #             zeros = np.zeros((pts_num, 1))
            #             ones = np.ones((pts_num, 1))
            #             world_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
            #             pix_coords = perspective(world_coords, P)
            #             x = np.array([pts[0] for pts in pix_coords], dtype='int')
            #             y = np.array([pts[1] for pts in pix_coords], dtype='int')
            #             for j in range(1, x.shape[0]):
            #                 img = cv2.line(img, (x[j-1], y[j-1]), (x[j], y[j]), color=tuple([255*c for c in to_rgb(colors_plt[line_type])]), thickness=2)
            #         if i > 2:
            #             img = cv2.flip(img, 1)
            #         img = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_CUBIC)
            #         text_size, _ = cv2.getTextSize(cam, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
            #         text_w, text_h = text_size
            #         cv2.rectangle(img, (0, 0), (0+text_w, 0+text_h), color=(0, 0, 0), thickness=-1)
            #         img = cv2.putText(img, cam, (0, 0 + text_h + 1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
            #         ax.imshow(img)
                    
            #     plt.subplots_adjust(wspace=0.0, hspace=0.0)
            #     plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
            #     plt.close()

            #     # Vector map
            #     impath = os.path.join(args.logdir, 'vector_pred_final')
            #     if not os.path.exists(impath):
            #         os.mkdir(impath)
            #     imname = os.path.join(impath, f'{base_name}.png')
            #     # print('saving', imname)

            #     fig = plt.figure(figsize=(4, 2))
            #     plt.xlim(-30, 30)
            #     plt.ylim(-15, 15)
            #     plt.axis('off')

            #     for coord, confidence, line_type in zip(coords, confidences, line_types):
            #         coord = coord * dx + bx # [-30, -15, 30, 15]
            #         x = np.array([pt[0] for pt in coord])
            #         y = np.array([pt[1] for pt in coord])
            #         plt.scatter(coord[:, 0], coord[:, 1], 1.5, c=colors_plt[line_type])
            #         plt.plot(coord[:, 0], coord[:, 1], linewidth=2.0, color=colors_plt[line_type], alpha=0.7)
            #         # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
            #         # plt.plot(x, y, '-', c=colors_plt[line_type], linewidth=2)
            #     plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
            #     plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
            #     plt.close()

            #     # Instance map
            #     impath = os.path.join(args.logdir, 'instance_pred')
            #     if not os.path.exists(impath):
            #         os.mkdir(impath)
            #     imname = os.path.join(impath, f'{base_name}.png')
            #     # print('saving', imname)

            #     fig = plt.figure(figsize=(4, 2))
            #     plt.xlim(-30, 30)
            #     plt.ylim(-15, 15)
            #     plt.axis('off')

            #     for coord in coords:
            #         coord = coord * dx + bx # [-30, -15, 30, 15]
            #         plt.plot(coord[:, 0], coord[:, 1], linewidth=2)
            #     plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
            #     plt.savefig(imname, bbox_inches='tight', pad_inches=0, dpi=400)
            #     plt.close()
                
            if (idx + 1) % args.log_interval == 0:
                # model_fps = (idx + 1) / model_time_sum
                # post_fps = (idx + 1) / post_time_sum
                total_fps = (idx + 1) / total_time_sum
                print(f'Done image [{idx + 1:<3}/ {args.samples}], ',
                    #   f'Model FPS: {model_fps:>5.2f}    Post FPS: {post_fps:>5.2f}    Total FPS: {total_fps:>5.2f}',
                      f'fps: {total_fps:.1f} img / s')
                
            if (idx + 1) == args.samples:
                # model_fps = (idx + 1) / model_time_sum
                # post_fps = (idx + 1) / post_time_sum
                total_fps = (idx + 1) / total_time_sum
                print('Overall: '
                    #   f'Model FPS: {model_fps:>5.2f}    Post FPS: {post_fps:>5.2f}    Total FPS: {total_fps:>5.2f}',
                      f'fps: {total_fps:.1f} img / s')
                break

def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'backbone': args.backbone,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'sample_dist': args.sample_dist, # 1.5
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'dist_threshold': args.dist_threshold, # 10.0
        'cell_size': args.cell_size, # 8
        'num_vectors': args.num_vectors, # 100
        'pos_freq': args.pos_freq, # 10
        'feature_dim': args.feature_dim, # 256
        'gnn_layers': args.gnn_layers, # ['self']*7
        'sinkhorn_iterations': args.sinkhorn_iterations, # 100
        'vertex_threshold': args.vertex_threshold, # 0.015
        'match_threshold': args.match_threshold, # 0.1
    }

    dataset = VectorMapDataset(version=args.version, dataroot=args.dataroot, data_conf=data_conf, is_train=False)
    # _, val_loader = vectormap_dataset(args.version, args.dataroot, data_conf, 1, args.nworkers, False)
    # model = get_model(args.model, data_conf, True, args.embedding_dim, True, args.angle_class)
    norm_layer_dict = {'1d': torch.nn.BatchNorm1d, '2d': torch.nn.BatchNorm2d}
    model = get_model(args.model, data_conf, norm_layer_dict, False, False, args.embedding_dim, False, args.angle_class, args.distance_reg, args.vertex_pred, args.refine)
    total_params = 0
    print("Name : Param.")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        print(f"{name} : {params}")
        total_params += params
    print(f"Total trainable params : {total_params}")
    # wrap_fp16_model(model)
    if args.modelf is not None:
        model.load_state_dict(torch.load(args.modelf, map_location='cuda:0'), strict=False)
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0])

    test(dataset, model, args, data_conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs/offset_local_dt_nearest_effnet_b4')
    
    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')
    parser.add_argument("--backbone", type=str, default='efficientnet-b4',
                        choices=['efficientnet-b0', 'efficientnet-b4', 'efficientnet-b7', 'resnet-18', 'resnet-50'])
    parser.add_argument("--nworkers", type=int, default=4)

    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--sample_dist", type=float, default=1.5) # 1.5
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=50)

    # embedding config
    parser.add_argument("--embedding_dim", type=int, default=16)

    # direction config
    parser.add_argument('--angle_class', type=int, default=36)
    
    # distance transform config
    parser.add_argument("--distance_reg", action='store_true')
    parser.add_argument("--dist_threshold", type=float, default=10.0)

    # vertex location classification config
    parser.add_argument("--vertex_pred", action='store_false')
    parser.add_argument("--cell_size", type=int, default=8)

    # positional encoding frequencies
    parser.add_argument("--pos_freq", type=int, default=10,
                        help="log2 of max freq for positional encoding (2D vertex location)")

    # vector refinement config
    parser.add_argument("--refine", action='store_true')

    # VectorMapNet config
    parser.add_argument("--num_vectors", type=int, default=400) # 100 * 3 classes = 300 in total
    parser.add_argument("--vertex_threshold", type=float, default=0.01)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=7)
    parser.add_argument("--sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--match_threshold", type=float, default=0.1)

    args = parser.parse_args()
    main(args)