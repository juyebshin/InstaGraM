import argparse
import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision
from tensorboardX import SummaryWriter

from data.dataset import semantic_dataset, vectormap_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou, get_batch_cd
from model import get_model
from data.visualize import colorise, colors_plt
from data.image import denormalize_img
from loss import GraphLoss, gen_dx_bx

def onehot_encoding(logits, dim=1):
    # logits: b, C, 200, 400
    max_idx = torch.argmax(logits, dim, keepdim=True) # b, 1, 200, 400
    one_hot = logits.new_full(logits.shape, 0) # zeros b, C, 200, 400
    one_hot.scatter_(dim, max_idx, 1) # b, C, 200, 400 one hot
    return one_hot

def visualize(writer: SummaryWriter, title, imgs: torch.Tensor, dt_mask: torch.Tensor, vt_mask: torch.Tensor, 
                vectors_gt: list, matches_gt: torch.Tensor, semantics_gt: torch.Tensor, 
                dt: torch.Tensor, heatmap: torch.Tensor, matches: torch.Tensor, positions: torch.Tensor, semantics: torch.Tensor, 
                masks: torch.Tensor, step: int, args):
    if writer is None:
        return
    # imgs: b, 6, 3, 128, 352
    # dt: b, 3, 200, 400 tensor
    # heatmap: b, 65, 25, 50 tensor
    imgs = imgs.detach().cpu().float()[0] # 6, 3, 128, 352
    # imgs = imgs.fliplr()
    imgs[3:] = torch.flip(imgs[3:], [3,])
    # imgs = torch.index_select(imgs, 0, torch.LongTensor([0, 1, 2, 5, 4, 3]))
    imgs_grid = torchvision.utils.make_grid(imgs, nrow=3) # 3, 262, 1064
    imgs_grid = np.array(denormalize_img(imgs_grid)) # 262, 1064, 3
    writer.add_image(f'{title}/images', imgs_grid, step, dataformats='HWC')

    dx, bx, nx = gen_dx_bx(args.xbound, args.ybound)


    if dt is not None:
        dt = dt.detach().cpu().float().numpy()
        dt_mask = dt_mask.detach().cpu().float().numpy()
        writer.add_image(f'{title}/distance_transform_gt', colorise(dt_mask[0], 'magma'), step, dataformats='NHWC')
        writer.add_image(f'{title}/distance_transform_pred', colorise(dt[0], 'magma'), step, dataformats='NHWC')
    
    if heatmap is not None:
        # vertex = onehot_encoding(heatmap)
        heatmap = heatmap.detach().cpu().float().numpy()[0] # 65, 25, 50
        # vertex = vertex.detach().cpu().float().numpy()[0] # 65, 25, 50, onehot
        vt_mask = vt_mask.detach().cpu().float().numpy()[0] # 65, 25, 50

        nodust_gt = vt_mask[:-1, :, :] # 64, 25, 50
        Hc, Wc = vt_mask.shape[1:] # 25, 50
        nodust_gt = nodust_gt.transpose(1, 2, 0) # 25, 50, 64
        heatmap_gt = np.reshape(nodust_gt, [Hc, Wc, args.cell_size, args.cell_size]) # 25, 50, 8, 8
        heatmap_gt = np.transpose(heatmap_gt, [0, 2, 1, 3]) # 25, 8, 50, 8
        heatmap_gt = np.reshape(heatmap_gt, [Hc*args.cell_size, Wc*args.cell_size]) # 200, 400

        nodust = heatmap[:-1, :, :] # 64, 25, 50
        nodust = nodust.transpose(1, 2, 0) # 25, 50, 64
        heatmap = np.reshape(nodust, [Hc, Wc, args.cell_size, args.cell_size]) # 25, 50, 8, 8
        heatmap = np.transpose(heatmap, [0, 2, 1, 3]) # 25, 8, 50, 8
        heatmap = np.reshape(heatmap, [Hc*args.cell_size, Wc*args.cell_size]) # 200, 400

        writer.add_image(f'{title}/vertex_heatmap_gt', colorise(heatmap_gt, 'hot', 0.0, 1.0), step, dataformats='HWC')
        heatmap[heatmap < args.vertex_threshold] = 0.0
        heatmap[heatmap > 0.0] = 1.0
        writer.add_image(f'{title}/vertex_heatmap_bin', colorise(heatmap, 'hot', 0.0, 1.0), step, dataformats='HWC')
    
    if matches is not None and positions is not None and masks is not None:
        # matches: [b, N+1, N+1]
        # positions: [b, N, 2], x y
        # masks: [b, N, 1]
        # attentions: [b, L, H, N, N] L: 7 layers, H: 4 heads
        # vectors_gt: [b] list of [instance] list of dict
        # matches_gt: [b, N, N+1]
        # xbound: []
        # ybound: []
        matches = matches[0].detach().cpu().float().numpy() # [N+1, N+1]
        positions = positions[0].detach().cpu().float().numpy() # [N, 3]
        masks = masks[0].detach().cpu().int().numpy().squeeze(-1) # [N]
        # attentions = attentions[0].detach().cpu().float().numpy()[-1] # [H, N, N] attention of the last layer
        masks_bins = np.concatenate([masks, [1]], 0) # [N + 1]
        vectors_gt = vectors_gt[0]
        matches_gt = matches_gt.detach().cpu().float().numpy()[0] # [N+1, N+1]
        positions = positions * dx + bx # [30, 60]
        positions_valid = positions[masks == 1] # [M, 3]

        # Vector prediction
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')
        # plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=0.5, c=positions_valid[:, 2], cmap='jet', vmin=0.0, vmax=1.0)

        matches = matches[masks_bins == 1][:, masks_bins == 1] # masked [M+1, M+1]
        matches_nodust = matches[:-1, :-1] # [M, M]
        rows, cols = np.where(matches_nodust > args.match_threshold)
        for row, col in zip(rows, cols):
            plt.plot([positions_valid[row, 0], positions_valid[col, 0]], [positions_valid[row, 1], positions_valid[col, 1]], 'o-', c=cm.jet(matches_nodust[row, col]), linewidth=0.5, markersize=1.0)
        # matches_idx = matches.argmax(1) if len(matches) > 0 else None # [M, ]
        # for i, pos in enumerate(positions_valid): # [3,]
        #     if matches_idx is not None:
        #         match = matches_idx[i]
        #         if matches[i, match] > 0.1 and match < len(matches)-1: # 0.8 too high?
        #             # plt.plot([pos[0], positions_valid[match][0]], [pos[1], positions_valid[match][1]], '-', color=colorise(matches[i, match], 'jet', 0.0, 1.0))
        #             plt.quiver(pos[0], pos[1], positions_valid[match][0] - pos[0], positions_valid[match][1] - pos[1], 
        #                        color=colorise(matches[i, match], 'jet', 0.0, 1.0), scale_units='xy', angles='xy', scale=1)
        
        writer.add_figure(f'{title}/vector_pred', fig, step)
        plt.close()

        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')

        for vector in vectors_gt:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = pts[:pts_num]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])
        
        writer.add_figure(f'{title}/vector_gt', fig, step)
        plt.close()

        # Match prediction
        fig = plt.figure()
        plt.grid(False)
        plt.imshow(matches, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0) # [M, M]
        plt.colorbar()
        writer.add_figure(f'{title}/match_pred', fig, step)
        plt.close()

        # Aligned GT matches
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')

        matches_gt = matches_gt[masks_bins == 1][:, masks_bins == 1] # [M+1, M+1]
        matches_idx = matches_gt.argmax(1) if len(matches_gt) > 0 else None # [M, ]
        
        plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=0.5, c='b', vmin=0.0, vmax=1.0)
        for i, pos in enumerate(positions_valid): # [3,]
            if matches_idx is not None:
                match = matches_idx[i]
                if matches_gt[i, match] == 1.0 and match < len(matches_gt)-1: # less then N
                    # plt.plot([pos[0], positions_valid[match][0]], [pos[1], positions_valid[match][1]], '-', color=colorise(matches_gt[i, match], 'jet', 0.0, 1.0))
                    plt.quiver(pos[0], pos[1], positions_valid[match][0] - pos[0], positions_valid[match][1] - pos[1], 
                               color=cm.jet(matches_gt[i, match]), scale_units='xy', angles='xy', scale=1)
        
        writer.add_figure(f'{title}/match_aligned', fig, step)
        plt.close()

        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')

        # Match gt
        fig = plt.figure()
        plt.grid(False)
        plt.imshow(matches_gt, cmap='hot', interpolation='nearest') # [M, M]
        plt.colorbar()
        writer.add_figure(f'{title}/match_gt', fig, step)
        plt.close()
    
    if semantics is not None and semantics_gt is not None:
        # semantics: [b, 3, N] 0: divider, 1: ped-crossing, 2: boundary
        # semantics_gt: [b, 3, N]
        semantic = semantics[0].exp().detach().cpu().float().numpy() # [3, N]
        semantic_gt = semantics_gt[0].detach().cpu().float().numpy().astype('uint8') # [3, N]
        
        semantic_onehot = semantic.argmax(0)[masks == 1] # [M]
        semantic_gt_onehot = semantic_gt.argmax(0)[masks == 1] # [M]

        # Semantic prediction
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')
        plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=1.0, c=[colors_plt[c] for c in semantic_onehot])
        
        writer.add_figure(f'{title}/vector_semantic_pred', fig, step)
        plt.close()
        
        # Semantic gt
        fig = plt.figure()
        plt.xlim(args.xbound[0], args.xbound[1])
        plt.ylim(args.ybound[0], args.ybound[1])
        plt.axis('off')
        plt.scatter(positions_valid[:, 0], positions_valid[:, 1], s=1.0, c=[colors_plt[c] for c in semantic_gt_onehot])
        
        writer.add_figure(f'{title}/vector_semantic_gt', fig, step)
        plt.close()



def eval_iou(model, val_loader, args, writer=None, step=None, vis_interval=0, is_master=False):
    # st
    graph_loss_fn = GraphLoss(args.xbound, args.ybound, num_classes=NUM_CLASSES).cuda()

    model.eval()
    counter = 0
    total_intersects = 0
    total_union = 0
    total_cdist_p = 0.0
    total_cdist_l = 0.0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, distance_gt, vertex_gt, vectors_gt in tqdm.tqdm(val_loader):

            semantic, distance, vertex, embedding, direction, matches, positions, masks = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            heatmap = vertex.softmax(1) # b, 65, 25, 50
            matches = matches.exp() # b, N+1, N+1
            vertex_gt = vertex_gt.cuda().float() # b, 65, 25, 50
            intersects, union = get_batch_iou(onehot_encoding(heatmap), vertex_gt)
            total_intersects += intersects
            total_union += union

            cdist_p, cdist_l = get_batch_cd(positions, vectors_gt, masks, args.xbound, args.ybound)
            total_cdist_p += cdist_p
            total_cdist_l += cdist_l

            _, _, seg_loss, matches_gt, vector_semantics_gt = graph_loss_fn(matches, positions, semantic, masks, vectors_gt)

            if writer is not None and vis_interval > 0:
                if counter % vis_interval == 0 and is_master:                
                        if args.distance_reg:
                            distance = distance.relu().clamp(max=args.dist_threshold).cuda() # b, 3, 200, 400
                        heatmap_onehot = onehot_encoding(heatmap)
                        visualize(writer, 'eval', imgs, distance_gt, vertex_gt, vectors_gt, matches_gt, vector_semantics_gt, distance, heatmap, matches, positions, semantic, masks, step, args)
            
            counter += 1

    total_cdist_p, total_cdist_l = float(total_cdist_p/counter), float(total_cdist_l/counter)
    total_cdist = float((total_cdist_p + total_cdist_l)*0.5)
    print(f'CD_p: {total_cdist_p:.4f}, CD_l: {total_cdist_l:.4f}, CD: {total_cdist:.4f}')
    return total_intersects / (total_union + 1e-7), total_cdist


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
        'match_threshold': args.match_threshold, # 0.2
    }

    train_loader, val_loader = vectormap_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.segmentation, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class, args.distance_reg, args.vertex_pred, args.refine)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    print(eval_iou(model, val_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs/instagram_lss_eff_b4_128x352_30e')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='LSS')
    parser.add_argument("--backbone", type=str, default='efficientnet-b4',
                        choices=['efficientnet-b0', 'efficientnet-b4', 'efficientnet-b7', 'resnet-18', 'resnet-50'])

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default='./runs/instagram_lss_eff_b4_128x352_30e/model_best.pt')

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--sample_dist", type=float, default=1.5)

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    parser.add_argument("--scale_dt", type=float, default=1.0)
    parser.add_argument("--scale_cdist", type=float, default=1.0, 
                        help="Scale of Chamfer distance loss")
    parser.add_argument("--scale_match", type=float, default=1.0, 
                        help="Scale of matching loss")

    # distance transform config
    parser.add_argument("--distance_reg", action='store_true')
    parser.add_argument("--dist_threshold", type=float, default=10.0)

    # vertex location classification config
    parser.add_argument("--vertex_pred", action='store_false') # defaults to true
    parser.add_argument("--cell_size", type=int, default=8)

    # positional encoding frequencies
    parser.add_argument("--pos_freq", type=int, default=10,
                        help="log2 of max freq for positional encoding (2D vertex location)")

    # semantic segmentation config
    parser.add_argument("--segmentation", action='store_true')

    # vector refinement config
    parser.add_argument("--refine", action='store_true')

    # InstaGraM config
    parser.add_argument("--num_vectors", type=int, default=300) # 100 * 3 classes = 300 in total
    parser.add_argument("--vertex_threshold", type=float, default=0.01)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=7)
    parser.add_argument("--sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--match_threshold", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
