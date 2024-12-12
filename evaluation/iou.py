import torch
import numpy as np

from loss import gen_dx_bx


def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)

def get_batch_cd(pred_positions: torch.Tensor, gt_vectors: list, masks: torch.Tensor, xbound: list, ybound: list, nonsense: float = 5.0):
    # pred_positions: [b, N, 2]
    # gt_vectors: [b] list of [instance] list of dict
    # masks: [b, N, 1]

    dx, bx, nx = gen_dx_bx(xbound, ybound)

    cdist_p_list = []
    cdist_l_list = []
    with torch.no_grad():
        for pred_position, gt_vector, mask in zip(pred_positions, gt_vectors, masks):
            # pred_position: [N, 3]
            # gt_vector: [instance] list of dict
            mask = mask.squeeze(-1) # [N]
            position_valid = pred_position * torch.tensor(dx).cuda() + torch.tensor(bx).cuda() # de-normalize, [N, 2]
            position_valid = position_valid[mask == 1] # [M, 2] x, y c
            pts_list = []
            for ins, vector in enumerate(gt_vector): # dict
                pts, pts_num, type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num] # [p, 2] array
                [pts_list.append(pt) for pt in pts]
            
            gt_position = torch.tensor(np.array(pts_list)).float().cuda()

            if len(gt_position) > 0 and len(position_valid) > 0:
                # compute chamfer distance # [N, P] shaped tensor
                cdist = torch.cdist(position_valid, gt_position) # [M, P] prediction to label
                # nearest ground truth vectors
                cdist_p_mean = torch.mean(cdist.min(dim=-1).values) # [N,]
                cdist_l_mean = torch.mean(cdist.min(dim=0).values) # [P,]
            else:
                cdist_p_mean = torch.tensor(nonsense).float().cuda()
                cdist_l_mean = torch.tensor(nonsense).float().cuda()
            
            cdist_p_list.append(cdist_p_mean)
            cdist_l_list.append(cdist_l_mean)
        
        batch_cdist_p = torch.stack(cdist_p_list) # [b,]
        batch_cdist_l = torch.stack(cdist_l_list) # [b,]
        mask_p = batch_cdist_p != -1.0
        mask_l = batch_cdist_l != -1.0

        batch_cdist_p_mean = (batch_cdist_p*mask_p).sum(dim=0) / mask_p.sum(dim=0)
        batch_cdist_l_mean = (batch_cdist_l*mask_l).sum(dim=0) / mask_l.sum(dim=0)
    return batch_cdist_p_mean, batch_cdist_l_mean

def get_batch_vector_iou(pred_vectors: torch.Tensor, matches: torch.Tensor, masks: torch.Tensor, gt_map: torch.Tensor, thickness: int = 5):
    return None
