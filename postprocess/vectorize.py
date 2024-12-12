import numpy as np
import torch
import torch.nn as nn

from .cluster import LaneNetPostProcessor
from .connect import sort_points_by_dist, connect_by_direction
from .connect import sort_indexed_points_by_dist, connect_by_adj_list


def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True) # [1, 200, 400]
    one_hot = logits.new_full(logits.shape, 0) # [4, 200, 400]
    one_hot.scatter_(dim, max_idx, 1) # [4, 200, 400]
    return one_hot


def onehot_encoding_spread(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-1, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-2, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+1, max=logits.shape[dim]-1), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+2, max=logits.shape[dim]-1), 1)

    return one_hot


def get_pred_top2_direction(direction, dim=1):
    direction = torch.softmax(direction, dim)
    idx1 = torch.argmax(direction, dim)
    idx1_onehot_spread = onehot_encoding_spread(direction, dim)
    idx1_onehot_spread = idx1_onehot_spread.bool()
    direction[idx1_onehot_spread] = 0
    idx2 = torch.argmax(direction, dim)
    direction = torch.stack([idx1, idx2], dim) - 1
    return direction


def vectorize(segmentation, embedding, direction, angle_class):
    segmentation = segmentation.softmax(0) # [4, 200, 400]
    embedding = embedding.cpu() # [16, 200, 400]
    direction = direction.permute(1, 2, 0).cpu() # [200, 400, 37]
    direction = get_pred_top2_direction(direction, dim=-1) # [200, 400, 2]

    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)

    oh_pred = onehot_encoding(segmentation).cpu().numpy() # [4, 200, 400]
    confidences = []
    line_types = []
    simplified_coords = []
    for i in range(1, oh_pred.shape[0]): # 1, 2, 3
        single_mask = oh_pred[i].astype('uint8') # [200, 400]
        single_embedding = embedding.permute(1, 2, 0) # [200, 400, 16]

        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedding) # [200, 400], [N, 2] 2: x, y
        if single_class_inst_mask is None:
            continue

        num_inst = len(single_class_inst_coords)

        prob = segmentation[i]
        prob[single_class_inst_mask == 0] = 0
        nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
        nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
        vertical_mask = avg_mask_1 > avg_mask_2
        horizontal_mask = ~vertical_mask
        nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

        for j in range(1, num_inst + 1):
            full_idx = np.where((single_class_inst_mask == j)) # [J], [J] row, col
            full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose() # [J, 2]
            confidence = prob[single_class_inst_mask == j].mean().item()

            idx = np.where(nms_mask & (single_class_inst_mask == j)) # [K], [K] row, col
            if len(idx[0]) == 0:
                continue
            lane_coordinate = np.vstack((idx[1], idx[0])).transpose() # [K, 2]

            range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
            range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
            if range_0 > range_1:
                lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
            else:
                lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])

            lane_coordinate = np.stack(lane_coordinate)
            lane_coordinate = sort_points_by_dist(lane_coordinate)
            lane_coordinate = lane_coordinate.astype('int32')
            lane_coordinate = connect_by_direction(lane_coordinate, direction, step=7, per_deg=360 / angle_class)

            simplified_coords.append(lane_coordinate)
            confidences.append(confidence)
            line_types.append(i-1)

    return simplified_coords, confidences, line_types

def vectorize_graph(positions: torch.Tensor, match: torch.Tensor, segmentation: torch.Tensor, mask: torch.Tensor, cls_threshold=0.5, match_threshold=0.1):
    """ Vectorize from graph representations

    Parameters
    ----------
    @ positions: [N, 2]
    @ match: [N+1, N+1]
    @ segmentation: [4, N] 
    @ mask: [N, 1] 
    @ patch_size: (30.0, 60.0)

    Returns:
    simplified_coords: [ins] list of [K, 2] poiny arrays
    confidences: [ins] list of float
    line_types: [ins] list of index
    """
    confidences = []
    line_types = []
    simplified_coords = []

    assert match.shape[0] == match.shape[1], f"match.shape[0]: {match.shape[0]} != match.shape[1]: {match.shape[1]}"
    assert positions.shape[0] == segmentation.shape[1] == mask.shape[0], f"Following shapes mismatch: positions.shape[0]({positions.shape[0]}), segmentation.shape[1]({segmentation.shape[1]}), mask.shape[0]({mask.shape[0]}"

    mask = mask.squeeze(-1).cpu() # [N]
    mask_bin = torch.cat([mask, mask.new_tensor(1).view(1)], 0) # [N+1]
    match = match.exp().cpu()[mask_bin == 1][:, mask_bin == 1] # [M+1, M+1]
    positions = positions.cpu().numpy()[mask == 1] # [M, 3]
    if match.shape[0] < 3:
        return simplified_coords, confidences, line_types
    adj_mat = match[:-1, :-1] > match_threshold # [M, M] for > threshold
    t2scores, t2indices = torch.topk(match[:-1, :-1], 2, -1) # [M, 2]? for top-2
    t2mat = match.new_full(match[:-1, :-1].shape, 0, dtype=torch.bool)
    t2mat = t2mat.scatter_(1, t2indices, 1) # [M, M]
    adj_mat = adj_mat & t2mat # [M, M]
    segmentation = segmentation.exp() # [3, N] # if NLLLoss: torch.sigmoid(segmentation)
    seg_onehot = onehot_encoding(segmentation).cpu()[:, mask == 1].numpy() # [3, M] 0, 1, 2
    segmentation = segmentation.cpu().numpy()[:, mask == 1] # [3, M]
    
    for i in range(seg_onehot.shape[0]): # 0, 1, 2
        single_mask = np.expand_dims(seg_onehot[i].astype('uint8'), 1) # [M, 1]
        single_match_mask = single_mask @ single_mask.T # [M, M] symmetric

        single_class_adj_list = torch.nonzero(adj_mat & single_match_mask).numpy() # [M', 2] symmetric single_class_adj_list[:, 0] -> single_class_adj_list[:, 1]
        if single_class_adj_list.shape[0] == 0:
            continue
        single_inst_adj_list = single_class_adj_list
        single_class_adj_score = match[single_class_adj_list[:, 0], single_class_adj_list[:, 1]].numpy() # [M'] confidence

        prob = segmentation[i] # [M,]

        while True:
            if single_inst_adj_list.shape[0] == 0:
                break

            cur, next = single_inst_adj_list[0] # cur -> next
            init_cur_idx, _ = np.where(single_inst_adj_list[:, :-1] == cur) # two or one
            single_inst_coords = np.expand_dims(positions[cur], 0) # [1, 2]
            single_inst_confidence = np.expand_dims(prob[cur], 0) # [1, 1] np array
            cur_taken = [cur]

            for ici in init_cur_idx: # one or two
                cur, next = single_inst_adj_list[0] # cur -> next
                single_inst_adj_list = np.delete(single_inst_adj_list, 0, 0)
                while True:
                    next_idx = np.where(single_inst_adj_list[:, :-1] == next)[0]
                    next_adj = single_inst_adj_list[next_idx, -1]
                    if cur not in cur_taken and cur in next_adj:
                        single_inst_coords = np.vstack((single_inst_coords, positions[cur])) # [num, 2]
                        single_inst_confidence = np.vstack((single_inst_confidence, prob[cur])) # [num, 1]
                        cur_taken.append(cur)
                    if cur == next:
                        break
                    cur = next
                    cur_idx, _ = np.where(single_inst_adj_list[:, :-1] == cur) # two or one
                    for ci in cur_idx:
                        next_candidate = single_inst_adj_list[ci, 1]
                        if next_candidate not in cur_taken:
                            next = next_candidate # update next
                    single_inst_adj_list = np.delete(single_inst_adj_list, cur_idx, 0)
                
                # reverse
                single_inst_coords = np.flipud(single_inst_coords)
                single_inst_confidence = np.flipud(single_inst_confidence)
                cur_taken.reverse()
            
            assert len(cur_taken) == len(single_inst_coords)
            
            if single_inst_coords.shape[0] < 2:
                continue
            
            simplified_coords.append(single_inst_coords) # [num, 2]
            confidences.append(single_inst_confidence.mean())
            line_types.append(i)
            
    return simplified_coords, confidences, line_types
