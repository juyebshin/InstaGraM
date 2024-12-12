import math
from operator import index
import random
import numpy as np
from copy import deepcopy

import torch


def sort_points_by_dist(coords):
    coords = coords.astype('float')
    num_points = coords.shape[0] # [num_pt]
    diff_matrix = np.repeat(coords[:, None], num_points, 1) - coords # [num_pt, num_pt, 2]
    # x_range = np.max(np.abs(diff_matrix[..., 0]))
    # y_range = np.max(np.abs(diff_matrix[..., 1]))
    # diff_matrix[..., 1] *= x_range / y_range
    dist_matrix = np.sqrt(((diff_matrix) ** 2).sum(-1))
    dist_matrix_full = deepcopy(dist_matrix)
    direction_matrix = diff_matrix / (dist_matrix.reshape(num_points, num_points, 1) + 1e-6)

    sorted_points = [coords[0]]
    sorted_indices = [0]
    dist_matrix[:, 0] = np.inf

    last_direction = np.array([0, 0])
    for i in range(num_points - 1):
        last_idx = sorted_indices[-1]
        dist_metric = dist_matrix[last_idx] - 0 * (last_direction * direction_matrix[last_idx]).sum(-1)
        idx = np.argmin(dist_metric) % num_points
        new_direction = direction_matrix[last_idx, idx]
        if dist_metric[idx] > 3 and min(dist_matrix_full[idx][sorted_indices]) < 5:
            dist_matrix[:, idx] = np.inf
            continue
        if dist_metric[idx] > 10 and i > num_points * 0.9:
            break
        sorted_points.append(coords[idx])
        sorted_indices.append(idx)
        dist_matrix[:, idx] = np.inf
        last_direction = new_direction

    return np.stack(sorted_points, 0)

def sort_indexed_points_by_dist(indexed_coords):
    coords, indices = indexed_coords[:, :-1], indexed_coords[:, -1]
    coords = coords.astype('float')
    num_points = coords.shape[0] # [num_pt]
    diff_matrix = np.repeat(coords[:, None], num_points, 1) - coords # [num_pt, num_pt, 2]
    # x_range = np.max(np.abs(diff_matrix[..., 0]))
    # y_range = np.max(np.abs(diff_matrix[..., 1]))
    # diff_matrix[..., 1] *= x_range / y_range
    dist_matrix = np.sqrt(((diff_matrix) ** 2).sum(-1))
    dist_matrix_full = deepcopy(dist_matrix)
    direction_matrix = diff_matrix / (dist_matrix.reshape(num_points, num_points, 1) + 1e-6)

    sorted_points = [indexed_coords[0]]
    sorted_indices = [0]
    dist_matrix[:, 0] = np.inf

    last_direction = np.array([0, 0])
    for i in range(num_points - 1):
        last_idx = sorted_indices[-1]
        dist_metric = dist_matrix[last_idx] - 0 * (last_direction * direction_matrix[last_idx]).sum(-1)
        idx = np.argmin(dist_metric) % num_points
        new_direction = direction_matrix[last_idx, idx]
        if dist_metric[idx] > 3 and min(dist_matrix_full[idx][sorted_indices]) < 5:
            dist_matrix[:, idx] = np.inf
            continue
        if dist_metric[idx] > 10 and i > num_points * 0.9:
            break
        sorted_points.append(indexed_coords[idx])
        sorted_indices.append(idx)
        dist_matrix[:, idx] = np.inf
        last_direction = new_direction

    return np.stack(sorted_points, 0)
    

def connect_by_step(coords, direction_mask, sorted_points, taken_direction, step=5, per_deg=10):
    # direction_mask: [200, 400, 2]
    # sorted_points: [2]
    # taken_direction: [200, 400, 2]
    while True:
        last_point = tuple(np.flip(sorted_points[-1]))
        if not taken_direction[last_point][0]:
            direction = direction_mask[last_point][0]
            taken_direction[last_point][0] = True
        elif not taken_direction[last_point][1]:
            direction = direction_mask[last_point][1]
            taken_direction[last_point][1] = True
        else:
            break

        if direction == -1:
            continue

        deg = per_deg * direction
        vector_to_target = step * np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
        last_point = deepcopy(sorted_points[-1])

        # NMS
        coords = coords[np.linalg.norm(coords - last_point, axis=-1) > step-1]

        if len(coords) == 0:
            break

        target_point = np.array([last_point[0] + vector_to_target[0], last_point[1] + vector_to_target[1]])
        dist_metric = np.linalg.norm(coords - target_point, axis=-1)
        idx = np.argmin(dist_metric)

        if dist_metric[idx] > 50:
           continue

        sorted_points.append(deepcopy(coords[idx]))

        vector_to_next = coords[idx] - last_point
        deg = np.rad2deg(math.atan2(vector_to_next[1], vector_to_next[0]))
        inverse_deg = (180 + deg) % 360
        target_direction = per_deg * direction_mask[tuple(np.flip(sorted_points[-1]))] # [2]
        tmp = np.abs(target_direction - inverse_deg)
        tmp = torch.min(tmp, 360 - tmp)
        taken = np.argmin(tmp)
        taken_direction[tuple(np.flip(sorted_points[-1]))][taken] = True

def connect_by_adj(indexed_coords, adj_list, adj_score, sorted_points, taken):
    while True:
        last_point = sorted_points[-1] # x y index
        last_point_idx = np.where(indexed_coords[:, -1] == last_point[-1])[0]
        if last_point[-1] not in taken:
            taken.append(last_point[-1]) # vector index
        else:
            break

        if last_point_idx >= (len(indexed_coords) - 1): # end of indexed_coords
            break
        
        cur_idx, _ = np.where(adj_list[:, :-1] == last_point[-1])
        last_point_adj_list = adj_list[cur_idx, -1]

        next_point = indexed_coords[last_point_idx + 1].squeeze(0)
        next_idx, _ = np.where(adj_list[:, :-1] == next_point[-1])
        next_point_adj_list = adj_list[next_idx, -1]

        # no connection in neither direction
        if next_point[-1] not in last_point_adj_list and last_point[-1] not in next_point_adj_list:
            valid_next = -1
            last_point_adj_score = adj_score[cur_idx]
            max_conf = 0.0
            for candidate, confidence in zip(last_point_adj_list, last_point_adj_score):
                if candidate not in taken:
                    if confidence > max_conf:
                        max_conf = confidence
                        valid_next = candidate
            if valid_next == -1:
                # for candidate in next_point_adj_list:
                #     if candidate not in taken:
                #         valid_next = candidate
                #         break
                # if valid_next == -1:
                #     break
                sorted_points.append(deepcopy(next_point))
                break
            next_point_idx = np.where(indexed_coords[:, -1] == valid_next)[0]
            if not next_point_idx.shape[0]:
                break
            next_point = indexed_coords[np.where(indexed_coords[:, -1] == valid_next)[0]].squeeze(0)
            sorted_points.append(deepcopy(next_point))
            continue
        # # only last_point -> next_point
        # elif last_point[-1] not in next_point_adj_list:
        #     break
        # # only next_point -> last_point
        # elif next_point[-1] not in last_point_adj_list:
        #     break
        # last_point and next_point have bidirectional connections
        else:
            sorted_points.append(deepcopy(next_point))
            continue



def connect_by_direction(coords, direction_mask, step=5, per_deg=10):
    sorted_points = [deepcopy(coords[random.randint(0, coords.shape[0]-1)])]
    taken_direction = np.zeros_like(direction_mask, dtype=np.bool) # [200, 400, 2]

    connect_by_step(coords, direction_mask, sorted_points, taken_direction, step, per_deg)
    sorted_points.reverse()
    connect_by_step(coords, direction_mask, sorted_points, taken_direction, step, per_deg)
    return np.stack(sorted_points, 0)

def connect_by_adj_list(indexed_coords, adj_list, adj_score):
    # indexed_coords: [N, 3]
    # adj_list: [M, 2]

    # dist = np.linalg.norm(indexed_coords[:-1, :-1] - indexed_coords[1:, :-1], axis=1)
    # indices = np.where(dist > 10.0)[0]
    # vector_indices = indexed_coords[indices, -1]
    # del_count = 0
    # for idx, vi in zip(indices, vector_indices):
    #     a1, a2 = np.where(adj_list[:, :-1] == vi)
    #     idx = idx - del_count
    #     check = np.array([indexed_coords[idx, -1], indexed_coords[idx+1, -1]])
    #     if check[-1] in adj_list[a1, -1]: # connection is valid
    #         continue
    #     valid_connect = adj_list[a1, -1]
    #     if valid_connect.shape[0] < 2 and idx >= 0: # this vector is outlier
    #         indexed_coords = np.delete(indexed_coords, idx, 0)
    #         del_count += 1
    #     elif idx > 0:
    #         indexed_coords[[idx-1, idx]] = indexed_coords[[idx, idx-1]]
    
    sorted_points = [deepcopy(indexed_coords[random.randint(0, indexed_coords.shape[0]-1)])]
    taken = []

    connect_by_adj(indexed_coords, adj_list, adj_score, sorted_points, taken)
    sorted_points.reverse()
    indexed_coords = np.flip(indexed_coords, 0)
    taken = []
    connect_by_adj(indexed_coords, adj_list, adj_score, sorted_points, taken)
    return np.stack(sorted_points, 0)
