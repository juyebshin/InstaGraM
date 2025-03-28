import cv2
import numpy as np

import torch

from shapely import affinity
from shapely.geometry import LineString, box


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2)) # (N, 2), cols, rows
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    elif type == 'vertex':
        for coord in coords:
            cv2.circle(mask, coord, radius=0, color=idx, thickness=-1)
        # mask[ [coord[1] for coord in coords], [coord[0] for coord in coords] ] = 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            # new_line: vectors in BEV pixel coordinate
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, patch_size, canvas_size, num_classes, thickness, angle_class, cell_size=8):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    distance_masks = []
    vertex_masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)
        distance_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, 1, 1)
        distance_masks.append(distance_mask)
        vertex_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, 1, 1, type='vertex')
        vertex_masks.append(vertex_mask)

    # canvas_size: tuple (int, int)
    # vertex_masks: 3, 200, 400
    vertex_masks = np.stack(vertex_masks)
    vertex_masks = vertex_masks.max(0)
    H, W = vertex_mask.shape # 200, 400
    Hc, Wc = int(H/cell_size), int(W/cell_size)
    vertex_masks = np.reshape(vertex_masks, [Hc, cell_size, Wc, cell_size]) # Hc, 8, Wc, 8
    vertex_masks = np.transpose(vertex_masks, [0, 2, 1, 3]) # Hc, Wc, 8, 8
    vertex_masks = np.reshape(vertex_masks, [Hc, Wc, cell_size*cell_size]) # Hc, Wc, 64
    vertex_masks = vertex_masks.transpose(2, 0, 1) # 64, Hc, Wc
    vertex_sum = vertex_masks.sum(0) # number of vertex in each cell, [Hc, Wc]
    # find cell with more then one vertex
    rows, cols = np.where(vertex_sum > 1)
    # N == len(rows) == len(cols)
    if len(rows):
        multi_vertex = vertex_masks[:, [row for row in rows], [col for col in cols]].transpose(1, 0) # N, 64
        index, depth = np.where(multi_vertex > 0)
        nums_multi_vertex = np.histogram(index, bins=len(rows), range=(0, len(rows)))[0]
        select = np.random.randint(nums_multi_vertex)
        nums_cum = np.insert(np.cumsum(nums_multi_vertex[:-1]), 0, 0)
        select_cum = select + nums_cum
        remove_index = np.delete(index, select_cum)
        remove_depth = np.delete(depth, select_cum)
        multi_vertex[[i for i in remove_index], [d for d in remove_depth]] = 0
        vertex_masks[:, [row for row in rows], [col for col in cols]] = multi_vertex.transpose(1, 0)
        vertex_sum = vertex_masks.sum(0) # number of vertex in each cell, Hc, Wc
    assert np.max(vertex_sum) <= 1, f"max(vertex_sum) expected less than 1, but got: {np.max(vertex_sum)}" # make sure one vertex per cell
    # randomly select one vertex and remove all others
    dust = np.zeros_like(vertex_sum, dtype='uint8') # Hc, Wc
    dust[vertex_sum == 0] = 1
    dust = np.expand_dims(dust, axis=0) # 1, Hc, Wc
    vertex_masks = np.concatenate((vertex_masks, dust), axis=0) # 65, Hc, Wc
    assert np.min(vertex_masks.sum(0)) == 1

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)
    distance_masks = np.stack(distance_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    distance_masks = distance_masks != 0

    return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks), distance_masks, torch.tensor(vertex_masks)


def rasterize_map(vectors, patch_size, canvas_size, num_classes, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
