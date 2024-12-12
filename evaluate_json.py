import torch
import tqdm
import csv

from evaluation.dataset import HDMapNetEvalDataset, VectorMapEvalDataset
from evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
from evaluation.AP import instance_mask_AP
from evaluation.iou import get_batch_iou

SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
THRESHOLDS = [0.5, 1.0, 1.5] # [0.2, 0.5, 1.0]? [2, 4, 6]


def get_val_info(args):
    data_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'thickness': args.thickness,
        'sample_dist': args.sample_dist, # 1.5
    }

    dataset = HDMapNetEvalDataset(args.version, args.dataroot, args.eval_set, args.result_path, data_conf)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bsz, shuffle=False, drop_last=False)

    total_CD1 = torch.zeros(args.max_channel).cuda()
    total_CD2 = torch.zeros(args.max_channel).cuda()
    total_CD_num1 = torch.zeros(args.max_channel).cuda()
    total_CD_num2 = torch.zeros(args.max_channel).cuda()
    total_intersect = torch.zeros(args.max_channel).cuda()
    total_union = torch.zeros(args.max_channel).cuda()
    AP_matrix = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()
    AP_count_matrix = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()

    # save metric for every frame
    results_list = []
    results_list.append(['Frame', 'IoU', 'CD_p', 'CD_l', 'CD', 'AP'])
    print('running eval...')
    for batchi, (pred_map, confidence_level, gt_map) in enumerate(tqdm.tqdm(data_loader)):
        # iou
        pred_map = pred_map.cuda()
        confidence_level = confidence_level.cuda()
        gt_map = gt_map.cuda()

        intersect, union = get_batch_iou(pred_map, gt_map)
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(pred_map, gt_map, args.xbound[2], args.ybound[2], threshold=args.CD_threshold)

        AP = instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, args.xbound[2], args.ybound[2],
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS)

        total_intersect += intersect.cuda()
        total_union += union.cuda()
        total_CD1 += CD1
        total_CD2 += CD2
        total_CD_num1 += num1
        total_CD_num2 += num2

        if args.bsz == 1:
            rec = data_loader.dataset.samples[batchi]
            scene_name = data_loader.dataset.nusc.get('scene', rec['scene_token'])['name']
            lidar_top_path = data_loader.dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
            frame_name = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0].split('_')[-1] # timestamp
            frame_name = scene_name + '_' + frame_name # {scene_name}_{timestamp}
            results_list.append([frame_name, (intersect/union).cpu().numpy(), (CD1/num1).cpu().numpy(), (CD2/num2).cpu().numpy(), ((CD1 + CD2) / (num1 + num2)).cpu().numpy(), AP.cpu().numpy()])

    if args.bsz == 1:
        filename = args.result_path.split('.')[0]
        with open(f'{filename}.csv', 'w') as f:
            print(f'saving number of vectors list to {filename}.csv...')
            write = csv.writer(f)
            write.writerows(results_list)
    
    CD_pred = total_CD1 / total_CD_num1
    CD_label = total_CD2 / total_CD_num2
    CD = (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2)
    CD_pred[CD_pred > args.CD_threshold] = args.CD_threshold
    CD_label[CD_label > args.CD_threshold] = args.CD_threshold
    CD[CD > args.CD_threshold] = args.CD_threshold
    return {
        'iou': total_intersect / total_union,
        'CD_pred': CD_pred,
        'CD_label': CD_label,
        'CD': CD,
        'Average_precision': AP_matrix / AP_count_matrix,
        'AP_class': torch.mean((AP_matrix / AP_count_matrix), 1),
        'AP_mean': torch.mean((AP_matrix / AP_count_matrix)),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result_path', type=str, default='vectormapnet_vector_softmax.json')
    parser.add_argument('--dataroot', type=str, default='./nuscenes')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='val', choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    parser.add_argument('--thickness', type=int, default=5)
    parser.add_argument('--max_channel', type=int, default=3)
    parser.add_argument('--CD_threshold', type=int, default=5)
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--sample_dist", type=float, default=1.5)

    args = parser.parse_args()

    print(get_val_info(args))
