from tqdm import tqdm
import numpy as np
from functools import partial
import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.ShapeNetTest import *
from option.parse import parse
from model.loss.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.mmd_cov import minimum_matching_distance, coverage
from metrics.miou_shape import calc_miou
from model import define_net

category_list = ['Chair', 'Car', 'Lamp', 'Airplane', 'Table']
#test_data_root = '/orion/u/jiangthu/datasets/ShapeNet/test_data/'
#data_root = '/orion/u/jiangthu/projects/CycleConsistentDeformation/data/dataset_shapenet/'

nnd = chamfer_3DDist()
def calc_nnd(pc1, pc2, nnd):
    dist1, dist2, _, _ = nnd(pc1, pc2)
    return dist1.mean(1) + dist2.mean(1)
func_nnd = partial(calc_nnd, nnd=nnd)

class ShapeNetMIOUSeg(ShapeNetMIOU):
    def get_data(self):
        src_list = []
        tgt_list = []
        for index in range(self.num):
            src_idx = self.src_idx[index]
            src_shape = self.train_src_shapes[[src_idx[0]], src_idx[1], :]
            src_list.append(src_shape)
        src = np.concatenate(src_list, axis=0).reshape(10, -1, 2048, 4)
        return src, self.test_shapes

def read_cat2id_dict(path):
    with open(path) as f:
        lines = f.read().splitlines()
    cat2id_dict = {}
    for l in lines:
        cat, identity = l.split('\t')
        cat2id_dict[cat] = identity
    return cat2id_dict

def calc_miou_top(category, deform_array, label, test_data_root):
    # src_array: 10, B, 2048, 4
    # tgt_array: B, 2048, 4
    # deform_array: 10, B, 2048, 3
    # label: list
    ds = ShapeNetMIOUSeg(category, 'tf', root=test_data_root)
    src_array, tgt_array = ds.get_data()

    tgt_shape = torch.from_numpy(tgt_array).float()  # B, 2048, 4
    tgt_seg = tgt_shape[:, :, -1].long()
    tgt_shape = tgt_shape[:, :, :3].transpose(1, 2).contiguous().cuda()
    results = {'miou': []}
    for trial in range(src_array.shape[0]):
        src_shape = torch.from_numpy(src_array[trial]).float()  # B, 2048, 4
        src_seg = src_shape[:, :, -1].long()
        src_shape = src_shape[:, :, :3].transpose(1, 2).contiguous().cuda()

        deform_shape = torch.from_numpy(
            deform_array[trial]).float()  # B, 2048, 3
        deform_shape = deform_shape.transpose(1, 2).contiguous().cuda()
        miou_sum = 0.0
        for idx in range(src_array.shape[1]):
            miou = calc_miou(tgt_shape[[idx]],
                             deform_shape[[idx]],
                             src_seg[[idx]],
                             tgt_seg[[idx]],
                             label,
                             nnd,
                             return_CD=False)
            miou_sum += miou
        miou_sum /= src_array.shape[1]
        results['miou'].append(miou_sum)
    for k, v in results.items():
        results[k] = np.mean(v)
    return results['miou']

def calc_mmd_cov(ref_array, transfer_array):
    # ref_array: B, 2048, 3
    # transfer_array: 10, B, 2048, 3
    ref_set = torch.from_numpy(ref_array).float().cuda()
    MMD_CD = []
    COV_CD = []
    for trial in range(transfer_array.shape[0]):
        transfer_set = torch.from_numpy(transfer_array[trial]).float().cuda()
        mmd_cd, _, _ = minimum_matching_distance(ref_set, transfer_set, 100,
                                                 func_nnd)
        cov_cd, _ = coverage(ref_set, transfer_set, 100, func_nnd)
        MMD_CD.append(mmd_cd)
        COV_CD.append(cov_cd)
    return np.mean(MMD_CD), np.mean(COV_CD)

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option yaml file.')
    parser.add_argument('-test_data_root', type=str, help='Path to test data dir.')
    parser.add_argument('-data_root', type=str, help='Path to full data dir.')
    parser.add_argument('-out_dir', type=str, default=None, help='Output directory.')
    parser.add_argument('-load_path', type=str, default=None, help='Pretrained model path.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True, gpu0=True)
    test_data_root = args.test_data_root
    data_root = args.data_root
    category = opt['dataset']['train']['category']
    cat2id_dict = read_cat2id_dict(os.path.join(data_root, 'synsetoffset2category.txt'))
    
    if args.out_dir is None:
        result_root = '../result/ShapeNet/%s/%s' % (opt['model']['model_type'], category)
    else:
        result_root = args.out_dir
    print('Will save to %s' % result_root)
    if not os.path.isdir(result_root):
        os.makedirs(result_root)
    
    # get and load model
    torch.set_grad_enabled(False)
    model = define_net(opt['model'])
    if args.load_path is None:
        load_path = os.path.join(opt['path']['model'], 'best_model.pth'))
    else:
        load_path = args.load_path
    print('Loading from %s' % load_path)
    sd = torch.load(load_path)
    model.load_state_dict(sd)
    model = model.eval().cuda()
    
    # Fitting
    deform_shape_list = []
    ds = ShapeNetCD(category, 'torch', root=test_data_root)
    tgt_shape_list = []
    for src, tgt in tqdm(ds, total=len(ds)):
        src_shape = src.cuda()
        tgt_shape = tgt.cuda()
        tgt_shape_list.append(tgt_shape)
        deform_shape = model(src_shape, tgt_shape)
        deform_shape_list.append(deform_shape)

    tgt_shapes = torch.cat(tgt_shape_list, dim=0).transpose(1, 2)

    deform_shapes = torch.cat(deform_shape_list, dim=0).transpose(1, 2)
    cd = func_nnd(tgt_shapes, deform_shapes)
    print('Fitting Chamfer Distance: %.3e' % cd.mean().item())
    deform_shapes = deform_shapes.cpu().numpy()
    np.save(os.path.join(result_root, 'cd_deform_shapes.npy'), deform_shapes)
    print('Saved cd_deform_shapes(' + str(deform_shapes.shape) + ') to', os.path.join(result_root, 'cd_deform_shapes.npy'))


    # MIOU
    deform_shape_list = []
    ds = ShapeNetMIOU(category, 'torch', root=test_data_root)
    for src, tgt in tqdm(ds, total=len(ds)):
        src_shape = src.cuda()
        tgt_shape = tgt.cuda()
        deform_shape = model(src_shape, tgt_shape)
        deform_shape_list.append(deform_shape.cpu())

    cat_id = cat2id_dict[category]
    with open(os.path.join(data_root, '%s_label.json' % cat_id)) as f:
        label = json.load(f)
    deform_shapes = torch.cat(deform_shape_list, dim=0).transpose(1, 2).view(10, -1, 2048, 3).numpy()
    miou = calc_miou_top(category, deform_shapes, label, test_data_root)
    print('mIoU: %.3f' % miou)
    np.save(os.path.join(result_root, 'miou_deform_shapes.npy'), deform_shapes)
    print('Saved miou_deform_shapes(' + str(deform_shapes.shape) + ') to', os.path.join(result_root, 'miou_deform_shapes.npy'))
    
    
    # Transfer
    transfer_shape_list = []
    deform_shape_list = []
    ds = ShapeNetTransfer(category, 'torch', root=test_data_root)
    for src, tgt, new in tqdm(ds, total=len(ds)):
        src_shape = src.cuda()
        tgt_shape = tgt.cuda()
        new_shape = new.cuda()
        deform_shape, transfer_shape = model.deformation_transfer(src_shape, tgt_shape, new_shape)
        transfer_shape_list.append(transfer_shape.cpu())
        deform_shape_list.append(deform_shape.cpu())
        
    ref_path = os.path.join(test_data_root, category, 'ref_shapes.npy')
    ref_shapes = np.load(ref_path)
    
    deform_shapes = torch.cat(deform_shape_list, dim=0).transpose(1, 2).view(10, -1, 2048, 3).numpy()
    transfer_shapes = torch.cat(transfer_shape_list, dim=0).transpose(1, 2).view(10, -1, 2048, 3).numpy()
    
    MMD_CD, COV_CD = calc_mmd_cov(ref_shapes, transfer_shapes)
    print('MMD-CD: %.3e; Cov-CD %.3f' % (MMD_CD, COV_CD))
    np.save(os.path.join(result_root, 'transfer_deform_shapes.npy'), deform_shapes)
    np.save(os.path.join(result_root, 'transfer_shapes.npy'), transfer_shapes)
    print('Saved transfer_shapes(%s), transfer_deform_shapes(%s) to %s' % (str(transfer_shapes.shape), str(deform_shapes.shape), os.path.join(result_root, 'transfer_(deform_)shapes.npy')))
    
    # Parallelogram
    deform1_shape_list = []
    deform2_shape_list = []
    transfer12_shape_list = []
    transfer21_shape_list = []
    ds = ShapeNetParallel(category, 'torch', root=test_data_root)
    for src, tgt1, tgt2 in tqdm(ds, total=len(ds)):
        src_shape = src.cuda()
        tgt1_shape = tgt1.cuda()
        tgt2_shape = tgt2.cuda()
        deform1_shape, transfer12_shape = model.deformation_transfer(src_shape, tgt1_shape, tgt2_shape)
        deform2_shape, transfer21_shape = model.deformation_transfer(src_shape, tgt2_shape, tgt1_shape)
        deform1_shape_list.append(deform1_shape.cpu())
        deform2_shape_list.append(deform2_shape.cpu())
        transfer12_shape_list.append(transfer12_shape.cpu())
        transfer21_shape_list.append(transfer21_shape.cpu())
        

    deform1_shapes = torch.cat(deform1_shape_list, dim=0).transpose(1, 2)
    deform2_shapes = torch.cat(deform2_shape_list, dim=0).transpose(1, 2)
    transfer12_shapes = torch.cat(transfer12_shape_list, dim=0).transpose(1, 2)
    transfer21_shapes = torch.cat(transfer21_shape_list, dim=0).transpose(1, 2)
    parallel_cd = func_nnd(transfer12_shapes.cuda(), transfer21_shapes.cuda())
    print('Parallelogram Chamfer distance: %.3e' % parallel_cd.mean().item())
    
    np.save(os.path.join(result_root, 'parallel_deform1_shapes.npy'), deform1_shapes.numpy())
    np.save(os.path.join(result_root, 'parallel_deform2_shapes.npy'), deform2_shapes.numpy())
    np.save(os.path.join(result_root, 'parallel_transfer12_shapes.npy'), transfer12_shapes.numpy())
    np.save(os.path.join(result_root, 'parallel_transfer21_shapes.npy'), transfer21_shapes.numpy())
    print('Saved paraller deform/transfer shapes (%s/%s) to %s' % (deform1_shapes.shape, transfer12_shapes.shape, os.path.join(result_root, 'parallel_deform/transfer1/2_shapes.npy')))
        
if __name__ == '__main__':
    main()
