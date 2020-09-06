from tqdm import tqdm
import numpy as np
from functools import partial
import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.ComplementMeTest import *
from option.parse import parse
from model.loss.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.mmd_cov import minimum_matching_distance, coverage
from model import define_net

category_list = ['Chair', 'Car', 'Sofa', 'Airplane', 'Table']

def project(point, mat):
    # point: 1, 3, N
    # mat: 1, 3N, 3N
    flat = point.transpose(1, 2).contiguous().view(1, -1, 1)
    projected = torch.bmm(mat, flat)
    projected = projected.view(1, -1, 3).transpose(1, 2)
    return projected

nnd = chamfer_3DDist()
def calc_nnd(pc1, pc2, nnd):
    dist1, dist2, _, _ = nnd(pc1, pc2)
    return dist1.mean(1) + dist2.mean(1)
func_nnd = partial(calc_nnd, nnd=nnd)

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
    parser.add_argument('-out_dir', type=str, default=None, help='Output directory.')
    parser.add_argument('-load_path', type=str, default=None, help='Pretrained model path.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True, gpu0=True)
    test_data_root = args.test_data_root
    # get category
    flag = False
    for category in category_list:
        if category in opt['dataset']['train']['file_list_path']:
            flag = True
            break
    assert flag, "Category not found in file_list_path!"
    
    if args.out_dir is None:
        result_root = '../result/ComplementMe/%s' % (category)
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
    tgt_shape_list = []
    ds = ComplementMeCD(category, 'torch', root=test_data_root)
    for src, tgt, mat in tqdm(ds, total=len(ds)):
        src_shape = src.cuda()
        tgt_shape = tgt.cuda()
        deform_shape = model(src_shape, tgt_shape)
        deform_shape = project(deform_shape, mat.cuda())
        deform_shape_list.append(deform_shape)
        tgt_shape_list.append(tgt_shape)

    tgt_shapes = torch.cat(tgt_shape_list, dim=0).transpose(1, 2)
    deform_shapes = torch.cat(deform_shape_list, dim=0).transpose(1, 2)
    cd = func_nnd(tgt_shapes, deform_shapes)
    print('Fitting Chamfer Distance: %.3e' % cd.mean().item())
    deform_shapes = deform_shapes.cpu().numpy()
    np.save(os.path.join(result_root, 'cd_deform_shapes.npy'), deform_shapes)
    print('Saved cd_deform_shapes(' + str(deform_shapes.shape) + ') to', os.path.join(result_root, 'cd_deform_shapes.npy'))
    
    # Transfer
    transfer_shape_list = []
    ds = ComplementMeTransfer(category, 'torch', root=test_data_root)
    for src, tgt, new, mat in tqdm(ds, total=len(ds)):
        src_shape = src.cuda()
        tgt_shape = tgt.cuda()
        new_shape = new.cuda()
        deform_shape, transfer_shape = model.deformation_transfer(src_shape, tgt_shape, new_shape)
        transfer_shape = project(transfer_shape, mat.cuda())
        transfer_shape_list.append(transfer_shape.cpu())
        
    ref_path = os.path.join(test_data_root, category, 'ref_shapes.npy')
    ref_shapes = np.load(ref_path)
    
    transfer_shapes = torch.cat(transfer_shape_list, dim=0).transpose(1, 2).view(10, -1, 2048, 3).numpy()
    MMD_CD, COV_CD = calc_mmd_cov(ref_shapes, transfer_shapes)
    print('MMD-CD: %.3e; Cov-CD %.3f' % (MMD_CD, COV_CD))
    np.save(os.path.join(result_root, 'transfer_shapes.npy'), transfer_shapes)
    print('Saved transfer_shapes(%s) to %s' % (str(transfer_shapes.shape), os.path.join(result_root, 'transfer_shapes.npy')))
        
if __name__ == '__main__':
    main()
