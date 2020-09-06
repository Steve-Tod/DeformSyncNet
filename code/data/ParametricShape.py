import os
import h5py
import numpy as np
import json
from itertools import product

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ParametricPairTrainV0(Dataset):
    # source and target with everything
    def __init__(self, opt):
        self.is_train = (opt['phase'] == 'train')
        
        with open(opt['file_list_path']) as f:
            self.file_list = [os.path.join(opt['root'], x) for x in f.read().splitlines()]
            if self.is_train:
                self.target_file_list = [x for x in self.file_list]
                np.random.shuffle(self.target_file_list)
            else:
                self.target_file_list = [self.file_list[-1]]
                self.target_file_list.extend(self.file_list[:-1])
        
        with h5py.File(self.file_list[0], 'r') as f:
            tmp_mat = np.array(f['points_mat'])
            
        assert tmp_mat.shape[0] % 3 == 0
        self.num_point_all = tmp_mat.shape[0] / 3
        self.num_point = opt['num_point'] if 'num_point' in opt.keys() else self.num_point_all
        
        self.max_num_param = opt['max_num_param']
        assert self.num_point == self.num_point_all, 'Can not select subset, or the pinv need to be computed again'
        
    def read_and_pad(self, path):
        with h5py.File(path, 'r') as f:
            points_mat = np.array(f['points_mat'])
            points_pinv_mat = np.array(f['points_pinv_mat'])
            point_cloud = np.array(f['points'])
            default_param = np.array(f['default_param'])
        pc = torch.from_numpy(point_cloud).float()
        num_param = default_param.shape[0]
        # padding
        A = torch.zeros((int(self.num_point * 3), self.max_num_param))
        A[:, :num_param] = torch.from_numpy(points_mat).float()
        
        P = torch.zeros((self.max_num_param, int(self.num_point * 3)))
        P[:num_param, :] = torch.from_numpy(points_pinv_mat).float()
        if num_param < self.max_num_param:
            P[num_param:, :] += P[[0], :] #max_num_param * (3*num_point)
            
        param = torch.zeros(self.max_num_param)
        param[:num_param] = torch.from_numpy(default_param).float()
        
        param_mask = torch.zeros(self.max_num_param)
        param_mask[:num_param] += 1
        if pc.size(0) != 3:
            pc = pc.transpose(0, 1).contiguous()
        return pc, A, P, param, param_mask    
        
    def __getitem__(self, index):
        source_path = self.file_list[index]
        if self.is_train:
            target_idx = torch.randint(len(self.file_list), size=(1,)).int().item()
        else:
            target_idx = index
        target_path = self.target_file_list[target_idx]
        
        src_pc, src_A, src_P, src_param, src_param_mask = self.read_and_pad(source_path)
        tgt_pc, tgt_A, tgt_P, tgt_param, tgt_param_mask = self.read_and_pad(target_path)
        
        
        return {
            'src_shape': src_pc,
            'src_A': src_A,
            'src_P': src_P,
            'src_param': src_param,
            'src_param_mask': src_param_mask,
            'src_path': source_path,
            
            'tgt_shape': tgt_pc,
            'tgt_A': tgt_A,
            'tgt_P': tgt_P,
            'tgt_param': tgt_param,
            'tgt_param_mask': tgt_param_mask,
            'tgt_path': target_path
        }
    
    def __len__(self):
        return len(self.file_list)
    
    
class ParametricPairTrainV1(ParametricPairTrainV0):
    # with source projection matrix only 
    
    def read_points(self, path):
        with h5py.File(path, 'r') as f:
            point_cloud = np.array(f['points']).astype(np.float32)
        pc = torch.from_numpy(point_cloud.T).float()
        return pc
        
    def __getitem__(self, index):
        source_path = self.file_list[index]
        if self.is_train:
            target_idx = torch.randint(len(self.file_list), size=(1,)).int().item()
        else:
            target_idx = index
        target_path = self.target_file_list[target_idx]
        
        src_pc, src_A, src_P, _, _ = self.read_and_pad(source_path)
        tgt_pc = self.read_points(target_path)
        
        
        return {
            'src_shape': src_pc,
            'src_A': src_A,
            'src_P': src_P,
            'src_path': source_path,
            
            'tgt_shape': tgt_pc,
            'tgt_path': target_path
        }
    
    def __len__(self):
        return len(self.file_list)