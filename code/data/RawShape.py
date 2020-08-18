import os
import h5py
import numpy as np
import json
from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader

from .util import *

def read_cat2id_dict(path):
    with open(path) as f:
        lines = f.read().splitlines()
    cat2id_dict = {}
    for l in lines:
        cat, identity = l.split('\t')
        cat2id_dict[cat] = identity
    return cat2id_dict

class RawShapeNetPairTrainV0(Dataset):
    def __init__(self, opt):
        self.phase = opt['phase']
        data_root = opt['root']
        self.norm = opt['norm']
        cat2id_dict = read_cat2id_dict(
            os.path.join(data_root, 'synsetoffset2category.txt'))
        cat_id = cat2id_dict[opt['category']]
        with open(os.path.join(data_root, 'train_test_split/shuffled_%s_file_list.json' % self.phase)) as f:
            self.file_list = [os.path.join(data_root, x[11:] + '.txt') for x in json.load(f) if cat_id in x]
            self.file_list.sort()
            self.target_file_list = [x for x in self.file_list]
            
        np.random.shuffle(self.target_file_list)
        self.num_point = opt['num_point']
        with open(os.path.join(data_root, '%s_label.json' % cat_id)) as f:
            self.label = json.load(f)

    def read(self, path):
        pc = np.loadtxt(path)
        num_point = pc.shape[0]
        if num_point >= self.num_point:
            index = torch.randperm(num_point)[:self.num_point].sort()[0]
        else:
            index = torch.zeros(self.num_point)
            index[:num_point] = torch.arange(num_point)
            index = index.long()
        label = torch.from_numpy(pc[index, -1]).long()
        points = torch.from_numpy(pc[index, :3]).float()
        if self.norm:
            points = BoundingBox(points)
        points = points.transpose(0, 1).contiguous()
        return points, label

    def __getitem__(self, index):
        source_path = self.file_list[index]
        if self.phase == 'train':
            target_idx = torch.randint(len(self.file_list),
                                       size=(1, )).int().item()
        else:
            target_idx = index
        target_path = self.target_file_list[target_idx]

        src_pc, src_seg = self.read(source_path)
        tgt_pc, tgt_seg = self.read(target_path)
        

        return {
            'src_shape': src_pc,
            'src_seg': src_seg,
            'src_path': source_path,
            'tgt_shape': tgt_pc,
            'tgt_seg': tgt_seg,
            'tgt_path': target_path,
            'label': self.label
        }

    def __len__(self):
        return len(self.file_list)
