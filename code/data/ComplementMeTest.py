import os
import numpy as np
import torch

class ComplementMeCD:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ComplementMe/test_data/', return_AP=False):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.test_A = np.load(os.path.join(self.root, 'test_A.npy'))
        self.test_P = np.load(os.path.join(self.root, 'test_P.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'cd_src_idx.npy'))
        self.tgt_idx = np.load(os.path.join(self.root, 'cd_tgt_idx.npy'))
        self.num = self.src_idx.shape[0]
        self.frame = frame
        self.return_AP = return_AP
            
    def __getitem__(self, index, return_AP=False):
        src_idx = [self.src_idx[index]]
        src_shape = self.test_shapes[src_idx, :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        if self.return_AP:
            return src_shape, tgt_shape, self.test_A[src_idx], self.test_P[src_idx]
        src_mat = np.matmul(self.test_A[src_idx], self.test_P[src_idx])
        
        if self.frame == 'tf':
            return src_shape, tgt_shape, src_mat
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous().float()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous().float()
            src_mat = torch.from_numpy(src_mat).float()
            return src_shape, tgt_shape, src_mat
        
    def __len__(self):
        return self.num
    
class ComplementMeRawCD:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ComplementMe/test_data/'):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'cd_src_idx.npy'))
        self.tgt_idx = np.load(os.path.join(self.root, 'cd_tgt_idx.npy'))
        self.num = self.src_idx.shape[0]
        self.frame = frame
            
    def __getitem__(self, index, return_AP=False):
        src_idx = [self.src_idx[index]]
        src_shape = self.test_shapes[src_idx, :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        
        if self.frame == 'tf':
            return src_shape, tgt_shape
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous().float()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous().float()
            return src_shape, tgt_shape
        
    def __len__(self):
        return self.num
    
class ComplementMeTransfer:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ComplementMe/test_data/', simple=False, return_AP=False):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'transfer_src_idx.npy')) # 200
        self.tgt_idx = np.load(os.path.join(self.root, 'transfer_tgt_idx.npy')) # 100
        self.new_idx = np.load(os.path.join(self.root, 'transfer_new_idx.npy')) # 200
        self.test_A = np.load(os.path.join(self.root, 'test_A.npy'))
        self.test_P = np.load(os.path.join(self.root, 'test_P.npy'))
        
        if not simple:
            num_trial = self.tgt_idx.shape[0]
            num_shape = self.src_idx.shape[0]
            self.src_idx = np.tile(self.src_idx, num_trial)
            self.new_idx = np.tile(self.new_idx, num_trial)
            self.tgt_idx = np.repeat(self.tgt_idx, num_shape)
        
        self.num = self.src_idx.shape[0]
        self.frame = frame
        self.return_AP = return_AP
            
    def __getitem__(self, index):
        new_idx = [self.new_idx[index]]
        src_shape = self.test_shapes[[self.src_idx[index]], :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        new_shape = self.test_shapes[new_idx, :, :3]
        if self.return_AP:
            return src_shape, tgt_shape, new_shape, self.test_A[new_idx], self.test_P[new_idx]
        
        new_mat = np.matmul(self.test_A[new_idx], self.test_P[new_idx])
        
        if self.frame == 'tf':
            return src_shape, tgt_shape, new_shape, new_mat
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous().float()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous().float()
            new_shape = torch.from_numpy(new_shape).transpose(1, 2).contiguous().float()
            new_mat = torch.from_numpy(new_mat).float()
            return src_shape, tgt_shape, new_shape, new_mat
        
    def __len__(self):
        return self.num
    
    
class ComplementMeCDTransfer:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ComplementMe/test_data/', return_AP=False):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'transfer_src_idx.npy')) # 200
        self.tgt_idx = np.load(os.path.join(self.root, 'transfer_tgt_idx.npy')) # 100
        self.test_A = np.load(os.path.join(self.root, 'test_A.npy'))
        self.test_P = np.load(os.path.join(self.root, 'test_P.npy'))
        num_trial = self.tgt_idx.shape[0]
        num_shape = self.src_idx.shape[0]
        self.src_idx = np.tile(self.src_idx, num_trial)
        self.tgt_idx = np.repeat(self.tgt_idx, num_shape)
        
        self.num = self.src_idx.shape[0]
        self.frame = frame
        self.return_AP = return_AP
            
    def __getitem__(self, index):
        src_idx = [self.src_idx[index]]
        src_shape = self.test_shapes[[self.src_idx[index]], :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        if self.return_AP:
            return src_shape, tgt_shape, self.test_A[src_idx], self.test_P[src_idx]
        
        src_mat = np.matmul(self.test_A[src_idx], self.test_P[src_idx])
        
        if self.frame == 'tf':
            return src_shape, tgt_shape, src_mat
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous().float()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous().float()
            src_mat = torch.from_numpy(src_mat).float()
            return src_shape, tgt_shape, src_mat
        
    def __len__(self):
        return self.num
    

class ComplementMeRawTransfer:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ComplementMe/test_data/'):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'transfer_src_idx.npy')) # 200
        self.tgt_idx = np.load(os.path.join(self.root, 'transfer_tgt_idx.npy')) # 100
        self.new_idx = np.load(os.path.join(self.root, 'transfer_new_idx.npy')) # 200
        
        num_trial = self.tgt_idx.shape[0]
        num_shape = self.src_idx.shape[0]
        self.src_idx = np.tile(self.src_idx, num_trial)
        self.new_idx = np.tile(self.new_idx, num_trial)
        self.tgt_idx = np.repeat(self.tgt_idx, num_shape)
        
        self.num = self.src_idx.shape[0]
        self.frame = frame
            
    def __getitem__(self, index):
        new_idx = [self.new_idx[index]]
        src_shape = self.test_shapes[[self.src_idx[index]], :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        new_shape = self.test_shapes[new_idx, :, :3]
        
        if self.frame == 'tf':
            return src_shape, tgt_shape, new_shape
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous().float()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous().float()
            new_shape = torch.from_numpy(new_shape).transpose(1, 2).contiguous().float()
            return src_shape, tgt_shape, new_shape
        
    def __len__(self):
        return self.num