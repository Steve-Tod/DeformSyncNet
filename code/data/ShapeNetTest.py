import os
import numpy as np
import torch

class ShapeNetCD:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ShapeNet/test_data/'):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'cd_src_idx.npy'))
        self.tgt_idx = np.load(os.path.join(self.root, 'cd_tgt_idx.npy'))
        self.num = self.src_idx.shape[0]
        self.frame = frame
            
            
    def __getitem__(self, index):
        src_shape = self.test_shapes[[self.src_idx[index]], :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        if self.frame == 'tf':
            return src_shape, tgt_shape
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous()
            return src_shape, tgt_shape
        
    def __len__(self):
        return self.num
    

class ShapeNetMIOU:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ShapeNet/test_data/'):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.train_src_shapes = np.load(os.path.join(self.root, 'train_src_shapes.npy')) # 10, 10, 2048, 4
        self.src_idx = np.load(os.path.join(self.root, 'miou_src_idx.npy')) # 10, N, 2
        self.src_idx = self.src_idx.reshape(-1, 2) # 10N, 2
        self.tgt_idx = np.load(os.path.join(self.root, 'miou_tgt_idx.npy')) # 10, N
        self.tgt_idx = self.tgt_idx.reshape(-1) # 10N
        self.num = self.tgt_idx.shape[0]
        self.frame = frame
            
    def __getitem__(self, index):
        src_idx = self.src_idx[index]
        src_shape = self.train_src_shapes[[src_idx[0]], src_idx[1], :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        if self.frame == 'tf':
            return src_shape, tgt_shape
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous()
            return src_shape, tgt_shape
        
    def __len__(self):
        return self.num
    
    
class ShapeNetTransfer:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ShapeNet/test_data/', simple=False):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'transfer_src_idx.npy')) # 200/50
        self.tgt_idx = np.load(os.path.join(self.root, 'transfer_tgt_idx.npy')) # 10/50
        self.new_idx = np.load(os.path.join(self.root, 'transfer_new_idx.npy')) # 200/50
        
        if not simple:
            num_trial = self.tgt_idx.shape[0]
            num_shape = self.src_idx.shape[0]
            self.src_idx = np.tile(self.src_idx, num_trial)
            self.new_idx = np.tile(self.new_idx, num_trial)
            self.tgt_idx = np.repeat(self.tgt_idx, num_shape)
        assert self.src_idx.shape == self.new_idx.shape and self.src_idx.shape == self.tgt_idx.shape
        
        self.num = self.src_idx.shape[0]
        self.frame = frame
            
    def __getitem__(self, index):
        src_shape = self.test_shapes[[self.src_idx[index]], :, :3]
        tgt_shape = self.test_shapes[[self.tgt_idx[index]], :, :3]
        new_shape = self.test_shapes[[self.new_idx[index]], :, :3]
        if self.frame == 'tf':
            return src_shape, tgt_shape, new_shape
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous()
            tgt_shape = torch.from_numpy(tgt_shape).transpose(1, 2).contiguous()
            new_shape = torch.from_numpy(new_shape).transpose(1, 2).contiguous()
            return src_shape, tgt_shape, new_shape
        
    def __len__(self):
        return self.num
    
class ShapeNetParallel:
    def __init__(self, category, frame, root='/orion/u/jiangthu/datasets/ShapeNet/test_data/'):
        self.root = os.path.join(root, category)
        self.test_shapes = np.load(os.path.join(self.root, 'test_shapes.npy'))
        self.src_idx = np.load(os.path.join(self.root, 'parallel_src_idx.npy'))
        self.tgt1_idx = np.load(os.path.join(self.root, 'parallel_tgt1_idx.npy'))
        self.tgt2_idx = np.load(os.path.join(self.root, 'parallel_tgt2_idx.npy'))
        self.num = self.src_idx.shape[0]
        self.frame = frame
            
            
    def __getitem__(self, index):
        src_shape = self.test_shapes[[self.src_idx[index]], :, :3]
        tgt1_shape = self.test_shapes[[self.tgt1_idx[index]], :, :3]
        tgt2_shape = self.test_shapes[[self.tgt2_idx[index]], :, :3]
        if self.frame == 'tf':
            return src_shape, tgt1_shape, tgt2_shape
        elif self.frame == 'torch':
            src_shape = torch.from_numpy(src_shape).transpose(1, 2).contiguous()
            tgt1_shape = torch.from_numpy(tgt1_shape).transpose(1, 2).contiguous()
            tgt2_shape = torch.from_numpy(tgt2_shape).transpose(1, 2).contiguous()
            return src_shape, tgt1_shape, tgt2_shape
        
    def __len__(self):
        return self.num