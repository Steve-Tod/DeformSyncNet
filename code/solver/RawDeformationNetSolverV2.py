import logging
import os
import sys
import h5py
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch import nn

from .RawDeformationNetSolverV0 import RawDeformationNetSolverV0
from model import define_net
from model.loss.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from util.util_dir import mkdir
from util.util_visual import plot_3d_point_cloud

logger = logging.getLogger('base')

class RawDeformationNetSolverV2(RawDeformationNetSolverV0):
    # Use deformation handle and hard projection on points
    def feed_data(self, data, test=False):
        self.src_shape = data['src_shape'].to(self.device)
        self.src_path = data['src_path']
        self.src_A = data['src_A'].to(self.device)
        self.src_P = data['src_P'].to(self.device)
        self.src_mat = torch.bmm(self.src_A, self.src_P)
        
        self.tgt_shape = data['tgt_shape'].to(self.device)
        self.tgt_path = data['tgt_path']
    # train one step
    def optimize_parameter(self):
        self.model.train()
        self.update_weight()
        self.optimizer.zero_grad()
        batch_size = self.src_shape.size(0)
        num_points = self.src_shape.size(2)
        deform_shape_raw = self.model(self.src_shape, self.tgt_shape)
        deform_shape_raw = deform_shape_raw.transpose(1, 2).contiguous().view(batch_size, -1, 1) # B,3N,1
        deform_shape_proj = torch.bmm(self.src_mat, deform_shape_raw) # B,3N,1
        self.deform_shape = deform_shape_proj.view(batch_size, -1, 3).transpose(1, 2).contiguous()
        
        loss = 0.0
        for k, v in self.cri_dict.items():
            if 'fit' in k:
                loss_fit = v['cri'](self.tgt_shape, self.deform_shape)
                loss += v['weight'] * loss_fit
                self.log_dict['loss_' + k] = loss_fit.item()
            elif 'sym' in k:
                flipped = v['sym_vec'] * self.deform_shape
                loss_sym = v['cri'](flipped, self.deform_shape)
                loss += v['weight'] * loss_sym
                self.log_dict['loss_' + k] = loss_sym.item()
                
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.update_learning_rate()

    def test(self):
        batch_size = self.src_shape.size(0)
        self.model.eval()
        with torch.no_grad():
            deform_shape_raw = self.model(self.src_shape, self.tgt_shape)
            deform_shape_raw = deform_shape_raw.transpose(1, 2).contiguous().view(batch_size, -1, 1) # B,3N,1
            deform_shape_proj = torch.bmm(self.src_mat, deform_shape_raw) # B,3N,1
            self.deform_shape = deform_shape_proj.view(batch_size, -1, 3).transpose(1, 2).contiguous()
        self.model.train()

    def get_current_visual(self):
        if 'plot' in self.opt['train'].keys():
            lim = [self.opt['train']['plot']['lim']] * 3
            seq = self.opt['train']['plot']['seq']
        else:
            lim = [(-1, 1)] * 3
            seq = [0, 1, 2]
        
        fig = plt.figure(figsize=(9, 3))
        num_point = 2048
        colors = np.linspace(start=0, stop=2*np.pi, num=2048)
        
        ax_src = fig.add_subplot(1, 3, 1, projection='3d')
        pc_src = self.src_shape.cpu().numpy()[0]
        plot_3d_point_cloud(pc_src[seq[0]], pc_src[seq[1]], pc_src[seq[2]], 
                            axis=ax_src, show=False, lim=lim,
                            c=colors, cmap='hsv')
        ax_src.set_title('source shape')
        
        ax_tgt = fig.add_subplot(1, 3, 2, projection='3d')
        pc_tgt = self.tgt_shape.cpu().numpy()[0]
        plot_3d_point_cloud(pc_tgt[seq[0]], pc_tgt[seq[1]], pc_tgt[seq[2]], 
                            axis=ax_tgt, show=False, lim=lim, 
                            c=colors, cmap='hsv')
        ax_tgt.set_title('target shape')
        
        ax_deform = fig.add_subplot(1, 3, 3, projection='3d')
        pc_deform = self.deform_shape.cpu().numpy()[0]
        plot_3d_point_cloud(pc_deform[seq[0]], pc_deform[seq[1]], pc_deform[seq[2]], 
                            axis=ax_deform, show=False, lim=lim, 
                            c=colors, cmap='hsv')
        ax_deform.set_title('deform shape')
        
        plt.tight_layout()
        
        return fig