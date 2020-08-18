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

from .BaseSolver import BaseSolver
from model import define_net
from model.loss.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from model.loss.emd import EMD
from util.util_dir import mkdir
from util.util_visual import plot_3d_point_cloud

logger = logging.getLogger('base')

class RawDeformationNetSolverV0(BaseSolver):
    def __init__(self, opt):
        super().__init__(opt)
        train_opt = opt['train']
        model_opt = opt['model']
        self.model = define_net(model_opt).to(self.device)

        if self.parallel:
            self.model = DataParallel(self.model)
        if self.is_train:
            self.model.train()
            self.cri_dict = {}
            self.nnd = chamfer_3DDist()
            self.build_loss(train_opt)
            # optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_opt['learning_rate'],
                weight_decay=train_opt['weight_decay'])
            self.optimizer_list.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizer_list:
                    self.scheduler_list.append(
                        torch.optim.lr_scheduler.MultiStepLR(
                            self.optimizer,
                            train_opt['lr_steps'],
                            gamma=train_opt['lr_gamma']))
            else:
                raise NotImplementedError(
                    'Learning rate schedule scheme %s is not implemented' %
                    train_opt['lr_scheme'])

            self.log_dict = OrderedDict()
        self.print_network()  # print network
        self.load()  # load model if needed
        
    def build_loss(self, train_opt):
        for k, v in train_opt['loss'].items():
            self.cri_dict[k] = {}
            self.cri_dict[k]['weight'] = v['weight']
            if v['loss_type'] == 'CD':
                self.cri_dict[k]['cri'] = self.calc_nnd
            elif v['loss_type'] == 'EMD':
                if not hasattr(self, 'emd'):
                    self.emd = EMD()
                self.cri_dict[k]['cri'] = self.calc_emd
            elif v['loss_type'] == 'l2':
                self.cri_dict[k]['cri'] = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type %s not implemented!' % v['loss_type'])

            if 'gamma' in v.keys(): # weight decay
                self.cri_dict[k]['gamma'] = v['gamma']
                self.cri_dict[k]['steps'] = v['steps']
            if 'step_weight' in v.keys(): # weight decay
                self.cri_dict[k]['step_weight'] = v['step_weight']
                self.cri_dict[k]['steps'] = v['steps']
            if 'sym_axis' in v.keys(): # symmetry axis
                sym_vec = torch.ones((1, 3, 1)).to(self.device)
                sym_vec[:, int(v['sym_axis']), :] *= -1
                self.cri_dict[k]['sym_vec'] = sym_vec
                
    def feed_data(self, data, test=False):
        self.src_shape = data['src_shape'].to(self.device)
        self.src_path = data['src_path']
        self.tgt_shape = data['tgt_shape'].to(self.device)
        self.tgt_path = data['tgt_path']

    # train one step
    def optimize_parameter(self):
        self.model.train()
        self.update_weight()
        self.optimizer.zero_grad()
        self.deform_shape = self.model(self.src_shape, self.tgt_shape)

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
            elif 'reg_dict' in k:
                batch_size, num_point_3, num_coeff = self.model.dict.shape
                dict_norm = self.model.dict.abs().sum()
                loss_reg_dict = dict_norm / (batch_size * num_coeff)
                loss += v['weight'] * loss_reg_dict
                self.log_dict['loss_' + k] = loss_reg_dict.item()
                
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.update_learning_rate()

    def update_weight(self):
        def _new_weight(init_weight, steps, gamma):
            return init_weight * (gamma ** bisect_right(steps, self.step))
        
        for k, v in self.cri_dict.items():
            if 'gamma' in v.keys():
                if 'init_weight' not in v.keys():
                    v['init_weight'] = v['weight']
                new_weight = _new_weight(v['init_weight'], v['steps'], v['gamma'])
                v['weight'] = new_weight

    def test(self):
        self.model.eval()
        with torch.no_grad():
            self.deform_shape = self.model(self.src_shape, self.tgt_shape)
        self.model.train()

    def evaluate(self):
        return_dict = OrderedDict()
        with torch.no_grad():
            for k, v in self.cri_dict.items():
                if 'fit' in k:
                    loss_fit = v['cri'](self.tgt_shape, self.deform_shape)
                    return_dict['loss_' + k] = loss_fit.item()
                elif 'sym' in k:
                    flipped = v['sym_vec'] * self.deform_shape
                    loss_sym = v['cri'](flipped, self.deform_shape)
                    return_dict['loss_' + k] = loss_sym.item()
        return return_dict

    def update_learning_rate(self):
        for s in self.scheduler_list:
            s.step(self.step)

    def calc_nnd(self, pc1, pc2):
        dist1, dist2, _, _ = self.nnd(
            pc1.transpose(2, 1).contiguous(),
            pc2.transpose(2, 1).contiguous())
        return dist1.mean() + dist2.mean()

    def calc_emd(self, pc1, pc2):
        emdist = self.emd(
            pc1.transpose(2, 1).contiguous(),
            pc2.transpose(2, 1).contiguous())
        return emdist.mean()
    
    def get_current_log(self):
        return self.log_dict

    def log_current(self, epoch, tb_logger=None):
        logs = self.log_dict
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
            epoch, self.step, self.get_current_learning_rate())
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            # tensorboard logger
        logger.info(message)
        if tb_logger is not None:
            for k, v in logs.items():
                tb_logger.add_scalar('loss/%s' % k, v,
                                     self.step)
            # log loss weights
            for k, v in self.cri_dict.items():
                tb_logger.add_scalar('weight/weight_%s' % k, v['weight'], self.step)

    def get_current_visual(self):
        fig = plt.figure(figsize=(9, 3))
        num_point = 2048
        colors = np.linspace(start=0, stop=2*np.pi, num=2048)
        
        ax_src = fig.add_subplot(1, 3, 1, projection='3d')
        pc_src = self.src_shape.cpu().numpy()[0]
        plot_3d_point_cloud(pc_src[0], -pc_src[2], pc_src[1], 
                            axis=ax_src, show=False, lim=[(-1, 1)] * 3,
                            c=colors, cmap='hsv')
        ax_src.set_title('source shape')
        
        ax_tgt = fig.add_subplot(1, 3, 2, projection='3d')
        pc_tgt = self.tgt_shape.cpu().numpy()[0]
        plot_3d_point_cloud(pc_tgt[0], -pc_tgt[2], pc_tgt[1], 
                            axis=ax_tgt, show=False, lim=[(-1, 1)] * 3, 
                            c=colors, cmap='hsv')
        ax_tgt.set_title('target shape')
        
        ax_deform = fig.add_subplot(1, 3, 3, projection='3d')
        pc_deform = self.deform_shape.cpu().numpy()[0]
        plot_3d_point_cloud(pc_deform[0], -pc_deform[2], pc_deform[1], 
                            axis=ax_deform, show=False, lim=[(-1, 1)] * 3, 
                            c=colors, cmap='hsv')
        ax_deform.set_title('deform shape')
        
        plt.tight_layout()
        
        return fig

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.model.__class__.__name__,
                self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(
            net_struc_str, n))
        logger.info(s)

    def load(self):
        if self.opt['path']['strict_load'] is None:
            strict = True
        else:
            strict = self.opt['path']['strict_load']

        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model from [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.model, strict)

    def save(self, save_label):
        self.save_network(self.model, 'model', save_label)