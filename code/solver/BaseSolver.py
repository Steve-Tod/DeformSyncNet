import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger('base')

class BaseSolver:
    def __init__(self, opt):
        self.opt = opt
        if opt['gpu_id'] is not None and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        if opt['gpu_id'] is not None and len(opt['gpu_id']) >= 2:
            self.parallel = True
        else:
            self.parallel = False
        self.is_train = opt['is_train']
        self.scheduler_list = []
        self.optimizer_list = []
        self.step = 0
        self.best_step = 0
        self.best_res = 1e5
    
    def feed_data(self, data):
        pass
    
    def optimize_parameter(self):
        pass
    
    def get_current_visual(self):
        pass
    
    def get_current_loss(self):
        pass
    
    def print_network(self):
        pass
    
    def save(self, label):
        pass
    
    def load(self):
        pass
    
    def update_learning_rate(self):
        for s in self.scheduler_list:
            s.step()
            
    def get_current_learning_rate(self):
        return self.optimizer_list[0].param_groups[0]['lr']
    
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
        #logger.info(s)
        
    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['model'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  
        # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 
                 'iter': self.step, 
                 'best_step': self.best_step,
                 'best_res': self.best_res,
                 'schedulers': [], 
                 'optimizers': []}
        for s in self.scheduler_list:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizer_list:
            state['optimizers'].append(o.state_dict())
            
        save_filename = '{}.state'.format(self.step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizer_list = resume_state['optimizers']
        resume_scheduler_list = resume_state['schedulers']
        assert len(resume_optimizer_list) == len(self.optimizer_list), 'Wrong lengths of optimizers'
        assert len(resume_scheduler_list) == len(self.scheduler_list), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizer_list):
            self.optimizer_list[i].load_state_dict(o)
        for i, s in enumerate(resume_scheduler_list):
            self.scheduler_list[i].load_state_dict(s)
        if 'best_step' in resume_state.keys():
            self.best_step = resume_state['best_step'] 
        if 'best_res' in resume_state.keys():
            self.best_res = resume_state['best_res']