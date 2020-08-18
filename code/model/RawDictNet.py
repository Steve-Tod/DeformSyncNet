import torch
from torch import nn
from .block import create_conv_stack, create_fc_stack
from .encoder_decoder import PointNetFeat
from .PointNet import PointNetSeg, PointNetCls, PointNetClsMix

class RawDictNetV0(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt['arch'] == 'MLP':
            self.feat = create_conv_stack(opt['point_conv_channel'],
                                          nn.LeakyReLU,
                                          negative_slope=0.2)
        elif opt['arch'] == 'PointNetSeg':
            self.feat = PointNetSeg(opt)
        else:
            raise NotImplementedError('Feature extractor [%s] not implemented' % opt['arch'])
            
        self.norm_column = opt['norm_column']
        

    def forward(self, src_pc, debug=False):
        '''
        default_pc: B * 3 * n
        P: B * k * 3n
        '''
        num_point = src_pc.size(2)
        batch_size = src_pc.size(0)
        
        default_feature = self.feat(src_pc) # B * (3*k) * n
        # B * (3*n) * k
        dictionary = default_feature.transpose(1, 2).contiguous().view(batch_size, 3 * num_point, -1)
        if self.norm_column:
            dictionary = dictionary / (dictionary ** 2).sum(dim=1, keepdim=True).sqrt()
        return dictionary
        
        
class RawCoefficientNetV0(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt['arch'] == 'PointNetCls':
            self.net = PointNetCls(opt)
        else:
            raise NotImplementedError('Coefficient predictor [%s] not implemented' % opt['arch'])
        self.tanh = opt['tanh']
        # self.feat = PointNetFeat(opt['conv_channel'])
        # self.fc = create_fc_stack(opt['fc_channel'])
    def forward(self, pc_src, pc_dst):
        if self.tanh:
            src_coeff = torch.tanh(self.net(pc_src))
            dst_coeff = torch.tanh(self.net(pc_dst))
        else:
            src_coeff = self.net(pc_src)
            dst_coeff = self.net(pc_dst)
        # between -2, 2
        coeff_offset = dst_coeff - src_coeff
        return coeff_offset
    
class RawCoefficientNetV1(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt['arch'] == 'PointNetClsMix':
            self.net = PointNetClsMix(opt)
        else:
            raise NotImplementedError('Coefficient predictor [%s] not implemented' % opt['arch'])
        self.tanh = opt['tanh']
    def forward(self, pc_src, pc_dst):
        # between -2, 2
        if self.tanh:
            coeff_offset = 2 * torch.tanh(self.net(pc_src, pc_dst))
        else:
            coeff_offset = self.net(pc_src, pc_dst)
        return coeff_offset
    
    
class RawDeformationDictNetV0(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt['dict']['version'] == 0:
            self.dict_net = RawDictNetV0(opt['dict'])
        else:
            raise NotImplementedError('Dict net version %d not implemented' % opt['dict']['version'])
            
        if opt['coeff']['version'] == 0:
            self.coeff_net = RawCoefficientNetV0(opt['coeff'])
        elif opt['coeff']['version'] == 1: # concatenation
            self.coeff_net = RawCoefficientNetV1(opt['coeff'])
        else:
            raise NotImplementedError('Coeff net version %d not implemented' % opt['coeff']['version'])

    def forward(self,
                pc_source,
                pc_target):
        batch_size = pc_target.size(0)
        self.dict = self.dict_net(pc_source) # b,3n,k
        self.coeff = self.coeff_net(pc_source, pc_target) # b,k

        pc_offset = torch.bmm(self.dict, self.coeff.unsqueeze(-1)) # b,3n,1
        pc_offset = pc_offset.view(batch_size, -1, 3).transpose(1, 2).contiguous() # b,3,n
        pc_deformed = pc_source + pc_offset
        return pc_deformed
        
    def deformation_transfer(self, pc_source, pc_target, pc_source_new):
        batch_size = pc_target.size(0)
        self.dict = self.dict_net(pc_source) # b,3n,k
        self.coeff = self.coeff_net(pc_source, pc_target) # b,k

        pc_offset = torch.bmm(self.dict, self.coeff.unsqueeze(-1)) # b,3n,1
        pc_offset = pc_offset.view(batch_size, -1, 3).transpose(1, 2).contiguous() # b,3,n
        pc_deformed = pc_source + pc_offset
        
        new_dict = self.dict_net(pc_source_new)
        pc_offset_new = torch.bmm(new_dict, self.coeff.unsqueeze(-1)) # b,3n,1
        pc_offset_new = pc_offset_new.view(batch_size, -1, 3).transpose(1, 2).contiguous()
        pc_deformed_new = pc_source_new + pc_offset_new
        
        return pc_deformed, pc_deformed_new
    

class RawDeformationDictNetV1(RawDeformationDictNetV0):
    def forward(self, pc_source, pc_target, src_A, src_P):
        batch_size = pc_target.size(0)
        self.dict = self.dict_net(pc_source) # b,3n,k
        self.dict = torch.bmm(src_P, self.dict) # b, k', k
        self.coeff = self.coeff_net(pc_source, pc_target) # b, k

        param_offset = torch.bmm(self.dict, self.coeff.unsqueeze(-1)) # b, k',1
        pc_offset = torch.bmm(src_A, param_offset) # b, 3n, 1
        pc_offset = pc_offset.view(batch_size, -1, 3).transpose(1, 2).contiguous() # b,3,n
        pc_deformed = pc_source + pc_offset
        return pc_deformed
