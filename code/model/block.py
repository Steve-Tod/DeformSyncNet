import torch
from torch import nn

def create_fc_stack(fc_channel, act=nn.ReLU, **act_kwargs):
    fc = nn.Sequential()
    for i in range(len(fc_channel) - 1):
        in_c = fc_channel[i]
        out_c = fc_channel[i + 1]
        fc_name = 'fc_%d_%d' % (in_c, out_c)
        fc.add_module(fc_name, nn.Linear(in_c, out_c))
        act_name = 'fc_act_%d_%d' % (in_c, out_c)
        if i < len(fc_channel) - 2:
            fc.add_module(act_name, act(**act_kwargs))

    return fc

def create_conv_stack(conv_channel, act=nn.ReLU, **act_kwargs):
    conv = nn.Sequential()
    for i in range(len(conv_channel) - 1):
        in_c = conv_channel[i]
        out_c = conv_channel[i + 1]
        conv_name = 'conv1d_%d_%d' % (in_c, out_c)
        conv.add_module(conv_name, nn.Conv1d(in_c, out_c, 1))
        act_name = 'conv_act_%d_%d' % (in_c, out_c)
        conv.add_module(act_name, act(**act_kwargs))
    return conv

class FMNetResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bnorm=False):
        super().__init__()
        self.block = nn.Sequential()
        self.block.add_module('conv_1', nn.Conv1d(dim_in, dim_out, 1))
        if bnorm:
            self.block.add_module('bnorm_1', nn.BatchNorm1d(dim_out))
        self.block.add_module('relu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.block.add_module('conv_2', nn.Conv1d(dim_out, dim_out, 1))
        if bnorm:
            self.block.add_module('bnorm_2', nn.BatchNorm1d(dim_out))
        
        if dim_in != dim_out:
            self.affine = nn.Conv1d(dim_in, dim_out, 1)
        else:
            self.affine = None
        
    def forward(self, x):
        x_out = self.block(x)
        if self.affine:
            x_out += self.affine(x)
        else:
            x_out += x
        return torch.relu(x_out)