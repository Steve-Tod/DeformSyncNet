import torch
from torch import nn

def xavier_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
