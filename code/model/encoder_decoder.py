import torch
from torch import nn

from .block import create_conv_stack, create_fc_stack

class PointNetFeat(nn.Module):
    def __init__(self, conv_channel):
        super().__init__()
        self.conv = create_conv_stack(conv_channel, nn.LeakyReLU, negative_slope=0.2)
        self.symmetry = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        after_conv = self.conv(x)
        global_feat = self.symmetry(after_conv)
        return global_feat.squeeze(-1)
    

class PointNetAE(nn.Module):
    def __init__(self,
                 dec_fc_channel,
                 enc_conv_channel=[3, 64, 128, 1024],
                 enc_fc_channel=[1024, 256, 100]):
        super().__init__()
        self.encoder = PointNetFeat(enc_conv_channel)
        if len(enc_fc_channel) != 0:
            self.encoder.add_module(
                'enc_fc',
                create_fc_stack(enc_fc_channel, nn.LeakyReLU,
                                **{'negative_slope': 0.2}))
        self.decoder = create_fc_stack(dec_fc_channel, nn.LeakyReLU,
                                       **{'negative_slope': 0.2})

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        b_size = x.size(0)
        x = self.decoder(x)
        x = x.view(b_size, 3, -1)
        return x
