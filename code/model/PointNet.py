import torch
from torch import nn

class PointNetConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=bias))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm1d(out_channel))
        self.layer.add_module('act', nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.layer(x)

class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = nn.Sequential(PointNetConv1d(3, 64), 
                                   PointNetConv1d(64, 64))
        self.conv2 = nn.Sequential(PointNetConv1d(64, 64),
                                   PointNetConv1d(64, 128), 
                                   PointNetConv1d(128, 1024))
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        point_feat = self.conv1(x)
        global_feat = self.pool(self.conv2(point_feat)).squeeze(-1)
        return global_feat, point_feat

class PointNetCls(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.feat = PointNetfeat()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, opt['out_dim'])
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x, _ = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x
    
class PointNetClsMix(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.feat = PointNetfeat()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, opt['out_dim'])
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        x, _ = self.feat(x)
        y, _ = self.feat(y)
        feat = torch.cat((x, y), dim=1)
        x = self.relu(self.bn1(self.fc1(feat)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x
    
class PointNetSeg(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.feat = PointNetfeat()
        self.conv = nn.Sequential(PointNetConv1d(1088, 512),
                                  PointNetConv1d(512, 256), 
                                  PointNetConv1d(256, 128),
                                  PointNetConv1d(128, opt['feature_dim']))
        
    def forward(self, x):
        num_point = x.size(-1)
        global_feat, point_feat = self.feat(x)
        global_feat = global_feat.unsqueeze(-1).repeat(1, 1, num_point)
        feat = torch.cat((global_feat, point_feat), dim=1)
        return self.conv(feat)