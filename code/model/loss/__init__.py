import torch
import torch.nn as nn


# regularization losses

def norm_l1(tensor):
    # B, M, N
    batch_size, num_point3, num_column = tensor.shape
    return tensor.abs().sum() / (batch_size * num_column)

def norm_l21(tensor):
    batch_size, num_point3, num_column = tensor.shape
    t2 = tensor ** 2
    t2_sq = t2.sum(1).sqrt()
    return t2_sq.abs().sum() / (batch_size * num_column)

def reg_ortg(tensor):
    # orthogonal regularization
    # |A^T A - I|_1
    # B, M, N
    batch_size, num_param, num_column = tensor.shape
    I = torch.eye(num_column, device=tensor.device, dtype=tensor.dtype).unsqueeze(0)
    I = I.repeat(batch_size, 1, 1)

    colomn_norm = (tensor ** 2).sum(1, keepdim=True).sqrt()
    normed = tensor / colomn_norm
    err = torch.bmm(normed.transpose(1, 2), normed) - I
    loss = (err**2).sum((1, 2)).sqrt().mean()
    return loss
    