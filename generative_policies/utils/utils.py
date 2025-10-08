import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

def inputs_to_torch(x, device):
    if type(x) == np.ndarray or type(x) == list:
        x = torch.FloatTensor(x)
    return x.to(device)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GaussianFourierEmb(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, dim, scale=16, trainable=True):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.trainable = trainable

        # Initialize W with a normal distribution and set requires_grad based on trainable
        W_init = torch.randn(self.dim // 2) * self.scale
        self.W = nn.Parameter(W_init, requires_grad=self.trainable)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)
    
    def to(self, device):
        super().to(device)
        return self

