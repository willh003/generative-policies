from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from .utils import inputs_to_torch


class IdentityEncoder(nn.Module):
    """
    Handles observations that are numpy arrays or torch tensors 
    of shapes (B, D) or (B, 1, D) or (D)
    puts them on the correct device
    """

    def __init__(self, input_dim: int, device: str = 'cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def __call__(self, input_data):
        input_data = inputs_to_torch(input_data, self.device)

        if len(input_data.shape) == 1: # no batch provided
            input_data = input_data.unsqueeze(0)
        elif len(input_data.shape) == 3: # seq dim 1 provided
            assert input_dataobs.shape[1] == 1, "No sequence length expected"
            input_data = input_data.squeeze(dim=1)

        input_data = input_data.to(self.device)
        
        return input_data

    @property
    def output_dim(self):
        return self.input_dim
    
class IdentityTwoInputEncoder(nn.Module):
    """
    Encodes two input tensors/arrays into a single tensor, handling batching and device
    Useful for multiple conditioning variables (e.g. obs and action) 
    """
    
    def __init__(self, input_1_dim, input_2_dim, device='cuda'):
        super(IdentityTwoInputEncoder, self).__init__()
        self.input_1_dim = input_1_dim
        self.input_2_dim = input_2_dim
        self.input_1_encoder = IdentityEncoder(input_1_dim, device=device)
        self.input_2_encoder = IdentityEncoder(input_2_dim, device=device)
        self.device = device

    def __call__(self, input_1, input_2) -> torch.Tensor:
        input_1_cond = self.input_1_encoder(input_1)
        input_2_cond = self.input_2_encoder(input_2)
        cond = torch.cat([input_1_cond, input_2_cond], dim=-1)
        return cond

    @property
    def output_dim(self):
        return self.input_1_dim + self.input_2_dim

    def to(self, device):
        self.device = device
        self.input_1_encoder.to(device)
        self.input_2_encoder.to(device)
        return self

class DictObservationEncoder(nn.Module):
    def __init__(self, shape_meta: dict, num_frames: int):
        super().__init__()
        
        # Calculate lowdim_obs_dim from shape_meta
        lowdim_obs_dim = 0
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            if attr["type"] == "low_dim":
                lowdim_obs_dim += sum(attr["shape"])
        
        self.lowdim_obs_dim = lowdim_obs_dim

        low_dim_keys = list()
        key_shape_map = dict()
        key_transform_map = nn.ModuleDict()

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_shape = tuple(attr["shape"])
            key_shape_map[key] = obs_shape

            obs_type = attr.get("type", "low_dim")
            if obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        self.low_dim_keys = low_dim_keys
        self.num_frames = num_frames
        self.key_shape_map = key_shape_map

    def __call__(self, obs_dict) -> torch.Tensor:

        low_dims = list()
        for key in self.low_dim_keys:
            low_dim = obs_dict[key].flatten(0, 1)
            assert low_dim.shape[1:] == self.key_shape_map[key]
            low_dims.append(low_dim)

        
        low_dims = torch.cat(low_dims, dim=-1)  # (B*T, D_low_dim)
        low_dims = rearrange(low_dims, "(b t) d -> b (t d)", t=self.num_frames)


        return low_dims

    @property
    def output_dim(self):
        return self.num_frames * sum(self.key_shape_map.values())
