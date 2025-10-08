from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from .utils import inputs_to_torch


class IdentityObservationEncoder(nn.Module):
    
    def __init__(self, obs_dim: int, device: str = 'cuda'):
        """
        Handles observations that are numpy arrays or torch tensors 
        of shapes (B, D) or (B, 1, D) or (D)
        puts them on the correct device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def __call__(self, obs):
        obs = inputs_to_torch(obs, self.device)

        if len(obs.shape) == 1: # no batch provided
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 3: # seq dim 1 provided
            assert obs.shape[1] == 1, "No sequence length expected"
            obs = obs.squeeze(dim=1)

        obs = obs.to(self.device)
        
        return obs

    @property
    def output_dim(self):
        return self.obs_dim
    
class IdentityObservationActionEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, device='cuda'):
        super(IdentityObservationActionEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_encoder = IdentityObservationEncoder(obs_dim, device=device)
        self.action_encoder = IdentityObservationEncoder(action_dim, device=device)
        self.device = device

    def __call__(self, obs, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations and actions.
        For ObservationActionEncoder, this would need to be implemented based on the flow policy's loss function.
        """
        obs_cond = self.obs_encoder(obs)
        action_cond = self.action_encoder(action)
        cond = torch.cat([obs_cond, action_cond], dim=-1)
        return cond

    @property
    def output_dim(self):
        return self.obs_dim + self.action_dim

    def to(self, device):
        self.device = device
        self.obs_encoder.to(device)
        self.action_encoder.to(device)
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
