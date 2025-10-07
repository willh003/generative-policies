from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

class IdentityObservationEncoder(nn.Module):
    
    def __init__(self, obs_dim: int):
        """
        Handles observations that are already encoded as tensors of shape (B, D)
        """
        super().__init__()
        self.obs_dim = obs_dim
    
    def __call__(self, obs):

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 3:
            # (B, 1, D) -> (B, D), where 1 is the time dimension
            obs = obs.squeeze(1)

        return obs

    @property
    def output_dim(self):
        return self.obs_dim
    

class LowDimObservationEncoder(nn.Module):
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
